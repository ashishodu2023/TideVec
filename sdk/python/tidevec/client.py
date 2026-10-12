"""
TideVec Python SDK — REST/HTTP client.

Temporally-aware causal vector database. Connects to the tidevec-server
REST API (default port 6399) via plain HTTP/JSON — no gRPC, no protobuf,
no code generation required.

Quick start:
    from tidevec import TideVec, HalfLife

    db = TideVec("localhost:6399")
    db.create_collection("docs", dim=768,
        half_life_ms=HalfLife.ONE_WEEK, temporal_blend=0.3)

    db.upsert("docs", [{
        "id": "v1",
        "embedding": [0.1, 0.2, ...],
        "payload": {"source": "wiki"},
    }])

    results = db.search("docs", query_vector, top_k=10)
    for hit in results:
        print(hit.id, hit.score, hit.temporal_score)
"""

from __future__ import annotations

import json as _json
import os as _os
import time as _time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional


# ------------------------------------------------------------------
# Exceptions
# ------------------------------------------------------------------

class TideVecError(Exception):
    """Base exception for all TideVec client errors."""


class ConnectionError_(TideVecError):
    """Raised when the server cannot be reached."""


class CollectionNotFoundError(TideVecError):
    """Raised when a collection doesn't exist (HTTP 404)."""


class APIError(TideVecError):
    """Raised for any other non-2xx server response."""
    def __init__(self, message: str, status: int):
        super().__init__(f"[{status}] {message}")
        self.status = status


class UnauthorizedError(APIError):
    """Raised when X-Api-Key is missing or invalid (HTTP 401)."""


class ForbiddenError(APIError):
    """Raised when tenant lacks access (HTTP 403)."""


class RateLimitError(APIError):
    """Raised when rate limit is exceeded (HTTP 429)."""


class NotLeaderError(APIError):
    """
    Raised when a write reaches a Raft follower (HTTP 421, cluster mode).

    ``leader`` is the leader's base URL, or None during an election.
    The client follows the hint automatically; you only see this error
    when no leader emerged within the retry budget.
    """
    def __init__(self, message: str, status: int, leader: Optional[str] = None):
        super().__init__(message, status)
        self.leader = leader


# ------------------------------------------------------------------
# Result types
# ------------------------------------------------------------------

@dataclass
class SearchHit:
    """One result from a vector search."""
    id:                str
    score:              float
    vector_score:       float
    temporal_score:     float
    payload:            Dict[str, str]
    created_at:         int
    staleness_warning:  bool = False
    staleness_reason:   str  = ""
    causal_neighbors:   List[str] = field(default_factory=list)
    contradicted_by:    List[str] = field(default_factory=list)
    superseded_by:      List[str] = field(default_factory=list)
    text_score:         float = 0.0   # BM25 score in keyword / hybrid search

    @property
    def is_superseded(self) -> bool:
        return bool(self.superseded_by)

    @classmethod
    def _from_json(cls, j: dict) -> "SearchHit":
        return cls(
            id               = j["id"],
            score            = j["score"],
            vector_score     = j["vector_score"],
            temporal_score   = j["temporal_score"],
            payload          = j.get("payload", {}),
            created_at       = j.get("created_at", 0),
            staleness_warning= j.get("staleness_warning", False),
            staleness_reason = j.get("staleness_reason", ""),
            causal_neighbors = j.get("causal_neighbors", []),
            contradicted_by  = j.get("contradicted_by", []),
            superseded_by    = j.get("superseded_by", []),
            text_score       = j.get("text_score", 0.0),
        )

    def __repr__(self) -> str:
        return (f"SearchHit(id={self.id!r}, score={self.score:.4f}, "
                f"temporal={self.temporal_score:.3f})")


@dataclass
class Record:
    """A stored vector, as returned by get / scroll."""
    id:            str
    payload:       Dict[str, str] = field(default_factory=dict)
    created_at:    int = 0
    valid_from:    int = 0
    valid_until:   Optional[int] = None
    embedding:     Optional[List[float]] = None
    edges:         List[dict] = field(default_factory=list)
    superseded_by: List[str] = field(default_factory=list)
    supersedes:    List[str] = field(default_factory=list)

    @property
    def is_superseded(self) -> bool:
        return bool(self.superseded_by)

    @classmethod
    def _from_json(cls, j: dict) -> "Record":
        return cls(
            id            = j["id"],
            payload       = j.get("payload", {}),
            created_at    = j.get("created_at", 0),
            valid_from    = j.get("valid_from", 0),
            valid_until   = j.get("valid_until"),
            embedding     = j.get("embedding"),
            edges         = j.get("edges", []),
            superseded_by = j.get("superseded_by", []),
            supersedes    = j.get("supersedes", []),
        )


@dataclass
class ScrollPage:
    """One page of a scroll. Pass ``next_cursor`` back to get the next
    page; it is None on the last page."""
    records:     List[Record]
    next_cursor: Optional[str] = None

    def __iter__(self) -> Iterator[Record]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)


@dataclass
class SearchResponse:
    """Search results, optionally with a RetrievalTrace."""
    hits:  List[SearchHit]
    trace: Optional[dict] = None

    def __iter__(self) -> Iterator[SearchHit]:
        return iter(self.hits)

    def __len__(self) -> int:
        return len(self.hits)

    def __getitem__(self, idx: int) -> SearchHit:
        return self.hits[idx]


@dataclass
class CollectionInfo:
    name:          str
    n_vectors:     int = 0
    n_shards:      int = 0
    total_writes:  int = 0
    total_queries: int = 0
    dim:           int = 0
    index_type:    str = "tvindex"
    metric:        str = "cosine"
    backend:       str = "durable"
    supersession:  Optional[dict] = None
    text_field:    str = ""
    payload_index: bool = True
    aliases:       List[str] = field(default_factory=list)

    @classmethod
    def _from_json(cls, d: dict) -> "CollectionInfo":
        return cls(
            name=d["name"], n_vectors=d.get("n_vectors", 0),
            n_shards=d.get("n_shards", 0),
            total_writes=d.get("total_writes", 0),
            total_queries=d.get("total_queries", 0),
            dim=d.get("dim", 0),
            index_type=d.get("index_type", "tvindex"),
            metric=d.get("metric", "cosine"),
            backend=d.get("backend", "durable"),
            supersession=d.get("supersession"),
            text_field=d.get("text_field", ""),
            payload_index=d.get("payload_index", True),
            aliases=d.get("aliases", []),
        )


@dataclass
class DriftStatus:
    collection:    str
    phase:         str
    total_vectors: int = 0
    migrated:      int = 0
    skipped:       int = 0
    pct_complete:  float = 0.0
    error:         str = ""


@dataclass
class CompactionStatus:
    """Index compaction state of a collection (counts are summed over shards)."""
    started:        int = 0      # shards this compact() call started rebuilding
    running:        bool = False
    running_shards: int = 0
    runs:           int = 0      # completed compactions
    tombstones:     int = 0      # deleted/re-embedded slots awaiting compaction
    index_bytes:    int = 0      # approximate graph memory
    last_reclaimed: int = 0
    last_replayed:  int = 0      # writes replayed onto the rebuilt index
    last_ms:        float = 0.0

    @classmethod
    def _from_json(cls, d: dict) -> "CompactionStatus":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


class HalfLife:
    """Temporal decay half-life presets, in milliseconds."""
    ONE_HOUR  = 3_600_000
    ONE_DAY   = 86_400_000
    ONE_WEEK  = 604_800_000
    ONE_MONTH = 2_592_000_000
    ONE_YEAR  = 31_536_000_000


# ------------------------------------------------------------------
# Request builders (shared by TideVec and tidevec.Local)
# ------------------------------------------------------------------

def _collection_body(
    name: str,
    dim: int,
    index_type: str,
    metric: str,
    n_shards: int,
    n_replicas: int,
    write_quorum: int,
    half_life_ms: int,
    temporal_blend: float,
    staleness_threshold: float,
    agent_id: str,
    memory_ttl_seconds: int,
    dedup_threshold: float,
    agent_enabled: bool,
    supersession: bool,
    supersede_threshold: float,
    supersede_match_field: str,
    text_field: Optional[str] = None,
    payload_index: bool = True,
    compaction_ratio: Optional[float] = None,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "name": name,
        "dim": dim,
        "index_type": index_type,
        "metric": metric,
        "n_shards": n_shards,
        "n_replicas": n_replicas,
        "write_quorum": write_quorum,
        "temporal": {
            "half_life_ms": half_life_ms,
            "temporal_blend": temporal_blend,
            "staleness_threshold": staleness_threshold,
        },
    }
    if agent_enabled or memory_ttl_seconds > 0 or dedup_threshold > 0:
        body["agent"] = {
            "enabled": agent_enabled or memory_ttl_seconds > 0,
            "agent_id": agent_id,
            "memory_ttl_seconds": memory_ttl_seconds,
            "dedup_threshold": dedup_threshold,
        }
    if supersession:
        body["supersession"] = {
            "enabled": True,
            "similarity_threshold": supersede_threshold,
            "match_field": supersede_match_field,
        }
    if text_field:
        body["text_field"] = text_field
    if not payload_index:
        body["payload_index"] = False
    if compaction_ratio is not None:
        body["compaction_ratio"] = compaction_ratio
    return body


# Filters: either the legacy "key=value" string or a JSON filter object,
#   {"lang": "en"}                                   equality
#   {"lang": {"$in": ["en", "de"]}}                  $eq $ne $in $nin
#   {"year": {"$gte": 2020, "$lt": 2025}}            $gt $gte $lt $lte
#   {"created_at": {"$gte": "2025-01-01"}}           ISO-8601 dates work in ranges
#   {"$or": [{...}, {...}]}, {"$and": [...]}, {"$not": {...}}, {"k": {"$exists": True}}
# Several keys in one object are ANDed. ``created_at`` (the vector's
# timestamp) and ``_id`` are always available. Payload values are strings;
# numeric comparisons parse them as numbers or dates.
Filter = Any


def _to_ms(value: Any) -> Any:
    """datetime -> epoch ms (naive = UTC); date -> ISO string; else as-is."""
    import datetime as _dt
    if isinstance(value, _dt.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=_dt.timezone.utc)
        return int(value.timestamp() * 1000)
    if isinstance(value, _dt.date):
        return value.isoformat()
    return value


def _encode_filter(filter: Filter) -> Any:
    if filter is None or filter == "":
        return None
    if isinstance(filter, str):
        _check_filter(filter)
        return filter
    if isinstance(filter, dict):
        return {str(k): _encode_filter_value(v) for k, v in filter.items()}
    raise TypeError(f"filter must be a 'key=value' string or a dict, got {type(filter).__name__}")


def _encode_filter_value(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _encode_filter_value(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set, frozenset)):
        return [_encode_filter_value(x) for x in v]
    return _to_ms(v)


def _filter_fields(filter: Filter = None, as_of: Any = None) -> Dict[str, Any]:
    body: Dict[str, Any] = {}
    f = _encode_filter(filter)
    if f is not None:
        body["filter"] = f
    if as_of is not None:
        body["as_of"] = _to_ms(as_of)
    return body


def _search_body(
    top_k: int,
    temporal_blend: Optional[float],
    mode: str,
    causal_hops: int,
    filter: Filter,
    metric: str,
    ef_search: int,
    include_trace: bool,
    include_staleness_warnings: bool,
    superseded_penalty: float,
    *,
    as_of: Any = None,
    text: Optional[str] = None,
    hybrid: Optional[str] = None,
    alpha: float = 0.5,
    rrf_k: int = 60,
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "top_k": top_k,
        "mode": mode,
        "causal_hops": causal_hops,
        "metric": metric,
        "ef_search": ef_search,
        "include_trace": include_trace,
        "include_staleness_warnings": include_staleness_warnings,
        "superseded_penalty": superseded_penalty,
    }
    body.update(_filter_fields(filter, as_of))
    if temporal_blend is not None:
        body["temporal_blend"] = temporal_blend
    if text:
        body["text"] = text
        if hybrid is not None or alpha != 0.5 or rrf_k != 60:
            body["hybrid"] = {"fusion": hybrid or "rrf", "alpha": alpha, "rrf_k": rrf_k}
    elif hybrid is not None:
        raise ValueError("hybrid search needs text=...")
    return body


def _scroll_body(limit: int, cursor: Optional[str], filter: Filter, as_of: Any,
                 with_vector: bool, with_payload: bool) -> Dict[str, Any]:
    body: Dict[str, Any] = {"limit": limit, "with_vector": with_vector,
                            "with_payload": with_payload}
    if cursor is not None:
        body["cursor"] = cursor
    body.update(_filter_fields(filter, as_of))
    return body


def _page_from_data(data: dict) -> ScrollPage:
    return ScrollPage(records=[Record._from_json(v) for v in data.get("vectors", [])],
                      next_cursor=data.get("next_cursor"))


class _RetrievalMixin:
    """get / scroll helpers shared by TideVec and Local; subclasses
    implement ``get_many`` and ``scroll``."""

    def get_many(self, collection: str, ids: List[str], *, with_vector: bool = False,
                 with_payload: bool = True) -> List[Record]:
        raise NotImplementedError

    def scroll(self, collection: str, *, limit: int = 100, cursor: Optional[str] = None,
               filter: Filter = None, as_of: Any = None, with_vector: bool = False,
               with_payload: bool = True) -> ScrollPage:
        raise NotImplementedError

    def get(self, collection: str, id: str, *, with_vector: bool = True) -> Optional[Record]:
        """Fetch one stored vector by id (None if absent). The record
        carries its payload, timestamps, edges and supersession links."""
        found = self.get_many(collection, [id], with_vector=with_vector)
        return found[0] if found else None

    def iterate(self, collection: str, *, batch_size: int = 500, filter: Filter = None,
                as_of: Any = None, with_vector: bool = False) -> Iterator[Record]:
        """Yield every (matching) record in id order, one page at a time."""
        cursor: Optional[str] = None
        while True:
            page = self.scroll(collection, limit=batch_size, cursor=cursor, filter=filter,
                               as_of=as_of, with_vector=with_vector)
            yield from page.records
            if page.next_cursor is None:
                return
            cursor = page.next_cursor

    def list_ids(self, collection: str, *, filter: Filter = None, as_of: Any = None,
                 limit: Optional[int] = None, cursor: Optional[str] = None) -> List[str]:
        """Ids in sorted order (all of them unless ``limit`` is given),
        starting after ``cursor``."""
        out: List[str] = []
        while limit is None or len(out) < limit:
            want = 1000 if limit is None else min(1000, limit - len(out))
            page = self.scroll(collection, limit=want, cursor=cursor, filter=filter,
                               as_of=as_of, with_payload=False)
            out.extend(r.id for r in page.records)
            if page.next_cursor is None:
                break
            cursor = page.next_cursor
        return out


def payload_str(value: Any) -> str:
    """How a payload value is stored (payloads are string -> string):
    strings as-is, everything else JSON-encoded. Filters must use the
    same encoding, e.g. ``f"page={payload_str(3)}"``."""
    return value if isinstance(value, str) else _json.dumps(value)


def _normalise_vector(v: Dict[str, Any]) -> Dict[str, Any]:
    """Copy a vector dict, mapping timestamp_ms -> created_at and
    stringifying payload values."""
    v = dict(v)
    if "timestamp_ms" in v and "created_at" not in v:
        v["created_at"] = v.pop("timestamp_ms")
    if v.get("payload"):
        v["payload"] = {str(k): payload_str(val) for k, val in v["payload"].items()}
    return v


def _response_from_data(data: dict) -> "SearchResponse":
    hits = [SearchHit._from_json(r) for r in data.get("results", [])]
    return SearchResponse(hits=hits, trace=data.get("trace"))


def _check_filter(filter: str) -> None:
    key, sep, _ = filter.partition("=")
    if not sep or not key:
        raise ValueError(f"filter must look like 'key=value', got {filter!r}")


# ------------------------------------------------------------------
# Client
# ------------------------------------------------------------------

class TideVec(_RetrievalMixin):
    """
    TideVec REST client.

    Args:
        host: "host:port", default port is 6399 if omitted.
        api_key: API key sent as X-Api-Key header. Falls back to
                 TIDEVEC_API_KEY env var if not provided.
        timeout: request timeout in seconds (default 30).
        tls: use HTTPS instead of HTTP.
        max_retries: retry count for 429 rate-limit responses.
    """

    def __init__(
        self,
        host: str = "localhost:6399",
        api_key: str = "",
        timeout: float = 30.0,
        tls: bool = False,
        max_retries: int = 3,
    ):
        if ":" not in host:
            host = f"{host}:6399"
        scheme = "https" if tls else "http"
        self._base_url = f"{scheme}://{host}"
        self._api_key = api_key or _os.environ.get("TIDEVEC_API_KEY", "")
        self._timeout = timeout
        self._max_retries = max_retries

    def __enter__(self) -> "TideVec":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        """No-op — urllib does not hold persistent connections."""

    # ---- internal HTTP helpers -----------------------------------

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        base_url = self._base_url
        attempt = 0
        hops = 0
        # Cluster mode: leader elections take ~1s; give redirects a budget
        # comfortably above that.
        redirect_deadline = _time.monotonic() + max(10.0, self._timeout)
        while True:
            try:
                return self._request_once(method, path, body, base_url)
            except RateLimitError:
                if attempt >= self._max_retries:
                    raise
                _time.sleep(min(2 ** attempt, 8))
                attempt += 1
            except NotLeaderError as e:
                if _time.monotonic() > redirect_deadline:
                    raise
                hops += 1
                hint = e.leader.rstrip("/") if e.leader else None
                if hint and hint != base_url and hops <= 2:
                    base_url = hint
                    continue
                # No leader yet, or hints are stale mid-election: back off.
                _time.sleep(min(0.05 * hops, 0.5))
                base_url = hint or self._base_url
            except ConnectionError_:
                # A stale leader hint may point at a node that just died.
                if base_url == self._base_url or _time.monotonic() > redirect_deadline:
                    raise
                _time.sleep(0.1)
                base_url = self._base_url

    def _request_once(self, method: str, path: str, body: Optional[dict] = None,
                      base_url: Optional[str] = None) -> dict:
        url = (base_url or self._base_url) + path
        data = _json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        if self._api_key:
            req.add_header("X-Api-Key", self._api_key)

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read()
                return _json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            raw = e.read()
            payload = {}
            try:
                payload = _json.loads(raw)
                msg = payload.get("error", str(e))
            except Exception:
                msg = raw.decode("utf-8", errors="replace") or str(e)
            if e.code == 421:
                raise NotLeaderError(msg, e.code, payload.get("leader")) from e
            if e.code == 404:
                raise CollectionNotFoundError(msg) from e
            if e.code == 401:
                raise UnauthorizedError(msg, e.code) from e
            if e.code == 403:
                raise ForbiddenError(msg, e.code) from e
            if e.code == 429:
                raise RateLimitError(msg, e.code) from e
            raise APIError(msg, e.code) from e
        except (urllib.error.URLError, OSError) as e:
            reason = getattr(e, "reason", e)
            raise ConnectionError_(
                f"Cannot connect to TideVec at {base_url or self._base_url}: {reason}"
            ) from e

    def _get(self, path: str) -> dict:
        return self._request("GET", path)

    def _post(self, path: str, body: dict) -> dict:
        return self._request("POST", path, body)

    def _put(self, path: str, body: dict) -> dict:
        return self._request("PUT", path, body)

    def _delete(self, path: str, body: Optional[dict] = None) -> dict:
        return self._request("DELETE", path, body)

    # ---- health ----------------------------------------------------

    def health(self) -> dict:
        """Return the raw /health response: status, version, collections, timestamp_ms."""
        return self._get("/health")

    def ping(self) -> bool:
        """Return True if the server is reachable, False otherwise."""
        try:
            self.health()
            return True
        except TideVecError:
            return False

    def cluster_status(self) -> dict:
        """
        Raft status of this node (experimental cluster mode): role, term,
        leader URL, commit/applied indexes. Returns {"mode": "standalone"}
        on a single server.
        """
        return self._get("/v1/cluster/status").get("data", {})

    def info(self) -> dict:
        """Return server info: name, version, description, features."""
        resp = self._get("/v1/info")
        return resp.get("data", resp)

    def metrics(self) -> str:
        """Return Prometheus metrics text from GET /metrics."""
        url = self._base_url + "/metrics"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return resp.read().decode("utf-8")

    # ---- collections -------------------------------------------

    def create_collection(
        self,
        name: str,
        dim: int,
        index_type: str = "tvindex",
        metric: str = "cosine",
        n_shards: int = 4,
        n_replicas: int = 1,
        write_quorum: int = 1,
        half_life_ms: int = HalfLife.ONE_MONTH,
        temporal_blend: float = 0.3,
        staleness_threshold: float = 0.2,
        agent_id: str = "",
        memory_ttl_seconds: int = 0,
        dedup_threshold: float = 0.0,
        agent_enabled: bool = False,
        supersession: bool = False,
        supersede_threshold: float = 0.92,
        supersede_match_field: str = "",
        text_field: Optional[str] = None,
        payload_index: bool = True,
        compaction_ratio: Optional[float] = None,
    ) -> str:
        """
        Create a new collection. Returns the collection name.

        With ``supersession=True`` every upsert is compared against its
        nearest neighbours: a near-duplicate (``vector_score >=
        supersede_threshold``, and equal ``payload[supersede_match_field]``
        when set) is linked as an older version, and searches report it via
        ``hit.superseded_by``.

        ``text_field`` names the payload field indexed for BM25 keyword and
        hybrid search (``search(..., text=...)``). ``payload_index`` indexes
        every payload key so selective filters stay fast; turn it off to
        save memory on write-heavy collections.

        ``compaction_ratio`` (default 1.0): indexes are rebuilt in the
        background once deleted/re-embedded slots reach this fraction of the
        live vectors; lower values reclaim memory sooner. See ``compact()``.

        Scores: for the cosine metric ``vector_score`` is ``(1 + cos) / 2``
        in [0, 1] (so a threshold of 0.92 means cos >= 0.84), and
        ``score = (1 - temporal_blend) * vector_score + temporal_blend * decay``.
        """
        body = _collection_body(
            name, dim, index_type, metric, n_shards, n_replicas, write_quorum,
            half_life_ms, temporal_blend, staleness_threshold, agent_id,
            memory_ttl_seconds, dedup_threshold, agent_enabled,
            supersession, supersede_threshold, supersede_match_field,
            text_field, payload_index, compaction_ratio)
        resp = self._post("/v1/collections", body)
        return resp["data"]["name"]

    def drop_collection(self, name: str) -> bool:
        """Delete a collection. Returns True if it existed."""
        try:
            self._delete(f"/v1/collections/{name}")
            return True
        except CollectionNotFoundError:
            return False

    def has_collection(self, name: str) -> bool:
        try:
            self._get(f"/v1/collections/{name}")
            return True
        except CollectionNotFoundError:
            return False

    def get_collection(self, name: str) -> CollectionInfo:
        """Fetch metadata for a single collection. Raises CollectionNotFoundError if missing."""
        resp = self._get(f"/v1/collections/{name}")
        return CollectionInfo._from_json(resp["data"])

    def list_collections(self) -> List[CollectionInfo]:
        """List all collections on the server."""
        resp = self._get("/v1/collections")
        return [
            CollectionInfo(
                name=c["name"], n_vectors=c.get("n_vectors", 0),
                n_shards=c.get("n_shards", 0),
                dim=c.get("dim", 0),
                index_type=c.get("index_type", "tvindex"),
                metric=c.get("metric", "cosine"),
                backend=c.get("backend", "durable"),
            ) for c in resp.get("data", [])
        ]

    def set_temporal(
        self, name: str, half_life_ms: int, temporal_blend: float = 0.3,
        staleness_threshold: float = 0.2,
    ) -> None:
        """Update temporal decay settings for an existing collection."""
        self._put(f"/v1/collections/{name}/temporal", {
            "half_life_ms": half_life_ms,
            "temporal_blend": temporal_blend,
            "staleness_threshold": staleness_threshold,
        })

    # ---- vectors -------------------------------------------------

    def upsert(self, collection: str, vectors: List[Dict[str, Any]]) -> int:
        """
        Upsert vectors into a collection.

        Each vector dict:
            {
                "id":          str,
                "embedding":   List[float],
                "payload":     Dict[str, str]              (optional),
                "created_at":  int (ms since epoch)        (optional, defaults to now),
                "timestamp_ms": int (alias for created_at) (optional),
                "ttl_seconds": int                          (optional),
                "edges":       [{"target_id":str,"type":"CAUSES","weight":0.9}]  (optional),
            }

        Returns the number of vectors inserted.
        """
        mapped = [_normalise_vector(v) for v in vectors]
        for v in mapped:
            emb = v.get("embedding")
            if emb is not None and not isinstance(emb, list):
                v["embedding"] = [float(x) for x in emb]
        resp = self._post(f"/v1/collections/{collection}/upsert",
                          {"vectors": mapped})
        return resp["data"]["inserted"]

    def upsert_one(self, collection: str, vector: Dict[str, Any]) -> int:
        """Convenience wrapper for upserting a single vector."""
        return self.upsert(collection, [vector])

    def delete(self, collection: str, ids: Optional[List[str]] = None,
               *, filter: Filter = None) -> int:
        """Delete vectors by id and/or by a filter (``"key=value"`` or a
        filter dict). Returns the number actually removed."""
        if ids is None and filter is None:
            raise ValueError("delete() needs ids or filter")
        body: Dict[str, Any] = {}
        if ids is not None:
            body["ids"] = list(ids)
        if filter is not None:
            f = _encode_filter(filter)
            if f is None:
                raise ValueError("delete() filter must not be empty")
            body["filter"] = f
        resp = self._post(f"/v1/collections/{collection}/delete", body)
        return resp["data"]["deleted"]

    def count(self, collection: str, *, filter: Filter = None, as_of: Any = None) -> int:
        """Number of live vectors, optionally only those matching ``filter``
        / created at or before ``as_of``."""
        if filter is None and as_of is None:
            return self.get_collection(collection).n_vectors
        resp = self._post(f"/v1/collections/{collection}/count",
                          _filter_fields(filter, as_of))
        return resp["data"]["count"]

    # ---- retrieval by id ------------------------------------------

    def get_many(self, collection: str, ids: List[str], *, with_vector: bool = False,
                 with_payload: bool = True) -> List[Record]:
        """Fetch stored vectors by id, in request order; missing ids are skipped."""
        resp = self._post(f"/v1/collections/{collection}/get",
                          {"ids": list(ids), "with_vector": with_vector,
                           "with_payload": with_payload})
        return [Record._from_json(v) for v in resp["data"]["vectors"]]

    def scroll(self, collection: str, *, limit: int = 100, cursor: Optional[str] = None,
               filter: Filter = None, as_of: Any = None, with_vector: bool = False,
               with_payload: bool = True) -> ScrollPage:
        """One page of records in id order. Pass ``page.next_cursor`` back
        as ``cursor`` until it is None. See also ``iterate``."""
        resp = self._post(f"/v1/collections/{collection}/scroll",
                          _scroll_body(limit, cursor, filter, as_of, with_vector,
                                       with_payload))
        return _page_from_data(resp["data"])

    # ---- aliases -------------------------------------------------

    def set_alias(self, alias: str, collection: str) -> Optional[str]:
        """Point ``alias`` at ``collection`` (atomically repointing it if it
        exists). Every collection method accepts the alias as a name.
        Returns the collection it pointed at before, if any."""
        resp = self._put(f"/v1/aliases/{alias}", {"collection": collection})
        return resp["data"].get("previous")

    def delete_alias(self, alias: str) -> bool:
        try:
            self._delete(f"/v1/aliases/{alias}")
            return True
        except CollectionNotFoundError:
            return False

    def list_aliases(self) -> Dict[str, str]:
        """{alias: collection}."""
        resp = self._get("/v1/aliases")
        return {a["alias"]: a["collection"] for a in resp["data"]["aliases"]}

    # ---- maintenance -----------------------------------------------

    def compact(self, collection: str, *, wait: bool = False) -> CompactionStatus:
        """Rebuild the collection's vector, payload and text indexes without
        the tombstones left by deletes and re-embeds. Runs in the background
        (reads and writes continue); ``wait=True`` returns once it is done.
        Collections also compact by themselves at ``compaction_ratio``."""
        resp = self._post(f"/v1/collections/{collection}/compact", {"wait": wait})
        return CompactionStatus._from_json(resp["data"])

    def compaction_status(self, collection: str) -> CompactionStatus:
        resp = self._get(f"/v1/collections/{collection}/compact")
        return CompactionStatus._from_json(resp["data"])

    # ---- search ----------------------------------------------------

    def search(
        self,
        collection: str,
        query_vector: Optional[List[float]] = None,
        top_k: int = 10,
        temporal_blend: Optional[float] = None,
        mode: str = "vector_only",
        causal_hops: int = 1,
        filter: Filter = "",
        metric: str = "cosine",
        ef_search: int = 128,
        include_trace: bool = False,
        include_staleness_warnings: bool = True,
        superseded_penalty: float = 0.0,
        *,
        as_of: Any = None,
        text: Optional[str] = None,
        hybrid: Optional[str] = None,
        alpha: float = 0.5,
        rrf_k: int = 60,
    ) -> SearchResponse:
        """
        Search a collection. mode is one of:
        "vector_only" | "causal_expand" | "contradiction_check" | "entity_resolve"

        temporal_blend: None uses the collection's configured blend;
            0.0 is pure semantic similarity.
        superseded_penalty: score multiplier (1 - penalty) for hits that a
            newer version supersedes; 1.0 pushes them to the bottom.
        filter: ``"key=value"`` or a filter dict (see ``Filter``).
        as_of: ms timestamp, ISO-8601 string or datetime — search the
            collection as it was then (only vectors created by then;
            supersession as of then).
        text: BM25 keyword query over the collection's ``text_field``.
            With no ``query_vector`` this is keyword-only search; with both
            the rankings are fused (``hybrid="rrf"`` default, or
            ``"weighted"``; ``alpha`` weights the vector side).
        """
        body = _search_body(top_k, temporal_blend, mode, causal_hops, filter,
                            metric, ef_search, include_trace,
                            include_staleness_warnings, superseded_penalty,
                            as_of=as_of, text=text, hybrid=hybrid, alpha=alpha,
                            rrf_k=rrf_k)
        if query_vector is not None:
            body["vector"] = [float(x) for x in query_vector]
        elif not text:
            raise ValueError("search() needs query_vector or text")
        resp = self._post(f"/v1/collections/{collection}/search", body)
        return _response_from_data(resp["data"])

    def search_one(self, collection: str, query_vector: List[float], **kwargs) -> Optional[SearchHit]:
        """Convenience: return only the top hit, or None if no results."""
        resp = self.search(collection, query_vector, top_k=1, **kwargs)
        return resp.hits[0] if resp.hits else None

    def batch_search(
        self, collection: str, query_vectors: List[List[float]], **kwargs
    ) -> List[SearchResponse]:
        """Run multiple searches sequentially. (No batched server endpoint yet.)"""
        return [self.search(collection, q, **kwargs) for q in query_vectors]

    # ---- causal edges --------------------------------------------

    def add_edges(self, collection: str, edges: List[Dict[str, Any]]) -> int:
        """
        Add causal edges. Each edge dict:
            {"src": str, "tgt": str, "type": "CAUSES"|"CONTRADICTS"|"UPDATES"|
             "RELATED_TO"|"ENTITY_OF"|"SUPPORTS", "weight": float}
        Returns the number of edges added.
        """
        resp = self._post(f"/v1/collections/{collection}/edges", {"edges": edges})
        return resp["data"]["added"]

    # ---- DriftBridge (zero-downtime model migration) ------------

    def start_drift(
        self,
        collection: str,
        reembed_url: str,
        *,
        M: int = 16,
        ef_construction: int = 200,
        half_life_ms: Optional[int] = None,
        temporal_blend: Optional[float] = None,
        new_dim: Optional[int] = None,
    ) -> dict:
        """
        Start a zero-downtime embedding model migration.

        reembed_url must be an http(s) endpoint the server allows (see
        ``--reembed-allow`` / ``--reembed-allow-private``) that accepts
        POST {"id":..., "payload":...} and returns {"embedding":[...]}.
        The new model must keep the collection's dimension; ``new_dim``,
        if given, is checked against it. Not available in cluster mode.
        """
        body: Dict[str, Any] = {
            "reembed_url": reembed_url,
            "M": M,
            "ef_construction": ef_construction,
        }
        if new_dim is not None:
            body["new_dim"] = new_dim
        if half_life_ms is not None or temporal_blend is not None:
            body["temporal"] = {}
            if half_life_ms is not None:
                body["temporal"]["half_life_ms"] = half_life_ms
            if temporal_blend is not None:
                body["temporal"]["temporal_blend"] = temporal_blend
        resp = self._post(f"/v1/collections/{collection}/drift/start", body)
        return resp["data"]

    def drift_status(self, collection: str) -> DriftStatus:
        """Poll migration progress for a collection."""
        resp = self._get(f"/v1/collections/{collection}/drift/status")
        d = resp["data"]
        return DriftStatus(
            collection=d.get("collection", collection),
            phase=d.get("phase", "IDLE"),
            total_vectors=d.get("total_vectors", 0),
            migrated=d.get("migrated", 0),
            skipped=d.get("skipped", 0),
            pct_complete=float(d.get("pct_complete", 0)),
            error=d.get("error", ""),
        )

    def abort_drift(self, collection: str) -> None:
        """Abort an in-progress migration; live index is kept."""
        self._post(f"/v1/collections/{collection}/drift/abort", {})

    # ---- Admin (backups, requires auth) --------------------------

    def trigger_backup(self) -> str:
        """Trigger a manual snapshot. Returns the snapshot filename."""
        resp = self._post("/v1/admin/backup", {})
        return resp["data"]["snapshot"]

    def list_backups(self) -> List[str]:
        """List available backup snapshot filenames."""
        resp = self._get("/v1/admin/backups")
        return resp["data"].get("snapshots", [])

    def list_backup_manifests(self) -> List[dict]:
        """Return PITR manifest history with timestamps and cloud URIs."""
        resp = self._get("/v1/admin/backups/manifests")
        return resp["data"].get("manifests", [])

    def restore_backup(self, snapshot: str) -> dict:
        """
        Restore data from a snapshot (point-in-time recovery).
        Creates a safety snapshot first. Server restart required after restore.
        """
        resp = self._post("/v1/admin/restore", {
            "snapshot": snapshot,
            "confirm": True,
        })
        return resp["data"]


# ------------------------------------------------------------------
# Async client — thin wrapper using a thread pool, since the server
# is synchronous REST. For true async I/O, use httpx/aiohttp directly
# against the same routes documented above.
# ------------------------------------------------------------------

class AsyncTideVec:
    """
    Async-friendly wrapper around TideVec using asyncio.to_thread().

    Usage:
        async with AsyncTideVec("localhost:6399") as db:
            await db.upsert("docs", vectors)
            results = await db.search("docs", query)
    """

    def __init__(self, host: str = "localhost:6399", api_key: str = "",
                 timeout: float = 30.0, tls: bool = False):
        self._sync = TideVec(host, api_key=api_key, timeout=timeout, tls=tls)

    async def __aenter__(self) -> "AsyncTideVec":
        return self

    async def __aexit__(self, *_) -> None:
        self._sync.close()

    async def health(self) -> dict:
        import asyncio
        return await asyncio.to_thread(self._sync.health)

    async def ping(self) -> bool:
        import asyncio
        return await asyncio.to_thread(self._sync.ping)

    async def create_collection(self, *args, **kwargs) -> str:
        import asyncio
        return await asyncio.to_thread(self._sync.create_collection, *args, **kwargs)

    async def upsert(self, collection: str, vectors: List[Dict[str, Any]]) -> int:
        import asyncio
        return await asyncio.to_thread(self._sync.upsert, collection, vectors)

    async def search(self, collection: str, query_vector: List[float], **kwargs) -> SearchResponse:
        import asyncio
        return await asyncio.to_thread(self._sync.search, collection, query_vector, **kwargs)

    async def delete(self, collection: str, ids: Optional[List[str]] = None,
                     *, filter: Filter = None) -> int:
        import asyncio
        return await asyncio.to_thread(self._sync.delete, collection, ids, filter=filter)

    async def add_edges(self, collection: str, edges: List[Dict[str, Any]]) -> int:
        import asyncio
        return await asyncio.to_thread(self._sync.add_edges, collection, edges)

    async def get(self, collection: str, id: str, **kwargs) -> Optional[Record]:
        import asyncio
        return await asyncio.to_thread(self._sync.get, collection, id, **kwargs)

    async def get_many(self, collection: str, ids: List[str], **kwargs) -> List[Record]:
        import asyncio
        return await asyncio.to_thread(self._sync.get_many, collection, ids, **kwargs)

    async def scroll(self, collection: str, **kwargs) -> ScrollPage:
        import asyncio
        return await asyncio.to_thread(self._sync.scroll, collection, **kwargs)

    async def count(self, collection: str, **kwargs) -> int:
        import asyncio
        return await asyncio.to_thread(self._sync.count, collection, **kwargs)

    async def compact(self, collection: str, *, wait: bool = False) -> CompactionStatus:
        import asyncio
        return await asyncio.to_thread(self._sync.compact, collection, wait=wait)
