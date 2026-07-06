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
        )

    def __repr__(self) -> str:
        return (f"SearchHit(id={self.id!r}, score={self.score:.4f}, "
                f"temporal={self.temporal_score:.3f})")


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


@dataclass
class DriftStatus:
    collection:    str
    phase:         str
    total_vectors: int = 0
    migrated:      int = 0
    skipped:       int = 0
    pct_complete:  float = 0.0
    error:         str = ""


class HalfLife:
    """Temporal decay half-life presets, in milliseconds."""
    ONE_HOUR  = 3_600_000
    ONE_DAY   = 86_400_000
    ONE_WEEK  = 604_800_000
    ONE_MONTH = 2_592_000_000
    ONE_YEAR  = 31_536_000_000


# ------------------------------------------------------------------
# Client
# ------------------------------------------------------------------

class TideVec:
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
        last_err: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                return self._request_once(method, path, body)
            except RateLimitError as e:
                last_err = e
                if attempt >= self._max_retries:
                    raise
                _time.sleep(min(2 ** attempt, 8))
        raise last_err  # type: ignore[misc]

    def _request_once(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        url = self._base_url + path
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
            try:
                payload = _json.loads(raw)
                msg = payload.get("error", str(e))
            except Exception:
                msg = raw.decode("utf-8", errors="replace") or str(e)
            if e.code == 404:
                raise CollectionNotFoundError(msg) from e
            if e.code == 401:
                raise UnauthorizedError(msg, e.code) from e
            if e.code == 403:
                raise ForbiddenError(msg, e.code) from e
            if e.code == 429:
                raise RateLimitError(msg, e.code) from e
            raise APIError(msg, e.code) from e
        except urllib.error.URLError as e:
            raise ConnectionError_(
                f"Cannot connect to TideVec at {self._base_url}: {e.reason}"
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
    ) -> str:
        """Create a new collection. Returns the collection name."""
        body = {
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
        resp = self._post("/v1/collections", body)
        return resp["data"]["name"]

    def drop_collection(self, name: str) -> bool:
        """Delete a collection. Returns True if it existed."""
        try:
            self._delete(f"/v1/collections/{name}")
            return True
        except CollectionNotFoundError:
            return False

    def get_collection(self, name: str) -> CollectionInfo:
        """Fetch metadata for a single collection. Raises CollectionNotFoundError if missing."""
        resp = self._get(f"/v1/collections/{name}")
        d = resp["data"]
        return CollectionInfo(
            name=d["name"], n_vectors=d.get("n_vectors", 0),
            n_shards=d.get("n_shards", 0),
            total_writes=d.get("total_writes", 0),
            total_queries=d.get("total_queries", 0),
            dim=d.get("dim", 0),
            index_type=d.get("index_type", "tvindex"),
            metric=d.get("metric", "cosine"),
            backend=d.get("backend", "durable"),
        )

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
        # Remap timestamp_ms -> created_at so callers can use either name.
        # The server only reads "created_at"; "timestamp_ms" is silently ignored
        # by the C++ json_serializers.hpp parser.
        mapped = []
        for v in vectors:
            v = dict(v)  # don't mutate caller's dict
            if "timestamp_ms" in v and "created_at" not in v:
                v["created_at"] = v.pop("timestamp_ms")
            mapped.append(v)
        resp = self._post(f"/v1/collections/{collection}/upsert",
                          {"vectors": mapped})
        return resp["data"]["inserted"]

    def upsert_one(self, collection: str, vector: Dict[str, Any]) -> int:
        """Convenience wrapper for upserting a single vector."""
        return self.upsert(collection, [vector])

    def delete(self, collection: str, ids: List[str]) -> int:
        """Delete vectors by id. Returns the number actually removed."""
        resp = self._post(f"/v1/collections/{collection}/delete", {"ids": ids})
        return resp["data"]["deleted"]

    # ---- search ----------------------------------------------------

    def search(
        self,
        collection: str,
        query_vector: List[float],
        top_k: int = 10,
        temporal_blend: float = 0.3,
        mode: str = "vector_only",
        causal_hops: int = 1,
        filter: str = "",
        metric: str = "cosine",
        ef_search: int = 128,
        include_trace: bool = False,
        include_staleness_warnings: bool = True,
    ) -> SearchResponse:
        """
        Search a collection. mode is one of:
        "vector_only" | "causal_expand" | "contradiction_check" | "entity_resolve"
        """
        body = {
            "vector": query_vector,
            "top_k": top_k,
            "temporal_blend": temporal_blend,
            "mode": mode,
            "causal_hops": causal_hops,
            "filter": filter,
            "metric": metric,
            "ef_search": ef_search,
            "include_trace": include_trace,
            "include_staleness_warnings": include_staleness_warnings,
        }
        resp = self._post(f"/v1/collections/{collection}/search", body)
        data = resp["data"]
        hits = [SearchHit._from_json(r) for r in data.get("results", [])]
        trace = data.get("trace")
        return SearchResponse(hits=hits, trace=trace)

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
    ) -> dict:
        """
        Start a zero-downtime embedding model migration.

        reembed_url must be a public http(s) endpoint on the server allowlist
        that accepts POST {"id":..., "payload":...} and returns {"embedding":[...]}.
        """
        body: Dict[str, Any] = {
            "reembed_url": reembed_url,
            "M": M,
            "ef_construction": ef_construction,
        }
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

    async def delete(self, collection: str, ids: List[str]) -> int:
        import asyncio
        return await asyncio.to_thread(self._sync.delete, collection, ids)
