"""
Embedded TideVec — the full engine in-process, no server.

    from tidevec import Local

    db = Local("./tidevec_data")
    db.create_collection("docs", dim=384, supersession=True)
    db.upsert("docs", [{"id": "v1", "embedding": [...], "payload": {"doc": "refunds"}}])
    hits = db.search("docs", query_vector, top_k=5)

``Local`` has the same methods as the REST client ``TideVec``, so code
written against one runs against the other. Data is persisted (WAL +
segments) under ``path`` and recovered on the next open.

Requires the native extension: ``pip install "tidevec[embedded]"``.
"""

from __future__ import annotations

import functools
import json as _json
import os as _os
import threading
from typing import Any, Callable, Dict, List, Optional, TypeVar

from .client import (
    CollectionInfo,
    CollectionNotFoundError,
    CompactionStatus,
    Filter,
    HalfLife,
    Record,
    ScrollPage,
    SearchHit,
    SearchResponse,
    TideVecError,
    _RetrievalMixin,
    _collection_body,
    _encode_filter,
    _filter_fields,
    _normalise_vector,
    _page_from_data,
    _response_from_data,
    _scroll_body,
    _search_body,
)

_F = TypeVar("_F", bound=Callable[..., Any])


def _load_native():
    try:
        import tidevec_native  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "tidevec.Local needs the native engine. Install it with:\n"
            '    pip install "tidevec[embedded]"'
        ) from e
    return tidevec_native


def _translate_errors(fn: _F) -> _F:
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except RuntimeError as e:
            msg = str(e)
            if msg.startswith("Collection not found"):
                raise CollectionNotFoundError(msg) from e
            raise TideVecError(msg) from e
    return wrapper  # type: ignore[return-value]


class Local(_RetrievalMixin):
    """
    In-process TideVec database.

    Args:
        path: data directory (created if missing).
        segment_store: persist vectors in mmap segment files as well as
            the WAL (faster recovery for large collections).
    """

    def __init__(self, path: str = "./tidevec_data", *, segment_store: bool = True):
        native = _load_native()
        self._np = __import__("numpy")
        self.path = str(path)
        _os.makedirs(self.path, exist_ok=True)
        self._db = native.Database(self.path, segment_store)
        self._dims: Dict[str, int] = {}
        self._lock = threading.Lock()

    def __enter__(self) -> "Local":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        """Flush and release the engine. The object is unusable afterwards."""
        self._db = None

    def __repr__(self) -> str:
        return f"Local({self.path!r})"

    # ---- health (parity with the REST client) -------------------

    def health(self) -> dict:
        return {"status": "ok", "mode": "embedded",
                "collections": len(self.list_collections())}

    def ping(self) -> bool:
        return self._db is not None

    def info(self) -> dict:
        return {"name": "TideVec", "mode": "embedded", "path": self.path}

    # ---- collections ---------------------------------------------

    @_translate_errors
    def create_collection(
        self,
        name: str,
        dim: int,
        index_type: str = "tvindex",
        metric: str = "cosine",
        n_shards: int = 1,
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
        """Create a collection. See ``TideVec.create_collection``."""
        body = _collection_body(
            name, dim, index_type, metric, n_shards, n_replicas, write_quorum,
            half_life_ms, temporal_blend, staleness_threshold, agent_id,
            memory_ttl_seconds, dedup_threshold, agent_enabled,
            supersession, supersede_threshold, supersede_match_field,
            text_field, payload_index, compaction_ratio)
        self._db.create_collection(_json.dumps(body))
        self._dims[name] = dim
        return name

    def drop_collection(self, name: str) -> bool:
        with self._lock:
            self._dims.clear()   # aliases may have pointed at it
        return bool(self._db.drop_collection(name))

    def has_collection(self, name: str) -> bool:
        return bool(self._db.has_collection(name))

    def list_collections(self) -> List[CollectionInfo]:
        return [
            CollectionInfo(name=c["name"], n_vectors=c.get("n_vectors", 0),
                           n_shards=c.get("n_shards", 0), dim=c.get("dim", 0),
                           index_type=c.get("index_type", "tvindex"),
                           metric=c.get("metric", "cosine"), backend="embedded")
            for c in _json.loads(self._db.list_collections())
        ]

    @_translate_errors
    def get_collection(self, name: str) -> CollectionInfo:
        return CollectionInfo._from_json(_json.loads(self._db.describe(name)))

    @_translate_errors
    def count(self, collection: str, *, filter: Filter = None, as_of: Any = None) -> int:
        if filter is None and as_of is None:
            return int(self._db.count(collection))
        return int(self._db.count_where(collection, _json.dumps(_filter_fields(filter, as_of))))

    # ---- retrieval by id ------------------------------------------

    @_translate_errors
    def get_many(self, collection: str, ids: List[str], *, with_vector: bool = False,
                 with_payload: bool = True) -> List[Record]:
        data = _json.loads(self._db.get(collection, list(ids), with_vector, with_payload))
        return [Record._from_json(v) for v in data["vectors"]]

    @_translate_errors
    def scroll(self, collection: str, *, limit: int = 100, cursor: Optional[str] = None,
               filter: Filter = None, as_of: Any = None, with_vector: bool = False,
               with_payload: bool = True) -> ScrollPage:
        body = _scroll_body(limit, cursor, filter, as_of, with_vector, with_payload)
        return _page_from_data(_json.loads(self._db.scroll(collection, _json.dumps(body))))

    # ---- aliases -------------------------------------------------

    @_translate_errors
    def set_alias(self, alias: str, collection: str) -> Optional[str]:
        with self._lock:
            self._dims.pop(alias, None)
        prev = self._db.set_alias(alias, collection)
        return prev or None

    def delete_alias(self, alias: str) -> bool:
        with self._lock:
            self._dims.pop(alias, None)
        return bool(self._db.delete_alias(alias))

    def list_aliases(self) -> Dict[str, str]:
        return {a["alias"]: a["collection"] for a in _json.loads(self._db.list_aliases())}

    # ---- maintenance -----------------------------------------------

    @_translate_errors
    def compact(self, collection: str, *, wait: bool = False) -> CompactionStatus:
        """Rebuild indexes without tombstones. See ``TideVec.compact``."""
        return CompactionStatus._from_json(_json.loads(self._db.compact(collection, wait)))

    @_translate_errors
    def compaction_status(self, collection: str) -> CompactionStatus:
        return CompactionStatus._from_json(_json.loads(self._db.compaction_status(collection)))

    @_translate_errors
    def set_temporal(self, name: str, half_life_ms: int, temporal_blend: float = 0.3,
                     staleness_threshold: float = 0.2) -> None:
        self._db.set_temporal(name, _json.dumps({
            "half_life_ms": half_life_ms,
            "temporal_blend": temporal_blend,
            "staleness_threshold": staleness_threshold,
        }))

    # ---- vectors -------------------------------------------------

    def _dim(self, collection: str) -> int:
        with self._lock:
            if collection not in self._dims:
                self._dims[collection] = self.get_collection(collection).dim
            return self._dims[collection]

    @_translate_errors
    def upsert(self, collection: str, vectors: List[Dict[str, Any]]) -> int:
        """Upsert vector dicts (same shape as ``TideVec.upsert``)."""
        if not vectors:
            return 0
        dim = self._dim(collection)
        meta, rows = [], []
        for v in vectors:
            v = _normalise_vector(v)
            emb = v.pop("embedding")
            if len(emb) != dim:
                raise TideVecError(
                    f"vector {v.get('id')!r} has dimension {len(emb)}, "
                    f"collection {collection!r} expects {dim}")
            rows.append(emb)
            meta.append(v)
        arr = self._np.asarray(rows, dtype=self._np.float32).reshape(len(rows), dim)
        return int(self._db.upsert(collection, arr, _json.dumps(meta)))

    def upsert_one(self, collection: str, vector: Dict[str, Any]) -> int:
        return self.upsert(collection, [vector])

    @_translate_errors
    def delete(self, collection: str, ids: Optional[List[str]] = None,
               *, filter: Filter = None) -> int:
        """Delete by id and/or filter (``"key=value"`` or a filter dict)."""
        if ids is None and filter is None:
            raise ValueError("delete() needs ids or filter")
        f = None
        if filter is not None:
            f = _encode_filter(filter)
            if f is None:
                raise ValueError("delete() filter must not be empty")
        n = 0
        if ids is not None:
            n += int(self._db.remove(collection, list(ids)))
        if f is not None:
            n += int(self._db.remove_where_json(collection, _json.dumps(f)))
        return n

    @_translate_errors
    def add_edges(self, collection: str, edges: List[Dict[str, Any]]) -> int:
        """Add typed edges: ``{"src", "tgt", "type", "weight"}``."""
        return int(self._db.add_edges(collection, _json.dumps(edges)))

    # ---- search ----------------------------------------------------

    @_translate_errors
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
        """Search a collection. See ``TideVec.search``."""
        body = _search_body(top_k, temporal_blend, mode, causal_hops, filter,
                            metric, ef_search, include_trace,
                            include_staleness_warnings, superseded_penalty,
                            as_of=as_of, text=text, hybrid=hybrid, alpha=alpha,
                            rrf_k=rrf_k)
        if query_vector is None:
            if not text:
                raise ValueError("search() needs query_vector or text")
            q = self._np.zeros(0, dtype=self._np.float32)
        else:
            q = self._np.asarray(query_vector, dtype=self._np.float32).reshape(-1)
            dim = self._dim(collection)
            if q.shape[0] != dim:
                raise TideVecError(f"query has dimension {q.shape[0]}, "
                                   f"collection {collection!r} expects {dim}")
        data = _json.loads(self._db.search(collection, q, _json.dumps(body)))
        return _response_from_data(data)

    def search_one(self, collection: str, query_vector: List[float],
                   **kwargs) -> Optional[SearchHit]:
        resp = self.search(collection, query_vector, top_k=1, **kwargs)
        return resp.hits[0] if resp.hits else None

    def batch_search(self, collection: str, query_vectors: List[List[float]],
                     **kwargs) -> List[SearchResponse]:
        return [self.search(collection, q, **kwargs) for q in query_vectors]
