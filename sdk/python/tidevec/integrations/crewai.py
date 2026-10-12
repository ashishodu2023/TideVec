"""
CrewAI memory storage backed by TideVec (CrewAI >= 1.10 unified ``Memory``).

    from crewai import Crew, Memory
    from tidevec import TideVec
    from tidevec.integrations.crewai import TideVecStorage

    storage = TideVecStorage(TideVec("localhost:6399"), collection="crew_memory")
    crew = Crew(agents=[...], tasks=[...], memory=Memory(storage=storage))

    # or make it the default for every Memory() created in this process:
    storage.install_as_default()

``TideVecStorage`` implements CrewAI's ``StorageBackend`` protocol:
records keep their hierarchical scope (``/project/alpha``), categories,
metadata, importance and timestamps; search pushes scope and category
filters down to TideVec and returns similarity scores in ``[0, 1]``.

Limitations:
    * Listing, scope/category statistics and filtered deletes scan the
      matching records. With the retrieval-completeness REST API this is a
      paged server-side scan; against servers and native modules before
      0.3.0 it enumerates the collection, which only suits small memories.
    * Scope prefixes are pushed down for the first 8 path components.
"""

from __future__ import annotations

import asyncio
import json as _json
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from crewai.memory.types import MemoryRecord, ScopeInfo
except ImportError as e:  # pragma: no cover
    raise ImportError('pip install "tidevec[crewai]"') from e

from ..client import payload_str
from ._backend import Backend, Record, and_filters, dumps

_CONTENT = "content"
_SCOPE = "_cr_scope"
_CATS = "_cr_cats"
_META = "_cr_meta"
_IMPORTANCE = "_cr_importance"
_ACCESSED = "_cr_accessed"
_SOURCE = "_cr_source"
_PRIVATE = "_cr_private"
_NO_EMBEDDING = "_cr_noemb"
_MAX_DEPTH = 8


def _parts(scope: Optional[str]) -> List[str]:
    return [p for p in (scope or "/").split("/") if p]


def _norm_scope(scope: Optional[str]) -> str:
    return "/" + "/".join(_parts(scope))


def _in_scope(scope: str, prefix: Optional[str]) -> bool:
    p = _norm_scope(prefix)
    return p == "/" or scope == p or scope.startswith(p + "/")


def _ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _dt(ms: int) -> datetime:
    # CrewAI compares record timestamps with naive datetime.utcnow().
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).replace(tzinfo=None)


class TideVecStorage:
    """
    Args:
        db: ``tidevec.TideVec`` or ``tidevec.Local``.
        collection: collection name. Created on the first ``save`` with the
            dimension of the first embedding (or ``vector_dim``).
        vector_dim: create the collection eagerly with this dimension.
        temporal_blend: recency weight inside TideVec's ranking. Defaults to
            0 because CrewAI applies its own recency/importance scoring.
        **collection_kwargs: extra ``create_collection`` arguments.
    """

    def __init__(self, db: Any, *, collection: str = "crewai_memory",
                 vector_dim: Optional[int] = None, temporal_blend: float = 0.0,
                 **collection_kwargs: Any):
        self.db = db
        self.collection = collection
        self.temporal_blend = temporal_blend
        self.collection_kwargs = collection_kwargs
        self._lock = threading.Lock()
        self._dim: Optional[int] = None
        self._backend = Backend(db, collection)
        if db.has_collection(collection):
            self._dim = int(db.get_collection(collection).dim)
        elif vector_dim:
            self._create(vector_dim)

    def install_as_default(self) -> None:
        """Make every ``crewai.Memory()`` without explicit storage use this backend."""
        from crewai.memory.storage.factory import set_memory_storage_factory
        set_memory_storage_factory(lambda *args, **kwargs: self)

    # ---- helpers ---------------------------------------------------------

    def _create(self, dim: int) -> None:
        with self._lock:
            if self._dim is None:
                if not self.db.has_collection(self.collection):
                    self.db.create_collection(self.collection, dim=dim, **self.collection_kwargs)
                    self._dim = dim
                else:
                    self._dim = int(self.db.get_collection(self.collection).dim)

    def _filter(self, scope_prefix: Optional[str] = None,
                categories: Optional[List[str]] = None,
                metadata_filter: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        conds: List[Optional[Dict[str, Any]]] = []
        parts = _parts(scope_prefix)[:_MAX_DEPTH]
        if parts:
            conds.append({f"_cr_s{i}": p for i, p in enumerate(parts)})
        if categories:
            ors = [{f"_cr_c.{c}": "1"} for c in categories]
            conds.append(ors[0] if len(ors) == 1 else {"$or": ors})
        for k, v in (metadata_filter or {}).items():
            if isinstance(v, (str, int, float, bool)):
                conds.append({f"_cr_m.{k}": payload_str(v)})
        return and_filters(*conds)

    @staticmethod
    def _matches(rec: MemoryRecord, scope_prefix: Optional[str],
                 categories: Optional[List[str]],
                 metadata_filter: Optional[Dict[str, Any]]) -> bool:
        if scope_prefix is not None and not _in_scope(rec.scope, scope_prefix):
            return False
        if categories and not any(c in rec.categories for c in categories):
            return False
        if metadata_filter and not all(rec.metadata.get(k) == v for k, v in metadata_filter.items()):
            return False
        return True

    def _to_vector(self, r: MemoryRecord, dim: int) -> Dict[str, Any]:
        scope = _norm_scope(r.scope)
        payload = {_CONTENT: r.content, _SCOPE: scope, _CATS: dumps(list(r.categories)),
                   _META: dumps(r.metadata or {}), _IMPORTANCE: repr(float(r.importance)),
                   _ACCESSED: str(_ms(r.last_accessed)), _PRIVATE: "1" if r.private else "0"}
        if r.source is not None:
            payload[_SOURCE] = r.source
        for i, p in enumerate(_parts(scope)[:_MAX_DEPTH]):
            payload[f"_cr_s{i}"] = p
        for c in r.categories:
            payload[f"_cr_c.{c}"] = "1"
        for k, v in (r.metadata or {}).items():
            if isinstance(v, (str, int, float, bool)):
                payload[f"_cr_m.{k}"] = payload_str(v)
        if r.embedding:
            emb = [float(x) for x in r.embedding]
        else:
            emb = [1.0] + [0.0] * (dim - 1)
            payload[_NO_EMBEDDING] = "1"
        return {"id": r.id, "embedding": emb, "payload": payload, "created_at": _ms(r.created_at)}

    @staticmethod
    def _to_record(rec: Record) -> MemoryRecord:
        p = rec.payload
        embedding = rec.embedding if rec.embedding and _NO_EMBEDDING not in p else None
        return MemoryRecord(
            id=rec.id, content=p.get(_CONTENT, ""), scope=p.get(_SCOPE, "/"),
            categories=_json.loads(p.get(_CATS, "[]")), metadata=_json.loads(p.get(_META, "{}")),
            importance=float(p.get(_IMPORTANCE, 0.5)), created_at=_dt(rec.created_at),
            last_accessed=_dt(int(p.get(_ACCESSED, rec.created_at))), embedding=embedding,
            source=p.get(_SOURCE), private=p.get(_PRIVATE) == "1")

    def _scan(self, scope_prefix: Optional[str] = None, *, with_vector: bool = False,
              categories: Optional[List[str]] = None,
              metadata_filter: Optional[Dict[str, Any]] = None) -> List[MemoryRecord]:
        if self._dim is None:
            return []
        flt = self._filter(scope_prefix, categories, metadata_filter)
        out = [self._to_record(r) for r in self._backend.scan(flt, with_vector=with_vector)
               if _SCOPE in r.payload]
        return [r for r in out if self._matches(r, scope_prefix, categories, metadata_filter)]

    # ---- StorageBackend protocol ----------------------------------------

    def save(self, records: List[MemoryRecord]) -> None:
        if not records:
            return
        if self._dim is None:
            dim = next((len(r.embedding) for r in records if r.embedding), None)
            if dim is None:
                raise ValueError("cannot create the collection before an embedded record is saved; "
                                 "pass vector_dim=")
            self._create(dim)
        assert self._dim is not None
        for r in records:
            if r.embedding and len(r.embedding) != self._dim:
                raise ValueError(f"embedding dimension {len(r.embedding)} != "
                                 f"collection dimension {self._dim}")
        rows = [self._to_vector(r, self._dim) for r in records]
        for i in range(0, len(rows), 256):
            self._backend.upsert(rows[i:i + 256])

    def update(self, record: MemoryRecord) -> None:
        if record.embedding is None and self._backend.structured:
            prev = self._backend.get([record.id], with_vector=True)
            if prev and prev[0].embedding and _NO_EMBEDDING not in prev[0].payload:
                record = record.model_copy(update={"embedding": prev[0].embedding})
        self.save([record])

    def touch_records(self, record_ids: List[str]) -> None:
        """Refresh ``last_accessed`` (needs the retrieval-completeness API; no-op otherwise)."""
        if not record_ids or self._dim is None or not self._backend.structured:
            return
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        recs = [self._to_record(r) for r in self._backend.get(record_ids, with_vector=True)]
        self.save([r.model_copy(update={"last_accessed": now}) for r in recs])

    def get_record(self, record_id: str) -> Optional[MemoryRecord]:
        if self._dim is None:
            return None
        recs = self._backend.get([record_id], with_vector=True)
        return self._to_record(recs[0]) if recs else None

    def search(self, query_embedding: List[float], scope_prefix: Optional[str] = None,
               categories: Optional[List[str]] = None,
               metadata_filter: Optional[Dict[str, Any]] = None, limit: int = 10,
               min_score: float = 0.0) -> List[tuple]:
        if self._dim is None:
            return []
        if len(query_embedding) != self._dim:
            raise ValueError(f"query dimension {len(query_embedding)} != collection dimension {self._dim}")
        flt = self._filter(scope_prefix, categories, metadata_filter)
        fetch = limit * 3 + 10 if (categories or metadata_filter) else limit + 5
        out = []
        for r in self._backend.search(query_embedding, fetch, flt,
                                      temporal_blend=self.temporal_blend):
            if _SCOPE not in r.payload or _NO_EMBEDDING in r.payload:
                continue
            rec = self._to_record(r)
            if not self._matches(rec, scope_prefix, categories, metadata_filter):
                continue
            score = float(r.score or 0.0)
            if score >= min_score:
                out.append((rec, score))
            if len(out) >= limit:
                break
        return out

    def delete(self, scope_prefix: Optional[str] = None, categories: Optional[List[str]] = None,
               record_ids: Optional[List[str]] = None, older_than: Optional[datetime] = None,
               metadata_filter: Optional[Dict[str, Any]] = None) -> int:
        if self._dim is None:
            return 0
        scoped = scope_prefix is not None and _norm_scope(scope_prefix) != "/"
        if record_ids and not (scoped or categories or metadata_filter or older_than):
            return self._backend.delete(ids=record_ids)
        recs = self._scan(scope_prefix, categories=categories, metadata_filter=metadata_filter)
        if record_ids:
            wanted = set(record_ids)
            recs = [r for r in recs if r.id in wanted]
        if older_than is not None:
            cutoff = _ms(older_than)
            recs = [r for r in recs if _ms(r.created_at) < cutoff]
        ids = [r.id for r in recs]
        return self._backend.delete(ids=ids) if ids else 0

    def list_records(self, scope_prefix: Optional[str] = None, limit: int = 200,
                     offset: int = 0) -> List[MemoryRecord]:
        recs = self._scan(scope_prefix)
        recs.sort(key=lambda r: r.created_at, reverse=True)
        return recs[offset:offset + limit]

    def get_scope_info(self, scope: str) -> ScopeInfo:
        path = _norm_scope(scope)
        recs = self._scan(path)
        child_prefix = path.rstrip("/") + "/"
        children = set()
        cats = set()
        for r in recs:
            cats.update(r.categories)
            if r.scope.startswith(child_prefix):
                first = r.scope[len(child_prefix):].split("/", 1)[0]
                if first:
                    children.add(child_prefix + first)
        stamps = [r.created_at for r in recs]
        return ScopeInfo(path=path, record_count=len(recs), categories=sorted(cats),
                         oldest_record=min(stamps) if stamps else None,
                         newest_record=max(stamps) if stamps else None,
                         child_scopes=sorted(children))

    def list_scopes(self, parent: str = "/") -> List[str]:
        return self.get_scope_info(parent).child_scopes

    def list_categories(self, scope_prefix: Optional[str] = None) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in self._scan(scope_prefix):
            for c in r.categories:
                counts[c] = counts.get(c, 0) + 1
        return counts

    def count(self, scope_prefix: Optional[str] = None) -> int:
        if self._dim is None:
            return 0
        if scope_prefix is None or _norm_scope(scope_prefix) == "/":
            return self._backend.count()
        return len(self._scan(scope_prefix))

    def reset(self, scope_prefix: Optional[str] = None) -> None:
        if self._dim is None:
            return
        if scope_prefix is None or _norm_scope(scope_prefix) == "/":
            self.db.drop_collection(self.collection)
            with self._lock:
                self._dim = None
            return
        self.delete(scope_prefix=scope_prefix)

    # ---- async -----------------------------------------------------------

    async def asave(self, records: List[MemoryRecord]) -> None:
        await asyncio.to_thread(self.save, records)

    async def asearch(self, query_embedding: List[float], scope_prefix: Optional[str] = None,
                      categories: Optional[List[str]] = None,
                      metadata_filter: Optional[Dict[str, Any]] = None, limit: int = 10,
                      min_score: float = 0.0) -> List[tuple]:
        return await asyncio.to_thread(self.search, query_embedding, scope_prefix, categories,
                                       metadata_filter, limit, min_score)

    async def adelete(self, scope_prefix: Optional[str] = None,
                      categories: Optional[List[str]] = None,
                      record_ids: Optional[List[str]] = None,
                      older_than: Optional[datetime] = None,
                      metadata_filter: Optional[Dict[str, Any]] = None) -> int:
        return await asyncio.to_thread(self.delete, scope_prefix, categories, record_ids,
                                       older_than, metadata_filter)


__all__ = ["TideVecStorage"]
