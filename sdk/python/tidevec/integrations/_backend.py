"""
Shared plumbing for the framework adapters (Haystack, LangGraph, CrewAI).

Those frameworks need get-by-id, listing and rich filters. Current
servers and embedded builds provide them (``get_many``, ``scroll``,
filtered ``count``/``delete`` and JSON filters); against 0.3.0 servers or
older native modules the backend falls back to enumerating a collection
with an exhaustive search and evaluating filters client-side, which only
suits small collections (thousands of vectors).

Filters use TideVec's Mongo-style syntax::

    {"lang": "en", "year": {"$gte": 2020}, "tag": ["a", "b"],
     "$or": [{...}, {...}], "$not": {...}}

``created_at`` refers to the vector timestamp (ms since epoch).
"""

from __future__ import annotations

import json as _json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Sequence

from ..client import APIError, CollectionNotFoundError, payload_str

Filter = Optional[Dict[str, Any]]


@dataclass
class Record:
    """A stored vector (from get/scroll) or a search hit, normalised."""

    id: str
    created_at: int
    payload: Dict[str, str]
    superseded_by: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    score: Optional[float] = None

    @classmethod
    def of(cls, obj: Any, score: Optional[float] = None) -> "Record":
        return cls(id=obj.id, created_at=int(obj.created_at or 0), payload=dict(obj.payload or {}),
                   superseded_by=list(getattr(obj, "superseded_by", None) or []),
                   embedding=getattr(obj, "embedding", None),
                   score=score if score is not None else getattr(obj, "score", None))


# ---- client-side filter evaluation ---------------------------------------

def _to_number(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            pass
        try:
            dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp() * 1000.0
    return None


def _eq(stored: Optional[str], want: Any) -> bool:
    if stored is None:
        return False
    if isinstance(want, str):
        return stored == want
    a, b = _to_number(stored), _to_number(want)
    if a is not None and b is not None:
        return a == b
    return stored == payload_str(want)


def _field_matches(stored: Optional[str], cond: Any) -> bool:
    if isinstance(cond, list):
        return any(_eq(stored, c) for c in cond)
    if not isinstance(cond, dict):
        return _eq(stored, cond)
    for op, arg in cond.items():
        if op == "$eq":
            ok = _eq(stored, arg)
        elif op == "$ne":
            ok = not _eq(stored, arg)
        elif op == "$in":
            ok = any(_eq(stored, a) for a in arg)
        elif op == "$nin":
            ok = not any(_eq(stored, a) for a in arg)
        elif op == "$exists":
            ok = (stored is not None) == bool(arg)
        elif op in ("$gt", "$gte", "$lt", "$lte"):
            a, b = _to_number(stored), _to_number(arg)
            if a is None or b is None or math.isnan(a):
                return False
            ok = {"$gt": a > b, "$gte": a >= b, "$lt": a < b, "$lte": a <= b}[op]
        else:
            raise ValueError(f"unknown filter operator {op!r}")
        if not ok:
            return False
    return True


def matches(flt: Filter, record: Record) -> bool:
    """Evaluate a Mongo-style filter against a record (mirrors the server)."""
    if not flt:
        return True
    for key, cond in flt.items():
        if key == "$and":
            ok = all(matches(c, record) for c in cond)
        elif key == "$or":
            ok = any(matches(c, record) for c in cond)
        elif key == "$not":
            ok = not matches(cond, record)
        elif key == "created_at":
            ok = _field_matches(str(record.created_at), cond)
        elif key == "_id":
            ok = _field_matches(record.id, cond)
        else:
            ok = _field_matches(record.payload.get(key), cond)
        if not ok:
            return False
    return True


def legacy_pushdown(flt: Filter) -> str:
    """A ``key=value`` filter implied by ``flt`` (a superset of its matches), or ""."""
    if not flt:
        return ""
    for key, cond in flt.items():
        if key == "$and":
            for c in cond:
                s = legacy_pushdown(c)
                if s:
                    return s
        elif not key.startswith("$") and key not in ("created_at", "_id"):
            if isinstance(cond, dict) and set(cond) == {"$eq"}:
                cond = cond["$eq"]
            if isinstance(cond, (str, int, float)) and not isinstance(cond, bool):
                return f"{key}={payload_str(cond)}"
    return ""


def and_filters(*filters: Filter) -> Filter:
    parts = [f for f in filters if f]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else {"$and": parts}


# ---- backend ---------------------------------------------------------------

class Backend:
    """
    Collection operations using the best API the database offers.

    Args:
        db: ``tidevec.TideVec`` or ``tidevec.Local``.
        collection: collection name (must exist before the first call).
        structured: force (True) or disable (False) the get/scroll/JSON-filter
            API; None probes it once.
    """

    def __init__(self, db: Any, collection: str, structured: Optional[bool] = None):
        self.db = db
        self.collection = collection
        self._structured = structured

    @property
    def structured(self) -> bool:
        if self._structured is None:
            try:
                self.db.scroll(self.collection, limit=1, with_payload=False)
                self._structured = True
            except (AttributeError, NotImplementedError, TypeError, CollectionNotFoundError):
                # Old client/native module, or a 0.3.0 server answering 404 for /scroll.
                self._structured = False
            except APIError as e:
                if e.status not in (400, 404, 405):
                    raise
                self._structured = False
        return self._structured

    # ---- reads -----------------------------------------------------------

    def search(self, vector: Sequence[float], top_k: int = 10, flt: Filter = None, *,
               temporal_blend: Optional[float] = None, superseded_penalty: float = 0.0,
               **kwargs: Any) -> List[Record]:
        vec = [float(x) for x in vector]
        if self.structured:
            hits = self.db.search(self.collection, vec, top_k=top_k, filter=flt or "",
                                  temporal_blend=temporal_blend,
                                  superseded_penalty=superseded_penalty,
                                  include_staleness_warnings=False, **kwargs).hits
            return [Record.of(h) for h in hits]
        pushdown = legacy_pushdown(flt)
        fetch = max(top_k * 5, 50) if flt else top_k
        hits = self.db.search(self.collection, vec, top_k=fetch, filter=pushdown,
                              temporal_blend=temporal_blend,
                              superseded_penalty=superseded_penalty,
                              include_staleness_warnings=False).hits
        return [r for r in (Record.of(h) for h in hits) if matches(flt, r)][:top_k]

    def _enumerate(self, flt: Filter) -> List[Record]:
        n = int(self.db.count(self.collection))
        if n == 0:
            return []
        dim = int(self.db.get_collection(self.collection).dim)
        probe = [1.0] + [0.0] * (dim - 1)
        hits = self.db.search(self.collection, probe, top_k=n, ef_search=max(128, n),
                              filter=legacy_pushdown(flt), temporal_blend=0.0,
                              include_staleness_warnings=False).hits
        return [r for r in (Record.of(h) for h in hits) if matches(flt, r)]

    def scan(self, flt: Filter = None, *, with_vector: bool = False,
             page_size: int = 500) -> Iterator[Record]:
        if not self.structured:
            yield from self._enumerate(flt)
            return
        for rec in self.db.iterate(self.collection, batch_size=page_size, filter=flt,
                                   with_vector=with_vector):
            yield Record.of(rec)

    def get(self, ids: Sequence[str], *, with_vector: bool = False) -> List[Record]:
        ids = list(ids)
        if not ids:
            return []
        if self.structured:
            return [Record.of(r) for r in self.db.get_many(self.collection, ids,
                                                           with_vector=with_vector)]
        wanted = set(ids)
        found = {r.id: r for r in self._enumerate(None) if r.id in wanted}
        return [found[i] for i in ids if i in found]

    def count(self, flt: Filter = None) -> int:
        if not flt:
            return int(self.db.count(self.collection))
        if self.structured:
            return int(self.db.count(self.collection, filter=flt))
        return len(self._enumerate(flt))

    # ---- writes ----------------------------------------------------------

    def upsert(self, vectors: List[Dict[str, Any]]) -> int:
        return int(self.db.upsert(self.collection, vectors)) if vectors else 0

    def delete(self, ids: Optional[Sequence[str]] = None, flt: Filter = None) -> int:
        n = 0
        if ids:
            n += int(self.db.delete(self.collection, list(ids)))
        if flt:
            if self.structured:
                n += int(self.db.delete(self.collection, filter=flt))
            else:
                doomed = [r.id for r in self._enumerate(flt)]
                if doomed:
                    n += int(self.db.delete(self.collection, doomed))
        return n


def dumps(value: Any) -> str:
    return _json.dumps(value, default=str, separators=(",", ":"))
