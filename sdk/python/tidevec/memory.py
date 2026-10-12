"""
Versioned long-term memory for agents.

    from tidevec import Local
    from tidevec.memory import AgentMemory

    mem = AgentMemory(Local("~/.tidevec/memory"))
    mem.remember("User's favourite editor is Vim", key="user.editor")
    mem.remember("User switched to Helix", key="user.editor")   # supersedes the Vim memory
    mem.recall("which editor does the user like?")              # -> Helix only

A ``key`` names the fact being remembered; a new memory with the same key
supersedes the previous ones. Unkeyed near-identical restatements are
linked automatically by TideVec's supersession. ``recall`` returns current
memories ranked by similarity blended with recency; superseded versions
stay available through ``history`` and ``include_superseded=True``.

``recall`` and ``history`` accept ``as_of`` to answer "what did I know on
date X": only memories stored by then count, and a memory counts as
current if it had not yet been superseded at that moment.
"""

from __future__ import annotations

import os as _os
import time as _time
import uuid
from datetime import date, datetime, time, timezone
from typing import Any, Dict, Iterable, List, Optional, Union

from .client import HalfLife, Record, SearchHit, payload_str
from .contradictions import Judge, detect_contradictions

_KEY = "memory_key"
_TEXT = "text"

AsOf = Union[int, float, str, datetime, date, None]


def _now_ms() -> int:
    return int(_time.time() * 1000)


def _as_of_ms(value: AsOf) -> Optional[int]:
    """Epoch ms for an ``as_of`` argument. A bare date means the end of that
    day (UTC); naive datetimes are UTC; strings are ISO 8601 or epoch ms."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise TypeError("as_of must be a datetime, date, ISO 8601 string or epoch ms")
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if s.lstrip("-").isdigit():
            return int(s)
        try:
            if len(s) == 10:
                value = date.fromisoformat(s)
            else:
                value = datetime.fromisoformat(s.replace("Z", "+00:00").replace("z", "+00:00"))
        except ValueError:
            raise ValueError(f"as_of: not an ISO 8601 date/time or epoch ms: {value!r}") from None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return int(value.timestamp() * 1000)
    if isinstance(value, date):
        end = datetime.combine(value, time.max, tzinfo=timezone.utc)
        return int(end.timestamp() * 1000)
    raise TypeError("as_of must be a datetime, date, ISO 8601 string or epoch ms")


def _and(*parts: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    present = [p for p in parts if p]
    if not present:
        return None
    return present[0] if len(present) == 1 else {"$and": present}


def _age(ms: int) -> str:
    secs = max(0, (_now_ms() - ms) // 1000)
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if secs >= size:
            return f"{secs // size}{unit} ago"
    return "just now"


class AgentMemory:
    """
    Args:
        db: ``tidevec.TideVec`` or ``tidevec.Local``.
        collection: collection holding the memories.
        embedder: object with ``dim``/``embed_query``; defaults to
            ``tidevec.embeddings.default_embedder()``.
        half_life_ms: how fast old memories lose rank.
        temporal_blend: weight of recency in recall ranking.
        supersede_threshold: vector_score ((1 + cos) / 2) at which an unkeyed
            memory counts as a restatement of an earlier one; 0.95 = cos 0.9.
        contradiction_judge: optional LLM judge; when set, each new memory
            is checked against similar ones and CONTRADICTS/SUPPORTS edges
            are recorded.
    """

    def __init__(self, db: Any, collection: str = "agent_memory", embedder: Any = None, *,
                 half_life_ms: int = HalfLife.ONE_MONTH, temporal_blend: float = 0.2,
                 supersede_threshold: float = 0.95,
                 contradiction_judge: Optional[Judge] = None):
        if embedder is None:
            from .embeddings import default_embedder
            embedder = default_embedder()
        self.db = db
        self.collection = collection
        self.embedder = embedder
        self.judge = contradiction_judge
        if not db.has_collection(collection):
            dim = getattr(embedder, "dim", None) or len(embedder.embed_query("probe"))
            db.create_collection(collection, dim=dim, half_life_ms=half_life_ms,
                                 temporal_blend=temporal_blend, supersession=True,
                                 supersede_threshold=supersede_threshold)

    # ---- writes ----------------------------------------------------

    def remember(self, text: str, key: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 created_at: Optional[int] = None) -> Dict[str, Any]:
        """Store a memory. Returns its id and the ids it superseded."""
        if not text.strip():
            raise ValueError("text must not be empty")
        mem_id = uuid.uuid4().hex
        ts = created_at or _now_ms()
        emb = self.embedder.embed_query(text)
        payload = {k: payload_str(v) for k, v in (metadata or {}).items()}
        payload[_TEXT] = text
        payload[_KEY] = key or ""

        superseded: List[str] = []
        edges = []
        if key:
            for prior in self._versions(key):
                if prior.created_at <= ts:
                    if not prior.superseded_by:
                        superseded.append(prior.id)
                        edges.append({"src": mem_id, "tgt": prior.id, "type": "UPDATES"})
                else:
                    edges.append({"src": prior.id, "tgt": mem_id, "type": "UPDATES"})

        self.db.upsert(self.collection, [{"id": mem_id, "embedding": emb,
                                          "payload": payload, "created_at": ts}])
        if edges:
            self.db.add_edges(self.collection, [dict(e, weight=1.0) for e in edges])

        out: Dict[str, Any] = {"id": mem_id, "superseded": superseded}
        if self.judge is not None:
            out["relations"] = detect_contradictions(
                self.db, self.collection, mem_id, text, emb, self.judge,
                text_key=_TEXT, skip_ids=superseded)
        return out

    def forget(self, memory_id: Optional[str] = None, key: Optional[str] = None) -> int:
        """Delete one memory by id, or every version of a key."""
        if not memory_id and not key:
            raise ValueError("forget() needs memory_id or key")
        n = 0
        if memory_id:
            n += self.db.delete(self.collection, [memory_id])
        if key:
            n += self.db.delete(self.collection, filter={_KEY: key})
        return n

    # ---- reads -----------------------------------------------------

    def _versions(self, key: str, as_of: Optional[int] = None) -> List[Record]:
        """Every stored version of ``key`` (created by ``as_of``), with
        supersession links. Scroll finds them exactly; get-by-id attaches
        the links, which scroll pages do not carry."""
        ids = [r.id for r in self.db.iterate(self.collection, filter={_KEY: key}, as_of=as_of)]
        return self.db.get_many(self.collection, ids) if ids else []

    def _superseded_by_as_of(self, links: Iterable[str], as_of: int,
                             created: Dict[str, int]) -> List[str]:
        """The subset of ``links`` whose newer version existed at ``as_of``."""
        links = list(links)
        unknown = [i for i in links if i not in created]
        if unknown:
            created.update({r.id: r.created_at
                            for r in self.db.get_many(self.collection, unknown,
                                                      with_payload=False)})
        return [i for i in links if i in created and created[i] <= as_of]

    @staticmethod
    def _as_dict(hit: SearchHit) -> Dict[str, Any]:
        out = AgentMemory._base_dict(hit.id, hit.payload, hit.created_at,
                                     hit.superseded_by)
        out.update(score=round(hit.score, 4), stale=hit.staleness_warning,
                   contradicted_by=hit.contradicted_by)
        return out

    @staticmethod
    def _base_dict(mem_id: str, payload: Dict[str, str], created_at: int,
                   superseded_by: List[str]) -> Dict[str, Any]:
        payload = dict(payload)
        text = payload.pop(_TEXT, "")
        key = payload.pop(_KEY, "")
        return {
            "id": mem_id,
            "text": text,
            "key": key or None,
            "created_at": datetime.fromtimestamp(created_at / 1000, timezone.utc).isoformat(),
            "age": _age(created_at),
            "superseded_by": list(superseded_by),
            "metadata": payload,
        }

    @staticmethod
    def _record_dict(rec: Record, superseded_by: Optional[List[str]] = None) -> Dict[str, Any]:
        out = AgentMemory._base_dict(
            rec.id, rec.payload, rec.created_at,
            rec.superseded_by if superseded_by is None else superseded_by)
        out["contradicts"] = [e["target_id"] for e in rec.edges
                              if e.get("type") == "CONTRADICTS"]
        return out

    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """One memory by id (None if absent)."""
        rec = self.db.get(self.collection, memory_id, with_vector=False)
        return self._record_dict(rec) if rec is not None else None

    def recall(self, query: str, k: int = 5, *, include_superseded: bool = False,
               key: Optional[str] = None, filter: Optional[Dict[str, Any]] = None,
               as_of: AsOf = None) -> List[Dict[str, Any]]:
        """Most relevant memories for ``query``, newest versions only by default.

        Args:
            key: only versions of this fact.
            filter: metadata filter, e.g. ``{"project": "alpha"}`` or
                ``{"priority": {"$gte": 2}}`` (same operators as search).
            as_of: recall as of this moment (datetime, date = end of that
                day UTC, ISO 8601 string or epoch ms): memories stored later
                are ignored and supersession is judged at that time.
        """
        emb = self.embedder.embed_query(query)
        fetch = k if include_superseded else max(k * 4, 20)
        hits = self.db.search(self.collection, emb, top_k=fetch,
                              filter=_and({_KEY: key} if key else None, filter),
                              as_of=_as_of_ms(as_of)).hits
        if not include_superseded:
            hits = [h for h in hits if not h.superseded_by]
        return [self._as_dict(h) for h in hits[:k]]

    def history(self, key: str, *, as_of: AsOf = None) -> List[Dict[str, Any]]:
        """Every version of a key, oldest first. With ``as_of``, only the
        versions stored by then, with supersession as it stood at that time."""
        cutoff = _as_of_ms(as_of)
        recs = sorted(self._versions(key, cutoff), key=lambda r: r.created_at)
        if cutoff is None:
            return [self._record_dict(r) for r in recs]
        created = {r.id: r.created_at for r in recs}
        return [self._record_dict(r, self._superseded_by_as_of(r.superseded_by, cutoff, created))
                for r in recs]


def default_memory_path() -> str:
    return _os.path.expanduser(_os.environ.get("TIDEVEC_MEMORY_PATH", "~/.tidevec/memory"))
