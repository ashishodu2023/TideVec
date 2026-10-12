"""
LangGraph long-term memory store backed by TideVec.

    from langgraph.prebuilt import create_react_agent
    from tidevec import TideVec
    from tidevec.integrations.langgraph import TideVecStore

    store = TideVecStore(TideVec("localhost:6399"), collection="lg_memories",
                         index={"dims": 1536, "embed": "openai:text-embedding-3-small"})
    store.put(("users", "u1", "memories"), "diet", {"text": "User is vegetarian"})
    store.search(("users", "u1"), query="what does the user eat?", limit=3)

    agent = create_react_agent(model, tools, store=store)

``TideVecStore`` implements ``langgraph.store.base.BaseStore`` (get, put,
delete, search with namespace prefixes, value filters and semantic query,
list_namespaces, TTL). Items keep ``created_at`` across updates and
``updated_at`` is the vector timestamp, so the collection's temporal decay
ranks recently-updated memories higher when ``temporal_blend > 0``.

For versioned agent memory (supersession chains, history, contradiction
flags) use :func:`memory_tools`, which exposes ``tidevec.AgentMemory`` as
LangChain tools for any LangGraph agent.

Limitations:
    * Value filters are evaluated client-side over a candidate set fetched
      by namespace prefix; without the retrieval-completeness REST API
      (servers before 0.4, embedded mode) listing enumerates the
      collection, so keep those collections small.
    * Namespace prefixes are matched up to ``max_namespace_depth`` (8)
      labels on the server; deeper prefixes are completed client-side.
"""

from __future__ import annotations

import asyncio
import hashlib
import json as _json
import operator
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from langgraph.store.base import (
        BaseStore,
        GetOp,
        IndexConfig,
        Item,
        ListNamespacesOp,
        MatchCondition,
        Op,
        PutOp,
        Result,
        SearchItem,
        SearchOp,
        TTLConfig,
    )
    from langgraph.store.base.embed import ensure_embeddings, get_text_at_path, tokenize_path
except ImportError as e:  # pragma: no cover
    raise ImportError('pip install "tidevec[langgraph]"') from e

from ._backend import Backend, Record, dumps

_NS = "_lg_ns"
_KEY = "_lg_key"
_VALUE = "_lg_value"
_CREATED = "_lg_created"
_DEPTH = "_lg_depth"
_NO_EMBEDDING = "_lg_noemb"
_MAX_DEPTH = 8
_DEFAULT_DIM = 16


def _prefix_key(i: int) -> str:
    return f"_lg_p{i}"


def _item_id(namespace: Tuple[str, ...], key: str) -> str:
    h = hashlib.sha256(_json.dumps([list(namespace), key]).encode()).hexdigest()
    return f"lg-{h[:32]}"


def _dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


_OPS = {"$eq": operator.eq, "$ne": operator.ne, "$gt": operator.gt,
        "$gte": operator.ge, "$lt": operator.lt, "$lte": operator.le}


def _compare(actual: Any, cond: Any) -> bool:
    if not isinstance(cond, dict):
        return actual == cond
    for op, want in cond.items():
        if op not in _OPS:
            raise ValueError(f"unsupported filter operator {op!r}")
        try:
            if not _OPS[op](actual, want):
                return False
        except TypeError:
            return False
    return True


def _value_matches(value: Dict[str, Any], flt: Optional[Dict[str, Any]]) -> bool:
    if not flt:
        return True
    for path, cond in flt.items():
        actual: Any = value
        for part in path.split("."):
            if not isinstance(actual, dict) or part not in actual:
                actual = None
                break
            actual = actual[part]
        if not _compare(actual, cond):
            return False
    return True


def _ns_matches(namespace: Tuple[str, ...], cond: MatchCondition) -> bool:
    path = tuple(cond.path)
    if len(namespace) < len(path):
        return False
    window = namespace[:len(path)] if cond.match_type == "prefix" else namespace[len(namespace) - len(path):]
    return all(p == "*" or p == n for p, n in zip(path, window))


class TideVecStore(BaseStore):
    """
    Args:
        db: ``tidevec.TideVec`` or ``tidevec.Local``.
        collection: collection name, created on first use.
        index: LangGraph ``IndexConfig`` (``dims``, ``embed``, optional
            ``fields``) enabling semantic ``search(query=...)``. Without it,
            ``query`` is ignored and search lists items newest first.
        ttl: LangGraph ``TTLConfig``; ``default_ttl`` (minutes) applies to
            puts without an explicit ``ttl``. Expiry uses TideVec's per-vector
            TTL. ``refresh_on_read`` is not supported (TideVec TTLs are fixed
            at write time).
        temporal_blend: recency weight for semantic search (None = the
            collection's setting).
        **collection_kwargs: extra ``create_collection`` arguments.
    """

    supports_ttl = True

    def __init__(self, db: Any, *, collection: str = "langgraph_store",
                 index: Optional[IndexConfig] = None, ttl: Optional[TTLConfig] = None,
                 temporal_blend: Optional[float] = None, **collection_kwargs: Any):
        self.db = db
        self.collection = collection
        self.ttl_config = ttl
        self.temporal_blend = temporal_blend
        self.index_config: Optional[Dict[str, Any]] = None
        self.embeddings = None
        if index:
            self.index_config = dict(index)
            self.embeddings = ensure_embeddings(index.get("embed"))
            fields = index.get("fields") or ["$"]
            self.index_config["__tokenized_fields"] = [
                (f, "$" if f == "$" else tokenize_path(f)) for f in fields]
        self.dim = int(self.index_config["dims"]) if self.index_config else _DEFAULT_DIM
        if not db.has_collection(collection):
            db.create_collection(collection, dim=self.dim, **collection_kwargs)
        self._backend = Backend(db, collection)

    # ---- helpers ---------------------------------------------------------

    def _ns_filter(self, prefix: Sequence[str]) -> Optional[Dict[str, Any]]:
        conds = {_prefix_key(i): label for i, label in enumerate(prefix[:_MAX_DEPTH])}
        return conds or None

    def _to_item(self, rec: Record, cls: type = Item, score: Optional[float] = None) -> Any:
        p = rec.payload
        ns = tuple(_json.loads(p[_NS]))
        created = int(p.get(_CREATED, rec.created_at))
        kwargs = dict(namespace=ns, key=p[_KEY], value=_json.loads(p[_VALUE]),
                      created_at=_dt(created), updated_at=_dt(rec.created_at))
        if cls is SearchItem:
            kwargs["score"] = score
        return cls(**kwargs)

    def _texts(self, value: Dict[str, Any], index: Any) -> List[str]:
        if index is False or not self.index_config:
            return []
        if isinstance(index, list):
            paths = [(f, "$" if f == "$" else tokenize_path(f)) for f in index]
        else:
            paths = self.index_config["__tokenized_fields"]
        texts: List[str] = []
        for _, path in paths:
            texts.extend(t for t in get_text_at_path(value, path) if t)
        return texts

    def _get_records(self, keys: List[Tuple[Tuple[str, ...], str]]) -> Dict[str, Record]:
        ids = [_item_id(ns, k) for ns, k in keys]
        recs = self._backend.get(ids)
        now = time.time() * 1000
        out = {}
        for r in recs:
            exp = r.payload.get("_lg_expires")
            if exp and float(exp) <= now:
                continue
            out[r.id] = r
        return out

    # ---- batch -----------------------------------------------------------

    def batch(self, ops: Iterable[Op]) -> List[Result]:
        ops = list(ops)
        results: List[Result] = [None] * len(ops)

        # Last write per (namespace, key) wins, as in InMemoryStore.
        puts: Dict[Tuple[Tuple[str, ...], str], PutOp] = {}
        for op in ops:
            if isinstance(op, PutOp):
                puts[(tuple(op.namespace), op.key)] = op
        if puts:
            self._apply_puts(list(puts.values()))

        gets = [(i, op) for i, op in enumerate(ops) if isinstance(op, GetOp)]
        if gets:
            found = self._get_records([(tuple(op.namespace), op.key) for _, op in gets])
            for i, op in gets:
                rec = found.get(_item_id(tuple(op.namespace), op.key))
                results[i] = self._to_item(rec) if rec else None

        for i, op in enumerate(ops):
            if isinstance(op, SearchOp):
                results[i] = self._search(op)
            elif isinstance(op, ListNamespacesOp):
                results[i] = self._list_namespaces(op)
        return results

    async def abatch(self, ops: Iterable[Op]) -> List[Result]:
        return await asyncio.get_running_loop().run_in_executor(None, self.batch, list(ops))

    def _apply_puts(self, puts: List[PutOp]) -> None:
        deletes = [_item_id(tuple(op.namespace), op.key) for op in puts if op.value is None]
        writes = [op for op in puts if op.value is not None]
        if deletes:
            self._backend.delete(ids=deletes)
        if not writes:
            return
        existing = self._get_records([(tuple(op.namespace), op.key) for op in writes])
        texts_per_op = [self._texts(op.value, op.index) for op in writes]
        flat = [t for texts in texts_per_op for t in texts]
        vectors = self.embeddings.embed_documents(flat) if flat and self.embeddings else []
        now = int(time.time() * 1000)
        rows = []
        offset = 0
        for op, texts in zip(writes, texts_per_op):
            ns = tuple(op.namespace)
            vid = _item_id(ns, op.key)
            if texts:
                # Several indexed fields: average them (cosine search over the mean).
                chunk = vectors[offset:offset + len(texts)]
                offset += len(texts)
                emb = [sum(col) / len(chunk) for col in zip(*chunk)]
                no_emb = False
            else:
                emb, no_emb = [1.0] + [0.0] * (self.dim - 1), True
            prev = existing.get(vid)
            created = int(prev.payload.get(_CREATED, prev.created_at)) if prev else now
            payload = {_NS: dumps(list(ns)), _KEY: op.key, _VALUE: dumps(op.value),
                       _CREATED: str(created), _DEPTH: str(len(ns))}
            for i, label in enumerate(ns[:_MAX_DEPTH]):
                payload[_prefix_key(i)] = label
            if no_emb:
                payload[_NO_EMBEDDING] = "1"
            row: Dict[str, Any] = {"id": vid, "embedding": [float(x) for x in emb],
                                   "payload": payload, "created_at": now}
            ttl = op.ttl
            if ttl is None and self.ttl_config:
                ttl = self.ttl_config.get("default_ttl")
            if ttl:
                row["ttl_seconds"] = max(1, int(round(ttl * 60)))
                payload["_lg_expires"] = str(now + int(ttl * 60_000))
            rows.append(row)
        self._backend.upsert(rows)

    def _candidates(self, prefix: Sequence[str]) -> List[Record]:
        now = time.time() * 1000
        out = []
        for r in self._backend.scan(self._ns_filter(prefix)):
            if _NS not in r.payload:
                continue
            ns = tuple(_json.loads(r.payload[_NS]))
            exp = r.payload.get("_lg_expires")
            if ns[:len(prefix)] == tuple(prefix) and not (exp and float(exp) <= now):
                out.append(r)
        return out

    def _search(self, op: SearchOp) -> List[SearchItem]:
        prefix = tuple(op.namespace_prefix)
        if op.query and self.embeddings:
            qvec = self.embeddings.embed_query(op.query)
            want = op.offset + op.limit
            fetch = want * 4 + 20 if op.filter else want + 10
            now = time.time() * 1000
            scored: List[SearchItem] = []
            seen = set()
            recs = self._backend.search(qvec, fetch, self._ns_filter(prefix),
                                        temporal_blend=self.temporal_blend)
            for r in recs:
                p = r.payload
                if _NS not in p or _NO_EMBEDDING in p or r.id in seen:
                    continue
                exp = p.get("_lg_expires")
                if exp and float(exp) <= now:
                    continue
                if tuple(_json.loads(p[_NS]))[:len(prefix)] != prefix:
                    continue
                if not _value_matches(_json.loads(p[_VALUE]), op.filter):
                    continue
                seen.add(r.id)
                scored.append(self._to_item(r, SearchItem, score=r.score))
            page = scored[op.offset:want]
            if len(page) < op.limit:
                # As in InMemoryStore: fill with items that have no embedding.
                for r in self._candidates(prefix):
                    if len(page) >= op.limit:
                        break
                    if _NO_EMBEDDING in r.payload and _value_matches(
                            _json.loads(r.payload[_VALUE]), op.filter):
                        page.append(self._to_item(r, SearchItem))
            return page
        recs = [r for r in self._candidates(prefix)
                if _value_matches(_json.loads(r.payload[_VALUE]), op.filter)]
        recs.sort(key=lambda r: r.created_at, reverse=True)
        return [self._to_item(r, SearchItem) for r in recs[op.offset:op.offset + op.limit]]

    def _list_namespaces(self, op: ListNamespacesOp) -> List[Tuple[str, ...]]:
        prefix: Tuple[str, ...] = ()
        for cond in op.match_conditions or ():
            if cond.match_type == "prefix" and "*" not in cond.path:
                prefix = tuple(cond.path)
                break
        namespaces = {tuple(_json.loads(r.payload[_NS])) for r in self._candidates(prefix)}
        if op.match_conditions:
            namespaces = {ns for ns in namespaces
                          if all(_ns_matches(ns, c) for c in op.match_conditions)}
        if op.max_depth is not None:
            namespaces = {ns[:op.max_depth] for ns in namespaces}
        ordered = sorted(namespaces)
        return ordered[op.offset:op.offset + op.limit]


def memory_tools(memory: Any, *, namespace_key: Optional[str] = None) -> List[Any]:
    """
    LangChain tools (``remember``, ``recall``, ``memory_history``) over a
    ``tidevec.AgentMemory``, for ``create_react_agent(model, tools=...)``.
    Remembering with a ``key`` that already exists supersedes the old fact.

    Args:
        memory: a ``tidevec.AgentMemory``.
        namespace_key: optional prefix prepended to every key (e.g. a user id).
    """
    from langchain_core.tools import tool

    def full_key(key: Optional[str]) -> Optional[str]:
        if not key:
            return None
        return f"{namespace_key}:{key}" if namespace_key else key

    @tool
    def remember(text: str, key: Optional[str] = None) -> str:
        """Store a fact in long-term memory. Pass a stable `key` (e.g. "user.diet")
        so a later fact with the same key replaces this one."""
        out = memory.remember(text, key=full_key(key))
        note = f" (replaced {len(out['superseded'])} older version(s))" if out["superseded"] else ""
        return f"stored memory {out['id']}{note}"

    @tool
    def recall(query: str, k: int = 5) -> str:
        """Retrieve the most relevant current memories for a query."""
        mems = memory.recall(query, k=k)
        if not mems:
            return "no relevant memories"
        return "\n".join(f"- {m['text']} ({m['age']})" for m in mems)

    @tool
    def memory_history(key: str) -> str:
        """Show every version of a remembered fact, oldest first."""
        versions = memory.history(full_key(key) or key)
        if not versions:
            return f"no memories with key {key!r}"
        return "\n".join(f"- {m['text']}" + (" [superseded]" if m["superseded_by"] else "")
                         for m in versions)

    return [remember, recall, memory_history]


__all__ = ["TideVecStore", "memory_tools"]
