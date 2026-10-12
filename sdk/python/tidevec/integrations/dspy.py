"""
DSPy retriever backed by TideVec.

    import dspy
    from tidevec import TideVec
    from tidevec.integrations.dspy import TideVecRetriever

    retrieve = TideVecRetriever(TideVec("localhost:6399"), "kb",
                                embedder=dspy.Embedder("openai/text-embedding-3-small"), k=5)
    retrieve.add(["Refunds take 3 days"], metadatas=[{"doc_id": "refunds"}])

    class RAG(dspy.Module):
        def __init__(self):
            self.retrieve = retrieve
            self.answer = dspy.ChainOfThought("context, question -> answer")
        def forward(self, question):
            context = self.retrieve(question).passages
            return self.answer(context=context, question=question)

Results drop document versions that a newer one superseded (set
``exclude_superseded=False`` to keep them), and are ranked by similarity
blended with recency according to the collection's temporal settings.
``retrieve.as_rm()`` returns a callable for the legacy
``dspy.configure(rm=...)`` + ``dspy.Retrieve(k)`` pattern.
"""

from __future__ import annotations

import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

try:
    import dspy as _dspy
    from dspy import Prediction
    from dspy.dsp.utils import dotdict
except ImportError as e:  # pragma: no cover
    raise ImportError('pip install "tidevec[dspy]"') from e

from ..client import payload_str


def _embed_fn(embedder: Any) -> Callable[[List[str]], List[List[float]]]:
    """Accept a tidevec/LangChain embedder, a ``dspy.Embedder`` or a callable."""
    if hasattr(embedder, "embed_documents"):
        return lambda texts: [list(map(float, v)) for v in embedder.embed_documents(texts)]
    if callable(embedder):
        def call(texts: List[str]) -> List[List[float]]:
            out = embedder(texts)
            rows = out.tolist() if hasattr(out, "tolist") else list(out)
            if rows and not isinstance(rows[0], (list, tuple)):
                rows = [rows]
            return [list(map(float, r)) for r in rows]
        return call
    raise TypeError("embedder must have embed_documents() or be callable")


class TideVecRetriever(_dspy.Retrieve):
    """
    Args:
        db: ``tidevec.TideVec`` or ``tidevec.Local``.
        collection: collection name; created on the first ``add`` if missing.
        embedder: ``dspy.Embedder``, a ``tidevec.embeddings`` embedder, a
            LangChain ``Embeddings`` or ``fn(list[str]) -> vectors``.
        k: default number of passages.
        text_key: payload key holding the passage text.
        filter: fixed payload filter (``"key=value"``) applied to every query.
        exclude_superseded: drop passages a newer version supersedes.
        temporal_blend: per-query recency weight (None = collection setting).
        **collection_kwargs: ``create_collection`` arguments, e.g.
            ``supersession=True, supersede_match_field="doc_id"``.
    """

    def __init__(self, db: Any, collection: str, embedder: Any, k: int = 3, *,
                 text_key: str = "text", filter: str = "", exclude_superseded: bool = True,
                 temporal_blend: Optional[float] = None, callbacks: Optional[list] = None,
                 **collection_kwargs: Any):
        super().__init__(k=k, callbacks=callbacks)
        self.db = db
        self.collection = collection
        self.text_key = text_key
        self.filter = filter
        self.exclude_superseded = exclude_superseded
        self.temporal_blend = temporal_blend
        self.collection_kwargs = collection_kwargs
        self._embed = _embed_fn(embedder)

    def _ensure_collection(self, dim: int) -> None:
        if not self.db.has_collection(self.collection):
            self.db.create_collection(self.collection, dim=dim, **self.collection_kwargs)

    def add(self, texts: Sequence[str], metadatas: Optional[Sequence[Dict[str, Any]]] = None, *,
            ids: Optional[Sequence[str]] = None,
            created_at: Optional[Union[int, Sequence[int]]] = None) -> List[str]:
        """Embed and store passages. ``created_at`` (ms) backdates them."""
        texts = list(texts)
        metadatas = list(metadatas or [{} for _ in texts])
        ids = list(ids or [uuid.uuid4().hex for _ in texts])
        stamps = ([created_at] * len(texts) if isinstance(created_at, int)
                  else list(created_at or [None] * len(texts)))
        if not (len(texts) == len(metadatas) == len(ids) == len(stamps)):
            raise ValueError("texts, metadatas, ids and created_at must have the same length")
        vectors = self._embed(texts)
        if vectors:
            self._ensure_collection(len(vectors[0]))
        rows = []
        for text, meta, vid, emb, ts in zip(texts, metadatas, ids, vectors, stamps):
            payload = {k: payload_str(v) for k, v in meta.items()}
            payload[self.text_key] = text
            row: Dict[str, Any] = {"id": vid, "embedding": emb, "payload": payload}
            if ts is not None:
                row["created_at"] = int(ts)
            rows.append(row)
        for i in range(0, len(rows), 256):
            self.db.upsert(self.collection, rows[i:i + 256])
        return ids

    def _search(self, query: str, k: int, filter: str) -> List[Dict[str, Any]]:
        emb = self._embed([query])[0]
        fetch = max(k * 4, 20) if self.exclude_superseded else k
        hits = self.db.search(self.collection, emb, top_k=fetch, filter=filter,
                              temporal_blend=self.temporal_blend).hits
        if self.exclude_superseded:
            hits = [h for h in hits if not h.superseded_by]
        out = []
        for h in hits[:k]:
            meta = dict(h.payload)
            text = meta.pop(self.text_key, "")
            out.append({"id": h.id, "text": text, "score": h.score,
                        "created_at": h.created_at, "metadata": meta})
        return out

    def forward(self, query: Union[str, List[str]], k: Optional[int] = None,
                filter: Optional[str] = None, **kwargs: Any) -> Prediction:
        """Passages for one query, or merged (best score per passage) for a list of queries."""
        k = k if k is not None else self.k
        flt = self.filter if filter is None else filter
        queries = [query] if isinstance(query, str) else [q for q in query if q]
        best: Dict[str, Dict[str, Any]] = {}
        for q in queries:
            for r in self._search(q, k, flt):
                if r["id"] not in best or r["score"] > best[r["id"]]["score"]:
                    best[r["id"]] = r
        ranked = sorted(best.values(), key=lambda r: r["score"], reverse=True)[:k]
        return Prediction(passages=[r["text"] for r in ranked], ids=[r["id"] for r in ranked],
                          scores=[r["score"] for r in ranked],
                          metadata=[r["metadata"] for r in ranked])

    def as_rm(self) -> Callable[..., List[dotdict]]:
        """Callable for ``dspy.configure(rm=...)``, returning ``dotdict(long_text=...)`` items."""
        def rm(query: Union[str, List[str]], k: Optional[int] = None, **kwargs: Any) -> List[dotdict]:
            pred = self.forward(query, k=k, **kwargs)
            return [dotdict(long_text=t, id=i, score=s)
                    for t, i, s in zip(pred.passages, pred.ids, pred.scores)]
        return rm
