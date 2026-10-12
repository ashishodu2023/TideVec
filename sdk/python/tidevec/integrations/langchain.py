"""
LangChain vector store backed by TideVec.

    from tidevec import Local
    from tidevec.integrations.langchain import TideVecVectorStore

    store = TideVecVectorStore(Local("./data"), "kb", embeddings,
                               supersession=True, supersede_match_field="doc_id")
    store.add_texts(["Refunds take 14 days"], metadatas=[{"doc_id": "refunds"}])
    store.add_texts(["Refunds now take 3 days"], metadatas=[{"doc_id": "refunds"}])
    docs = store.similarity_search("how long do refunds take?", k=3,
                                   exclude_superseded=True)

Works with the REST client (``tidevec.TideVec``) or embedded mode
(``tidevec.Local``). Metadata round-trips exactly; top-level scalar keys
are also stored flat so they can be used in filters — ``{"key": value}``,
or the full TideVec filter syntax (``{"year": {"$gte": 2020}}``,
``{"$or": [...]}``, ``{"lang": {"$in": [...]}}``, ...).
Each returned Document carries ``metadata["tidevec"]`` with the score
breakdown, staleness warning and ``superseded_by`` ids.
"""

from __future__ import annotations

import json as _json
import uuid
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore
except ImportError as e:  # pragma: no cover
    raise ImportError('pip install "tidevec[langchain]"') from e

from ..client import Record, SearchHit, payload_str

_META_KEY = "_lc_metadata"
_BATCH = 256


def _filter(filter: Union[None, str, dict]) -> Union[None, str, dict]:
    return filter or None


class TideVecVectorStore(VectorStore):
    """
    Args:
        client: ``tidevec.TideVec`` or ``tidevec.Local``.
        collection: collection name (created on first use if missing).
        embedding: LangChain ``Embeddings`` (or a ``tidevec.embeddings`` embedder).
        text_key: payload key holding the document text.
        temporal_blend: default per-query blend (None = collection setting).
        superseded_penalty: default score penalty for superseded documents.
        **collection_kwargs: passed to ``create_collection`` when the
            collection does not exist (e.g. ``half_life_ms``,
            ``supersession=True``, ``supersede_match_field="doc_id"``).
    """

    def __init__(self, client: Any, collection: str, embedding: Embeddings, *,
                 text_key: str = "text", temporal_blend: Optional[float] = None,
                 superseded_penalty: float = 0.0, **collection_kwargs: Any):
        self._client = client
        self.collection = collection
        self._embedding = embedding
        self.text_key = text_key
        self.temporal_blend = temporal_blend
        self.superseded_penalty = superseded_penalty
        if not client.has_collection(collection):
            dim = getattr(embedding, "dim", None) or len(embedding.embed_query("dimension probe"))
            client.create_collection(collection, dim=dim, **collection_kwargs)

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    @property
    def client(self) -> Any:
        return self._client

    # ---- writes ----------------------------------------------------

    def add_texts(self, texts: Iterable[str], metadatas: Optional[List[dict]] = None, *,
                  ids: Optional[List[str]] = None,
                  created_at: Optional[Union[int, Sequence[int]]] = None,
                  **kwargs: Any) -> List[str]:
        """
        Embed and upsert texts. ``created_at`` (ms since epoch, scalar or
        one per text) backdates documents; a numeric ``metadata["created_at"]``
        is used the same way.
        """
        texts = list(texts)
        metadatas = metadatas or [{} for _ in texts]
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        if not (len(texts) == len(metadatas) == len(ids)):
            raise ValueError("texts, metadatas and ids must have the same length")
        stamps: List[Optional[int]]
        if created_at is None:
            stamps = [None] * len(texts)
        elif isinstance(created_at, int):
            stamps = [created_at] * len(texts)
        else:
            stamps = list(created_at)

        for start in range(0, len(texts), _BATCH):
            chunk = slice(start, start + _BATCH)
            vectors = []
            embs = self._embedding.embed_documents(texts[chunk])
            for text, meta, vid, emb, ts in zip(texts[chunk], metadatas[chunk],
                                                 ids[chunk], embs, stamps[chunk]):
                payload = {k: payload_str(v) for k, v in meta.items()
                           if isinstance(v, (str, int, float, bool))}
                payload[self.text_key] = text
                payload[_META_KEY] = _json.dumps(meta, default=str)
                vec = {"id": vid, "embedding": list(emb), "payload": payload}
                if ts is None and isinstance(meta.get("created_at"), (int, float)):
                    ts = int(meta["created_at"])
                if ts is not None:
                    vec["created_at"] = int(ts)
                vectors.append(vec)
            self._client.upsert(self.collection, vectors)
        return ids

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete by ids and/or by a ``filter`` (dict or ``"key=value"``)."""
        flt = _filter(kwargs.get("filter"))
        if ids is None and flt is None:
            return False
        self._client.delete(self.collection, ids, filter=flt)
        return True

    # ---- reads -----------------------------------------------------

    def _metadata(self, payload: dict) -> Tuple[str, dict]:
        payload = dict(payload)
        text = payload.pop(self.text_key, "")
        raw = payload.pop(_META_KEY, None)
        return text, (_json.loads(raw) if raw else payload)

    def get_by_ids(self, ids: Sequence[str], /) -> List[Document]:
        """Documents for the given ids (missing ids are skipped)."""
        if not ids:
            return []
        records: List[Record] = self._client.get_many(self.collection, list(ids))
        docs = []
        for r in records:
            text, metadata = self._metadata(r.payload)
            metadata["tidevec"] = {"created_at": r.created_at,
                                   "superseded_by": r.superseded_by}
            docs.append(Document(id=r.id, page_content=text, metadata=metadata))
        return docs

    def _to_document(self, hit: SearchHit) -> Document:
        text, metadata = self._metadata(hit.payload)
        metadata["tidevec"] = {
            "score": hit.score,
            "vector_score": hit.vector_score,
            "temporal_score": hit.temporal_score,
            "created_at": hit.created_at,
            "staleness_warning": hit.staleness_warning,
            "superseded_by": hit.superseded_by,
            "contradicted_by": hit.contradicted_by,
        }
        return Document(id=hit.id, page_content=text, metadata=metadata)

    def similarity_search_with_score_by_vector(
            self, embedding: List[float], k: int = 4,
            filter: Union[None, str, dict] = None, *,
            exclude_superseded: bool = False, **kwargs: Any) -> List[Tuple[Document, float]]:
        fetch = k * 3 if exclude_superseded else k
        resp = self._client.search(
            self.collection, list(embedding), top_k=fetch, filter=_filter(filter) or "",
            as_of=kwargs.get("as_of"),
            temporal_blend=kwargs.get("temporal_blend", self.temporal_blend),
            superseded_penalty=kwargs.get("superseded_penalty", self.superseded_penalty))
        hits = [h for h in resp.hits if not (exclude_superseded and h.superseded_by)]
        return [(self._to_document(h), h.score) for h in hits[:k]]

    def similarity_search_with_score(self, query: str, k: int = 4,
                                     filter: Union[None, str, dict] = None,
                                     **kwargs: Any) -> List[Tuple[Document, float]]:
        return self.similarity_search_with_score_by_vector(
            self._embedding.embed_query(query), k, filter, **kwargs)

    def similarity_search_by_vector(self, embedding: List[float], k: int = 4,
                                    filter: Union[None, str, dict] = None,
                                    **kwargs: Any) -> List[Document]:
        return [d for d, _ in self.similarity_search_with_score_by_vector(
            embedding, k, filter, **kwargs)]

    def similarity_search(self, query: str, k: int = 4,
                          filter: Union[None, str, dict] = None,
                          **kwargs: Any) -> List[Document]:
        return [d for d, _ in self.similarity_search_with_score(query, k, filter, **kwargs)]

    def _select_relevance_score_fn(self):
        # Scores are already similarities blended with recency, in [0, 1].
        return lambda score: score

    # ---- construction ----------------------------------------------

    @classmethod
    def from_texts(cls, texts: List[str], embedding: Embeddings,
                   metadatas: Optional[List[dict]] = None, *,
                   ids: Optional[List[str]] = None, client: Any = None,
                   collection: str = "langchain", path: str = "./tidevec_data",
                   **kwargs: Any) -> "TideVecVectorStore":
        """Build a store from texts. Without ``client`` an embedded
        ``tidevec.Local(path)`` database is used."""
        if client is None:
            from ..local import Local
            client = Local(path)
        store = cls(client, collection, embedding, **kwargs)
        store.add_texts(texts, metadatas, ids=ids)
        return store
