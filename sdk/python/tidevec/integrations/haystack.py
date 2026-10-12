"""
Haystack 2.x/3.x ``DocumentStore`` and embedding retriever backed by TideVec.

    from haystack import Document
    from tidevec.integrations.haystack import TideVecDocumentStore, TideVecEmbeddingRetriever

    store = TideVecDocumentStore(url="localhost:6399", collection="kb", embedding_dim=384,
                                 supersede_match_field="doc_id")
    store.write_documents([Document(content="Refunds take 3 days", embedding=emb,
                                    meta={"doc_id": "refunds"})])
    retriever = TideVecEmbeddingRetriever(document_store=store, top_k=5)
    retriever.run(query_embedding=query_emb)["documents"]   # current versions only

Filters use Haystack's syntax (``{"field": "meta.lang", "operator": "==", "value": "en"}``,
nested ``AND``/``OR``/``NOT``) and are evaluated with Haystack's own
``document_matches_filter``, so semantics match the in-memory store. The
parts TideVec can index (equality, ``in`` and numeric ranges on top-level
meta keys) are pushed down to the server as a pre-filter.

Limitations:
    * ``filter_documents``, ``count_documents(filters)`` and the SKIP/FAIL
      duplicate policies need the retrieval-completeness REST API
      (``/scroll`` and ``/get``, 0.3.0+); against older servers and native
      modules they enumerate the collection, which only suits small collections.
    * Documents without an embedding are stored with a placeholder vector
      and never returned by embedding retrieval.
    * ``blob`` and ``sparse_embedding`` are not stored.
"""

from __future__ import annotations

import json as _json
import os as _os
from typing import Any, Dict, List, Optional, Union

try:
    from haystack import Document, component, default_from_dict, default_to_dict
    from haystack.document_stores.errors import DuplicateDocumentError
    from haystack.document_stores.types import DuplicatePolicy, FilterPolicy
    from haystack.document_stores.types.filter_policy import apply_filter_policy
    from haystack.errors import FilterError
    from haystack.utils.filters import document_matches_filter
except ImportError as e:  # pragma: no cover
    raise ImportError('pip install "tidevec[haystack]"') from e

from ..client import TideVec, payload_str
from ._backend import Backend, Record, dumps

_CONTENT = "content"
_META = "_hs_meta"
_NO_EMBEDDING = "_hs_noemb"
_BATCH = 256

_RANGE = {">": "$gt", ">=": "$gte", "<": "$lt", "<=": "$lte"}


def _pushdown(filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """TideVec pre-filter that matches a superset of the Haystack filter, or None."""
    if not filters:
        return None
    op = filters.get("operator")
    if op in ("AND", "OR"):
        parts = [_pushdown(c) for c in filters.get("conditions", [])]
        if op == "AND":
            kept = [p for p in parts if p]
            return None if not kept else kept[0] if len(kept) == 1 else {"$and": kept}
        return {"$or": parts} if parts and all(parts) else None
    field, value = filters.get("field", ""), filters.get("value")
    if not field.startswith("meta.") or "." in field[5:]:
        return None
    key = field[5:]

    def scalar(v: Any) -> bool:
        return isinstance(v, (str, int, float)) and not isinstance(v, bool)

    if op == "==" and scalar(value):
        return {key: value}
    if op == "in" and isinstance(value, list) and value and all(scalar(v) for v in value):
        return {key: {"$in": value}}
    if op in _RANGE and isinstance(value, (int, float)) and not isinstance(value, bool):
        return {key: {_RANGE[op]: value}}
    return None


class TideVecDocumentStore:
    """
    Args:
        db: ``tidevec.TideVec`` or ``tidevec.Local``. When omitted, a client
            is built from ``url``/``api_key`` or an embedded ``Local(path)``.
        url: server ``host:port`` (default ``$TIDEVEC_URL`` or localhost:6399).
        api_key: defaults to ``$TIDEVEC_API_KEY``; not serialized by ``to_dict``.
        path: data directory for embedded mode (used when ``url`` is not set
            and ``db`` is not given).
        collection: collection name, created on first use.
        embedding_dim: dimension used when creating the collection.
        supersede_match_field: meta key whose equal values mark a newer
            document as superseding an older one (e.g. ``"doc_id"``).
        supersede_threshold: vector_score needed for supersession.
        **collection_kwargs: extra ``create_collection`` arguments.
    """

    def __init__(self, db: Any = None, *, url: Optional[str] = None, api_key: Optional[str] = None,
                 path: Optional[str] = None, collection: str = "haystack",
                 embedding_dim: int = 768, supersede_match_field: str = "",
                 supersede_threshold: float = 0.5, **collection_kwargs: Any):
        self.url = url
        self.path = path
        self.collection = collection
        self.embedding_dim = embedding_dim
        self.supersede_match_field = supersede_match_field
        self.supersede_threshold = supersede_threshold
        self.collection_kwargs = collection_kwargs
        if db is None:
            if url or not path:
                db = TideVec(url or _os.environ.get("TIDEVEC_URL", "localhost:6399"),
                             api_key=api_key or "")
            else:
                from ..local import Local
                db = Local(path)
        self.db = db
        if not db.has_collection(collection):
            kwargs = dict(collection_kwargs)
            if supersede_match_field:
                kwargs.update(supersession=True, supersede_match_field=supersede_match_field,
                              supersede_threshold=supersede_threshold)
            db.create_collection(collection, dim=embedding_dim, **kwargs)
        self._backend = Backend(db, collection)

    # ---- serialization ---------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        url = self.url
        if url is None and isinstance(self.db, TideVec):
            url = self.db._base_url.split("://", 1)[-1]
        path = self.path if self.path is not None else getattr(self.db, "path", None)
        return default_to_dict(self, url=url, path=None if url else path,
                               collection=self.collection, embedding_dim=self.embedding_dim,
                               supersede_match_field=self.supersede_match_field,
                               supersede_threshold=self.supersede_threshold,
                               **self.collection_kwargs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TideVecDocumentStore":
        return default_from_dict(cls, data)

    # ---- conversion ------------------------------------------------------

    def _to_vector(self, doc: Document) -> Dict[str, Any]:
        meta = dict(doc.meta or {})
        payload = {k: payload_str(v) for k, v in meta.items()
                   if isinstance(v, (str, int, float, bool)) and not k.startswith("_hs_")}
        payload[_META] = dumps(meta)
        if doc.content is not None:
            payload[_CONTENT] = doc.content
        if doc.embedding is not None:
            emb = [float(x) for x in doc.embedding]
        else:
            emb = [1.0] + [0.0] * (self.embedding_dim - 1)
            payload[_NO_EMBEDDING] = "1"
        vec: Dict[str, Any] = {"id": doc.id, "embedding": emb, "payload": payload}
        created = meta.get("created_at")
        if isinstance(created, (int, float)) and not isinstance(created, bool):
            vec["created_at"] = int(created)
        return vec

    @staticmethod
    def _to_document(rec: Record, *, score: Optional[float] = None,
                     with_tidevec_meta: bool = False) -> Document:
        payload = rec.payload
        meta = _json.loads(payload[_META]) if _META in payload else {
            k: v for k, v in payload.items() if k not in (_CONTENT, _NO_EMBEDDING)}
        if with_tidevec_meta:
            meta["tidevec"] = {"created_at": rec.created_at, "superseded_by": rec.superseded_by}
        embedding = rec.embedding if rec.embedding and _NO_EMBEDDING not in payload else None
        return Document(id=rec.id, content=payload.get(_CONTENT), meta=meta,
                        embedding=embedding, score=score)

    # ---- DocumentStore protocol -----------------------------------------

    def count_documents(self) -> int:
        return self._backend.count()

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        if filters and "operator" not in filters:
            raise FilterError("Invalid filter syntax. See https://docs.haystack.deepset.ai/docs/metadata-filtering")
        docs = [self._to_document(r) for r in self._backend.scan(_pushdown(filters), with_vector=True)]
        if filters:
            docs = [d for d in docs if document_matches_filter(filters, d)]
        return docs

    def write_documents(self, documents: List[Document],
                        policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
        for d in documents:
            if not isinstance(d, Document):
                raise ValueError("write_documents() expects a list of Documents")
        if policy == DuplicatePolicy.NONE:
            policy = DuplicatePolicy.FAIL
        unique: Dict[str, Document] = {}
        for d in documents:
            if d.id in unique and policy == DuplicatePolicy.FAIL:
                raise DuplicateDocumentError(f"ID '{d.id}' appears twice in the batch")
            if d.id not in unique or policy == DuplicatePolicy.OVERWRITE:
                unique[d.id] = d
        if policy in (DuplicatePolicy.SKIP, DuplicatePolicy.FAIL):
            existing = {r.id for r in self._backend.get(list(unique))}
            if existing and policy == DuplicatePolicy.FAIL:
                raise DuplicateDocumentError(f"IDs already exist: {sorted(existing)}")
            unique = {k: v for k, v in unique.items() if k not in existing}
        docs = list(unique.values())
        for i in range(0, len(docs), _BATCH):
            self._backend.upsert([self._to_vector(d) for d in docs[i:i + _BATCH]])
        return len(docs)

    def delete_documents(self, document_ids: List[str]) -> None:
        if document_ids:
            self._backend.delete(ids=document_ids)

    # ---- retrieval -------------------------------------------------------

    def embedding_retrieval(self, query_embedding: List[float], *,
                            filters: Optional[Dict[str, Any]] = None, top_k: int = 10,
                            exclude_superseded: bool = True,
                            temporal_blend: Optional[float] = None,
                            scale_score: bool = True) -> List[Document]:
        fetch = top_k * 4 + 10 if (filters or exclude_superseded) else top_k + 5
        recs = self._backend.search(query_embedding, fetch, _pushdown(filters),
                                    temporal_blend=temporal_blend)
        out: List[Document] = []
        for r in recs:
            if _NO_EMBEDDING in r.payload or (exclude_superseded and r.superseded_by):
                continue
            score = r.score if scale_score or r.score is None else 2.0 * r.score - 1.0
            doc = self._to_document(r, score=score, with_tidevec_meta=True)
            if filters and not document_matches_filter(filters, doc):
                continue
            out.append(doc)
            if len(out) >= top_k:
                break
        return out


@component
class TideVecEmbeddingRetriever:
    """
    Retrieves documents from a ``TideVecDocumentStore`` by embedding. Superseded
    document versions are dropped unless ``exclude_superseded=False``; each
    returned document carries ``meta["tidevec"]`` with ``created_at`` and
    ``superseded_by``.
    """

    def __init__(self, document_store: TideVecDocumentStore, *,
                 filters: Optional[Dict[str, Any]] = None, top_k: int = 10,
                 exclude_superseded: bool = True, temporal_blend: Optional[float] = None,
                 filter_policy: Union[str, FilterPolicy] = FilterPolicy.REPLACE):
        if not isinstance(document_store, TideVecDocumentStore):
            raise ValueError("document_store must be a TideVecDocumentStore")
        self.document_store = document_store
        self.filters = filters or {}
        self.top_k = top_k
        self.exclude_superseded = exclude_superseded
        self.temporal_blend = temporal_blend
        self.filter_policy = (FilterPolicy.from_str(filter_policy)
                              if isinstance(filter_policy, str) else filter_policy)

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(self, document_store=self.document_store.to_dict(),
                               filters=self.filters, top_k=self.top_k,
                               exclude_superseded=self.exclude_superseded,
                               temporal_blend=self.temporal_blend,
                               filter_policy=self.filter_policy.value)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TideVecEmbeddingRetriever":
        params = data["init_parameters"]
        params["document_store"] = TideVecDocumentStore.from_dict(params["document_store"])
        if "filter_policy" in params:
            params["filter_policy"] = FilterPolicy.from_str(params["filter_policy"])
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(self, query_embedding: List[float], filters: Optional[Dict[str, Any]] = None,
            top_k: Optional[int] = None) -> Dict[str, List[Document]]:
        filters = apply_filter_policy(self.filter_policy, self.filters, filters)
        docs = self.document_store.embedding_retrieval(
            query_embedding, filters=filters or None, top_k=top_k or self.top_k,
            exclude_superseded=self.exclude_superseded, temporal_blend=self.temporal_blend)
        return {"documents": docs}
