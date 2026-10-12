"""
LlamaIndex vector store backed by TideVec.

    from llama_index.core import StorageContext, VectorStoreIndex
    from tidevec import Local
    from tidevec.integrations.llamaindex import TideVecVectorStore

    vector_store = TideVecVectorStore(Local("./data"), "kb",
                                      supersession=True, supersede_match_field="ref_doc_id")
    index = VectorStoreIndex.from_documents(
        docs, storage_context=StorageContext.from_defaults(vector_store=vector_store))

The collection is created on the first ``add`` using the embedding size.
Filters: ``MetadataFilters`` with ==, !=, >, >=, <, <=, in, nin and
is_empty, nested, combined with AND / OR / NOT. ``node_ids`` and
``doc_ids`` restrict a query. Node scores are TideVec's blended
(similarity + recency) scores; each returned node's metadata gains
``tidevec_superseded_by`` and ``tidevec_staleness_warning``.
"""

from __future__ import annotations

import json as _json
from typing import Any, Dict, List, Optional, Sequence

try:
    from llama_index.core.bridge.pydantic import PrivateAttr
    from llama_index.core.schema import BaseNode
    from llama_index.core.vector_stores.types import (
        BasePydanticVectorStore,
        FilterCondition,
        FilterOperator,
        MetadataFilter,
        MetadataFilters,
        VectorStoreQuery,
        VectorStoreQueryResult,
    )
    from llama_index.core.vector_stores.utils import (
        metadata_dict_to_node,
        node_to_metadata_dict,
    )
except ImportError as e:  # pragma: no cover
    raise ImportError('pip install "tidevec[llamaindex]"') from e

from ..client import payload_str

_NODE_KEY = "_li_node"

_OPS = {
    FilterOperator.EQ: "$eq", FilterOperator.NE: "$ne",
    FilterOperator.GT: "$gt", FilterOperator.GTE: "$gte",
    FilterOperator.LT: "$lt", FilterOperator.LTE: "$lte",
    FilterOperator.IN: "$in", FilterOperator.NIN: "$nin",
}


def _filter_value(v: Any) -> Any:
    if isinstance(v, (list, tuple)):
        return [_filter_value(x) for x in v]
    return v if isinstance(v, (str, int, float, bool)) else payload_str(v)


def _filter_to_json(f: Any) -> Dict[str, Any]:
    if isinstance(f, MetadataFilters):
        parts = [_filter_to_json(x) for x in f.filters]
        if f.condition == FilterCondition.NOT:
            inner = parts[0] if len(parts) == 1 else {"$and": parts}
            return {"$not": inner}
        if len(parts) == 1:
            return parts[0]
        return {"$or" if f.condition == FilterCondition.OR else "$and": parts}
    assert isinstance(f, MetadataFilter)
    if f.operator == FilterOperator.IS_EMPTY:
        return {f.key: {"$exists": False}}
    op = _OPS.get(f.operator)
    if op is None:
        raise NotImplementedError(f"TideVec does not support the {f.operator.value!r} filter")
    return {f.key: {op: _filter_value(f.value)}}


def _filters_to_json(filters: Optional[MetadataFilters], node_ids: Optional[List[str]] = None,
                     doc_ids: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    parts = []
    if filters is not None and filters.filters:
        parts.append(_filter_to_json(filters))
    if node_ids:
        parts.append({"_id": {"$in": list(node_ids)}})
    if doc_ids:
        parts.append({"ref_doc_id": {"$in": list(doc_ids)}})
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else {"$and": parts}


class TideVecVectorStore(BasePydanticVectorStore):
    """
    Args:
        client: ``tidevec.TideVec`` or ``tidevec.Local``.
        collection: collection name.
        temporal_blend: per-query blend (None = collection setting).
        superseded_penalty: score penalty for superseded nodes.
        exclude_superseded: drop nodes that a newer version supersedes.
        collection_kwargs: passed to ``create_collection`` on first add.
    """

    stores_text: bool = True
    flat_metadata: bool = False

    collection: str
    temporal_blend: Optional[float] = None
    superseded_penalty: float = 0.0
    exclude_superseded: bool = False
    collection_kwargs: Dict[str, Any] = {}

    _client: Any = PrivateAttr()

    def __init__(self, client: Any, collection: str, *,
                 temporal_blend: Optional[float] = None,
                 superseded_penalty: float = 0.0,
                 exclude_superseded: bool = False,
                 **collection_kwargs: Any) -> None:
        super().__init__(collection=collection, temporal_blend=temporal_blend,
                         superseded_penalty=superseded_penalty,
                         exclude_superseded=exclude_superseded,
                         collection_kwargs=collection_kwargs)
        self._client = client

    @classmethod
    def class_name(cls) -> str:
        return "TideVecVectorStore"

    @property
    def client(self) -> Any:
        return self._client

    def add(self, nodes: Sequence[BaseNode], **add_kwargs: Any) -> List[str]:
        if not nodes:
            return []
        if not self._client.has_collection(self.collection):
            self._client.create_collection(
                self.collection, dim=len(nodes[0].get_embedding()), **self.collection_kwargs)
        vectors = []
        for node in nodes:
            meta = node.metadata or {}
            payload = {k: payload_str(v) for k, v in meta.items()
                       if isinstance(v, (str, int, float, bool))}
            payload["ref_doc_id"] = node.ref_doc_id or ""
            payload[_NODE_KEY] = _json.dumps(
                node_to_metadata_dict(node, remove_text=False, flat_metadata=False),
                default=str)
            vec: Dict[str, Any] = {"id": node.node_id,
                                   "embedding": list(node.get_embedding()),
                                   "payload": payload}
            if isinstance(meta.get("created_at"), (int, float)):
                vec["created_at"] = int(meta["created_at"])
            vectors.append(vec)
        self._client.upsert(self.collection, vectors)
        return [n.node_id for n in nodes]

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        if self._client.has_collection(self.collection):
            self._client.delete(self.collection, filter={"ref_doc_id": ref_doc_id})

    def delete_nodes(self, node_ids: Optional[List[str]] = None,
                     filters: Optional[MetadataFilters] = None, **delete_kwargs: Any) -> None:
        if not self._client.has_collection(self.collection):
            return
        flt = _filters_to_json(filters)
        if node_ids is None and flt is None:
            return
        if node_ids is not None and flt is not None:
            flt = {"$and": [flt, {"_id": {"$in": list(node_ids)}}]}
            node_ids = None
        self._client.delete(self.collection, node_ids, filter=flt)

    def get_nodes(self, node_ids: Optional[List[str]] = None,
                  filters: Optional[MetadataFilters] = None, **kwargs: Any) -> List[BaseNode]:
        """Fetch stored nodes by id and/or filter (no vector search)."""
        if not self._client.has_collection(self.collection):
            return []
        if node_ids is not None and filters is None:
            records = self._client.get_many(self.collection, list(node_ids))
        else:
            records = list(self._client.iterate(
                self.collection, filter=_filters_to_json(filters, node_ids)))
        return [metadata_dict_to_node(_json.loads(r.payload[_NODE_KEY])) for r in records
                if _NODE_KEY in r.payload]

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        if query.query_embedding is None:
            raise ValueError("TideVecVectorStore needs query.query_embedding")
        if not self._client.has_collection(self.collection):
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        k = query.similarity_top_k
        flt = _filters_to_json(query.filters, query.node_ids, query.doc_ids)
        resp = self._client.search(
            self.collection, list(query.query_embedding),
            top_k=k * 3 if self.exclude_superseded else k,
            filter=flt or "",
            temporal_blend=self.temporal_blend,
            superseded_penalty=self.superseded_penalty)

        nodes, sims, ids = [], [], []
        for hit in resp.hits:
            if self.exclude_superseded and hit.superseded_by:
                continue
            node = metadata_dict_to_node(_json.loads(hit.payload[_NODE_KEY]))
            node.metadata["tidevec_superseded_by"] = hit.superseded_by
            node.metadata["tidevec_staleness_warning"] = hit.staleness_warning
            nodes.append(node)
            sims.append(hit.score)
            ids.append(hit.id)
            if len(nodes) == k:
                break
        return VectorStoreQueryResult(nodes=nodes, similarities=sims, ids=ids)
