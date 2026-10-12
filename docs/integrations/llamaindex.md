# LlamaIndex

```bash
pip install "tidevec[llamaindex,embedded]"
```

```python
from llama_index.core import StorageContext, VectorStoreIndex
from tidevec import Local
from tidevec.integrations.llamaindex import TideVecVectorStore

vector_store = TideVecVectorStore(
    Local("./data"), "kb",
    supersession=True, supersede_match_field="ref_doc_id",
    exclude_superseded=True,
)
index = VectorStoreIndex.from_documents(
    docs, storage_context=StorageContext.from_defaults(vector_store=vector_store))
print(index.as_query_engine().query("What is the current refund window?"))
```

## Constructor

`TideVecVectorStore(client, collection, *, temporal_blend=None, superseded_penalty=0.0,
exclude_superseded=False, **collection_kwargs)`

- `client` is `TideVec` (server) or `Local` (embedded).
- The collection is created on the first `add`, with the dimension of the first node's embedding;
  `collection_kwargs` go to `create_collection`.
- Each node's `ref_doc_id` is stored in the payload, so `supersede_match_field="ref_doc_id"` links
  re-ingested versions of the same source document, and `delete(ref_doc_id)` removes all its nodes.
- A numeric `metadata["created_at"]` (ms since epoch) is used as the node's timestamp.

## Queries

- Node scores are TideVec's blended similarity + recency scores.
- Returned nodes gain `metadata["tidevec_superseded_by"]` and `metadata["tidevec_staleness_warning"]`.
- `exclude_superseded=True` fetches `3k` candidates and drops superseded nodes.
- Filters: `MetadataFilters` with a **single** `==` filter. Other operators, several filters, and
  `node_ids`/`doc_ids` restrictions raise `NotImplementedError`.

## Example

[`examples/03_llamaindex.py`](https://github.com/ashishodu2023/TideVec/blob/main/examples/03_llamaindex.py)
runs offline with a hashing embedder; swap in any LlamaIndex embedding model.
