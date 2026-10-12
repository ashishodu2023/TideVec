# LangChain

```bash
pip install "tidevec[langchain,embedded]"
```

`TideVecVectorStore` is a LangChain `VectorStore` that works with the REST client (`TideVec`) or
embedded mode (`Local`).

```python
from tidevec import Local
from tidevec.integrations.langchain import TideVecVectorStore

store = TideVecVectorStore(
    Local("./data"), "kb", embeddings,            # any LangChain Embeddings
    supersession=True, supersede_match_field="doc_id", supersede_threshold=0.5,
)
store.add_texts(["Refunds take 14 days"], metadatas=[{"doc_id": "refunds"}],
                created_at=1735689600000)        # optional backdating (ms since epoch)
store.add_texts(["Refunds now take 3 days"], metadatas=[{"doc_id": "refunds"}])

docs = store.similarity_search("how long do refunds take?", k=3, exclude_superseded=True)
retriever = store.as_retriever(search_kwargs={"k": 4, "exclude_superseded": True})
```

## Constructor

`TideVecVectorStore(client, collection, embedding, *, text_key="text", temporal_blend=None,
superseded_penalty=0.0, **collection_kwargs)`

- The collection is created on first use if it does not exist; `collection_kwargs` go to
  `create_collection` (`half_life_ms`, `temporal_blend`, `supersession`, `supersede_match_field`, ...).
  The dimension is taken from the embedding model.
- `temporal_blend` and `superseded_penalty` are per-query defaults; both can be overridden as keyword
  arguments to any search call. `None` uses the collection's blend.
- `TideVecVectorStore.from_texts(texts, embedding, client=None, collection="langchain",
  path="./tidevec_data")` uses embedded mode when no client is given.

## Metadata and filters

Metadata round-trips exactly (stored as JSON in the payload). Top-level scalar values are also stored
as flat payload fields so they can be filtered on: `filter={"doc_id": "refunds"}` or
`filter="doc_id=refunds"`. Only **one** equality condition is supported; more raise `ValueError`.
A numeric `metadata["created_at"]` is used as the document's timestamp.

Each returned `Document` has `metadata["tidevec"]` with `score`, `vector_score`, `temporal_score`,
`created_at`, `staleness_warning`, `superseded_by` and `contradicted_by`, so a chain can tell the
model that a source is outdated instead of hiding it.

`exclude_superseded=True` fetches `3k` candidates and drops superseded ones before returning `k`.
Scores are blended similarity + recency in \([0, 1]\) and are returned as-is as relevance scores.

## Example

[`examples/02_langchain_rag.py`](https://github.com/ashishodu2023/TideVec/blob/main/examples/02_langchain_rag.py)
builds a small knowledge base in which two policies were revised and compares the context a RAG chain
would receive from a plain similarity retriever with TideVec's. It runs offline with hashing
embeddings, or with `--openai`.

!!! note "More integrations in progress"
    Haystack, DSPy and LangGraph adapters are being added under `tidevec.integrations` on `main`.
    They are not part of this release and are not documented yet.
    <!-- TODO(docs): add pages when the Phase 4 integrations ship. -->
