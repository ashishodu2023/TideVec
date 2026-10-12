# Haystack, DSPy, LangGraph, CrewAI, Vercel AI SDK, n8n

Each adapter drops document versions that a newer version superseded. The Python adapters live in
`tidevec.integrations` (install the matching extra) and work against a server (`TideVec("host:6399")`)
or embedded mode (`Local(path)`); CI tests them in both modes (`sdk/python/tests/test_ecosystem.py`).
The TypeScript packages talk to a server and are tested against one (`sdk/typescript/packages/*/test`).

| Framework | Install | Import | Example |
|---|---|---|---|
| Haystack 2.x | `pip install "tidevec[haystack]"` | `from tidevec.integrations.haystack import TideVecDocumentStore, TideVecEmbeddingRetriever` | [`06_haystack_pipeline.py`](https://github.com/ashishodu2023/TideVec/blob/main/examples/06_haystack_pipeline.py) |
| LangGraph | `pip install "tidevec[langgraph]"` | `from tidevec.integrations.langgraph import TideVecStore, memory_tools` | [`07_langgraph_memory.py`](https://github.com/ashishodu2023/TideVec/blob/main/examples/07_langgraph_memory.py) |
| CrewAI ≥ 1.10 | `pip install "tidevec[crewai]"` | `from tidevec.integrations.crewai import TideVecStorage` | [`08_crewai_memory.py`](https://github.com/ashishodu2023/TideVec/blob/main/examples/08_crewai_memory.py) |
| DSPy | `pip install "tidevec[dspy]"` | `from tidevec.integrations.dspy import TideVecRetriever` | [`09_dspy_rag.py`](https://github.com/ashishodu2023/TideVec/blob/main/examples/09_dspy_rag.py) |
| Vercel AI SDK 5-7 | `@tidevec/ai-sdk` (npm, [not published yet](../sdks/typescript.md)) | `import { tidevecMemoryTools } from "@tidevec/ai-sdk"` | [`10_typescript/ai_sdk_agent.mjs`](https://github.com/ashishodu2023/TideVec/tree/main/examples/10_typescript) |
| n8n | `n8n-nodes-tidevec` community node (not published yet) | Settings → Community Nodes | [package README](https://github.com/ashishodu2023/TideVec/tree/main/sdk/typescript/packages/n8n-nodes-tidevec) |

## Haystack

`TideVecDocumentStore` implements Haystack's `DocumentStore` protocol and `TideVecEmbeddingRetriever`
is a pipeline component:

```python
from haystack import Document
from tidevec.integrations.haystack import TideVecDocumentStore, TideVecEmbeddingRetriever

store = TideVecDocumentStore(url="localhost:6399", collection="kb", embedding_dim=384,
                             supersede_match_field="doc_id")
store.write_documents([Document(content="Refunds take 3 days", embedding=emb,
                                meta={"doc_id": "refunds"})])
retriever = TideVecEmbeddingRetriever(document_store=store, top_k=5)
retriever.run(query_embedding=query_emb)["documents"]       # current versions only
```

Filters use Haystack's syntax and Haystack's own matcher, so semantics equal the in-memory store; the
parts TideVec can index (equality, `in`, numeric ranges on top-level meta keys) are pushed down.
`blob` and `sparse_embedding` are not stored.

## LangGraph

`TideVecStore` is a `langgraph.store.base.BaseStore` (namespaced `put`/`get`/`search`/`delete`, value
filters, semantic search, TTL), usable as `create_react_agent(model, tools, store=store)`.
`memory_tools(AgentMemory(...))` gives an agent `remember` / `recall` / `memory_history` tools where a
fact remembered under an existing key supersedes the old one.

## CrewAI

`TideVecStorage` implements CrewAI's unified-memory `StorageBackend`: pass it as
`Memory(storage=storage)` or call `storage.install_as_default()`. Scopes and categories are pushed down
as filters.

## DSPy

`TideVecRetriever(client, collection, embedder=..., k=5)` is a `dspy.Retrieve` module returning a
`Prediction` with `passages`, `ids`, `scores` and `metadata`; `retrieve.as_rm()` supports the legacy
`dspy.configure(rm=...)` pattern.

## Vercel AI SDK and n8n

`@tidevec/ai-sdk` provides memory tools (`remember` / `recall` / `history` / `forget`) with the same
storage layout as the Python `AgentMemory` and `tidevec-mcp`, a knowledge-search tool and RAG
middleware. `n8n-nodes-tidevec` is an early n8n node for upsert and search that can be attached to an
AI Agent node as a tool. See the [TypeScript SDK](../sdks/typescript.md) page.
