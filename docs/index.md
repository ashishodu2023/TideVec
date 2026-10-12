# TideVec

**A vector database for RAG and AI agents over knowledge that changes.**

When a document is revised, most vector search keeps both versions and ranks them by similarity alone,
so an assistant can confidently quote the refund policy that changed three weeks ago. TideVec handles
this in the database:

1. **Recency is part of the score inside the HNSW search**, with a per-collection half-life and a
   per-query blend. See [Recency and half-life](concepts/recency.md).
2. **Supersession:** on upsert, TideVec links a new version to the older versions it replaces (same
   document key, or similarity above a threshold). Results carry `superseded_by`, and superseded
   versions can be demoted or dropped. See [Supersession](concepts/supersession.md).
3. **Typed edges** (`UPDATES`, `CONTRADICTS`, `CAUSES`, ...) are stored next to the vectors, so a search
   can flag contradicted results or expand to related facts. See
   [Edges and contradictions](concepts/edges.md).

The engine is C++20. You can run it **in-process from Python** (`pip install "tidevec[embedded]"`) or
as a **REST server** (`docker run averm004/tidevec`), with clients for Python, Go, Java and C++.

## Where to start

| If you want to... | Read |
|---|---|
| Try it in two minutes | [Quickstart](quickstart.md) |
| Use it from LangChain or LlamaIndex | [LangChain](integrations/langchain.md), [LlamaIndex](integrations/llamaindex.md) |
| Give an agent memory that handles updated facts | [Agent memory and MCP](integrations/mcp.md) |
| Call the HTTP API directly | [REST API reference](api/rest.md) |
| Run it in production | [Deployment](operations/deployment.md), [Durability](operations/durability.md) |
| See the evidence | [Benchmarks](benchmarks.md) |

## What the evidence says

On 270 real Wikipedia revisions of 55 articles ([benchmark](benchmarks.md)):

- Plain similarity returned an outdated version as the top hit for 55-85% of queries.
- Recency weighting fixed the top hit, but TideVec's in-search blend ranked **exactly the same as a
  rerank** with the same formula. Its advantage there is operational (one stage, works with filters),
  not accuracy.
- **Supersession is what removed stale versions from the top 5** (0.80-0.91 outdated versions per
  query with recency alone in the mid-2025 scenario, 0 with supersession).

## Status

TideVec is young. Established engines (Qdrant, Milvus, Weaviate, Pinecone) are far ahead on scale,
operations and ecosystem. Today TideVec runs as a single server or embedded library; a replicated
[cluster mode](operations/cluster.md) is experimental, and the GPU/TPU paths are not validated on
hardware. The [roadmap](https://github.com/ashishodu2023/TideVec/blob/main/ROADMAP.md) lists what is
planned and in what order.
