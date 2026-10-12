# Examples

All scripts run offline with a hashing embedder unless noted, so you can try them without API keys.

```bash
pip install "tidevec[embedded,langchain,llamaindex,haystack,langgraph,crewai,dspy,mcp]"   # or only the ones you need
```

| Example | Shows |
|---------|-------|
| [`01_quickstart_embedded.py`](01_quickstart_embedded.py) | Embedded mode, temporal scoring vs. pure similarity, automatic supersession |
| [`02_langchain_rag.py`](02_langchain_rag.py) | LangChain vector store / retriever that drops superseded policy versions (`--openai` for real embeddings) |
| [`03_llamaindex.py`](03_llamaindex.py) | LlamaIndex `VectorStoreIndex` on TideVec, versioned inserts, `delete_ref_doc` |
| [`04_agent_memory/`](04_agent_memory/) | `AgentMemory` and the `tidevec-mcp` server for Claude Desktop / Cursor |
| [`05_drift_migration.py`](05_drift_migration.py) | DriftBridge: re-embed a live collection with a new model while it keeps serving queries; writes made mid-migration are preserved (needs the server binary) |
| [`06_haystack_pipeline.py`](06_haystack_pipeline.py) | Haystack pipeline with `TideVecDocumentStore` + `TideVecEmbeddingRetriever`; superseded document versions are dropped, Haystack filter syntax |
| [`07_langgraph_memory.py`](07_langgraph_memory.py) | LangGraph `TideVecStore` (namespaced items, semantic search, filters) and `remember` / `recall` / `memory_history` agent tools |
| [`08_crewai_memory.py`](08_crewai_memory.py) | CrewAI `Memory(storage=TideVecStorage(...))`; shallow recall without an LLM |
| [`09_dspy_rag.py`](09_dspy_rag.py) | DSPy RAG module with `TideVecRetriever` (retrieval runs offline; `--lm openai/gpt-4o-mini` for answers) |
| [`10_typescript/`](10_typescript/) | TypeScript client quickstart and a Vercel AI SDK agent with `@tidevec/ai-sdk` memory tools (needs a server; the agent needs `OPENAI_API_KEY`) |
| [`mcp/`](mcp/) | `tidevec-mcp` setup for Claude Desktop, Claude Code, Cursor, Windsurf, Zed, VS Code and remote HTTP clients |
| [`notebooks/tidevec_quickstart.ipynb`](notebooks/tidevec_quickstart.ipynb) | Colab notebook with a real sentence-transformers model |

To run against a server instead of embedded mode, replace `Local(path)` with `TideVec("localhost:6399")`;
the two clients have the same methods. Examples 06-09 take `--url localhost:6399` for that.
