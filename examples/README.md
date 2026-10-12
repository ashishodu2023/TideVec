# Examples

All scripts run offline with a hashing embedder unless noted, so you can try them without API keys.

```bash
pip install "tidevec[embedded,langchain,llamaindex,mcp]"
```

| Example | Shows |
|---------|-------|
| [`01_quickstart_embedded.py`](01_quickstart_embedded.py) | Embedded mode, temporal scoring vs. pure similarity, automatic supersession |
| [`02_langchain_rag.py`](02_langchain_rag.py) | LangChain vector store / retriever that drops superseded policy versions (`--openai` for real embeddings) |
| [`03_llamaindex.py`](03_llamaindex.py) | LlamaIndex `VectorStoreIndex` on TideVec, versioned inserts, `delete_ref_doc` |
| [`04_agent_memory/`](04_agent_memory/) | `AgentMemory` and the `tidevec-mcp` server for Claude Desktop / Cursor |
| [`05_drift_migration.py`](05_drift_migration.py) | DriftBridge: re-embed a live collection with a new model while it keeps serving queries; writes made mid-migration are preserved (needs the server binary) |
| [`notebooks/tidevec_quickstart.ipynb`](notebooks/tidevec_quickstart.ipynb) | Colab notebook with a real sentence-transformers model |

To run against a server instead of embedded mode, replace `Local(path)` with `TideVec("localhost:6399")`;
the two clients have the same methods.
