# Examples

The scripts in [`examples/`](https://github.com/ashishodu2023/TideVec/tree/main/examples) run offline
with a hashing embedder unless noted, so they need no API keys.

```bash
pip install "tidevec[embedded,langchain,llamaindex,mcp]"
```

| Example | Shows |
|---|---|
| [`01_quickstart_embedded.py`](https://github.com/ashishodu2023/TideVec/blob/main/examples/01_quickstart_embedded.py) | Embedded mode; recency blend versus pure similarity; automatic supersession |
| [`02_langchain_rag.py`](https://github.com/ashishodu2023/TideVec/blob/main/examples/02_langchain_rag.py) | LangChain vector store and retriever that drops superseded policy versions (`--openai` for real embeddings) |
| [`03_llamaindex.py`](https://github.com/ashishodu2023/TideVec/blob/main/examples/03_llamaindex.py) | LlamaIndex `VectorStoreIndex` on TideVec, versioned inserts, `delete_ref_doc` |
| [`04_agent_memory/`](https://github.com/ashishodu2023/TideVec/tree/main/examples/04_agent_memory) | `AgentMemory` and the `tidevec-mcp` server for Claude Desktop and Cursor |
| [`05_drift_migration.py`](https://github.com/ashishodu2023/TideVec/blob/main/examples/05_drift_migration.py) | DriftBridge: re-embed a live collection with a new model while it keeps serving queries (needs the server binary) |
| [`notebooks/tidevec_quickstart.ipynb`](https://github.com/ashishodu2023/TideVec/blob/main/examples/notebooks/tidevec_quickstart.ipynb) | Colab notebook with a real sentence-transformers model |

To run an example against a server instead of embedded mode, replace `Local(path)` with
`TideVec("localhost:6399")`; the two clients have the same methods.
