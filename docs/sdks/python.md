# Python SDK

```bash
pip install tidevec                 # REST client only (no dependencies beyond the standard library)
pip install "tidevec[embedded]"     # + in-process engine (tidevec-embedded wheel)
pip install "tidevec[langchain]"    # + LangChain integration
pip install "tidevec[llamaindex]"   # + LlamaIndex integration
pip install "tidevec[mcp]"          # + tidevec-mcp server
```

Python 3.9-3.13. Two entry points with the same methods:

```python
from tidevec import TideVec, Local

db = TideVec("localhost:6399", api_key="...")   # server; api_key defaults to $TIDEVEC_API_KEY
db = Local("./tidevec_data")                    # embedded, no server
```

`TideVec(host, api_key="", timeout=30.0, tls=False, max_retries=3)` retries `429` responses with
backoff and, in [cluster mode](../operations/cluster.md), follows `421` redirects to the leader.
`AsyncTideVec` wraps the same API for asyncio.

## Collections

```python
from tidevec import HalfLife

db.create_collection(
    "kb", dim=384,
    metric="cosine",                     # "cosine" | "l2" | "dot"
    half_life_ms=HalfLife.ONE_MONTH,     # recency half-life
    temporal_blend=0.3,                  # weight of recency in the score
    staleness_threshold=0.2,             # age warning below this recency weight
    supersession=True, supersede_threshold=0.92, supersede_match_field="doc_id",
)
db.list_collections()     # [CollectionInfo, ...]
db.get_collection("kb")   # raises CollectionNotFoundError if missing
db.has_collection("kb")
db.count("kb")
db.set_temporal("kb", half_life_ms=HalfLife.ONE_WEEK, temporal_blend=0.2)
db.drop_collection("kb")  # True if it existed
```

`create_collection` also accepts `index_type`, `n_shards` (default 4), `n_replicas`, `write_quorum`
and agent-memory settings (`agent_id`, `memory_ttl_seconds`, `dedup_threshold`, `agent_enabled`).

## Writing

```python
db.upsert("kb", [{
    "id": "refunds-2026",
    "embedding": vec,                        # list or numpy array; length must equal dim
    "payload": {"doc_id": "refunds", "text": "..."},   # string values
    "created_at": 1767225600000,             # optional, ms since epoch; default now
    "ttl_seconds": 86400,                    # optional
    "edges": [{"target_id": "refunds-2025", "type": "UPDATES", "weight": 1.0}],  # optional
}])
db.delete("kb", ids=["a", "b"])
db.delete("kb", filter="source=crm")         # returns the number removed
db.add_edges("kb", [{"src": "a", "tgt": "b", "type": "CONTRADICTS", "weight": 1.0}])
```

A vector whose length differs from the collection's `dim` raises `TideVecError` (embedded) or
`APIError` with status 400 (server). Payload values are stored as strings; `tidevec.payload_str`
converts other types the same way the integrations do.

## Searching

```python
resp = db.search(
    "kb", query_vec, top_k=10,
    temporal_blend=None,              # None = collection setting; 0.0 = pure similarity
    filter="lang=en",                 # or a filter dict, see Concepts > Retrieval, filters
    superseded_penalty=1.0,           # push superseded versions to the bottom
    mode="vector_only",               # or "contradiction_check", "causal_expand"
    ef_search=128, include_trace=False, include_staleness_warnings=True,
)
for hit in resp:                      # SearchResponse is iterable; resp.trace has the trace
    hit.id, hit.score, hit.vector_score, hit.temporal_score, hit.payload, hit.created_at
    hit.superseded_by, hit.is_superseded, hit.contradicted_by, hit.causal_neighbors
    hit.staleness_warning, hit.staleness_reason
db.search_one("kb", query_vec)        # top hit or None
db.batch_search("kb", [q1, q2])       # sequential searches
```

`search` also takes `filter` as a dict (`{"price": {"$lt": 20}}`), `as_of=` (ms, ISO-8601 string
or `datetime`) and `text=` for BM25 keyword or hybrid search (`hybrid="rrf"|"weighted"`, `alpha`).

## Retrieval, aliases and compaction

`get`, `get_many`, `scroll`, `iterate`, `list_ids`, `count(filter=, as_of=)`, `delete(ids, filter=)`,
`set_alias` / `delete_alias` / `list_aliases`, `compact` / `compaction_status`. Same methods on
`TideVec` and `Local`; see [Retrieval, filters, as-of and hybrid](../concepts/filters.md).

## Server-only methods

`health()`, `info()`, `metrics()`, `cluster_status()`, DriftBridge (`start_drift`, `drift_status`,
`abort_drift`), backups (`trigger_backup`, `list_backups`, `list_backup_manifests`,
`restore_backup`). `Local` has `health()`, `info()` and `close()`.

## Errors

| Exception | When |
|---|---|
| `CollectionNotFoundError` | `404`, unknown collection |
| `UnauthorizedError` | `401`, missing or wrong API key |
| `ForbiddenError` | `403`, tenant, quota or URL not allowed |
| `RateLimitError` | `429` after `max_retries` |
| `NotLeaderError` | `421` with no reachable leader (cluster mode); `.leader` has the hint |
| `APIError` | any other non-2xx; `.status` has the code |
| `ConnectionError_` | server unreachable |

All derive from `TideVecError`.

## Embedders

`tidevec.embeddings` has small helpers used by the integrations and the MCP server:

- `HashingEmbedder(dim)`: deterministic, dependency-free, for tests and demos only (no semantics).
- `OpenAIEmbedder(...)`: any OpenAI-compatible embeddings endpoint.
- `SentenceTransformerEmbedder(...)`: local models via `sentence-transformers`.
- `default_embedder()`: picks one based on what is installed and configured.

Each has `dim`, `embed_query(text)` and `embed_documents(texts)`.
