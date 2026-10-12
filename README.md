<div align="center">

# TideVec

**Your RAG keeps quoting last month's policy. TideVec fixes that.**

*A vector database for AI agents and RAG over knowledge that changes:
time-aware ranking, automatic detection of superseded versions, and typed
links between facts, in a C++20 engine you can embed in Python or run as a server.*

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![CI](https://github.com/ashishodu2023/TideVec/actions/workflows/ci.yml/badge.svg)](https://github.com/ashishodu2023/TideVec/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/tidevec)](https://pypi.org/project/tidevec/)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-orange)](https://en.cppreference.com/w/cpp/20)

[**tidevec.com**](https://tidevec.com) · [**Docs**](docs/) · [**Examples**](examples/) · [**Benchmarks**](benchmarks/temporal_real_data/) · [**Changelog**](CHANGELOG.md) · [**Roadmap**](ROADMAP.md)

</div>

---

## The problem

Most vector search ranks purely on semantic similarity. When a document is revised, the old version
usually stays in the index, and because it is often longer or more "definitional" it can score
*higher* than the current one. Your assistant confidently cites a refund policy that changed three
weeks ago.

The usual workarounds each have a failure mode:

- **Hard timestamp filters** (`age <= N days`) silently drop authoritative documents that simply
  haven't been rewritten recently.
- **Recency reranking** (decay formulas in Qdrant's Query API, Milvus decay rankers) reorders the
  candidates similarity search already returned. It fixes the top hit, but older versions of the same
  document still fill the rest of the context window.
- **Manual deletes / TTLs** require someone to know which documents were superseded.

TideVec handles this in the database:

1. **Recency is part of the score inside the HNSW search**, with a per-collection half-life and a
   per-query blend.
2. **Supersession:** on upsert, TideVec links a new version to the older versions it replaces (same
   document key, or similarity above a threshold). Results carry `superseded_by`, and superseded
   versions can be demoted or dropped, no time window to tune.
3. **Typed edges** (`UPDATES`, `CONTRADICTS`, `CAUSES`, ...) are stored with the vectors, so a search
   can flag contradicted results or expand to related facts.

---

## Quick start

### Embedded (no server)

```bash
pip install "tidevec[embedded]"      # Linux x86_64/arm64, macOS arm64; Python 3.9-3.13
```

```python
from tidevec import Local, HalfLife

db = Local("./tidevec_data")
db.create_collection("policies", dim=384, half_life_ms=HalfLife.ONE_MONTH, temporal_blend=0.3,
                     # link a new upsert to older vectors with the same doc_id that it resembles
                     supersession=True, supersede_match_field="doc_id", supersede_threshold=0.5)

db.upsert("policies", [
    {"id": "refunds-2025", "embedding": old_vec, "payload": {"doc_id": "refunds"}},
    {"id": "refunds-2026", "embedding": new_vec, "payload": {"doc_id": "refunds"}},
])

for hit in db.search("policies", query_vec, top_k=5, superseded_penalty=1.0):
    print(hit.id, round(hit.score, 3), "superseded by" if hit.superseded_by else "", hit.superseded_by)
```

`Local` and the server client `TideVec("host:6399")` have the same methods.
Runnable version: [`examples/01_quickstart_embedded.py`](examples/01_quickstart_embedded.py).

### Server

```bash
export TIDEVEC_API_KEY="your-secret-key"
docker run -p 6399:6399 -e TIDEVEC_API_KEY \
  -v tidevec_data:/data -v tidevec_backups:/backups averm004/tidevec:latest
```

```python
from tidevec import TideVec
db = TideVec("localhost:6399")          # reads TIDEVEC_API_KEY from the environment
```

`latest` is the newest release (`0.3.0`, `0.3` and `X.Y.Z-cpu` tags are also published, for amd64 and
arm64); `main` tracks the main branch. The amd64 image needs an x86-64-v3 CPU (AVX2 + FMA).

---

## Integrations

Every integration drops document versions that a newer version superseded. The Python ones live in
[`sdk/python/tidevec/integrations/`](sdk/python/tidevec/integrations/) and work with both the server
client and embedded mode; install the matching extra (`pip install "tidevec[haystack]"`, ...).

| Framework | Use | Docs / example |
|---|---|---|
| LangChain | `from tidevec.integrations.langchain import TideVecVectorStore` | [docs](docs/integrations/langchain.md) · [`02_langchain_rag.py`](examples/02_langchain_rag.py) |
| LlamaIndex | `from tidevec.integrations.llamaindex import TideVecVectorStore` | [docs](docs/integrations/llamaindex.md) · [`03_llamaindex.py`](examples/03_llamaindex.py) |
| Haystack | `from tidevec.integrations.haystack import TideVecDocumentStore, TideVecEmbeddingRetriever` | [docs](docs/integrations/frameworks.md#haystack) · [`06_haystack_pipeline.py`](examples/06_haystack_pipeline.py) |
| LangGraph | `from tidevec.integrations.langgraph import TideVecStore, memory_tools` | [docs](docs/integrations/frameworks.md#langgraph) · [`07_langgraph_memory.py`](examples/07_langgraph_memory.py) |
| CrewAI | `from tidevec.integrations.crewai import TideVecStorage` | [docs](docs/integrations/frameworks.md#crewai) · [`08_crewai_memory.py`](examples/08_crewai_memory.py) |
| DSPy | `from tidevec.integrations.dspy import TideVecRetriever` | [docs](docs/integrations/frameworks.md#dspy) · [`09_dspy_rag.py`](examples/09_dspy_rag.py) |
| Vercel AI SDK (TypeScript) | `@tidevec/ai-sdk`: memory tools, search tool, RAG middleware | [docs](docs/integrations/frameworks.md#vercel-ai-sdk-and-n8n) · [`10_typescript/`](examples/10_typescript/) |
| n8n (TypeScript) | `n8n-nodes-tidevec` community node | [package](sdk/typescript/packages/n8n-nodes-tidevec/) |
| MCP server | `uvx tidevec-mcp`: `remember` / `recall` / `history` / `forget` for Claude Desktop, Cursor and other MCP clients | [docs](docs/integrations/mcp.md) · [`examples/mcp/`](examples/mcp/) · [`04_agent_memory/`](examples/04_agent_memory/) |

Agent memory without a framework: `from tidevec.memory import AgentMemory` (a new fact under an existing
key supersedes the old one). The TypeScript packages are built and tested in CI but not on npm yet.

---

## Benchmark on real revision history

[`benchmarks/temporal_real_data/`](benchmarks/temporal_real_data/): the lead sections of 55 Wikipedia
articles at five points in time (270 documents, 2-5 versions per article), embedded with
`bge-small-en-v1.5`. The question for each article is whether search returns the **current** version.

| Strategy | Current@1 (today) | Current@1 (as of mid-2025) | Stale versions in top-5 (mid-2025) |
|---|---:|---:|---:|
| Cosine only | 18.2% | 27.3% | 2.89 |
| Hard filter, 1 year | 100% | **0%** | 0.00 |
| Cosine top-50, then recency rerank | 100% | 100% | 0.91 |
| TideVec recency blend | 100% | 100% | 0.91 |
| TideVec + supersession | 100% | 100% | **0.00** |

What this shows, and what it does not:

- Plain similarity returned an outdated version for 55–85% of queries across query sets.
- **TideVec's in-search blend produced the same rankings as a rerank with the same formula.** On this
  data its advantage is operational (one stage, no candidate-pool size, works with filters), not
  accuracy.
- A 1-year hard filter was perfect while documents were fresh and returned nothing relevant once the
  current versions were 1.5 years old.
- **Supersession is what removes stale context:** recency alone left ~0.9 outdated versions in each
  top-5; supersession removed all of them. Without a document key, a similarity threshold of 0.95
  (`vector_score`, raw cosine 0.90) linked versions with 100% precision and flagged 95.8% of stale
  documents.

Caveats: 55 articles, one small embedding model, templated queries. Full tables and the scripts are
in the benchmark folder. The older [`temporal_vs_qdrant`](benchmarks/temporal_vs_qdrant/) benchmark
uses hand-set similarities to illustrate one failure mode; use the real-data one for claims.

**More benchmarks** (summaries in [`docs/benchmarks.md`](docs/benchmarks.md)):

- [`temporal_v2`](benchmarks/temporal_v2/): Wikipedia policy pages and PyPI READMEs, three embedding
  models, natural-language queries, reimplemented Qdrant/Milvus decay formulas as baselines.
  Supersession by document key had the highest Current@1 everywhere (82–91% pooled vs. 70–83% for
  the best decay); a recency blend alone was fragile.
- [`agent_memory`](benchmarks/agent_memory/): LongMemEval knowledge-update, 72 questions. `AgentMemory`
  returned an outdated fact as the top hit for 4.2% of questions vs. 27.8% for a plain vector store.
- [`ann`](benchmarks/ann/): SIFT-1M and GloVe-100 against hnswlib and FAISS HNSW. **TideVec is
  slower**: 3.5–6× lower QPS at matched recall and builds take longer.
- [`gpu`](benchmarks/gpu/README.md): the CUDA path fails recall on real hardware and is unsupported.

---

## Features

### Time-aware ranking (TVIndex)

```
final_score = (1 − β) · similarity + β · 2^(−age / half_life)
```

`similarity` is the metric's score in [0, 1] (for cosine, `(1 + cos) / 2`). `β` is the collection's
`temporal_blend`, overridable per query (`temporal_blend=0.0` gives pure similarity). The blended score
is used inside the HNSW beam search, not only on the final candidates. Results also carry
`vector_score`, `temporal_score` and a `staleness_warning` when the recency weight drops below the
collection's `staleness_threshold`. Half-life presets: `HalfLife.ONE_HOUR` ... `ONE_YEAR`.

### Supersession

```python
db.create_collection("kb", dim=768, supersession=True,
                     supersede_threshold=0.92,          # vector_score needed to link two versions
                     supersede_match_field="doc_id")    # optional: only link equal doc_ids
```

On each upsert TideVec looks for older vectors the new one replaces and adds an `UPDATES` edge (mirrored
as `superseded_by` on the old one, across shards). Search results expose `hit.superseded_by`;
`superseded_penalty=1.0` pushes superseded versions to the bottom. Linking happens at write time, so a
query never has to guess which version is current.

### Typed edges and contradictions

`CAUSES · CONTRADICTS · UPDATES · ENTITY_OF · RELATED_TO · SUPPORTS`, stored per shard next to the
vectors. `mode="contradiction_check"` fills `hit.contradicted_by` from `CONTRADICTS` edges (stored
symmetrically); `mode="causal_expand"` adds neighbours up to `causal_hops` away.
TideVec does not judge contradictions itself: `tidevec.contradictions.detect_contradictions` asks an
LLM judge (OpenAI-compatible, or your own function) whether a new document contradicts or supports its
nearest neighbours and records the answer as a `CONTRADICTS` or `SUPPORTS` edge.

### Filters, retrieval, as-of and hybrid search

In every SDK (Python server and embedded, Go, Java, TypeScript); details in
[`docs/concepts/filters.md`](docs/concepts/filters.md).

```python
# Rich filters: key=value, or $eq/$ne/$in/$nin/$gt/$gte/$lt/$lte/$exists, $and/$or/$not, ISO dates
db.search("kb", q, filter={"lang": {"$in": ["en", "de"]}, "published": {"$gte": "2025-01-01"}})
db.delete("kb", filter={"source": "crm"})               # delete by filter (or ids=[...])

# Read back without a query vector
db.get("kb", "refunds-2026"); db.get_many("kb", ids)
page = db.scroll("kb", limit=100, filter="lang=en")     # cursor paging; also iterate(), list_ids()
db.count("kb", filter={"lang": "en"})

# Point in time: what was current then (visibility, recency and supersession as of that moment)
db.search("kb", q, as_of="2025-07-01T00:00:00Z")

# BM25 keyword + vector hybrid (collection created with text_field="text")
db.search("kb", q, text="TV-4711 shutdown", hybrid="rrf")   # or "weighted", alpha=...

# Aliases: point readers at "kb_live", then cut over atomically to a rebuilt collection
db.set_alias("kb_live", "kb_v2")
```

- **Payload indexes** (inverted for keywords, sorted for numbers and dates, always on for
  `created_at`) let the planner run very selective filters as an exact search over the matching ids
  instead of a graph walk that finds too few results; filtered-out nodes still route graph searches.
- **Compaction** rebuilds the indexes in the background once deleted and replaced vectors reach
  `compaction_ratio` × live vectors (default 1.0; `db.compact("kb")` forces it); reads and writes continue.

### DriftBridge: model migration without downtime

When you change embedding models, `start_drift` re-embeds every stored vector in the background by
calling your endpoint (`POST {"id", "payload"}` → `{"embedding": [...]}`) into a shadow index, while the
live index keeps serving queries. Writes during the migration are mirrored to the shadow, and the swap
skips vectors deleted or rewritten in the meantime.

Limits: the new model must keep the collection's dimension; the swap applies vectors one by one, so
queries during the swap (about a second per few hundred vectors) see a mix of old and new embeddings;
standalone servers only. Re-embed URLs must be allowlisted (`--reembed-allow host`); loopback and
private addresses additionally need `--reembed-allow-private`, and cloud metadata endpoints are always
blocked. Walkthrough: [`examples/05_drift_migration.py`](examples/05_drift_migration.py).

### Retrieval trace

`include_trace=True` returns a per-query trace (strategy, latency, staleness warnings, contradiction
alerts) that can be exported over OTLP with `--otel`.

---

## Index and search internals

**TVIndex** (`include/tidevec/index/tv_index.hpp`) is an HNSW graph (Malkov & Yashunin) with:

- the blended temporal score in the layer-0 beam search;
- the HNSW neighbour-selection heuristic, so outlying vectors keep incoming links;
- updates that move an embedding tombstone the old node and insert a new one (metadata-only updates
  stay in place); tombstones keep routing searches and are compacted away by a rebuild once they
  reach the number of live vectors;
- filters, validity windows and hard staleness exclusion applied during traversal;
- optional 8-bit product quantization (`use_pq_compression`) with ADC scoring and exact rescoring
  through a `VectorStore` callback (32× smaller vector storage in the unit test: 500 × 96-dim).

| Parameter | Default | |
|---|---|---|
| `M` | 16 | edges per node per layer |
| `ef_construction` | 200 | build beam width |
| `ef_search` | 128 | query beam width (per query) |
| `temporal_blend` | 0.3 | weight on recency |
| `half_life_ms` | 30 days | decay rate |

Other paths: **FlatIndex** (exact, AVX2 dot/L2 kernels), a **CPU IVF** engine, and accelerator
back-ends selected by `--device`:

- **GPU (unsupported):** an independent CUDA implementation of the search design from *CAGRA*
  (Ootomo et al., ICDE 2024), built with `-DTIDEVEC_CUDA=ON`. It is not NVIDIA cuVS. Benchmarked on
  real hardware it does not work: the graph engine's recall@10 is 0.003 and the brute-force engine is
  not exact ([`benchmarks/gpu`](benchmarks/gpu/README.md)). The CPU fallback is what the tests cover.
- **TPU:** exact batched matmul via XLA when built against `libxla_client`; CI builds the stub only.

Metrics: cosine (default), L2, dot product.

---

## Durability

A standalone server writes every change to a per-shard **write-ahead log** (CRC-checked, replayed on
start) and, with `--segment-store`, to disk-backed vector segments. `--ultra-durable` (default in the
Docker image) adds:

- **Reed-Solomon RS(10,4)** encoding of segments (GF(2⁸), any 10 of 14 shards reconstruct);
- WAL **group commit** (one `fsync` per batch of writes);
- a **health monitor** that scrubs checksums, tracks lost shards and rebuilds them;
- an in-process **Raft log** that orders writes.

**Modelled durability**, from `ReedSolomon::durability_nines`: independent disks, each lost with 0.4%
annual probability (Backblaze AFR), no repair during the year.

| Scheme | Storage overhead | Shard losses survived | Modelled durability |
|---|---:|---:|---:|
| 3-way replication | 3.0× | 2 | 7.2 nines |
| RS(6,3) | 1.5× | 3 | 7.5 nines |
| **RS(10,4)** | **1.4×** | **4** | **8.7 nines** |
| RS(14,4) | 1.29× | 4 | 8.1 nines |

**Read this table as a property of the coding scheme, not of a deployment.** Today all shards of a
standalone server live on **one host**, so losing that host's disk loses the data regardless of the
coding; use backups (`--backup`, S3/GCS upload, point-in-time restore). Repairing lost shards within
hours would add orders of magnitude to the modelled figures; that is not modelled. The in-process Raft
group gives ordering, not availability: every member runs in the same process.

---

## Experimental: replicated cluster mode

Three or more `tidevec-server` processes can replicate every write through Raft over HTTP:

```bash
PEERS="0=http://10.0.0.1:6399,1=http://10.0.0.2:6399,2=http://10.0.0.3:6399"
export TIDEVEC_CLUSTER_SECRET=...        # same on every node; guards the /raft/* endpoints
tidevec-server --port 6399 --data-dir ./data --node-id 0 --raft-peers "$PEERS"   # node 1, 2 likewise
```

How it behaves:

- **Writes** (create/drop collection, upsert, delete, edges, temporal settings) go to the leader,
  which appends them to the Raft log and answers once the write is committed and applied locally.
- **Followers reject writes** with HTTP `421` and the leader's URL (`X-TideVec-Leader` header and
  `leader` field). The Python, Go and Java clients follow the hint automatically; Python raises
  `NotLeaderError` if no leader is reachable.
- **Reads are served locally** by whichever node you ask unless the request asks for
  `consistency=linearizable` (ReadIndex) or `consistency=lease`.
- Term, vote and log are persisted with fsync and survive restarts; a torn log tail is discarded.
  Log snapshots (`InstallSnapshot` for lagging or new nodes), single-server membership changes
  (`--raft-join`, `POST /v1/admin/cluster/add_node` / `remove_node`) and pre-vote are implemented.
  `GET /v1/cluster/status` reports role, term, leader, members and commit/applied indexes.

**Not production-ready.** Every node holds a full copy (no sharding across hosts; the multi-Raft
`tidevec-router` is an experiment), DriftBridge is disabled and `--ultra-durable` is turned off. It
is covered by in-process Raft tests, multi-process tests and a chaos harness that injects crashes,
partitions, pauses, disk and clock faults and checks every history for linearizability
([`tests/chaos`](tests/chaos/README.md)), not by production use. Guide:
[`docs/operations/cluster.md`](docs/operations/cluster.md).

---

## SDKs

| Language | Package | Notes |
|---|---|---|
| Python | `pip install tidevec` (`[embedded]`, `[langchain]`, `[llamaindex]`, `[haystack]`, `[dspy]`, `[langgraph]`, `[crewai]`, `[mcp]`, `[all]` extras) | server client, `Local` embedded mode, embedders, integrations |
| Go | `sdk/go` | `go get github.com/ashishodu2023/TideVec/sdk/go` |
| Java | `sdk/java` (Maven, Java 17+) | no runtime dependencies |
| C++ | `sdk/cpp` | header-only REST client; not updated for 0.3.0 (no supersession settings, and `drop_collection` does not drop), see [docs](docs/sdks/cpp.md) |
| TypeScript | `sdk/typescript` (`tidevec`, `@tidevec/ai-sdk`, `n8n-nodes-tidevec`) | ESM + CJS, Node 18+, Deno, Bun, browsers; tested in CI, not on npm yet |

All clients speak the REST API below. **0.3.0 breaking changes:** Go `SearchOptions.TemporalBlend` is
now `*float32` (use `tidevec.Blend(x)`; `nil` means "collection default"), Go `Delete` takes
`(ids, filter)` and returns the count; Java `SearchOptions.temporalBlend` is a nullable `Float` and
`deleteVectors` returns the count.

```go
db, _ := tidevec.New("localhost:6399", tidevec.ClientConfig{APIKey: os.Getenv("TIDEVEC_API_KEY")})
resp, _ := db.Search(ctx, "docs", query, tidevec.SearchOptions{TopK: 10, Filter: "lang=en"})
n, _ := db.Delete(ctx, "docs", nil, "source=crm")
```

```java
try (TideVecClient db = TideVecClient.builder("localhost", 6399)
        .apiKey(System.getenv("TIDEVEC_API_KEY")).build()) {
    SearchResponse r = db.search("docs", query, SearchOptions.builder().topK(10).build());
    for (SearchHit h : r.hits()) if (h.isSuperseded()) System.out.println(h.id() + " is outdated");
}
```

---

## REST API

```
GET    /health                               Server status (no auth)
GET    /v1/info                              Version and feature manifest (no auth)
GET    /metrics                              Prometheus metrics (no auth)
GET    /v1/cluster/status                    Raft role/term/leader, or {"mode":"standalone"}

GET    /v1/collections                       List collections
POST   /v1/collections                       Create collection (temporal, agent, supersession settings)
GET    /v1/collections/{name}                Collection stats
DELETE /v1/collections/{name}                Drop collection
POST   /v1/collections/{name}/upsert         Upsert vectors                  [rate-limited]
POST   /v1/collections/{name}/delete         Delete by ids and/or filter
POST   /v1/collections/{name}/search         Vector, keyword (BM25) or hybrid search  [rate-limited]
GET    /v1/collections/{name}/vectors/{id}   Get one vector
POST   /v1/collections/{name}/get            Get many by id
POST   /v1/collections/{name}/scroll         Page through vectors (filter, as_of)
POST   /v1/collections/{name}/ids            List ids (filter, as_of)
POST   /v1/collections/{name}/count          Count (filter, as_of)
POST   /v1/collections/{name}/compact        Start compaction (GET: status)
POST   /v1/collections/{name}/edges          Add typed edges
PUT    /v1/collections/{name}/temporal       Update half-life / blend

GET    /v1/aliases                           List aliases
PUT    /v1/aliases/{alias}                   Create or atomically repoint an alias
DELETE /v1/aliases/{alias}                   Remove an alias

POST   /v1/collections/{name}/drift/start    Start model migration (standalone only)
GET    /v1/collections/{name}/drift/status   Migration progress
POST   /v1/collections/{name}/drift/abort    Abort migration

POST   /v1/admin/backup                      Trigger snapshot
GET    /v1/admin/backups                     List snapshots
GET    /v1/admin/backups/manifests           Point-in-time manifest history
POST   /v1/admin/restore                     Restore from snapshot
POST   /v1/admin/cluster/add_node            Cluster mode: add a node (remove_node likewise)
```

`/v1/*` routes need `X-Api-Key` when auth is enabled. Errors: `400` bad request, `401`, `403`
(tenant, quota or URL not allowed), `404` unknown collection, `421` not the Raft leader, `429` rate
limited, `503` leadership changed during a write.

The full reference, with request fields, defaults and status codes, is generated from the route
registrations by `scripts/gen_api_docs.py` into [`docs/api/rest.md`](docs/api/rest.md).

---

## Running in production

| Feature | Docker default | Flag / env |
|---|---|---|
| Auth | on (key generated and printed if unset) | `--require-auth`, `TIDEVEC_API_KEY`, `TIDEVEC_REQUIRE_AUTH=0` to disable |
| Ultra-durable stack | on | `--ultra-durable`, `TIDEVEC_ULTRA_DURABLE=0` |
| Segment store | on | `--segment-store`, `TIDEVEC_SEGMENT_STORE=0` |
| Backups (local, S3, GCS) | off | `--backup --backup-s3 s3://... --backup-gcs gs://...`, `TIDEVEC_BACKUP=1`, `TIDEVEC_BACKUP_S3`, `TIDEVEC_BACKUP_GCS`; snapshots go to the `/backups` volume (S3/GCS upload needs the `aws`/`gsutil` CLIs, not in the image) |
| Rate limit | 100 req/s per client | `--rate-limit N`, `TIDEVEC_RATE_LIMIT` |
| OTel export | off | `--otel --otel-endpoint URL`, `TIDEVEC_OTEL_ENDPOINT` (JSON log lines; OTLP span export is experimental) |
| TLS | off | `--tls-cert/--tls-key` (build with `-DTIDEVEC_TLS=ON`; the Docker images have it: `TIDEVEC_TLS_CERT`, `TIDEVEC_TLS_KEY`), or a reverse proxy such as the compose `tls` profile (nginx) |
| Multi-tenancy | off | `--multi-tenant`, `TIDEVEC_MULTI_TENANT=1` + `tenants.json` (per-tenant keys, prefixes, quotas); not an isolation boundary, see [docs](docs/operations/auth.md) |
| DriftBridge allowlist | none | `--reembed-allow h1,h2`, `--reembed-allow-private`, `TIDEVEC_REEMBED_ALLOW` |
| Cluster mode (experimental) | off | `--node-id`, `--raft-peers`, `TIDEVEC_NODE_ID`, `TIDEVEC_RAFT_PEERS`, `TIDEVEC_CLUSTER_SECRET` |

The data directory holds `registry.json`, a WAL per shard, segments, `tenants.json` and (in cluster
mode) `raft/`. Keep backups outside it: snapshots archive the whole data directory. **Changing the API
key does not revoke the old one** in 0.3.0, because the first key is stored in `tenants.json`; edit
that file to revoke it ([details](docs/operations/auth.md)).

A Helm chart (`deploy/helm/tidevec`, StatefulSet + PVC + Secret) and a compose file with
Prometheus/Grafana/Jaeger (`docker compose -f docker/docker-compose.yml --profile monitoring up`) are
included. The env variables above apply to the CPU image; the GPU and TPU images start the server
with fixed flags. Operations guide: [`docs/operations/`](docs/operations/deployment.md).

---

## Architecture

```
            Python (embedded: tidevec_native)          REST clients (Python, Go, Java, C++)
                          │                                         │
                          │                          tidevec-server: REST :6399 (cpp-httplib)
                          │                            auth · rate limit · tenants · OTel
                          │                            [experimental] Raft over HTTP between servers
                          ▼                                         ▼
              ┌───────────────────────── CollectionRegistry ─────────────────────────┐
              │  DurableCollection (default) │ UltraDurableCollection │ Accelerated   │
              │   ShardRouter → per-shard WAL → Collection                            │
              │      TVIndex (HNSW + recency + supersession)  ·  CausalGraph (edges)  │
              │   SegmentStore (disk vectors, PQ rescoring) · RS(10,4) · HealthMonitor │
              └───────────────────────────────────────────────────────────────────────┘
                     DriftBridge (shadow re-embedding) · Backups / PITR · RetrievalTrace
```

---

## Tests

| Suite | Covers |
|---|---|
| `tidevec_tests` | TVIndex recall vs. exact search, filters, upserts, tombstone compaction, outliers, temporal math, supersession, causal graph |
| `tidevec_scale_tests` | WAL crash recovery, sharding, replica sets, PQ encode/ADC, concurrent writes |
| `tidevec_api_tests` | every REST endpoint, delete-by-filter, supersession, error codes, re-embed URL allowlist |
| `tidevec_accel_tests` | AVX2 kernels, IVF, dispatcher, GPU/TPU fallbacks |
| `tidevec_eleven_nines_tests` | GF(256), RS encode/decode, Raft election/replication/persistence, health monitor, durability model |
| `tidevec_security_tests` | API keys and rotation, tenant isolation and quotas, re-embed SSRF/DNS pinning, backup/restore, OTLP export, TLS |
| `tidevec_cpp_sdk_tests` | C++ SDK against a live server |
| `tidevec_cluster_tests` | Raft snapshots, membership changes, ReadIndex/leases, pre-vote, injected faults |
| Python (`sdk/python/tests`) | embedded and server modes, every framework integration, agent memory, MCP, DriftBridge, 3-process Raft cluster |
| Go / Java / TypeScript SDKs | JSON encoding, plus live-server round trips: supersession, filters, get/scroll, as_of, hybrid, aliases |
| Chaos (`tests/chaos`) | multi-process Raft cluster under faults, checked for linearizability (nightly CI) |
| Release smoke test (`scripts/release_smoke_test.py`) | installed package or running server: version, supersession, dimension validation |

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
for t in build/tidevec_*tests; do $t || break; done
cd sdk/python && pytest                     # uses build/tidevec-server
```

CI runs all of the above on Ubuntu 22.04/24.04, Python 3.9-3.13 and Node 20/22, lints and renders the
Helm chart, builds embedded wheels for Linux x86_64, Linux arm64 and macOS (arm64, x86_64 cross-compiled),
and builds the docs with `mkdocs build --strict`. A nightly run adds the chaos harness, the Python cluster
and multi-Raft router tests, a 1M-vector filtered-search benchmark and musllinux wheels. Pushing a
`v*` tag runs [`release.yml`](.github/workflows/release.yml), which also installs the built wheels on
each platform and smoke-tests the Docker image before publishing.

---

## Platform support

| Platform | Server | Embedded wheel | Notes |
|---|:---:|:---:|---|
| Linux x86_64 | ✅ CI | ✅ | Docker CPU image amd64 + arm64 |
| Linux arm64 | ✅ (Docker) | ✅ | |
| Linux musl (Alpine) | — | ✅ from 0.3.0 releases | musllinux wheels, built and smoke-tested in release CI |
| macOS arm64 | not built in CI | ✅ | the wheel job compiles the engine headers |
| macOS x86_64 | not built in CI | ✅ from 0.3.0 releases | cross-compiled on arm64; not tested on Intel hardware |
| Windows | via WSL2 / Docker | ❌ | storage layer uses POSIX APIs |
| NVIDIA GPU | optional build | ❌ | unsupported: fails recall on hardware ([benchmark](benchmarks/gpu/README.md)) |

Compiler: GCC 12+ / Clang 15+, C++20.

---

## Scale projections (not measured)

The figures below are design targets derived from published results for the underlying algorithms,
not TideVec measurements. They are here to show the intended scaling path, not as benchmarks.

| Path | Target |
|---|---|
| PQ (96 subspaces), 1B × 768-dim | ~96 GB of codes vs. 3.1 TB raw float32 |
| GPU graph search (CAGRA design) | order-of-magnitude higher batch QPS than CPU HNSW, per the CAGRA paper |
| TPU exact matmul | exact recall for batch workloads |

---

## How TideVec compares

Capabilities as documented by each project in 2025; check their current docs before relying on this.

| | TideVec | Qdrant | Milvus | Weaviate | Pinecone |
|---|---|---|---|---|---|
| Recency in ranking | in-search blend | decay formulas over prefetched candidates | decay rankers | — | — |
| Automatic version supersession | ✅ | — | — | — | — |
| Typed edges between records | ✅ | — | — | cross-references | — |
| Embedded / in-process mode | ✅ (Python) | local mode (Python client) | Milvus Lite | embedded (experimental) | — |
| Production distributed clustering | experimental (full replicas) | ✅ | ✅ | ✅ | ✅ (managed) |
| Managed cloud | not yet | ✅ | ✅ (Zilliz) | ✅ | ✅ |
| GPU | — (CUDA path unsupported) | GPU indexing | ✅ (cuVS, incl. CAGRA) | — | n/a |
| License | Apache-2.0 | Apache-2.0 | Apache-2.0 | BSD-3-Clause | proprietary |

TideVec is young: the established engines are far ahead on scale, operations and ecosystem. Its focus
is narrower: retrieval over content that changes, and memory for agents.

---

## Research foundation

| Component | Reference | Status |
|---|---|---|
| TVIndex temporal scoring inside HNSW | Original contribution | Implemented |
| Write-time supersession linking | Original contribution | Implemented |
| CausalEdge co-located typed graph | Original contribution | Implemented |
| HNSW | Malkov & Yashunin, IEEE TPAMI 2020 | Modified (temporal scoring) |
| CAGRA search design | Ootomo et al., ICDE 2024 | Independent CUDA implementation; fails recall on hardware, unsupported |
| Reed-Solomon erasure coding | Plank, FAST 2005 | Implemented |
| Raft consensus | Ongaro & Ousterhout, USENIX ATC 2014 | Implemented (snapshots, membership changes, ReadIndex, pre-vote); experimental |

A paper on temporally-aware retrieval for agents is in preparation.

---

## Repository layout

```
include/tidevec/
  core/           cortex_vector, temporal_scorer, metrics, supersession, collection
  index/          tv_index (HNSW + recency), flat_index
  graph/          causal_graph
  cluster/        durable_collection, shard_router, replica_set, ultra_durable_collection
  consensus/      raft (log, election, persistence), raft_http (HTTP transport)
  erasure/        reed_solomon
  health/         health_monitor
  storage/        wal, segment_store
  drift/          drift_bridge, drift_http
  accelerator/    cpu/gpu/tpu engines, dispatcher
  api/            rest_server, collection_registry, server_security, json_serializers
src/              server_main.cpp, CUDA and XLA kernels
sdk/              python, python-embedded (pybind11), go, java, cpp, typescript
examples/         quickstart, LangChain, LlamaIndex, Haystack, LangGraph, CrewAI, DSPy, TypeScript, MCP, DriftBridge, Colab notebook
benchmarks/       temporal_real_data, temporal_v2, agent_memory, ann, gpu, temporal_vs_qdrant (synthetic)
tests/            unit, integration, cluster (Raft), chaos (fault injection + linearizability checker)
docker/ deploy/   images, compose, Helm chart
packaging/        tidevec-mcp (PyPI launcher), MCP registry entry, Homebrew formula
docs/             MkDocs site (mkdocs.yml at the root); docs/api/rest.md is generated
scripts/          gen_api_docs.py, check_release_versions.py, release_smoke_test.py, sync_embedded_headers.sh
```

---

## Roadmap

See [ROADMAP.md](ROADMAP.md): phased plan with exit criteria covering filters and payload indexes,
hybrid search, multi-host sharding, Raft hardening, ecosystem integrations, distribution and a
managed service.

---

## License

Apache 2.0, see [LICENSE](LICENSE).

<div align="center">

Built by [Ashish Verma](https://github.com/ashishodu2023) · Part of the [Cortex Platform](https://getcortexops.com)

**[tidevec.com](https://tidevec.com) · [GitHub](https://github.com/ashishodu2023/TideVec) · [PyPI](https://pypi.org/project/tidevec/)**

</div>
