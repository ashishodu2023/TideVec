# TideVec Roadmap

Status as of 0.3.0 (October 2026). Durations are estimates for a small team (1-2 engineers) and
phases overlap where noted. Each phase ends with exit criteria that can be checked, not just shipped
features.

---

## 1. Where TideVec stands

**Works and is tested** (217 C++ tests across 8 suites, 171 Python, 8 Go, 12 Java, 34 TypeScript,
7 Python cluster tests and 11 linearizability-checker tests; CI on every push, with the Python
cluster tests and a chaos run nightly):

- HNSW index with recency blended into the search, per-query blend, `ef_search`, optional PQ.
- Retrieval: get-by-id, scroll, list ids and count; JSON filters (`$in`, ranges, dates,
  `$and`/`$or`/`$not`) with payload indexes and a filter-aware planner; `as_of` point-in-time
  queries; BM25 hybrid search; collection aliases; delete by ids or filter; background compaction.
- Write-time **supersession** (version linking by key or similarity) and typed edges
  (`UPDATES`, `CONTRADICTS`, ...), mirrored across shards.
- WAL with replay; disk segments; backups with S3/GCS upload and point-in-time restore; auth, rate
  limits, multi-tenant keys and quotas; OTel traces.
- Embedded Python mode (Linux x86_64/arm64, macOS arm64; musllinux and macOS x86_64 wheels from the
  0.3.0 release workflow), integrations for LangChain, LlamaIndex, Haystack, DSPy, LangGraph and
  CrewAI, `AgentMemory` + `tidevec-mcp` server, Go/Java/C++ REST clients, TypeScript client with
  Vercel AI SDK and n8n packages (not on npm yet), Helm chart.
- DriftBridge model migration that preserves concurrent writes (same dimension, standalone only).
- Published benchmarks: real-data temporal, temporal v2, LongMemEval agent memory, SIFT-1M/GloVe-100
  ANN (where TideVec trails hnswlib and FAISS).

**Experimental:** multi-process Raft cluster mode (full replicas; snapshots, membership changes,
ReadIndex/lease reads and pre-vote; chaos-tested with a linearizability checker) and the multi-Raft
`tidevec-router`. **Unsupported:** the GPU (CUDA, CAGRA-style) path, which fails recall on real
hardware, and the TPU path.

**Missing:** multi-host sharding in production form, atomic DriftBridge swap, real gRPC (the
`.proto` exists, no server), Windows, packages on npm and Maven Central, a managed service.

---

## 2. Positioning

**Wedge: retrieval over knowledge that changes, and memory for agents.**

General-purpose vector databases (Qdrant, Milvus, Weaviate, Pinecone) are far ahead on scale, ops and
ecosystem; competing there head-on is not winnable. What none of them do is know that a document *was
replaced*. TideVec's defensible claims, each backed by the real-data benchmark:

1. Plain similarity returns outdated versions most of the time (55–85% on Wikipedia revisions).
2. Recency reranking fixes the top hit but leaves ~1 stale version in every top-5 context.
3. Write-time supersession removes them, with no time window to tune.

Target users, in order:

- **Agent builders** who need memory where facts get updated: MCP users, LangGraph/CrewAI
  developers.
- **RAG teams with versioned corpora:** support knowledge bases, policies and handbooks, product
  docs, contracts, changelogs.
- **Local-first / embedded users** who want `pip install` without running a server.

Positioning to avoid: "fastest" or "most durable" anything until benchmarks support it.

---

## 3. Phases

### Phase 0: Release 0.3.0 (now, ~2-4 weeks)

- Tag `v0.3.0`: CI publishes `tidevec` and `tidevec-embedded` (cibuildwheel: manylinux x86_64/aarch64,
  macOS arm64; CPython 3.9-3.13) to PyPI and versioned Docker tags.
- Tag `sdk/go/v0.3.0` for the Go module. Publishing to Maven Central needs a Sonatype account and a
  signing key (account decision).
- Docs site: quickstart, concepts (recency, supersession, edges), API reference generated from the
  REST handlers, the five examples.

**Exit criteria:** on a clean Linux x86_64, Linux arm64 and macOS arm64 machine,
`pip install "tidevec[embedded]"` followed by `examples/01_quickstart_embedded.py` works in under
5 minutes; `docker run` plus the README Python snippet works; release CI fully green.

### Phase 1: Retrieval completeness (1-3 months, starts with Phase 0)

**Status (0.3.0):** items 1-4, aliases and background compaction shipped in every SDK except C++;
the atomic DriftBridge swap and WAL/segment compaction are open. Exit criteria: a nightly CI test
asserts recall@10 ≥ 0.95 for 1% range, keyword, AND and `created_at` filters on 1M vectors; the BEIR
hybrid comparison has not been run.

Ordered by what integrations and users hit first:

1. **Get-by-id, scroll, list ids** (REST, all SDKs, embedded). LangChain `get_by_ids` and LlamaIndex
   node fetch depend on it, and so do MCP `history` and export.
2. **Filters:** AND/OR across keys, `IN`, numeric and date ranges (including `created_at`), plus
   **payload indexes** (inverted for keywords, sorted for numbers). Filter-aware planning: exact scan
   when a filter is very selective, graph traversal otherwise.
3. **As-of queries:** `as_of=<timestamp>` returns what was current at that time (vectors created
   before t, supersession state at t). This extends the wedge from "latest" to "versioned retrieval"
   (audits, reproducing an answer an agent gave last month).
4. **Hybrid search:** BM25 sparse index with reciprocal-rank or weighted fusion; exact-match terms
   such as product codes and error IDs fail with dense-only retrieval.
5. **DriftBridge v2:** atomic per-shard index swap logged in the WAL; dimension changes through a new
   collection plus **collection aliases** for zero-downtime cut-over.
6. **Storage compaction:** background (non-blocking) index rebuilds (today compaction runs under the
   write lock); WAL and segment compaction for update-heavy workloads.

**Exit criteria:**

- Range and multi-key filters hold recall@10 ≥ 0.95 at 1% selectivity on 1M vectors.
- Hybrid beats dense-only nDCG@10 on at least 3 BEIR datasets.
- `as_of` is covered by tests against a brute-force oracle.
- Every SDK exposes get, scroll, filters and hybrid.

### Phase 2: Benchmarks and validation (1-3 months, parallel with Phase 1)

**Status (0.3.0):** ANN (SIFT-1M, GloVe-100 vs. hnswlib and FAISS), temporal v2 (reimplemented
Qdrant/Milvus decay baselines, not live instances), LongMemEval agent memory (mem0/Zep and LoCoMo not
yet) and the GPU run (result: unsupported) are published in `benchmarks/`; Cohere-1M is not.

- **Standard ANN numbers** with VectorDBBench or ann-benchmarks (SIFT-1M, GloVe-100, Cohere-1M):
  recall vs. QPS, build time, memory, p99 latency. Publish them even where TideVec trails.
- **Temporal benchmark v2:**
  - corpora: about 5k Wikipedia articles, a policy/handbook corpus, changelog Q&A
  - queries written by people, not templates
  - three embedding models
  - live baselines: Qdrant decay formulas and Milvus decay rankers
- **Agent-memory benchmark:** the knowledge-update subsets of LongMemEval and LoCoMo. Compare
  `AgentMemory` against mem0, Zep and a plain vector store.
- **GPU validation:** run the CUDA path on A100/H100 against cuVS CAGRA. Either publish results and
  support it, or mark it unsupported and stop advertising it.

**Exit criteria:** results reproducible from scripts in `benchmarks/` and published. Every performance
claim in the README links to one of them.

### Phase 3: Distributed TideVec (3-6 months, after Phase 1 items 1-2)

**Status (0.3.0):** snapshots/`InstallSnapshot`, single-server membership changes, ReadIndex and
lease reads, pre-vote and the chaos harness with a linearizability checker (nightly CI) are done;
multi-Raft exists only as the experimental `tidevec-router`. The 24-hour chaos run, replicated
DriftBridge, cross-host erasure coding and the 100M-vector test are open.

- **Raft hardening:**
  - log snapshots and `InstallSnapshot`
  - membership changes (single-server changes)
  - linearizable reads (ReadIndex or leases), pre-vote
  - DriftBridge as replicated log operations
- **Fault testing:** partitions, `kill -9`, disk-full and clock-skew runs checked with a
  linearizability checker (Porcupine/Knossos style), run in nightly CI.
- **Multi-host sharding:** one Raft group per shard range (multi-Raft), placement and rebalancing,
  scatter-gather queries across hosts.
- **Cross-host erasure coding** of segments, which makes the RS(10,4) durability model apply to a
  real deployment.

**Exit criteria:**

- A 3-node cluster survives a 24-hour chaos run with no linearizability violations and no data loss.
- A node can be replaced without downtime.
- 100M vectors across 6 hosts, with published query latency.

### Phase 4: Ecosystem and distribution (continuous, from Phase 0)

**Status (0.3.0):** in-repo Haystack, DSPy, LangGraph and CrewAI integrations, the TypeScript client,
`@tidevec/ai-sdk` and the n8n node, MCP setups for more clients, a Helm chart, a Homebrew formula and
musllinux/macOS x86_64 wheels exist; upstream listings and npm publishing are open.

| Integration | Prerequisite | Notes |
|---|---|---|
| LangChain partner package (`langchain-tidevec`), LlamaIndex upstream | Phase 1.1 | Moves the existing integrations into the official directories |
| Haystack `DocumentStore` | Phase 1.1-1.2 | Haystack needs rich filters |
| DSPy retriever | none | Small; research audience |
| TypeScript client on npm, then the Vercel AI SDK provider | none | Opens the JS agent ecosystem |
| n8n node | TS client | Low-code agent builders |
| More MCP clients (Claude Code, Windsurf, Zed, VS Code), listings on MCP directories | none | Highest-leverage channel for agent memory |
| LangGraph / CrewAI memory adapters | none | Built on `AgentMemory` |

Distribution:

- npm (TypeScript client)
- Helm chart on Artifact Hub
- Homebrew formula for the server
- `musllinux` and macOS x86_64 wheels
- **Windows**: port the POSIX storage layer (`open`/`fsync`/`mmap`) behind a platform abstraction,
  then add Windows wheels.

**Exit criteria (per integration):** documented, tested in CI against a live server, and listed in
the upstream project's integration directory.

### Phase 5: Managed cloud (gated; earliest after Phase 3)

A multi-tenant service is not responsible to run on full-replica, unsharded clusters. That gate comes
first. Groundwork already exists: per-tenant keys and quotas, backups/PITR, OTel, Helm.

**Business decisions required before engineering starts:**

- **Billing:** usage-based (stored vectors and queries) or tiers; payment provider; free tier.
- **Cloud accounts and regions:** start with one provider and region; data-residency options.
- **SLA:** availability target, support hours, on-call rotation.
- **Security and compliance:**
  - SOC 2 (Type I, then Type II) and a GDPR DPA
  - encryption at rest with customer-managed keys (KMS)
  - audit logs, SSO
- **Commercial model:** pricing against Qdrant Cloud, Zilliz and Pinecone serverless; open-core
  boundary (which features stay Apache-2.0); legal entity and terms.

**Engineering:**

- control plane: tenant provisioning on Kubernetes, upgrades, backups to object storage
- metering pipeline and usage dashboards
- per-tenant encryption
- status page

**Exit criteria:** private beta with 3-5 design partners; 99.9% measured availability over 30 days;
zero data-loss incidents; documented incident response.

---

## 4. Go-to-market (alongside Phases 0-2)

1. **Launch post:** "Your RAG is quoting last year's policy". Built on the real-data benchmark,
   including where TideVec only ties recency reranking.
2. **Channels:** Show HN, r/LocalLLaMA, r/MachineLearning, the LangChain and LlamaIndex community
   showcases.
3. **Agent-memory launch:**
   - publish `tidevec-mcp` to the MCP directories (Smithery, mcp.so, Cursor's list)
   - show a short demo of a fact being updated and recalled correctly
4. **Notebook and preprint:**
   - a Colab notebook (exists) and an arXiv preprint of the temporal and supersession method
   - temporal benchmark v2 as the paper's evaluation
5. **Design partners:** 5-10 teams with versioned knowledge (support, legal/policy, internal docs,
   agent products).
   - Offer: direct support and roadmap influence.
   - Ask: a corpus or anonymised eval set, and a monthly feedback call.
   - Selection criteria: documents revised at least monthly, an existing RAG or agent in production,
     and willingness to share retrieval metrics.

Rule for all material: every number links to a reproducible benchmark or test; projections are
labelled as such.

---

## 5. Metrics

| Area | Metric | Phase 0-2 target |
|---|---|---|
| Adoption | PyPI downloads/week (`tidevec`, `tidevec-embedded`), Docker pulls, GitHub stars | 1k downloads/week by end of Phase 2 |
| Activation | Share of new users who run a search (opt-in telemetry or docs funnel; decide policy first) | measured, baseline set |
| Community | Issues and PRs from outside contributors per month, integrations listed upstream | 5 external PRs/month; 4 upstream listings |
| Design partners | Signed, active (monthly call held), in production | 5 signed, 2 in production |
| Retrieval quality | Recall@10 on SIFT-1M / Cohere-1M per release; Current@1 and stale-in-top-5 on temporal benchmark v2 | no regression release to release |
| Performance | p50/p99 search latency and QPS at fixed recall, build throughput | published per release |
| Reliability | CI pass rate, flaky-test count, data-loss/corruption bugs, time to fix | 0 open data-loss bugs; flake rate < 1% |
| Business (Phase 5) | Beta tenants, availability, gross margin per tenant, paid conversion | set when Phase 5 starts |

---

## 6. Open decisions

- **gRPC:** implement the server from `proto/tidevec.proto`, or delete the proto and stay
  REST-only? Decide when a user needs streaming or lower per-call overhead.
- **Telemetry:** none today. Opt-in only, if any, and announced before it ships.
- **Windows priority:** port the storage layer, or keep WSL2/Docker as the answer?
- **Open-core boundary:** which features (multi-host, tenancy, cloud control plane) stay Apache-2.0.
