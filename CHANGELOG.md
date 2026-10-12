# Changelog

All notable changes to TideVec are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/) (pre-1.0: minor versions may break APIs).

Versions apply to the server, the Docker images, `tidevec`, `tidevec-embedded` and `tidevec-mcp`
on PyPI, the Go module (`sdk/go/vX.Y.Z` tag), the Java SDK, the C++ SDK, the TypeScript packages
and the Helm chart (`scripts/check_release_versions.py` checks all of them).

## [Unreleased]

Nothing yet.

## [0.3.0] - unreleased

Focus of this release: retrieval over knowledge that changes (supersession), correctness of the
HNSW index, retrieval APIs (get/scroll, rich filters, `as_of`, hybrid search, aliases), embedded
Python mode, framework integrations in Python and TypeScript, published benchmarks, a hardened
experimental Raft cluster mode, and documentation that only claims what is tested.

### Breaking changes

- **Per-query `temporal_blend` now defaults to the collection's setting.** A search request
  without `temporal_blend` used to be scored with a fixed 0.3 (and an explicit value was ignored,
  see Fixed). The REST field is now optional; omitting it uses the collection's
  `temporal.temporal_blend`, and `0.0` gives pure similarity.
- **Python:** `TideVec.search(..., temporal_blend=None)` is the new default (previously `0.3`).
  `None` means "use the collection setting".
- **Go:** `SearchOptions.TemporalBlend` is now `*float32` (use `tidevec.Blend(x)`; `nil` means
  collection default). `Delete` takes `(ids, filter)` and returns the count.
- **Java:** `SearchOptions.temporalBlend` is a nullable `Float`; `deleteVectors` returns the count.
- **Dimension validation:** upserting a vector whose length differs from the collection's `dim`
  is rejected with `400` (Python `APIError`; embedded mode raises `TideVecError`). Previously one such vector could change the
  collection's dimension and break every later search.
- **`404` for a missing collection** on search, upsert, delete, edges and temporal updates
  (previously `400`). The Python client raises `CollectionNotFoundError`.
- **Docker images no longer expose port 6400.** No gRPC server exists; only REST on 6399 is served.
- **Docker CPU image writes backups to `/backups`** (a new volume) instead of `/data/backups`.
  Snapshots archive all of `/data`, so with the old default every snapshot contained all earlier
  ones. Move existing snapshots, or set `TIDEVEC_BACKUP_DIR`.
- **Docker images serve HTTPS when `TIDEVEC_TLS_CERT` and `TIDEVEC_TLS_KEY` are set** (all three
  images are built with `-DTIDEVEC_TLS=ON`). Previously the server ignored the certificate and
  served plain HTTP. The GPU and TPU images now use the same entrypoint and `TIDEVEC_*` variables
  as the CPU image (they used to start the server with fixed flags, ignoring most of them).
- **`--no-require-auth` / `TIDEVEC_REQUIRE_AUTH=0` turn authentication off even when an API key is
  configured** (with a warning at startup); before, the key was still demanded. Combining it with
  `--multi-tenant` is refused at startup. The Docker entrypoint no longer generates a key when
  auth is off, and `TIDEVEC_ULTRA_DURABLE=0` / `TIDEVEC_SEGMENT_STORE=0` now actually turn those
  features off.
- **Durability figures corrected.** `ReedSolomon::durability_nines` added two unjustified nines;
  RS(10,4) models at 8.7 nines (not 10.7). Anything that displayed the old figure changes.
- **The configured API key is the only admin key.** It is no longer written to `tenants.json`;
  a `default` entry left there by earlier versions is removed at startup, so the previous key
  stops working. `tenants.json` keys are only accepted with `--multi-tenant`.
- **Tenants are isolated.** A tenant without `collection_prefixes` is limited to `<id>_*`
  (`[""]` grants everything); admin routes (`/v1/admin/*`, `/v1/cluster/*`) need the admin key;
  quotas count the tenant's own collections and vectors.
- **Collection and alias names** must match `[A-Za-z0-9_.-]{1,128}`, not start with `.`, and not be
  `raft`, `registry.json` or `tenants.json` (`400` otherwise).
- **`--tls-cert` / `--tls-key` on a build without TLS** now exit with an error instead of serving
  plain HTTP.
- **Restore** reloads collections in place (`restart_required: false`) and is refused in cluster
  mode. Snapshots now archive the data directory's contents; older snapshots still restore.
- **DriftBridge `reembed_url` hosts are resolved when the migration starts.** A host name that
  resolves to a loopback, private or link-local address is refused (`403`) unless it is listed in
  `--reembed-allow` and `--reembed-allow-private` is set; an unresolvable host is a `400`.

### Added

- **Supersession:** on upsert, a new vector is linked to the older versions it replaces (same
  `match_field` value and/or `vector_score` above `similarity_threshold`) with an `UPDATES` edge,
  mirrored as `superseded_by` across shards. Configured per collection (`supersession` block on
  create); search accepts `superseded_penalty` (1.0 pushes superseded versions to the bottom).
- **Retrieval APIs** (REST, Python server and embedded clients, Go, Java, TypeScript; not the C++
  SDK): `GET /v1/collections/{name}/vectors/{id}`, `POST .../get` (many ids, unknown ones listed in
  `missing`), `POST .../scroll` (cursor paging in id order), `GET`/`POST .../ids` and
  `POST .../count`, each with optional filter and `as_of`. Python adds `iterate()` and `list_ids()`.
- **Rich JSON filters** on search, scroll, ids, count and delete: equality, `$eq`, `$ne`, `$in`,
  `$nin`, `$gt`/`$gte`/`$lt`/`$lte` on numbers and ISO dates, `$exists`, `$and`/`$or`/`$not`, `_id`
  and `created_at`. The old `"key=value"` string form still works. **Payload indexes** (inverted for
  keywords, sorted for numbers and dates, always on for `created_at`; `payload_index: false` per
  collection turns them off) feed a filter planner that picks graph traversal, an exact search over
  the indexed matches, or an exact scan.
- **Delete by filter:** `POST /v1/collections/{name}/delete` accepts `filter` (string or JSON) in
  addition to `ids`, and returns `deleted`. Python: `delete(collection, ids=None, *, filter=None)`.
- **`as_of` point-in-time queries** on search, scroll, ids and count: only vectors created at or
  before the timestamp, with recency and supersession evaluated as of that moment.
- **Hybrid search:** collections created with `text_field` keep a BM25 index; search takes `text`
  (keyword-only without a vector) and `hybrid: "rrf" | "weighted"` with `rrf_k` (default 60) and
  `alpha`. Results carry `text_score`.
- **Collection aliases:** `GET /v1/aliases`, `PUT`/`DELETE /v1/aliases/{alias}`; any collection
  route accepts an alias, and repointing is atomic.
- **Compaction:** indexes are rebuilt in the background (reads and writes continue) once deleted and
  replaced vectors reach `compaction_ratio` × live vectors (default 1.0, per collection);
  `POST /v1/collections/{name}/compact` forces it and `GET` reports its status.
- **Embedded mode:** `tidevec-embedded` (pybind11 wheels for Linux x86_64/aarch64 and macOS
  arm64, CPython 3.9-3.13) and `tidevec.Local(path)` with the same methods as the REST client.
  Install with `pip install "tidevec[embedded]"`. The release workflow also builds musllinux
  (x86_64/aarch64) wheels and macOS x86_64 wheels (cross-compiled on arm64, not tested on Intel),
  and installs the musl wheels on Alpine before publishing.
- **Python integrations:** LangChain `TideVecVectorStore`, LlamaIndex `TideVecVectorStore`,
  Haystack `TideVecDocumentStore` + `TideVecEmbeddingRetriever`, DSPy `TideVecRetriever`, LangGraph
  `TideVecStore` + `memory_tools`, CrewAI `TideVecStorage` (extras `haystack`, `dspy`, `langgraph`,
  `crewai`; all drop superseded versions and accept a server client or `Local`).
  `tidevec.memory.AgentMemory` (`remember` / `recall` / `history` / `forget`, keyed facts
  supersede older ones) and the `tidevec-mcp` MCP server. Optional LLM-judged contradiction
  detection (`tidevec.contradictions.detect_contradictions`) that records `CONTRADICTS` /
  `SUPPORTS` edges. Offline `HashingEmbedder`, plus OpenAI and sentence-transformers embedders.
- **TypeScript SDK** (`sdk/typescript`, not on npm yet): `tidevec` client (ESM + CJS, Node 18+,
  Deno, Bun, browsers) with the retrieval APIs and `AgentMemory`; `@tidevec/ai-sdk` (Vercel AI SDK
  memory tools, search tool and RAG middleware); `n8n-nodes-tidevec` community node.
- **MCP packaging:** `tidevec-mcp` launcher package (`packaging/pypi/tidevec-mcp`, for
  `uvx tidevec-mcp`; built and checked in CI, uploaded to PyPI separately), an MCP registry
  entry (`packaging/mcp/server.json`), a Dockerfile and Smithery config, and setup guides for
  Claude Desktop, Claude Code, Cursor, Windsurf, Zed, VS Code and remote HTTP clients
  (`examples/mcp/`). Homebrew formula for the server (`packaging/homebrew`).
- **Experimental replicated cluster mode:** Raft over HTTP between `tidevec-server` processes
  (`--node-id`, `--raft-peers`, `TIDEVEC_CLUSTER_SECRET`), persisted term/vote/log, followers
  answer writes with `421` and the leader URL (clients follow it), `GET /v1/cluster/status`.
  - Log snapshots every `--raft-snapshot-threshold` applied entries (default 4096) and
    `InstallSnapshot` for lagging or new nodes.
  - Single-server membership changes: `--raft-join`, `POST /v1/admin/cluster/add_node` /
    `remove_node`; new nodes catch up as learners before voting; one change at a time (`409`).
  - Linearizable reads: `?consistency=linearizable` (ReadIndex) or `lease`, or the
    `X-TideVec-Consistency` header, on search, get, scroll, ids, count and collection stats.
  - Pre-vote; `--raft-election-timeout-ms` (default 400) and `--raft-heartbeat-ms` (default 100).
  - Chaos harness (`tests/chaos/`): a real multi-process cluster under crashes, partitions, pauses,
    disk-full and clock faults, with a linearizability checker for the recorded history. CI-mode
    runs recorded in `tests/chaos/README.md` (7 seeds on 3 nodes, plus a 5-node run) passed with 0
    violations; a `local`-reads self-test run caught the expected stale read.
  - `tidevec-router` (experimental): routes ids across several Raft groups (multi-Raft).
- **Benchmarks** (scripts and results in `benchmarks/`, summary in `docs/benchmarks.md`):
  - `ann/`: SIFT-1M and GloVe-100 against hnswlib and FAISS HNSW. TideVec trails both: at
    recall@10 ≥ 0.95 on SIFT-1M it reaches 2,656 QPS (1 thread) vs. 11,401 (hnswlib) and 5,074
    (FAISS), and builds take 1,767 s vs. 231 s and 547 s.
  - `temporal_v2/`: Wikipedia policy pages and PyPI READMEs, three embedding models, template and
    natural-language queries, reimplemented Qdrant/Milvus decay baselines. Supersession by key had
    the highest pooled Current@1 in every corpus and scenario (82.1-91.4% vs. 69.6-82.9% for the
    best decay and 32.0-34.6% for cosine).
  - `agent_memory/`: LongMemEval knowledge-update (72 questions, k = 5). `AgentMemory` returned an
    outdated fact as the top hit for 4.2% of questions vs. 27.8% for a plain vector store.
  - `gpu/`: the CUDA path on real hardware. Graph search recall@10 was 0.003, so the GPU path is
    unsupported.
- **CI:** TypeScript SDK on Node 20 and 22 against a live server; Helm lint and `helm template`
  variants; `tidevec-mcp` build and `twine check`; all Python framework extras on Python 3.10 and
  3.13 (fails on skipped tests); security, C++ SDK and Raft cluster suites. Nightly (and on manual
  dispatch): 1M-vector filtered-search test, chaos run, Python cluster tests, musllinux wheels.
- **DriftBridge:** `--reembed-allow-private` to allow a local or in-VPC re-embedding service
  (cloud metadata endpoints stay blocked); dimension-changing migrations are rejected up front.
- **Server / Docker / Helm:** `--backup-interval-hours` (`TIDEVEC_BACKUP_INTERVAL_HOURS`,
  `backup.intervalHours`); `TIDEVEC_LOG_LEVEL` (`debug`/`info` log each request, `warn`/`error`
  do not; `config.logLevel`) and `TIDEVEC_QUIET=1`; native TLS in the images
  (`TIDEVEC_TLS_CERT` / `TIDEVEC_TLS_KEY`) and the chart (`tls.enabled`, `tls.secretName`: probes,
  ServiceMonitor and `helm test` switch to HTTPS); arguments given to the container are passed to
  the server after the generated ones.
- **Search:** `ef_search` per query; HNSW neighbour-selection heuristic.
- **Health monitor** lowers the durability estimate when shards are lost.
- **Examples** (`examples/01`-`09`, `10_typescript/`, `mcp/` and a Colab notebook) and the real-data temporal benchmark
  (`benchmarks/temporal_real_data/`: 270 Wikipedia revisions of 55 articles).
- **Go and Java SDKs** rewritten without gRPC dependencies, with live-server tests and the
  retrieval, filter, `as_of`, hybrid, alias and compaction APIs; Python client `count`,
  `has_collection`, `cluster_status`.
- **Release tooling:** `.github/workflows/release.yml` (PyPI, Docker Hub, Go module tag, GitHub
  Release), `scripts/check_release_versions.py`, `scripts/release_smoke_test.py`, docs site
  (`mkdocs.yml`, `docs/`) with a REST reference generated by `scripts/gen_api_docs.py`.
- `CMakeLists.txt`: `TIDEVEC_MARCH` option (default `native`); release Docker images are built
  for `x86-64-v3` so they run on CPUs other than the build machine's.
- `LICENSE` file (Apache-2.0).

### Fixed

- **Index (TVIndex):**
  - search used reversed priority queues; recall@10 on collections larger than ~128 vectors was
    near 0 and is now 0.97-0.999 in the unit tests
  - re-embedding the entry-point node disconnected the graph
  - re-embedding an id in place left stale graph links; it now tombstones the old node and inserts
    a new one (metadata-only updates stay in place)
  - outlying vectors and vectors re-embedded during a migration could become unreachable
  - a full re-embed could leave queries with no results
  - the per-query `temporal_blend` was ignored
  - one wrongly-sized vector could change the collection's dimension
- **Collections and API:**
  - deleting a non-existent id was counted as deleted
  - missing collection returned `400` instead of `404`
  - `CONTRADICTS` edges were not symmetric and not mirrored across shards; edges were lost on
    restart
  - dropping a collection left its WAL behind
  - tenant checks were missing on drop, delete, edges and temporal updates
  - score-space mix-ups between raw cosine and the `[0, 1]` similarity
- **Raft:** dangling reference while applying entries; commit rule ignored the current term; no
  no-op entry after election; replication RPCs ran one peer at a time; nothing was persisted; a
  failed `listen()` was silent. Found by the chaos harness: a candidate whose election timed out
  never retried after a split vote (leaderless windows); a leader whose disk stayed full kept
  leadership and rejected every write; a write could be confirmed by a different entry at the same
  index after an election (the term is now checked).
- **DriftBridge:** deleted vectors were resurrected by the swap; concurrent updates were rolled
  back; duplicate shadow nodes; use-after-free on abort; a dimension change silently did nothing.
- **Health monitor:** a hang; lost shards were ignored in the durability estimate.
- **Security:**
  - rotating the API key did not revoke the old one; key comparisons were not constant-time
  - any tenant could list all collections, reach others' collections through aliases and call
    backup/restore; quotas used server-wide totals
  - snapshot names were passed to a shell (`tar` via `system()`), allowing command injection and
    path traversal through `/v1/admin/restore`; backups now run `tar`/`aws`/`gsutil` without a
    shell and only accept `[A-Za-z0-9_.-]+.tar.gz` names
  - restore renamed the data directory, which fails on a mount point (Docker); it now swaps the
    directory's contents; `GET /v1/admin/backups` always returned an empty list
  - the DriftBridge URL allowlist could be bypassed with `user@host`, IPv6 brackets or numeric IPv4
    forms (`2130706433`, `127.1`), and blocked public `172.2.x.x` addresses
  - rate limiting was keyed by the unverified `X-Api-Key` header, so made-up keys bypassed it
  - the Docker entrypoint passed the API key on the command line (visible in `ps`) and split
    environment values containing spaces into extra flags
  - DriftBridge only checked the `reembed_url` host name, so a public name resolving to a private
    address (or re-resolving to one later, DNS rebinding) reached internal services; the host is
    now resolved once, every address is checked, and callbacks connect to the pinned address
    (with the original `Host` header and TLS name)
  - dropping or restoring a collection while requests were using it freed it under them
    (use-after-free, crashes); requests now hold a reference, and drop/restore wait for them
    before closing the collection and touching its files
- **TLS:** the `-DTIDEVEC_TLS=ON` build did not compile.
- **Embedded Python:** invalid arguments rejected by the engine (for example
  `compaction_ratio=0`) raised a bare `ValueError`; they now raise
  `tidevec.local.InvalidArgumentError`, a `TideVecError` that is still a `ValueError`.
- **OTLP export** posted to `/` instead of the endpoint's path, with invalid trace/span ids and
  timestamps, synchronously on the search thread; it now sends valid OTLP/JSON from a background
  thread.
- **Missing-collection `404`s** returned the body `Endpoint not found`.
- **C++ SDK:** `drop_collection` did not drop; `temporal_blend` was always sent; no supersession
  settings; upsert/edges/temporal used wrong status codes, routes or fields; search results were
  never parsed; `tls = true` was silently ignored (now rejected); the CMake include path pointed at
  the server headers.
- **Go SDK:** `temporal_blend` always sent; `Info` decoding; unrunnable test file; unused gRPC
  dependencies in `go.mod`.
- **Java SDK:** no JSON escaping; wrong drop route; edges and timestamps ignored; blend always
  forced.
- **Docker / CI:** phantom gRPC port; Docker Hub page said auth was off by default (it is on);
  the generated API key fell back to a guessable `tv_dev_<timestamp>` (the image has no `openssl`)
  and is now 128 random bits; entrypoint now passes cluster, rate-limit, re-embed, multi-tenant and
  S3/GCS backup settings (`TIDEVEC_MULTI_TENANT`, `TIDEVEC_BACKUP_S3`, `TIDEVEC_BACKUP_GCS`); the
  compose file no longer enables trace export to a Jaeger host that only exists with the
  `monitoring` profile; the embedded-wheel smoke test called a class that does not exist; protobuf
  stubs regenerated for protobuf >= 5.29.
- **Release:** every version string (server `/health`, CMake, PyPI packages, Java, C++ SDK,
  Docker labels) now says 0.3.0; Docker CPU images for amd64 and arm64 are published as one
  multi-arch manifest instead of two pushes to the same tag.

### Changed

- README and Docker Hub page rewritten to contain only claims backed by code, tests or the
  benchmarks; scale figures are labelled as projections, the GPU path as unsupported.
- `latest` / `cpu` Docker tags now track the latest release; builds from `main` are tagged `main`.
- Durability model figures: RS(10,4) 8.7 nines, RS(14,4) 8.1, RS(6,3) 7.5, 3-way replication 7.2
  (independent disks, 0.4% annual failure, no repair). A standalone server keeps all shards on one
  host.

### Known limitations

- DriftBridge swaps vectors one by one: queries during the swap can see a mix of old and new
  embeddings; standalone servers only; the dimension must stay the same.
- Cluster mode is experimental: every node keeps a full copy (`tidevec-router` multi-Raft is an
  experiment), DriftBridge returns `501`, `--ultra-durable` is off, restore is refused, and backups
  are per node.
- The GPU (CUDA) path fails recall on real hardware and is unsupported; the TPU path is not
  validated on hardware.
- The TypeScript packages are not on npm; the C++ SDK has no filter, retrieval or hybrid APIs.
- macOS x86_64 wheels are cross-compiled on arm64 and not tested in CI.
- No Windows support (POSIX storage layer); use WSL2 or Docker.
- `proto/tidevec.proto` has no server; REST is the only API.
- Native TLS needs a `-DTIDEVEC_TLS=ON` build (OpenSSL 3; the Docker images have it). Cluster mode
  talks plain HTTP between nodes, so the Helm chart refuses `tls.enabled` with `cluster.enabled`.
  The C++ SDK speaks plain HTTP only.
- During a restore, new requests wait (up to 60 s) until the collections are reloaded; requests
  already running finish against the old data, and their writes are discarded by the restore.

## [0.2.1]

Python SDK `tidevec` 0.2.1 on PyPI with updated project metadata.

## [0.2.0]

DriftBridge, AgentContext TTL, GPU (CUDA) and TPU Docker builds.

[Unreleased]: https://github.com/ashishodu2023/TideVec/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/ashishodu2023/TideVec/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/ashishodu2023/TideVec/releases/tag/v0.2.1
[0.2.0]: https://github.com/ashishodu2023/TideVec/releases/tag/v0.2.0
