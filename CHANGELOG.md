# Changelog

All notable changes to TideVec are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/) (pre-1.0: minor versions may break APIs).

Versions apply to the server, the Docker images, `tidevec` and `tidevec-embedded` on PyPI, the
Go module (`sdk/go/vX.Y.Z` tag), the Java SDK and the C++ SDK.

## [Unreleased]

Work in progress on the roadmap
([ROADMAP.md](https://github.com/ashishodu2023/TideVec/blob/main/ROADMAP.md), Phases 1-4). Entries move into a
release section when they are merged and tested.

<!-- TODO(release): before tagging, decide whether in-progress Phase 1-4 work ships in 0.3.0
     or in 0.4.0, and move the finished items into the matching section. -->

## [0.3.0] - unreleased

Focus of this release: retrieval over knowledge that changes (supersession), correctness of the
HNSW index, embedded Python mode, framework integrations, and documentation that only claims what
is tested.

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
- **Delete by filter:** `POST /v1/collections/{name}/delete` accepts `{"filter": "key=value"}`
  in addition to `ids`, and returns `deleted`. Python: `delete(collection, ids=None, *, filter=None)`.
- **Embedded mode:** `tidevec-embedded` (pybind11 wheels for Linux x86_64/aarch64 and macOS
  arm64, CPython 3.9-3.13) and `tidevec.Local(path)` with the same methods as the REST client.
  Install with `pip install "tidevec[embedded]"`.
- **Python integrations:** LangChain `TideVecVectorStore`, LlamaIndex `TideVecVectorStore`,
  `tidevec.memory.AgentMemory` (`remember` / `recall` / `history` / `forget`, keyed facts
  supersede older ones) and the `tidevec-mcp` MCP server. Optional LLM-judged contradiction
  detection (`tidevec.contradictions.detect_contradictions`) that records `CONTRADICTS` /
  `SUPPORTS` edges. Offline `HashingEmbedder`, plus OpenAI and sentence-transformers embedders.
- **Experimental replicated cluster mode:** Raft over HTTP between `tidevec-server` processes
  (`--node-id`, `--raft-peers`, `TIDEVEC_CLUSTER_SECRET`), persisted term/vote/log, followers
  answer writes with `421` and the leader URL (clients follow it), `GET /v1/cluster/status`.
  No log snapshots, membership changes or linearizable reads yet.
- **DriftBridge:** `--reembed-allow-private` to allow a local or in-VPC re-embedding service
  (cloud metadata endpoints stay blocked); dimension-changing migrations are rejected up front.
- **Server / Docker / Helm:** `--backup-interval-hours` (`TIDEVEC_BACKUP_INTERVAL_HOURS`,
  `backup.intervalHours`); `TIDEVEC_LOG_LEVEL` (`debug`/`info` log each request, `warn`/`error`
  do not; `config.logLevel`) and `TIDEVEC_QUIET=1`; native TLS in the images
  (`TIDEVEC_TLS_CERT` / `TIDEVEC_TLS_KEY`) and the chart (`tls.enabled`, `tls.secretName`: probes,
  ServiceMonitor and `helm test` switch to HTTPS); arguments given to the container are passed to
  the server after the generated ones.
- **Search:** `ef_search` per query; HNSW neighbour-selection heuristic; tombstoned nodes are
  compacted by a rebuild once they equal the live count.
- **Health monitor** lowers the durability estimate when shards are lost.
- **Examples** (`examples/01`-`05` and a Colab notebook) and the real-data temporal benchmark
  (`benchmarks/temporal_real_data/`: 270 Wikipedia revisions of 55 articles).
- **Go and Java SDKs** rewritten without gRPC dependencies, with live-server tests; Python client
  `count`, `has_collection`, `cluster_status`.
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
  failed `listen()` was silent.
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
  benchmark; scale figures are labelled as projections, the GPU path as unvalidated.
- `latest` / `cpu` Docker tags now track the latest release; builds from `main` are tagged `main`.
- Durability model figures: RS(10,4) 8.7 nines, RS(14,4) 8.1, RS(6,3) 7.5, 3-way replication 7.2
  (independent disks, 0.4% annual failure, no repair). A standalone server keeps all shards on one
  host.

### Known limitations

- DriftBridge swaps vectors one by one: queries during the swap can see a mix of old and new
  embeddings; standalone servers only; the dimension must stay the same.
- Cluster mode keeps a full copy on every node and has no snapshots or membership changes.
- GPU (CUDA) and TPU paths are not validated on hardware.
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
