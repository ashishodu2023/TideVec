# TideVec

**A vector database for RAG and AI agents over knowledge that changes.**

Time-aware ranking inside the HNSW search, automatic detection of superseded document versions, and
typed links (`UPDATES`, `CONTRADICTS`, `CAUSES`, ...) stored next to the vectors.

| | |
|---|---|
| **Docker** | [hub.docker.com/r/averm004/tidevec](https://hub.docker.com/r/averm004/tidevec) |
| **GitHub** | [github.com/ashishodu2023/TideVec](https://github.com/ashishodu2023/TideVec) |
| **Website** | [tidevec.com](https://tidevec.com) |
| **License** | [Apache 2.0](https://github.com/ashishodu2023/TideVec/blob/main/LICENSE) |

---

## Quick start (CPU)

Multi-arch: **linux/amd64** (x86-64-v3: Haswell or newer, AVX2 + FMA) and **linux/arm64** (Apple
Silicon via Docker Desktop).

```bash
export TIDEVEC_API_KEY="your-secret-key"
docker run -d -p 6399:6399 -e TIDEVEC_API_KEY \
  -v tidevec_data:/data -v tidevec_backups:/backups averm004/tidevec:latest
```

Auth is **on** by default. If you don't set `TIDEVEC_API_KEY`, the container generates one and prints
it in the logs (`docker logs <container>`). Send it as `X-Api-Key` on `/v1/*` requests.

```bash
curl -s http://localhost:6399/health
curl -s -H "X-Api-Key: $TIDEVEC_API_KEY" http://localhost:6399/v1/collections
```

```python
from tidevec import TideVec, HalfLife      # pip install tidevec

db = TideVec("localhost:6399")              # reads TIDEVEC_API_KEY
db.create_collection("docs", dim=768, half_life_ms=HalfLife.ONE_MONTH,
                     supersession=True, supersede_match_field="doc_id")
db.upsert("docs", [{"id": "refunds-v2", "embedding": vec, "payload": {"doc_id": "refunds"}}])

for hit in db.search("docs", query, top_k=5, superseded_penalty=1.0):
    print(hit.id, hit.score, hit.superseded_by)
```

No server needed for local use: `pip install "tidevec[embedded]"` and `tidevec.Local("./data")`.

---

## Configuration (all images)

| Variable | Default | Description |
|---|---|---|
| `TIDEVEC_API_KEY` | generated | API key |
| `TIDEVEC_REQUIRE_AUTH` | `1` | `0` disables auth, even if `TIDEVEC_API_KEY` is set (local testing only) |
| `TIDEVEC_ULTRA_DURABLE` | `1` | RS(10,4) segments, WAL group commit, health monitor, in-process Raft log |
| `TIDEVEC_SEGMENT_STORE` | `1` | disk-backed vector segments |
| `TIDEVEC_BACKUP` | `0` | `1` takes a snapshot of `/data` periodically (newest 48 kept) |
| `TIDEVEC_BACKUP_INTERVAL_HOURS` | `6` | hours between snapshots |
| `TIDEVEC_BACKUP_DIR` | `/backups` | snapshot directory; keep it outside `/data` |
| `TIDEVEC_BACKUP_S3` / `TIDEVEC_BACKUP_GCS` | unset | also upload snapshots to `s3://...` / `gs://...`; needs the `aws` / `gsutil` CLI, which the image does not include |
| `TIDEVEC_MULTI_TENANT` | `0` | `1` accepts per-tenant keys from `/data/tenants.json`; each tenant only sees its own collections |
| `TIDEVEC_RATE_LIMIT` | `100` | requests per second per client |
| `TIDEVEC_OTEL_ENDPOINT` | unset | enables JSON event logs and OTLP trace export |
| `TIDEVEC_REEMBED_ALLOW` | unset | hosts DriftBridge may call to re-embed (`a.com,b.com`) |
| `TIDEVEC_REEMBED_ALLOW_PRIVATE` | `0` | `1` lets listed hosts be private/loopback addresses |
| `TIDEVEC_LOG_LEVEL` | `info` | `debug`/`info` log every request; `warn`/`error` do not (`TIDEVEC_QUIET=1` is the same) |
| `TIDEVEC_TLS_CERT` / `TIDEVEC_TLS_KEY` | unset | PEM certificate and key: serve HTTPS on 6399 (set both) |
| `TIDEVEC_THREADS` / `TIDEVEC_DEVICE` / `TIDEVEC_PORT` / `TIDEVEC_DATA_DIR` | `8` / `cpu` / `6399` / `/data` | |

Only port **6399** (REST) is served: plain HTTP, or HTTPS with `TIDEVEC_TLS_CERT` and
`TIDEVEC_TLS_KEY` pointing at mounted PEM files (`-v ./tls:/tls:ro -e TIDEVEC_TLS_CERT=/tls/tls.crt
-e TIDEVEC_TLS_KEY=/tls/tls.key`). Extra arguments after the image name are passed to the server.

**Rotating the API key:** change `TIDEVEC_API_KEY` and restart; the old key stops working.

### Durability, honestly

Every write goes to a CRC-checked write-ahead log under `/data` and is replayed on restart. The
RS(10,4) erasure coding models at 8.7 nines **across independent disks** (0.4% annual failure, no
repair), but a single container keeps every shard on **one volume**, so that volume is your failure
domain. Enable backups (`TIDEVEC_BACKUP=1`), copy `/backups` off the host, and keep `/data` on
durable storage. To restore, stop the container and extract a snapshot into the data volume
(`tar -xzf tidevec_<ts>.tar.gz -C <volume> --strip-components=1`); the restore API cannot replace a
mounted `/data`.

### Experimental: replicated cluster

Three or more containers can replicate writes with Raft over HTTP:

```bash
-e TIDEVEC_NODE_ID=0 \
-e TIDEVEC_RAFT_PEERS="0=http://node0:6399,1=http://node1:6399,2=http://node2:6399" \
-e TIDEVEC_CLUSTER_SECRET=...
```

The leader accepts writes; followers answer `421` with the leader's address (the SDKs follow it), and
serve possibly slightly stale reads unless a request asks for `consistency=linearizable` or `lease`.
Log snapshots, adding/removing nodes and pre-vote are implemented. Every node holds a full copy and
DriftBridge is disabled. Experimental, not for production.

---

## Tags

| Tag | Arch | Description |
|---|---|---|
| `latest`, `cpu` | amd64, arm64 | newest stable release, CPU server (recommended) |
| `0.3.0`, `0.3.0-cpu`, `0.3`, `0.3-cpu` | amd64, arm64 | pinned release / latest patch of a minor release |
| `gpu`, `0.3.0-gpu`, `0.3-gpu` | amd64 | CUDA 12 runtime, `TIDEVEC_DEVICE=gpu`; same entrypoint and variables as the CPU image (unsupported, see below) |
| `tpu`, `0.3.0-tpu`, `0.3-tpu` | amd64 | JAX/XLA base, `TIDEVEC_DEVICE=tpu`; same entrypoint and variables as the CPU image |
| `main`, `sha-<commit>` (`-gpu`/`-tpu` suffixes for the accelerator images) | | builds of the main branch, not releases |

Pre-releases (`0.4.0-rc1`) get their own version tags only and never move `latest`, `cpu` or `0.x`.

The GPU path is an independent CUDA implementation of the CAGRA search design (Ootomo et al.,
ICDE 2024), not NVIDIA cuVS. On real GPU hardware it fails recall (graph search recall@10 0.003), so it
is unsupported; use the CPU image. On Apple Silicon use `latest`/`cpu`. The GPU/TPU tags are
amd64-only (`no matching manifest for linux/arm64` means you pulled one of them).

---

## What's inside

- **Recency-aware HNSW:** `final = (1 − β)·similarity + β·2^(−age/half_life)` used during the search;
  `β` per collection, overridable per query.
- **Supersession:** new versions are linked to the ones they replace at write time; results carry
  `superseded_by` and superseded versions can be demoted.
- **Typed edges:** contradiction checks and causal expansion in the same query.
- **Delete by id or filter**, payload equality filters applied during traversal.
- **DriftBridge:** re-embed a live collection with a new model in the background.
- **Integrations:** LangChain, LlamaIndex, and an MCP memory server (`tidevec-mcp`) for Claude Desktop
  and Cursor.

On 270 real Wikipedia revisions, plain cosine returned an outdated version for 55–85% of queries;
TideVec with supersession returned the current version and kept outdated versions out of the top 5.
Recency weighting alone matched a recency rerank, no better. Details:
[benchmarks/temporal_real_data](https://github.com/ashishodu2023/TideVec/tree/main/benchmarks/temporal_real_data).

---

## Links

- [GitHub](https://github.com/ashishodu2023/TideVec) · [Examples](https://github.com/ashishodu2023/TideVec/tree/main/examples) · [Roadmap](https://github.com/ashishodu2023/TideVec/blob/main/ROADMAP.md)
- [tidevec.com](https://tidevec.com) · [contact@tidevec.com](mailto:contact@tidevec.com)
