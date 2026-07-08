# TideVec

**The world's first temporally-aware causal vector database.**

| | |
|---|---|
| **Docker** | [hub.docker.com/r/averm004/tidevec](https://hub.docker.com/r/averm004/tidevec) |
| **GitHub** | [github.com/ashishodu2023/TideVec](https://github.com/ashishodu2023/TideVec) |
| **Website** | [gettidevec.com](https://gettidevec.com) |
| **License** | [Apache 2.0](https://github.com/ashishodu2023/TideVec/blob/main/LICENSE) |

---

## Quick Start

### CPU — any machine (Linux, Mac, Windows/WSL)

Multi-arch: **linux/amd64** + **linux/arm64** (Apple Silicon).

```bash
docker pull averm004/tidevec:latest
docker run -d \
  -p 6399:6399 \
  -v "$(pwd)/data:/data" \
  averm004/tidevec:latest
```

### GPU — NVIDIA (CUDA runtime, amd64 only)

Requires [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
docker pull averm004/tidevec:gpu
docker run -d --gpus all \
  -p 6399:6399 \
  -v "$(pwd)/data:/data" \
  -e TIDEVEC_DEVICE=gpu \
  averm004/tidevec:gpu
```

Supports A100, H100, RTX 30/40 series. Image includes CUDA 12.6 runtime; server starts with `--device gpu`.

### TPU — Google Cloud TPU (amd64 only)

For GKE TPU nodes (v4/v5e). Native XLA matmul when built with libxla_client; JAX runtime included.

```bash
docker pull averm004/tidevec:tpu
docker run -d \
  -p 6399:6399 \
  -v "$(pwd)/data:/data" \
  -e TIDEVEC_DEVICE=tpu \
  averm004/tidevec:tpu
```

### Full observability stack

```bash
curl -O https://raw.githubusercontent.com/ashishodu2023/TideVec/main/docker/docker-compose.yml
docker compose up -d
# Optional GPU profile:
docker compose --profile gpu up -d
```

---

## Available Tags

| Tag | Image | Arch | Description |
|-----|-------|------|-------------|
| `latest`, `cpu` | `averm004/tidevec:latest` | amd64, arm64 | CPU-only server (default) |
| `gpu` | `averm004/tidevec:gpu` | amd64 | CUDA 12.6 runtime, `--device gpu` |
| `tpu` | `averm004/tidevec:tpu` | amd64 | JAX/XLA base for Cloud TPU |
| `0.2.0-cpu` | versioned CPU | amd64, arm64 | Pin to release |
| `0.2.0-gpu` | versioned GPU | amd64 | Pin to release |
| `0.2.0-tpu` | versioned TPU | amd64 | Pin to release |
| `sha-<commit>-cpu` | CI snapshot | amd64, arm64 | Exact commit (debugging) |
| `sha-<commit>-gpu` | CI snapshot | amd64 only | Exact GPU commit (Linux NVIDIA hosts) |

> **Note:** `gpu` and `tpu` are **amd64-only** and larger than CPU (~1.4 GB GPU, ~225 MB TPU). They are rebuilt on every `main` push and on version tags (`v*`).

---

## Apple Silicon (Mac M1/M2/M3/M4) and ARM64

**Use the CPU image on Mac** — it is multi-arch (`linux/arm64` + `linux/amd64`):

```bash
docker pull averm004/tidevec:latest
# or
docker pull averm004/tidevec:cpu
```

**Do not pull `:gpu`, `:tpu`, or `sha-*-gpu` on Apple Silicon** unless you force amd64 emulation. Those tags are built for **Linux amd64** servers with NVIDIA GPUs or Google Cloud TPU. macOS has no CUDA runtime, so the GPU image cannot accelerate workloads locally anyway.

If you see:

```text
no matching manifest for linux/arm64/v8 in the manifest list entries
```

you pulled an **amd64-only** tag on an ARM Mac. Switch to `:latest` or `:cpu`.

| Your machine | Recommended tag |
|--------------|-----------------|
| Mac (Apple Silicon) | `latest`, `cpu` |
| Mac (Intel) | `latest`, `cpu` |
| Linux + NVIDIA GPU | `gpu` |
| GKE / Cloud TPU node | `tpu` |

**Optional** — pull a GPU image under amd64 emulation (slow, no real GPU; for inspect/export only):

```bash
docker pull --platform linux/amd64 averm004/tidevec:gpu
```

---

## Verify

```bash
curl -s http://localhost:6399/health | jq .
curl -s http://localhost:6399/v1/info   | jq .
```

---

## Python SDK

```python
from tidevec import TideVec, HalfLife

db = TideVec("http://localhost:6399")
db.create_collection("docs", dim=768, half_life_ms=HalfLife.ONE_WEEK)
db.upsert("docs", [{"id": "v1", "embedding": [...], "payload": {"src": "wiki"}}])

for hit in db.search("docs", query, top_k=10, temporal_blend=0.3):
    print(hit.id, hit.score, hit.temporal_score)
```

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `TIDEVEC_PORT` | `6399` | REST API port |
| `TIDEVEC_DATA_DIR` | `/data` | Persistent data directory |
| `TIDEVEC_THREADS` | `8` | Worker threads |
| `TIDEVEC_DEVICE` | `auto` | `auto` \| `cpu` \| `gpu` \| `tpu` |
| `TIDEVEC_API_KEY` | *(empty)* | API key; set to enable auth |
| `TIDEVEC_REQUIRE_AUTH` | `false` | Require `X-Api-Key` on all endpoints |

Production flags (also via CLI): `--ultra-durable`, `--segment-store`, `--backup`, `--multi-tenant`, `--otel`.

---

## What Makes TideVec Different

- **TVIndex** — HNSW with Ebbinghaus temporal decay. Fresh vectors rank higher automatically.
- **CausalEdge** — Native typed-edge graph (CAUSES, CONTRADICTS, UPDATES). No separate Neo4j.
- **GPU CAGRA** — CUDA warp-level beam search (33–77× faster than CPU HNSW at ~95% recall).
- **TPU XLA** — Exact batch matmul path for highest throughput on Google TPU pods.
- **RS(10,4)** — Reed–Solomon erasure coding for 11-nines durability.
- **Raft** — Multi-node consensus with sub-150 ms failover.
- **RetrievalTrace** — OTel-compatible per-query trace with staleness warnings.

---

## Links

- 🐳 [Docker Hub — averm004/tidevec](https://hub.docker.com/r/averm004/tidevec)
- 🌐 [gettidevec.com](https://gettidevec.com)
- 📚 [Documentation](https://gettidevec.com/docs)
- ⭐ [GitHub — TideVec](https://github.com/ashishodu2023/TideVec)
- 💬 [contact@tidevec.com](mailto:contact@tidevec.com)
