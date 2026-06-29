# TideVec

**The world's first temporally-aware causal vector database.**

[![GitHub](https://img.shields.io/badge/GitHub-ashishodu2023%2FCortexdb-blue?logo=github)](https://github.com/ashishodu2023/TideVec)
[![Website](https://img.shields.io/badge/Website-tidevec.com-indigo)](https://tidevec.com)
[![License](https://img.shields.io/badge/License-Apache_2.0-green)](https://github.com/ashishodu2023/TideVec/blob/main/LICENSE)

## Quick Start

```bash
# CPU (any machine — Linux, Mac, Windows via WSL)
docker run -d -p 6399:6399 -v $(pwd)/data:/data ashishodu2023/TideVec:latest

# GPU (NVIDIA, requires nvidia-container-toolkit)
docker run -d --gpus all -p 6399:6399 ashishodu2023/TideVec:gpu

# Full stack (server + Prometheus + Grafana + Jaeger)
curl -O https://raw.githubusercontent.com/ashishodu2023/TideVec/main/docker/docker-compose.yml
docker compose up -d
```

## Available Tags

| Tag | Description | Arch |
|---|---|---|
| `latest`, `cpu` | CPU only, multi-arch | amd64, arm64 |
| `gpu` | CUDA 12.3 — A100/H100/RTX | amd64 |
| `tpu` | XLA/JAX — Google Cloud TPU | amd64 |
| `1.0.0-cpu` | Versioned CPU | amd64, arm64 |
| `1.0.0-gpu` | Versioned GPU | amd64 |

## What Makes TideVec Different

- **TVIndex** — HNSW with Ebbinghaus temporal decay scoring. Fresh vectors rank higher automatically.
- **CausalEdge** — Native typed-edge graph (CAUSES, CONTRADICTS, UPDATES). No separate Neo4j.
- **GPU CAGRA** — 33–77× faster than HNSW at 95% recall. CUDA warp-level beam search.
- **RS(10,4)** — Reed-Solomon erasure coding. 11-nines durability at 1.4× storage overhead.
- **Raft consensus** — 5-node, <150ms failover, 9-nines availability.
- **RetrievalTrace** — OTel-compatible per-query trace with staleness warnings.

## Connect

```python
from tidevec import TideVec, HalfLife

db = TideVec("localhost:6399")
db.create_collection("docs", dim=768, half_life_ms=HalfLife.ONE_WEEK)
db.upsert("docs", [{"id":"v1","embedding":[...],"payload":{"src":"wiki"}}])

for hit in db.search("docs", query, top_k=10, temporal_blend=0.3):
    print(hit.id, hit.score, hit.temporal_score)
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `TIDEVEC_PORT` | `6399` | REST API port |
| `TIDEVEC_DATA_DIR` | `/data` | Data directory |
| `TIDEVEC_THREADS` | `8` | Worker threads |
| `TIDEVEC_DEVICE` | `auto` | `auto` \| `cpu` \| `gpu` \| `tpu` |
| `TIDEVEC_API_KEY` | `` | Auth key (empty = no auth) |

## Links

- 🌐 [tidevec.com](https://tidevec.com)
- 📚 [Docs](https://tidevec.com/docs)
- ⭐ [GitHub](https://github.com/ashishodu2023/TideVec)
- 💬 [contact@tidevec.com](mailto:contact@tidevec.com)
