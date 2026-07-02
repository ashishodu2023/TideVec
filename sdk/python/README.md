# tidevec-py

Python SDK for [TideVec](https://github.com/ashishodu2023/TideVec) — the world's first temporally-aware, causally-indexed vector database.

## Install

```bash
pip install tidevec-py
```

## Quick Start

```python
from tidevec import TideVec, HalfLife

# Connect
db = TideVec("localhost:6399")

# Create a collection
db.create_collection(
    "docs",
    dim=768,
    half_life_ms=HalfLife.ONE_WEEK,   # fresh vectors rank higher
    temporal_blend=0.3,               # 30% time, 70% semantic
)

# Upsert vectors
db.upsert("docs", [
    {
        "id":        "doc_001",
        "embedding": [0.1, 0.2, ...],  # 768-dim
        "payload":   {"source": "confluence", "team": "platform"},
        "ttl_seconds": 86400,          # expire in 24 hours
    },
    {
        "id":        "doc_002",
        "embedding": [0.3, 0.4, ...],
        "payload":   {"source": "jira"},
        "edges":     [{"target_id": "doc_001", "type": "UPDATES", "weight": 0.9}],
    },
])

# Search with temporal scoring
results = db.search(
    "docs",
    query_vector=[0.1, 0.2, ...],
    top_k=10,
    temporal_blend=0.3,
    include_staleness_warnings=True,
)

for hit in results:
    print(f"{hit.id}  score={hit.score:.4f}  temporal={hit.temporal_score:.3f}")
    if hit.staleness_warning:
        print(f"  ⚠️  {hit.staleness_reason}")

# Causal expansion
results = db.search(
    "docs",
    query_vector=[0.1, 0.2, ...],
    mode="causal_expand",
    causal_hops=2,
)

# Contradiction detection
results = db.search(
    "docs",
    query_vector=[0.1, 0.2, ...],
    mode="contradiction_check",
)
for hit in results:
    if hit.contradicted_by:
        print(f"{hit.id} is contradicted by {hit.contradicted_by}")

# Batch search (GPU/TPU accelerated)
responses = db.batch_search(
    "docs",
    query_vectors=[[0.1, ...], [0.5, ...], [0.9, ...]],
    top_k=5,
    device="gpu",   # "auto" | "gpu" | "tpu" | "cpu"
)

# With trace for observability (CortexOps integration)
results = db.search("docs", query_vector=[...], include_trace=True)
print(f"latency={results.latency_ms:.1f}ms  strategy={results.strategy}")
```

## Async

```python
import asyncio
from tidevec import AsyncTideVec

async def main():
    async with AsyncTideVec("localhost:6399") as db:
        await db.upsert("docs", [{"id": "v1", "embedding": [...]}])
        results = await db.search("docs", query_vector=[...], top_k=5)
        for hit in results:
            print(hit.id, hit.score)

asyncio.run(main())
```

## Context manager

```python
with TideVec("localhost:6399", api_key="mykey") as db:
    db.create_collection("agents", dim=1536, half_life_ms=HalfLife.ONE_HOUR)
    # ... operations
# connection auto-closed
```

## HalfLife presets

```python
from tidevec import HalfLife

HalfLife.AGENT_SESSION  # 1 hour   — per-session agent memory
HalfLife.ONE_DAY        # 1 day    — news / feeds
HalfLife.ONE_WEEK       # 7 days   — support tickets
HalfLife.ONE_MONTH      # 30 days  — documents (default)
HalfLife.ONE_YEAR       # 365 days — long-term knowledge base
```
