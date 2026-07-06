# tidevec-py

Python SDK for [TideVec](https://github.com/ashishodu2023/TideVec) — the world's first temporally-aware, causally-indexed vector database.

## Install

```bash
pip install tidevec-py
```

## Quick Start

```python
import os
from tidevec import TideVec, HalfLife

# Auth: pass api_key or set TIDEVEC_API_KEY env var
db = TideVec("localhost:6399", api_key=os.environ.get("TIDEVEC_API_KEY", ""))

db.create_collection(
    "docs",
    dim=768,
    half_life_ms=HalfLife.ONE_WEEK,
    temporal_blend=0.3,
)

db.upsert("docs", [
    {"id": "doc_001", "embedding": [...], "payload": {"source": "wiki"}},
])

results = db.search("docs", query_vector=[...], top_k=10)
for hit in results:
    print(f"{hit.id}  score={hit.score:.4f}  temporal={hit.temporal_score:.3f}")
```

## Authentication

Production TideVec requires an API key on all `/v1/*` routes:

```python
# Option 1: environment variable (recommended)
export TIDEVEC_API_KEY="your-secret-key"
db = TideVec("localhost:6399")  # reads env automatically

# Option 2: explicit
db = TideVec("localhost:6399", api_key="your-secret-key")

# Option 3: TLS
db = TideVec("localhost:6399", api_key="...", tls=True)
```

Errors:

```python
from tidevec import UnauthorizedError, ForbiddenError, RateLimitError

try:
    db.search("docs", query)
except UnauthorizedError:   # 401 — missing/invalid API key
    ...
except ForbiddenError:      # 403 — tenant or SSRF blocked
    ...
except RateLimitError:       # 429 — auto-retried up to 3 times
    ...
```

## DriftBridge — Model Migration

Upgrade embedding models without downtime:

```python
import time

db.start_drift(
    "docs",
    reembed_url="https://embed.example.com/v1/embed",
)

while True:
    status = db.drift_status("docs")
    print(f"{status.phase}: {status.pct_complete:.0f}%")
    if status.phase in ("COMPLETE", "IDLE", "FAILED"):
        break
    time.sleep(5)

# Abort if needed
db.abort_drift("docs")
```

## Backups

```python
snapshot = db.trigger_backup()   # "tidevec_1712345678.tar.gz"
backups  = db.list_backups()     # list all snapshots
```

## Observability

```python
# Prometheus metrics
print(db.metrics())

# Per-query trace (OTel-compatible)
results = db.search("docs", query, include_trace=True)
if results.trace:
    print(results.trace["strategy"], results.trace["latency_ms"])
```

## Async

```python
import asyncio
from tidevec import AsyncTideVec

async def main():
    async with AsyncTideVec("localhost:6399", api_key="...") as db:
        await db.upsert("docs", [{"id": "v1", "embedding": [...]}])
        results = await db.search("docs", query_vector=[...], top_k=5)

asyncio.run(main())
```

## API Reference

| Method | Description |
|--------|-------------|
| `health()` | Server health check (no auth) |
| `info()` | Server feature manifest |
| `metrics()` | Prometheus text metrics |
| `create_collection(name, dim, ...)` | Create collection |
| `list_collections()` | List all collections |
| `get_collection(name)` | Collection stats + backend type |
| `drop_collection(name)` | Delete collection |
| `upsert(collection, vectors)` | Insert/update vectors |
| `delete(collection, ids)` | Delete vectors by ID |
| `search(collection, query, ...)` | ANN search with temporal scoring |
| `add_edges(collection, edges)` | Add causal graph edges |
| `set_temporal(name, half_life_ms, ...)` | Update decay config |
| `start_drift(collection, reembed_url)` | Start model migration |
| `drift_status(collection)` | Poll migration progress |
| `abort_drift(collection)` | Cancel migration |
| `trigger_backup()` | Manual snapshot |
| `list_backups()` | List snapshots |
| `list_backup_manifests()` | PITR manifest history |
| `restore_backup(snapshot)` | Point-in-time restore |

## HalfLife presets

```python
from tidevec import HalfLife

HalfLife.ONE_HOUR   # agent session memory
HalfLife.ONE_DAY    # news / feeds
HalfLife.ONE_WEEK   # support tickets
HalfLife.ONE_MONTH  # documents (default)
HalfLife.ONE_YEAR   # long-term knowledge base
```
