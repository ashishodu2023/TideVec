# Quickstart

TideVec runs two ways with the same API: **embedded** inside a Python process (no server), or as a
**REST server**. Start embedded if you are prototyping, and switch to the server when several processes
or languages need the same data.

## Embedded (no server)

```bash
pip install "tidevec[embedded]"   # Linux x86_64/arm64, macOS arm64; CPython 3.9-3.13
```

The `[embedded]` extra pulls in `tidevec-embedded`, a compiled wheel that contains the C++ engine.
On other platforms pip falls back to building it from the source distribution, which needs a C++20
compiler (scikit-build-core fetches CMake if it is missing).

```python
import time
from tidevec import HalfLife, Local
from tidevec.embeddings import HashingEmbedder   # toy embedder; use a real model in practice

DAY = 86_400_000
now = int(time.time() * 1000)
emb = HashingEmbedder(256)

db = Local("./tidevec_data")
db.create_collection(
    "policies", dim=emb.dim,
    half_life_ms=HalfLife.ONE_MONTH, temporal_blend=0.3,
    # link a new upsert to older vectors with the same doc_id that it resembles
    supersession=True, supersede_threshold=0.5, supersede_match_field="doc_id",
)

for vid, text, ts in [
    ("refunds-2025", "Refund policy: refunds are issued within 14 days.", now - 200 * DAY),
    ("refunds-2026", "Refund policy update: refunds are now issued within 3 days.", now - 2 * DAY),
]:
    db.upsert("policies", [{"id": vid, "embedding": emb.embed_query(text), "created_at": ts,
                            "payload": {"text": text, "doc_id": "refunds"}}])

query = emb.embed_query("How many days until refunds are issued?")
for hit in db.search("policies", query, top_k=2, superseded_penalty=1.0):
    print(hit.id, round(hit.score, 3), hit.superseded_by)
db.close()
```

`refunds-2026` comes first, and `refunds-2025` reports `superseded_by == ["refunds-2026"]`.
A longer runnable version that prints pure similarity next to the blended ranking is
[`examples/01_quickstart_embedded.py`](https://github.com/ashishodu2023/TideVec/blob/main/examples/01_quickstart_embedded.py).

!!! note "One process per data directory"
    Nothing stops two processes from opening the same directory, and they would overwrite each
    other's files. Open a path from one process only, and never from `Local` and a server at once.

## Server

```bash
export TIDEVEC_API_KEY="your-secret-key"
docker run -p 6399:6399 -e TIDEVEC_API_KEY -v "$(pwd)/data:/data" averm004/tidevec:latest
```

If you do not set `TIDEVEC_API_KEY`, the container generates a random key and prints it on startup.
The amd64 image needs an x86-64-v3 CPU (Haswell or newer, with AVX2 and FMA); an arm64 image is
published as well.

```python
from tidevec import TideVec

db = TideVec("localhost:6399")   # reads TIDEVEC_API_KEY from the environment
db.create_collection("policies", dim=384, supersession=True, supersede_match_field="doc_id")
print(db.list_collections())
```

`TideVec` (server client) and `Local` (embedded) expose the same methods, so code moves between them by
changing one line. The same calls over plain HTTP:

```bash
curl -s localhost:6399/health
curl -s -X POST localhost:6399/v1/collections -H "X-Api-Key: $TIDEVEC_API_KEY" \
     -H 'Content-Type: application/json' -d '{"name": "notes", "dim": 3}'
curl -s -X POST localhost:6399/v1/collections/notes/upsert -H "X-Api-Key: $TIDEVEC_API_KEY" \
     -H 'Content-Type: application/json' \
     -d '{"vectors": [{"id": "a", "embedding": [0.1, 0.2, 0.3], "payload": {"text": "hello"}}]}'
curl -s -X POST localhost:6399/v1/collections/notes/search -H "X-Api-Key: $TIDEVEC_API_KEY" \
     -H 'Content-Type: application/json' -d '{"vector": [0.1, 0.2, 0.3], "top_k": 1}'
```

See the [REST API reference](api/rest.md) for every endpoint.

### Building the server from source

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j"$(nproc)" --target tidevec-server
./build-release/tidevec-server --port 6399 --data-dir ./data --api-key "$TIDEVEC_API_KEY"
```

By default the build uses `-march=native`; pass `-DTIDEVEC_MARCH=x86-64-v3` (or an empty value) for a
binary that runs on other machines.

## Next steps

- [Recency and half-life](concepts/recency.md): pick a half-life for your data.
- [Supersession](concepts/supersession.md): choose between a document key and a similarity threshold.
- [LangChain](integrations/langchain.md) / [LlamaIndex](integrations/llamaindex.md) /
  [Agent memory and MCP](integrations/mcp.md).
