# Supersession

Supersession records, at write time, that one vector is a newer version of another. Queries then know
which results are outdated without guessing from timestamps.

## How linking works

With supersession enabled, every upsert runs a pure-similarity search (no recency blend) for the
incoming vector's nearest neighbours (`candidates`, default 5). A neighbour is treated as another
version of the same document when:

1. its `vector_score` is at least `similarity_threshold` (default 0.92, which for cosine means a
   cosine of 0.84), **and**
2. if `match_field` is set, both vectors have the same non-empty value in that payload field
   (for example `doc_id`, `url` or `policy_key`). If the incoming vector has no value for the field,
   nothing is linked.

The vector with the later `created_at` becomes the newer version. TideVec stores an `UPDATES` edge
from newer to older and records `superseded_by` on the older one. This works across shards and in
embedded mode.

```python
db.create_collection(
    "kb", dim=384,
    supersession=True,
    supersede_match_field="doc_id",   # optional, strongly recommended when you have a key
    supersede_threshold=0.5,          # with a key, a low threshold is safe
)
```

Over REST the same settings are the `supersession` object of
[`POST /v1/collections`](../api/rest.md#post-v1-collections): `enabled`, `similarity_threshold`,
`match_field`, `candidates`.

## At query time

Every hit that has a newer version gets:

- `superseded_by`: the ids of the newer versions (`hit.is_superseded` in Python);
- `staleness_warning = true` and `staleness_reason = "Superseded by <ids>"` (unless the query sets
  `include_staleness_warnings=false`).

A query can also demote superseded hits with `superseded_penalty` in \([0, 1]\): the hit's score is
multiplied by `1 - penalty` and results are re-sorted. `1.0` pushes every superseded version to the
bottom.

```python
for hit in db.search("kb", q, top_k=5, superseded_penalty=1.0):
    print(hit.id, hit.superseded_by)
```

To drop superseded versions entirely, filter them out on the client
(`[h for h in hits if not h.is_superseded]`), or use `exclude_superseded=True` in the
[LangChain integration](../integrations/langchain.md).

## Choosing a threshold

**With a document key** (`match_field`), the threshold only has to separate "a revision of this
document" from "a different chunk of this document", so values around 0.5 work. This is the
recommended setup.

**Without a key**, the threshold alone decides. The [benchmark](../benchmarks.md) measured this on
Wikipedia revisions (bge-small-en-v1.5):

| `similarity_threshold` (`vector_score`) | Raw cosine | Link precision | Stale documents flagged |
|---:|---:|---:|---:|
| 0.80 | 0.60 | 41.3% | 100% |
| 0.85 | 0.70 | 51.5% | 100% |
| 0.90 | 0.80 | 79.2% | 100% |
| 0.95 | 0.90 | 100% | 95.8% |

Lower thresholds link unrelated articles; on that data 0.95 was the first value with no wrong
links. The right value depends on your embedding model and chunking, so check a sample of links before
relying on it.

## Limits

- Linking only happens on upsert, and supersession is configured when the collection is created
  (there is no endpoint to switch it on later). Existing vectors are not linked retroactively.
- The comparison only sees the `candidates` nearest neighbours of the new vector.
- Supersession says "this is older than that". It does not judge whether two *different* sources
  disagree: that is a [contradiction](edges.md#contradictions).
