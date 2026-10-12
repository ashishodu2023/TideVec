# Recency and half-life

Every vector has a `created_at` timestamp (milliseconds since the Unix epoch; defaults to the time of
the upsert). At query time TideVec turns its age into a recency weight and blends it with similarity:

\[
\text{recency} = 2^{-\text{age} / \text{half\_life}}
\qquad
\text{score} = (1 - \beta)\cdot\text{similarity} + \beta\cdot\text{recency}
\]

- `similarity` is the metric's score mapped to \([0, 1]\). For cosine it is \((1 + \cos)/2\), so a
  `vector_score` of 0.92 means a cosine of 0.84.
- `half_life_ms` is how long it takes the recency weight to halve: a vector one half-life old gets
  0.5, two half-lives 0.25. Future-dated vectors get 1.0.
- \(\beta\) is `temporal_blend`. 0 is pure similarity; the default is 0.3.

The blended score is used **inside the HNSW beam search** (layer 0), not only to reorder a final
candidate list. Each hit reports all three numbers: `score`, `vector_score` and `temporal_score`.

## Collection setting and per-query override

`half_life_ms`, `temporal_blend`, `staleness_threshold` and `temporal_cutoff` are collection
settings, set at creation and changeable later:

```python
from tidevec import HalfLife

db.create_collection("kb", dim=384, half_life_ms=HalfLife.ONE_YEAR, temporal_blend=0.3)
db.set_temporal("kb", half_life_ms=HalfLife.ONE_MONTH, temporal_blend=0.2)
```

A search can override the blend for one query. Leaving it out (`temporal_blend=None` in Python, `nil`
in Go, `null` in Java) uses the collection's setting:

```python
db.search("kb", q)                       # collection's blend
db.search("kb", q, temporal_blend=0.0)   # pure similarity for this query
```

!!! warning "Changed in 0.3.0"
    Before 0.3.0 the server used 0.3 for any search that did not send `temporal_blend`, and the
    Python client always sent 0.3, so a collection's own blend was ignored. See the
    [changelog](../changelog.md).

## Choosing a half-life

The half-life should match how fast your content goes out of date, not how often you write.
`HalfLife` has presets (`ONE_HOUR`, `ONE_DAY`, `ONE_WEEK`, `ONE_MONTH`, `ONE_YEAR`); any number of
milliseconds works.

| Content | Starting point |
|---|---|
| Agent session scratch memory | 1 hour |
| News, feeds, chat | 1 day |
| Support tickets, incidents | 1 month (the default is 30 days) |
| Policies, documentation, wikis | 1 year |

The [real-data benchmark](../benchmarks.md) used a 365-day half-life with blend 0.3 on Wikipedia
revisions. Those values have not been tuned across other datasets; measure on your own queries.

## What recency does not fix

Recency changes the *order*. It does not remove an old version that is still similar enough to the
query: in the benchmark, recency fixed the top hit but left about one outdated version in every top-5.
To remove those, link versions with [supersession](supersession.md). To cut old content off entirely,
set `temporal_cutoff` (see [Staleness signals](staleness.md)), but remember that a hard cut-off
returns nothing once the current version is itself older than the cut-off.

## Compared with reranking

A common alternative is to fetch the top N by similarity and rerank with the same formula. On the
benchmark this produced **identical rankings** to TideVec's in-search blend. The differences are
operational: no candidate-pool size to tune, one stage, and it composes with filters during
traversal. It is not more accurate on that data.
