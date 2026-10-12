# Staleness signals

TideVec gives a caller several ways to notice, demote or exclude old content. They are independent
and can be combined.

| Signal | Set where | Effect |
|---|---|---|
| `temporal_score` | always | the recency weight \(2^{-\text{age}/\text{half\_life}}\) of each hit |
| `staleness_warning` (age) | collection `staleness_threshold`, default 0.20 | hit flagged when its recency weight is below the threshold, with a reason like `"Vector is 400 days old"` |
| `staleness_warning` (superseded) | [supersession](supersession.md) | hit flagged with `"Superseded by <ids>"` and `superseded_by` filled |
| `superseded_penalty` | per query | superseded hits' scores multiplied by `1 - penalty`, then re-sorted |
| `temporal_cutoff` | collection, default 0 (off); REST `temporal.temporal_cutoff` only, not in the SDK helpers | vectors whose recency weight is below the cut-off are skipped during the search |
| `valid_until` | per vector (ms since epoch) | the vector is not returned by queries after this time |
| `ttl_seconds` | per vector | shorthand for `valid_until = created_at + ttl_seconds` |

`valid_from` is stored and returned but not enforced at query time today.

With the default threshold of 0.20, the age warning fires once a vector is about 2.3 half-lives old
(\(2^{-2.32} \approx 0.2\)): roughly 70 days with the default 30-day half-life.

Set `include_staleness_warnings=false` on a query to skip both kinds of warning (and the reason
strings).

## What to do with the warnings

A RAG pipeline can pass the flags to the model ("this source was superseded on ...") instead of
silently dropping the hit. The [LangChain integration](../integrations/langchain.md) copies them into
`Document.metadata["tidevec"]`, and [`AgentMemory`](../integrations/mcp.md) reports them as `stale`
and `superseded_by` on each recalled memory.

## Hard cut-offs

`temporal_cutoff` is a hard filter on age. It is simple and predictable while content is fresh, but it
fails abruptly: in the [benchmark](../benchmarks.md), a one-year cut-off was perfect while the current
versions were recent, and returned **no correct version at all** once they were 1.5 years old. Prefer
supersession plus a soft blend unless "never return anything older than X" is a real requirement.
