# Benchmarks

Every number on this page comes from a script in
[`benchmarks/`](https://github.com/ashishodu2023/TideVec/tree/main/benchmarks) and its committed
results file. If a figure is not here, it has not been measured.

## Temporal retrieval on real revision history

[`benchmarks/temporal_real_data`](https://github.com/ashishodu2023/TideVec/tree/main/benchmarks/temporal_real_data)
asks whether vector search returns the **current** version of a document when older versions are also
indexed. The corpus is the lead section of 55 Wikipedia articles as of 2018, 2020, 2022, 2024 and
today: 270 documents with 2-5 versions per article, timestamped with the revision date, embedded with
`BAAI/bge-small-en-v1.5` (via fastembed). k = 5, half-life 365 days, blend 0.3.

```bash
cd benchmarks/temporal_real_data
pip install "tidevec[embedded]" fastembed numpy
python fetch_corpus.py      # about 6 minutes, MediaWiki API
python run_benchmark.py
```

**Current@1** is the share of queries whose top hit is the current version; **stale in top-5** is
the average number of outdated versions among the 5 results a RAG prompt would receive.

| Strategy | Current@1 (today) | Current@1 (as of mid-2025) | Stale in top-5 (mid-2025) |
|---|---:|---:|---:|
| Cosine only | 18.2% | 27.3% | 2.89 |
| Hard filter, last 1 year | 100% | **0%** | 0.00 |
| Cosine top-50, then recency rerank | 100% | 100% | 0.91 |
| TideVec recency blend | 100% | 100% | 0.91 |
| TideVec + supersession | 100% | 100% | **0.00** |

("Today" and "mid-2025" use the neutral "What is {article}?" queries. In the mid-2025 scenario the
current version of every article is about 1.5 years old.)

What this shows:

- **Plain similarity returns stale content most of the time:** an outdated version was the top hit for
  55-85% of queries across the query sets and scenarios.
- **TideVec's in-search blend ranked exactly like a rerank with the same formula.** Its advantage is
  operational (one stage, no candidate-pool size, recency applied to filtered searches), not accuracy.
- **Hard time filters fail when the window is wrong:** one year was perfect for fresh content and
  returned nothing relevant once the current documents were 1.5 years old.
- **Supersession is what removes stale context:** recency alone left 0.80-0.91 outdated versions in
  each top-5 (depending on the query set); supersession removed all of them, even with no recency
  weighting at all.
- **Without a document key, linking needs a high threshold:** 0.95 (`vector_score`, raw cosine 0.90)
  linked versions with 100% precision and flagged 95.8% of stale documents; 0.80 had 41.3% precision.
  See the table in [Supersession](concepts/supersession.md#choosing-a-threshold).

**Caveats.** 55 articles, one small embedding model, templated queries rather than real user
questions. Wikipedia leads are rewritten rather than appended to, which suits similarity-based
linking. Full per-query-set numbers are in
[`results/results.json`](https://github.com/ashishodu2023/TideVec/blob/main/benchmarks/temporal_real_data/results/results.json).

## Illustrative: `temporal_vs_qdrant`

[`benchmarks/temporal_vs_qdrant`](https://github.com/ashishodu2023/TideVec/tree/main/benchmarks/temporal_vs_qdrant)
uses hand-set similarities to reproduce one failure mode. It is a demonstration, not evidence; use the
real-data benchmark for claims.

## In progress

!!! info "No published results yet"
    ANN recall and throughput (`benchmarks/ann`), GPU (`benchmarks/gpu`), agent memory
    (`benchmarks/agent_memory`) and a larger temporal benchmark with natural queries
    (`benchmarks/temporal_v2`) are being built. Until each has a README and committed results, there
    are **no** TideVec throughput, latency, recall-at-scale or GPU numbers to quote.
    <!-- TODO(docs): link each benchmark here once its README and results are committed. -->

## Scale projections

The README's "scale projections" table is arithmetic from vector sizes and index parameters, not
measurements. Treat it as a sizing estimate only.
