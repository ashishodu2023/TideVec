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
  55–85% of queries across the query sets and scenarios.
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

## Temporal benchmark v2: more corpora, natural queries, decay baselines

[`benchmarks/temporal_v2`](https://github.com/ashishodu2023/TideVec/tree/main/benchmarks/temporal_v2)
([summary](https://github.com/ashishodu2023/TideVec/blob/main/benchmarks/temporal_v2/results/summary.md)):
Wikipedia policy/guideline leads (628 documents, 185 titles) and PyPI READMEs (492 documents,
128 titles), 2-5 dated versions each, three embedding models (all-MiniLM-L6-v2, bge-small-en-v1.5,
jina-embeddings-v2-small-en), template plus hand-written natural-language queries. The Qdrant decay
formulas and Milvus decay rankers are reimplemented as documented over a top-100 cosine prefetch;
they are not live Qdrant or Milvus instances.

Current@1, all query sets pooled, averaged over the three models:

| Strategy | policy, today | policy, as of mid-2025 | docs, today | docs, as of mid-2025 |
|---|---:|---:|---:|---:|
| cosine | 32.5% | 32.0% | 32.8% | 34.6% |
| best Qdrant/Milvus decay | 77.0% | 69.6% | 82.9% | 72.9% |
| TideVec recency blend 0.3 | 58.6% | 60.9% | 67.0% | 10.0% |
| supersession by key, no recency | **82.8%** | **82.1%** | **90.0%** | **91.4%** |

- Supersession by document key had the highest Current@1 in every corpus and scenario (on two of
  the natural-query subsets a hard 1-2 year filter tied it).
- **Recency weighting is fragile:** with a 365-day half-life the 0.3 blend fell to 10.0% on the docs
  corpus as of mid-2025 (current versions were a median 580 days old), far below plain cosine. The
  best decay formulas also lost 7-10 points between scenarios.

## Agent memory: LongMemEval knowledge-update

[`benchmarks/agent_memory`](https://github.com/ashishodu2023/TideVec/tree/main/benchmarks/agent_memory)
stores every user turn of each LongMemEval-S (cleaned) knowledge-update haystack as a memory dated by
its session, then asks the question at its question date. Retrieval only, no LLM reader; 72
questions, k = 5, `bge-small-en-v1.5`.

| Strategy | current@1 | stale@1 | stale in top-5 |
|---|---:|---:|---:|
| plain vector store | 0.250 | 0.278 | 0.931 |
| cosine top-50 + recency rerank | 0.389 | 0.069 | 0.306 |
| `AgentMemory` (defaults: blend 0.2, supersede 0.95) | 0.389 | 0.042 | 0.208 |
| `AgentMemory` + oracle key (upper bound) | 0.403 | 0.000 | 0.000 |

One embedding model; mem0 and Zep have not been compared yet.

## ANN: SIFT-1M and GloVe-100 vs. hnswlib and FAISS

[`benchmarks/ann`](https://github.com/ashishodu2023/TideVec/tree/main/benchmarks/ann)
([summary](https://github.com/ashishodu2023/TideVec/blob/main/benchmarks/ann/results/summary.md)):
M = 16, ef_construction = 200, k = 10, 10,000 queries, single-threaded unless noted.

| | TideVec | hnswlib | FAISS HNSW |
|---|---:|---:|---:|
| SIFT-1M, QPS at recall@10 ≥ 0.95 | 2,656 | 11,401 | 5,074 |
| SIFT-1M, QPS at recall@10 ≥ 0.99 | 1,185 | 4,149 | 3,046 |
| SIFT-1M, build (1 thread) | 1,767 s | 231 s | 547 s |
| GloVe-100 (1.18M), QPS at recall@10 ≥ 0.9 | 351 | 2,111 | 2,058 |
| GloVe-100, build (1 thread) | 2,497 s | 375 s | 331 s |

**TideVec trails both libraries:** 3.5-6× lower QPS than hnswlib at equal recall, about 7× slower to
build than hnswlib (its insert path is single-threaded) and 1.5-1.9× the index memory. Its value is in what it
ranks (recency, supersession, filters), not raw ANN speed.

## GPU: not usable

[`benchmarks/gpu`](https://github.com/ashishodu2023/TideVec/tree/main/benchmarks/gpu/README.md) ran the
CUDA engines against NVIDIA cuVS on an RTX 4060 Ti. The CAGRA-style graph engine returned essentially
random results (recall@10 0.003 on GloVe-1.2M) and the cuBLAS brute-force engine was not exact
(0.9945) and 4× slower than cuVS brute force. The CUDA path is unsupported until it is rewritten or
replaced by cuVS.

## Scale projections

The README's "scale projections" table is arithmetic from vector sizes and index parameters, not
measurements. Treat it as a sizing estimate only.
