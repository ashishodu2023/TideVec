# Temporal retrieval on real revision history

Does a vector search return the **current** version of a document when older versions are also indexed?
This benchmark answers that on real data: the lead section of 55 Wikipedia articles as it stood on
2018-01-01, 2020-01-01, 2022-01-01, 2024-01-01 and today (270 documents, 2-5 distinct versions per
article, `created_at` = revision timestamp). The newest version of each article is ground truth.

```bash
pip install "tidevec[embedded]" fastembed numpy
python fetch_corpus.py        # ~6 min, MediaWiki API; writes data/corpus.jsonl (CC BY-SA)
python run_benchmark.py       # embeddings: BAAI/bge-small-en-v1.5 via fastembed
```

## Setup

**Strategies** (k = 5, half-life 365 days, blend 0.3):

| Strategy | What it is |
|---|---|
| `cosine` | Plain ANN similarity, what any vector database returns by default |
| `filter-1y`, `filter-2y` | Hard filter `created_at >= now - window`, then cosine |
| `rerank` | Cosine top-50, then rerank by `0.7·sim + 0.3·decay` (the pattern behind Qdrant's decay formulas and Milvus decay rankers) |
| `tidevec` | TideVec with the same formula applied inside the HNSW search |
| `tidevec+super` | Plus supersession (`match_field="article"`), superseded versions excluded |
| `super only (blend 0)` | Supersession and exclusion, no recency weighting at all |

**Query sets** (one query per article):

- `question`: "What is {article}?" (neutral wording).
- `old`: the first sentence of the oldest version, which is how someone with outdated knowledge phrases it.
- `new`: the first sentence of the current version, which favours plain cosine.

**Scenarios:**

- `today`: the current versions are fresh (a few days old at run time).
- `as_of_2025-07-01`: the corpus as it stood mid-2025, where the current version of every article is the
  2024 snapshot, about 550 days old. This models documents that are revised every year or two, such as
  policies, handbooks and specs.

## Results (bge-small-en-v1.5)

Current@1 is the share of queries whose top hit is the current version. Stale@1 is the share whose top
hit is an older version of the right article. Stale-in-top-5 is the average number of outdated versions
in the 5 results a RAG prompt would receive.

**Scenario `today`, `question` queries:**

| Strategy | Current@1 | Stale@1 | Stale in top-5 |
|---|---:|---:|---:|
| cosine | 18.2% | 81.8% | 3.89 |
| filter-1y / filter-2y | 100% | 0% | 0.00 |
| rerank | 100% | 0% | 0.00 |
| tidevec | 100% | 0% | 0.00 |
| tidevec+super | 100% | 0% | 0.00 |

**Scenario `as_of_2025-07-01`** (current versions about 1.5 years old):

| Strategy | Current@1 (question / old / new) | Stale in top-5 (question) |
|---|---:|---:|
| cosine | 27.3% / 16.4% / 45.5% | 2.89 |
| filter-1y | **0% / 0% / 0%** (returns nothing relevant) | 0.00 |
| filter-2y | 100% / 98.2% / 100% | 0.00 |
| rerank | 100% / 98.2% / 100% | 0.91 |
| tidevec | 100% / 98.2% / 100% | 0.91 |
| tidevec+super | 100% / 98.2% / 100% | **0.00** |
| super only (blend 0) | 100% / 98.2% / 100% | **0.00** |

**Automatic supersession without a document key** (similarity only; all 270 docs inserted in time order):

| Threshold (vector_score) | Raw cosine | Links | Precision | Stale docs flagged |
|---:|---:|---:|---:|---:|
| 0.80 | 0.60 | 1283 | 41.3% | 100% |
| 0.85 | 0.70 | 1030 | 51.5% | 100% |
| 0.90 | 0.80 | 668 | 79.2% | 100% |
| 0.95 | 0.90 | 476 | 100% | 95.8% |

Full numbers for every query set are in [`results/results.json`](results/results.json).

## What this shows

1. **Plain similarity returns stale content most of the time.** On real revisions, cosine picked an
   outdated version for 55–85% of queries, and about 3-4 of the top 5 results were outdated versions.
   Older lead sections are often shorter and more "definitional", so they match questions well.
2. **Any recency signal fixes the top hit, and TideVec's in-search blend is not more accurate than a
   rerank.** It produced exactly the same rankings as a top-50 rerank with the same formula. The
   advantage is operational: one stage, no candidate-pool size to tune, and recency applied to filtered
   searches too. It is not a quality advantage on this data.
3. **Hard filters fail when the window is wrong.** A 1-year window was perfect for fresh content and
   returned nothing relevant once the current documents were 1.5 years old. Choosing the window requires
   knowing how often each document changes.
4. **Supersession is what removes stale context.** Recency weighting still left about 0.9 outdated
   versions in each top-5 context window. Supersession removed all of them, and with `blend 0` it does so
   without any time-based tuning: the newest version wins because the older ones are marked as replaced,
   not because of their age.
5. **Automatic linking needs either a key or a high threshold.** With a document key (`match_field`),
   linking is exact. Without one, a vector_score threshold of 0.95 (raw cosine 0.90) linked versions with
   100% precision and flagged 95.8% of stale documents. Lower thresholds start linking different
   articles together.

## Caveats

- 55 articles and one small embedding model. The queries are templated or taken from the documents
  themselves, not written by real users.
- Wikipedia leads are rewritten rather than appended to, which suits supersession. Corpora where an
  update is a short delta ("price is now $12") will have lower version-to-version similarity, so use
  `match_field`.
- For the cosine metric TideVec's `vector_score` is `(1 + cos) / 2`. Supersession thresholds and the
  blend operate in that space; the tables report both scales.
- The older synthetic benchmark in [`../temporal_vs_qdrant`](../temporal_vs_qdrant) uses hand-set cosine
  values to reproduce one failure mode. Use this one for claims about real data.
