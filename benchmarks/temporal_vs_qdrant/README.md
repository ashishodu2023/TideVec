# TideVec vs Qdrant — Temporal Ranking Benchmark

Reproducible answer to: **"How does TideVec compare to Qdrant with a timestamp metadata filter?"**

## Quick run

```bash
# No deps required (stdlib only)
python3 run_benchmark.py

# Optional: validate hard-filter path against a live Qdrant
pip install qdrant-client
docker run -p 6333:6333 qdrant/qdrant
python3 run_benchmark.py --qdrant-url http://localhost:6333
```

Results are written to `results/benchmark_results.json`.

## Corpus

| Slice | Count | Role |
|---|---|---|
| Policy topics × 4 dated versions | 32 | Current + stale refund/return policies |
| FAQ / support distractors | 20 | Semantically adjacent noise |
| Off-topic distractors | 12 | Shipping, account, billing, privacy |
| **Total** | **64** | |
| Policy queries (1 per topic) | 8 | Each has a known gold current doc |

Embeddings for policy docs use **controlled cosines** matching the live demo failure mode:

```
stale  v1  cosine ≈ 0.837
stale  v2  cosine ≈ 0.832
stale  v3  cosine ≈ 0.827
current v4 cosine ≈ 0.822   ← lower semantic, but authoritative
```

FAQ/distractor docs use hashed bag-of-words (no ML dependency).

### Age cohorts (the hard-filter trap)

| Cohort | Topics | Current policy age | What a ≤14d filter does |
|---|---|---|---|
| Fresh current | electronics, software, shipping, digital | 1–5 days | Includes gold ✓ |
| Current >14d | apparel, warranty, subscription, damaged | 18–22 days | **Excludes gold** ✗ |

## Strategies

1. **Qdrant cosine-only** — default ANN, no time awareness  
2. **Qdrant filter ≤14d** — `age_days <= 14` metadata filter, then cosine  
3. **Qdrant filter ≤30d** — wider window (common “just increase the TTL” fix)  
4. **TideVec blend** — `final = 0.7·cosine + 0.3·2^(−age/7d)` (matches `TemporalScorer`)

## Latest results

```
Strategy             Gold@1 ↑  Stale@1 ↓  Mean gold rank ↓  nDCG@5 ↑
Qdrant cosine-only     0.0%    100.0%      4.00             0.601
Qdrant filter ≤14d    50.0%      0.0%      6.00             0.409
Qdrant filter ≤30d     0.0%    100.0%      2.00             0.633
TideVec blend β=0.3  100.0%      0.0%      1.00             1.000

Cohort Gold@1          Fresh current    Current >14d
Qdrant cosine-only       0%               0%
Qdrant filter ≤14d     100%               0%
Qdrant filter ≤30d       0%               0%
TideVec blend β=0.3    100%             100%
```

## Interpretation

| Approach | Failure mode |
|---|---|
| Cosine-only | Always prefers the longer/older policy text when its semantic score is slightly higher |
| Hard ≤14d filter | Works only while the *current* doc is inside the window; silently drops authoritative policies that just haven’t been rewritten recently |
| Hard ≤30d filter | Re-admits stale versions that still beat current on cosine — back to square one |
| TideVec soft decay | Continuously demotes stale docs; no window to tune; older knowledge stays searchable |

**A timestamp metadata filter is binary include/exclude. TideVec is a continuous rerank.**
