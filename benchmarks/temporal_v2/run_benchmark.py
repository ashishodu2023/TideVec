"""
Temporal retrieval benchmark v2.

Corpora (fetch_corpus.py): Wikipedia article leads, Wikipedia policy and
guideline leads, PyPI READMEs, each with 2-5 dated versions per document.
A query about a document should return its CURRENT version.

Strategies (all see the same embeddings; k = 5, nDCG over the top 10):

  cosine                    plain similarity (what any vector DB returns)
  filter-1y / filter-2y     hard created_at >= now - window, then cosine
  rerank (linear 0.3)       cosine top-100, then 0.7*sim01 + 0.3*2^(-age/hl)
  qdrant {exp,gauss,lin}    Qdrant formula query, exactly as documented:
                            $score + <decay>(now - t), scale = hl, midpoint 0.5,
                            over a top-100 cosine prefetch
  qdrant 0.1*exp            same with the decay weighted 0.1 (the calibration
                            Qdrant's hybrid-search docs recommend)
  milvus {gauss,exp,linear} Milvus decay rankers, exactly as documented:
                            cosine * decay(|now - t|), origin = now,
                            scale = hl, offset 0, decay 0.5, over top-100
  tidevec                   recency blend inside the HNSW search (blend 0.3)
  tidevec+super (key)       + supersession with match_field = document title,
                            superseded versions excluded
  super only (key, blend 0) supersession and exclusion, no recency weighting
  tidevec+auto-super        supersession by similarity only (threshold 0.95,
                            no key), superseded excluded, blend 0.3
  auto-super only           the same without recency weighting (blend 0)

Half-life / scale hl = 365 days everywhere.

Metrics: current@1, stale@1 (top hit is an older version of the right
document), wrong@1, stale_in_top5 (older versions of the right document in
the top 5), ndcg@10 (relevance 1 for the current version only).

    python fetch_corpus.py
    python run_benchmark.py                                   # 3 models
    python run_benchmark.py --models BAAI/bge-small-en-v1.5 --corpora policy
"""

import argparse
import calendar
import hashlib
import json
import re
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "sdk" / "python"))

from tidevec import Local  # noqa: E402

DAY = 86_400_000
K, K_NDCG, POOL = 5, 10, 100
MODELS = ["BAAI/bge-small-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2",
          "jinaai/jina-embeddings-v2-small-en"]
_COORDS = re.compile(r"^[\d°′″NSEW.\s/;\ufeff\-]+(?=[A-Z])")


# ---- data -------------------------------------------------------------

def load_corpus(corpus):
    seen, docs = set(), []
    for line in open(HERE / "data" / f"{corpus}.jsonl"):
        d = json.loads(line)
        key = (d["title"], d["version"])
        if key in seen:
            continue
        seen.add(key)
        d["text"] = _COORDS.sub("", d["text"]).strip()
        d["id"] = f"{d['title']}#{d['version']}"
        docs.append(d)
    return docs


def first_sentence(text):
    m = re.match(r"(.+?[.!?])\s", text + " ")
    return (m.group(1) if m else text)[:300]


def template_question(corpus, title):
    if corpus == "policy":
        return f"What is Wikipedia's guidance on {title.split(':', 1)[1].lower()}?"
    if corpus == "docs":
        return f"What is the {title} Python package?"
    return f"What is {title}?"


def build_queries(corpus, docs, natural):
    by_title = defaultdict(list)
    for d in docs:
        by_title[d["title"]].append(d)
    sets = {"question": [], "old": [], "new": [], "natural": []}
    for title, versions in by_title.items():
        if len(versions) < 2:
            continue
        versions.sort(key=lambda d: d["created_at"])
        sets["question"].append((title, template_question(corpus, title)))
        sets["old"].append((title, first_sentence(versions[0]["text"])))
        sets["new"].append((title, first_sentence(versions[-1]["text"])))
    sets["natural"] = [(q["title"], q["query"]) for q in natural
                       if q["corpus"] == corpus and len(by_title.get(q["title"], [])) > 1]
    return sets


# ---- embeddings ----------------------------------------------------------

class Embedder:
    def __init__(self, model):
        from fastembed import TextEmbedding
        self.name = model
        self.model = TextEmbedding(model, cache_dir=str(ROOT / ".uv-cache" / "fastembed"),
                                   threads=4)

    def _cached(self, texts, kind, fn):
        h = hashlib.sha1(("\x00".join(texts) + kind + self.name).encode()).hexdigest()[:16]
        path = HERE / "data" / "emb" / f"{h}.npy"
        if path.exists():
            return np.load(path)
        E = np.array(list(fn(texts)), dtype=np.float32)
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, E)
        return E

    def docs(self, texts):
        return self._cached(texts, "p", lambda t: self.model.passage_embed(t, batch_size=32))

    def queries(self, texts):
        return self._cached(texts, "q", self.model.query_embed)


# ---- strategies ------------------------------------------------------------

def topn(scores, n=K_NDCG):
    order = np.argsort(-scores)[:n]
    return [i for i in order if np.isfinite(scores[i])]


def rank_cosine(q, E, ts, now, hl):
    return topn(E @ q)


def rank_filter(window_days):
    def fn(q, E, ts, now, hl):
        s = E @ q
        s[ts < now - window_days * DAY] = -np.inf
        return topn(s)
    return fn


def _rerank(q, E, ts, combine):
    sims = E @ q
    cand = np.argsort(-sims)[:POOL]
    final = combine(sims[cand], ts[cand])
    return list(cand[np.argsort(-final)][:K_NDCG])


def rank_linear(blend):
    def fn(q, E, ts, now, hl):
        return _rerank(q, E, ts, lambda s, t: (1 - blend) * (1 + s) / 2
                       + blend * np.power(0.5, np.maximum(now - t, 0) / hl))
    return fn


# Qdrant (https://qdrant.tech/documentation/search/search-relevance/#decay-functions):
#   lin_decay(x)   = max(0, -(1 - m)/s * |x - target| + 1)
#   exp_decay(x)   = exp(ln(m)/s * |x - target|)
#   gauss_decay(x) = exp(ln(m)/s^2 * (x - target)^2)
# Time-based boosting example: score = $score + exp_decay(now - point_time).
def qdrant_decay(kind, x, scale, m=0.5):
    if kind == "lin":
        return np.maximum(0.0, -(1 - m) / scale * np.abs(x) + 1)
    if kind == "exp":
        return np.exp(np.log(m) / scale * np.abs(x))
    return np.exp(np.log(m) / scale ** 2 * x ** 2)


def rank_qdrant(kind, weight=1.0):
    def fn(q, E, ts, now, hl):
        return _rerank(q, E, ts, lambda s, t: s + weight * qdrant_decay(kind, now - t, hl))
    return fn


# Milvus decay rankers (milvus-docs v2.6.x, userGuide/embeddings-reranking/reranking):
#   final = normalized_score * decay; COSINE scores are used as is.
#   gauss:  exp(-(max(0,|x-origin|-offset))^2 / (2 sigma^2)), sigma^2 = -scale^2 / (2 ln decay)
#   exp:    exp(lambda * max(0,|x-origin|-offset)), lambda = ln(decay)/scale
#   linear: max((s - max(0,|x-origin|-offset)) / s, 0), s = scale / (1 - decay)
def milvus_decay(kind, x, scale, decay=0.5, offset=0.0):
    d = np.maximum(0.0, np.abs(x) - offset)
    if kind == "gauss":
        sigma2 = -scale ** 2 / (2 * np.log(decay))
        return np.exp(-d ** 2 / (2 * sigma2))
    if kind == "exp":
        return np.exp(np.log(decay) / scale * d)
    s = scale / (1 - decay)
    return np.maximum((s - d) / s, 0.0)


def rank_milvus(kind):
    def fn(q, E, ts, now, hl):
        return _rerank(q, E, ts, lambda s, t: s * milvus_decay(kind, now - t, hl))
    return fn


NUMPY_STRATEGIES = {
    "cosine": rank_cosine,
    "filter-1y": rank_filter(365),
    "filter-2y": rank_filter(730),
    "rerank (linear 0.3)": rank_linear(0.3),
    "qdrant exp_decay": rank_qdrant("exp"),
    "qdrant gauss_decay": rank_qdrant("gauss"),
    "qdrant lin_decay": rank_qdrant("lin"),
    "qdrant 0.1*exp_decay": rank_qdrant("exp", 0.1),
    "milvus gauss": rank_milvus("gauss"),
    "milvus exp": rank_milvus("exp"),
    "milvus linear": rank_milvus("linear"),
}


def load_tidevec(path, docs, E, hl, blend, now, supersession=None):
    shift = int(time.time() * 1000 - now)  # engine ages are measured against the wall clock
    db = Local(path, segment_store=False)
    kw = {}
    if supersession:
        kw = dict(supersession=True, supersede_threshold=supersession["threshold"],
                  supersede_match_field=supersession.get("match_field", ""))
    db.create_collection("c", dim=E.shape[1], half_life_ms=int(hl), temporal_blend=blend, **kw)
    for i in sorted(range(len(docs)), key=lambda i: docs[i]["created_at"]):
        d = docs[i]
        db.upsert("c", [{"id": d["id"], "embedding": E[i], "created_at": d["created_at"] + shift,
                         "payload": {"title": d["title"]}}])
    return db


def tidevec_rank(db, Q, id_to_idx, exclude=False, **kw):
    out = []
    for q in Q:
        hits = db.search("c", q, top_k=K_NDCG * 4 if exclude else K_NDCG,
                         include_staleness_warnings=False, **kw).hits
        if exclude:
            hits = [h for h in hits if not h.superseded_by]
        out.append([id_to_idx[h.id] for h in hits[:K_NDCG]])
    return out


# ---- metrics -----------------------------------------------------------------

def score(rankings, queries, docs):
    m = defaultdict(float)
    for (title, _), ranked in zip(queries, rankings):
        top = [docs[i] for i in ranked]
        if top:
            t = top[0]
            m["current@1"] += t["title"] == title and t["is_current"]
            m["stale@1"] += t["title"] == title and not t["is_current"]
            m["wrong@1"] += t["title"] != title
        m["stale_in_top5"] += sum(d["title"] == title and not d["is_current"] for d in top[:K])
        rank = next((r for r, d in enumerate(top) if d["title"] == title and d["is_current"]), None)
        m["ndcg@10"] += 0.0 if rank is None else 1.0 / np.log2(rank + 2)
    return {k: v / len(queries) for k, v in m.items()}


# ---- main ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--corpora", nargs="+", default=["wiki", "policy", "docs"])
    ap.add_argument("--blend", type=float, default=0.3)
    ap.add_argument("--half-life-days", type=float, default=365)
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    args = ap.parse_args()

    natural = [json.loads(line) for line in open(HERE / "queries_natural.jsonl")]
    hl = args.half_life_days * DAY
    as_of = calendar.timegm(time.strptime("2025-07-01", "%Y-%m-%d")) * 1000
    scenarios = {"today": time.time() * 1000, "as_of_2025-07-01": as_of}
    tmp = tempfile.mkdtemp(prefix="tidevec-tv2-")
    try:
        for model in args.models:
            emb = Embedder(model)
            for corpus in args.corpora:
                all_docs = load_corpus(corpus)
                t0 = time.time()
                E_all = emb.docs([d["text"] for d in all_docs])
                print(f"[{model}] {corpus}: {len(all_docs)} docs, "
                      f"{len({d['title'] for d in all_docs})} titles (embedded in {time.time() - t0:.0f}s)")
                cres = {"n_docs": len(all_docs), "n_titles": len({d["title"] for d in all_docs}),
                        "scenarios": {}}
                for label, now in scenarios.items():
                    keep = [i for i, d in enumerate(all_docs) if d["created_at"] <= now]
                    docs = [dict(all_docs[i]) for i in keep]
                    newest = defaultdict(int)
                    for d in docs:
                        newest[d["title"]] = max(newest[d["title"]], d["created_at"])
                    for d in docs:
                        d["is_current"] = d["created_at"] == newest[d["title"]]
                    E = E_all[keep]
                    ts = np.array([d["created_at"] for d in docs], dtype=np.float64)
                    id_to_idx = {d["id"]: i for i, d in enumerate(docs)}
                    ages = [(now - d["created_at"]) / DAY for d in docs if d["is_current"]]
                    qsets = build_queries(corpus, docs, natural)

                    base = f"{tmp}/{abs(hash((model, corpus, label)))}"
                    tv = load_tidevec(base + "-plain", docs, E, hl, args.blend, now)
                    tvk = load_tidevec(base + "-key", docs, E, hl, args.blend, now,
                                       {"threshold": 0.5, "match_field": "title"})
                    tva = load_tidevec(base + "-auto", docs, E, hl, args.blend, now,
                                       {"threshold": 0.95})
                    sc = {"n_docs": len(docs),
                          "current_age_days": {"median": float(np.median(ages)),
                                               "p90": float(np.percentile(ages, 90))},
                          "query_sets": {}}
                    for qname, queries in qsets.items():
                        if not queries:
                            continue
                        Q = emb.queries([q for _, q in queries])
                        ranks = {name: [fn(q, E, ts, now, hl) for q in Q]
                                 for name, fn in NUMPY_STRATEGIES.items()}
                        ranks["tidevec"] = tidevec_rank(tv, Q, id_to_idx)
                        ranks["tidevec+super (key)"] = tidevec_rank(tvk, Q, id_to_idx, exclude=True)
                        ranks["super only (key, blend 0)"] = tidevec_rank(
                            tvk, Q, id_to_idx, exclude=True, temporal_blend=0.0)
                        ranks["tidevec+auto-super (0.95)"] = tidevec_rank(tva, Q, id_to_idx,
                                                                          exclude=True)
                        ranks["auto-super only (0.95, blend 0)"] = tidevec_rank(
                            tva, Q, id_to_idx, exclude=True, temporal_blend=0.0)
                        sc["query_sets"][qname] = {"n": len(queries), "strategies": {
                            s: score(r, queries, docs) for s, r in ranks.items()}}
                    for db in (tv, tvk, tva):
                        db.close()
                        shutil.rmtree(db.path, ignore_errors=True)
                    cres["scenarios"][label] = sc
                    print(f"  {label}: {len(docs)} docs, current age median "
                          f"{sc['current_age_days']['median']:.0f} d")
                    for qname, qs in sc["query_sets"].items():
                        print(f"    {qname} (n={qs['n']})")
                        for s, m in qs["strategies"].items():
                            print(f"      {s:<28} cur@1 {m['current@1'] * 100:5.1f}%  "
                                  f"stale@1 {m['stale@1'] * 100:5.1f}%  "
                                  f"wrong@1 {m['wrong@1'] * 100:5.1f}%  "
                                  f"stale/top5 {m['stale_in_top5']:.2f}  ndcg {m['ndcg@10']:.3f}")
                cres.update(model=model, corpus=corpus, blend=args.blend,
                            half_life_days=args.half_life_days, pool=POOL, k=K)
                out = Path(args.out_dir) / f"{corpus}_{model.split('/')[-1]}.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(cres, indent=2))
                print(f"wrote {out}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
