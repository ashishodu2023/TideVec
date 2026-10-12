"""
Temporal retrieval on real revision history (Wikipedia lead sections).

Each article appears in the corpus 2-5 times: its lead section as of
2018, 2020, 2022, 2024 and today. A query about an article should return
the *current* version. Strategies compared:

  cosine            plain ANN similarity (any vector DB)
  filter-1y/2y      hard filter created_at >= now - window, then cosine
  rerank            cosine top-50, then rerank by (1-b)*cos + b*decay
                    (what Qdrant's decay formula / Milvus decay ranker do)
  tidevec           TideVec, decay blended inside the HNSW search
  tidevec+super     TideVec with supersession on (match_field=article),
                    superseded versions excluded from results
  super only        supersession + exclusion, no recency blend

Query sets:
  question   "What is <article>?"                 neutral
  old        first sentence of the OLDEST version  phrased like stale text
  new        first sentence of the CURRENT version phrased like current text

Also reports precision/recall of fully automatic (similarity-only)
supersession links at several thresholds.

    python fetch_corpus.py
    python run_benchmark.py                     # bge-small via fastembed if installed
    python run_benchmark.py --embedder hashing  # no model download
"""

import argparse
import calendar
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
sys.path.insert(0, str(HERE.parents[1] / "sdk" / "python"))

from tidevec import Local  # noqa: E402

DAY = 86_400_000
_COORDS = re.compile(r"^[\d°′″NSEW.\s/;\ufeff\-]+(?=[A-Z])")


# ---- embedders ------------------------------------------------------

class FastEmbedder:
    name = "BAAI/bge-small-en-v1.5 (fastembed)"

    def __init__(self):
        from fastembed import TextEmbedding
        self.model = TextEmbedding("BAAI/bge-small-en-v1.5",
                                   cache_dir=str(HERE.parents[1] / ".uv-cache" / "fastembed"))

    def docs(self, texts):
        return np.array(list(self.model.passage_embed(texts)), dtype=np.float32)

    def queries(self, texts):
        return np.array(list(self.model.query_embed(texts)), dtype=np.float32)


class HashEmbedder:
    name = "HashingEmbedder(384) (lexical)"

    def __init__(self):
        from tidevec.embeddings import HashingEmbedder
        self.e = HashingEmbedder(384)

    def docs(self, texts):
        return np.array(self.e.embed_documents(texts), dtype=np.float32)

    queries = docs


def make_embedder(kind):
    if kind in ("auto", "fastembed"):
        try:
            return FastEmbedder()
        except ImportError:
            if kind == "fastembed":
                raise
    return HashEmbedder()


# ---- data -----------------------------------------------------------

def load_corpus(path):
    docs = [json.loads(line) for line in open(path)]
    for d in docs:
        d["text"] = _COORDS.sub("", d["text"]).strip()
        d["id"] = f"{d['article']}#{d['version']}"
    return docs


def first_sentence(text):
    m = re.match(r"(.+?[.!?])\s", text + " ")
    return (m.group(1) if m else text)[:300]


def build_queries(docs):
    by_article = defaultdict(list)
    for d in docs:
        by_article[d["article"]].append(d)
    sets = {"question": [], "old": [], "new": []}
    for article, versions in by_article.items():
        versions.sort(key=lambda d: d["created_at"])
        sets["question"].append((article, f"What is {article}?"))
        sets["old"].append((article, first_sentence(versions[0]["text"])))
        sets["new"].append((article, first_sentence(versions[-1]["text"])))
    return sets


# ---- strategies -----------------------------------------------------

def decay(ages_ms, half_life_ms):
    return np.power(0.5, np.maximum(ages_ms, 0) / half_life_ms)


def rank_cosine(q, E, meta, k, **_):
    return list(np.argsort(-(E @ q))[:k])


def rank_filter(q, E, meta, k, window_ms, now, **_):
    sims = E @ q
    sims[meta["created_at"] < now - window_ms] = -np.inf
    order = np.argsort(-sims)
    return [i for i in order[:k] if np.isfinite(sims[i])]


def rank_rerank(q, E, meta, k, blend, half_life_ms, now, pool=50, **_):
    # Same score space as TideVec: cosine mapped to [0, 1].
    sims = (1 + E @ q) / 2
    cand = np.argsort(-sims)[:pool]
    tw = decay(now - meta["created_at"][cand], half_life_ms)
    scores = (1 - blend) * sims[cand] + blend * tw
    return list(cand[np.argsort(-scores)][:k])


# ---- metrics --------------------------------------------------------

def score(rankings, queries, docs):
    n = len(queries)
    cur1 = stale1 = wrong1 = 0
    stale5 = cur5 = 0.0
    for (article, _), ranked in zip(queries, rankings):
        top = [docs[i] for i in ranked]
        if top:
            t = top[0]
            if t["article"] != article:
                wrong1 += 1
            elif t["is_current"]:
                cur1 += 1
            else:
                stale1 += 1
        stale5 += sum(1 for d in top[:5] if d["article"] == article and not d["is_current"])
        cur5 += any(d["article"] == article and d["is_current"] for d in top[:5])
    return {"current@1": cur1 / n, "stale@1": stale1 / n, "wrong@1": wrong1 / n,
            "current_in_top5": cur5 / n, "stale_in_top5": stale5 / n}


def tidevec_rankings(db, coll, Q, queries, id_to_idx, k, **search_kwargs):
    out = []
    for q in Q:
        hits = db.search(coll, q, top_k=k * 4 if search_kwargs.get("_exclude") else k,
                         **{kk: v for kk, v in search_kwargs.items() if not kk.startswith("_")})
        if search_kwargs.get("_exclude"):
            hits = [h for h in hits if not h.superseded_by]
        out.append([id_to_idx[h.id] for h in hits[:k]])
    return out


def load_tidevec(path, name, docs, E, half_life_ms, blend, supersession=None, now=None):
    # The engine measures age against the wall clock; shift timestamps so
    # ages are relative to the scenario's `now`.
    shift = int(time.time() * 1000 - now) if now else 0
    db = Local(path)
    kwargs = {}
    if supersession:
        kwargs = dict(supersession=True, supersede_threshold=supersession["threshold"],
                      supersede_match_field=supersession.get("match_field", ""))
    db.create_collection(name, dim=E.shape[1], n_shards=1, half_life_ms=half_life_ms,
                         temporal_blend=blend, **kwargs)
    order = sorted(range(len(docs)), key=lambda i: docs[i]["created_at"])  # arrive in time order
    for i in order:
        d = docs[i]
        db.upsert(name, [{"id": d["id"], "embedding": E[i], "created_at": d["created_at"] + shift,
                          "payload": {"article": d["article"]}}])
    return db


def link_quality(db, name, docs, E):
    """Precision/recall of automatic supersession links."""
    by_id = {d["id"]: d for d in docs}
    links = set()
    for i, d in enumerate(docs):
        for h in db.search(name, E[i], top_k=10, temporal_blend=0.0):
            for newer in h.superseded_by:
                links.add((h.id, newer))
    correct = sum(1 for old, new in links
                  if by_id[old]["article"] == by_id[new]["article"]
                  and by_id[old]["created_at"] < by_id[new]["created_at"])
    stale = [d for d in docs if not d["is_current"]]
    flagged = {old for old, new in links
               if by_id[old]["article"] == by_id[new]["article"]}
    return {"links": len(links),
            "precision": correct / len(links) if links else 1.0,
            "stale_docs_flagged": len([d for d in stale if d["id"] in flagged]) / len(stale)}


# ---- main -----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(HERE / "data" / "corpus.jsonl"))
    ap.add_argument("--embedder", default="auto", choices=["auto", "fastembed", "hashing"])
    ap.add_argument("--blend", type=float, default=0.3)
    ap.add_argument("--half-life-days", type=float, default=365)
    ap.add_argument("--out", default=str(HERE / "results" / "results.json"))
    args = ap.parse_args()

    all_docs = load_corpus(args.corpus)
    emb = make_embedder(args.embedder)
    print(f"corpus: {len(all_docs)} docs, {len(set(d['article'] for d in all_docs))} articles; "
          f"embedder: {emb.name}")
    E_all = emb.docs([d["text"] for d in all_docs])
    E_all /= np.linalg.norm(E_all, axis=1, keepdims=True)
    hl = args.half_life_days * DAY
    k = 5

    as_of = calendar.timegm(time.strptime("2025-07-01", "%Y-%m-%d")) * 1000
    scenarios = {"today": time.time() * 1000, "as_of_2025-07-01": as_of}

    results = {"embedder": emb.name, "n_docs": len(all_docs), "blend": args.blend,
               "half_life_days": args.half_life_days, "scenarios": {}}
    tmp = tempfile.mkdtemp(prefix="tidevec-bench-")
    try:
        for label, now in scenarios.items():
            # The corpus as it existed at `now`: newest surviving version is current.
            keep = [i for i, d in enumerate(all_docs) if d["created_at"] <= now]
            docs = [dict(all_docs[i]) for i in keep]
            newest = {}
            for d in docs:
                newest[d["article"]] = max(newest.get(d["article"], 0), d["created_at"])
            for d in docs:
                d["is_current"] = d["created_at"] == newest[d["article"]]
            docs_by_article = defaultdict(int)
            for d in docs:
                docs_by_article[d["article"]] += 1
            E = E_all[keep]
            meta = {"created_at": np.array([d["created_at"] for d in docs], dtype=np.float64)}
            id_to_idx = {d["id"]: i for i, d in enumerate(docs)}
            ages = [(now - d["created_at"]) / DAY for d in docs if d["is_current"]]
            query_sets = build_queries([d for d in docs if docs_by_article[d["article"]] > 1])

            tv = load_tidevec(f"{tmp}/{label}-plain", "wiki", docs, E, hl, args.blend, now=now)
            tvs = load_tidevec(f"{tmp}/{label}-super", "wiki", docs, E, hl, args.blend, now=now,
                               supersession={"threshold": 0.5, "match_field": "article"})
            sc = {"n_docs": len(docs), "now": int(now),
                  "current_version_age_days": {"median": float(np.median(ages)),
                                               "max": float(np.max(ages))},
                  "query_sets": {}}
            for qname, queries in query_sets.items():
                Q = emb.queries([q for _, q in queries])
                Q /= np.linalg.norm(Q, axis=1, keepdims=True)
                strategies = {
                    "cosine": [rank_cosine(q, E, meta, k) for q in Q],
                    "filter-1y": [rank_filter(q, E, meta, k, 365 * DAY, now) for q in Q],
                    "filter-2y": [rank_filter(q, E, meta, k, 730 * DAY, now) for q in Q],
                    "rerank": [rank_rerank(q, E, meta, k, args.blend, hl, now) for q in Q],
                    "tidevec (blend 0)": tidevec_rankings(tv, "wiki", Q, queries, id_to_idx, k,
                                                          temporal_blend=0.0),
                    "tidevec": tidevec_rankings(tv, "wiki", Q, queries, id_to_idx, k),
                    "tidevec+super": tidevec_rankings(tvs, "wiki", Q, queries, id_to_idx, k,
                                                      _exclude=True),
                    "super only (blend 0)": tidevec_rankings(tvs, "wiki", Q, queries, id_to_idx,
                                                             k, _exclude=True, temporal_blend=0.0),
                }
                sc["query_sets"][qname] = {s: score(r, queries, docs)
                                           for s, r in strategies.items()}
            tv.close()
            tvs.close()
            results["scenarios"][label] = sc

        results["auto_supersession"] = {}
        for thr in (0.80, 0.85, 0.90, 0.95):
            db = load_tidevec(f"{tmp}/auto{thr}", "wiki", all_docs, E_all, hl, args.blend,
                              supersession={"threshold": thr})
            results["auto_supersession"][str(thr)] = link_quality(db, "wiki", all_docs, E_all)
            db.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))

    cols = ["current@1", "stale@1", "wrong@1", "current_in_top5", "stale_in_top5"]
    for label, sc in results["scenarios"].items():
        a = sc["current_version_age_days"]
        print(f"\n=== scenario {label}: {sc['n_docs']} docs, current versions median age "
              f"{a['median']:.0f} d (max {a['max']:.0f} d)")
        for qname, table in sc["query_sets"].items():
            print(f"queries: {qname}")
            print(f"  {'strategy':<20}" + "".join(f"{c:>17}" for c in cols))
            for s, m in table.items():
                print(f"  {s:<20}" + "".join(
                    f"{m[c]:>17.2f}" if c == "stale_in_top5" else f"{m[c] * 100:>16.1f}%"
                    for c in cols))
    print("\nautomatic supersession (similarity only, no match_field):")
    for thr, m in results["auto_supersession"].items():
        print(f"  threshold {thr} (cos {2 * float(thr) - 1:.2f}): {m['links']:>4} links, "
              f"precision {m['precision'] * 100:5.1f}%, "
              f"stale docs flagged {m['stale_docs_flagged'] * 100:5.1f}%")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
