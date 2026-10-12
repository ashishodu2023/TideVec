"""
Agent-memory benchmark on the knowledge-update subset of LongMemEval.

LongMemEval (Wu et al., 2024, https://github.com/xiaowu0162/LongMemEval)
gives each question a haystack of ~50 chat sessions. In the 78
`knowledge-update` questions the user states a fact in one session and
changes it in a later one ("my 5K personal best is 27:12" ... "beat my
personal best of 25:50"); the question asks for the current value. User
turns that carry the evidence are labelled `has_answer`.

Every user turn of the haystack is stored as one memory with
created_at = its session date, then the question is asked at its
question_date. This measures retrieval only (no LLM reader):

  current@1        top memory is an evidence turn from the LATEST answer session
  current@5        such a turn is in the top 5
  stale@1          top memory is an evidence turn from an OLDER answer session
  stale_in_top5    older evidence turns in the top 5 (what misleads a reader LLM)
  stale_first      an older evidence turn ranks above every latest-evidence turn
  mrr              reciprocal rank of the first latest-evidence turn

    python run_longmemeval.py                     # bge-small-en-v1.5 via fastembed
    python run_longmemeval.py --model sentence-transformers/all-MiniLM-L6-v2
"""

import argparse
import calendar
import hashlib
import json
import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "sdk" / "python"))

from tidevec import Local  # noqa: E402
from tidevec.memory import AgentMemory  # noqa: E402

DATA_URL = ("https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/"
            "longmemeval_s_cleaned.json")
DAY = 86_400_000
K = 5


def parse_date(s):
    return calendar.timegm(time.strptime(s.split(" (")[0] + s.split(")")[1], "%Y/%m/%d %H:%M")) * 1000


class CachedEmbedder:
    """fastembed model with every text embedded up front in batches.

    Stored memories use passage embeddings, questions use query embeddings,
    identically for every strategy.
    """

    def __init__(self, model):
        from fastembed import TextEmbedding
        self.name = model
        self.model = TextEmbedding(model, cache_dir=str(ROOT / ".uv-cache" / "fastembed"),
                                   threads=4)
        self.cache = {}
        self.dim = None

    def _key(self, text, kind):
        return kind + hashlib.sha1(text.encode()).hexdigest()

    def prime(self, passages, queries, cache_file):
        if cache_file.exists():
            z = np.load(cache_file, allow_pickle=True)
            self.cache = dict(zip(z["keys"].tolist(), z["vecs"]))
        todo_p = [t for t in dict.fromkeys(passages) if self._key(t, "p") not in self.cache]
        todo_q = [t for t in dict.fromkeys(queries) if self._key(t, "q") not in self.cache]
        if todo_p or todo_q:
            print(f"embedding {len(todo_p)} passages, {len(todo_q)} queries with {self.name}")
            for t, v in zip(todo_p, self.model.passage_embed(todo_p, batch_size=64)):
                self.cache[self._key(t, "p")] = v / np.linalg.norm(v)
            for t, v in zip(todo_q, self.model.query_embed(todo_q)):
                self.cache[self._key(t, "q")] = v / np.linalg.norm(v)
            np.savez(cache_file, keys=np.array(list(self.cache)), vecs=np.stack(list(self.cache.values())))
        self.dim = len(next(iter(self.cache.values())))
        self.queries = set(queries)

    def embed_query(self, text):
        kind = "q" if text in self.queries else "p"
        return self.cache[self._key(text, kind)].tolist()


def load_questions(path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, path)
    data = json.load(open(path))
    out = []
    for q in data:
        if q["question_type"] != "knowledge-update":
            continue
        turns, answer_dates = [], {}
        for sid, date, sess in zip(q["haystack_session_ids"], q["haystack_dates"],
                                   q["haystack_sessions"]):
            t0 = parse_date(date)
            for i, t in enumerate(sess):
                if t["role"] != "user" or not t["content"].strip():
                    continue
                turns.append({"id": f"{sid}:{i}", "session": sid, "text": t["content"],
                              "created_at": t0 + i * 60_000, "evidence": bool(t.get("has_answer"))})
                if t.get("has_answer"):
                    answer_dates[sid] = t0
        if len(answer_dates) < 1:
            continue
        latest = max(answer_dates, key=answer_dates.get)
        for t in turns:
            t["latest"] = t["evidence"] and t["session"] == latest
            t["older"] = t["evidence"] and t["session"] != latest
        out.append({"id": q["question_id"], "question": q["question"], "answer": q["answer"],
                    "now": parse_date(q["question_date"]), "turns": turns,
                    "n_answer_sessions": len(answer_dates)})
    return out


def score(ranked, turns_by_id):
    r = [turns_by_id[i] for i in ranked[:K]]
    first_latest = next((n for n, t in enumerate(r) if t["latest"]), None)
    first_older = next((n for n, t in enumerate(r) if t["older"]), None)
    return {
        "current@1": float(bool(r) and r[0]["latest"]),
        "current@5": float(first_latest is not None),
        "stale@1": float(bool(r) and r[0]["older"]),
        "stale_in_top5": float(sum(t["older"] for t in r)),
        "stale_first": float(first_older is not None
                             and (first_latest is None or first_older < first_latest)),
        "mrr": 0.0 if first_latest is None else 1.0 / (first_latest + 1),
    }


# ---- strategies -----------------------------------------------------

def plain_cosine(q, emb, turns, now, **_):
    E = np.array([emb.embed_query(t["text"]) for t in turns])
    sims = E @ np.array(emb.embed_query(q["question"]))
    return [turns[i]["id"] for i in np.argsort(-sims)[:K]]


def recency_rerank(q, emb, turns, now, blend=0.2, half_life_days=30, pool=50, **_):
    E = np.array([emb.embed_query(t["text"]) for t in turns])
    sims = (1 + E @ np.array(emb.embed_query(q["question"]))) / 2
    cand = np.argsort(-sims)[:pool]
    ages = np.array([now - turns[i]["created_at"] for i in cand], dtype=np.float64)
    decay = np.power(0.5, np.maximum(ages, 0) / (half_life_days * DAY))
    scores = (1 - blend) * sims[cand] + blend * decay
    return [turns[i]["id"] for i in cand[np.argsort(-scores)][:K]]


def agent_memory(q, emb, turns, now, tmp, blend=0.2, threshold=0.95, oracle_key=False, **_):
    # AgentMemory ranks recency against the wall clock: shift so the
    # question is asked "now".
    shift = int(time.time() * 1000) - now
    db = Local(tempfile.mkdtemp(dir=tmp), segment_store=False)
    mem = AgentMemory(db, collection="m", embedder=emb, temporal_blend=blend,
                      supersede_threshold=threshold)
    ids = {}
    for t in sorted(turns, key=lambda t: t["created_at"]):
        key = q["id"] if oracle_key and t["evidence"] else None
        ids[mem.remember(t["text"], key=key, created_at=t["created_at"] + shift)["id"]] = t["id"]
    hits = mem.recall(q["question"], k=K)
    db.close()
    shutil.rmtree(db.path, ignore_errors=True)
    return [ids[h["id"]] for h in hits]


STRATEGIES = {
    "vector store (cosine)": dict(fn=plain_cosine),
    "recency rerank (cos top-50, blend 0.2, hl 30d)": dict(fn=recency_rerank),
    "AgentMemory (default: blend 0.2, supersede 0.95)": dict(fn=agent_memory),
    "AgentMemory (supersede 0.90)": dict(fn=agent_memory, threshold=0.90),
    "AgentMemory (supersede 0.98)": dict(fn=agent_memory, threshold=0.98),
    "AgentMemory (blend 0, supersede 0.95)": dict(fn=agent_memory, blend=0.0),
    "AgentMemory + oracle key (upper bound)": dict(fn=agent_memory, oracle_key=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "data" / "longmemeval_s_cleaned.json"))
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--only", nargs="*", default=[],
                    help="re-run only strategies whose name contains one of these strings, "
                         "merging into the existing results file")
    args = ap.parse_args()

    qs = load_questions(Path(args.data))
    if args.limit:
        qs = qs[: args.limit]
    emb = CachedEmbedder(args.model)
    emb.prime([t["text"] for q in qs for t in q["turns"]], [q["question"] for q in qs],
              HERE / "data" / f"emb_{args.model.replace('/', '_')}.npz")
    n_turns = [len(q["turns"]) for q in qs]
    print(f"{len(qs)} knowledge-update questions, {np.mean(n_turns):.0f} user turns each; "
          f"{sum(q['n_answer_sessions'] > 1 for q in qs)} with evidence in 2+ sessions")

    out = Path(args.out or HERE / "results" / f"longmemeval_ku_{args.model.split('/')[-1]}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    results = {"dataset": "LongMemEval-S (cleaned), knowledge-update", "model": args.model,
               "n_questions": len(qs), "k": K, "strategies": {}}
    if args.only and out.exists():
        results["strategies"] = json.loads(out.read_text())["strategies"]
    tmp = tempfile.mkdtemp(prefix="tidevec-lme-")
    try:
        for name, cfg in STRATEGIES.items():
            if args.only and not any(s in name for s in args.only):
                continue
            cfg = dict(cfg)
            fn = cfg.pop("fn")
            t0 = time.time()
            try:
                rows = []
                for q in qs:
                    by_id = {t["id"]: t for t in q["turns"]}
                    rows.append(score(fn(q, emb, q["turns"], q["now"], tmp=tmp, **cfg), by_id))
            except Exception as e:  # noqa: BLE001  keep the other strategies' results
                results["strategies"][name] = {"error": f"{type(e).__name__}: {e}"}
                print(f"  {name:<50} ERROR {type(e).__name__}: {e}")
                continue
            agg = {m: float(np.mean([r[m] for r in rows])) for m in rows[0]}
            agg["seconds"] = time.time() - t0
            results["strategies"][name] = agg
            print(f"  {name:<50} " + "  ".join(f"{m} {v:.3f}" for m, v in agg.items()))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
