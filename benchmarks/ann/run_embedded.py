"""
The same benchmark through `tidevec.Local` (the pip `tidevec-embedded` wheel):
portable -O3 build, WAL writes on upsert, JSON encoding per call.

    python run_embedded.py --dataset sift-128-euclidean-100000
"""

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import psutil

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "sdk" / "python"))

from tidevec import Local  # noqa: E402
from run_refs import read_bin  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ef", default="10,32,64,128,256")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--nq", type=int, default=2000)
    args = ap.parse_args()

    d = HERE / "data" / args.dataset
    meta = json.loads((d / "meta.json").read_text())
    X = read_bin(d / "base.fbin", np.float32)
    Q = read_bin(d / "query.fbin", np.float32)[: args.nq]
    gt = read_bin(d / "gt.ibin", np.int32)[: args.nq, : args.k]

    tmp = tempfile.mkdtemp(prefix="tidevec-embedded-bench-")
    try:
        db = Local(tmp, segment_store=False)
        db.create_collection("c", dim=X.shape[1], metric=meta["metric"], temporal_blend=0.0)
        before = psutil.Process().memory_info().rss / 2**20
        t0 = time.perf_counter()
        for s in range(0, len(X), 1000):
            db.upsert("c", [{"id": str(i), "embedding": X[i], "created_at": 0}
                            for i in range(s, min(s + 1000, len(X)))])
        out = {"engine": "tidevec-embedded (Local)", "n": len(X), "d": X.shape[1], "nq": len(Q),
               "build_s": time.perf_counter() - t0,
               "index_rss_mb": psutil.Process().memory_info().rss / 2**20 - before, "runs": []}
        for ef in map(int, args.ef.split(",")):
            lat = np.empty(len(Q))
            found = []
            t0 = time.perf_counter()
            for i, q in enumerate(Q):
                s = time.perf_counter()
                hits = db.search("c", q, top_k=args.k, ef_search=ef, metric=meta["metric"],
                                 include_staleness_warnings=False).hits
                lat[i] = (time.perf_counter() - s) * 1e6
                found.append([int(h.id) for h in hits])
            wall = time.perf_counter() - t0
            recall = np.mean([len(set(f) & set(g)) / args.k for f, g in zip(found, gt)])
            out["runs"].append({"ef_search": ef, "recall": float(recall), "qps": len(Q) / wall,
                                "p50_us": float(np.percentile(lat, 50)),
                                "p99_us": float(np.percentile(lat, 99))})
            print(f"  ef={ef} recall={recall:.4f} qps={len(Q) / wall:.0f}", file=sys.stderr)
        db.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
