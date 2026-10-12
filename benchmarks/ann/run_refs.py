"""
Reference HNSW implementations on the same data and parameters as tvbench.

    python run_refs.py --engine hnswlib --dataset sift-128-euclidean
    python run_refs.py --engine faiss   --dataset glove-100-angular
    python run_refs.py --engine hnswlib --dataset sift-128-euclidean --build-threads 20 --build-only

Prints one JSON object with the same fields as tvbench. Latency is measured
per query from Python (one call per query, one thread), so it includes a few
microseconds of Python call overhead that tvbench does not have; `qps_batch`
is the same queries submitted as one single-threaded batch call.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import psutil

HERE = Path(__file__).parent


def read_bin(path, dtype):
    with open(path, "rb") as f:
        n, d = np.fromfile(f, dtype=np.int32, count=2)
        return np.fromfile(f, dtype=dtype, count=n * d).reshape(n, d)


def rss_mb():
    return psutil.Process().memory_info().rss / 2**20


class Hnswlib:
    def __init__(self, d, metric, M, efc, threads):
        import hnswlib
        self.index = hnswlib.Index(space="l2" if metric == "l2" else "cosine", dim=d)
        self.M, self.efc, self.threads = M, efc, threads

    def build(self, X):
        self.index.init_index(max_elements=len(X), ef_construction=self.efc, M=self.M)
        self.index.add_items(X, np.arange(len(X)), num_threads=self.threads)

    def set_ef(self, ef):
        self.index.set_ef(ef)

    def search(self, Q, k, threads=1):
        return self.index.knn_query(Q, k=k, num_threads=threads)[0]


class Faiss:
    def __init__(self, d, metric, M, efc, threads):
        import faiss
        self.faiss = faiss
        mt = faiss.METRIC_L2 if metric == "l2" else faiss.METRIC_INNER_PRODUCT
        self.index = faiss.IndexHNSWFlat(d, M, mt)
        self.index.hnsw.efConstruction = efc
        self.threads = threads

    def build(self, X):
        self.faiss.omp_set_num_threads(self.threads)
        self.index.add(X)

    def set_ef(self, ef):
        self.index.hnsw.efSearch = ef

    def search(self, Q, k, threads=1):
        self.faiss.omp_set_num_threads(threads)
        return self.index.search(Q, k)[1]


ENGINES = {"hnswlib": Hnswlib, "faiss": Faiss}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=ENGINES, required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--efc", type=int, default=200)
    ap.add_argument("--ef", default="10,16,24,32,48,64,96,128,192,256,384,512")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--build-threads", type=int, default=1)
    ap.add_argument("--mt-threads", type=int, default=8)
    ap.add_argument("--build-only", action="store_true")
    args = ap.parse_args()

    d = HERE / "data" / args.dataset
    meta = json.loads((d / "meta.json").read_text())
    X = read_bin(d / "base.fbin", np.float32)
    Q = read_bin(d / "query.fbin", np.float32)
    gt = read_bin(d / "gt.ibin", np.int32)[:, : args.k]

    eng = ENGINES[args.engine](X.shape[1], meta["metric"], args.M, args.efc, args.build_threads)
    before = rss_mb()
    t0 = time.perf_counter()
    eng.build(X)
    build_s = time.perf_counter() - t0
    out = {"engine": args.engine, "n": len(X), "d": X.shape[1], "nq": len(Q),
           "metric": meta["metric"], "M": args.M, "ef_construction": args.efc,
           "build_threads": args.build_threads, "build_s": build_s,
           "index_rss_mb": rss_mb() - before,
           "peak_rss_mb": psutil.Process().memory_info().rss / 2**20, "runs": []}
    print(f"  built in {build_s:.1f}s", file=sys.stderr)
    if args.build_only:
        print(json.dumps(out))
        return

    for ef in map(int, args.ef.split(",")):
        eng.set_ef(ef)
        lat = np.empty(len(Q))
        found = np.empty((len(Q), args.k), dtype=np.int64)
        t0 = time.perf_counter()
        for i in range(len(Q)):
            s = time.perf_counter()
            found[i] = eng.search(Q[i: i + 1], args.k)[0]
            lat[i] = (time.perf_counter() - s) * 1e6
        wall = time.perf_counter() - t0
        recall = np.mean([len(set(found[i]) & set(gt[i])) / args.k for i in range(len(Q))])

        t0 = time.perf_counter()
        eng.search(Q, args.k, threads=1)
        qps_batch = len(Q) / (time.perf_counter() - t0)
        t0 = time.perf_counter()
        eng.search(Q, args.k, threads=args.mt_threads)
        qps_mt = len(Q) / (time.perf_counter() - t0)

        out["runs"].append({"ef_search": ef, "recall": float(recall), "qps": len(Q) / wall,
                            "qps_batch": qps_batch, "mean_us": float(lat.mean()),
                            "p50_us": float(np.percentile(lat, 50)),
                            "p95_us": float(np.percentile(lat, 95)),
                            "p99_us": float(np.percentile(lat, 99)),
                            "qps_threads": args.mt_threads, "qps_mt": qps_mt})
        print(f"  ef={ef} recall={recall:.4f} qps={len(Q) / wall:.0f}", file=sys.stderr)
    print(json.dumps(out))


if __name__ == "__main__":
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    main()
