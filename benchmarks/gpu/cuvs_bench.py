"""
Reference: NVIDIA cuVS CAGRA and cuVS brute force on the same files gpubench reads.

    pip install --extra-index-url https://pypi.nvidia.com cuvs-cu12 numpy
    python cuvs_bench.py --dir ../ann/data/glove-100-angular-100000
"""

import argparse
import json
import time
from pathlib import Path

import cupy as cp
import numpy as np
from cuvs.neighbors import brute_force, cagra


def read_bin(path, dtype):
    with open(path, "rb") as f:
        n, d = np.fromfile(f, dtype=np.int32, count=2)
        return np.fromfile(f, dtype=dtype, count=n * d).reshape(n, d)


def recall(found, gt, k):
    return float(np.mean([len(set(found[i, :k]) & set(gt[i, :k])) / k for i in range(len(gt))]))


def timed(fn):
    fn()  # warm-up
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    out = fn()
    cp.cuda.Device().synchronize()
    return out, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--itopk", default="32,64,128,256")
    ap.add_argument("--degree", type=int, default=32)
    args = ap.parse_args()
    d = Path(args.dir)
    X = cp.asarray(read_bin(d / "base.fbin", np.float32))
    Q = cp.asarray(read_bin(d / "query.fbin", np.float32))
    gt = read_bin(d / "gt.ibin", np.int32)
    k = args.k
    out = {"n": int(X.shape[0]), "d": int(X.shape[1]), "nq": int(Q.shape[0]), "k": k,
           "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(), "runs": []}

    bf = brute_force.build(X, metric="inner_product")
    (_, idx), s = timed(lambda: brute_force.search(bf, Q, k))
    out["runs"].append({"engine": "cuvs brute_force", "recall": recall(cp.asnumpy(idx), gt, k),
                        "qps": len(Q) / s})

    t0 = time.perf_counter()
    index = cagra.build(cagra.IndexParams(metric="inner_product",
                                          graph_degree=args.degree,
                                          intermediate_graph_degree=2 * args.degree), X)
    cp.cuda.Device().synchronize()
    build_s = time.perf_counter() - t0
    for itopk in map(int, args.itopk.split(",")):
        sp = cagra.SearchParams(itopk_size=itopk)
        (_, idx), s = timed(lambda: cagra.search(sp, index, Q, k))
        out["runs"].append({"engine": "cuvs cagra", "graph_degree": args.degree, "itopk": itopk,
                            "build_s": build_s, "recall": recall(cp.asnumpy(idx), gt, k),
                            "qps": len(Q) / s})
    print(json.dumps(out))


if __name__ == "__main__":
    main()
