"""
Download an ann-benchmarks dataset and convert it to the flat binary files
tvbench and run_refs.py read.

    python prepare.py sift-128-euclidean            # full 1M
    python prepare.py glove-100-angular
    python prepare.py sift-128-euclidean --n 100000  # subset, exact GT recomputed

Writes data/<name>[-<n>]/{base.fbin,query.fbin,gt.ibin,meta.json}.
Angular datasets are L2-normalised (cosine is unchanged by this).
"""

import argparse
import json
import urllib.request
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).parent
URL = "http://ann-benchmarks.com/{}.hdf5"


def write_fbin(path, a):
    a = np.ascontiguousarray(a, dtype=np.float32)
    with open(path, "wb") as f:
        np.array(a.shape, dtype=np.int32).tofile(f)
        a.tofile(f)


def write_ibin(path, a):
    a = np.ascontiguousarray(a, dtype=np.int32)
    with open(path, "wb") as f:
        np.array(a.shape, dtype=np.int32).tofile(f)
        a.tofile(f)


def exact_gt(base, query, metric, k):
    import faiss
    index = faiss.IndexFlatL2(base.shape[1]) if metric == "l2" else faiss.IndexFlatIP(base.shape[1])
    index.add(base)
    return index.search(query, k)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="e.g. sift-128-euclidean, glove-100-angular")
    ap.add_argument("--n", type=int, default=0, help="use the first N base vectors")
    ap.add_argument("--k", type=int, default=100)
    args = ap.parse_args()

    data = HERE / "data"
    data.mkdir(exist_ok=True)
    h5 = data / f"{args.name}.hdf5"
    if not h5.exists():
        print(f"downloading {URL.format(args.name)}")
        urllib.request.urlretrieve(URL.format(args.name), h5)

    with h5py.File(h5, "r") as f:
        metric = "l2" if f.attrs["distance"] == "euclidean" else "cosine"
        base = np.asarray(f["train"], dtype=np.float32)
        query = np.asarray(f["test"], dtype=np.float32)
        gt = np.asarray(f["neighbors"])[:, : args.k]
    if metric == "cosine":
        base /= np.maximum(np.linalg.norm(base, axis=1, keepdims=True), 1e-12)
        query /= np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    if args.n and args.n < len(base):
        base = base[: args.n]
        gt = exact_gt(base, query, metric, args.k)

    out = data / (args.name + (f"-{args.n}" if args.n else ""))
    out.mkdir(exist_ok=True)
    write_fbin(out / "base.fbin", base)
    write_fbin(out / "query.fbin", query)
    write_ibin(out / "gt.ibin", gt)
    (out / "meta.json").write_text(json.dumps(
        {"dataset": args.name, "n": len(base), "nq": len(query), "d": base.shape[1],
         "metric": metric}))
    print(f"wrote {out}: n={len(base)} nq={len(query)} d={base.shape[1]} metric={metric}")


if __name__ == "__main__":
    main()
