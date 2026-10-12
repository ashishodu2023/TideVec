#!/usr/bin/env bash
# Full ANN benchmark: SIFT-1M and GloVe-100 on TideVec, hnswlib and FAISS HNSW.
# Usage (from anywhere, with the bench venv's python on PATH):
#   benchmarks/ann/run_all.sh [dataset ...]
set -euo pipefail
cd "$(dirname "$0")"
root="$(cd ../.. && pwd)"
build="$root/build-bench/ann"
datasets=("$@")
[ ${#datasets[@]} -eq 0 ] && datasets=(sift-128-euclidean glove-100-angular)
EF=10,16,24,32,48,64,96,128,192,256,384,512
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
mkdir -p results logs

cmake -S . -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build "$build"

load() { echo "$(date -Is) $1 $(cut -d' ' -f1-3 /proc/loadavg)" >> logs/load.log; }

for ds in "${datasets[@]}"; do
  [ -f "data/$ds/base.fbin" ] || python prepare.py "$ds"
  metric=$(python -c "import json;print(json.load(open('data/$ds/meta.json'))['metric'])")
  load "start tidevec $ds"
  "$build/tvbench" --dir "data/$ds" --metric "$metric" --ef $EF --threads 8 \
      > "results/tidevec_$ds.json" 2> "logs/tidevec_$ds.log"
  for eng in hnswlib faiss; do
    load "start $eng $ds"
    python run_refs.py --engine $eng --dataset "$ds" --ef $EF \
        > "results/${eng}_$ds.json" 2> "logs/${eng}_$ds.log"
    python run_refs.py --engine $eng --dataset "$ds" --build-threads 16 --build-only \
        > "results/${eng}_${ds}_build16.json" 2>> "logs/${eng}_$ds.log"
  done
  load "done $ds"
done
python report.py > /dev/null
