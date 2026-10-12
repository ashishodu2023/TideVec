# GPU validation: TideVec CUDA engines vs. NVIDIA cuVS

**Verdict: the CUDA path is not usable.** Run on real hardware for the first time (October 2026), the
CAGRA-style graph engine returns essentially random results (recall@10 = 0.003 on GloVe-1.2M), and the
"exact" cuBLAS brute-force engine is not exact (recall@10 = 0.995) and is 4x slower than cuVS brute
force. It should be marked unsupported and no longer advertised until it is rewritten or replaced by
cuVS itself.

Hardware: NVIDIA GeForce RTX 4060 Ti (8 GB, Ada, sm_89), driver 595.99, CUDA 12.8 toolkit (conda-forge).
This is a consumer card, not the A100/H100 the roadmap targets. Absolute QPS will differ on data-centre
GPUs, but the recall numbers are properties of the algorithms and do not depend on the card.

## Results

Dataset: GloVe-100 angular (ann-benchmarks; unit-normalised), 10,000 queries, k = 10, all queries
submitted as one batch (TideVec brute force: batches of 256, because it allocates an
n_queries x n distance matrix per call).

**GloVe-100, 1,183,514 vectors**

| Engine | Setting | Build (s) | Recall@10 | QPS |
|---|---|---:|---:|---:|
| TideVec `GpuBruteForceEngine` (cuBLAS SGEMM + custom top-k) | batch 256 | 0.3 (upload) | 0.9945 | 5,667 |
| cuVS `brute_force` | batch 10,000 | — | **1.0000** | 22,283 |
| TideVec `CagraStyleEngine` | degree 64, itopk 64 | 116 | **0.0031** | 574,413 |
| TideVec `CagraStyleEngine` | degree 64, itopk 256 | 109 | **0.0035** | 106,231 |
| cuVS `cagra` (defaults, not tuned) | degree 32, itopk 64 | 2.2 | 0.6978 | 347,193 |
| cuVS `cagra` | degree 32, itopk 128 | 2.2 | 0.7935 | 183,711 |
| cuVS `cagra` | degree 32, itopk 256 | 2.2 | 0.8601 | 96,712 |

**GloVe-100, first 100,000 vectors** (exact ground truth recomputed)

| Engine | Setting | Build (s) | Recall@10 | QPS |
|---|---|---:|---:|---:|
| TideVec `GpuBruteForceEngine` | batch 256 | 0.05 | 0.9880 | 29,641 |
| cuVS `brute_force` | batch 10,000 | — | 1.0000 | 159,303 |
| TideVec `CagraStyleEngine` | degree 64, itopk 64 / 128 / 256 | 6.4 | 0.012 / 0.011 / 0.013 | 456k / 182k / 87k |
| cuVS `cagra` | degree 32, itopk 64 / 128 / 256 | 4.2 | 0.817 / 0.891 / 0.942 | 722k / 362k / 144k |

Raw output: [`results/`](results/).

## Why (from reading `src/accelerator/gpu_kernels.cu`)

These are observations to guide a fix, not a full audit:

- **Graph build is not NN-Descent or CAGRA.** `nn_descent_kernel` starts from a uniformly random graph
  and, for each node independently, samples random neighbours-of-neighbours and keeps the closest.
  There is no reverse-edge (local join) step, no rank-based pruning or reverse-graph merge that CAGRA
  uses for reachability, and the graph is built with L2 while search uses cosine distance. The result
  is a graph a beam search cannot navigate.
- **Beam-search bookkeeping is broken.** `sorted_insert` shifts `s_beam_dists`/`s_beam_idxs` but not
  `s_visited`, so visited flags end up attached to the wrong candidates. The search starts from a fixed
  node (`N / 2`), not a medoid or random seeds.
- **Top-k for brute force has overlapping shared memory.** In `topk_per_row_kernel` each warp's
  `s_idxs` slice starts at `s_vals + 32*top_k`, which is the next warp's `s_vals` slice, so warps in
  the same block overwrite each other's candidates. That explains recall < 1 for an exact search. The
  kernel also limits k to 32 and scans each row with a single warp, which is why it is slower than
  cuVS.
- `cuda_pq_encode` is an empty stub, so the GPU IVF-PQ path cannot work either.
- `CMakeLists.txt` builds only for `sm_80`; other GPUs need `-arch` changed.

Recommendation: delete `CagraStyleEngine` and the hand-written kernels and, if GPU support is still
wanted, call cuVS (`cuvs::neighbors::cagra`, `brute_force`) through its C/C++ API, with this harness as
the acceptance test (recall must match the cuVS row on the same data).

## Reproduce

Needs an NVIDIA GPU with a working driver (`nvidia-smi`). No system CUDA toolkit is required; both
toolchains below install into gitignored `.venvs/`.

```bash
# 1. data (from benchmarks/ann)
cd benchmarks/ann
python prepare.py glove-100-angular
python prepare.py glove-100-angular --n 100000
cd ../..

# 2. CUDA 12.8 nvcc + cuBLAS via micromamba (the pip nvidia-cuda-nvcc wheel has no nvcc driver)
curl -sSfL https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64 \
     -o micromamba && chmod +x micromamba
MAMBA_ROOT_PREFIX=$PWD/.venvs/mamba ./micromamba create -y -p .venvs/cuda -c conda-forge \
    "cuda-nvcc=12.8" "cuda-cudart-dev=12.8" "libcublas-dev=12.8" "cuda-version=12.8" "gxx=13"

# 3. build the TideVec harness (set -arch for your GPU: sm_80 A100, sm_90 H100, sm_89 Ada)
C=$PWD/.venvs/cuda; mkdir -p build-bench/gpu
$C/bin/nvcc -std=c++20 -O3 -arch=sm_89 -ccbin $C/bin/g++ --extended-lambda -DTIDEVEC_CUDA_ENABLED \
    -Iinclude -I$C/include -I$C/targets/x86_64-linux/include \
    benchmarks/gpu/gpubench.cu src/accelerator/gpu_kernels.cu \
    -L$C/lib -L$C/targets/x86_64-linux/lib -lcublas -o build-bench/gpu/gpubench
export LD_LIBRARY_PATH=$C/lib
D=benchmarks/ann/data/glove-100-angular
build-bench/gpu/gpubench --dir $D --engine brute
build-bench/gpu/gpubench --dir $D --engine cagra --itopk 64,256

# 4. cuVS reference
uv venv .venvs/gpu && uv pip install --python .venvs/gpu/bin/python \
    --extra-index-url https://pypi.nvidia.com cuvs-cu12 cupy-cuda12x numpy
.venvs/gpu/bin/python benchmarks/gpu/cuvs_bench.py --dir $D
```

On an A100/H100, repeat with SIFT-1M normalised to unit length (or Cohere-1M) and report the same
table; acceptance criterion for re-advertising GPU support: recall within 0.01 of cuVS at equal
`itopk`, and QPS within 2x.
