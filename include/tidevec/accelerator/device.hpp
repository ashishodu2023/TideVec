#pragma once
// ================================================================
// accelerator/device.hpp — Hardware capability detection
//
// CortexDB accelerator stack:
//
//   ┌─────────────────────────────────────────────────┐
//   │           AcceleratorDispatcher                 │
//   │  routes queries to best available device        │
//   ├──────────────┬──────────────┬───────────────────┤
//   │  CUDA GPU    │   XLA/TPU    │  CPU (SIMD)       │
//   │  (cuVS-style)│  (matmul)    │  (baseline)       │
//   │  GpuAnnEngine│  TpuAnnEngine│  CpuAnnEngine     │
//   └──────────────┴──────────────┴───────────────────┘
//
// Design principle: ZERO hard deps on CUDA/XLA at compile time.
// Each engine is compiled conditionally:
//   -DCORTEXDB_CUDA_ENABLED   → link against libcuda, libcublas
//   -DCORTEXDB_XLA_ENABLED    → link against libxla
// Without those flags: only CPU path compiles (always works).
//
// Runtime detection: even if compiled with CUDA support,
// gracefully falls back to CPU if no GPU is found.
// ================================================================

#include <cortexdb/core/cortex_vector.hpp>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <immintrin.h>   // AVX2/AVX512 intrinsics

namespace cortexdb {
namespace accel {

// ------ Device types ------------------------------------------
enum class DeviceType {
    CPU   = 0,   // always available, SIMD-accelerated
    GPU   = 1,   // NVIDIA CUDA (cuVS / cuBLAS)
    TPU   = 2,   // Google XLA / Cloud TPU
    AUTO  = 99   // let dispatcher choose best
};

inline std::string device_type_str(DeviceType d) {
    switch (d) {
        case DeviceType::CPU:  return "CPU";
        case DeviceType::GPU:  return "GPU";
        case DeviceType::TPU:  return "TPU";
        case DeviceType::AUTO: return "AUTO";
        default:               return "UNKNOWN";
    }
}

// ------ Device capabilities -----------------------------------
struct DeviceInfo {
    DeviceType type;
    std::string name;
    int    device_id        = 0;
    size_t memory_bytes     = 0;       // total device memory
    size_t free_memory_bytes= 0;
    int    compute_units    = 0;       // CUDA SMs / TPU cores
    float  peak_tflops      = 0.0f;   // FP32 TFLOPS
    bool   available        = false;
    std::string driver_version;

    void print() const {
        std::cout << "  [" << device_type_str(type) << "] "
                  << name
                  << " | " << (memory_bytes / (1024*1024*1024)) << "GB"
                  << " | " << peak_tflops << " TFLOPS"
                  << " | " << (available ? "READY" : "UNAVAILABLE")
                  << "\n";
    }
};

// ------ Accelerated search result (batch) --------------------
struct AccelSearchResult {
    // For N queries, top_k results each:
    // indices[q * top_k + k] = index of k-th nearest for query q
    // distances[q * top_k + k] = distance to k-th nearest for query q
    std::vector<int64_t> indices;
    std::vector<float>   distances;
    int                  n_queries = 0;
    int                  top_k     = 0;
    double               latency_ms = 0.0;
    DeviceType           device_used;

    std::vector<int64_t> result_for(int q) const {
        std::vector<int64_t> r(top_k);
        for (int k = 0; k < top_k; ++k)
            r[k] = indices[q * top_k + k];
        return r;
    }
};

// ------ Abstract engine interface ----------------------------
class AnnEngine {
public:
    virtual ~AnnEngine() = default;

    // Add vectors to the engine's index
    virtual void add(const float* data, int64_t n, int64_t dim) = 0;

    // Batch search: queries[n_q * dim] → results
    virtual AccelSearchResult search(
        const float* queries,
        int64_t n_queries,
        int64_t dim,
        int top_k) = 0;

    // Reset the index
    virtual void reset() = 0;

    virtual DeviceInfo device_info() const = 0;
    virtual DeviceType device_type() const = 0;
};

// ================================================================
// SIMD helpers — AVX2 accelerated distance kernels
// ================================================================
namespace simd {

// Dot product of two float vectors using AVX2 (falls back to scalar)
inline float dot_avx2(const float* a, const float* b, int n) {
#if defined(__AVX2__)
    __m256 sum = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    // Horizontal sum
    __m128 lo  = _mm256_castps256_ps128(sum);
    __m128 hi  = _mm256_extractf128_ps(sum, 1);
    __m128 s   = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    float result = _mm_cvtss_f32(s);
    // Handle tail
    for (; i < n; ++i) result += a[i] * b[i];
    return result;
#else
    float result = 0.0f;
    for (int i = 0; i < n; ++i) result += a[i] * b[i];
    return result;
#endif
}

// L2 squared using AVX2
inline float l2sq_avx2(const float* a, const float* b, int n) {
#if defined(__AVX2__)
    __m256 sum = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 diff = _mm256_sub_ps(
            _mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
    float result = _mm_cvtss_f32(s);
    for (; i < n; ++i) { float d = a[i]-b[i]; result += d*d; }
    return result;
#else
    float result = 0.0f;
    for (int i = 0; i < n; ++i) { float d=a[i]-b[i]; result+=d*d; }
    return result;
#endif
}

// Batch matrix multiply: C[m,k] = A[m,n] * B[n,k]^T  (row-major)
// Computes distances from m queries against n database vectors
// Uses OpenMP + AVX2 for CPU parallelism
inline void batch_matmul_l2(
    const float* queries,  // [m x dim]
    const float* database, // [n x dim]
    float*       out,      // [m x n] output distances
    int m, int n, int dim)
{
    #pragma omp parallel for schedule(dynamic, 64) if(m > 16)
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            out[i * n + j] = l2sq_avx2(
                queries  + i * dim,
                database + j * dim, dim);
        }
    }
}

inline void batch_matmul_dot(
    const float* queries,
    const float* database,
    float*       out,
    int m, int n, int dim)
{
    #pragma omp parallel for schedule(dynamic, 64) if(m > 16)
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            out[i * n + j] = dot_avx2(
                queries  + i * dim,
                database + j * dim, dim);
        }
    }
}

} // namespace simd

} // namespace accel
} // namespace cortexdb
