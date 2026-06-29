#pragma once

#include <vector>
#include <cmath>
#include <stdexcept>
#include <string>

namespace cortexdb {

// ------------------------------------------------------------------
// Distance / similarity functions
// All operate on raw float spans for SIMD auto-vectorisation.
// Compiler flags: -O3 -march=native enable AVX2/AVX512 on supported hardware.
// ------------------------------------------------------------------

inline float dot_product(const float* a, const float* b, std::size_t n) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

inline float l2_squared(const float* a, const float* b, std::size_t n) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

inline float l2_distance(const float* a, const float* b, std::size_t n) {
    return std::sqrt(l2_squared(a, b, n));
}

inline float cosine_similarity(const float* a, const float* b, std::size_t n) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    float denom = std::sqrt(na) * std::sqrt(nb);
    if (denom < 1e-10f) return 0.0f;
    return dot / denom;
}

// Overloads for std::vector
inline float cosine_similarity(const std::vector<float>& a,
                                const std::vector<float>& b) {
    if (a.size() != b.size())
        throw std::invalid_argument("Vector dimension mismatch");
    return cosine_similarity(a.data(), b.data(), a.size());
}

inline float l2_distance(const std::vector<float>& a,
                          const std::vector<float>& b) {
    if (a.size() != b.size())
        throw std::invalid_argument("Vector dimension mismatch");
    return l2_distance(a.data(), b.data(), a.size());
}

inline float dot_product(const std::vector<float>& a,
                          const std::vector<float>& b) {
    if (a.size() != b.size())
        throw std::invalid_argument("Vector dimension mismatch");
    return dot_product(a.data(), b.data(), a.size());
}

// ------------------------------------------------------------------
// Unified metric dispatcher
// ------------------------------------------------------------------
enum class Metric { COSINE, L2, DOT };

inline Metric parse_metric(const std::string& s) {
    if (s == "cosine") return Metric::COSINE;
    if (s == "l2")     return Metric::L2;
    if (s == "dot")    return Metric::DOT;
    throw std::invalid_argument("Unknown metric: " + s);
}

// Returns a similarity score in [0,1] direction (higher = more similar)
// For L2 we convert distance to similarity: 1/(1+d)
inline float compute_similarity(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 Metric metric) {
    switch (metric) {
        case Metric::COSINE:
            return (cosine_similarity(a, b) + 1.0f) * 0.5f; // map [-1,1]→[0,1]
        case Metric::L2:
            return 1.0f / (1.0f + l2_distance(a, b));
        case Metric::DOT:
            return dot_product(a, b);
        default:
            return 0.0f;
    }
}

// ------------------------------------------------------------------
// Vector normalisation (in-place, for cosine pre-normalisation)
// ------------------------------------------------------------------
inline void normalise(std::vector<float>& v) {
    float norm = std::sqrt(dot_product(v, v));
    if (norm < 1e-10f) return;
    for (auto& x : v) x /= norm;
}

} // namespace cortexdb
