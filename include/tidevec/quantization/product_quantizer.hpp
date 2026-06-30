#pragma once
// ================================================================
// ProductQuantizer — lossy vector compression for billion-scale
//
// Splits a D-dimensional vector into M subspaces of D/M dims each.
// Each subspace is quantized to one of K=256 centroids (1 byte).
// Result: D*4 bytes → M bytes  (e.g. 768-dim → 96 bytes at M=96)
//
// Used in DiskANN-style search:
//   · Full-precision vectors stored on SSD
//   · Compressed PQ codes kept in RAM (~250GB for 1B vectors at M=256)
//   · PQ codes used to rank candidates; top-K re-scored from SSD
// ================================================================

#include <tidevec/core/metrics.hpp>

#include <vector>
#include <array>
#include <cstdint>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <fstream>
#include <algorithm>
#include <numeric>

namespace tidevec {

static constexpr int PQ_K = 256;  // 256 centroids → 1 byte per subspace

class ProductQuantizer {
public:
    // m_subspaces: number of subvector splits (must divide dim evenly)
    // Training: pass ~100K sample vectors
    ProductQuantizer() = default;

    explicit ProductQuantizer(std::size_t dim, std::size_t m_subspaces = 64)
        : dim_(dim), M_(m_subspaces)
    {
        if (dim % M_ != 0)
            throw std::invalid_argument(
                "dim must be divisible by m_subspaces");
        sub_dim_ = dim_ / M_;
        codebooks_.resize(M_, std::vector<std::vector<float>>(
            PQ_K, std::vector<float>(sub_dim_, 0.0f)));
    }

    // ----- Training: k-means per subspace -------------------------
    void train(const std::vector<std::vector<float>>& samples,
               int n_iter = 25, unsigned seed = 42) {
        if (samples.empty()) return;
        if (samples[0].size() != dim_)
            throw std::invalid_argument("Training sample dim mismatch");

        std::mt19937 rng(seed);

        for (std::size_t m = 0; m < M_; ++m) {
            // Extract subvectors for this subspace
            std::vector<std::vector<float>> sub(samples.size(),
                                                std::vector<float>(sub_dim_));
            for (std::size_t i = 0; i < samples.size(); ++i)
                for (std::size_t j = 0; j < sub_dim_; ++j)
                    sub[i][j] = samples[i][m * sub_dim_ + j];

            // Init centroids: random sample
            std::vector<std::size_t> indices(samples.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), rng);

            int k = std::min(PQ_K, static_cast<int>(samples.size()));
            for (int c = 0; c < k; ++c)
                codebooks_[m][c] = sub[indices[c]];

            // k-means iterations
            std::vector<int> assignments(sub.size());
            for (int iter = 0; iter < n_iter; ++iter) {
                // Assign
                for (std::size_t i = 0; i < sub.size(); ++i) {
                    float best = std::numeric_limits<float>::max();
                    int   best_c = 0;
                    for (int c = 0; c < k; ++c) {
                        float d = l2_squared(sub[i].data(),
                                             codebooks_[m][c].data(),
                                             sub_dim_);
                        if (d < best) { best = d; best_c = c; }
                    }
                    assignments[i] = best_c;
                }
                // Update centroids
                std::vector<std::vector<float>> new_c(k,
                    std::vector<float>(sub_dim_, 0.0f));
                std::vector<int> counts(k, 0);
                for (std::size_t i = 0; i < sub.size(); ++i) {
                    int c = assignments[i];
                    ++counts[c];
                    for (std::size_t j = 0; j < sub_dim_; ++j)
                        new_c[c][j] += sub[i][j];
                }
                for (int c = 0; c < k; ++c)
                    if (counts[c] > 0) {
                        for (auto& v : new_c[c]) v /= counts[c];
                        codebooks_[m][c] = new_c[c];
                    }
            }
        }
        trained_ = true;
    }

    // ----- Encode: vector → M bytes (PQ code) --------------------
    std::vector<uint8_t> encode(const std::vector<float>& vec) const {
        if (!trained_) throw std::runtime_error("PQ not trained");
        if (vec.size() != dim_) throw std::invalid_argument("Dim mismatch");

        std::vector<uint8_t> code(M_);
        for (std::size_t m = 0; m < M_; ++m) {
            float best = std::numeric_limits<float>::max();
            uint8_t best_c = 0;
            for (int c = 0; c < PQ_K; ++c) {
                float d = l2_squared(vec.data() + m * sub_dim_,
                                     codebooks_[m][c].data(), sub_dim_);
                if (d < best) { best = d; best_c = static_cast<uint8_t>(c); }
            }
            code[m] = best_c;
        }
        return code;
    }

    // ----- Decode: PQ code → approximate vector ------------------
    std::vector<float> decode(const std::vector<uint8_t>& code) const {
        std::vector<float> vec(dim_);
        for (std::size_t m = 0; m < M_; ++m)
            for (std::size_t j = 0; j < sub_dim_; ++j)
                vec[m * sub_dim_ + j] = codebooks_[m][code[m]][j];
        return vec;
    }

    // ----- Asymmetric Distance Computation (ADC) -----------------
    // Pre-compute distance tables from query to all centroids.
    // Then approximate distance(query, encoded_vec) in O(M) not O(D).
    struct DistanceTable {
        // table[m][c] = distance from query subvector m to centroid c
        std::vector<std::array<float, PQ_K>> table;
    };

    DistanceTable precompute_adc(const std::vector<float>& query) const {
        DistanceTable dt;
        dt.table.resize(M_);
        for (std::size_t m = 0; m < M_; ++m)
            for (int c = 0; c < PQ_K; ++c)
                dt.table[m][c] = l2_squared(
                    query.data() + m * sub_dim_,
                    codebooks_[m][c].data(), sub_dim_);
        return dt;
    }

    float adc_distance(const DistanceTable& dt,
                       const std::vector<uint8_t>& code) const {
        float dist = 0.0f;
        for (std::size_t m = 0; m < M_; ++m)
            dist += dt.table[m][code[m]];
        return dist;
    }

    // ----- Persistence -------------------------------------------
    void save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(&dim_),    sizeof(dim_));
        f.write(reinterpret_cast<const char*>(&M_),      sizeof(M_));
        f.write(reinterpret_cast<const char*>(&sub_dim_),sizeof(sub_dim_));
        for (auto& cb : codebooks_)
            for (auto& centroid : cb)
                f.write(reinterpret_cast<const char*>(centroid.data()),
                        sub_dim_ * sizeof(float));
    }

    void load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        f.read(reinterpret_cast<char*>(&dim_),    sizeof(dim_));
        f.read(reinterpret_cast<char*>(&M_),      sizeof(M_));
        f.read(reinterpret_cast<char*>(&sub_dim_),sizeof(sub_dim_));
        codebooks_.resize(M_, std::vector<std::vector<float>>(
            PQ_K, std::vector<float>(sub_dim_)));
        for (auto& cb : codebooks_)
            for (auto& centroid : cb)
                f.read(reinterpret_cast<char*>(centroid.data()),
                       sub_dim_ * sizeof(float));
        trained_ = true;
    }

    // ----- Accessors ---------------------------------------------
    std::size_t dim()        const { return dim_; }
    std::size_t n_subspaces()const { return M_; }
    std::size_t code_size()  const { return M_; }  // bytes per vector
    bool        is_trained() const { return trained_; }

    // Memory estimate for N vectors
    std::size_t memory_bytes(std::size_t n_vectors) const {
        return n_vectors * M_;  // M bytes per vector
    }

private:
    std::size_t dim_ = 0, M_ = 0, sub_dim_ = 0;
    bool trained_ = false;
    // codebooks_[subspace][centroid][dimension]
    std::vector<std::vector<std::vector<float>>> codebooks_;
};

} // namespace tidevec
