#pragma once

#include <tidevec/core/cortex_vector.hpp>
#include <tidevec/core/metrics.hpp>
#include <tidevec/core/temporal_scorer.hpp>

#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <shared_mutex>
#include <mutex>
#include <stdexcept>

namespace tidevec {

// ------------------------------------------------------------------
// FlatIndex — O(N·D) exact nearest-neighbour search
//
// Role: ground truth baseline + small-collection (<100K) use case.
// Every query scores ALL stored vectors — 100% recall guaranteed.
// The temporal scorer is applied after similarity to blend scores.
// ------------------------------------------------------------------
class FlatIndex {
public:
    explicit FlatIndex(std::size_t expected_dim = 0,
                       Metric metric = Metric::COSINE,
                       TemporalConfig tcfg = {})
        : expected_dim_(expected_dim)
        , metric_(metric)
        , scorer_(tcfg)
    {}

    // ------ write path --------------------------------------------

    void insert(const CortexVector& vec) {
        if (vec.embedding.empty())
            throw std::invalid_argument("Cannot insert empty embedding");
        if (expected_dim_ == 0) expected_dim_ = vec.dim();
        if (vec.dim() != expected_dim_)
            throw std::invalid_argument("Dimension mismatch: expected "
                + std::to_string(expected_dim_) + " got "
                + std::to_string(vec.dim()));

        std::unique_lock lock(mutex_);
        id_to_idx_[vec.id] = vectors_.size();
        vectors_.push_back(vec);
    }

    void upsert(const CortexVector& vec) {
        std::unique_lock lock(mutex_);
        auto it = id_to_idx_.find(vec.id);
        if (it != id_to_idx_.end()) {
            vectors_[it->second] = vec;
        } else {
            lock.unlock();
            insert(vec);
        }
    }

    bool remove(const std::string& id) {
        std::unique_lock lock(mutex_);
        auto it = id_to_idx_.find(id);
        if (it == id_to_idx_.end()) return false;
        // Tombstone: mark as deleted (lazy deletion)
        vectors_[it->second].payload["__deleted__"] = "1";
        id_to_idx_.erase(it);
        return true;
    }

    // ------ read path ---------------------------------------------

    std::vector<SearchResult> search(const std::vector<float>& query,
                                     const QueryOptions& opts) const {
        std::shared_lock lock(mutex_);

        if (vectors_.empty()) return {};
        if (query.size() != expected_dim_)
            throw std::invalid_argument("Query dimension mismatch");

        Timestamp qt = now_ms();
        Metric m = parse_metric(opts.metric);

        struct Candidate {
            std::size_t idx;
            float final_score;
            TemporalScorer::Score ts;
        };

        std::vector<Candidate> candidates;
        candidates.reserve(vectors_.size());

        for (std::size_t i = 0; i < vectors_.size(); ++i) {
            const auto& v = vectors_[i];
            if (v.payload.count("__deleted__")) continue;
            if (!v.is_valid_at(qt)) continue;
            if (scorer_.is_hard_excluded(v, qt)) continue;

            // Apply payload filter (simple key=value)
            if (!opts.filter.empty() && !matches_filter(v, opts.filter)) continue;

            float sim = compute_similarity(query, v.embedding, m);
            auto  ts  = scorer_.score(v, sim, qt);

            candidates.push_back({i, ts.final_score, ts});
        }

        // Sort descending by final_score
        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate& a, const Candidate& b){
                      return a.final_score > b.final_score;
                  });

        // Build results
        int k = std::min(opts.top_k, static_cast<int>(candidates.size()));
        std::vector<SearchResult> results;
        results.reserve(k);

        for (int i = 0; i < k; ++i) {
            const auto& c = candidates[i];
            const auto& v = vectors_[c.idx];
            SearchResult r;
            r.id              = v.id;
            r.score           = c.final_score;
            r.vector_score    = c.ts.vector_score;
            r.temporal_score  = c.ts.temporal_weight;
            r.payload         = v.payload;
            r.created_at      = v.created_at;
            if (opts.include_staleness_warnings) {
                r.staleness_warning = c.ts.staleness_warning;
                r.staleness_reason  = c.ts.staleness_reason;
            }
            results.push_back(std::move(r));
        }
        return results;
    }

    // ------ utility -----------------------------------------------

    std::size_t size() const {
        std::shared_lock lock(mutex_);
        std::size_t n = 0;
        for (const auto& v : vectors_)
            if (!v.payload.count("__deleted__")) ++n;
        return n;
    }

    std::size_t dim() const { return expected_dim_; }

    bool contains(const std::string& id) const {
        std::shared_lock lock(mutex_);
        return id_to_idx_.count(id) > 0;
    }

    const CortexVector* get(const std::string& id) const {
        std::shared_lock lock(mutex_);
        auto it = id_to_idx_.find(id);
        if (it == id_to_idx_.end()) return nullptr;
        return &vectors_[it->second];
    }

    void set_temporal_config(const TemporalConfig& cfg) {
        scorer_.set_config(cfg);
    }

private:
    // Simple filter: "key=value" format
    static bool matches_filter(const CortexVector& v, const std::string& filter) {
        auto eq = filter.find('=');
        if (eq == std::string::npos) return true;
        std::string key = filter.substr(0, eq);
        std::string val = filter.substr(eq + 1);
        auto it = v.payload.find(key);
        return (it != v.payload.end() && it->second == val);
    }

    std::size_t expected_dim_;
    Metric metric_;
    TemporalScorer scorer_;

    std::vector<CortexVector> vectors_;
    std::unordered_map<std::string, std::size_t> id_to_idx_;

    mutable std::shared_mutex mutex_;
};

} // namespace tidevec
