#pragma once
// ================================================================
// accelerated_collection.hpp
//
// Wraps DurableCollection with an AcceleratorDispatcher.
//
// Query path with acceleration:
//
//   search(query, opts)
//     │
//     ├─ batch_size >= gpu_threshold?
//     │   └─ GPU engine (CAGRA/cuBLAS): exact search
//     │       → merge with TVIndex temporal scores
//     │
//     ├─ batch_size >= tpu_threshold?
//     │   └─ TPU engine (XLA matmul): exact search
//     │       → merge with TVIndex temporal scores
//     │
//     └─ CPU path (TVIndex HNSW + CausalEdge)
//
// The GPU/TPU path does EXACT search (no recall loss).
// Temporal scoring is applied POST-SEARCH using the score merge:
//   final_score = α·accel_similarity + β·temporal_weight(created_at)
// ================================================================

#include <tidevec/cluster/durable_collection.hpp>
#include <tidevec/accelerator/dispatcher.hpp>
#include <tidevec/core/temporal_scorer.hpp>

#include <unordered_map>
#include <vector>
#include <string>
#include <shared_mutex>

namespace tidevec {

class AcceleratedCollection {
public:
    struct Config {
        DurableCollection::Config durable;
        accel::AcceleratorDispatcher::Config accel;
        bool use_accel_for_search = true;   // false = always CPU TVIndex
        int  accel_batch_threshold = 1;     // use accel when batch >= this
    };

    explicit AcceleratedCollection(Config cfg)
        : cfg_(std::move(cfg))
        , durable_(cfg_.durable)
        , dispatcher_(cfg_.accel)
    {}

    // ------ Write (delegates to DurableCollection) ---------------
    void upsert(const CortexVector& vec) {
        durable_.upsert(vec);
        // Also add to accelerator engine for GPU/TPU search
        {
            std::unique_lock lock(accel_mutex_);
            accel_id_to_idx_[vec.id] = accel_n_;
            accel_idx_to_id_.push_back(vec.id);
            dispatcher_.add(vec.embedding.data(), 1,
                            static_cast<int64_t>(vec.embedding.size()));
            ++accel_n_;
        }
        // Store metadata for temporal re-scoring
        {
            std::unique_lock lock(meta_mutex_);
            meta_[vec.id] = {vec.created_at, vec.payload, vec.valid_until};
        }
    }

    bool remove(const std::string& id) {
        std::unique_lock lock(accel_mutex_);
        accel_deleted_.insert(id);
        return durable_.remove(id);
    }

    void add_edge(const std::string& src, const std::string& tgt,
                  EdgeType type, float weight = 1.0f) {
        durable_.add_edge(src, tgt, type, weight);
    }

    // ------ Search -----------------------------------------------
    std::vector<SearchResult> search(
        const std::vector<float>& query,
        const QueryOptions& opts,
        RetrievalTrace* trace = nullptr,
        accel::DeviceType device_hint = accel::DeviceType::AUTO)
    {
        // Choose path
        bool use_accel = cfg_.use_accel_for_search
            && (static_cast<int>(1) >= cfg_.accel_batch_threshold);

        if (!use_accel) {
            return durable_.search(query, opts, trace);
        }

        // ---- GPU/TPU accelerated exact search -------------------
        auto t0 = std::chrono::steady_clock::now();

        // Over-fetch: get 4× top_k from accelerator, apply temporal
        // scoring to refine, return top_k
        int fetch_k = std::min(opts.top_k * 4,
                               static_cast<int>(accel_n_));
        if (fetch_k == 0) return {};

        accel::AccelSearchResult accel_res;
        {
            std::shared_lock lock(accel_mutex_);
            accel_res = dispatcher_.search(
                query.data(), 1,
                static_cast<int64_t>(query.size()),
                fetch_k, device_hint);
        }

        // ------ Apply temporal scoring + filter ------------------
        TemporalScorer scorer(cfg_.durable.temporal);
        Timestamp qt = now_ms();
        // Metric used by durable path; accel path uses cosine always
        
        std::vector<SearchResult> candidates;
        candidates.reserve(fetch_k);

        for (int i = 0; i < static_cast<int>(accel_res.indices.size()); ++i) {
            int64_t idx = accel_res.indices[i];
            if (idx < 0 || idx >= static_cast<int64_t>(accel_idx_to_id_.size()))
                continue;

            const std::string& id = accel_idx_to_id_[idx];
            if (accel_deleted_.count(id)) continue;

            // Get metadata for temporal scoring
            VecMeta meta;
            {
                std::shared_lock lock(meta_mutex_);
                auto it = meta_.find(id);
                if (it == meta_.end()) continue;
                meta = it->second;
            }

            // Check validity
            if (meta.valid_until.has_value() && qt > *meta.valid_until) continue;

            // Apply payload filter
            if (!opts.filter.empty() && !_filter_match(meta.payload, opts.filter))
                continue;

            // Accelerator distance → similarity (1 - cosine_dist)
            float accel_sim = 1.0f - accel_res.distances[i];
            // Temporal score
            float tw = scorer.temporal_weight_raw(meta.created_at, qt);
            float alpha = 1.0f - opts.temporal_blend;
            float beta  = opts.temporal_blend;
            float final_score = alpha * accel_sim + beta * tw;

            SearchResult r;
            r.id             = id;
            r.score          = final_score;
            r.vector_score   = accel_sim;
            r.temporal_score = tw;
            r.payload        = meta.payload;
            r.created_at     = meta.created_at;
            if (opts.include_staleness_warnings &&
                tw < cfg_.durable.temporal.staleness_threshold) {
                r.staleness_warning = true;
                r.staleness_reason  = "Temporal score " +
                    std::to_string(tw).substr(0,5) + " below threshold";
            }
            candidates.push_back(std::move(r));
        }

        // Sort by final_score descending, take top_k
        int k = std::min(opts.top_k, static_cast<int>(candidates.size()));
        std::partial_sort(candidates.begin(), candidates.begin() + k,
                          candidates.end(),
            [](const SearchResult& a, const SearchResult& b){
                return a.score > b.score;
            });
        candidates.resize(k);

        // ------ Fill trace ---------------------------------------
        if (trace) {
            auto t1 = std::chrono::steady_clock::now();
            trace->strategy       = "ACCEL_" +
                accel::device_type_str(accel_res.device_used) + "_EXACT";
            trace->collection_name= cfg_.durable.name;
            trace->latency_ms     = std::chrono::duration<double,std::milli>(t1-t0).count();
            trace->query_id       = "q_" + std::to_string(qt);
            for (const auto& r : candidates) {
                if (r.staleness_warning)
                    trace->staleness_warnings.push_back(
                        {r.id, 0, r.temporal_score, r.staleness_reason});
            }
        }

        return candidates;
    }

    // ------ Batch search (native accelerator mode) ---------------
    // For true GPU batch throughput: send B queries in one shot
    std::vector<std::vector<SearchResult>> batch_search(
        const std::vector<std::vector<float>>& queries,
        const QueryOptions& opts,
        accel::DeviceType device_hint = accel::DeviceType::AUTO)
    {
        if (queries.empty()) return {};
        int B   = static_cast<int>(queries.size());
        int dim = static_cast<int>(queries[0].size());
        int k   = opts.top_k;

        // Flatten queries for GPU batch call
        std::vector<float> flat(B * dim);
        for (int q = 0; q < B; ++q)
            std::copy(queries[q].begin(), queries[q].end(),
                      flat.data() + q * dim);

        accel::AccelSearchResult accel_res;
        {
            std::shared_lock lock(accel_mutex_);
            accel_res = dispatcher_.search(flat.data(), B, dim, k * 4,
                                           device_hint);
        }

        TemporalScorer scorer(cfg_.durable.temporal);
        Timestamp qt = now_ms();

        std::vector<std::vector<SearchResult>> all_results(B);
        for (int q = 0; q < B; ++q) {
            auto& results = all_results[q];
            results.reserve(k);
            for (int i = 0; i < k * 4 && static_cast<int>(results.size()) < k; ++i) {
                int64_t idx = accel_res.indices[q * k * 4 + i];
                if (idx < 0 || idx >= static_cast<int64_t>(accel_idx_to_id_.size()))
                    continue;
                const std::string& id = accel_idx_to_id_[idx];
                if (accel_deleted_.count(id)) continue;

                VecMeta meta;
                {
                    std::shared_lock lock(meta_mutex_);
                    auto it = meta_.find(id);
                    if (it == meta_.end()) continue;
                    meta = it->second;
                }
                if (meta.valid_until.has_value() && qt > *meta.valid_until) continue;
                if (!opts.filter.empty() && !_filter_match(meta.payload, opts.filter)) continue;

                float sim = 1.0f - accel_res.distances[q * k * 4 + i];
                float tw  = scorer.temporal_weight_raw(meta.created_at, qt);
                float fs  = (1.0f - opts.temporal_blend) * sim + opts.temporal_blend * tw;

                SearchResult r;
                r.id = id; r.score = fs; r.vector_score = sim;
                r.temporal_score = tw; r.created_at = meta.created_at;
                results.push_back(std::move(r));
            }
        }
        return all_results;
    }

    // ------ Stats ------------------------------------------------
    std::size_t total_vectors() const { return durable_.total_vectors(); }
    std::size_t accel_n()       const { return accel_n_; }
    bool        gpu_available() const { return dispatcher_.gpu_available(); }
    bool        tpu_available() const { return dispatcher_.tpu_available(); }

    accel::AcceleratorDispatcher::Stats accel_stats() const {
        return dispatcher_.stats();
    }

    void print_device_info() const {
        dispatcher_.system_devices().print();
    }

    void set_temporal_config(const TemporalConfig& cfg) {
        cfg_.durable.temporal = cfg;
        durable_.set_temporal_config(cfg);
    }

    std::vector<CortexVector> snapshot_vectors() const {
        return durable_.snapshot_vectors();
    }

    void swap_index(std::unique_ptr<TVIndex> idx) {
        durable_.swap_index(std::move(idx));
    }

    TemporalConfig temporal_config() const {
        return durable_.temporal_config();
    }

    std::size_t recover() { return durable_.recover(); }
    std::size_t n_shards() const { return durable_.n_shards(); }
    uint64_t total_writes()  const { return durable_.total_writes(); }
    uint64_t total_queries() const { return durable_.total_queries(); }

private:
    struct VecMeta {
        Timestamp created_at;
        std::unordered_map<std::string,std::string> payload;
        std::optional<Timestamp> valid_until;
    };

    static bool _filter_match(
        const std::unordered_map<std::string,std::string>& payload,
        const std::string& filter)
    {
        auto eq = filter.find('=');
        if (eq == std::string::npos) return true;
        auto key = filter.substr(0, eq);
        auto val = filter.substr(eq + 1);
        auto it = payload.find(key);
        return (it != payload.end() && it->second == val);
    }

    Config cfg_;
    DurableCollection durable_;
    accel::AcceleratorDispatcher dispatcher_;

    // Accelerator index state
    std::vector<std::string> accel_idx_to_id_;
    std::unordered_map<std::string, int64_t> accel_id_to_idx_;
    std::unordered_set<std::string> accel_deleted_;
    int64_t accel_n_ = 0;
    mutable std::shared_mutex accel_mutex_;

    // Vector metadata for temporal scoring
    std::unordered_map<std::string, VecMeta> meta_;
    mutable std::shared_mutex meta_mutex_;
};

} // namespace tidevec
