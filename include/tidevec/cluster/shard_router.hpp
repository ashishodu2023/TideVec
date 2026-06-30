#pragma once
// ================================================================
// ShardRouter — horizontal sharding for 1B+ vector search
//
// Strategy:
//   · Consistent hashing: vector_id → shard (stable across rebalance)
//   · Scatter-gather query: broadcast to all shards, merge top-K
//   · Shard = Collection instance owning ~1B/N vectors
//
// At 1B vectors, 10 shards each hold ~100M vectors.
// Each shard runs a TVIndex over its slice → fits in ~16GB RAM per shard
// (HNSW graph: 100M * 16 * 4 * 4 ≈ 25.6GB; use PQ for in-RAM graph).
//
// For a single process (embedded mode), shards are in-process.
// For distributed mode (v0.3), shards are remote gRPC nodes.
// ================================================================

#include <tidevec/core/collection.hpp>

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <algorithm>
#include <functional>
#include <thread>
#include <future>
#include <numeric>
#include <shared_mutex>

namespace tidevec {

// ---- Consistent hash ring helper --------------------------------
inline std::size_t hash_to_shard(const std::string& key, std::size_t n_shards) {
    // FNV-1a
    uint64_t h = 14695981039346656037ULL;
    for (char c : key) {
        h ^= static_cast<uint8_t>(c);
        h *= 1099511628211ULL;
    }
    return static_cast<std::size_t>(h % n_shards);
}

// ================================================================
// ShardRouter — manages N in-process Collection shards
// ================================================================
class ShardRouter {
public:
    struct Config {
        std::string collection_name;
        std::size_t n_shards       = 8;     // 8 shards for ~800M vectors
        Collection::Config shard_cfg;        // template config per shard
        bool parallel_search       = true;   // scatter-gather via threads
    };

    explicit ShardRouter(Config cfg) : cfg_(std::move(cfg)) {
        shards_.reserve(cfg_.n_shards);
        for (std::size_t i = 0; i < cfg_.n_shards; ++i) {
            auto shard_cfg = cfg_.shard_cfg;
            shard_cfg.name = cfg_.collection_name + "_shard_" + std::to_string(i);
            shards_.push_back(std::make_unique<Collection>(shard_cfg));
        }
    }

    // ------ Write (route by id) ----------------------------------

    void upsert(const CortexVector& vec) {
        std::size_t shard_idx = hash_to_shard(vec.id, cfg_.n_shards);
        shards_[shard_idx]->upsert(vec);
    }

    bool remove(const std::string& id) {
        std::size_t shard_idx = hash_to_shard(id, cfg_.n_shards);
        return shards_[shard_idx]->remove(id);
    }

    void add_edge(const std::string& src, const std::string& tgt,
                  EdgeType type, float weight = 1.0f) {
        // Edges stored on the src vector's shard
        std::size_t shard_idx = hash_to_shard(src, cfg_.n_shards);
        shards_[shard_idx]->add_edge(src, tgt, type, weight);
    }

    // ------ Search (scatter-gather) ------------------------------
    // Broadcasts query to all N shards in parallel,
    // collects top-K from each, merges and re-sorts globally.

    std::vector<SearchResult> search(const std::vector<float>& query,
                                     const QueryOptions& opts) const {
        int global_top_k = opts.top_k;
        // Each shard returns top_k results; we merge all
        QueryOptions shard_opts = opts;
        shard_opts.top_k = global_top_k * 2;  // over-fetch per shard

        if (cfg_.parallel_search) {
            return _parallel_search(query, shard_opts, global_top_k);
        } else {
            return _serial_search(query, shard_opts, global_top_k);
        }
    }

    // ------ Stats ------------------------------------------------

    std::size_t total_size() const {
        std::size_t n = 0;
        for (const auto& s : shards_) n += s->size();
        return n;
    }

    std::size_t n_shards() const { return cfg_.n_shards; }

    std::size_t shard_size(std::size_t i) const {
        if (i >= shards_.size()) return 0;
        return shards_[i]->size();
    }

    // Which shard owns a given id?
    std::size_t owner_shard(const std::string& id) const {
        return hash_to_shard(id, cfg_.n_shards);
    }

    void set_temporal_config(const TemporalConfig& cfg) {
        for (auto& s : shards_) s->set_temporal_config(cfg);
    }

private:
    std::vector<SearchResult>
    _parallel_search(const std::vector<float>& query,
                     const QueryOptions& shard_opts,
                     int global_top_k) const {
        // Launch one future per shard
        std::vector<std::future<std::vector<SearchResult>>> futures;
        futures.reserve(cfg_.n_shards);

        for (const auto& shard : shards_) {
            futures.push_back(std::async(std::launch::async,
                [&shard, &query, &shard_opts]() {
                    return shard->search(query, shard_opts);
                }));
        }

        // Collect and merge
        std::vector<SearchResult> all;
        all.reserve(cfg_.n_shards * shard_opts.top_k);
        for (auto& f : futures) {
            auto res = f.get();
            all.insert(all.end(), res.begin(), res.end());
        }
        return _merge_top_k(std::move(all), global_top_k);
    }

    std::vector<SearchResult>
    _serial_search(const std::vector<float>& query,
                   const QueryOptions& shard_opts,
                   int global_top_k) const {
        std::vector<SearchResult> all;
        for (const auto& shard : shards_) {
            auto res = shard->search(query, shard_opts);
            all.insert(all.end(), res.begin(), res.end());
        }
        return _merge_top_k(std::move(all), global_top_k);
    }

    static std::vector<SearchResult>
    _merge_top_k(std::vector<SearchResult> all, int k) {
        // Partial sort: O(N log k) not O(N log N)
        int take = std::min(k, static_cast<int>(all.size()));
        std::partial_sort(all.begin(), all.begin() + take, all.end(),
            [](const SearchResult& a, const SearchResult& b){
                return a.score > b.score;
            });
        all.resize(take);
        return all;
    }

    Config cfg_;
    std::vector<std::unique_ptr<Collection>> shards_;
};

} // namespace tidevec
