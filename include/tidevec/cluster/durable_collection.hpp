#pragma once
// ================================================================
// DurableCollection — billion-scale, durable, highly-available
//
// Combines:
//   ShardRouter  → horizontal scale (N shards × ~100M vectors each)
//   ReplicaSet   → fault tolerance (primary + 2 replicas per shard)
//   WriteAheadLog→ crash durability (fsync before ACK)
//   SegmentStore → disk-resident storage (SSD, mmap'd reads)
//   TVIndex      → temporal HNSW per shard
//
// Capacity targets:
//   1B vectors, dim=768, 10 shards, 3-way replication
//   Storage:  ~2.9TB SSD (raw) per replica = ~8.7TB total
//   RAM:      ~25GB per shard node (HNSW graph) + OS page cache
//   Latency:  <50ms P99 at 95%+ recall (scatter-gather + PQ pre-rank)
//   Writes:   ~100K inserts/sec sustained (WAL batching)
//   QPS:      ~10K queries/sec at 10 shard nodes
// ================================================================

#include <tidevec/cluster/shard_router.hpp>
#include <tidevec/cluster/replica_set.hpp>
#include <tidevec/storage/wal.hpp>
#include <tidevec/observability/retrieval_trace.hpp>
#include <iostream>

#include <string>
#include <memory>
#include <chrono>
#include <atomic>
#include <sstream>

namespace tidevec {

// ================================================================
// DurableCollection
// ================================================================
class DurableCollection {
public:
    struct Config {
        std::string name;
        std::size_t dim              = 768;
        std::size_t n_shards         = 10;
        int         n_replicas       = 2;       // per shard
        int         write_quorum     = 2;
        std::string data_dir         = "./tidevec_data";
        TemporalConfig temporal;
        TVIndexConfig  tvindex;
        bool parallel_search         = true;
    };

    explicit DurableCollection(Config cfg) : cfg_(std::move(cfg)) {
        _build_shards();
    }

    // ------ Crash recovery ---------------------------------------
    // Call once after construction, before accepting requests.
    // Replays all shard WALs to restore in-memory state.
    // Returns total records replayed across all shards.
    std::size_t recover() {
        std::size_t total = 0;
        for (auto& rs : shard_replicas_)
            total += rs->recover_primary();
        if (total > 0)
            std::cout << "  Recovered " << total
                      << " WAL records for collection '" << cfg_.name << "'\n";
        return total;
    }

    // ------ Write API (WAL → shard → replicas) ------------------

    void upsert(const CortexVector& vec) {
        _shard_for(vec.id).upsert(vec);
        ++total_writes_;
    }

    bool remove(const std::string& id) {
        return _shard_for(id).remove(id);
    }

    void add_edge(const std::string& src, const std::string& tgt,
                  EdgeType type, float weight = 1.0f) {
        _shard_for(src).add_edge(src, tgt, type, weight);
    }

    // ------ Read API (scatter-gather across shards) -------------

    std::vector<SearchResult> search(const std::vector<float>& query,
                                     QueryOptions opts,
                                     RetrievalTrace* trace = nullptr) {
        auto t0 = std::chrono::steady_clock::now();

        // Scatter-gather: each shard's ReplicaSet handles replica routing
        int global_k = opts.top_k;
        opts.top_k *= 2;  // over-fetch per shard

        std::vector<SearchResult> all;
        all.reserve(cfg_.n_shards * opts.top_k);

        if (cfg_.parallel_search) {
            std::vector<std::future<std::vector<SearchResult>>> futs;
            futs.reserve(shard_replicas_.size());
            for (auto& rs : shard_replicas_)
                futs.push_back(std::async(std::launch::async,
                    [&rs, &query, &opts]() { return rs->search(query, opts); }));
            for (auto& f : futs) {
                auto r = f.get();
                all.insert(all.end(), r.begin(), r.end());
            }
        } else {
            for (auto& rs : shard_replicas_) {
                auto r = rs->search(query, opts);
                all.insert(all.end(), r.begin(), r.end());
            }
        }

        // Global merge
        int take = std::min(global_k, static_cast<int>(all.size()));
        std::partial_sort(all.begin(), all.begin() + take, all.end(),
            [](const SearchResult& a, const SearchResult& b){
                return a.score > b.score;
            });
        all.resize(take);

        // Fill trace
        if (trace) {
            auto t1 = std::chrono::steady_clock::now();
            trace->strategy = "DURABLE_SHARDED_TVINDEX";
            trace->collection_name = cfg_.name;
            trace->latency_ms = std::chrono::duration<double,std::milli>(t1-t0).count();
            trace->query_id = _new_query_id();
            for (const auto& r : all) {
                if (r.staleness_warning)
                    trace->staleness_warnings.push_back(
                        {r.id, 0, r.temporal_score, r.staleness_reason});
                if (!r.contradicted_by.empty())
                    for (const auto& c : r.contradicted_by)
                        trace->contradiction_alerts.push_back({r.id, c, 0.0f});
            }
        }

        ++total_queries_;
        return all;
    }

    // ------ Stats ------------------------------------------------

    std::size_t total_vectors() const {
        std::size_t n = 0;
        for (const auto& rs : shard_replicas_) n += rs->primary_size();
        return n;
    }

    std::size_t n_shards()   const { return cfg_.n_shards; }

    void set_temporal_config(const TemporalConfig& cfg) {
        cfg_.temporal = cfg;
        for (auto& rs : shard_replicas_) {
            // update primary + replicas via their primary collection
            // (simplified: update config for future queries)
            (void)rs;  // distributed update in v0.3
        }
    }
    uint64_t total_writes()  const { return total_writes_.load(); }
    uint64_t total_queries() const { return total_queries_.load(); }

    void print_stats() const {
        std::cout << "=== DurableCollection: " << cfg_.name << " ===\n";
        std::cout << "  Shards:         " << cfg_.n_shards << "\n";
        std::cout << "  Replicas/shard: " << cfg_.n_replicas << "\n";
        std::cout << "  Total vectors:  " << total_vectors() << "\n";
        std::cout << "  Total writes:   " << total_writes()  << "\n";
        std::cout << "  Total queries:  " << total_queries() << "\n";
        for (std::size_t i = 0; i < shard_replicas_.size(); ++i)
            std::cout << "  Shard " << i << " size:   "
                      << shard_replicas_[i]->primary_size() << "\n";
    }

private:
    void _build_shards() {
        shard_replicas_.reserve(cfg_.n_shards);

        WalConfig wal_cfg;
        wal_cfg.sync_on_write = true;

        for (std::size_t i = 0; i < cfg_.n_shards; ++i) {
            Collection::Config ccfg;
            ccfg.name        = cfg_.name + "_shard_" + std::to_string(i);
            ccfg.dim         = cfg_.dim;
            ccfg.index_type  = IndexType::TVINDEX;
            ccfg.metric      = Metric::COSINE;
            ccfg.temporal    = cfg_.temporal;
            ccfg.tvindex_cfg = cfg_.tvindex;
            ccfg.tvindex_cfg.metric   = Metric::COSINE;
            ccfg.tvindex_cfg.temporal = cfg_.temporal;

            ReplicaSet::Config rscfg;
            rscfg.collection_cfg = ccfg;
            rscfg.n_replicas     = cfg_.n_replicas;
            rscfg.write_quorum   = cfg_.write_quorum;
            rscfg.wal_cfg        = wal_cfg;
            rscfg.wal_cfg.dir    = cfg_.data_dir + "/wal/shard_" + std::to_string(i);

            shard_replicas_.push_back(std::make_unique<ReplicaSet>(rscfg));
        }
    }

    ReplicaSet& _shard_for(const std::string& id) {
        return *shard_replicas_[hash_to_shard(id, cfg_.n_shards)];
    }

    std::string _new_query_id() const {
        return "q_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count());
    }

    Config cfg_;
    std::vector<std::unique_ptr<ReplicaSet>> shard_replicas_;
    std::atomic<uint64_t> total_writes_{0};
    std::atomic<uint64_t> total_queries_{0};
};

} // namespace tidevec
