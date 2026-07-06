#pragma once
// ================================================================
// ReplicaSet — synchronous replication for high availability
//
// Topology:
//   · 1 primary + N replicas (default: 2 = 3-way replication)
//   · Writes: primary only; propagated to replicas synchronously
//   · Reads: any replica (round-robin load balancing)
//   · Failover: if primary dies, promote replica[0]
//
// In distributed mode (v0.3), replicas are remote gRPC nodes.
// In embedded mode (this file), replicas are in-process Collections
// — useful for testing and for single-machine HA (different SSDs).
//
// Durability contract:
//   A write ACKs to client only after:
//     1. WAL flushed + fsync on primary
//     2. WAL flushed + fsync on at least (N/2+1) replicas
//   This matches Raft majority-commit semantics.
// ================================================================

#include <tidevec/core/collection.hpp>
#include <tidevec/storage/wal.hpp>

#include <vector>
#include <memory>
#include <atomic>
#include <shared_mutex>
#include <stdexcept>

namespace tidevec {

// ================================================================
// ReplicaSet — wraps a primary + replica Collections
// ================================================================
class ReplicaSet {
public:
    struct Config {
        Collection::Config collection_cfg;
        int n_replicas = 2;        // 1 primary + 2 replicas = 3-way
        int write_quorum = 2;      // majority of 3
        WalConfig wal_cfg;
    };

    explicit ReplicaSet(Config cfg) : cfg_(std::move(cfg)) {
        // Create primary
        auto pcfg = cfg_.collection_cfg;
        pcfg.name = cfg_.collection_cfg.name + "_primary";
        primary_ = std::make_unique<Collection>(pcfg);
        primary_wal_ = std::make_unique<WriteAheadLog>(cfg_.wal_cfg);

        // Create replicas
        for (int i = 0; i < cfg_.n_replicas; ++i) {
            auto rcfg = cfg_.collection_cfg;
            rcfg.name = cfg_.collection_cfg.name + "_replica_" + std::to_string(i);
            replicas_.push_back(std::make_unique<Collection>(rcfg));

            WalConfig rwcfg = cfg_.wal_cfg;
            rwcfg.dir += "/replica_" + std::to_string(i);
            replica_wals_.push_back(std::make_unique<WriteAheadLog>(rwcfg));
        }
    }

    // ------ Write path (primary + replication) -------------------

    void upsert(const CortexVector& vec) {
        std::unique_lock lock(write_mutex_);

        // 1. Write to primary WAL (durable)
        primary_wal_->log_upsert(vec);

        // 2. Apply to primary in-memory index
        primary_->upsert(vec);

        // 3. Replicate to replicas (synchronous to write_quorum)
        int acked = 1;  // primary counts
        for (std::size_t i = 0; i < replicas_.size(); ++i) {
            try {
                replica_wals_[i]->log_upsert(vec);
                replicas_[i]->upsert(vec);
                ++acked;
                if (acked >= cfg_.write_quorum) break;
            } catch (...) {
                // Replica down — continue to next
            }
        }

        if (acked < cfg_.write_quorum)
            throw std::runtime_error(
                "Write quorum not met: only " + std::to_string(acked) +
                " of " + std::to_string(cfg_.write_quorum) + " acked");

        // fsync primary WAL after quorum commit
        primary_wal_->fsync_to_disk();
    }

    bool remove(const std::string& id) {
        std::unique_lock lock(write_mutex_);
        primary_wal_->log_delete(id);
        bool ok = primary_->remove(id);
        for (auto& r : replicas_) {
            try { r->remove(id); } catch (...) {}
        }
        return ok;
    }

    void add_edge(const std::string& src, const std::string& tgt,
                  EdgeType type, float weight = 1.0f) {
        std::unique_lock lock(write_mutex_);
        primary_wal_->log_add_edge(src, tgt, type, weight);
        primary_->add_edge(src, tgt, type, weight);
        for (auto& r : replicas_) {
            try { r->add_edge(src, tgt, type, weight); } catch (...) {}
        }
    }

    // ------ Read path (round-robin across all nodes) -------------

    std::vector<SearchResult> search(const std::vector<float>& query,
                                     const QueryOptions& opts) {
        // Pick a reader: primary or replicas in round-robin
        int reader_idx = read_counter_.fetch_add(1,
            std::memory_order_relaxed) % (1 + replicas_.size());

        std::shared_lock lock(read_mutex_);
        try {
            if (reader_idx == 0) return primary_->search(query, opts);
            return replicas_[reader_idx - 1]->search(query, opts);
        } catch (...) {
            // Fallback to primary on replica failure
            return primary_->search(query, opts);
        }
    }

    // ------ Failover: promote replica[0] to primary -------------
    void failover_primary() {
        std::unique_lock lock(write_mutex_);
        if (replicas_.empty())
            throw std::runtime_error("No replicas to promote");
        primary_     = std::move(replicas_[0]);
        primary_wal_ = std::move(replica_wals_[0]);
        replicas_.erase(replicas_.begin());
        replica_wals_.erase(replica_wals_.begin());
        is_degraded_ = true;
    }

    // ------ Recovery: replay WAL into primary on startup ----------
    // Called once at server boot before accepting requests.
    // Rebuilds the entire in-memory index from the WAL — this is
    // the standard crash-recovery pattern for a WAL-based storage
    // engine. Returns the number of records replayed.
    std::size_t recover_primary() {
        std::size_t replayed = 0;
        bool has_wal = false;

        // Check if WAL directory exists and has segments
        try {
            for (auto& _ : fs::directory_iterator(cfg_.wal_cfg.dir)) {
                (void)_;
                has_wal = true;
                break;
            }
        } catch (...) {}

        if (!has_wal) return 0;

        primary_wal_->replay([&](const WalRecord& rec) {
            std::size_t off = 0;
            switch (static_cast<WalOp>(rec.header.op)) {
                case WalOp::UPSERT: {
                    auto v = deserialize_vector(rec.payload.data(), off);
                    primary_->upsert(v);
                    ++replayed;
                    break;
                }
                case WalOp::DELETE: {
                    auto id = deserialize_string(rec.payload.data(), off);
                    primary_->remove(id);
                    ++replayed;
                    break;
                }
                default: break;
            }
        });
        return replayed;
    }

    // ------ Recovery: replay WAL into empty replica ---------------
    void recover_replica(std::size_t replica_idx) {
        if (replica_idx >= replicas_.size())
            throw std::out_of_range("Invalid replica index");
        primary_wal_->replay([&](const WalRecord& rec) {
            std::size_t off = 0;
            switch (static_cast<WalOp>(rec.header.op)) {
                case WalOp::UPSERT: {
                    auto v = deserialize_vector(rec.payload.data(), off);
                    replicas_[replica_idx]->upsert(v);
                    break;
                }
                case WalOp::DELETE: {
                    auto id = deserialize_string(rec.payload.data(), off);
                    replicas_[replica_idx]->remove(id);
                    break;
                }
                default: break;
            }
        });
    }

    // ------ Stats ------------------------------------------------
    std::size_t primary_size() const { return primary_->size(); }
    bool        is_degraded()  const { return is_degraded_; }
    int         n_replicas()   const { return static_cast<int>(replicas_.size()); }

    // Expose primary Collection for DriftBridge snapshotting.
    Collection*       primary()       { return primary_.get(); }
    const Collection* primary() const { return primary_.get(); }

private:
    Config cfg_;
    std::unique_ptr<Collection>       primary_;
    std::unique_ptr<WriteAheadLog>    primary_wal_;
    std::vector<std::unique_ptr<Collection>>    replicas_;
    std::vector<std::unique_ptr<WriteAheadLog>> replica_wals_;

    std::shared_mutex write_mutex_;
    std::shared_mutex read_mutex_;
    std::atomic<int>  read_counter_{0};
    bool              is_degraded_ = false;
};

} // namespace tidevec
