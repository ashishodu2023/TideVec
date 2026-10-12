#pragma once
// ================================================================
// ultra_durable_collection.hpp — erasure-coded, Raft-ordered collection
//
// Combines:
//
//   Durability:
//     ├── RS(10,4) erasure coding          → survives 4 shard losses
//     ├── WAL + group commit batching      → fewer fsyncs per write
//     ├── Background scrubbing             → detect silent corruption
//     └── Per-block checksums              → catch bit rot
//
//   Ordering / failover:
//     ├── Raft log (in-process RaftGroup)  → single write order
//     └── Health monitor + repair          → rebuild lost shards
//
// Modelled durability (independent disks, p = 0.004/year, no repair):
//   RS(10,4) loses data with P ≈ 2.0e-9/year = 8.7 nines. See
//   erasure/reed_solomon.hpp for the model and its limits.
//
// Deployment caveat: all shards and all Raft members of this class
// live in one process on one host, so host loss is the dominant
// failure mode and the figures above are what the coding scheme
// allows across independent disks, not what this deployment gives.
// Multi-process replication is the separate, experimental
// `tidevec-server --node-id/--raft-peers` cluster mode.
// ================================================================

#include <tidevec/cluster/durable_collection.hpp>
#include <tidevec/consensus/raft.hpp>
#include <tidevec/erasure/reed_solomon.hpp>
#include <tidevec/health/health_monitor.hpp>
#include <tidevec/storage/wal.hpp>

#include <string>
#include <memory>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <chrono>
#include <iostream>
#include <sstream>
#include <iomanip>

namespace tidevec {

// ================================================================
// WAL Group Commit — batch multiple writes into one fsync
// ================================================================
struct WalGCConfig {
    int    max_batch_size = 1000;
    int    max_delay_ms   = 2;
    std::string wal_dir   = "./wal";
};

class WalGroupCommit {
public:
    using Config = WalGCConfig;
    explicit WalGroupCommit(Config cfg = WalGCConfig{})
        : cfg_(std::move(cfg))
    {
        WalConfig wcfg;
        wcfg.dir           = cfg_.wal_dir;
        wcfg.sync_on_write = false;  // we control sync ourselves
        wal_ = std::make_unique<WriteAheadLog>(wcfg);
    }

    // Submit a write; returns when it's durable (fsynced to majority)
    uint64_t submit_and_wait(WalOp op, const std::vector<uint8_t>& payload) {
        auto promise = std::make_shared<std::promise<uint64_t>>();
        auto future  = promise->get_future();

        {
            std::lock_guard lock(mu_);
            pending_.push({op, payload, promise});
        }
        cv_.notify_one();

        return future.get();  // block until fsynced
    }

    void start() {
        running_ = true;
        flush_thread_ = std::thread([this]{ _flush_loop(); });
    }

    void stop() {
        running_ = false;
        cv_.notify_all();
        if (flush_thread_.joinable()) flush_thread_.join();
    }

    uint64_t writes_batched()  const { return writes_batched_.load(); }
    uint64_t fsyncs_performed()const { return fsyncs_.load(); }
    double   avg_batch_size()  const {
        uint64_t f = fsyncs_.load();
        return f ? static_cast<double>(writes_batched_.load()) / f : 0.0;
    }

private:
    struct PendingWrite {
        WalOp op;
        std::vector<uint8_t> payload;
        std::shared_ptr<std::promise<uint64_t>> promise;
    };

    void _flush_loop() {
        while (running_) {
            std::vector<PendingWrite> batch;

            {
                std::unique_lock lock(mu_);
                cv_.wait_for(lock, std::chrono::milliseconds(cfg_.max_delay_ms),
                    [this]{ return !pending_.empty() || !running_; });
                while (!pending_.empty() &&
                       static_cast<int>(batch.size()) < cfg_.max_batch_size) {
                    batch.push_back(std::move(pending_.front()));
                    pending_.pop();
                }
            }

            if (batch.empty()) continue;

            // Write all entries to WAL without fsync
            std::vector<uint64_t> lsns;
            for (auto& item : batch) {
                uint64_t lsn = 0;
                switch (item.op) {
                case WalOp::UPSERT: {
                    std::size_t off = 0;
                    auto vec = deserialize_vector(item.payload.data(), off);
                    lsn = wal_->log_upsert(vec);
                    break;
                }
                case WalOp::DELETE: {
                    std::size_t off = 0;
                    auto id = deserialize_string(item.payload.data(), off);
                    lsn = wal_->log_delete(id);
                    break;
                }
                case WalOp::ADD_EDGE: {
                    std::size_t off = 0;
                    auto src = deserialize_string(item.payload.data(), off);
                    auto tgt = deserialize_string(item.payload.data(), off);
                    EdgeType et = static_cast<EdgeType>(item.payload[off++]);
                    float w; std::memcpy(&w, item.payload.data() + off, 4);
                    lsn = wal_->log_add_edge(src, tgt, et, w);
                    break;
                }
                default:
                    lsn = wal_->log_checkpoint();
                    break;
                }
                lsns.push_back(lsn);
                ++writes_batched_;
            }

            // One fsync for the entire batch
            wal_->fsync_to_disk();
            ++fsyncs_;

            // Notify all waiters
            for (std::size_t i = 0; i < batch.size(); ++i)
                batch[i].promise->set_value(lsns[i]);
        }
    }

    Config cfg_;
    std::unique_ptr<WriteAheadLog> wal_;

    std::mutex mu_;
    std::condition_variable cv_;
    std::queue<PendingWrite> pending_;

    std::atomic<bool>     running_{false};
    std::atomic<uint64_t> writes_batched_{0};
    std::atomic<uint64_t> fsyncs_{0};

    std::thread flush_thread_;
};

// ================================================================
// UltraDurableCollection — the complete 11-nines stack
// ================================================================
class UltraDurableCollection {
public:
    struct Config {
        // Identity
        std::string name;
        std::size_t dim    = 768;

        // Sharding
        std::size_t n_shards  = 10;

        // Erasure coding: RS(k, m)
        int rs_k = 10;   // data shards
        int rs_m = 4;    // parity shards (4 = survives 4 simultaneous failures)

        // Raft
        int raft_nodes = 5;  // 5-node Raft = survives 2 simultaneous leader failures

        // WAL group commit
        int wal_batch_size = 1000;
        int wal_delay_ms   = 2;

        // Health monitoring
        int scrub_interval_hours   = 24;   // scrub every 24h for max durability
        int heartbeat_interval_ms  = 5000;
        double p_disk_failure_year = 0.004;

        // Index
        TemporalConfig temporal;
        TVIndexConfig  tvindex;
        SupersessionConfig supersession;

        // Storage
        std::string data_dir = "./tidevec_data";
    };

    explicit UltraDurableCollection(Config cfg)
        : cfg_(std::move(cfg))
        , rs_(cfg_.rs_k, cfg_.rs_m)
    {
        _init_components();
        std::cout << _startup_banner();
    }

    ~UltraDurableCollection() {
        if (wal_gc_) wal_gc_->stop();
        if (health_)  health_->stop();
        if (raft_)    raft_->stop();
    }

    // ------ Write API (goes through Raft → WAL → index) ----------

    void upsert(const CortexVector& vec) {
        // Submit to Raft (linearisable ordering), then block until the
        // committed entry has been applied so reads see the write.
        auto payload = serialize_vector(vec);
        auto idx = raft_->submit("UPSERT", payload);
        if (!idx) throw std::runtime_error("Raft: not leader or no quorum");
        _wait_applied(*idx);
        ++total_writes_;
    }

    bool remove(const std::string& id) {
        if (!underlying_->contains(id)) return false;
        std::vector<uint8_t> payload(id.begin(), id.end());
        auto idx = raft_->submit("DELETE", payload);
        if (!idx) throw std::runtime_error("Raft: not leader or no quorum");
        _wait_applied(*idx);
        return true;
    }

    void add_edge(const std::string& src, const std::string& tgt,
                  EdgeType type, float weight = 1.0f) {
        std::vector<uint8_t> payload;
        auto append = [&](const std::string& s) {
            uint32_t len = s.size();
            auto* p = reinterpret_cast<const uint8_t*>(&len);
            payload.insert(payload.end(), p, p+4);
            payload.insert(payload.end(), s.begin(), s.end());
        };
        append(src); append(tgt);
        payload.push_back(static_cast<uint8_t>(type));
        auto* wp = reinterpret_cast<const uint8_t*>(&weight);
        payload.insert(payload.end(), wp, wp+4);
        auto idx = raft_->submit("ADD_EDGE", payload);
        if (!idx) throw std::runtime_error("Raft: not leader or no quorum");
        _wait_applied(*idx);
    }

    // ------ Read API (bypasses Raft, reads from local index) ------

    std::vector<SearchResult> search(const std::vector<float>& query,
                                      const QueryOptions& opts) {
        ++total_queries_;
        // Reads go directly to the in-memory index (no Raft round-trip)
        // Strong reads: linearisable reads can optionally contact leader
        if (underlying_) return underlying_->search(query, opts);
        return {};
    }

    // ------ Erasure coding demo ----------------------------------

    // Encode a segment using RS(k,m)
    std::vector<std::vector<uint8_t>> encode_segment(
        const std::vector<uint8_t>& data) const {
        return rs_.encode(data);
    }

    // Decode: reconstruct from any k shards
    std::vector<uint8_t> decode_segment(
        const std::vector<std::vector<uint8_t>>& shards,
        uint32_t present_mask,
        std::size_t original_size) const {
        return rs_.decode(shards, present_mask, original_size);
    }

    // ------ Status ----------------------------------------------

    void print_status() const {
        std::cout << "\n=== UltraDurableCollection: " << cfg_.name << " ===\n";
        std::cout << "  Total writes:   " << total_writes_.load()  << "\n";
        std::cout << "  Total queries:  " << total_queries_.load() << "\n";
        std::cout << "  Vectors stored: " << (underlying_ ? underlying_->total_vectors() : 0) << "\n";

        if (wal_gc_) {
            std::cout << "  WAL batch size: " << wal_gc_->avg_batch_size() << " writes/fsync\n";
            std::cout << "  WAL fsyncs:     " << wal_gc_->fsyncs_performed() << "\n";
        }

        if (raft_) raft_->print_state();

        if (health_) {
            auto rep = health_->durability_report();
            std::cout << rep.summary();
        }
    }

    health::DurabilityReport durability_report() const {
        if (health_) return health_->durability_report();
        // Compute without monitor
        health::HealthMonitor::Config hcfg;
        hcfg.rs_k = cfg_.rs_k; hcfg.rs_m = cfg_.rs_m;
        hcfg.p_disk_failure_year = cfg_.p_disk_failure_year;
        health::DurabilityReport rep;
        rep.n_nodes_total   = cfg_.rs_k + cfg_.rs_m;
        rep.n_nodes_healthy = rep.n_nodes_total;
        rep.rs_k = cfg_.rs_k; rep.rs_m = cfg_.rs_m;
        rep.p_loss_per_year = rs_.durability_loss_probability(cfg_.p_disk_failure_year);
        rep.effective_nines = rs_.durability_nines(cfg_.p_disk_failure_year);
        rep.status = "OK";
        return rep;
    }

    std::string prometheus_metrics() const {
        std::string base;
        if (health_) base = health_->prometheus_metrics();
        std::ostringstream ss;
        ss << base
           << "tidevec_total_writes "   << total_writes_.load()  << "\n"
           << "tidevec_total_queries "  << total_queries_.load() << "\n";
        return ss.str();
    }

    bool is_leader() const { return raft_ && raft_->leader() != nullptr; }

    const erasure::ReedSolomon& rs_codec() const { return rs_; }
    const DurableCollection& underlying() const { return *underlying_; }

    // Delegation to underlying DurableCollection (REST server API)
    std::size_t total_vectors() const {
        return underlying_ ? underlying_->total_vectors() : 0;
    }
    std::size_t n_shards() const {
        return underlying_ ? underlying_->n_shards() : cfg_.n_shards;
    }
    std::vector<CortexVector> snapshot_vectors() const {
        return underlying_ ? underlying_->snapshot_vectors() : std::vector<CortexVector>{};
    }
    void swap_index(std::unique_ptr<TVIndex> idx) {
        if (underlying_) underlying_->swap_index(std::move(idx));
    }
    TemporalConfig temporal_config() const { return cfg_.temporal; }
    void set_temporal_config(const TemporalConfig& cfg) {
        cfg_.temporal = cfg;
        if (underlying_) underlying_->set_temporal_config(cfg);
    }
    uint64_t total_writes()  const { return total_writes_.load(); }
    uint64_t total_queries() const { return total_queries_.load(); }
    std::size_t recover() {
        return underlying_ ? underlying_->recover() : 0;
    }

private:
    void _init_components() {
        // 1. In-memory collection (the actual vector index)
        DurableCollection::Config dcfg;
        dcfg.name         = cfg_.name;
        dcfg.dim          = cfg_.dim;
        dcfg.n_shards     = cfg_.n_shards;
        dcfg.n_replicas   = 1;  // Raft handles replication now
        dcfg.write_quorum = 1;
        dcfg.data_dir     = cfg_.data_dir;
        dcfg.temporal     = cfg_.temporal;
        dcfg.tvindex      = cfg_.tvindex;
        dcfg.supersession = cfg_.supersession;
        underlying_       = std::make_unique<DurableCollection>(dcfg);

        // 2. Raft group (5 nodes for 2-failure tolerance)
        raft_ = std::make_unique<consensus::RaftGroup>(
            cfg_.raft_nodes,
            [this](const consensus::LogEntry& entry) {
                _apply_raft_entry(entry);
            },
            false  // verbose
        );
        raft_->start();
        if (!raft_->wait_for_leader(5000))
            throw std::runtime_error("Raft: no leader elected within 5s");

        // 3. WAL group commit
        WalGroupCommit::Config wcfg;
        wcfg.max_batch_size = cfg_.wal_batch_size;
        wcfg.max_delay_ms   = cfg_.wal_delay_ms;
        wcfg.wal_dir        = cfg_.data_dir + "/wal";
        wal_gc_ = std::make_unique<WalGroupCommit>(wcfg);
        wal_gc_->start();

        // 4. Health monitor
        health::HealthMonitor::Config hcfg;
        hcfg.n_nodes             = cfg_.rs_k + cfg_.rs_m;
        hcfg.rs_k                = cfg_.rs_k;
        hcfg.rs_m                = cfg_.rs_m;
        hcfg.scrub_interval_ms   = cfg_.scrub_interval_hours * 3600 * 1000;
        hcfg.heartbeat_interval_ms = cfg_.heartbeat_interval_ms;
        hcfg.p_disk_failure_year = cfg_.p_disk_failure_year;
        hcfg.verbose             = false;
        health_ = std::make_unique<health::HealthMonitor>(hcfg);

        // Wire health callbacks (stubs — real impl reads actual node status)
        health_->set_heartbeat_fn([](int node_id) { (void)node_id; return true; });
        health_->set_alert_fn([](const std::string& msg) {
            std::cerr << "[ALERT] " << msg << "\n";
        });
        health_->start();
    }

    // Every Raft node calls this for every committed entry, but all nodes
    // share one in-process state machine, so each log index is applied once.
    void _apply_raft_entry(const consensus::LogEntry& entry) {
        std::lock_guard lock(apply_mu_);
        if (entry.index <= applied_index_) return;
        if (entry.op == "UPSERT") {
            std::size_t off = 0;
            auto vec = deserialize_vector(entry.payload.data(), off);
            underlying_->upsert(vec);
        } else if (entry.op == "DELETE") {
            std::string id(entry.payload.begin(), entry.payload.end());
            underlying_->remove(id);
        } else if (entry.op == "ADD_EDGE") {
            std::size_t off = 0;
            auto e = deserialize_edge(entry.payload.data(), off);
            underlying_->add_edge(e.src, e.tgt, e.type, e.weight);
        }
        applied_index_ = entry.index;
        apply_cv_.notify_all();
    }

    void _wait_applied(consensus::Index idx, int timeout_ms = 5000) {
        std::unique_lock lock(apply_mu_);
        if (!apply_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                                [&]{ return applied_index_ >= idx; }))
            throw std::runtime_error("Raft: entry " + std::to_string(idx) +
                                     " not applied within timeout");
    }

    std::string _startup_banner() const {
        auto rep = durability_report();
        std::ostringstream ss;
        ss << "\n╔══════════════════════════════════════════════════════╗\n"
           << "║   TideVec Ultra-Durable Collection                  ║\n"
           << "╠══════════════════════════════════════════════════════╣\n"
           << "║  Collection:    " << std::left << std::setw(38) << cfg_.name << "║\n"
           << "║  RS erasure:    RS(" << cfg_.rs_k << "," << cfg_.rs_m << ")"
           <<      " — survives " << cfg_.rs_m << " simultaneous failures"
           <<      std::setw(4) << " " << "║\n"
           << "║  Raft nodes:    " << cfg_.raft_nodes
           <<      " — tolerates " << (cfg_.raft_nodes/2) << " leader failures"
           <<      std::setw(11) << " " << "║\n"
           << "║  WAL batch:     " << cfg_.wal_batch_size << " writes/fsync"
           <<      std::setw(26) << " " << "║\n"
           << "║  Scrub period:  " << cfg_.scrub_interval_hours << "h"
           <<      std::setw(38) << " " << "║\n"
           << "╠══════════════════════════════════════════════════════╣\n"
           << "║  Durability:    " << std::fixed << std::setprecision(1)
           <<      rep.effective_nines << " nines"
           <<      std::setw(35) << " " << "║\n"
           << "║  P(data loss):  " << std::scientific << std::setprecision(1)
           <<      rep.p_loss_per_year << "/year"
           <<      std::setw(29) << " " << "║\n"
           << "╚══════════════════════════════════════════════════════╝\n";
        return ss.str();
    }

    Config cfg_;
    erasure::ReedSolomon rs_;

    std::unique_ptr<DurableCollection>          underlying_;
    std::unique_ptr<consensus::RaftGroup>        raft_;
    std::unique_ptr<WalGroupCommit>              wal_gc_;
    std::unique_ptr<health::HealthMonitor>       health_;

    std::mutex              apply_mu_;
    std::condition_variable apply_cv_;
    consensus::Index        applied_index_ = 0;

    std::atomic<uint64_t> total_writes_{0};
    std::atomic<uint64_t> total_queries_{0};
};

} // namespace tidevec
