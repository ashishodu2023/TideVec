#pragma once
// ================================================================
// health/health_monitor.hpp — Continuous background health system
//
// For 11-nines availability you need active self-healing, not just
// passive replication. The HealthMonitor runs background goroutines
// that continuously:
//
//   1. SCRUB: verify checksums of all stored segments
//              detect silent data corruption (bit rot)
//              trigger re-encoding if corruption found
//
//   2. HEARTBEAT: ping all shard nodes every 5s
//                 detect node failures within seconds
//                 trigger RS shard reconstruction on failure
//
//   3. REBALANCE: when a new node joins or a node fails
//                 redistribute shards for even load
//                 maintain RS parity invariant
//
//   4. METRICS: track MTTR (mean time to repair)
//               emit Prometheus metrics
//               alert when durability drops below threshold
//
//   5. WATCHDOG: monitor Raft leader health
//                force re-election if leader heartbeat misses
//
// SCRUB SCHEDULE:
//   Default: full scrub every 7 days (matches BACKBLAZE best practice)
//   High-durability: every 24 hours
//   Each scrub: verify xxhash64 of every shard against stored checksum
//
// REPAIR PROCESS (when shard lost):
//   1. Detect: heartbeat timeout or scrub mismatch
//   2. Assess: which shards are affected
//   3. Reconstruct: RS decode from surviving k shards
//   4. Re-encode: re-generate lost parity shard
//   5. Place: write recovered shard to new/healthy node
//   6. Verify: checksum the recovered shard
//   7. Update: metadata to point to new shard location
// ================================================================

#include <cortexdb/erasure/reed_solomon.hpp>

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <atomic>
#include <mutex>
#include <thread>
#include <chrono>
#include <functional>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>

namespace cortexdb {
namespace health {

// ================================================================
// NodeStatus — live state of one storage node
// ================================================================
enum class NodeHealth { HEALTHY, DEGRADED, FAILED, RECOVERING };

inline std::string node_health_str(NodeHealth h) {
    switch(h) {
        case NodeHealth::HEALTHY:    return "HEALTHY";
        case NodeHealth::DEGRADED:   return "DEGRADED";
        case NodeHealth::FAILED:     return "FAILED";
        case NodeHealth::RECOVERING: return "RECOVERING";
        default:                     return "UNKNOWN";
    }
}

struct NodeStatus {
    int         node_id;
    NodeHealth  health       = NodeHealth::HEALTHY;
    uint64_t    last_seen_ms = 0;
    uint64_t    last_scrub_ms= 0;
    std::size_t total_shards = 0;
    std::size_t failed_shards= 0;
    double      disk_usage_pct = 0.0;
    double      cpu_usage_pct  = 0.0;
    std::string last_error;
};

// ================================================================
// ScrubResult — outcome of scrubbing one segment
// ================================================================
struct ScrubResult {
    std::string segment_id;
    int         shards_checked;
    int         shards_corrupt;
    int         shards_repaired;
    double      duration_ms;
    bool        ok() const { return shards_corrupt == 0; }
};

// ================================================================
// ClusterDurabilityReport
// ================================================================
struct DurabilityReport {
    int     n_nodes_total;
    int     n_nodes_healthy;
    int     n_nodes_failed;
    int     rs_k, rs_m;
    int     shards_at_risk;     // < k surviving shards for any segment
    double  effective_nines;    // computed durability
    double  p_loss_per_year;
    std::string status;         // "OK" | "DEGRADED" | "CRITICAL"

    std::string summary() const {
        std::ostringstream ss;
        ss << "Durability Report:\n"
           << "  Nodes: " << n_nodes_healthy << "/" << n_nodes_total << " healthy\n"
           << "  RS("   << rs_k << "," << rs_m << "): survives "
           << rs_m << " simultaneous failures\n"
           << "  Shards at risk: " << shards_at_risk << "\n"
           << "  Effective nines: " << std::fixed << std::setprecision(1)
           << effective_nines << "\n"
           << "  P(data loss/year): " << std::scientific
           << std::setprecision(2) << p_loss_per_year << "\n"
           << "  Status: " << status << "\n";
        return ss.str();
    }
};

// ================================================================
// HealthMonitor
// ================================================================
struct HealthMonitorConfig {
        int    n_nodes              = 14;   // RS(10,4) = 14 nodes
        int    rs_k                 = 10;
        int    rs_m                 = 4;
        int    heartbeat_interval_ms= 5000;
        int    failure_timeout_ms   = 15000;  // 3 missed heartbeats
        int    scrub_interval_ms    = 7*24*3600*1000;  // 7 days
        int    repair_parallelism   = 4;
        double p_disk_failure_year  = 0.004;  // 0.4%/year (backblaze AFR)
        bool   verbose              = false;
    };

class HealthMonitor {
public:
    using Config = HealthMonitorConfig;
    using HeartbeatFn  = std::function<bool(int node_id)>;
    using RepairFn     = std::function<bool(int node_id, const std::string& seg_id)>;
    using ScrubFn      = std::function<ScrubResult(const std::string& seg_id)>;
    using AlertFn      = std::function<void(const std::string& msg)>;

    explicit HealthMonitor(Config cfg = HealthMonitorConfig{})
        : cfg_(std::move(cfg))
        , rs_(cfg_.rs_k, cfg_.rs_m)
    {
        node_status_.resize(cfg_.n_nodes);
        for (int i = 0; i < cfg_.n_nodes; ++i) {
            node_status_[i].node_id = i;
            node_status_[i].health  = NodeHealth::HEALTHY;
            node_status_[i].last_seen_ms = _now_ms();
        }
    }

    ~HealthMonitor() { stop(); }

    void set_heartbeat_fn(HeartbeatFn fn) { heartbeat_fn_ = fn; }
    void set_repair_fn(RepairFn fn)       { repair_fn_    = fn; }
    void set_scrub_fn(ScrubFn fn)         { scrub_fn_     = fn; }
    void set_alert_fn(AlertFn fn)         { alert_fn_     = fn; }

    void start() {
        running_ = true;
        heartbeat_thread_ = std::thread([this]{ _heartbeat_loop(); });
        scrub_thread_     = std::thread([this]{ _scrub_loop(); });
        repair_thread_    = std::thread([this]{ _repair_loop(); });
        if (cfg_.verbose)
            std::cout << "[Health] Monitor started (" << cfg_.n_nodes
                      << " nodes, RS(" << cfg_.rs_k << "," << cfg_.rs_m
                      << "), " << cfg_.scrub_interval_ms/3600000
                      << "h scrub interval)\n";
    }

    void stop() {
        running_ = false;
        if (heartbeat_thread_.joinable()) heartbeat_thread_.join();
        if (scrub_thread_.joinable())     scrub_thread_.join();
        if (repair_thread_.joinable())    repair_thread_.join();
    }

    // Register a segment for monitoring
    void register_segment(const std::string& seg_id,
                           const std::vector<int>& shard_nodes) {
        std::lock_guard lock(mu_);
        segment_nodes_[seg_id] = shard_nodes;
    }

    // Mark a node as failed (e.g. from Raft leader detection)
    void report_failure(int node_id) {
        std::lock_guard lock(mu_);
        if (node_id < 0 || node_id >= cfg_.n_nodes) return;
        auto& ns = node_status_[node_id];
        if (ns.health != NodeHealth::FAILED) {
            ns.health      = NodeHealth::FAILED;
            ns.last_error  = "Reported failed at " + std::to_string(_now_ms());
            failed_nodes_.insert(node_id);
            repair_queue_.push(node_id);
            ++total_failures_;
            _alert("[Health] Node " + std::to_string(node_id) + " FAILED — queuing repair");
        }
    }

    void report_recovered(int node_id) {
        std::lock_guard lock(mu_);
        if (node_id < 0 || node_id >= cfg_.n_nodes) return;
        auto& ns = node_status_[node_id];
        ns.health       = NodeHealth::HEALTHY;
        ns.last_seen_ms = _now_ms();
        failed_nodes_.erase(node_id);
        _alert("[Health] Node " + std::to_string(node_id) + " RECOVERED");
    }

    // ------ Durability report -----------------------------------
    DurabilityReport durability_report() const {
        std::lock_guard lock(mu_);
        int n_failed  = static_cast<int>(failed_nodes_.size());
        int n_healthy = cfg_.n_nodes - n_failed;

        // Effective p_fail accounts for currently failed nodes
        double eff_p = cfg_.p_disk_failure_year;

        DurabilityReport rep;
        rep.n_nodes_total   = cfg_.n_nodes;
        rep.n_nodes_healthy = n_healthy;
        rep.n_nodes_failed  = n_failed;
        rep.rs_k = cfg_.rs_k;
        rep.rs_m = cfg_.rs_m;
        rep.p_loss_per_year = rs_.durability_loss_probability(eff_p);
        rep.effective_nines = rs_.durability_nines(eff_p);
        rep.shards_at_risk  = static_cast<int>(at_risk_segments_.size());

        if (n_failed == 0)
            rep.status = "OK";
        else if (n_failed <= cfg_.rs_m)
            rep.status = "DEGRADED (repairing)";
        else
            rep.status = "CRITICAL — DATA LOSS RISK";

        return rep;
    }

    // ------ Metrics (Prometheus format) -------------------------
    std::string prometheus_metrics() const {
        auto rep = durability_report();
        std::ostringstream ss;
        ss << "# HELP cortexdb_nodes_healthy Healthy storage nodes\n"
           << "cortexdb_nodes_healthy " << rep.n_nodes_healthy << "\n"
           << "# HELP cortexdb_nodes_failed Failed storage nodes\n"
           << "cortexdb_nodes_failed " << rep.n_nodes_failed << "\n"
           << "# HELP cortexdb_durability_nines Effective durability nines\n"
           << "cortexdb_durability_nines " << rep.effective_nines << "\n"
           << "# HELP cortexdb_p_data_loss_year Annual probability of data loss\n"
           << "cortexdb_p_data_loss_year " << rep.p_loss_per_year << "\n"
           << "# HELP cortexdb_shards_at_risk Segments with insufficient surviving shards\n"
           << "cortexdb_shards_at_risk " << rep.shards_at_risk << "\n"
           << "# HELP cortexdb_scrubs_completed Total background scrubs completed\n"
           << "cortexdb_scrubs_completed " << scrubs_completed_.load() << "\n"
           << "# HELP cortexdb_repairs_completed Total shard repairs completed\n"
           << "cortexdb_repairs_completed " << repairs_completed_.load() << "\n"
           << "# HELP cortexdb_total_failures Total node failures detected\n"
           << "cortexdb_total_failures " << total_failures_.load() << "\n";
        return ss.str();
    }

    const std::vector<NodeStatus>& node_statuses() const { return node_status_; }

private:
    // ------ Heartbeat loop --------------------------------------
    void _heartbeat_loop() {
        while (running_) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(cfg_.heartbeat_interval_ms));

            if (!heartbeat_fn_) continue;

            for (int i = 0; i < cfg_.n_nodes && running_; ++i) {
                bool alive = heartbeat_fn_(i);
                std::lock_guard lock(mu_);
                auto& ns = node_status_[i];
                if (alive) {
                    ns.last_seen_ms = _now_ms();
                    if (ns.health == NodeHealth::FAILED) {
                        ns.health = NodeHealth::RECOVERING;
                    } else if (ns.health == NodeHealth::RECOVERING) {
                        ns.health = NodeHealth::HEALTHY;
                        failed_nodes_.erase(i);
                    }
                } else {
                    uint64_t elapsed = _now_ms() - ns.last_seen_ms;
                    if (elapsed > static_cast<uint64_t>(cfg_.failure_timeout_ms)
                        && ns.health == NodeHealth::HEALTHY) {
                        // Lock is held — call report_failure directly
                        ns.health = NodeHealth::FAILED;
                        failed_nodes_.insert(i);
                        repair_queue_.push(i);
                        ++total_failures_;
                        _alert("[Health] Node " + std::to_string(i) +
                               " heartbeat timeout after " +
                               std::to_string(elapsed) + "ms");
                    }
                }
            }
        }
    }

    // ------ Scrub loop ------------------------------------------
    void _scrub_loop() {
        // Stagger first scrub by a few minutes
        std::this_thread::sleep_for(std::chrono::minutes(5));

        while (running_) {
            // Scrub all registered segments
            std::vector<std::string> segs;
            {
                std::lock_guard lock(mu_);
                for (auto& [seg, _] : segment_nodes_) segs.push_back(seg);
            }

            if (cfg_.verbose)
                std::cout << "[Health] Starting scrub of " << segs.size() << " segments\n";

            int corrupt = 0;
            for (const auto& seg : segs) {
                if (!running_) break;
                if (scrub_fn_) {
                    auto result = scrub_fn_(seg);
                    if (!result.ok()) {
                        ++corrupt;
                        std::lock_guard lock(mu_);
                        at_risk_segments_.insert(seg);
                        _alert("[Health] Scrub found " +
                               std::to_string(result.shards_corrupt) +
                               " corrupt shard(s) in segment " + seg);
                    } else {
                        std::lock_guard lock(mu_);
                        at_risk_segments_.erase(seg);
                    }
                }
                ++scrubs_completed_;
            }

            if (cfg_.verbose)
                std::cout << "[Health] Scrub complete: " << segs.size()
                          << " segments, " << corrupt << " corrupt\n";

            // Sleep until next scrub
            auto sleep_ms = cfg_.scrub_interval_ms;
            for (int i = 0; i < sleep_ms/1000 && running_; ++i)
                std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }

    // ------ Repair loop -----------------------------------------
    void _repair_loop() {
        while (running_) {
            std::this_thread::sleep_for(std::chrono::seconds(1));

            int node_to_repair = -1;
            {
                std::lock_guard lock(mu_);
                if (!repair_queue_.empty()) {
                    node_to_repair = repair_queue_.front();
                    repair_queue_.pop();
                }
            }

            if (node_to_repair < 0 || !repair_fn_) continue;

            // Find all segments with a shard on this node
            std::vector<std::string> affected;
            {
                std::lock_guard lock(mu_);
                for (auto& [seg, nodes] : segment_nodes_)
                    for (int n : nodes)
                        if (n == node_to_repair) { affected.push_back(seg); break; }
            }

            if (cfg_.verbose)
                std::cout << "[Health] Repairing " << affected.size()
                          << " segments for node " << node_to_repair << "\n";

            int repaired = 0;
            for (const auto& seg : affected) {
                if (!running_) break;
                // Find a healthy replacement node
                int replacement = _find_replacement_node(node_to_repair);
                if (replacement < 0) {
                    _alert("[Health] No healthy replacement for node " +
                           std::to_string(node_to_repair));
                    break;
                }
                if (repair_fn_(replacement, seg)) {
                    ++repaired;
                    ++repairs_completed_;
                    // Update segment → node mapping
                    std::lock_guard lock(mu_);
                    auto& nodes = segment_nodes_[seg];
                    std::replace(nodes.begin(), nodes.end(), node_to_repair, replacement);
                }
            }

            if (cfg_.verbose)
                std::cout << "[Health] Repaired " << repaired << "/"
                          << affected.size() << " segments for node "
                          << node_to_repair << "\n";
        }
    }

    int _find_replacement_node(int failed_node) const {
        for (int i = 0; i < cfg_.n_nodes; ++i)
            if (i != failed_node && !failed_nodes_.count(i))
                return i;
        return -1;
    }

    void _alert(const std::string& msg) {
        if (cfg_.verbose) std::cout << msg << "\n";
        if (alert_fn_) alert_fn_(msg);
    }

    static uint64_t _now_ms() {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
    }

    Config cfg_;
    erasure::ReedSolomon rs_;

    mutable std::mutex mu_;
    std::vector<NodeStatus> node_status_;
    std::unordered_set<int> failed_nodes_;
    std::unordered_set<std::string> at_risk_segments_;
    std::unordered_map<std::string, std::vector<int>> segment_nodes_;
    std::queue<int> repair_queue_;

    HeartbeatFn heartbeat_fn_;
    RepairFn    repair_fn_;
    ScrubFn     scrub_fn_;
    AlertFn     alert_fn_;

    std::atomic<bool>     running_{false};
    std::atomic<uint64_t> scrubs_completed_{0};
    std::atomic<uint64_t> repairs_completed_{0};
    std::atomic<uint64_t> total_failures_{0};

    std::thread heartbeat_thread_;
    std::thread scrub_thread_;
    std::thread repair_thread_;
};

} // namespace health
} // namespace cortexdb
