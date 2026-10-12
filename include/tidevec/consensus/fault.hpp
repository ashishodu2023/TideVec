#pragma once
// ================================================================
// consensus/fault.hpp — fault injection for Raft testing
//
// EXPERIMENTAL / TEST-ONLY. A FaultInjector is shared by one node's
// RaftNode and its transports. The server exposes it through
// POST /v1/admin/fault, only when started with --enable-fault-injection.
//
//   · network partitions: drop RPCs to / from chosen node ids
//   · lossy or slow links: random drop probability, fixed RPC delay
//   · slow disk: delay before every Raft fsync
//   · disk full: every Raft write fails (the node refuses entries it
//     cannot persist, as it would on ENOSPC)
//   · clock skew: timer_scale multiplies this node's election and
//     heartbeat timers (a fast clock is < 1), clock_offset_ms shifts
//     the wall clock the leader stamps on writes
// ================================================================

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <thread>

namespace tidevec {
namespace consensus {

class FaultInjector {
public:
    void set_blocked(std::set<int> to, std::set<int> from) {
        std::lock_guard l(mu_);
        blocked_to_ = std::move(to);
        blocked_from_ = std::move(from);
    }
    std::set<int> blocked_to() const { std::lock_guard l(mu_); return blocked_to_; }
    std::set<int> blocked_from() const { std::lock_guard l(mu_); return blocked_from_; }

    bool blocks_to(int id) const { std::lock_guard l(mu_); return blocked_to_.count(id) > 0; }
    bool blocks_from(int id) const { std::lock_guard l(mu_); return blocked_from_.count(id) > 0; }

    bool should_drop() {
        double p = drop_prob.load();
        if (p <= 0) return false;
        std::lock_guard l(mu_);
        return std::uniform_real_distribution<double>(0, 1)(rng_) < p;
    }

    // Called before every Raft disk write.
    void before_io() const {
        if (int d = fsync_delay_ms.load(); d > 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(d));
        if (disk_full.load())
            throw std::runtime_error("disk full (injected fault)");
    }

    void delay_rpc() const {
        if (int d = rpc_delay_ms.load(); d > 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(d));
    }

    double timer_scale() const {
        double s = timer_scale_.load();
        return s > 0.01 ? s : 0.01;
    }
    void set_timer_scale(double s) { timer_scale_ = s; }

    void clear() {
        set_blocked({}, {});
        rpc_delay_ms = 0;
        fsync_delay_ms = 0;
        disk_full = false;
        drop_prob = 0;
        timer_scale_ = 1.0;
        clock_offset_ms = 0;
    }

    std::atomic<int>     rpc_delay_ms{0};
    std::atomic<int>     fsync_delay_ms{0};
    std::atomic<bool>    disk_full{false};
    std::atomic<double>  drop_prob{0};
    std::atomic<int64_t> clock_offset_ms{0};

private:
    mutable std::mutex mu_;
    std::set<int> blocked_to_, blocked_from_;
    std::atomic<double> timer_scale_{1.0};
    std::mt19937 rng_{std::random_device{}()};
};

} // namespace consensus
} // namespace tidevec
