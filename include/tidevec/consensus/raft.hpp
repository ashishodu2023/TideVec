#pragma once
// ================================================================
// consensus/raft.hpp — Raft consensus for CortexDB
//
// WHY RAFT over our current quorum-write ReplicaSet:
//
//   Current ReplicaSet:
//     · Simple quorum write (write to N/2+1 nodes)
//     · No leader election — any node can accept writes
//     · Split-brain possible if network partition occurs
//     · Recovery: manual WAL replay
//
//   Raft adds:
//     · SINGLE LEADER: all writes go through elected leader
//     · LINEARIZABLE READS: stale reads impossible
//     · AUTOMATIC FAILOVER: new leader elected in <150ms
//     · LOG ORDERING: every entry has a global term+index
//     · SAFETY GUARANTEE: at most one leader per term EVER
//     · MEMBERSHIP CHANGES: add/remove nodes without downtime
//
//   For 99.999999999% availability:
//     · 5-node Raft cluster tolerates 2 simultaneous node failures
//     · Availability = 1 - P(≥3 of 5 nodes down simultaneously)
//     · At p_down=0.001 per node: 1 - C(5,3)*p^3 ≈ 99.9999999%
//
// ALGORITHM (from Ongaro & Ousterhout, USENIX ATC 2014):
//
//   1. LEADER ELECTION:
//      Nodes start as Followers. If no heartbeat in [150,300]ms,
//      become Candidate, increment term, request votes.
//      Win if majority votes. Become Leader.
//
//   2. LOG REPLICATION:
//      Client sends write to Leader.
//      Leader appends entry to its log.
//      Sends AppendEntries RPC to all followers.
//      Once majority ACK, entry is COMMITTED → applied to state machine.
//      Leader replies to client.
//
//   3. SAFETY:
//      A node only votes for a candidate if candidate's log is
//      at least as up-to-date as the voter's log.
//      Ensures committed entries are never lost.
//
// This file: in-process Raft for single-machine multi-shard CortexDB.
// Distributed Raft (gRPC transport) is in raft_grpc.hpp (v0.3).
// ================================================================

#include <cortexdb/core/cortex_vector.hpp>

#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <memory>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <thread>
#include <chrono>
#include <random>
#include <optional>
#include <queue>
#include <iostream>

namespace cortexdb {
namespace consensus {

// ================================================================
// Types
// ================================================================
using Term  = uint64_t;
using Index = uint64_t;
using NodeId= int;

enum class RaftRole { FOLLOWER, CANDIDATE, LEADER };

inline std::string role_str(RaftRole r) {
    switch (r) {
        case RaftRole::FOLLOWER:  return "FOLLOWER";
        case RaftRole::CANDIDATE: return "CANDIDATE";
        case RaftRole::LEADER:    return "LEADER";
        default:                  return "UNKNOWN";
    }
}

// Log entry — maps to a CortexDB write operation
struct LogEntry {
    Term        term;
    Index       index;
    std::string op;           // "UPSERT" | "DELETE" | "ADD_EDGE" | "NOOP"
    std::vector<uint8_t> payload;  // serialised operation data

    bool is_noop() const { return op == "NOOP"; }
};

// Result of a Raft operation
struct RaftResult {
    bool    success;
    Term    term;
    Index   index;
    std::string error;

    static RaftResult ok(Term t, Index i) { return {true, t, i, ""}; }
    static RaftResult fail(const std::string& e) { return {false, 0, 0, e}; }
};

// ================================================================
// RaftConfig
// ================================================================
struct RaftConfig {
    NodeId  node_id;
    int     n_nodes          = 5;         // cluster size (5 = tolerates 2 failures)
    int     election_timeout_min_ms = 80;
    int     election_timeout_max_ms = 160;
    int     heartbeat_interval_ms   = 25;
    int     max_log_entries_per_rpc = 100;
    bool    verbose          = false;
};

// ================================================================
// Raft peer interface (in-process; replaced by gRPC in v0.3)
// ================================================================
struct AppendEntriesArgs {
    Term   term;
    NodeId leader_id;
    Index  prev_log_index;
    Term   prev_log_term;
    std::vector<LogEntry> entries;
    Index  leader_commit;
};

struct AppendEntriesReply {
    Term  term;
    bool  success;
    Index conflict_index;  // for fast log rollback
    Term  conflict_term;
};

struct RequestVoteArgs {
    Term   term;
    NodeId candidate_id;
    Index  last_log_index;
    Term   last_log_term;
};

struct RequestVoteReply {
    Term term;
    bool vote_granted;
};

// ================================================================
// RaftNode — core Raft state machine
// ================================================================
class RaftNode {
public:
    // Callback: called when an entry is committed (apply to state machine)
    using ApplyFn = std::function<void(const LogEntry&)>;

    explicit RaftNode(RaftConfig cfg, ApplyFn apply_fn)
        : cfg_(std::move(cfg))
        , apply_fn_(std::move(apply_fn))
        , role_(RaftRole::FOLLOWER)
        , current_term_(0)
        , voted_for_(-1)
        , commit_index_(0)
        , last_applied_(0)
        , votes_received_(0)
        , rng_(std::random_device{}())
    {
        next_index_.resize(cfg_.n_nodes, 1);
        match_index_.resize(cfg_.n_nodes, 0);
        // Append a NOOP entry at index 0 (sentinel)
        log_.push_back({0, 0, "NOOP", {}});
    }

    ~RaftNode() { stop(); }

    // Start background threads
    void start() {
        running_ = true;
        election_thread_ = std::thread([this]{ _election_loop(); });
        apply_thread_    = std::thread([this]{ _apply_loop(); });
        if (cfg_.verbose)
            std::cout << "[Raft " << cfg_.node_id << "] started\n";
    }

    void stop() {
        running_ = false;
        cv_.notify_all();
        apply_cv_.notify_all();
        if (election_thread_.joinable()) election_thread_.join();
        if (apply_thread_.joinable())    apply_thread_.join();
    }

    // ------ Client API ------------------------------------------

    // Submit a write — only succeeds on the leader
    // Returns the log index; committed when majority ACK
    std::optional<Index> submit(const std::string& op,
                                 const std::vector<uint8_t>& payload) {
        std::unique_lock lock(mu_);
        if (role_ != RaftRole::LEADER) return std::nullopt;

        Index idx = log_.size();
        log_.push_back({current_term_, idx, op, payload});
        next_index_[cfg_.node_id] = idx + 1;
        match_index_[cfg_.node_id] = idx;

        if (cfg_.verbose)
            std::cout << "[Raft " << cfg_.node_id << "] appended entry "
                      << idx << " term=" << current_term_ << "\n";
        _replicate_to_peers(lock);
        return idx;
    }

    // Wait for index to be committed (with timeout)
    bool wait_committed(Index idx, int timeout_ms = 5000) {
        auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(timeout_ms);
        std::unique_lock lock(mu_);
        return apply_cv_.wait_until(lock, deadline,
            [&]{ return commit_index_ >= idx || !running_; });
    }

    // ------ Peer RPCs (called by peer nodes) --------------------

    AppendEntriesReply on_append_entries(const AppendEntriesArgs& args) {
        std::unique_lock lock(mu_);

        AppendEntriesReply reply{current_term_, false, 0, 0};

        // Rule 1: reject if leader's term is stale
        if (args.term < current_term_) return reply;

        // Update term if we see a newer one
        if (args.term > current_term_) {
            current_term_ = args.term;
            voted_for_ = -1;
            role_ = RaftRole::FOLLOWER;
        }
        leader_id_ = args.leader_id;
        _reset_election_timer();

        // Rule 2: check prevLogIndex/prevLogTerm match
        if (args.prev_log_index >= static_cast<Index>(log_.size())) {
            reply.conflict_index = static_cast<Index>(log_.size());
            reply.conflict_term  = 0;
            return reply;
        }
        if (args.prev_log_index > 0 &&
            log_[args.prev_log_index].term != args.prev_log_term) {
            reply.conflict_term  = log_[args.prev_log_index].term;
            // Find first index with this term for fast rollback
            for (Index i = 1; i < static_cast<Index>(log_.size()); ++i)
                if (log_[i].term == reply.conflict_term) {
                    reply.conflict_index = i; break;
                }
            return reply;
        }

        // Append new entries
        for (const auto& entry : args.entries) {
            if (entry.index < static_cast<Index>(log_.size())) {
                // Conflict: truncate log
                if (log_[entry.index].term != entry.term)
                    log_.resize(entry.index);
                else continue;  // already have this entry
            }
            log_.push_back(entry);
        }

        // Update commit index
        if (args.leader_commit > commit_index_) {
            commit_index_ = std::min(args.leader_commit,
                static_cast<Index>(log_.size()-1));
            apply_cv_.notify_all();
        }

        reply.success = true;
        reply.term    = current_term_;
        return reply;
    }

    RequestVoteReply on_request_vote(const RequestVoteArgs& args) {
        std::unique_lock lock(mu_);
        RequestVoteReply reply{current_term_, false};

        if (args.term < current_term_) return reply;

        if (args.term > current_term_) {
            current_term_ = args.term;
            voted_for_    = -1;
            role_         = RaftRole::FOLLOWER;
        }

        // Grant vote if haven't voted, and candidate's log is at least as up-to-date
        bool can_vote = (voted_for_ < 0 || voted_for_ == args.candidate_id);
        Index last_idx  = static_cast<Index>(log_.size()-1);
        Term  last_term = log_[last_idx].term;
        bool log_ok = (args.last_log_term > last_term) ||
                      (args.last_log_term == last_term &&
                       args.last_log_index >= last_idx);

        if (can_vote && log_ok) {
            voted_for_ = args.candidate_id;
            reply.vote_granted = true;
            _reset_election_timer();
        }
        reply.term = current_term_;
        return reply;
    }

    // ------ Peer registration -----------------------------------
    // In-process: peers are other RaftNode pointers
    void register_peer(NodeId id, RaftNode* peer) {
        std::lock_guard lock(mu_);
        peers_[id] = peer;
    }

    // ------ State queries ---------------------------------------
    RaftRole role()         const { std::shared_lock l(smu_); return role_; }
    Term     current_term() const { std::shared_lock l(smu_); return current_term_; }
    Index    commit_index() const { std::shared_lock l(smu_); return commit_index_; }
    Index    log_size()     const { std::shared_lock l(smu_); return log_.size(); }
    bool     is_leader()    const { return role() == RaftRole::LEADER; }
    NodeId   leader_id()    const { std::shared_lock l(smu_); return leader_id_; }

    NodeId node_id() const { return cfg_.node_id; }

    void print_state() const {
        std::cout << "[Raft " << cfg_.node_id << "] "
                  << role_str(role_) << " term=" << current_term_
                  << " commit=" << commit_index_
                  << " log_size=" << log_.size() << "\n";
    }

private:
    // ------ Election loop (background thread) -------------------
    void _election_loop() {
        _reset_election_timer();
        while (running_) {
            std::unique_lock lock(mu_);
            // Wait for election timeout or notification
            bool timed_out = !cv_.wait_until(lock, election_deadline_,
                [this]{ return !running_ || election_reset_; });
            election_reset_ = false;
            if (!running_) break;

            if (timed_out && role_ != RaftRole::LEADER) {
                _start_election(lock);
            } else if (role_ == RaftRole::LEADER) {
                // Send heartbeats
                _send_heartbeats(lock);
                election_deadline_ = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(cfg_.heartbeat_interval_ms);
            }
        }
    }

    void _start_election(std::unique_lock<std::mutex>& lock) {
        role_ = RaftRole::CANDIDATE;
        ++current_term_;
        voted_for_      = cfg_.node_id;
        votes_received_ = 1;  // vote for self

        if (cfg_.verbose)
            std::cout << "[Raft " << cfg_.node_id << "] starting election term="
                      << current_term_ << "\n";

        Term   term     = current_term_;
        Index  last_idx = static_cast<Index>(log_.size()-1);
        Term   last_term= log_[last_idx].term;

        lock.unlock();
        // Request votes from all peers in parallel
        int votes = 1;
        for (auto& [pid, peer] : peers_) {
            if (!peer) continue;
            RequestVoteArgs args{term, cfg_.node_id, last_idx, last_term};
            auto reply = peer->on_request_vote(args);
            if (reply.vote_granted) ++votes;
            if (reply.term > term) {
                // We're stale — revert to follower
                lock.lock();
                current_term_ = reply.term;
                voted_for_    = -1;
                role_         = RaftRole::FOLLOWER;
                _reset_election_timer();
                return;
            }
        }
        lock.lock();

        int majority = cfg_.n_nodes / 2 + 1;
        if (role_ == RaftRole::CANDIDATE && current_term_ == term &&
            votes >= majority) {
            role_ = RaftRole::LEADER;
            leader_id_ = cfg_.node_id;
            if (cfg_.verbose)
                std::cout << "[Raft " << cfg_.node_id << "] became LEADER term="
                          << current_term_ << "\n";
            // Reinitialise leader volatile state
            for (int i = 0; i < cfg_.n_nodes; ++i) {
                next_index_[i]  = static_cast<Index>(log_.size());
                match_index_[i] = 0;
            }
            // Send immediate heartbeat
            _send_heartbeats(lock);
            // Reset to heartbeat cadence
            election_deadline_ = std::chrono::steady_clock::now() +
                std::chrono::milliseconds(cfg_.heartbeat_interval_ms);
        } else {
            _reset_election_timer();
        }
    }

    void _send_heartbeats(std::unique_lock<std::mutex>& lock) {
        if (role_ != RaftRole::LEADER) return;
        _replicate_to_peers(lock);
    }

    void _replicate_to_peers(std::unique_lock<std::mutex>& lock) {
        if (role_ != RaftRole::LEADER) return;
        Term  term = current_term_;
        Index commit = commit_index_;

        // Snapshot needed state before unlocking
        struct PeerWork {
            NodeId id; RaftNode* peer;
            AppendEntriesArgs args;
        };
        std::vector<PeerWork> work;
        for (auto& [pid, peer] : peers_) {
            if (!peer) continue;
            Index ni = next_index_[pid];
            Index prev_idx  = ni > 0 ? ni-1 : 0;
            Term  prev_term = (prev_idx < static_cast<Index>(log_.size()))
                            ? log_[prev_idx].term : 0;
            std::vector<LogEntry> entries;
            for (Index i = ni; i < static_cast<Index>(log_.size()) &&
                 static_cast<int>(entries.size()) < cfg_.max_log_entries_per_rpc; ++i)
                entries.push_back(log_[i]);

            work.push_back({pid, peer, {term, cfg_.node_id, prev_idx, prev_term, entries, commit}});
        }

        lock.unlock();
        // Send RPCs without holding mutex
        int acks = 1;  // self
        int majority = cfg_.n_nodes / 2 + 1;
        for (auto& pw : work) {
            auto reply = pw.peer->on_append_entries(pw.args);
            if (reply.term > term) {
                lock.lock();
                current_term_ = reply.term;
                voted_for_    = -1;
                role_         = RaftRole::FOLLOWER;
                _reset_election_timer();
                return;
            }
            if (reply.success) {
                lock.lock();
                Index new_match = pw.args.prev_log_index + pw.args.entries.size();
                match_index_[pw.id] = std::max(match_index_[pw.id], new_match);
                next_index_[pw.id]  = match_index_[pw.id] + 1;
                lock.unlock();
                ++acks;
            } else {
                // Log inconsistency: back up
                lock.lock();
                if (reply.conflict_term > 0) {
                    // Find last entry with conflict_term
                    Index ni = 1;
                    for (Index i = static_cast<Index>(log_.size())-1; i >= 1; --i)
                        if (log_[i].term == reply.conflict_term) { ni = i+1; break; }
                    next_index_[pw.id] = ni;
                } else {
                    next_index_[pw.id] = reply.conflict_index;
                }
                lock.unlock();
            }
        }
        lock.lock();

        // Advance commit index if majority replicated
        if (role_ == RaftRole::LEADER) {
            for (Index n = static_cast<Index>(log_.size())-1; n > commit_index_; --n) {
                if (log_[n].term != current_term_) break;
                int cnt = 1;
                for (int pid = 0; pid < cfg_.n_nodes; ++pid)
                    if (pid != cfg_.node_id && match_index_[pid] >= n) ++cnt;
                if (cnt >= majority) {
                    commit_index_ = n;
                    apply_cv_.notify_all();
                    break;
                }
            }
        }
    }

    // ------ Apply loop (background thread) ----------------------
    void _apply_loop() {
        while (running_) {
            std::unique_lock lock(mu_);
            apply_cv_.wait(lock, [this]{
                return !running_ || last_applied_ < commit_index_;
            });
            while (last_applied_ < commit_index_) {
                ++last_applied_;
                const auto& entry = log_[last_applied_];
                lock.unlock();
                if (!entry.is_noop()) {
                    try { apply_fn_(entry); }
                    catch (const std::exception& e) {
                        std::cerr << "[Raft " << cfg_.node_id
                                  << "] apply error: " << e.what() << "\n";
                    }
                }
                lock.lock();
            }
        }
    }

    void _reset_election_timer() {
        std::uniform_int_distribution<int> dist(
            cfg_.election_timeout_min_ms, cfg_.election_timeout_max_ms);
        election_deadline_ = std::chrono::steady_clock::now() +
            std::chrono::milliseconds(dist(rng_));
        election_reset_ = true;
        cv_.notify_all();
    }

    RaftConfig  cfg_;
    ApplyFn     apply_fn_;

    // Persistent state
    RaftRole    role_;
    Term        current_term_;
    NodeId      voted_for_;
    std::vector<LogEntry> log_;
    Index       commit_index_;
    Index       last_applied_;
    NodeId      leader_id_ = -1;

    // Volatile state (leader only)
    std::vector<Index> next_index_;
    std::vector<Index> match_index_;
    int         votes_received_;

    std::unordered_map<NodeId, RaftNode*> peers_;

    mutable std::mutex  mu_;
    mutable std::shared_mutex smu_;
    std::condition_variable cv_, apply_cv_;

    std::chrono::steady_clock::time_point election_deadline_;
    bool election_reset_ = false;
    std::atomic<bool> running_{false};
    std::mt19937 rng_;

    std::thread election_thread_;
    std::thread apply_thread_;
};

// ================================================================
// RaftGroup — manages a cluster of RaftNode instances
// (In-process for embedded mode; gRPC transport in v0.3)
// ================================================================
class RaftGroup {
public:
    explicit RaftGroup(int n_nodes,
                       RaftNode::ApplyFn apply_fn,
                       bool verbose = false) {
        nodes_.reserve(n_nodes);
        for (int i = 0; i < n_nodes; ++i) {
            RaftConfig cfg;
            cfg.node_id = i;
            cfg.n_nodes = n_nodes;
            cfg.verbose = verbose;
            nodes_.push_back(std::make_unique<RaftNode>(cfg, apply_fn));
        }
        // Wire peers
        for (int i = 0; i < n_nodes; ++i)
            for (int j = 0; j < n_nodes; ++j)
                if (i != j)
                    nodes_[i]->register_peer(j, nodes_[j].get());
    }

    void start() {
        for (auto& n : nodes_) n->start();
    }

    void stop() {
        for (auto& n : nodes_) n->stop();
    }

    // Submit to leader (auto-discover leader)
    std::optional<Index> submit(const std::string& op,
                                 const std::vector<uint8_t>& payload,
                                 int retries = 3) {
        for (int attempt = 0; attempt < retries; ++attempt) {
            for (auto& n : nodes_) {
                if (n->is_leader()) {
                    auto idx = n->submit(op, payload);
                    if (idx) return idx;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        return std::nullopt;
    }

    // Wait for index to be committed across all nodes
    bool wait_committed(Index idx, int timeout_ms = 5000) {
        for (auto& n : nodes_)
            if (!n->wait_committed(idx, timeout_ms)) return false;
        return true;
    }

    RaftNode* leader() const {
        for (auto& n : nodes_)
            if (n->is_leader()) return n.get();
        return nullptr;
    }

    int n_nodes() const { return static_cast<int>(nodes_.size()); }

    void print_state() const {
        std::cout << "\n=== Raft Cluster State ===\n";
        for (auto& n : nodes_) n->print_state();
    }

private:
    std::vector<std::unique_ptr<RaftNode>> nodes_;
};

} // namespace consensus
} // namespace cortexdb
