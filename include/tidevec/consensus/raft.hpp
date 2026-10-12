#pragma once
// ================================================================
// consensus/raft.hpp — Raft consensus for TideVec
//
// Implements leader election, log replication and the commit rule
// from Ongaro & Ousterhout, "In Search of an Understandable Consensus
// Algorithm" (USENIX ATC 2014), plus the extensions from Ongaro's
// dissertation ("Consensus: Bridging Theory and Practice", 2014):
//
//   · SINGLE LEADER per term; writes go through the leader.
//   · LOG MATCHING: entries are replicated in order; a follower whose
//     log diverges is rolled back to the leader's log.
//   · COMMIT RULE: an entry is committed once a majority stores it
//     and it belongs to the leader's current term (a new leader
//     appends a NOOP so that earlier-term entries commit promptly).
//   · ELECTION RESTRICTION: votes only go to candidates whose log is
//     at least as up-to-date as the voter's.
//   · LOG COMPACTION (§5): every snapshot_threshold applied entries the
//     state machine is snapshotted and the log prefix discarded.
//     Followers that fall behind the snapshot receive it in chunks via
//     InstallSnapshot.
//   · MEMBERSHIP CHANGES (§4.1): single-server add/remove through
//     CONFIG log entries. A server uses the latest configuration in its
//     log, committed or not; only one change may be pending, and only
//     after the leader has committed an entry in its term. New servers
//     first catch up as non-voting learners.
//   · LINEARIZABLE READS (§6.4): read_index() confirms leadership with a
//     heartbeat round before serving; lease_read() skips the round while
//     a majority acknowledged the leader within a lease (assumes bounded
//     clock drift; relies on check-quorum and leader stickiness below).
//   · PRE-VOTE and CHECK-QUORUM (§9.6, §6.2): a node only increments its
//     term after a majority says it could win; nodes ignore elections
//     while they hear from a current leader; a leader that cannot reach a
//     majority for an election timeout steps down.
//
// Transport is pluggable via RaftPeer:
//   · LocalPeer  — in-process (UltraDurableCollection, unit tests)
//   · HttpPeer   — multi-process, see consensus/raft_http.hpp
//
// Each peer gets its own replicator thread, so one slow or dead peer
// never delays heartbeats to the others.
//
// Persistence (RaftConfig::persist_dir non-empty): currentTerm,
// votedFor, the log and the latest snapshot are fsync'ed before any
// reply that depends on them. A failed write (e.g. disk full) is rolled
// back and the request refused instead of acknowledged.
// ================================================================

#include <tidevec/core/cortex_vector.hpp>
#include <tidevec/consensus/fault.hpp>

#include <algorithm>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <functional>
#include <memory>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <chrono>
#include <random>
#include <optional>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <unistd.h>

namespace tidevec {
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

// Log entry — maps to a TideVec write operation
struct LogEntry {
    Term        term;
    Index       index;
    std::string op;           // "UPSERT" | "DELETE" | "ADD_EDGE" | "CMD" | "NOOP" | "CONFIG"
    std::vector<uint8_t> payload;  // serialised operation data

    bool is_noop()   const { return op == "NOOP"; }
    bool is_config() const { return op == "CONFIG"; }
};

// One voting member of the cluster. addr is transport-specific (a base
// URL for HttpPeer, empty for in-process peers).
struct Member {
    NodeId      id = -1;
    std::string addr;
    bool operator==(const Member&) const = default;
};
using Membership = std::vector<Member>;

inline std::string encode_membership(const Membership& m) {
    std::string s;
    for (const auto& x : m) s += std::to_string(x.id) + " " + x.addr + "\n";
    return s;
}

inline Membership decode_membership(const std::string& s) {
    Membership out;
    std::istringstream in(s);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        auto sp = line.find(' ');
        Member m;
        m.id = std::stoi(line.substr(0, sp));
        if (sp != std::string::npos) m.addr = line.substr(sp + 1);
        out.push_back(std::move(m));
    }
    return out;
}

inline bool has_member(const Membership& m, NodeId id) {
    return std::any_of(m.begin(), m.end(), [id](const Member& x) { return x.id == id; });
}

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
    NodeId  node_id = 0;
    int     n_nodes          = 5;         // used when `members` is empty: ids 0..n_nodes-1
    Membership members;                   // bootstrap configuration (must include node_id)
    bool    join             = false;     // start with no configuration; wait to be added
    int     election_timeout_min_ms = 80;
    int     election_timeout_max_ms = 160;
    int     heartbeat_interval_ms   = 25;
    int     max_log_entries_per_rpc = 100;
    bool    verbose          = false;

    bool    pre_vote         = true;
    bool    check_quorum     = true;

    // Snapshot after this many applied entries (0 = never). Requires
    // RaftNode::set_snapshot_handlers().
    Index       snapshot_threshold   = 0;
    std::size_t snapshot_chunk_bytes = 1 << 20;

    // Directory for term/vote/log/snapshot. Empty = volatile (in-memory only).
    std::string persist_dir;

    // Entries up to this index are already reflected in the state
    // machine (e.g. after a restart) and are not re-applied.
    Index   initial_applied  = 0;

    // Optional fault injection (tests / chaos runs only).
    std::shared_ptr<FaultInjector> fault;
};

// ================================================================
// RPC messages
// ================================================================
struct AppendEntriesArgs {
    Term   term;
    NodeId leader_id;
    Index  prev_log_index;
    Term   prev_log_term;
    std::vector<LogEntry> entries;
    Index  leader_commit;
};

// success=false with conflict_term == 0 and conflict_index ==
// prev_log_index + 1 means a transient failure (the follower could not
// persist): the leader retries from the same point.
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
    bool   pre_vote = false;   // probe only: the receiver changes no state
};

struct RequestVoteReply {
    Term term;
    bool vote_granted;
};

struct InstallSnapshotArgs {
    Term   term;
    NodeId leader_id;
    Index  last_included_index;
    Term   last_included_term;
    Membership config;          // configuration as of last_included_index
    uint64_t offset;            // byte offset of this chunk
    std::vector<uint8_t> data;
    bool   done;
};

struct InstallSnapshotReply {
    Term     term;
    bool     success;
    uint64_t next_offset;       // bytes of this snapshot the follower holds
};

// ================================================================
// RaftPeer — transport to one remote node.
// Returning nullopt means the RPC failed (unreachable, timeout).
// ================================================================
class RaftPeer {
public:
    virtual ~RaftPeer() = default;
    virtual std::optional<AppendEntriesReply>
        append_entries(const AppendEntriesArgs& args) = 0;
    virtual std::optional<RequestVoteReply>
        request_vote(const RequestVoteArgs& args) = 0;
    virtual std::optional<InstallSnapshotReply>
        install_snapshot(const InstallSnapshotArgs&) { return std::nullopt; }
};

struct RaftStatus {
    NodeId   node_id;
    RaftRole role;
    Term     term;
    NodeId   leader_id;
    Index    commit_index;
    Index    last_applied;
    Index    last_log_index;
    Index    snapshot_index = 0;
    Term     snapshot_term  = 0;
    Membership members;
    Index    config_index   = 0;
    bool     config_pending = false;
    std::vector<NodeId> learners;
};

// Drops / delays outgoing RPCs according to the sender's FaultInjector.
class FaultyPeer : public RaftPeer {
public:
    FaultyPeer(std::shared_ptr<RaftPeer> inner, NodeId target,
               std::shared_ptr<FaultInjector> fault)
        : inner_(std::move(inner)), target_(target), fault_(std::move(fault)) {}

    std::optional<AppendEntriesReply>
    append_entries(const AppendEntriesArgs& a) override {
        if (_drop()) return std::nullopt;
        return inner_->append_entries(a);
    }
    std::optional<RequestVoteReply>
    request_vote(const RequestVoteArgs& a) override {
        if (_drop()) return std::nullopt;
        return inner_->request_vote(a);
    }
    std::optional<InstallSnapshotReply>
    install_snapshot(const InstallSnapshotArgs& a) override {
        if (_drop()) return std::nullopt;
        return inner_->install_snapshot(a);
    }

private:
    bool _drop() {
        if (!fault_) return false;
        fault_->delay_rpc();
        if (fault_->blocks_to(target_) || fault_->should_drop()) {
            // An unreachable peer looks like a short timeout, not an
            // instant refusal, so replicator loops don't spin.
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            return true;
        }
        return false;
    }

    std::shared_ptr<RaftPeer>      inner_;
    NodeId                         target_;
    std::shared_ptr<FaultInjector> fault_;
};

// ================================================================
// RaftNode — core Raft state machine
// ================================================================
class RaftNode {
public:
    // Called once per committed entry other than NOOP/CONFIG, in log order.
    using ApplyFn     = std::function<void(const LogEntry&)>;
    // Serialises the state machine as of the last applied entry. Runs on
    // the apply thread, so no apply happens concurrently.
    using SnapshotFn  = std::function<std::vector<uint8_t>()>;
    // Replaces the state machine with a snapshot taken at `index`.
    using RestoreFn   = std::function<void(const std::vector<uint8_t>&, Index)>;
    // Creates a transport for a member added at runtime.
    using PeerFactory = std::function<std::shared_ptr<RaftPeer>(const Member&)>;
    using Clock       = std::chrono::steady_clock;

    explicit RaftNode(RaftConfig cfg, ApplyFn apply_fn)
        : cfg_(std::move(cfg))
        , apply_fn_(std::move(apply_fn))
        , rng_(std::random_device{}())
    {
        if (cfg_.join) {
            // Learns its configuration from the leader.
        } else if (cfg_.members.empty()) {
            if (cfg_.n_nodes < 1 || cfg_.node_id < 0 || cfg_.node_id >= cfg_.n_nodes)
                throw std::invalid_argument("RaftConfig: node_id must be in [0, n_nodes)");
            for (int i = 0; i < cfg_.n_nodes; ++i) base_config_.push_back({i, ""});
        } else {
            if (!has_member(cfg_.members, cfg_.node_id))
                throw std::invalid_argument("RaftConfig: members must include node_id");
            base_config_ = cfg_.members;
        }
        snap_config_ = base_config_;
        log_.push_back({0, 0, "NOOP", {}});   // sentinel at snap_index_
        if (!cfg_.persist_dir.empty()) _load();
        _recompute_config();
        last_applied_ = std::min<Index>(cfg_.initial_applied, _last_index());
        commit_index_ = std::max(snap_index_, last_applied_);
    }

    ~RaftNode() {
        stop();
        if (log_fp_) std::fclose(log_fp_);
    }

    RaftNode(const RaftNode&) = delete;
    RaftNode& operator=(const RaftNode&) = delete;

    // Register a transport for a node id (in-process wiring, tests).
    void register_peer(NodeId id, std::shared_ptr<RaftPeer> peer) {
        std::lock_guard lock(mu_);
        if (id < 0 || id == cfg_.node_id)
            throw std::invalid_argument("register_peer: bad peer id");
        transports_[id] = std::move(peer);
        if (auto it = peers_.find(id); it != peers_.end()) it->second->peer = transports_[id];
        _reconcile_peers();
    }
    inline void register_peer(NodeId id, RaftNode* peer);

    void set_peer_factory(PeerFactory f) {
        std::lock_guard lock(mu_);
        peer_factory_ = std::move(f);
        _reconcile_peers();
    }

    void set_snapshot_handlers(SnapshotFn snap, RestoreFn restore) {
        std::lock_guard lock(mu_);
        snapshot_fn_ = std::move(snap);
        restore_fn_  = std::move(restore);
    }

    void start() {
        std::unique_lock lock(mu_);
        if (running_) return;
        running_ = true;
        _reset_election_timer();
        election_thread_ = std::thread([this]{ _election_loop(); });
        apply_thread_    = std::thread([this]{ _apply_loop(); });
        _reconcile_peers();
        if (cfg_.verbose)
            std::cout << "[Raft " << cfg_.node_id << "] started term="
                      << current_term_ << " log=" << _last_index()
                      << " snapshot=" << snap_index_ << "\n";
    }

    void stop() {
        {
            std::lock_guard lock(mu_);
            if (!running_) return;
            running_ = false;
        }
        cv_.notify_all();
        apply_cv_.notify_all();
        peer_cv_.notify_all();
        progress_cv_.notify_all();
        if (election_thread_.joinable()) election_thread_.join();
        if (apply_thread_.joinable())    apply_thread_.join();
        std::vector<std::thread> threads;
        {
            std::lock_guard lock(mu_);
            for (auto& [id, ps] : peers_)
                if (ps->thread.joinable()) threads.push_back(std::move(ps->thread));
            for (auto& t : finished_threads_) threads.push_back(std::move(t));
            finished_threads_.clear();
        }
        for (auto& t : threads) if (t.joinable()) t.join();
        std::lock_guard lock(mu_);
        for (auto& [id, ps] : peers_) ps->thread_alive = false;
    }

    // ------ Client API ------------------------------------------

    // Append a write to the leader's log. Returns its index, or nullopt
    // if this node is not the leader. Throws if the entry cannot be
    // persisted; a leader whose appends keep failing for longer than the
    // minimum election timeout steps down so a healthy node can take over
    // (it cannot win again until it can persist a new term).
    // Use wait_committed() to block until a majority has it.
    std::optional<Index> submit(const std::string& op,
                                 const std::vector<uint8_t>& payload) {
        if (op == "CONFIG")
            throw std::invalid_argument("submit: use add_server/remove_server for CONFIG");
        std::unique_lock lock(mu_);
        if (role_ != RaftRole::LEADER || !running_) return std::nullopt;
        Index idx;
        try { idx = _append_local(op, payload); }
        catch (const std::exception& e) {
            auto now = Clock::now();
            if (io_fail_since_ == Clock::time_point{}) {
                io_fail_since_ = now;
            } else if (now - io_fail_since_ >= _election_timeout_min()) {
                std::cerr << "[Raft " << cfg_.node_id << "] cannot persist log entries ("
                          << e.what() << "), stepping down\n";
                io_fail_since_ = {};
                _step_down(current_term_);
            }
            throw;
        }
        io_fail_since_ = {};
        if (cfg_.verbose)
            std::cout << "[Raft " << cfg_.node_id << "] appended entry "
                      << idx << " term=" << current_term_ << "\n";
        _advance_commit();
        peer_cv_.notify_all();
        return idx;
    }

    // Wait until idx is committed. Fails if leadership is lost and the
    // entry was overwritten, or on timeout.
    bool wait_committed(Index idx, int timeout_ms = 5000) {
        auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
        std::unique_lock lock(mu_);
        return apply_cv_.wait_until(lock, deadline,
            [&]{ return commit_index_ >= idx || !running_; }) && commit_index_ >= idx;
    }

    // Term of the entry at idx (0 if absent or compacted into a snapshot).
    Term term_at(Index idx) const {
        std::lock_guard l(mu_);
        if (idx < snap_index_ || idx > _last_index()) return 0;
        return _term(idx);
    }

    // Linearizable read barrier (ReadIndex). On the leader: waits until
    // an entry of the current term is committed, confirms leadership with
    // a majority heartbeat round, then waits until the state machine has
    // applied the commit index observed at the start. The caller may then
    // read local state. nullopt = not leader / leadership lost / timeout.
    std::optional<Index> read_index(int timeout_ms = 2000) {
        auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
        std::unique_lock lock(mu_);
        if (role_ != RaftRole::LEADER || !running_) return std::nullopt;
        const Term term = current_term_;
        auto still_leader = [&]{ return running_ && role_ == RaftRole::LEADER && current_term_ == term; };
        if (!progress_cv_.wait_until(lock, deadline, [&]{
                return !still_leader() || _term(commit_index_) == term; }) || !still_leader())
            return std::nullopt;
        const Index ri = commit_index_;
        if (!_quorum([&](NodeId id) { return id == cfg_.node_id; })) {
            const uint64_t round = ++hb_round_;
            auto now = Clock::now();
            for (auto& [id, ps] : peers_) ps->next_send = now;
            peer_cv_.notify_all();
            bool ok = progress_cv_.wait_until(lock, deadline, [&]{
                return !still_leader() || _quorum([&](NodeId id) {
                    if (id == cfg_.node_id) return true;
                    auto it = peers_.find(id);
                    return it != peers_.end() && it->second->ack_round >= round;
                });
            });
            if (!ok || !still_leader()) return std::nullopt;
        }
        if (!apply_cv_.wait_until(lock, deadline, [&]{ return last_applied_ >= ri || !running_; })
            || last_applied_ < ri)
            return std::nullopt;
        return ri;
    }

    // Lease-based read: like read_index() but skips the heartbeat round
    // while a majority acknowledged this leader within 80% of the minimum
    // election timeout. Unsafe if clocks drift by more than that margin.
    std::optional<Index> lease_read(int timeout_ms = 2000) {
        {
            auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
            std::unique_lock lock(mu_);
            if (role_ != RaftRole::LEADER || !running_) return std::nullopt;
            auto now = Clock::now();
            auto lease = std::chrono::duration_cast<Clock::duration>(_election_timeout_min() * 0.8);
            bool valid = _term(commit_index_) == current_term_ && _quorum([&](NodeId id) {
                if (id == cfg_.node_id) return true;
                auto it = peers_.find(id);
                return it != peers_.end() && now - it->second->last_ack < lease;
            });
            if (valid) {
                Index ri = commit_index_;
                if (apply_cv_.wait_until(lock, deadline, [&]{ return last_applied_ >= ri || !running_; })
                    && last_applied_ >= ri)
                    return ri;
                return std::nullopt;
            }
        }
        return read_index(timeout_ms);
    }

    // ------ Membership changes (leader only) ---------------------

    // Adds a voting member. The new server first replicates as a learner
    // until it is within one RPC batch of the leader, then a CONFIG entry
    // adding it is committed. Idempotent if the member already exists.
    RaftResult add_server(const Member& m, int timeout_ms = 15000) {
        auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
        std::unique_lock lock(mu_);
        if (m.id < 0) return RaftResult::fail("invalid node id");
        for (int pass = 0; pass < 2; ++pass) {
            for (const auto& x : config_)
                if (x.id == m.id)
                    return x.addr == m.addr ? RaftResult::ok(current_term_, config_index_)
                        : RaftResult::fail("node id already in the cluster with another address");
            if (pass == 0)
                if (auto e = _await_change_slot(lock, deadline); !e.empty()) return RaftResult::fail(e);
        }
        const Term term = current_term_;
        learners_[m.id] = m;
        _reconcile_peers();
        const Index margin = static_cast<Index>(std::max(1, cfg_.max_log_entries_per_rpc));
        auto still_leader = [&]{ return running_ && role_ == RaftRole::LEADER && current_term_ == term; };
        bool caught_up = progress_cv_.wait_until(lock, deadline, [&]{
            if (!still_leader()) return true;
            auto it = peers_.find(m.id);
            return it != peers_.end() && it->second->match_index + margin >= _last_index();
        });
        auto abandon = [&](const std::string& why) {
            learners_.erase(m.id);
            _reconcile_peers();
            return RaftResult::fail(why);
        };
        if (!still_leader()) return abandon("leadership lost during catch-up");
        if (!caught_up) return abandon("new server did not catch up before the timeout");
        if (auto e = _change_blocker(); !e.empty()) return abandon(e);
        Membership nc = config_;
        nc.push_back(m);
        Index idx;
        try { idx = _append_config(nc); }
        catch (const std::exception& e) { return abandon(e.what()); }
        learners_.erase(m.id);
        _advance_commit();
        peer_cv_.notify_all();
        return _wait_config_commit(lock, idx, term, deadline);
    }

    // Removes a member. Removing the leader itself makes it step down
    // once the change commits.
    RaftResult remove_server(NodeId id, int timeout_ms = 15000) {
        auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
        std::unique_lock lock(mu_);
        learners_.erase(id);
        if (!has_member(config_, id)) {
            _reconcile_peers();
            return RaftResult::ok(current_term_, config_index_);
        }
        if (auto e = _await_change_slot(lock, deadline); !e.empty()) return RaftResult::fail(e);
        if (!has_member(config_, id)) return RaftResult::ok(current_term_, config_index_);
        if (config_.size() == 1) return RaftResult::fail("cannot remove the last member");
        Membership nc;
        for (const auto& x : config_) if (x.id != id) nc.push_back(x);
        const Term term = current_term_;
        Index idx;
        try { idx = _append_config(nc); }
        catch (const std::exception& e) { return RaftResult::fail(e.what()); }
        _advance_commit();
        peer_cv_.notify_all();
        return _wait_config_commit(lock, idx, term, deadline);
    }

    // ------ Peer RPC handlers -----------------------------------

    AppendEntriesReply on_append_entries(const AppendEntriesArgs& args) {
        std::unique_lock lock(mu_);
        AppendEntriesReply reply{current_term_, false, 0, 0};
        if (args.term < current_term_) return reply;
        try {
            if (args.term > current_term_ || role_ != RaftRole::FOLLOWER)
                _step_down(args.term);
            leader_id_ = args.leader_id;
            last_leader_contact_ = Clock::now();
            prevoting_ = false;
            _reset_election_timer();
            reply.term = current_term_;

            Index prev      = args.prev_log_index;
            Term  prev_term = args.prev_log_term;
            std::size_t skip = 0;
            if (prev < snap_index_) {
                // Everything up to snap_index_ is committed and in our snapshot.
                if (prev + args.entries.size() <= snap_index_) {
                    reply.success = true;
                    return reply;
                }
                skip = static_cast<std::size_t>(snap_index_ - prev);
                prev = snap_index_;
                prev_term = snap_term_;
            }

            // Consistency check on (prev_log_index, prev_log_term).
            if (prev > _last_index()) {
                reply.conflict_index = _last_index() + 1;
                return reply;
            }
            if (_term(prev) != prev_term) {
                reply.conflict_term  = _term(prev);
                reply.conflict_index = prev;
                while (reply.conflict_index > snap_index_ + 1 &&
                       _term(reply.conflict_index - 1) == reply.conflict_term)
                    --reply.conflict_index;
                return reply;
            }

            // Find the first entry that is new or conflicts.
            std::size_t i = skip;
            Index trunc_at = 0;
            for (; i < args.entries.size(); ++i) {
                const auto& e = args.entries[i];
                if (e.index > _last_index()) break;
                if (_term(e.index) != e.term) { trunc_at = e.index; break; }
            }
            if (trunc_at && trunc_at <= commit_index_) {
                std::cerr << "[Raft " << cfg_.node_id
                          << "] refusing to truncate committed entry " << trunc_at << "\n";
                return reply;
            }
            if (i < args.entries.size()) {
                std::vector<LogEntry> tail(args.entries.begin() + static_cast<std::ptrdiff_t>(i),
                                           args.entries.end());
                if (trunc_at) {
                    std::vector<LogEntry> nl(log_.begin(),
                        log_.begin() + static_cast<std::ptrdiff_t>(trunc_at - snap_index_));
                    nl.insert(nl.end(), tail.begin(), tail.end());
                    _persist_rewrite(nl);
                    log_ = std::move(nl);
                    while (!log_configs_.empty() && log_configs_.back().first >= trunc_at)
                        log_configs_.pop_back();
                } else {
                    _persist_entries(tail);
                    log_.insert(log_.end(), tail.begin(), tail.end());
                }
                for (const auto& e : tail)
                    if (e.is_config())
                        log_configs_.push_back({e.index, _decode_config(e)});
                _recompute_config();
            }

            Index last_new = args.prev_log_index + args.entries.size();
            if (args.leader_commit > commit_index_) {
                commit_index_ = std::max(commit_index_, std::min(args.leader_commit, last_new));
                apply_cv_.notify_all();
            }
            reply.success = true;
        } catch (const std::exception& e) {
            _log_throttled(std::string("append_entries failed: ") + e.what());
            reply = {current_term_, false, args.prev_log_index + 1, 0};
        }
        return reply;
    }

    RequestVoteReply on_request_vote(const RequestVoteArgs& args) {
        std::unique_lock lock(mu_);
        RequestVoteReply reply{current_term_, false};
        auto now = Clock::now();
        Index last_idx  = _last_index();
        Term  last_term = _term(last_idx);
        bool log_ok = (args.last_log_term > last_term) ||
                      (args.last_log_term == last_term &&
                       args.last_log_index >= last_idx);

        if (args.pre_vote) {
            reply.vote_granted = args.term > current_term_ && log_ok && !_leader_recent(now);
            return reply;
        }
        if (args.term < current_term_) return reply;
        // Leader stickiness: ignore elections while a leader is active.
        if (args.term > current_term_ && cfg_.check_quorum && _leader_recent(now))
            return reply;
        try {
            if (args.term > current_term_) _step_down(args.term);
            bool can_vote = (voted_for_ < 0 || voted_for_ == args.candidate_id);
            if (can_vote && log_ok) {
                if (voted_for_ != args.candidate_id) {
                    _persist_meta(current_term_, args.candidate_id);
                    voted_for_ = args.candidate_id;
                }
                reply.vote_granted = true;
                _reset_election_timer();
            }
        } catch (const std::exception& e) {
            std::cerr << "[Raft " << cfg_.node_id << "] request_vote failed: "
                      << e.what() << "\n";
            reply.vote_granted = false;
        }
        reply.term = current_term_;
        return reply;
    }

    InstallSnapshotReply on_install_snapshot(const InstallSnapshotArgs& args) {
        std::unique_lock lock(mu_);
        InstallSnapshotReply reply{current_term_, false, 0};
        if (args.term < current_term_) return reply;
        try {
            if (args.term > current_term_ || role_ != RaftRole::FOLLOWER)
                _step_down(args.term);
            leader_id_ = args.leader_id;
            last_leader_contact_ = Clock::now();
            prevoting_ = false;
            _reset_election_timer();
            reply.term = current_term_;

            bool same = recv_index_ == args.last_included_index &&
                        recv_term_ == args.last_included_term;
            if (args.offset == 0) {
                recv_buf_.clear();
                recv_index_ = args.last_included_index;
                recv_term_  = args.last_included_term;
            } else if (!same || args.offset != recv_buf_.size()) {
                reply.next_offset = same ? recv_buf_.size() : 0;
                return reply;
            }
            recv_buf_.insert(recv_buf_.end(), args.data.begin(), args.data.end());
            reply.next_offset = recv_buf_.size();
            if (!args.done) { reply.success = true; return reply; }

            const Index idx = args.last_included_index;
            const Term  term = args.last_included_term;
            if (idx <= snap_index_ || idx <= last_applied_) {
                // Already covered by our own state.
                recv_buf_.clear();
                reply.success = true;
                return reply;
            }
            auto data = std::make_shared<const std::vector<uint8_t>>(std::move(recv_buf_));
            recv_buf_.clear();
            recv_index_ = 0;
            recv_term_  = 0;
            _persist_snapshot(idx, term, args.config, *data);

            std::vector<LogEntry> nl;
            nl.push_back({term, idx, "NOOP", {}});
            bool keep_suffix = idx <= _last_index() && _term(idx) == term;
            if (keep_suffix)
                nl.insert(nl.end(),
                    log_.begin() + static_cast<std::ptrdiff_t>(idx - snap_index_) + 1, log_.end());
            snap_index_  = idx;
            snap_term_   = term;
            snap_config_ = args.config;
            snap_data_   = data;
            log_ = std::move(nl);
            std::erase_if(log_configs_, [&](const auto& c) { return !keep_suffix || c.first <= idx; });
            try { _persist_rewrite(log_); }
            catch (const std::exception& e) {
                // The old log file is still valid: entries <= idx are skipped on load.
                std::cerr << "[Raft " << cfg_.node_id << "] log rewrite after snapshot failed: "
                          << e.what() << "\n";
            }
            _recompute_config();
            commit_index_ = std::max(commit_index_, idx);
            apply_cv_.notify_all();
            if (cfg_.verbose)
                std::cout << "[Raft " << cfg_.node_id << "] installed snapshot at "
                          << idx << " (" << data->size() << " bytes)\n";
            reply.success = true;
        } catch (const std::exception& e) {
            std::cerr << "[Raft " << cfg_.node_id << "] install_snapshot failed: "
                      << e.what() << "\n";
            recv_buf_.clear();
            recv_index_ = recv_term_ = 0;
            reply = {current_term_, false, 0};
        }
        return reply;
    }

    // ------ State queries ---------------------------------------
    RaftRole role()         const { std::lock_guard l(mu_); return role_; }
    Term     current_term() const { std::lock_guard l(mu_); return current_term_; }
    Index    commit_index() const { std::lock_guard l(mu_); return commit_index_; }
    Index    last_applied() const { std::lock_guard l(mu_); return last_applied_; }
    Index    log_size()     const { std::lock_guard l(mu_); return _last_index() + 1; }
    Index    snapshot_index() const { std::lock_guard l(mu_); return snap_index_; }
    bool     is_leader()    const { return role() == RaftRole::LEADER; }
    NodeId   leader_id()    const { std::lock_guard l(mu_); return leader_id_; }
    NodeId   node_id()      const { return cfg_.node_id; }
    Membership members()    const { std::lock_guard l(mu_); return config_; }

    // Address of a member or learner ("" if unknown).
    std::string address_of(NodeId id) const {
        std::lock_guard l(mu_);
        for (const auto& m : config_) if (m.id == id) return m.addr;
        if (auto it = learners_.find(id); it != learners_.end()) return it->second.addr;
        for (const auto& m : cfg_.members) if (m.id == id) return m.addr;
        return "";
    }

    RaftStatus status() const {
        std::lock_guard l(mu_);
        RaftStatus s{};
        s.node_id        = cfg_.node_id;
        s.role           = role_;
        s.term           = current_term_;
        s.leader_id      = leader_id_;
        s.commit_index   = commit_index_;
        s.last_applied   = last_applied_;
        s.last_log_index = _last_index();
        s.snapshot_index = snap_index_;
        s.snapshot_term  = snap_term_;
        s.members        = config_;
        s.config_index   = config_index_;
        s.config_pending = config_index_ > commit_index_;
        for (const auto& [id, m] : learners_) s.learners.push_back(id);
        return s;
    }

    void print_state() const {
        auto s = status();
        std::cout << "[Raft " << s.node_id << "] " << role_str(s.role)
                  << " term=" << s.term << " commit=" << s.commit_index
                  << " log_size=" << s.last_log_index + 1 << "\n";
    }

private:
    struct PeerState {
        NodeId id = -1;
        std::string addr;
        std::shared_ptr<RaftPeer> peer;
        Index next_index  = 1;
        Index match_index = 0;
        Term  vote_sent_term = 0;
        uint64_t prevote_round = 0;
        Clock::time_point next_send{}, retry_after{}, last_ack{};
        uint64_t ack_round = 0;
        Index    snap_sending = 0;
        uint64_t snap_offset  = 0;
        bool retired      = false;
        bool thread_alive = false;
        std::thread thread;
    };

    // All _-prefixed helpers below expect mu_ to be held.

    Index _last_index() const { return snap_index_ + static_cast<Index>(log_.size()) - 1; }
    const LogEntry& _entry(Index i) const { return log_[static_cast<std::size_t>(i - snap_index_)]; }
    Term  _term(Index i) const { return _entry(i).term; }

    bool _is_voter(NodeId id) const { return has_member(config_, id); }

    template <typename Pred>
    bool _quorum(Pred has) const {
        std::size_t n = config_.size(), c = 0;
        for (const auto& m : config_) if (has(m.id)) ++c;
        return n > 0 && 2 * c > n;
    }

    double _scale() const { return cfg_.fault ? cfg_.fault->timer_scale() : 1.0; }
    std::chrono::duration<double, std::milli> _election_timeout_min() const {
        return std::chrono::duration<double, std::milli>(cfg_.election_timeout_min_ms * _scale());
    }
    Clock::duration _election_timeout_max() const {
        return std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double, std::milli>(cfg_.election_timeout_max_ms * _scale()));
    }
    Clock::duration _hb_interval() const {
        return std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double, std::milli>(cfg_.heartbeat_interval_ms * _scale()));
    }

    bool _leader_recent(Clock::time_point now) const {
        if (role_ == RaftRole::LEADER) return true;
        return last_leader_contact_ != Clock::time_point{} &&
               now - last_leader_contact_ <
                   std::chrono::duration_cast<Clock::duration>(_election_timeout_min());
    }

    Index _append_local(const std::string& op, const std::vector<uint8_t>& payload) {
        Index idx = _last_index() + 1;
        std::vector<LogEntry> one{{current_term_, idx, op, payload}};
        _persist_entries(one);
        log_.push_back(std::move(one.front()));
        if (log_.back().is_config()) {
            log_configs_.push_back({idx, _decode_config(log_.back())});
            _recompute_config();
        }
        return idx;
    }

    Index _append_config(const Membership& nc) {
        auto s = encode_membership(nc);
        return _append_local("CONFIG", std::vector<uint8_t>(s.begin(), s.end()));
    }

    static Membership _decode_config(const LogEntry& e) {
        return decode_membership(std::string(e.payload.begin(), e.payload.end()));
    }

    std::string _change_blocker() const {
        if (role_ != RaftRole::LEADER || !running_) return "not the leader";
        if (config_index_ > commit_index_) return "another membership change is in progress";
        if (_term(commit_index_) != current_term_)
            return "leader has not committed an entry in its term yet; retry";
        return "";
    }

    // Waits (until deadline) for a change still in flight to commit, e.g.
    // right after a previous change or before a new leader's first commit.
    std::string _await_change_slot(std::unique_lock<std::mutex>& lock, Clock::time_point deadline) {
        for (;;) {
            auto e = _change_blocker();
            if (e.empty() || role_ != RaftRole::LEADER || !running_ || Clock::now() >= deadline)
                return e;
            apply_cv_.wait_until(lock, std::min(deadline, Clock::now() + std::chrono::milliseconds(20)));
        }
    }

    RaftResult _wait_config_commit(std::unique_lock<std::mutex>& lock, Index idx, Term term,
                                   Clock::time_point deadline) {
        apply_cv_.wait_until(lock, deadline, [&]{
            return !running_ || commit_index_ >= idx || current_term_ != term;
        });
        if (commit_index_ >= idx && current_term_ == term) return RaftResult::ok(term, idx);
        return RaftResult::fail("membership change not confirmed (leadership changed or timeout); "
                                "check /v1/cluster/status");
    }

    void _recompute_config() {
        if (log_configs_.empty()) { config_ = snap_config_; config_index_ = snap_index_; }
        else { config_ = log_configs_.back().second; config_index_ = log_configs_.back().first; }
        _reconcile_peers();
    }

    // Starts replicator threads for current members and learners, and
    // retires those of removed nodes.
    void _reconcile_peers() {
        std::map<NodeId, std::string> want;
        for (const auto& m : config_) if (m.id != cfg_.node_id) want[m.id] = m.addr;
        for (const auto& [id, m] : learners_) if (id != cfg_.node_id) want[id] = m.addr;
        for (const auto& [id, addr] : want) {
            auto& slot = peers_[id];
            if (!slot) {
                slot = std::make_unique<PeerState>();
                slot->id = id;
                slot->next_index = _last_index() + 1;
            }
            PeerState& ps = *slot;
            if (!ps.peer || (ps.addr != addr && !transports_.count(id))) {
                std::shared_ptr<RaftPeer> t;
                if (auto it = transports_.find(id); it != transports_.end()) t = it->second;
                else if (peer_factory_) t = peer_factory_(Member{id, addr});
                if (t) { ps.peer = std::move(t); ps.addr = addr; }
            }
            ps.retired = false;
            if (running_ && ps.peer && !ps.thread_alive) {
                if (ps.thread.joinable()) finished_threads_.push_back(std::move(ps.thread));
                ps.thread_alive = true;
                ps.thread = std::thread([this, p = &ps]{ _peer_loop(*p); });
            }
        }
        for (auto& [id, ps] : peers_) if (!want.count(id)) ps->retired = true;
        peer_cv_.notify_all();
    }

    void _reset_election_timer() {
        std::uniform_real_distribution<double> dist(
            cfg_.election_timeout_min_ms, cfg_.election_timeout_max_ms);
        election_deadline_ = Clock::now() + std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double, std::milli>(dist(rng_) * _scale()));
        election_reset_ = true;
        cv_.notify_all();
    }

    void _step_down(Term term) {
        if (term > current_term_) {
            try { _persist_meta(term, -1); }
            catch (const std::exception& e) {
                // Adopting the term in memory is still safe: votes in it
                // are only granted after a successful persist.
                std::cerr << "[Raft " << cfg_.node_id << "] cannot persist term: "
                          << e.what() << "\n";
            }
            current_term_ = term;
            voted_for_ = -1;
        }
        if (role_ == RaftRole::LEADER) leader_id_ = -1;
        role_ = RaftRole::FOLLOWER;
        prevoting_ = false;
        _reset_election_timer();
        progress_cv_.notify_all();
    }

    void _start_prevote() {
        prevoting_ = true;
        ++prevote_round_;
        prevotes_ = {cfg_.node_id};
        _reset_election_timer();
        peer_cv_.notify_all();
    }

    void _start_election() {
        Term t = current_term_ + 1;
        try { _persist_meta(t, cfg_.node_id); }
        catch (const std::exception& e) {
            std::cerr << "[Raft " << cfg_.node_id << "] cannot start election: "
                      << e.what() << "\n";
            _reset_election_timer();
            return;
        }
        role_ = RaftRole::CANDIDATE;
        current_term_ = t;
        voted_for_ = cfg_.node_id;
        votes_     = {cfg_.node_id};
        leader_id_ = -1;
        prevoting_ = false;
        if (cfg_.verbose)
            std::cout << "[Raft " << cfg_.node_id << "] starting election term="
                      << current_term_ << "\n";
        _reset_election_timer();
        if (_quorum([&](NodeId id) { return votes_.count(id) > 0; })) _become_leader();
        peer_cv_.notify_all();
    }

    void _become_leader() {
        role_      = RaftRole::LEADER;
        leader_id_ = cfg_.node_id;
        prevoting_ = false;
        auto now = Clock::now();
        leader_since_ = now;
        io_fail_since_ = {};
        for (auto& [id, ps] : peers_) {
            ps->next_index   = _last_index() + 1;
            ps->match_index  = 0;
            ps->next_send    = now;
            ps->retry_after  = now;
            ps->last_ack     = Clock::time_point{};
            ps->snap_sending = 0;
        }
        try { _append_local("NOOP", {}); }
        catch (const std::exception& e) {
            std::cerr << "[Raft " << cfg_.node_id << "] cannot append NOOP, stepping down: "
                      << e.what() << "\n";
            role_ = RaftRole::FOLLOWER;
            leader_id_ = -1;
            _reset_election_timer();
            return;
        }
        if (cfg_.verbose)
            std::cout << "[Raft " << cfg_.node_id << "] became LEADER term="
                      << current_term_ << "\n";
        _advance_commit();
        peer_cv_.notify_all();
        cv_.notify_all();
    }

    void _advance_commit() {
        if (role_ != RaftRole::LEADER) return;
        for (Index n = _last_index(); n > commit_index_; --n) {
            if (_term(n) != current_term_) break;
            bool ok = _quorum([&](NodeId id) {
                if (id == cfg_.node_id) return true;
                auto it = peers_.find(id);
                return it != peers_.end() && it->second->match_index >= n;
            });
            if (ok) {
                commit_index_ = n;
                apply_cv_.notify_all();
                progress_cv_.notify_all();
                break;
            }
        }
        // A leader removed from the configuration steps down once the
        // change is committed.
        if (config_index_ <= commit_index_ && !_is_voter(cfg_.node_id)) {
            if (cfg_.verbose)
                std::cout << "[Raft " << cfg_.node_id << "] removed from cluster, stepping down\n";
            role_ = RaftRole::FOLLOWER;
            leader_id_ = -1;
            _reset_election_timer();
            progress_cv_.notify_all();
        }
    }

    void _record_ack(PeerState& ps, uint64_t round, Clock::time_point sent_at) {
        ps.ack_round = std::max(ps.ack_round, round);
        if (sent_at > ps.last_ack) ps.last_ack = sent_at;
        progress_cv_.notify_all();
    }

    // ------ Election timer thread -------------------------------
    void _election_loop() {
        std::unique_lock lock(mu_);
        while (running_) {
            if (role_ == RaftRole::LEADER) {
                if (!cfg_.check_quorum) {
                    cv_.wait(lock, [this]{ return !running_ || role_ != RaftRole::LEADER; });
                    continue;
                }
                auto timeout = _election_timeout_max();
                cv_.wait_for(lock, timeout, [this]{ return !running_ || role_ != RaftRole::LEADER; });
                if (!running_ || role_ != RaftRole::LEADER) continue;
                auto now = Clock::now();
                if (now - leader_since_ < timeout) continue;
                bool ok = _quorum([&](NodeId id) {
                    if (id == cfg_.node_id) return true;
                    auto it = peers_.find(id);
                    return it != peers_.end() && now - it->second->last_ack < timeout;
                });
                if (!ok) {
                    if (cfg_.verbose)
                        std::cout << "[Raft " << cfg_.node_id
                                  << "] lost contact with a majority, stepping down\n";
                    role_ = RaftRole::FOLLOWER;
                    leader_id_ = -1;
                    _reset_election_timer();
                    progress_cv_.notify_all();
                }
                continue;
            }
            election_reset_ = false;
            bool woke = cv_.wait_until(lock, election_deadline_, [this]{
                return !running_ || election_reset_ || role_ == RaftRole::LEADER;
            });
            if (!running_) break;
            if (woke) continue;
            if (!_is_voter(cfg_.node_id)) { _reset_election_timer(); continue; }
            // A candidate whose election timed out (split vote) already
            // passed pre-vote; pre-votes are only sent by followers.
            if (cfg_.pre_vote && config_.size() > 1 && role_ == RaftRole::FOLLOWER) _start_prevote();
            else _start_election();
        }
    }

    // ------ Per-peer replicator thread --------------------------
    void _peer_loop(PeerState& ps) {
        std::unique_lock lock(mu_);
        while (running_ && !ps.retired) {
            const auto hb = _hb_interval();
            auto now = Clock::now();
            bool voter = _is_voter(ps.id);
            bool want_prevote = voter && prevoting_ && role_ == RaftRole::FOLLOWER &&
                                ps.prevote_round != prevote_round_;
            bool want_vote = voter && role_ == RaftRole::CANDIDATE &&
                             ps.vote_sent_term != current_term_;
            bool lagging = ps.next_index <= _last_index();
            bool want_ae = role_ == RaftRole::LEADER &&
                (now >= ps.next_send || (lagging && now >= ps.retry_after));

            if (!want_prevote && !want_vote && !want_ae) {
                auto wake = now + hb;
                if (role_ == RaftRole::LEADER) {
                    wake = ps.next_send;
                    if (lagging) wake = std::min(wake, ps.retry_after);
                }
                peer_cv_.wait_until(lock, wake);
                continue;
            }
            auto peer = ps.peer;

            if (want_prevote) {
                const uint64_t round = prevote_round_;
                ps.prevote_round = round;
                RequestVoteArgs args{current_term_ + 1, cfg_.node_id, _last_index(),
                                     _term(_last_index()), true};
                lock.unlock();
                auto r = peer->request_vote(args);
                lock.lock();
                if (!r) continue;
                if (r->term > current_term_ && !r->vote_granted) { _step_down(r->term); continue; }
                if (prevoting_ && prevote_round_ == round && role_ == RaftRole::FOLLOWER &&
                    r->vote_granted) {
                    prevotes_.insert(ps.id);
                    if (_quorum([&](NodeId id) { return prevotes_.count(id) > 0; })) {
                        prevoting_ = false;
                        _start_election();
                    }
                }
                continue;
            }

            if (want_vote) {
                Term term = current_term_;
                ps.vote_sent_term = term;
                RequestVoteArgs args{term, cfg_.node_id, _last_index(), _term(_last_index())};
                lock.unlock();
                auto r = peer->request_vote(args);
                lock.lock();
                if (!r) continue;
                if (r->term > current_term_) { _step_down(r->term); continue; }
                if (role_ == RaftRole::CANDIDATE && current_term_ == term && r->vote_granted) {
                    votes_.insert(ps.id);
                    if (_quorum([&](NodeId id) { return votes_.count(id) > 0; }))
                        _become_leader();
                }
                continue;
            }

            if (ps.next_index <= snap_index_) {
                _send_snapshot(ps, lock, peer, hb);
                continue;
            }

            const Term     term  = current_term_;
            const uint64_t round = hb_round_;
            Index ni = std::clamp<Index>(ps.next_index, snap_index_ + 1, _last_index() + 1);
            AppendEntriesArgs args;
            args.term           = term;
            args.leader_id      = cfg_.node_id;
            args.prev_log_index = ni - 1;
            args.prev_log_term  = _term(ni - 1);
            args.leader_commit  = commit_index_;
            for (Index i = ni; i <= _last_index() &&
                 static_cast<int>(args.entries.size()) < cfg_.max_log_entries_per_rpc; ++i)
                args.entries.push_back(_entry(i));
            const auto sent_at = now;
            ps.next_send = now + hb;

            lock.unlock();
            auto r = peer->append_entries(args);
            lock.lock();

            if (!r) { ps.retry_after = Clock::now() + hb; continue; }
            if (r->term > current_term_) { _step_down(r->term); continue; }
            if (role_ != RaftRole::LEADER || current_term_ != term) continue;
            _record_ack(ps, round, sent_at);

            if (r->success) {
                Index m = args.prev_log_index + args.entries.size();
                if (m > ps.match_index) ps.match_index = m;
                ps.next_index = ps.match_index + 1;
                _advance_commit();
            } else if (r->conflict_term == 0 && r->conflict_index == args.prev_log_index + 1) {
                ps.retry_after = Clock::now() + hb;   // follower could not persist; retry
            } else {
                Index back = r->conflict_index;
                if (r->conflict_term > 0) {
                    for (Index i = std::min(args.prev_log_index, _last_index()); i > snap_index_; --i)
                        if (_term(i) == r->conflict_term) { back = i + 1; break; }
                }
                ps.next_index = std::max<Index>(1, std::min(back, args.prev_log_index));
                ps.retry_after = Clock::now();
            }
        }
        ps.thread_alive = false;
    }

    void _send_snapshot(PeerState& ps, std::unique_lock<std::mutex>& lock,
                        const std::shared_ptr<RaftPeer>& peer, Clock::duration hb) {
        auto now = Clock::now();
        if (!snap_data_) {
            std::cerr << "[Raft " << cfg_.node_id << "] peer " << ps.id
                      << " needs a snapshot but none is loaded\n";
            ps.next_send = ps.retry_after = now + hb;
            return;
        }
        if (ps.snap_sending != snap_index_) { ps.snap_sending = snap_index_; ps.snap_offset = 0; }
        auto data = snap_data_;
        uint64_t off = std::min<uint64_t>(ps.snap_offset, data->size());
        uint64_t n = std::min<uint64_t>(cfg_.snapshot_chunk_bytes, data->size() - off);
        InstallSnapshotArgs args{current_term_, cfg_.node_id, snap_index_, snap_term_, snap_config_,
            off, std::vector<uint8_t>(data->begin() + static_cast<std::ptrdiff_t>(off),
                                      data->begin() + static_cast<std::ptrdiff_t>(off + n)),
            off + n == data->size()};
        const Term term = current_term_;
        const uint64_t round = hb_round_;
        ps.next_send = now + hb;

        lock.unlock();
        auto r = peer->install_snapshot(args);
        lock.lock();

        if (!r) { ps.retry_after = Clock::now() + hb; return; }
        if (r->term > current_term_) { _step_down(r->term); return; }
        if (role_ != RaftRole::LEADER || current_term_ != term) return;
        _record_ack(ps, round, now);
        if (ps.snap_sending != args.last_included_index) return;   // superseded
        ps.snap_offset = r->next_offset;
        if (r->success && args.done) {
            ps.match_index = std::max(ps.match_index, args.last_included_index);
            ps.next_index  = ps.match_index + 1;
            ps.snap_sending = 0;
            _advance_commit();
        } else {
            ps.retry_after = r->success ? Clock::now() : Clock::now() + hb;
        }
    }

    // ------ Apply thread ----------------------------------------
    void _apply_loop() {
        std::unique_lock lock(mu_);
        while (running_) {
            apply_cv_.wait(lock, [this]{
                return !running_ || last_applied_ < commit_index_;
            });
            while (running_ && last_applied_ < commit_index_) {
                if (last_applied_ < snap_index_) {
                    // A snapshot replaced the entries we have not applied.
                    auto data = snap_data_;
                    Index idx = snap_index_;
                    auto restore = restore_fn_;
                    lock.unlock();
                    if (restore && data) {
                        try { restore(*data, idx); }
                        catch (const std::exception& e) {
                            std::cerr << "[Raft " << cfg_.node_id
                                      << "] snapshot restore failed: " << e.what() << "\n";
                        }
                    } else {
                        std::cerr << "[Raft " << cfg_.node_id
                                  << "] snapshot at " << idx << " but no restore handler\n";
                    }
                    lock.lock();
                    last_applied_ = std::max(last_applied_, idx);
                    apply_cv_.notify_all();
                    continue;
                }
                LogEntry entry = _entry(last_applied_ + 1);
                lock.unlock();
                if (!entry.is_noop() && !entry.is_config()) {
                    try { apply_fn_(entry); }
                    catch (const std::exception& e) {
                        std::cerr << "[Raft " << cfg_.node_id
                                  << "] apply error at " << entry.index
                                  << ": " << e.what() << "\n";
                    }
                }
                lock.lock();
                if (entry.index == last_applied_ + 1) last_applied_ = entry.index;
                apply_cv_.notify_all();
                _maybe_snapshot(lock);
            }
        }
    }

    void _maybe_snapshot(std::unique_lock<std::mutex>& lock) {
        if (!cfg_.snapshot_threshold || !snapshot_fn_) return;
        if (last_applied_ < snap_index_ + cfg_.snapshot_threshold) return;
        const Index idx  = last_applied_;
        const Term  term = _term(idx);
        auto fn = snapshot_fn_;
        lock.unlock();
        std::vector<uint8_t> data;
        bool ok = true;
        try { data = fn(); }
        catch (const std::exception& e) {
            ok = false;
            std::cerr << "[Raft " << cfg_.node_id << "] snapshot failed: " << e.what() << "\n";
        }
        lock.lock();
        if (!ok || idx <= snap_index_ || idx > _last_index()) return;
        try { _compact(idx, term, std::move(data)); }
        catch (const std::exception& e) {
            std::cerr << "[Raft " << cfg_.node_id << "] snapshot persist failed: "
                      << e.what() << "\n";
        }
    }

    void _compact(Index idx, Term term, std::vector<uint8_t> data) {
        Membership cfg_at = snap_config_;
        for (const auto& [ci, c] : log_configs_) if (ci <= idx) cfg_at = c;
        auto shared = std::make_shared<const std::vector<uint8_t>>(std::move(data));
        _persist_snapshot(idx, term, cfg_at, *shared);
        std::vector<LogEntry> nl;
        nl.push_back({term, idx, "NOOP", {}});
        nl.insert(nl.end(), log_.begin() + static_cast<std::ptrdiff_t>(idx - snap_index_) + 1,
                  log_.end());
        snap_index_  = idx;
        snap_term_   = term;
        snap_config_ = cfg_at;
        snap_data_   = shared;
        log_ = std::move(nl);
        std::erase_if(log_configs_, [&](const auto& c) { return c.first <= idx; });
        if (cfg_.verbose)
            std::cout << "[Raft " << cfg_.node_id << "] snapshot at " << idx
                      << " (" << shared->size() << " bytes), log now "
                      << log_.size() - 1 << " entries\n";
        try { _persist_rewrite(log_); }
        catch (const std::exception& e) {
            std::cerr << "[Raft " << cfg_.node_id << "] log rewrite after snapshot failed: "
                      << e.what() << "\n";
        }
    }

    // ------ Persistence -----------------------------------------
    //
    // raft_meta     : "<term> <voted_for>\n", replaced atomically.
    // raft_log      : sequence of records
    //     u32 body_len | u32 fnv1a(body) | body
    //     body = u64 term | u64 index | u32 op_len | op | u32 len | payload
    //   A torn or corrupt tail record is discarded on load; records at or
    //   below the snapshot index are skipped.
    // raft_snapshot : "TVRS" | u32 version | u64 index | u64 term |
    //     u32 cfg_len | cfg | u64 data_len | data | u32 fnv1a(all before),
    //   replaced atomically.

    std::filesystem::path _meta_path() const { return std::filesystem::path(cfg_.persist_dir) / "raft_meta"; }
    std::filesystem::path _log_path()  const { return std::filesystem::path(cfg_.persist_dir) / "raft_log"; }
    std::filesystem::path _snap_path() const { return std::filesystem::path(cfg_.persist_dir) / "raft_snapshot"; }

    static uint32_t _fnv1a(const uint8_t* p, std::size_t n) {
        uint32_t h = 2166136261u;
        for (std::size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 16777619u; }
        return h;
    }

    template <typename T>
    static void _put(std::vector<uint8_t>& b, T v) {
        uint8_t raw[sizeof(T)];
        std::memcpy(raw, &v, sizeof(T));
        b.insert(b.end(), raw, raw + sizeof(T));
    }

    template <typename T>
    static bool _get(const std::vector<uint8_t>& b, std::size_t& off, T& v) {
        if (off + sizeof(T) > b.size()) return false;
        std::memcpy(&v, b.data() + off, sizeof(T));
        off += sizeof(T);
        return true;
    }

    static std::vector<uint8_t> _encode(const LogEntry& e) {
        std::vector<uint8_t> body;
        _put<uint64_t>(body, e.term);
        _put<uint64_t>(body, e.index);
        _put<uint32_t>(body, static_cast<uint32_t>(e.op.size()));
        body.insert(body.end(), e.op.begin(), e.op.end());
        _put<uint32_t>(body, static_cast<uint32_t>(e.payload.size()));
        body.insert(body.end(), e.payload.begin(), e.payload.end());
        std::vector<uint8_t> rec;
        _put<uint32_t>(rec, static_cast<uint32_t>(body.size()));
        _put<uint32_t>(rec, _fnv1a(body.data(), body.size()));
        rec.insert(rec.end(), body.begin(), body.end());
        return rec;
    }

    void _before_io() const { if (cfg_.fault) cfg_.fault->before_io(); }

    // At most one line per second for errors that repeat on every retry.
    void _log_throttled(const std::string& msg) {
        auto now = Clock::now();
        if (now - last_err_log_ < std::chrono::seconds(1)) { ++suppressed_errs_; return; }
        std::cerr << "[Raft " << cfg_.node_id << "] " << msg;
        if (suppressed_errs_) std::cerr << " (" << suppressed_errs_ << " similar suppressed)";
        std::cerr << "\n";
        last_err_log_ = now;
        suppressed_errs_ = 0;
    }

    static bool _fsync_file(std::FILE* fp) {
        return std::fflush(fp) == 0 && ::fsync(::fileno(fp)) == 0;
    }

    // Writes a whole file atomically (tmp + fsync + rename).
    void _write_atomic(const std::filesystem::path& path, const std::vector<uint8_t>& bytes) {
        auto tmp = path.string() + ".tmp";
        std::FILE* fp = std::fopen(tmp.c_str(), "wb");
        if (!fp) throw std::runtime_error("Raft: cannot write " + tmp);
        bool ok = std::fwrite(bytes.data(), 1, bytes.size(), fp) == bytes.size() && _fsync_file(fp);
        std::fclose(fp);
        if (!ok) {
            std::filesystem::remove(tmp);
            throw std::runtime_error("Raft: write failed for " + tmp);
        }
        std::filesystem::rename(tmp, path);
    }

    void _load() {
        std::filesystem::create_directories(cfg_.persist_dir);
        if (std::ifstream m(_meta_path()); m) {
            m >> current_term_ >> voted_for_;
            if (!m) { current_term_ = 0; voted_for_ = -1; }
        }
        _load_snapshot();
        log_.front() = {snap_term_, snap_index_, "NOOP", {}};

        std::vector<uint8_t> buf;
        if (std::ifstream f(_log_path(), std::ios::binary); f)
            buf.assign(std::istreambuf_iterator<char>(f), {});

        std::size_t off = 0, good = 0;
        while (true) {
            uint32_t len = 0, sum = 0;
            if (!_get(buf, off, len) || !_get(buf, off, sum)) break;
            if (off + len > buf.size()) break;
            std::vector<uint8_t> body(buf.begin() + static_cast<std::ptrdiff_t>(off),
                                      buf.begin() + static_cast<std::ptrdiff_t>(off + len));
            off += len;
            if (_fnv1a(body.data(), body.size()) != sum) break;
            std::size_t bo = 0;
            LogEntry e;
            uint32_t op_len = 0, pl_len = 0;
            if (!_get(body, bo, e.term) || !_get(body, bo, e.index) ||
                !_get(body, bo, op_len) || bo + op_len > body.size()) break;
            e.op.assign(body.begin() + static_cast<std::ptrdiff_t>(bo),
                        body.begin() + static_cast<std::ptrdiff_t>(bo + op_len));
            bo += op_len;
            if (!_get(body, bo, pl_len) || bo + pl_len > body.size()) break;
            e.payload.assign(body.begin() + static_cast<std::ptrdiff_t>(bo),
                             body.begin() + static_cast<std::ptrdiff_t>(bo + pl_len));
            if (e.index <= snap_index_) { good = off; continue; }
            if (e.index != _last_index() + 1) break;
            log_.push_back(std::move(e));
            if (log_.back().is_config())
                log_configs_.push_back({log_.back().index, _decode_config(log_.back())});
            good = off;
        }
        if (good != buf.size()) {
            std::cerr << "[Raft " << cfg_.node_id << "] discarding "
                      << buf.size() - good << " bytes of torn log tail\n";
            if (std::filesystem::exists(_log_path()))
                std::filesystem::resize_file(_log_path(), good);
        }
        log_fp_ = std::fopen(_log_path().c_str(), "ab");
        if (!log_fp_)
            throw std::runtime_error("Raft: cannot open " + _log_path().string());
    }

    void _load_snapshot() {
        if (!std::filesystem::exists(_snap_path())) return;
        std::vector<uint8_t> buf;
        if (std::ifstream f(_snap_path(), std::ios::binary); f)
            buf.assign(std::istreambuf_iterator<char>(f), {});
        auto corrupt = [&]{
            // The log prefix it replaced is gone: refuse to start rather
            // than silently lose committed entries.
            throw std::runtime_error("Raft: corrupt snapshot " + _snap_path().string());
        };
        if (buf.size() < 4 + 4 + 8 + 8 + 4 + 8 + 4 || std::memcmp(buf.data(), "TVRS", 4) != 0)
            corrupt();
        uint32_t want = 0;
        std::memcpy(&want, buf.data() + buf.size() - 4, 4);
        if (_fnv1a(buf.data(), buf.size() - 4) != want) corrupt();
        std::size_t off = 4;
        uint32_t version = 0, cfg_len = 0;
        uint64_t idx = 0, term = 0, data_len = 0;
        if (!_get(buf, off, version) || version != 1 || !_get(buf, off, idx) ||
            !_get(buf, off, term) || !_get(buf, off, cfg_len) || off + cfg_len > buf.size())
            corrupt();
        std::string cfg(buf.begin() + static_cast<std::ptrdiff_t>(off),
                        buf.begin() + static_cast<std::ptrdiff_t>(off + cfg_len));
        off += cfg_len;
        if (!_get(buf, off, data_len) || off + data_len + 4 != buf.size()) corrupt();
        snap_index_  = idx;
        snap_term_   = term;
        snap_config_ = decode_membership(cfg);
        snap_data_   = std::make_shared<const std::vector<uint8_t>>(
            buf.begin() + static_cast<std::ptrdiff_t>(off),
            buf.begin() + static_cast<std::ptrdiff_t>(off + data_len));
    }

    void _persist_meta(Term term, NodeId voted_for) {
        _before_io();
        if (cfg_.persist_dir.empty()) return;
        auto s = std::to_string(term) + " " + std::to_string(voted_for) + "\n";
        _write_atomic(_meta_path(), std::vector<uint8_t>(s.begin(), s.end()));
    }

    // Appends records; on failure the file is truncated back and the
    // in-memory log is left untouched by the caller.
    void _persist_entries(const std::vector<LogEntry>& entries) {
        _before_io();
        if (!log_fp_) return;
        std::fflush(log_fp_);
        long start = std::ftell(log_fp_);
        bool ok = true;
        for (const auto& e : entries) {
            auto rec = _encode(e);
            if (std::fwrite(rec.data(), 1, rec.size(), log_fp_) != rec.size()) { ok = false; break; }
        }
        if (ok) ok = _fsync_file(log_fp_);
        if (!ok) {
            std::clearerr(log_fp_);
            if (start >= 0 && ::ftruncate(::fileno(log_fp_), start) == 0)
                std::fseek(log_fp_, 0, SEEK_END);
            throw std::runtime_error("Raft: log write failed");
        }
    }

    void _persist_rewrite(const std::vector<LogEntry>& log) {
        _before_io();
        if (!log_fp_) return;
        std::vector<uint8_t> bytes;
        for (std::size_t i = 1; i < log.size(); ++i) {
            auto rec = _encode(log[i]);
            bytes.insert(bytes.end(), rec.begin(), rec.end());
        }
        _write_atomic(_log_path(), bytes);
        std::fclose(log_fp_);
        log_fp_ = std::fopen(_log_path().c_str(), "ab");
        if (!log_fp_)
            throw std::runtime_error("Raft: cannot reopen " + _log_path().string());
    }

    void _persist_snapshot(Index idx, Term term, const Membership& cfg,
                           const std::vector<uint8_t>& data) {
        _before_io();
        if (cfg_.persist_dir.empty()) return;
        std::vector<uint8_t> b;
        b.insert(b.end(), {'T', 'V', 'R', 'S'});
        _put<uint32_t>(b, 1);
        _put<uint64_t>(b, idx);
        _put<uint64_t>(b, term);
        auto c = encode_membership(cfg);
        _put<uint32_t>(b, static_cast<uint32_t>(c.size()));
        b.insert(b.end(), c.begin(), c.end());
        _put<uint64_t>(b, data.size());
        b.insert(b.end(), data.begin(), data.end());
        _put<uint32_t>(b, _fnv1a(b.data(), b.size()));
        _write_atomic(_snap_path(), b);
    }

    RaftConfig  cfg_;
    ApplyFn     apply_fn_;
    SnapshotFn  snapshot_fn_;
    RestoreFn   restore_fn_;
    PeerFactory peer_factory_;

    // Persistent state
    Term        current_term_ = 0;
    NodeId      voted_for_    = -1;
    std::vector<LogEntry> log_;          // log_[0] is a sentinel for snap_index_
    Index       snap_index_   = 0;
    Term        snap_term_    = 0;
    Membership  snap_config_;
    std::shared_ptr<const std::vector<uint8_t>> snap_data_;

    // Configuration
    Membership  base_config_;
    std::vector<std::pair<Index, Membership>> log_configs_;   // CONFIG entries in log_
    Membership  config_;                 // latest configuration (committed or not)
    Index       config_index_ = 0;
    std::map<NodeId, Member> learners_;

    // Volatile state
    RaftRole    role_         = RaftRole::FOLLOWER;
    Index       commit_index_ = 0;
    Index       last_applied_ = 0;
    NodeId      leader_id_    = -1;
    std::set<NodeId> votes_, prevotes_;
    bool        prevoting_    = false;
    uint64_t    prevote_round_ = 0;
    uint64_t    hb_round_     = 0;
    Clock::time_point last_leader_contact_{};
    Clock::time_point leader_since_{};
    Clock::time_point io_fail_since_{};   // first of a run of failed leader appends
    Clock::time_point last_err_log_{};
    uint64_t          suppressed_errs_ = 0;

    // Snapshot being received
    std::vector<uint8_t> recv_buf_;
    Index recv_index_ = 0;
    Term  recv_term_  = 0;

    std::map<NodeId, std::unique_ptr<PeerState>> peers_;
    std::unordered_map<NodeId, std::shared_ptr<RaftPeer>> transports_;
    std::vector<std::thread> finished_threads_;
    std::FILE* log_fp_ = nullptr;

    mutable std::mutex  mu_;
    std::condition_variable cv_, apply_cv_, peer_cv_, progress_cv_;

    Clock::time_point election_deadline_;
    bool election_reset_ = false;
    bool running_ = false;
    std::mt19937 rng_;

    std::thread election_thread_;
    std::thread apply_thread_;
};

// ================================================================
// LocalPeer — in-process transport (direct method calls)
// ================================================================
class LocalPeer : public RaftPeer {
public:
    explicit LocalPeer(RaftNode* node) : node_(node) {}
    std::optional<AppendEntriesReply>
    append_entries(const AppendEntriesArgs& a) override { return node_->on_append_entries(a); }
    std::optional<RequestVoteReply>
    request_vote(const RequestVoteArgs& a) override { return node_->on_request_vote(a); }
    std::optional<InstallSnapshotReply>
    install_snapshot(const InstallSnapshotArgs& a) override { return node_->on_install_snapshot(a); }
private:
    RaftNode* node_;
};

inline void RaftNode::register_peer(NodeId id, RaftNode* peer) {
    register_peer(id, std::make_shared<LocalPeer>(peer));
}

// ================================================================
// RaftGroup — a cluster of in-process RaftNode instances
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
        for (int i = 0; i < n_nodes; ++i)
            for (int j = 0; j < n_nodes; ++j)
                if (i != j)
                    nodes_[i]->register_peer(j, nodes_[j].get());
    }

    ~RaftGroup() { stop(); }

    void start() {
        for (auto& n : nodes_) n->start();
    }

    // Stop every node before any is destroyed: replicator threads hold
    // raw pointers to their peers.
    void stop() {
        for (auto& n : nodes_) n->stop();
    }

    bool wait_for_leader(int timeout_ms = 5000) const {
        auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(timeout_ms);
        while (!leader()) {
            if (std::chrono::steady_clock::now() >= deadline) return false;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        return true;
    }

    // Submit to leader (auto-discover leader). Retries span roughly one
    // election timeout so a write issued mid-election still lands.
    std::optional<Index> submit(const std::string& op,
                                 const std::vector<uint8_t>& payload,
                                 int retries = 25) {
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

    // Wait for index to be committed on every node
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

    const RaftNode& node(int i) const { return *nodes_.at(i); }

    void print_state() const {
        std::cout << "\n=== Raft Cluster State ===\n";
        for (auto& n : nodes_) n->print_state();
    }

private:
    std::vector<std::unique_ptr<RaftNode>> nodes_;
};

} // namespace consensus
} // namespace tidevec
