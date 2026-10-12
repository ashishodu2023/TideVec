// ================================================================
// test_raft_cluster.cpp — in-process tests for the Raft extensions
//   - log snapshots, compaction and chunked InstallSnapshot
//   - restart from a persisted snapshot
//   - single-server membership changes (add via learner, remove leader)
//   - ReadIndex / lease reads, check-quorum step-down
//   - pre-vote (a partitioned node does not inflate its term)
//   - disk-full faults (no acknowledgement without persistence)
//   - multi-Raft placement map (multi_raft_router.hpp)
// Nodes talk through SimPeer, which can partition any directed link.
// ================================================================

#include <tidevec/consensus/raft.hpp>
#include <tidevec/consensus/raft_http.hpp>
#include <tidevec/cluster/multi_raft_router.hpp>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace tidevec::consensus;
namespace fs = std::filesystem;
using namespace std::chrono_literals;

// ---- harness -------------------------------------------------
static int tests_run = 0, tests_passed = 0;
#define TEST(name) static void test_##name();\
    struct _r_##name{_r_##name(){run_test(#name,test_##name);}} _i_##name;\
    static void test_##name()
static bool g_failed = false;
#define ASSERT(c) do{if(!(c)){std::cerr<<"  FAIL: "#c" line "<<__LINE__<<"\n";g_failed=true;return;}}while(0)
static void run_test(const char* n, void (*f)()) {
    if (const char* filt = std::getenv("TIDEVEC_TEST_FILTER"); filt && !std::strstr(n, filt)) return;
    ++tests_run;
    std::cout << "[TEST] " << n << " ... " << std::flush;
    g_failed = false;
    f();
    if (g_failed) { std::cout << "FAILED\n"; return; }
    ++tests_passed;
    std::cout << "PASS\n";
}

template <typename F>
static bool eventually(F pred, std::chrono::milliseconds timeout = 5000ms) {
    auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (pred()) return true;
        std::this_thread::sleep_for(5ms);
    }
    return pred();
}

// ---- simulated network ----------------------------------------
struct Net {
    std::mutex mu;
    std::set<std::pair<int, int>> blocked;   // directed (from, to)
    bool cut(int a, int b) { std::lock_guard l(mu); return blocked.count({a, b}) > 0; }
    void isolate(int n, const std::vector<int>& others) {
        std::lock_guard l(mu);
        for (int o : others) { blocked.insert({n, o}); blocked.insert({o, n}); }
    }
    void heal() { std::lock_guard l(mu); blocked.clear(); }
};

struct SimPeer : RaftPeer {
    SimPeer(Net& net, int from, int to, RaftNode* node) : net(net), from(from), to(to), node(node) {}
    bool down() {
        if (net.cut(from, to)) { std::this_thread::sleep_for(2ms); return true; }
        return false;
    }
    std::optional<AppendEntriesReply> append_entries(const AppendEntriesArgs& a) override {
        if (down()) return std::nullopt;
        auto r = node->on_append_entries(a);
        if (net.cut(to, from)) return std::nullopt;
        return r;
    }
    std::optional<RequestVoteReply> request_vote(const RequestVoteArgs& a) override {
        if (down()) return std::nullopt;
        auto r = node->on_request_vote(a);
        if (net.cut(to, from)) return std::nullopt;
        return r;
    }
    std::optional<InstallSnapshotReply> install_snapshot(const InstallSnapshotArgs& a) override {
        if (down()) return std::nullopt;
        auto r = node->on_install_snapshot(a);
        if (net.cut(to, from)) return std::nullopt;
        return r;
    }
    Net& net;
    int from, to;
    RaftNode* node;
};

// ---- key/value state machine ----------------------------------
struct Kv {
    std::mutex mu;
    std::map<std::string, std::string> data;
    std::atomic<int> restores{0};

    void apply(const LogEntry& e) {
        std::string s(e.payload.begin(), e.payload.end());
        auto eq = s.find('=');
        std::lock_guard l(mu);
        data[s.substr(0, eq)] = s.substr(eq + 1);
    }
    std::vector<uint8_t> snapshot() {
        std::lock_guard l(mu);
        std::string s;
        for (auto& [k, v] : data) s += k + "=" + v + "\n";
        return {s.begin(), s.end()};
    }
    void restore(const std::vector<uint8_t>& b, Index) {
        std::lock_guard l(mu);
        data.clear();
        std::istringstream in(std::string(b.begin(), b.end()));
        std::string line;
        while (std::getline(in, line)) {
            auto eq = line.find('=');
            data[line.substr(0, eq)] = line.substr(eq + 1);
        }
        ++restores;
    }
    std::map<std::string, std::string> get() { std::lock_guard l(mu); return data; }
};

struct Cluster {
    Net net;
    std::vector<std::unique_ptr<Kv>> kv;
    std::vector<std::unique_ptr<RaftNode>> nodes;
    std::vector<std::shared_ptr<FaultInjector>> faults;
    RaftConfig base;

    explicit Cluster(int n, RaftConfig b = {}, const std::string& dir = "") : base(b) {
        Membership members;
        for (int i = 0; i < n; ++i) members.push_back({i, ""});
        for (int i = 0; i < n; ++i) add_node(i, members, false, dir);
    }

    RaftNode& add_node(int id, const Membership& members, bool join, const std::string& dir = "") {
        RaftConfig c = base;
        c.node_id = id;
        c.members = join ? Membership{} : members;
        c.join    = join;
        if (!dir.empty()) c.persist_dir = dir + "/n" + std::to_string(id);
        if ((int)faults.size() <= id) faults.resize(id + 1);
        faults[id] = std::make_shared<FaultInjector>();
        c.fault = faults[id];
        if ((int)kv.size() <= id) { kv.resize(id + 1); nodes.resize(id + 1); }
        kv[id] = std::make_unique<Kv>();
        Kv* k = kv[id].get();
        nodes[id] = std::make_unique<RaftNode>(c, [k](const LogEntry& e) { k->apply(e); });
        nodes[id]->set_snapshot_handlers([k] { return k->snapshot(); },
                                         [k](const std::vector<uint8_t>& b, Index i) { k->restore(b, i); });
        // Wire every pair of existing nodes (new ones get factories too).
        for (int j = 0; j < (int)nodes.size(); ++j) {
            if (!nodes[j] || j == id) continue;
            nodes[id]->register_peer(j, std::make_shared<SimPeer>(net, id, j, nodes[j].get()));
            nodes[j]->register_peer(id, std::make_shared<SimPeer>(net, j, id, nodes[id].get()));
        }
        return *nodes[id];
    }

    void start() { for (auto& n : nodes) if (n) n->start(); }
    ~Cluster() { for (auto& n : nodes) if (n) n->stop(); }

    RaftNode* leader(int exclude = -1) {
        for (int i = 0; i < (int)nodes.size(); ++i)
            if (nodes[i] && i != exclude && nodes[i]->is_leader()) return nodes[i].get();
        return nullptr;
    }
    RaftNode* wait_leader(int exclude = -1) {
        RaftNode* l = nullptr;
        eventually([&] { return (l = leader(exclude)) != nullptr; });
        return l;
    }
    std::optional<Index> set(const std::string& k, const std::string& v) {
        std::string s = k + "=" + v;
        for (int attempt = 0; attempt < 100; ++attempt) {
            if (auto* l = leader())
                if (auto idx = l->submit("SET", {s.begin(), s.end()})) return idx;
            std::this_thread::sleep_for(10ms);
        }
        return std::nullopt;
    }

    // Like set(), but retries until the entry is committed in the term it
    // was appended in (a stale leader's uncommitted entries can be lost).
    bool set_committed(const std::string& k, const std::string& v) {
        std::string s = k + "=" + v;
        for (int attempt = 0; attempt < 50; ++attempt) {
            if (auto* l = leader()) {
                const Term t = l->current_term();
                std::optional<Index> idx;
                try { idx = l->submit("SET", {s.begin(), s.end()}); } catch (const std::exception&) {}
                if (idx && l->wait_committed(*idx, 2000) && l->term_at(*idx) == t) return true;
            }
            std::this_thread::sleep_for(20ms);
        }
        return false;
    }
};

static RaftConfig fast_cfg() {
    RaftConfig c;
    c.election_timeout_min_ms = 80;
    c.election_timeout_max_ms = 160;
    c.heartbeat_interval_ms   = 20;
    return c;
}

// ================================================================
// Snapshots
// ================================================================

TEST(snapshot_compacts_log_on_every_node) {
    auto cfg = fast_cfg();
    cfg.snapshot_threshold = 20;
    Cluster c(3, cfg);
    c.start();
    ASSERT(c.wait_leader());
    std::optional<Index> last;
    for (int i = 0; i < 100; ++i) last = c.set("k" + std::to_string(i % 7), std::to_string(i));
    ASSERT(last && c.leader()->wait_committed(*last, 3000));
    ASSERT(eventually([&] {
        for (auto& n : c.nodes) if (n->last_applied() < *last || n->snapshot_index() < 80) return false;
        return true;
    }));
    for (auto& n : c.nodes) ASSERT(n->log_size() - n->snapshot_index() <= 25);
    ASSERT(c.kv[0]->get() == c.kv[1]->get() && c.kv[1]->get() == c.kv[2]->get());
    ASSERT(c.kv[0]->get().at("k6") == "97");
}

TEST(install_snapshot_catches_up_lagging_follower_in_chunks) {
    auto cfg = fast_cfg();
    cfg.snapshot_threshold   = 25;
    cfg.snapshot_chunk_bytes = 64;     // many chunks
    Cluster c(3, cfg);
    c.start();
    RaftNode* l = c.wait_leader();
    ASSERT(l);
    int lag = (l->node_id() + 1) % 3;
    c.net.isolate(lag, {0, 1, 2});
    std::optional<Index> last;
    for (int i = 0; i < 200; ++i) last = c.set("key" + std::to_string(i % 13), "v" + std::to_string(i));
    ASSERT(last);
    ASSERT(eventually([&] { return c.leader() && c.leader()->snapshot_index() > 100; }));
    ASSERT(c.nodes[lag]->last_applied() < 50);
    c.net.heal();
    ASSERT(eventually([&] { return c.nodes[lag]->last_applied() >= *last; }, 8000ms));
    ASSERT(c.kv[lag]->restores.load() >= 1);
    ASSERT(c.nodes[lag]->snapshot_index() > 100);
    int other = (lag + 1) % 3;
    ASSERT(eventually([&] { return c.kv[lag]->get() == c.kv[other]->get(); }));
    ASSERT(c.kv[lag]->get().at("key4") == "v199");
}

TEST(restart_restores_persisted_snapshot_then_log_suffix) {
    fs::path dir = fs::temp_directory_path() / "tidevec_raft_snap_restart";
    fs::remove_all(dir);
    auto cfg = fast_cfg();
    cfg.snapshot_threshold = 10;
    // fsync per append: leave headroom so a loaded disk does not cause churn.
    cfg.election_timeout_min_ms = 300;
    cfg.election_timeout_max_ms = 600;
    std::map<std::string, std::string> before;
    Index last_idx = 0;
    {
        Cluster c(3, cfg, dir.string());
        c.start();
        ASSERT(c.wait_leader());
        for (int i = 0; i < 45; ++i) ASSERT(c.set_committed("a" + std::to_string(i % 5), std::to_string(i)));
        ASSERT(eventually([&] {
            const Index n0 = c.nodes[0]->log_size();
            for (std::size_t i = 0; i < c.nodes.size(); ++i) {
                auto kv = c.kv[i]->get();
                if (!kv.count("a4") || kv.at("a4") != "44" || c.nodes[i]->log_size() != n0 ||
                    c.nodes[i]->last_applied() + 1 != n0) return false;
            }
            return true;
        }, 15000ms));
        before = c.kv[0]->get();
        last_idx = c.nodes[0]->log_size() - 1;
        ASSERT(last_idx >= 45);
        for (auto& n : c.nodes) ASSERT(n->snapshot_index() + 2 * cfg.snapshot_threshold > last_idx);
    }
    ASSERT(fs::exists(dir / "n0" / "raft_snapshot"));
    // Fresh state machines: everything comes back from snapshot + log.
    Cluster c(3, cfg, dir.string());
    for (auto& n : c.nodes)
        ASSERT(n->snapshot_index() >= 10 && n->log_size() == last_idx + 1);
    c.start();
    ASSERT(c.wait_leader());
    ASSERT(eventually([&] {
        for (auto& k : c.kv) if (k->get() != before) return false;
        return true;
    }));
    for (auto& k : c.kv) ASSERT(k->restores.load() == 1);
    fs::remove_all(dir);
}

TEST(corrupt_snapshot_refuses_to_start) {
    fs::path dir = fs::temp_directory_path() / "tidevec_raft_snap_corrupt";
    fs::remove_all(dir);
    auto cfg = fast_cfg();
    cfg.snapshot_threshold = 5;
    cfg.node_id = 0;
    cfg.n_nodes = 1;
    cfg.persist_dir = dir.string();
    {
        Kv kv;
        RaftNode n(cfg, [&](const LogEntry& e) { kv.apply(e); });
        n.set_snapshot_handlers([&] { return kv.snapshot(); },
                                [&](const std::vector<uint8_t>& b, Index i) { kv.restore(b, i); });
        n.start();
        ASSERT(eventually([&] { return n.is_leader(); }));
        std::optional<Index> last;
        for (int i = 0; i < 12; ++i) { std::string s = "x=" + std::to_string(i); last = n.submit("SET", {s.begin(), s.end()}); }
        ASSERT(last && n.wait_committed(*last));
        ASSERT(eventually([&] { return n.snapshot_index() >= 5; }));
        n.stop();
    }
    { std::fstream f(dir / "raft_snapshot", std::ios::in | std::ios::out | std::ios::binary);
      f.seekp(30); f.put('\x7f'); }
    bool threw = false;
    try { RaftNode n(cfg, [](const LogEntry&) {}); } catch (const std::runtime_error&) { threw = true; }
    ASSERT(threw);
    fs::remove_all(dir);
}

// ================================================================
// Membership changes
// ================================================================

TEST(add_server_via_learner_then_remove_leader) {
    auto cfg = fast_cfg();
    cfg.snapshot_threshold = 30;
    Cluster c(3, cfg);
    c.start();
    ASSERT(c.wait_leader());
    std::optional<Index> last;
    for (int i = 0; i < 80; ++i) last = c.set("m" + std::to_string(i % 4), std::to_string(i));
    ASSERT(last && c.leader()->wait_committed(*last, 3000));

    // Node 3 joins empty and learns everything (via snapshot) before voting.
    Membership m3{{0, ""}, {1, ""}, {2, ""}};
    auto& n3 = c.add_node(3, m3, /*join=*/true);
    n3.start();
    std::this_thread::sleep_for(300ms);
    ASSERT(!n3.is_leader() && n3.status().members.empty());
    RaftResult r = RaftResult::fail("no leader");
    ASSERT(eventually([&] {
        auto* l = c.leader();
        if (!l) return false;
        r = l->add_server({3, ""});
        return r.success;
    }, 8000ms));
    ASSERT(eventually([&] { return n3.status().members.size() == 4; }));
    ASSERT(eventually([&] { return c.kv[3]->get() == c.kv[0]->get(); }));
    ASSERT(c.kv[3]->restores.load() >= 1);

    // Remove the leader itself: it steps down, the rest elect a new one.
    RaftNode* old = c.leader();
    ASSERT(old);
    int old_id = old->node_id();
    ASSERT(old->remove_server(old_id).success);
    ASSERT(eventually([&] { return !old->is_leader(); }));
    RaftNode* nl = c.wait_leader(old_id);
    ASSERT(nl && nl->node_id() != old_id);
    ASSERT(nl->status().members.size() == 3);
    auto idx = c.set("after", "removal");
    ASSERT(idx && nl->wait_committed(*idx, 3000));
    // The removed node never campaigns again.
    Term t = old->current_term();
    std::this_thread::sleep_for(500ms);
    ASSERT(!old->is_leader() && old->current_term() == t);
    ASSERT(eventually([&] { return c.kv[3]->get().count("after") == 1; }));
}

TEST(membership_change_rejected_while_one_is_pending_or_on_follower) {
    Cluster c(3, fast_cfg());
    c.start();
    RaftNode* l = c.wait_leader();
    ASSERT(l);
    for (auto& n : c.nodes)
        if (n.get() != l) ASSERT(!n->remove_server(l->node_id()).success);
    // Removing a node that is not a member is a no-op success.
    ASSERT(l->remove_server(42).success);
    ASSERT(!l->add_server({l->node_id() == 0 ? 1 : 0, "elsewhere"}).success);
}

// ================================================================
// Reads, pre-vote, check-quorum
// ================================================================

TEST(read_index_needs_quorum_and_isolated_leader_steps_down) {
    Cluster c(3, fast_cfg());
    c.start();
    RaftNode* l = c.wait_leader();
    ASSERT(l);
    auto idx = c.set("r", "1");
    ASSERT(idx && l->wait_committed(*idx));
    auto ri = l->read_index(1000);
    ASSERT(ri && *ri >= *idx && l->last_applied() >= *ri);
    ASSERT(l->lease_read(1000).has_value());
    for (auto& n : c.nodes) if (n.get() != l) ASSERT(!n->read_index(200).has_value());

    int lid = l->node_id();
    c.net.isolate(lid, {0, 1, 2});
    ASSERT(!l->read_index(300).has_value());
    ASSERT(eventually([&] { return !l->is_leader(); }, 2000ms));   // check-quorum
    RaftNode* nl = c.wait_leader(lid);
    ASSERT(nl);
    auto idx2 = c.set("r", "2");
    ASSERT(idx2 && nl->wait_committed(*idx2));
    auto ri2 = nl->read_index(1000);
    ASSERT(ri2 && *ri2 >= *idx2);
    ASSERT(c.kv[nl->node_id()]->get().at("r") == "2");
}

TEST(pre_vote_keeps_partitioned_node_from_disrupting) {
    Cluster c(3, fast_cfg());
    c.start();
    RaftNode* l = c.wait_leader();
    ASSERT(l);
    std::this_thread::sleep_for(200ms);
    Term term = l->current_term();
    int f = (l->node_id() + 1) % 3;
    c.net.isolate(f, {0, 1, 2});
    std::this_thread::sleep_for(1500ms);   // ~10 election timeouts
    ASSERT(c.nodes[f]->current_term() == term);
    c.net.heal();
    std::this_thread::sleep_for(500ms);
    ASSERT(l->is_leader() && l->current_term() == term);
}

TEST(candidate_retries_after_failed_election) {
    Cluster c(3, fast_cfg());
    c.start();
    RaftNode* l = c.wait_leader();
    ASSERT(l);
    const int old = l->node_id(), a = (old + 1) % 3, d = (old + 2) % 3;
    c.faults[d]->disk_full = true;   // grants pre-votes but cannot persist a vote
    c.net.isolate(old, {0, 1, 2});
    ASSERT(eventually([&] { return c.nodes[a]->role() == RaftRole::CANDIDATE; }));
    const Term t0 = c.nodes[a]->current_term();
    std::this_thread::sleep_for(1000ms);    // several election timeouts
    ASSERT(c.nodes[a]->current_term() >= t0 + 2);
    c.faults[d]->disk_full = false;
    ASSERT(eventually([&] { return c.nodes[a]->is_leader() || c.nodes[d]->is_leader(); }));
    c.net.heal();
}

TEST(disk_full_follower_never_acknowledges_then_recovers) {
    Cluster c(3, fast_cfg());
    c.start();
    RaftNode* l = c.wait_leader();
    ASSERT(l);
    int f = (l->node_id() + 1) % 3;
    std::this_thread::sleep_for(100ms);
    Index before = c.nodes[f]->log_size();
    c.faults[f]->disk_full = true;
    std::optional<Index> last;
    for (int i = 0; i < 20; ++i) last = c.set("d", std::to_string(i));
    ASSERT(last && l->wait_committed(*last, 3000));    // the other follower suffices
    std::this_thread::sleep_for(200ms);
    ASSERT(c.nodes[f]->log_size() == before);
    // The leader cannot commit without a second healthy disk.
    int g = 3 - l->node_id() - f;
    c.faults[g]->disk_full = true;
    auto stuck = l->submit("SET", {'d', '=', 'x'});
    ASSERT(stuck && !l->wait_committed(*stuck, 400));
    c.faults[f]->disk_full = false;
    c.faults[g]->disk_full = false;
    ASSERT(eventually([&] { return c.nodes[f]->last_applied() >= *stuck; }));
    ASSERT(c.kv[f]->get().at("d") == "x");
}

TEST(leader_disk_full_rejects_submit) {
    Cluster c(3, fast_cfg());
    c.start();
    RaftNode* l = c.wait_leader();
    ASSERT(l);
    c.faults[l->node_id()]->disk_full = true;
    bool threw = false;
    try { l->submit("SET", {'a', '=', 'b'}); } catch (const std::runtime_error&) { threw = true; }
    ASSERT(threw);
    c.faults[l->node_id()]->disk_full = false;
}

TEST(leader_with_persistently_full_disk_steps_down) {
    Cluster c(3, fast_cfg());
    c.start();
    RaftNode* l = c.wait_leader();
    ASSERT(l);
    const int old = l->node_id();
    c.faults[old]->disk_full = true;
    // Appends keep failing; after one minimum election timeout it steps down.
    ASSERT(eventually([&] {
        try { l->submit("SET", {'a', '=', 'b'}); } catch (const std::runtime_error&) {}
        return !l->is_leader();
    }));
    // A healthy node takes over (the full-disk node cannot persist a new
    // term, so it cannot win) and writes succeed again.
    std::optional<Index> idx;
    ASSERT(eventually([&] {
        try { idx = c.set("after", "1"); } catch (const std::runtime_error&) {}
        return idx.has_value();
    }));
    for (auto& n : c.nodes)
        if (n->is_leader()) ASSERT(n->node_id() != old);
    c.faults[old]->disk_full = false;
    ASSERT(eventually([&] {
        return c.kv[old]->get().count("after") && c.kv[old]->get().at("after") == "1";
    }));
}

TEST(clock_skew_fast_timer_node_does_not_break_safety) {
    Cluster c(3, fast_cfg());
    c.faults[0]->set_timer_scale(0.3);   // node 0's clock runs ~3x fast
    c.start();
    std::map<Term, std::set<int>> leaders;
    std::optional<Index> last;
    for (int i = 0; i < 50; ++i) {
        last = c.set("s", std::to_string(i));
        for (int j = 0; j < 3; ++j)
            if (c.nodes[j]->is_leader()) leaders[c.nodes[j]->current_term()].insert(j);
    }
    ASSERT(last);
    for (auto& [t, ids] : leaders) ASSERT(ids.size() == 1);
    ASSERT(eventually([&] {
        return c.kv[0]->get() == c.kv[1]->get() && c.kv[1]->get() == c.kv[2]->get() &&
               c.kv[0]->get().count("s") && c.kv[0]->get().at("s") == "49";
    }));
}

TEST(install_snapshot_and_prevote_json_roundtrip) {
    InstallSnapshotArgs a{7, 2, 100, 6, {{0, "http://a:1"}, {2, "http://c:3"}}, 64, {1, 2, 3, 250}, true};
    auto b = snapshot_args_from_json(to_json(a));
    ASSERT(b.term == 7 && b.leader_id == 2 && b.last_included_index == 100 &&
           b.last_included_term == 6 && b.config == a.config && b.offset == 64 &&
           b.data == a.data && b.done);
    auto r = snapshot_reply_from_json(to_json(InstallSnapshotReply{9, true, 68}));
    ASSERT(r.term == 9 && r.success && r.next_offset == 68);
    RequestVoteArgs v{3, 1, 10, 2, true};
    ASSERT(vote_args_from_json(to_json(v)).pre_vote);
    rjson legacy = {{"term", 3}, {"candidate_id", 1}, {"last_log_index", 1}, {"last_log_term", 1}};
    ASSERT(!vote_args_from_json(legacy).pre_vote);
    ASSERT(decode_membership(encode_membership(a.config)) == a.config);
}

// ================================================================
// Multi-Raft placement map
// ================================================================

TEST(shard_placement_covers_hash_space_and_spreads_ids) {
    using namespace tidevec::cluster;
    auto p = Placement::uniform({{"http://a"}, {"http://b"}, {"http://c"}});
    ASSERT(p.groups.front().lo == 0 && p.groups.back().hi == UINT64_MAX);
    std::vector<int> per(3, 0);
    for (int i = 0; i < 3000; ++i) ++per[p.owner("doc-" + std::to_string(i))];
    for (int n : per) ASSERT(n > 800 && n < 1200);
    // Round trip through JSON keeps every owner.
    auto q = Placement::from_json(p.to_json());
    for (int i = 0; i < 100; ++i) {
        auto id = "x" + std::to_string(i);
        ASSERT(p.owner(id) == q.owner(id));
    }
    auto gap = p.to_json();
    gap["groups"][1]["range"][0] = "0x0000000000000005";
    bool threw = false;
    try { Placement::from_json(gap); } catch (const std::invalid_argument&) { threw = true; }
    ASSERT(threw);
    auto uneven = tidevec::cluster::rjson::parse(R"({"groups":[
        {"id":"hot","replicas":["http://a"],"range":["0x0","0x0fffffffffffffff"]},
        {"id":"rest","replicas":["http://b"],"range":["0x1000000000000000","0xffffffffffffffff"]}]})");
    auto u = Placement::from_json(uneven);
    ASSERT(u.groups[0].id == "hot" && u.owner("doc-1") < 2);
}

int main() {
    std::cout << "\n--- Raft cluster tests: " << tests_passed << " / " << tests_run
              << " passed ---\n";
    return tests_passed == tests_run ? 0 : 1;
}
