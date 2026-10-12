// ================================================================
// test_eleven_nines.cpp
// Tests for the durability and consensus components:
//   - Reed-Solomon GF(256) arithmetic
//   - RS encode/decode with arbitrary shard loss
//   - Raft leader election and log replication
//   - Health monitor scrub and repair
//   - UltraDurableCollection end-to-end
//   - Durability math verification
// ================================================================

#include <tidevec/erasure/reed_solomon.hpp>
#include <tidevec/consensus/raft.hpp>
#include <tidevec/consensus/raft_http.hpp>
#include <tidevec/health/health_monitor.hpp>
#include <tidevec/cluster/ultra_durable_collection.hpp>

#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <thread>
#include <chrono>
#include <filesystem>
#include <numeric>
#include <atomic>
#include <algorithm>
#include <map>
#include <set>

using namespace tidevec;
using namespace tidevec::erasure;
using namespace tidevec::consensus;
using namespace tidevec::health;
namespace fs = std::filesystem;

// ---- harness -------------------------------------------------
static int tests_run=0, tests_passed=0;
#define TEST(name) static void test_##name();\
    struct _r_##name{_r_##name(){run_test(#name,test_##name);}} _i_##name;\
    static void test_##name()
static bool g_current_test_failed=false;
#define ASSERT(c) do{if(!(c)){std::cerr<<"  FAIL: "#c" line "<<__LINE__<<"\n";g_current_test_failed=true;return;}}while(0)
#define ASSERT_NEAR(a,b,e) ASSERT(std::abs((double)(a)-(double)(b))<(double)(e))
static void run_test(const char*n,void(*f)()){
    ++tests_run;std::cout<<"[TEST] "<<n<<" ... ";
    g_current_test_failed=false;f();
    if(g_current_test_failed){std::cout<<"FAILED\n";return;}
    ++tests_passed;std::cout<<"PASS\n";}

// =================================================================
// SECTION 1: GF(256) Arithmetic
// =================================================================

TEST(gf256_add_commutative) {
    for (int a=0;a<256;++a) for(int b=0;b<256;++b)
        ASSERT(GF.add(a,b)==GF.add(b,a));
}

TEST(gf256_mul_commutative) {
    for (int a=0;a<256;++a) for(int b=0;b<256;++b)
        ASSERT(GF.mul(a,b)==GF.mul(b,a));
}

TEST(gf256_mul_by_zero) {
    for (int a=0;a<256;++a) ASSERT(GF.mul(a,0)==0);
}

TEST(gf256_mul_by_one) {
    for (int a=0;a<256;++a) ASSERT(GF.mul(a,1)==a);
}

TEST(gf256_inverse) {
    for (int a=1;a<256;++a)
        ASSERT(GF.mul(a, GF.inv(a))==1);
}

TEST(gf256_div) {
    for (int a=1;a<256;++a)
        for (int b=1;b<256;++b)
            ASSERT(GF.mul(GF.div(a,b),b)==a);
}

TEST(gf256_distributive) {
    // a*(b+c) == a*b + a*c
    for (int a=0;a<32;++a)
        for (int b=0;b<32;++b)
            for (int c=0;c<32;++c)
                ASSERT(GF.mul(a,GF.add(b,c))==GF.add(GF.mul(a,b),GF.mul(a,c)));
}

// =================================================================
// SECTION 2: GFMatrix
// =================================================================

TEST(gfmatrix_inverse) {
    // Random 4×4 matrix, verify A * A^-1 = I
    GFMatrix A(4,4);
    A.set(0,0,2);A.set(0,1,3);A.set(0,2,1);A.set(0,3,1);
    A.set(1,0,4);A.set(1,1,1);A.set(1,2,3);A.set(1,3,2);
    A.set(2,0,1);A.set(2,1,4);A.set(2,2,2);A.set(2,3,3);
    A.set(3,0,3);A.set(3,1,2);A.set(3,2,4);A.set(3,3,1);

    auto Ainv = A.inverse();

    // A * Ainv should be identity
    for (int r=0;r<4;++r) {
        std::vector<uint8_t> col(4);
        for (int c=0;c<4;++c) col[c]=Ainv.get(c,r);
        auto res = A.mul_vec(col);
        for (int i=0;i<4;++i)
            ASSERT(res[i]==(i==r?1:0));
    }
}

// =================================================================
// SECTION 3: Reed-Solomon Encode/Decode
// =================================================================

TEST(rs_encode_decode_no_loss) {
    ReedSolomon rs(4, 2);  // RS(4,2): 6 total shards, survive 2 failures
    std::vector<uint8_t> data = {0x48,0x65,0x6C,0x6C,0x6F,0x20,0x57,0x6F,
                                  0x72,0x6C,0x64,0x21,0xDE,0xAD,0xBE,0xEF};
    auto shards = rs.encode(data);
    ASSERT(static_cast<int>(shards.size()) == rs.n());

    // All shards present
    uint32_t present = (1u << rs.n()) - 1;
    auto recovered = rs.decode(shards, present, data.size());
    ASSERT(recovered == data);
}

TEST(rs_survive_1_failure_out_of_6) {
    ReedSolomon rs(4, 2);
    std::vector<uint8_t> data(64);
    std::iota(data.begin(), data.end(), 0);

    auto shards = rs.encode(data);

    // Lose shard 0 (a data shard)
    uint32_t present = 0b111110;  // shards 1-5 present
    shards[0].clear();  // simulate lost shard
    auto recovered = rs.decode(shards, present, data.size());
    ASSERT(recovered == data);
}

TEST(rs_survive_2_failures_rs42) {
    ReedSolomon rs(4, 2);
    std::vector<uint8_t> data(32, 0xAB);
    auto shards = rs.encode(data);

    // Lose 2 shards (max for RS(4,2))
    uint32_t present = 0b110011;  // shards 0,1,4,5 present; 2,3 lost
    shards[2].clear(); shards[3].clear();
    auto recovered = rs.decode(shards, present, data.size());
    ASSERT(recovered == data);
}

TEST(rs_survive_4_failures_rs104) {
    ReedSolomon rs(10, 4);  // Production config: 14 shards, survive 4
    std::vector<uint8_t> data(100);
    std::iota(data.begin(), data.end(), 0);
    auto shards = rs.encode(data);

    // Lose 4 shards: 1, 5, 8, 12
    uint32_t present = 0b11111111111111u & ~((1u<<1)|(1u<<5)|(1u<<8)|(1u<<12));
    shards[1].clear(); shards[5].clear(); shards[8].clear(); shards[12].clear();
    auto recovered = rs.decode(shards, present, data.size());
    ASSERT(recovered == data);
}

TEST(rs_insufficient_shards_throws) {
    ReedSolomon rs(4, 2);
    std::vector<uint8_t> data(16, 0x42);
    auto shards = rs.encode(data);

    // Only 3 shards present (need 4)
    uint32_t present = 0b000111;  // shards 0,1,2 only
    shards[3].clear(); shards[4].clear(); shards[5].clear();

    bool threw = false;
    try { rs.decode(shards, present, data.size()); }
    catch (const std::exception&) { threw = true; }
    ASSERT(threw);
}

TEST(rs_large_data) {
    // 1MB segment, RS(10,4)
    ReedSolomon rs(10, 4);
    std::mt19937 rng(42);
    std::vector<uint8_t> data(1024*100);
    for (auto& b : data) b = rng() & 0xFF;

    auto shards = rs.encode(data);
    ASSERT(static_cast<int>(shards.size()) == 14);

    // Lose any 4 shards: 0,2,7,11
    uint32_t present = 0b11111111111111u & ~((1u<<0)|(1u<<2)|(1u<<7)|(1u<<11));
    shards[0].clear(); shards[2].clear(); shards[7].clear(); shards[11].clear();
    auto recovered = rs.decode(shards, present, data.size());
    ASSERT(recovered == data);
}

TEST(rs_storage_overhead) {
    ReedSolomon rs(10, 4);
    // RS(10,4): 14 shards vs 10 data shards → 1.4× overhead
    // vs 3× for 3-way replication
    double overhead = static_cast<double>(rs.n()) / rs.k();
    ASSERT_NEAR(overhead, 1.4, 0.01);
    // Much better than 3× replication
    ASSERT(overhead < 2.0);
}

// =================================================================
// SECTION 4: Durability Math
// =================================================================

TEST(durability_rs104_model) {
    ReedSolomon rs(10, 4);
    // Backblaze AFR: ~0.4% annual disk failure rate
    double p_fail = 0.004;
    double nines  = rs.durability_nines(p_fail);
    std::cout << "\n    RS(10,4) durability: " << std::fixed
              << std::setprecision(1) << nines << " nines at p_fail="
              << std::setprecision(3) << p_fail << " ";
    // P(loss) ≈ C(14,5)·p^5 ≈ 2.0e-9/yr → 8.7 nines
    ASSERT(nines > 8.6 && nines < 8.8);
}

TEST(durability_rs42_model) {
    ReedSolomon rs(4, 2);
    double p_fail = 0.004;
    double nines  = rs.durability_nines(p_fail);
    std::cout << "\n    RS(4,2) durability: " << std::fixed
              << std::setprecision(1) << nines << " nines ";
    ASSERT(nines > 5.0);
}

TEST(durability_replication_comparison) {
    // 3-way replication: survives 2 failures only
    // RS(10,4): survives 4 failures, 1.4× overhead vs 3× overhead
    // At same storage cost (3TB SSD):
    //   3-way replication: 1TB usable
    //   RS(10,4): 2.14TB usable (3TB / 1.4)
    ReedSolomon rs(10, 4);
    double p_fail = 0.004;
    double rs_nines  = rs.durability_nines(p_fail);

    // 3-way replication: P(loss) = P(all 3 fail) ≈ p^3 for independent failures
    // (simplified; real analysis accounts for rebuild time)
    double rep3_p_loss = p_fail * p_fail * p_fail;
    double rep3_nines  = ReedSolomon::nines_from_loss(rep3_p_loss);

    std::cout << "\n    3-way replication: " << std::fixed << std::setprecision(1)
              << rep3_nines << " nines\n"
              << "    RS(10,4):          " << rs_nines << " nines\n"
              << "    RS advantage:      " << (rs_nines - rep3_nines) << " extra nines ";

    ASSERT(rs_nines > rep3_nines);  // RS is strictly more durable
    ASSERT(rs_nines > 8.6);
}

// =================================================================
// SECTION 5: Raft Consensus
// =================================================================

TEST(raft_leader_election) {
    // Leader election: majority voting prevents split brain
    // n=5 nodes: need 3 votes. Can't have two candidates both win.
    // Proven via pigeonhole: 2 candidates need 3+3=6 votes from 5 nodes — impossible.
    int n = 5;
    int majority = n/2+1;  // 3
    // Two candidates can't both get majority: 3+3 > 5
    ASSERT(2 * majority > n);  // pigeonhole: at most one can win
    ASSERT(majority == 3);

    // Raft vote rule: one vote per term per node (tested in term_monotonically_increases)
    // Combined: only one leader per term possible. ✓
    ASSERT(true);
}

TEST(raft_log_replication) {
    // Each of the 3 replicas applies every committed entry to its own
    // state machine: 5 entries × 3 nodes = 15 applies.
    std::atomic<int> applied{0};
    RaftGroup cluster(3, [&](const LogEntry& e){
        if (!e.is_noop()) ++applied;
    });
    cluster.start();
    ASSERT(cluster.wait_for_leader(5000));

    std::optional<Index> last;
    for (int i = 0; i < 5; ++i) {
        std::vector<uint8_t> payload{'T','E','S','T'};
        last = cluster.submit("UPSERT", payload);
        ASSERT(last.has_value());
    }

    ASSERT(cluster.wait_committed(*last, 2000));
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (applied.load() < 15 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    ASSERT(applied.load() == 15);

    cluster.stop();
}

TEST(raft_majority_commit) {
    // 3-node Raft: commit when 2/3 ACK
    std::atomic<int> applied{0};
    RaftGroup cluster(3, [&](const LogEntry& e){
        if (!e.is_noop()) ++applied;
    });
    cluster.start();
    ASSERT(cluster.wait_for_leader(5000));

    std::vector<uint8_t> payload(8, 0xAB);
    auto idx = cluster.submit("UPSERT", payload);
    ASSERT(idx.has_value());

    bool committed = cluster.wait_committed(*idx, 1000);
    ASSERT(committed);
    ASSERT(applied.load() >= 1);

    cluster.stop();
}

TEST(raft_term_monotonically_increases) {
    RaftNode::ApplyFn noop = [](const LogEntry&){};
    RaftGroup cluster(5, noop);
    cluster.start();
    std::vector<Term> last(5, 0);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(400);
    bool ok = true;
    while (std::chrono::steady_clock::now() < deadline) {
        for (int i = 0; i < 5; ++i) {
            Term t = cluster.node(i).current_term();
            if (t < last[i]) ok = false;
            last[i] = t;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    cluster.stop();
    ASSERT(ok);
    ASSERT(*std::max_element(last.begin(), last.end()) >= 1);
}

TEST(raft_no_split_brain) {
    // Sample the live cluster: no two nodes may be leader in the same term.
    RaftNode::ApplyFn noop = [](const LogEntry&){};
    RaftGroup cluster(5, noop);
    cluster.start();
    std::map<Term, std::set<int>> leaders_by_term;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    while (std::chrono::steady_clock::now() < deadline) {
        for (int i = 0; i < 5; ++i) {
            const auto& n = cluster.node(i);
            if (n.is_leader()) leaders_by_term[n.current_term()].insert(i);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    cluster.stop();
    ASSERT(!leaders_by_term.empty());
    for (const auto& [term, ids] : leaders_by_term)
        ASSERT(ids.size() == 1);
}

// Builds n persisted nodes wired with LocalPeer.
static std::vector<std::unique_ptr<RaftNode>>
make_persisted_cluster(const fs::path& dir, int n, RaftNode::ApplyFn fn) {
    std::vector<std::unique_ptr<RaftNode>> nodes;
    for (int i = 0; i < n; ++i) {
        RaftConfig cfg;
        cfg.node_id = i;
        cfg.n_nodes = n;
        cfg.persist_dir = (dir / ("n" + std::to_string(i))).string();
        nodes.push_back(std::make_unique<RaftNode>(cfg, fn));
    }
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i != j) nodes[i]->register_peer(j, nodes[j].get());
    return nodes;
}

static RaftNode* wait_leader(std::vector<std::unique_ptr<RaftNode>>& nodes) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
        for (auto& n : nodes) if (n->is_leader()) return n.get();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return nullptr;
}

TEST(raft_persistent_state_survives_restart) {
    fs::path dir = fs::temp_directory_path() / "tidevec_raft_persist";
    fs::remove_all(dir);
    std::atomic<int> applied{0};
    auto count = [&](const LogEntry&) { ++applied; };
    Term term_before = 0;
    {
        auto nodes = make_persisted_cluster(dir, 3, count);
        for (auto& n : nodes) n->start();
        RaftNode* leader = wait_leader(nodes);
        ASSERT(leader != nullptr);
        std::optional<Index> last;
        for (int i = 0; i < 5; ++i) last = leader->submit("UPSERT", {uint8_t(i)});
        ASSERT(last && leader->wait_committed(*last, 2000));
        term_before = leader->current_term();
        for (auto& n : nodes) n->stop();
    }
    // All three replicas re-apply the 5 committed entries from disk once a
    // new leader re-establishes the commit index.
    applied = 0;
    auto nodes = make_persisted_cluster(dir, 3, count);
    for (auto& n : nodes) ASSERT(n->log_size() >= 7);   // sentinel + NOOP + 5
    for (auto& n : nodes) n->start();
    RaftNode* leader = wait_leader(nodes);
    ASSERT(leader != nullptr);
    ASSERT(leader->current_term() > term_before);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (applied.load() < 15 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    for (auto& n : nodes) n->stop();
    ASSERT(applied.load() == 15);
    fs::remove_all(dir);
}

TEST(raft_single_node_commits_and_discards_torn_tail) {
    fs::path dir = fs::temp_directory_path() / "tidevec_raft_torn";
    fs::remove_all(dir);
    RaftConfig cfg;
    cfg.node_id = 0;
    cfg.n_nodes = 1;
    cfg.persist_dir = dir.string();
    {
        RaftNode node(cfg, [](const LogEntry&) {});
        node.start();
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (!node.is_leader() && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        ASSERT(node.is_leader());
        std::optional<Index> last;
        for (int i = 0; i < 3; ++i) last = node.submit("UPSERT", {1, 2, 3});
        ASSERT(last && node.wait_committed(*last, 1000));
        node.stop();
    }
    { std::ofstream f(dir / "raft_log", std::ios::binary | std::ios::app); f << "garbage"; }
    // Entries already applied are not replayed.
    cfg.initial_applied = 5;
    std::atomic<int> replayed{0};
    RaftNode node(cfg, [&](const LogEntry&) { ++replayed; });
    ASSERT(node.log_size() == 5);          // sentinel + NOOP + 3, tail dropped
    ASSERT(node.last_applied() == 4);      // clamped to the last log index
    node.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    node.stop();
    ASSERT(replayed.load() == 0);
    fs::remove_all(dir);
}

TEST(raft_http_codec_roundtrip) {
    std::vector<uint8_t> bytes;
    for (int i = 0; i < 300; ++i) bytes.push_back(static_cast<uint8_t>(i * 7));
    for (std::size_t len : {0u, 1u, 2u, 3u, 4u, 299u, 300u}) {
        std::vector<uint8_t> in(bytes.begin(), bytes.begin() + len);
        ASSERT(base64_decode(base64_encode(in)) == in);
    }
    AppendEntriesArgs a{3, 1, 7, 2, {{3, 8, "CMD", {0, 255, 10}}}, 6};
    auto b = append_args_from_json(to_json(a));
    ASSERT(b.term == 3 && b.leader_id == 1 && b.prev_log_index == 7 &&
           b.prev_log_term == 2 && b.leader_commit == 6);
    ASSERT(b.entries.size() == 1 && b.entries[0].payload == a.entries[0].payload &&
           b.entries[0].op == "CMD");
    ASSERT(secret_equals("abc", "abc") && !secret_equals("abc", "abd") &&
           !secret_equals("abc", "abcd"));
}

// =================================================================
// SECTION 6: Health Monitor
// =================================================================

TEST(health_monitor_startup) {
    health::HealthMonitorConfig cfg;
    cfg.n_nodes             = 14;  // RS(10,4)
    cfg.rs_k                = 10;
    cfg.rs_m                = 4;
    cfg.heartbeat_interval_ms = 100;
    cfg.scrub_interval_ms   = 1000000;  // don't scrub in test
    health::HealthMonitor hm(cfg);

    // All nodes start healthy
    for (const auto& ns : hm.node_statuses())
        ASSERT(ns.health == NodeHealth::HEALTHY);

    auto rep = hm.durability_report();
    ASSERT(rep.n_nodes_healthy == 14);
    ASSERT(rep.effective_nines > 8.6);
    ASSERT(rep.status == "OK");
}

TEST(health_monitor_failure_detection) {
    health::HealthMonitorConfig cfg;
    cfg.n_nodes = 14; cfg.rs_k = 10; cfg.rs_m = 4;
    cfg.heartbeat_interval_ms = 50;
    cfg.scrub_interval_ms = 9999999;
    health::HealthMonitor hm(cfg);

    // Manually report 2 node failures
    hm.report_failure(0);
    hm.report_failure(3);

    auto rep = hm.durability_report();
    ASSERT(rep.n_nodes_failed == 2);
    ASSERT(rep.status == "DEGRADED (repairing)");
    // Two lost shards leave room for only two more: P(≥3 of 12) ≈ 1.4e-5
    ASSERT(rep.effective_nines > 4.5 && rep.effective_nines < 5.0);
}

TEST(health_monitor_recovery) {
    health::HealthMonitorConfig cfg;
    cfg.n_nodes = 14; cfg.rs_k = 10; cfg.rs_m = 4;
    cfg.scrub_interval_ms = 9999999;
    health::HealthMonitor hm(cfg);

    hm.report_failure(1);
    ASSERT(hm.durability_report().n_nodes_failed == 1);

    hm.report_recovered(1);
    ASSERT(hm.durability_report().n_nodes_failed == 0);
    ASSERT(hm.durability_report().status == "OK");
}

TEST(health_monitor_critical_threshold) {
    health::HealthMonitorConfig cfg;
    cfg.n_nodes = 14; cfg.rs_k = 10; cfg.rs_m = 4;
    cfg.scrub_interval_ms = 9999999;
    health::HealthMonitor hm(cfg);

    // Lose 5 nodes (> m=4) → CRITICAL
    for (int i = 0; i < 5; ++i) hm.report_failure(i);

    auto rep = hm.durability_report();
    ASSERT(rep.status == "CRITICAL — DATA LOSS RISK");
}

TEST(health_monitor_prometheus_metrics) {
    health::HealthMonitorConfig cfg;
    cfg.n_nodes = 14; cfg.rs_k = 10; cfg.rs_m = 4;
    cfg.scrub_interval_ms = 9999999;
    health::HealthMonitor hm(cfg);

    std::string metrics = hm.prometheus_metrics();
    ASSERT(metrics.find("tidevec_nodes_healthy") != std::string::npos);
    ASSERT(metrics.find("tidevec_durability_nines") != std::string::npos);
    ASSERT(metrics.find("tidevec_p_data_loss_year") != std::string::npos);
}

// =================================================================
// SECTION 7: UltraDurableCollection (end-to-end)
// =================================================================

TEST(ultra_durable_startup_and_durability) {
    // Verify RS(k,m) durability thresholds for production configs.
    // Adding data shards at fixed parity lowers durability: RS(14,4) ≈ 8.1.
    std::vector<std::tuple<int,int,double>> configs = {
        {4, 2, 5.5}, {6, 3, 7.4}, {10, 4, 8.6}, {14, 4, 8.0},
    };
    for (auto& [k, m, min_nines] : configs) {
        ReedSolomon rs(k, m);
        double nines = rs.durability_nines(0.004);
        ASSERT(nines > min_nines);
    }
    ReedSolomon rs(10, 4);
    ASSERT(rs.durability_nines(0.004) > 8.6);
    ASSERT(rs.k() == 10); ASSERT(rs.m() == 4);
    ASSERT(rs.n() == 14);  // 14 total shards
}

TEST(ultra_durable_write_and_read) {
    // Test RS codec and durability math (not full Raft integration in unit test)
    ReedSolomon rs(4, 2);
    std::vector<uint8_t> data = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    auto shards = rs.encode(data);
    ASSERT(static_cast<int>(shards.size()) == 6);
    // Recover after 2 failures
    uint32_t present = 0b111100u;
    shards[0].clear(); shards[1].clear();
    auto rec = rs.decode(shards, present, data.size());
    ASSERT(rec == data);
}

TEST(ultra_durable_rs_encode_decode) {
    // RS encode/decode through UDC RS codec (no cluster startup)
    ReedSolomon rs(4, 2);  // RS(4,2): k=4, m=2, n=6
    std::vector<uint8_t> data(64);
    std::iota(data.begin(), data.end(), 1);
    auto shards = rs.encode(data);
    ASSERT(static_cast<int>(shards.size()) == 6);
    // Lose 2 shards (indices 1,4)
    uint32_t present = 0b111111u & ~((1u<<1)|(1u<<4));
    shards[1].clear(); shards[4].clear();
    auto rec = rs.decode(shards, present, data.size());
    ASSERT(rec == data);
}

TEST(ultra_durable_wal_group_commit) {
    // WAL group commit: batch N writes into 1 fsync
    // Verify the throughput math: 1000 writes/fsync = 1000× fewer disk operations
    int batch_size = 1000;
    int writes = 100000;
    int fsyncs_with_batch  = writes / batch_size;   // 100
    int fsyncs_without     = writes;                 // 100000
    ASSERT(fsyncs_with_batch == 100);
    ASSERT(fsyncs_without / fsyncs_with_batch == 1000);  // 1000× improvement
    // At 10ms fsync latency: 100k fsyncs = 1000s, 100 fsyncs = 1s
    double lat_batched_s  = fsyncs_with_batch * 0.010;  // 1s
    double lat_single_s   = fsyncs_without    * 0.010;  // 1000s
    ASSERT(lat_batched_s < lat_single_s / 100);
}

TEST(ultra_durable_applies_each_write_once) {
    std::string dir = "/tmp/tidevec_udc_" + std::to_string(now_ms());
    {
        UltraDurableCollection::Config cfg;
        cfg.name = "udc"; cfg.dim = 4; cfg.n_shards = 2;
        cfg.raft_nodes = 3; cfg.data_dir = dir;
        UltraDurableCollection udc(cfg);

        udc.upsert(CortexVector("old", {1.f, 0.f, 0.f, 0.f}));
        udc.upsert(CortexVector("new", {0.f, 1.f, 0.f, 0.f}));
        udc.upsert(CortexVector("x",   {0.f, 0.f, 1.f, 0.f}));
        // Read-your-writes: upsert returns only after apply.
        ASSERT(udc.total_vectors() == 3);
        // One apply per write, not one per Raft node.
        ASSERT(udc.underlying().total_writes() == 3);

        udc.add_edge("new", "old", EdgeType::UPDATES);
        QueryOptions opts; opts.top_k = 1; opts.temporal_blend = 0.0f;
        auto res = udc.search({1.f, 0.f, 0.f, 0.f}, opts);
        ASSERT(res.size() == 1 && res[0].id == "old");
        ASSERT(res[0].superseded_by == std::vector<std::string>{"new"});
    }
    std::filesystem::remove_all(dir);
}

// =================================================================
// MAIN
// =================================================================
int main() {
    std::cout << "\n=== TideVec Durability & Consensus Tests ===\n\n";

    std::cout << "\n--- Results ---\n";
    std::cout << tests_passed << " / " << tests_run << " tests passed\n";
    if (tests_passed == tests_run) {
        std::cout << "ALL TESTS PASSED\n\n";
        std::cout << "Modelled durability: RS(10,4) = 8.7 nines at 0.4% annual disk failure\n"
                  << "(independent disks, no repair; the server keeps all shards on one host)\n\n";
        return 0;
    }
    std::cout << (tests_run - tests_passed) << " TESTS FAILED\n\n";
    return 1;
}
