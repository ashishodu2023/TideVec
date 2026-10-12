// TideVec — Scale & Durability Tests
// Tests WAL, ShardRouter, ReplicaSet, DurableCollection, ProductQuantizer

#include <tidevec/quantization/product_quantizer.hpp>
#include <tidevec/storage/wal.hpp>
#include <tidevec/cluster/shard_router.hpp>
#include <tidevec/cluster/replica_set.hpp>
#include <tidevec/cluster/durable_collection.hpp>
#include <tidevec/api/json_serializers.hpp>

#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <string>
#include <filesystem>
#include <thread>
#include <chrono>
#include <random>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <unordered_set>

namespace fs = std::filesystem;
using namespace tidevec;

// ---- test harness -----------------------------------------------
static int tests_run = 0, tests_passed = 0;
static bool g_current_test_failed = false;

#define TEST(name) \
    void test_##name(); \
    struct _reg_##name { _reg_##name(){ run_test(#name, test_##name); } } _inst_##name; \
    void test_##name()

#define ASSERT(cond) do { \
    if (!(cond)) { \
        std::cerr << "  FAIL: " #cond " line " << __LINE__ << "\n"; \
        g_current_test_failed = true; \
        return; \
    } \
} while(0)

#define ASSERT_NEAR(a,b,eps) ASSERT(std::abs((float)(a)-(float)(b)) < (float)(eps))

static void run_test(const char* name, void(*fn)()) {
    ++tests_run;
    std::cout << "[TEST] " << name << " ... ";
    g_current_test_failed = false;
    fn();
    if (g_current_test_failed) {
        std::cout << "FAILED\n";
        return;
    }
    ++tests_passed;
    std::cout << "PASS\n";
}

// ---- helpers ----------------------------------------------------
static std::vector<float> rand_vec(std::size_t dim, unsigned seed=42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(dim);
    for (auto& x : v) x = dist(rng);
    return v;
}

// =================================================================
// SECTION 1: Product Quantizer
// =================================================================

TEST(pq_encode_decode_roundtrip) {
    const std::size_t dim = 64, M = 8;
    ProductQuantizer pq(dim, M);

    // Generate training samples
    std::vector<std::vector<float>> samples;
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> dist(-1.0f,1.0f);
    for (int i = 0; i < 500; ++i) {
        std::vector<float> v(dim);
        for (auto& x : v) x = dist(rng);
        samples.push_back(v);
    }
    pq.train(samples, 10);
    ASSERT(pq.is_trained());

    // Encode and decode
    auto code = pq.encode(samples[0]);
    ASSERT(code.size() == M);  // M bytes

    auto approx = pq.decode(code);
    ASSERT(approx.size() == dim);

    // Reconstruction error should be small-ish
    float err = l2_distance(samples[0], approx);
    ASSERT(err < 5.0f);  // loose bound — PQ is lossy
}

TEST(pq_adc_distance) {
    const std::size_t dim = 32, M = 4;
    ProductQuantizer pq(dim, M);

    std::vector<std::vector<float>> samples;
    std::mt19937 rng(2);
    std::uniform_real_distribution<float> dist(-1.f,1.f);
    for (int i = 0; i < 300; ++i) {
        std::vector<float> v(dim); for(auto& x:v) x=dist(rng);
        samples.push_back(v);
    }
    pq.train(samples, 5);

    auto query = samples[10];
    auto dt = pq.precompute_adc(query);
    auto code = pq.encode(samples[20]);
    float adc_dist = pq.adc_distance(dt, code);
    ASSERT(adc_dist >= 0.0f);  // distance must be non-negative
}

TEST(pq_memory_estimate) {
    ProductQuantizer pq(768, 96);
    // 1B vectors * 96 bytes = 96GB
    std::size_t mem = pq.memory_bytes(1'000'000'000ULL);
    ASSERT(mem == 96ULL * 1'000'000'000ULL);
}

// =================================================================
// SECTION 2: Write-Ahead Log
// =================================================================

TEST(wal_write_and_replay) {
    std::string wal_dir = "/tmp/tidevec_test_wal_" +
        std::to_string(now_ms());
    WriteAheadLog::Config cfg;
    cfg.dir = wal_dir;
    cfg.sync_on_write = false;  // faster for tests

    WriteAheadLog wal(cfg);

    CortexVector v("doc1", {0.1f, 0.2f, 0.3f});
    v.payload["source"] = "test";
    wal.log_upsert(v);
    wal.log_upsert(CortexVector("doc2", {0.4f, 0.5f, 0.6f}));
    wal.log_delete("doc1");
    wal.log_add_edge("doc2", "doc3", EdgeType::CAUSES, 0.9f);
    wal.flush_to_disk();

    // (WAL is still open — replay reads from disk so we flush first)
    // Replay and verify
    int upsert_count = 0, delete_count = 0, edge_count = 0;
    wal.replay([&](const WalRecord& rec) {
        switch (static_cast<WalOp>(rec.header.op)) {
            case WalOp::UPSERT:   ++upsert_count; break;
            case WalOp::DELETE:   ++delete_count; break;
            case WalOp::ADD_EDGE: ++edge_count;   break;
            default: break;
        }
    });

    ASSERT(upsert_count == 2);
    ASSERT(delete_count == 1);
    ASSERT(edge_count   == 1);

    fs::remove_all(wal_dir);
}

TEST(wal_vector_serialization_roundtrip) {
    CortexVector orig("vec_99", {1.0f, 2.0f, 3.0f, 4.0f});
    orig.payload["category"] = "finance";
    orig.payload["year"]     = "2025";
    orig.add_edge("vec_100", EdgeType::CONTRADICTS, 0.75f);
    orig.set_ttl_seconds(3600);

    auto payload = serialize_vector(orig);
    std::size_t off = 0;
    auto restored = deserialize_vector(payload.data(), off);

    ASSERT(restored.id == "vec_99");
    ASSERT(restored.dim() == 4);
    ASSERT(restored.payload.at("category") == "finance");
    ASSERT(restored.edges.size() == 1);
    ASSERT(restored.edges[0].type == EdgeType::CONTRADICTS);
    ASSERT(restored.valid_until.has_value());
}

TEST(wal_crash_recovery) {
    std::string wal_dir = "/tmp/tidevec_wal_crash_" + std::to_string(now_ms());
    WriteAheadLog::Config cfg; cfg.dir = wal_dir; cfg.sync_on_write = false;

    {   // "Process 1" — writes 50 vectors then "crashes"
        WriteAheadLog wal(cfg);
        for (int i = 0; i < 50; ++i)
            wal.log_upsert(CortexVector("v" + std::to_string(i),
                                        rand_vec(8, i)));
    }

    // "Process 2" — recovers from WAL
    WriteAheadLog wal2(cfg);
    int replayed = 0;
    wal2.replay([&](const WalRecord& rec) {
        if (static_cast<WalOp>(rec.header.op) == WalOp::UPSERT) ++replayed;
    });
    ASSERT(replayed == 50);

    fs::remove_all(wal_dir);
}

// =================================================================
// SECTION 3: ShardRouter
// =================================================================

TEST(shard_router_distribution) {
    Collection::Config ccfg;
    ccfg.dim        = 8;
    ccfg.index_type = IndexType::FLAT;

    ShardRouter::Config rcfg;
    rcfg.collection_name = "test_shard";
    rcfg.n_shards        = 4;
    rcfg.shard_cfg       = ccfg;
    rcfg.parallel_search = false;

    ShardRouter router(rcfg);

    // Insert 100 vectors
    for (int i = 0; i < 100; ++i) {
        CortexVector v("vec_" + std::to_string(i), rand_vec(8, i));
        router.upsert(v);
    }

    ASSERT(router.total_size() == 100);
    ASSERT(router.n_shards() == 4);

    // Check distribution is roughly even (no shard is empty)
    for (std::size_t s = 0; s < 4; ++s)
        ASSERT(router.shard_size(s) > 0);
}

TEST(shard_router_search) {
    Collection::Config ccfg;
    ccfg.dim        = 8;
    ccfg.index_type = IndexType::FLAT;

    ShardRouter::Config rcfg;
    rcfg.collection_name = "test_search";
    rcfg.n_shards        = 3;
    rcfg.shard_cfg       = ccfg;
    rcfg.parallel_search = true;

    ShardRouter router(rcfg);

    // Insert 60 vectors; target vector is "needle"
    std::vector<float> needle(8, 0.99f);
    router.upsert(CortexVector("needle", needle));
    for (int i = 0; i < 59; ++i) {
        auto v = rand_vec(8, i+1);
        router.upsert(CortexVector("noise_" + std::to_string(i), v));
    }

    QueryOptions opts;
    opts.top_k = 1;
    opts.temporal_blend = 0.0f;
    auto res = router.search(needle, opts);
    ASSERT(!res.empty());
    ASSERT(res[0].id == "needle");
}

TEST(shard_router_consistent_hashing) {
    // Same id always routes to same shard
    std::size_t s1 = hash_to_shard("user_123", 8);
    std::size_t s2 = hash_to_shard("user_123", 8);
    ASSERT(s1 == s2);
    ASSERT(s1 < 8);
}

TEST(shard_router_parallel_vs_serial_same_result) {
    Collection::Config ccfg;
    ccfg.dim = 4; ccfg.index_type = IndexType::FLAT;

    auto make_router = [&](bool parallel) {
        ShardRouter::Config rcfg;
        rcfg.collection_name = "par_test";
        rcfg.n_shards = 3;
        rcfg.shard_cfg = ccfg;
        rcfg.parallel_search = parallel;
        return ShardRouter(rcfg);
    };

    ShardRouter par  = make_router(true);
    ShardRouter ser  = make_router(false);

    for (int i = 0; i < 30; ++i) {
        CortexVector v("v" + std::to_string(i), rand_vec(4, i));
        par.upsert(v);
        ser.upsert(v);
    }

    std::vector<float> q = {1.f, 0.f, 0.f, 0.f};
    QueryOptions opts; opts.top_k = 3; opts.temporal_blend = 0.0f;
    auto r1 = par.search(q, opts);
    auto r2 = ser.search(q, opts);

    ASSERT(r1.size() == r2.size());
    if (!r1.empty() && !r2.empty())
        ASSERT(r1[0].id == r2[0].id);
}

// =================================================================
// SECTION 4: ReplicaSet
// =================================================================

TEST(replica_set_write_and_read) {
    std::string wal_dir = "/tmp/tidevec_rs_" + std::to_string(now_ms());

    Collection::Config ccfg;
    ccfg.dim = 4; ccfg.index_type = IndexType::FLAT;

    ReplicaSet::Config rscfg;
    rscfg.collection_cfg  = ccfg;
    rscfg.n_replicas      = 2;
    rscfg.write_quorum    = 2;
    rscfg.wal_cfg.dir     = wal_dir;
    rscfg.wal_cfg.sync_on_write = false;

    ReplicaSet rs(rscfg);

    rs.upsert(CortexVector("a", {1.f, 0.f, 0.f, 0.f}));
    rs.upsert(CortexVector("b", {0.f, 1.f, 0.f, 0.f}));
    rs.upsert(CortexVector("c", {0.f, 0.f, 1.f, 0.f}));

    ASSERT(rs.primary_size() == 3);

    QueryOptions opts; opts.top_k = 1; opts.temporal_blend = 0.0f;
    auto res = rs.search({1.f, 0.f, 0.f, 0.f}, opts);
    ASSERT(!res.empty() && res[0].id == "a");

    fs::remove_all(wal_dir);
}

TEST(replica_set_failover) {
    std::string wal_dir = "/tmp/tidevec_fo_" + std::to_string(now_ms());

    Collection::Config ccfg;
    ccfg.dim = 4; ccfg.index_type = IndexType::FLAT;

    ReplicaSet::Config rscfg;
    rscfg.collection_cfg  = ccfg;
    rscfg.n_replicas      = 2;
    rscfg.write_quorum    = 2;
    rscfg.wal_cfg.dir     = wal_dir;
    rscfg.wal_cfg.sync_on_write = false;

    ReplicaSet rs(rscfg);
    rs.upsert(CortexVector("x", {1.f, 1.f, 0.f, 0.f}));

    // Simulate primary failure: promote replica
    rs.failover_primary();
    ASSERT(rs.is_degraded());
    ASSERT(rs.n_replicas() == 1);

    // Reads still work after failover
    QueryOptions opts; opts.top_k = 1; opts.temporal_blend = 0.0f;
    auto res = rs.search({1.f, 1.f, 0.f, 0.f}, opts);
    ASSERT(!res.empty() && res[0].id == "x");

    fs::remove_all(wal_dir);
}

// =================================================================
// SECTION 5: DurableCollection (end-to-end)
// =================================================================

TEST(durable_collection_basic) {
    std::string data_dir = "/tmp/tidevec_dc_" + std::to_string(now_ms());

    DurableCollection::Config cfg;
    cfg.name      = "test_durable";
    cfg.dim       = 8;
    cfg.n_shards  = 3;
    cfg.n_replicas= 1;
    cfg.write_quorum = 1;
    cfg.data_dir  = data_dir;
    cfg.parallel_search = true;

    DurableCollection dc(cfg);

    // Insert 90 vectors
    for (int i = 0; i < 90; ++i) {
        CortexVector v("dv_" + std::to_string(i), rand_vec(8, i));
        dc.upsert(v);
    }

    ASSERT(dc.total_vectors() == 90);
    ASSERT(dc.n_shards() == 3);

    // Search
    QueryOptions opts;
    opts.top_k = 5;
    opts.temporal_blend = 0.0f;
    auto res = dc.search(rand_vec(8, 0), opts);
    ASSERT(res.size() == 5);

    fs::remove_all(data_dir);
}

TEST(durable_collection_with_trace) {
    std::string data_dir = "/tmp/tidevec_trace_" + std::to_string(now_ms());

    DurableCollection::Config cfg;
    cfg.name = "trace_test"; cfg.dim = 4;
    cfg.n_shards = 2; cfg.n_replicas = 1; cfg.write_quorum = 1;
    cfg.data_dir = data_dir;

    DurableCollection dc(cfg);
    dc.upsert(CortexVector("t1", {1.f, 0.f, 0.f, 0.f}));
    dc.upsert(CortexVector("t2", {0.f, 1.f, 0.f, 0.f}));

    QueryOptions opts; opts.top_k = 2; opts.temporal_blend = 0.0f;
    RetrievalTrace trace;
    auto res = dc.search({1.f, 0.f, 0.f, 0.f}, opts, &trace);

    ASSERT(!trace.query_id.empty());
    ASSERT(trace.latency_ms > 0.0);
    ASSERT(trace.strategy == "DURABLE_SHARDED_TVINDEX");
    ASSERT(!res.empty());

    fs::remove_all(data_dir);
}

TEST(durable_collection_temporal_scoring) {
    std::string data_dir = "/tmp/tidevec_ts_" + std::to_string(now_ms());

    DurableCollection::Config cfg;
    cfg.name = "temporal_dc"; cfg.dim = 4;
    cfg.n_shards = 2; cfg.n_replicas = 1; cfg.write_quorum = 1;
    cfg.data_dir = data_dir;
    cfg.temporal.half_life_ms   = 1000;  // 1 second
    cfg.temporal.temporal_blend = 0.4f;

    DurableCollection dc(cfg);

    CortexVector fresh("fresh", {1.f, 0.f, 0.f, 0.f});
    CortexVector stale("stale", {1.f, 0.f, 0.f, 0.f});
    stale.created_at = now_ms() - 5000;  // 5 seconds ago

    dc.upsert(fresh);
    dc.upsert(stale);

    QueryOptions opts; opts.top_k = 2; opts.temporal_blend = 0.4f;
    opts.include_staleness_warnings = true;
    auto res = dc.search({1.f, 0.f, 0.f, 0.f}, opts);

    ASSERT(res.size() == 2);
    float fresh_ts = 0.f, stale_ts = 0.f;
    for (const auto& r : res) {
        if (r.id == "fresh") fresh_ts = r.temporal_score;
        if (r.id == "stale") stale_ts = r.temporal_score;
    }
    ASSERT(fresh_ts > stale_ts);

    fs::remove_all(data_dir);
}

TEST(durable_collection_concurrent_writes) {
    std::string data_dir = "/tmp/tidevec_conc_" + std::to_string(now_ms());

    DurableCollection::Config cfg;
    cfg.name = "concurrent"; cfg.dim = 8;
    cfg.n_shards = 4; cfg.n_replicas = 1; cfg.write_quorum = 1;
    cfg.data_dir = data_dir;

    DurableCollection dc(cfg);

    // 4 threads each write 25 vectors concurrently
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&dc, t]() {
            for (int i = 0; i < 25; ++i) {
                std::string id = "cv_t" + std::to_string(t) + "_" + std::to_string(i);
                dc.upsert(CortexVector(id, rand_vec(8, t*100+i)));
            }
        });
    }
    for (auto& th : threads) th.join();

    ASSERT(dc.total_vectors() == 100);

    fs::remove_all(data_dir);
}

TEST(durable_supersession_across_shards) {
    std::string data_dir = "/tmp/tidevec_sup_" + std::to_string(now_ms());

    DurableCollection::Config cfg;
    cfg.name = "sup"; cfg.dim = 4;
    cfg.n_shards = 4; cfg.n_replicas = 1; cfg.write_quorum = 1;
    cfg.data_dir = data_dir;
    cfg.supersession.enabled = true;
    cfg.supersession.similarity_threshold = 0.9f;
    DurableCollection dc(cfg);

    // Pick ids that hash to different shards so the link must cross shards.
    std::string old_id = "policy_old", new_id;
    for (int i = 0; new_id.empty(); ++i) {
        std::string cand = "policy_new_" + std::to_string(i);
        if (hash_to_shard(cand, 4) != hash_to_shard(old_id, 4)) new_id = cand;
    }

    CortexVector old_v(old_id, {1.f, 0.f, 0.f, 0.f});
    old_v.created_at = now_ms() - 20LL * 86'400'000;
    dc.upsert(old_v);
    dc.upsert(CortexVector("unrelated", {0.f, 1.f, 0.f, 0.f}));
    dc.upsert(CortexVector(new_id, {0.98f, 0.1f, 0.f, 0.f}));

    ASSERT(dc.superseded_by(old_id) == std::vector<std::string>{new_id});
    ASSERT(dc.superseded_by(new_id).empty());

    QueryOptions opts; opts.top_k = 3; opts.temporal_blend = 0.0f;
    opts.superseded_penalty = 0.5f;
    auto res = dc.search({1.f, 0.f, 0.f, 0.f}, opts);
    ASSERT(!res.empty());
    ASSERT(res[0].id == new_id);
    bool flagged = false;
    for (const auto& r : res)
        if (r.id == old_id) flagged = r.staleness_warning && !r.superseded_by.empty();
    ASSERT(flagged);

    fs::remove_all(data_dir);
}

TEST(durable_edges_survive_recovery) {
    std::string data_dir = "/tmp/tidevec_edge_rec_" + std::to_string(now_ms());

    DurableCollection::Config cfg;
    cfg.name = "edges"; cfg.dim = 4;
    cfg.n_shards = 2; cfg.n_replicas = 1; cfg.write_quorum = 1;
    cfg.data_dir = data_dir;

    {
        DurableCollection dc(cfg);
        dc.upsert(CortexVector("a", {1.f, 0.f, 0.f, 0.f}));
        dc.upsert(CortexVector("b", {0.f, 1.f, 0.f, 0.f}));
        dc.upsert(CortexVector("c", {0.f, 0.f, 1.f, 0.f}));
        dc.add_edge("a", "b", EdgeType::CONTRADICTS);
        dc.add_edge("c", "a", EdgeType::UPDATES);
    }

    DurableCollection restored(cfg);
    restored.recover();
    ASSERT(restored.total_vectors() == 3);

    QueryOptions opts; opts.top_k = 1; opts.temporal_blend = 0.0f;
    opts.mode = QueryMode::CONTRADICTION_CHECK;
    auto res = restored.search({1.f, 0.f, 0.f, 0.f}, opts);
    ASSERT(res.size() == 1 && res[0].id == "a");
    ASSERT(res[0].contradicted_by == std::vector<std::string>{"b"});
    ASSERT(res[0].superseded_by == std::vector<std::string>{"c"});

    fs::remove_all(data_dir);
}

// =================================================================
// Filtered search at scale: recall@10 against a brute-force oracle and
// per-query latency for each filter shape. Defaults keep CI fast; set
// TIDEVEC_FILTER_BENCH_N (e.g. 100000 or 1000000) and optionally
// TIDEVEC_FILTER_BENCH_DIM / _QUERIES for a real measurement.
// Vectors are spread over shards built in parallel and searched with
// a fan-out + merge, like DurableCollection does.
// =================================================================
static long env_long(const char* name, long def) {
    const char* s = std::getenv(name);
    return s && *s ? std::strtol(s, nullptr, 10) : def;
}

TEST(filtered_recall_at_scale) {
    const long N  = env_long("TIDEVEC_FILTER_BENCH_N", 10000);
    const int  D  = static_cast<int>(env_long("TIDEVEC_FILTER_BENCH_DIM", 64));
    const int  NQ = static_cast<int>(env_long("TIDEVEC_FILTER_BENCH_QUERIES", 50));
    const int  K  = 10;
    const int  n_shards = static_cast<int>(std::clamp<long>(
        N / 100000, 1, std::max(1u, std::thread::hardware_concurrency())));
    const Timestamp t0 = 1'700'000'000'000LL;

    // Clustered data (embeddings have far lower intrinsic dimension than
    // uniform noise); payload attributes are independent of the vector.
    std::mt19937 rng(1234);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    const int n_centers = 256;
    std::vector<std::vector<float>> centers(n_centers, std::vector<float>(D));
    for (auto& c : centers) for (auto& x : c) x = gauss(rng);
    auto sample = [&](std::mt19937& r) {
        std::vector<float> v = centers[r() % n_centers];
        for (auto& x : v) x += 0.35f * gauss(r);
        return v;
    };

    std::vector<CortexVector> vecs;
    vecs.reserve(static_cast<std::size_t>(N));
    for (long i = 0; i < N; ++i) {
        std::unordered_map<std::string, std::string> p{
            {"bucket", std::to_string(i % 100)},             // 1% per value
            {"tenant", "t" + std::to_string(i % 10)},        // 10%
            {"cat",    "c" + std::to_string((i / 10) % 10)}, // 10%, independent of tenant
            {"half",   i % 2 ? "a" : "b"},                   // 50%
            {"score",  std::to_string((i * 7919) % 1000)}};  // numeric 0..999
        CortexVector v("v" + std::to_string(i), sample(rng), std::move(p));
        v.created_at = v.valid_from = t0 + i * 1000;
        vecs.push_back(std::move(v));
    }

    TVIndexConfig cfg;
    cfg.temporal.temporal_blend = 0.0f;
    std::vector<std::unique_ptr<TVIndex>> shards;
    for (int s = 0; s < n_shards; ++s) shards.push_back(std::make_unique<TVIndex>(cfg));
    auto build_start = std::chrono::steady_clock::now();
    {
        // Shard by id hash (as the router does) so shard membership is
        // independent of the payload attributes.
        std::vector<int> shard_of(static_cast<std::size_t>(N));
        for (long i = 0; i < N; ++i)
            shard_of[i] = static_cast<int>(std::hash<std::string>{}(vecs[i].id) % n_shards);
        std::vector<std::thread> ts;
        for (int s = 0; s < n_shards; ++s)
            ts.emplace_back([&, s] {
                for (long i = 0; i < N; ++i)
                    if (shard_of[i] == s) shards[s]->insert(vecs[i]);
            });
        for (auto& t : ts) t.join();
    }
    double build_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - build_start).count();
    std::cout << "\n  N=" << N << " dim=" << D << " shards=" << n_shards
              << " queries=" << NQ << " build=" << build_s << "s\n";

    std::vector<std::vector<float>> queries;
    std::mt19937 qrng(77);
    for (int q = 0; q < NQ; ++q) queries.push_back(sample(qrng));

    struct Case { const char* name; json filter; double min_recall; };
    const Timestamp t_1pct = t0 + (N / 100) * 1000;
    std::vector<Case> cases = {
        {"none",                    nullptr,                                         0.95},
        {"keyword 1%",              {{"bucket", "42"}},                               0.95},
        {"numeric range 1%",        {{"score", {{"$gte", 500}, {"$lt", 510}}}},       0.95},
        {"AND tenant&cat 1%",       {{"tenant", "t3"}, {"cat", "c7"}},                0.95},
        {"created_at range 1%",     {{"created_at", {{"$lt", t_1pct}}}},              0.95},
        {"OR bucket 2%",            {{"$or", {{{"bucket", "1"}}, {{"bucket", "2"}}}}}, 0.95},
        {"IN bucket 3%",            {{"bucket", {{"$in", {"5", "6", "7"}}}}},         0.95},
        {"NOT 1%",                  {{"$not", {{"bucket", {{"$gte", 1}}}}}},          0.95},
        {"keyword 10%",             {{"tenant", "t4"}},                               0.95},
        {"keyword 50%",             {{"half", "a"}},                                  0.95},
        {"ne 99%",                  {{"bucket", {{"$ne", "0"}}}},                     0.95},
    };

    bool ok = true;
    for (const auto& c : cases) {
        QueryOptions opts; opts.top_k = K;
        opts.filter_expr = filter_from_json(c.filter);
        const FilterExpr* f = opts.filter_expr.get();

        double hit = 0, total = 0;
        std::vector<double> lat;
        for (const auto& q : queries) {
            std::vector<std::pair<float, std::size_t>> s;
            for (std::size_t i = 0; i < vecs.size(); ++i) {
                if (f && !f->matches(vecs[i])) continue;
                s.emplace_back(compute_similarity(q, vecs[i].embedding, Metric::COSINE), i);
            }
            std::size_t kk = std::min<std::size_t>(K, s.size());
            std::partial_sort(s.begin(), s.begin() + kk, s.end(),
                              [](auto& a, auto& b) { return a.first > b.first; });
            std::unordered_set<std::string> truth;
            for (std::size_t i = 0; i < kk; ++i) truth.insert(vecs[s[i].second].id);

            auto qs = std::chrono::steady_clock::now();
            std::vector<SearchResult> merged;
            for (auto& sh : shards) {
                auto r = sh->search(q, opts);
                merged.insert(merged.end(), r.begin(), r.end());
            }
            std::sort(merged.begin(), merged.end(),
                      [](auto& a, auto& b) { return a.score > b.score; });
            if (merged.size() > static_cast<std::size_t>(K)) merged.resize(K);
            lat.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - qs).count());

            for (const auto& r : merged) {
                if (f && !f->matches(r.id, r.payload, r.created_at)) ok = false;
                hit += truth.count(r.id);
            }
            total += static_cast<double>(truth.size());
        }
        std::sort(lat.begin(), lat.end());
        double recall = total ? hit / total : 1.0;
        double mean = 0;
        for (double l : lat) mean += l;
        mean /= static_cast<double>(lat.size());
        char line[256];
        std::snprintf(line, sizeof line,
            "  %-22s plan=%-11s recall@10=%.3f  mean=%.2fms p50=%.2fms p95=%.2fms\n",
            c.name, shards[0]->explain(opts).name().c_str(), recall, mean,
            lat[lat.size() / 2], lat[std::min(lat.size() - 1, lat.size() * 95 / 100)]);
        std::cout << line;
        if (recall < c.min_recall) ok = false;
    }
    ASSERT(ok);
}

// =================================================================
// MAIN
// =================================================================
int main() {
    std::cout << "\n=== TideVec Scale & Durability Tests ===\n\n";
    std::cout << "\n--- Results ---\n";
    std::cout << tests_passed << " / " << tests_run << " tests passed\n";
    if (tests_passed == tests_run) {
        std::cout << "ALL TESTS PASSED\n\n";
        return 0;
    }
    std::cout << (tests_run - tests_passed) << " TESTS FAILED\n\n";
    return 1;
}
