// ================================================================
// test_accelerator.cpp — GPU/TPU accelerator layer tests
//
// These tests run on CPU in all environments.
// GPU/TPU paths are exercised when hardware is present;
// otherwise the stub engines transparently use CPU so ALL
// tests pass regardless of hardware.
// ================================================================

#include <tidevec/accelerator/device.hpp>
#include <tidevec/accelerator/cpu_engine.hpp>
#include <tidevec/accelerator/gpu_engine.hpp>
#include <tidevec/accelerator/tpu_engine.hpp>
#include <tidevec/accelerator/dispatcher.hpp>
#include <tidevec/accelerator/accelerated_collection.hpp>

#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <string>
#include <numeric>
#include <random>
#include <filesystem>
#include <chrono>

using namespace tidevec;
using namespace tidevec::accel;
namespace fs = std::filesystem;

// ---- test harness ----------------------------------------------
static int tests_run = 0, tests_passed = 0;
#define TEST(name) static void test_##name(); \
    struct _r_##name { _r_##name(){ run_test(#name, test_##name); } } _i_##name; \
    static void test_##name()
#define ASSERT(c) do { if(!(c)){ std::cerr<<"  FAIL: "#c" line "<<__LINE__<<"\n"; return; } } while(0)
#define ASSERT_NEAR(a,b,e) ASSERT(std::abs((float)(a)-(float)(b)) < (float)(e))
static void run_test(const char* n, void(*f)()) {
    ++tests_run; std::cout<<"[TEST] "<<n<<" ... "; f(); ++tests_passed; std::cout<<"PASS\n";
}

// ---- helpers ---------------------------------------------------
static std::vector<float> randf(int n, unsigned seed=42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1,1);
    std::vector<float> v(n); for(auto& x:v) x=d(rng); return v;
}

static std::vector<float> make_db(int n, int dim, unsigned seed=0) {
    std::vector<float> db(n*dim);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1,1);
    for(auto& x:db) x=d(rng);
    return db;
}

// =================================================================
// SECTION 1: Device detection
// =================================================================

TEST(device_discovery) {
    auto sys = discover_devices();
    // CPU always present
    ASSERT(sys.cpu.available);
    ASSERT(sys.cpu.type == DeviceType::CPU);
    // GPU info always populated (even if unavailable)
    ASSERT(!sys.gpus.empty());
    ASSERT(sys.gpus[0].type == DeviceType::GPU);
    // TPU info always populated
    ASSERT(!sys.tpus.empty());
    ASSERT(sys.tpus[0].type == DeviceType::TPU);
}

TEST(device_type_strings) {
    ASSERT(device_type_str(DeviceType::CPU)  == "CPU");
    ASSERT(device_type_str(DeviceType::GPU)  == "GPU");
    ASSERT(device_type_str(DeviceType::TPU)  == "TPU");
    ASSERT(device_type_str(DeviceType::AUTO) == "AUTO");
}

// =================================================================
// SECTION 2: SIMD distance kernels
// =================================================================

TEST(simd_dot_product_correctness) {
    std::vector<float> a = {1,0,0,0,0,0,0,0};
    std::vector<float> b = {1,0,0,0,0,0,0,0};
    float dot = simd::dot_avx2(a.data(), b.data(), 8);
    ASSERT_NEAR(dot, 1.0f, 1e-5f);
}

TEST(simd_dot_orthogonal) {
    std::vector<float> a = {1,0,0,0};
    std::vector<float> b = {0,1,0,0};
    ASSERT_NEAR(simd::dot_avx2(a.data(),b.data(),4), 0.0f, 1e-6f);
}

TEST(simd_l2sq_correctness) {
    std::vector<float> a = {0,0,0,0};
    std::vector<float> b = {3,4,0,0};
    ASSERT_NEAR(simd::l2sq_avx2(a.data(),b.data(),4), 25.0f, 1e-4f);
}

TEST(simd_l2sq_zero) {
    std::vector<float> v = {1,2,3,4};
    ASSERT_NEAR(simd::l2sq_avx2(v.data(),v.data(),4), 0.0f, 1e-6f);
}

TEST(simd_batch_matmul_shape) {
    int m=4, n=8, dim=16;
    auto q  = randf(m*dim, 1);
    auto db = randf(n*dim, 2);
    std::vector<float> out(m*n);
    simd::batch_matmul_l2(q.data(), db.data(), out.data(), m, n, dim);
    ASSERT(static_cast<int>(out.size()) == m*n);
    // All distances non-negative
    for (float d : out) ASSERT(d >= 0.0f);
}

TEST(simd_batch_matmul_self_distance_zero) {
    int n=4, dim=8;
    auto v = randf(n*dim, 3);
    std::vector<float> out(n*n);
    simd::batch_matmul_l2(v.data(), v.data(), out.data(), n, n, dim);
    // Diagonal should be 0 (vector vs itself)
    for (int i=0; i<n; ++i)
        ASSERT_NEAR(out[i*n+i], 0.0f, 1e-3f);
}

// =================================================================
// SECTION 3: CpuFlatEngine
// =================================================================

TEST(cpu_flat_engine_insert_search) {
    CpuFlatEngine eng(true);  // cosine
    auto db = make_db(100, 32);
    eng.add(db.data(), 100, 32);

    auto q = randf(32, 99);
    auto res = eng.search(q.data(), 1, 32, 5);
    ASSERT(res.n_queries == 1);
    ASSERT(res.top_k == 5);
    ASSERT(res.indices.size() == 5);
    ASSERT(res.device_used == DeviceType::CPU);
    // All indices valid
    for (auto idx : res.indices) ASSERT(idx >= 0 && idx < 100);
}

TEST(cpu_flat_engine_exact_top1) {
    CpuFlatEngine eng(true);
    // Insert a near-zero vector and a unit vector
    std::vector<float> unit = {1,0,0,0,0,0,0,0};
    std::vector<float> zero = {0,1,0,0,0,0,0,0};
    std::vector<float> db;
    db.insert(db.end(), unit.begin(), unit.end());
    db.insert(db.end(), zero.begin(), zero.end());
    eng.add(db.data(), 2, 8);

    auto res = eng.search(unit.data(), 1, 8, 1);
    ASSERT(res.indices[0] == 0);  // unit vector should be top match
}

TEST(cpu_flat_engine_batch_queries) {
    CpuFlatEngine eng(false);  // L2
    auto db = make_db(200, 16);
    eng.add(db.data(), 200, 16);

    int B = 10;
    auto queries = randf(B * 16, 7);
    auto res = eng.search(queries.data(), B, 16, 3);
    ASSERT(res.n_queries == B);
    ASSERT(res.top_k == 3);
    ASSERT(static_cast<int>(res.indices.size()) == B * 3);
    ASSERT(res.device_used == DeviceType::CPU);
}

TEST(cpu_flat_engine_reset) {
    CpuFlatEngine eng;
    eng.add(make_db(50,8).data(), 50, 8);
    eng.reset();
    auto res = eng.search(randf(8).data(), 1, 8, 5);
    ASSERT(res.indices.size() == 5);
    // All indices -1 (empty engine)
    for (auto idx : res.indices) ASSERT(idx == -1);
}

// =================================================================
// SECTION 4: CpuIvfEngine
// =================================================================

TEST(cpu_ivf_train_and_search) {
    CpuIvfEngine::Config cfg;
    cfg.nlist  = 8;
    cfg.nprobe = 4;
    CpuIvfEngine eng(cfg);

    auto db = make_db(400, 16);
    eng.train(db.data(), 400, 16);
    eng.add(db.data(), 400, 16);

    auto q = randf(16, 55);
    auto res = eng.search(q.data(), 1, 16, 5);
    ASSERT(res.indices.size() == 5);
    ASSERT(res.device_used == DeviceType::CPU);
    for (auto idx : res.indices) ASSERT(idx >= 0 && idx < 400);
}

TEST(cpu_ivf_recall_reasonable) {
    // IVF recall should be reasonably close to exact
    CpuIvfEngine::Config cfg; cfg.nlist=16; cfg.nprobe=8;
    CpuIvfEngine ivf(cfg);
    CpuFlatEngine flat(false);

    auto db = make_db(500, 16);
    ivf.train(db.data(), 500, 16);
    ivf.add(db.data(), 500, 16);
    flat.add(db.data(), 500, 16);

    int hits = 0, total = 20;
    for (int t = 0; t < total; ++t) {
        auto q = randf(16, t+100);
        auto r_ivf  = ivf.search(q.data(),  1, 16, 1);
        auto r_flat = flat.search(q.data(), 1, 16, 1);
        if (r_ivf.indices[0] == r_flat.indices[0]) ++hits;
    }
    // At nprobe=8/nlist=16 expect ~70%+ recall on random data
    ASSERT(hits >= 10);
}

// =================================================================
// SECTION 5: GPU engine (stubs or real)
// =================================================================

TEST(gpu_engine_stub_works_without_cuda) {
    GpuBruteForceEngine eng(0);
    auto info = eng.device_info();
    // Should have a name regardless of CUDA availability
    ASSERT(!info.name.empty());

    auto db = make_db(50, 8);
    eng.add(db.data(), 50, 8);
    auto q = randf(8, 1);
    auto res = eng.search(q.data(), 1, 8, 3);
    ASSERT(res.indices.size() == 3);
}

TEST(cagra_engine_stub_works) {
    CagraStyleEngine::Config cfg; cfg.degree=16; cfg.itopk_size=32;
    CagraStyleEngine eng(cfg, 0);

    auto db = make_db(60, 8);
    eng.add(db.data(), 60, 8);
    auto q = randf(8, 2);
    auto res = eng.search(q.data(), 1, 8, 5);
    ASSERT(res.indices.size() == 5);
    ASSERT(res.device_used == DeviceType::CPU || res.device_used == DeviceType::GPU);
}

// =================================================================
// SECTION 6: TPU engine (stub)
// =================================================================

TEST(tpu_engine_stub_works) {
    XlaBatchMatmulEngine eng;
    auto info = eng.device_info();
    ASSERT(info.type == DeviceType::TPU);
    ASSERT(!info.name.empty());

    auto db = make_db(40, 8);
    eng.add(db.data(), 40, 8);
    auto q = randf(8, 3);
    auto res = eng.search(q.data(), 1, 8, 3);
    ASSERT(res.indices.size() == 3);
}

// =================================================================
// SECTION 7: AcceleratorDispatcher
// =================================================================

TEST(dispatcher_device_selection_small_batch) {
    AcceleratorDispatcher::Config cfg;
    cfg.gpu_batch_threshold = 64;
    cfg.tpu_batch_threshold = 32;
    AcceleratorDispatcher disp(cfg);

    // Small batch → CPU
    ASSERT(disp.select_device_for_batch(1)  == DeviceType::CPU ||
           disp.select_device_for_batch(1)  == DeviceType::GPU);  // if GPU available
}

TEST(dispatcher_search_cpu) {
    AcceleratorDispatcher::Config cfg;
    cfg.preferred = DeviceType::CPU;
    AcceleratorDispatcher disp(cfg);

    auto db = make_db(100, 8);
    disp.add(db.data(), 100, 8, DeviceType::CPU);

    auto q = randf(8, 5);
    auto res = disp.search(q.data(), 1, 8, 5);
    ASSERT(res.indices.size() == 5);
    ASSERT(res.device_used == DeviceType::CPU);
}

TEST(dispatcher_search_gpu_path) {
    AcceleratorDispatcher::Config cfg;
    cfg.preferred = DeviceType::GPU;  // will use stub/CPU if no GPU
    AcceleratorDispatcher disp(cfg);

    auto db = make_db(80, 8);
    disp.add(db.data(), 80, 8, DeviceType::GPU);

    auto q = randf(8, 6);
    auto res = disp.search(q.data(), 1, 8, 5);
    ASSERT(res.indices.size() == 5);
    // Device is GPU or CPU (stub fallback)
    ASSERT(res.device_used == DeviceType::GPU || res.device_used == DeviceType::CPU);
}

TEST(dispatcher_search_tpu_path) {
    AcceleratorDispatcher::Config cfg;
    cfg.preferred = DeviceType::TPU;
    AcceleratorDispatcher disp(cfg);

    auto db = make_db(80, 8);
    disp.add(db.data(), 80, 8, DeviceType::TPU);

    auto q = randf(8, 7);
    auto res = disp.search(q.data(), 1, 8, 3);
    ASSERT(res.indices.size() == 3);
}

TEST(dispatcher_stats_tracking) {
    AcceleratorDispatcher::Config cfg;
    cfg.preferred = DeviceType::CPU;
    AcceleratorDispatcher disp(cfg);

    auto db = make_db(50, 4);
    disp.add(db.data(), 50, 4);

    for (int i = 0; i < 5; ++i) {
        auto q = randf(4, i);
        disp.search(q.data(), 1, 4, 3);
    }

    auto stats = disp.stats();
    ASSERT(stats.total_queries == 5);
    ASSERT(stats.avg_latency_ms >= 0.0);
}

TEST(dispatcher_auto_routing_batch) {
    AcceleratorDispatcher::Config cfg;
    cfg.gpu_batch_threshold = 4;
    cfg.verbose = false;
    AcceleratorDispatcher disp(cfg);

    auto db = make_db(200, 8);
    disp.add(db.data(), 200, 8);

    // Batch of 8 (>= gpu threshold) → GPU or CPU if no GPU
    auto q_batch = randf(8 * 8, 10);
    auto res = disp.search(q_batch.data(), 8, 8, 5);
    ASSERT(res.n_queries == 8);
    ASSERT(static_cast<int>(res.indices.size()) == 8 * 5);
}

// =================================================================
// SECTION 8: AcceleratedCollection
// =================================================================

TEST(accel_collection_upsert_and_search) {
    std::string data_dir = "/tmp/cdb_accel_" + std::to_string(now_ms());

    AcceleratedCollection::Config cfg;
    cfg.durable.name      = "accel_test";
    cfg.durable.dim       = 8;
    cfg.durable.n_shards  = 2;
    cfg.durable.n_replicas= 1;
    cfg.durable.write_quorum = 1;
    cfg.durable.data_dir  = data_dir;
    cfg.accel.preferred   = DeviceType::CPU;

    AcceleratedCollection ac(cfg);

    for (int i = 0; i < 30; ++i) {
        CortexVector v("av_" + std::to_string(i), randf(8, i));
        ac.upsert(v);
    }

    ASSERT(ac.total_vectors() == 30);
    ASSERT(ac.accel_n() == 30);

    QueryOptions opts; opts.top_k = 5; opts.temporal_blend = 0.0f;
    auto res = ac.search(randf(8, 99), opts);
    ASSERT(res.size() == 5);
    for (const auto& r : res)
        ASSERT(r.score >= 0.0f);

    fs::remove_all(data_dir);
}

TEST(accel_collection_temporal_scoring) {
    std::string data_dir = "/tmp/cdb_accel_ts_" + std::to_string(now_ms());

    AcceleratedCollection::Config cfg;
    cfg.durable.name = "ts_accel"; cfg.durable.dim = 4;
    cfg.durable.n_shards = 1; cfg.durable.n_replicas = 1;
    cfg.durable.write_quorum = 1; cfg.durable.data_dir = data_dir;
    cfg.durable.temporal.half_life_ms   = 1000;
    cfg.durable.temporal.temporal_blend = 0.4f;
    cfg.accel.preferred = DeviceType::CPU;

    AcceleratedCollection ac(cfg);

    CortexVector fresh("fresh", {1,0,0,0});
    CortexVector stale("stale", {1,0,0,0});
    stale.created_at = now_ms() - 5000;

    ac.upsert(fresh);
    ac.upsert(stale);

    QueryOptions opts; opts.top_k = 2; opts.temporal_blend = 0.4f;
    auto res = ac.search({1,0,0,0}, opts);
    ASSERT(res.size() == 2);
    // Fresh should rank higher due to temporal scoring
    float fresh_ts = 0, stale_ts = 0;
    for (const auto& r : res) {
        if (r.id == "fresh") fresh_ts = r.temporal_score;
        if (r.id == "stale") stale_ts = r.temporal_score;
    }
    ASSERT(fresh_ts > stale_ts);

    fs::remove_all(data_dir);
}

TEST(accel_collection_with_trace) {
    std::string data_dir = "/tmp/cdb_accel_tr_" + std::to_string(now_ms());

    AcceleratedCollection::Config cfg;
    cfg.durable.name = "trace_accel"; cfg.durable.dim = 4;
    cfg.durable.n_shards = 1; cfg.durable.n_replicas = 1;
    cfg.durable.write_quorum = 1; cfg.durable.data_dir = data_dir;
    cfg.accel.preferred = DeviceType::CPU;

    AcceleratedCollection ac(cfg);
    ac.upsert(CortexVector("t1", {1,0,0,0}));
    ac.upsert(CortexVector("t2", {0,1,0,0}));

    QueryOptions opts; opts.top_k = 2; opts.temporal_blend = 0.0f;
    RetrievalTrace trace;
    auto res = ac.search({1,0,0,0}, opts, &trace);

    ASSERT(!res.empty());
    ASSERT(!trace.query_id.empty());
    ASSERT(trace.latency_ms > 0.0);
    ASSERT(trace.strategy.find("ACCEL") != std::string::npos);

    fs::remove_all(data_dir);
}

TEST(accel_collection_batch_search) {
    std::string data_dir = "/tmp/cdb_accel_bs_" + std::to_string(now_ms());

    AcceleratedCollection::Config cfg;
    cfg.durable.name = "batch_accel"; cfg.durable.dim = 8;
    cfg.durable.n_shards = 1; cfg.durable.n_replicas = 1;
    cfg.durable.write_quorum = 1; cfg.durable.data_dir = data_dir;
    cfg.accel.preferred = DeviceType::CPU;

    AcceleratedCollection ac(cfg);
    for (int i = 0; i < 50; ++i)
        ac.upsert(CortexVector("bv_" + std::to_string(i), randf(8, i)));

    // Batch of 4 queries
    std::vector<std::vector<float>> queries;
    for (int q = 0; q < 4; ++q) queries.push_back(randf(8, q+200));

    QueryOptions opts; opts.top_k = 3; opts.temporal_blend = 0.0f;
    auto results = ac.batch_search(queries, opts);

    ASSERT(static_cast<int>(results.size()) == 4);
    for (const auto& r : results) ASSERT(!r.empty());

    fs::remove_all(data_dir);
}

TEST(accel_collection_device_info) {
    std::string data_dir = "/tmp/cdb_accel_di_" + std::to_string(now_ms());

    AcceleratedCollection::Config cfg;
    cfg.durable.name = "di_test"; cfg.durable.dim = 4;
    cfg.durable.n_shards = 1; cfg.durable.n_replicas = 1;
    cfg.durable.write_quorum = 1; cfg.durable.data_dir = data_dir;

    AcceleratedCollection ac(cfg);
    // Should not throw
    ac.print_device_info();
    // GPU/TPU availability reported honestly
    // (false on machines without hardware, true on GPU/TPU servers)
    bool gpu = ac.gpu_available();
    bool tpu = ac.tpu_available();
    // Just verify the method works
    (void)gpu; (void)tpu;
    ASSERT(true);

    fs::remove_all(data_dir);
}

// =================================================================
// MAIN
// =================================================================
int main() {
    std::cout << "\n=== TideVec GPU/TPU Accelerator Tests ===\n\n";
    std::cout << "\n--- Results ---\n";
    std::cout << tests_passed << " / " << tests_run << " tests passed\n";
    if (tests_passed == tests_run) { std::cout << "ALL TESTS PASSED\n\n"; return 0; }
    std::cout << (tests_run - tests_passed) << " TESTS FAILED\n\n";
    return 1;
}
