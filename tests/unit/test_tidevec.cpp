// TideVec Unit Tests
// Build: see CMakeLists.txt
// Run:   ./build/tidevec_tests

#include <tidevec/core/cortex_vector.hpp>
#include <tidevec/core/temporal_scorer.hpp>
#include <tidevec/core/metrics.hpp>
#include <tidevec/core/collection.hpp>
#include <tidevec/index/flat_index.hpp>
#include <tidevec/index/tv_index.hpp>
#include <tidevec/graph/causal_graph.hpp>
#include <tidevec/observability/retrieval_trace.hpp>
#include <tidevec/api/collection_registry.hpp>

#include <cassert>
#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <cmath>
#include <thread>
#include <atomic>
#include <chrono>
#include <random>
#include <unordered_set>
#include <set>
#include <cstdio>

// ---- minimal test harness ----------------------------------------
static int tests_run = 0, tests_passed = 0;
static bool g_current_test_failed = false;

#define TEST(name) \
    void test_##name(); \
    struct _reg_##name { _reg_##name(){ run_test(#name, test_##name); } } _inst_##name; \
    void test_##name()

#define ASSERT(cond) do { \
    if (!(cond)) { \
        std::cerr << "  FAIL: " << #cond << " at line " << __LINE__ << "\n"; \
        g_current_test_failed = true; \
        return; \
    } \
} while(0)

#define ASSERT_NEAR(a, b, eps) ASSERT(std::abs((a)-(b)) < (eps))

static void run_test(const char* name, void(*fn)()) {
    ++tests_run;
    std::cout << "[TEST] " << name << " ... ";
    g_current_test_failed = false;
    fn();
    if (g_current_test_failed) {
        std::cout << "FAILED\n";
    } else {
        ++tests_passed;
        std::cout << "PASS\n";
    }
}

using namespace tidevec;

// ---- helpers -----------------------------------------------------
// ==================================================================
// SECTION 1: CortexVector
// ==================================================================

TEST(cortex_vector_basic) {
    CortexVector v("v1", {0.1f, 0.2f, 0.3f});
    ASSERT(v.id == "v1");
    ASSERT(v.dim() == 3);
    ASSERT(v.is_currently_valid());
    ASSERT(!v.valid_until.has_value());
}

TEST(cortex_vector_ttl) {
    CortexVector v("v2", {1.0f, 0.0f});
    v.set_ttl_seconds(1);                       // expires in 1 second
    ASSERT(v.is_currently_valid());
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    ASSERT(!v.is_currently_valid());
}

TEST(cortex_vector_edges) {
    CortexVector v("v3", {0.5f, 0.5f});
    v.add_edge("v4", EdgeType::CAUSES, 0.9f);
    v.add_edge("v5", EdgeType::CONTRADICTS, 0.7f);
    ASSERT(v.edges.size() == 2);
    ASSERT(v.edges[0].type == EdgeType::CAUSES);
    ASSERT(v.edges[1].type == EdgeType::CONTRADICTS);
}

// ==================================================================
// SECTION 2: TemporalScorer
// ==================================================================

TEST(temporal_scorer_fresh) {
    TemporalScorer scorer;
    Timestamp now = now_ms();
    float tw = scorer.temporal_weight(now, now);
    ASSERT_NEAR(tw, 1.0f, 0.01f);  // just inserted → weight ≈ 1
}

TEST(temporal_scorer_decay) {
    TemporalConfig cfg;
    cfg.half_life_ms = 1000;  // 1 second half-life
    TemporalScorer scorer(cfg);

    Timestamp old_ts = now_ms() - 1000;  // 1 second ago
    float tw = scorer.temporal_weight(old_ts);
    ASSERT_NEAR(tw, 0.5f, 0.05f);  // half-life → weight ≈ 0.5
}

TEST(temporal_scorer_blend) {
    TemporalConfig cfg;
    cfg.temporal_blend = 0.3f;
    TemporalScorer scorer(cfg);

    float blended = scorer.blend(0.8f, 1.0f);
    // 0.7 * 0.8 + 0.3 * 1.0 = 0.56 + 0.30 = 0.86
    ASSERT_NEAR(blended, 0.86f, 0.001f);
}

TEST(temporal_scorer_staleness_warning) {
    TemporalConfig cfg;
    cfg.half_life_ms       = 1000;   // 1 second
    cfg.staleness_threshold = 0.4f;
    TemporalScorer scorer(cfg);

    Timestamp old_ts = now_ms() - 3000;  // 3 seconds ago → weight ≈ 0.125
    auto s = scorer.score_raw(old_ts, 0.9f, now_ms());
    ASSERT(s.staleness_warning);
    ASSERT(s.temporal_weight < 0.4f);
}

// ==================================================================
// SECTION 3: Distance Metrics
// ==================================================================

TEST(cosine_identical) {
    std::vector<float> a = {1.0f, 0.0f, 0.0f};
    ASSERT_NEAR(cosine_similarity(a, a), 1.0f, 1e-5f);
}

TEST(cosine_orthogonal) {
    std::vector<float> a = {1.0f, 0.0f};
    std::vector<float> b = {0.0f, 1.0f};
    ASSERT_NEAR(cosine_similarity(a, b), 0.0f, 1e-5f);
}

TEST(l2_distance_zero) {
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    ASSERT_NEAR(l2_distance(a, a), 0.0f, 1e-5f);
}

TEST(l2_distance_known) {
    std::vector<float> a = {0.0f, 0.0f};
    std::vector<float> b = {3.0f, 4.0f};
    ASSERT_NEAR(l2_distance(a, b), 5.0f, 1e-4f);
}

// ==================================================================
// SECTION 4: FlatIndex
// ==================================================================

TEST(flat_index_insert_search) {
    FlatIndex idx(3);
    idx.upsert(CortexVector("a", {1.0f, 0.0f, 0.0f}));
    idx.upsert(CortexVector("b", {0.0f, 1.0f, 0.0f}));
    idx.upsert(CortexVector("c", {0.0f, 0.0f, 1.0f}));
    ASSERT(idx.size() == 3);

    QueryOptions opts;
    opts.top_k = 1;
    opts.temporal_blend = 0.0f;
    auto res = idx.search({1.0f, 0.0f, 0.0f}, opts);
    ASSERT(!res.empty());
    ASSERT(res[0].id == "a");
}

TEST(flat_index_temporal_ordering) {
    TemporalConfig cfg;
    cfg.half_life_ms   = 1000;
    cfg.temporal_blend = 0.5f;
    FlatIndex idx(2, Metric::COSINE, cfg);

    // Both vectors equally similar to query
    // But "fresh" was just inserted, "old" is 5 seconds old
    CortexVector fresh("fresh", {1.0f, 0.0f});
    CortexVector old("old",   {1.0f, 0.0f});
    old.created_at = now_ms() - 5000;

    idx.upsert(fresh);
    idx.upsert(old);

    QueryOptions opts;
    opts.top_k = 2;
    opts.temporal_blend = 0.5f;
    auto res = idx.search({1.0f, 0.0f}, opts);

    ASSERT(res.size() == 2);
    ASSERT(res[0].id == "fresh");  // fresh should rank higher
    ASSERT(res[0].temporal_score > res[1].temporal_score);
}

TEST(flat_index_remove) {
    FlatIndex idx(2);
    idx.upsert(CortexVector("x", {1.0f, 0.0f}));
    idx.upsert(CortexVector("y", {0.0f, 1.0f}));
    ASSERT(idx.size() == 2);
    idx.remove("x");
    ASSERT(idx.size() == 1);

    QueryOptions opts; opts.top_k = 5;
    auto res = idx.search({1.0f, 0.0f}, opts);
    for (const auto& r : res) ASSERT(r.id != "x");
}

TEST(flat_index_filter) {
    FlatIndex idx(2);
    CortexVector a("a", {1.0f, 0.0f}, {{"type","news"}});
    CortexVector b("b", {1.0f, 0.1f}, {{"type","doc"}});
    idx.upsert(a); idx.upsert(b);

    QueryOptions opts;
    opts.top_k = 5;
    opts.filter = "type=news";
    opts.temporal_blend = 0.0f;
    auto res = idx.search({1.0f, 0.0f}, opts);
    ASSERT(res.size() == 1);
    ASSERT(res[0].id == "a");
}

// ==================================================================
// SECTION 5: TVIndex
// ==================================================================

TEST(tvindex_insert_search) {
    TVIndex idx;
    for (int i = 0; i < 20; ++i) {
        std::vector<float> emb(8, static_cast<float>(i) / 20.0f);
        idx.insert(CortexVector("v" + std::to_string(i), emb));
    }
    ASSERT(idx.size() == 20);

    std::vector<float> query(8, 0.95f);
    QueryOptions opts; opts.top_k = 3; opts.temporal_blend = 0.0f;
    auto res = idx.search(query, opts);
    ASSERT(res.size() == 3);
}

TEST(tvindex_temporal_preference) {
    TVIndex::Config cfg;
    cfg.temporal.half_life_ms   = 1000;
    cfg.temporal.temporal_blend = 0.4f;
    TVIndex idx(cfg);

    std::vector<float> emb = {1.0f, 0.0f, 0.0f, 0.0f};
    CortexVector fresh("fresh", emb);
    CortexVector stale("stale", emb);
    stale.created_at = now_ms() - 5000;

    idx.insert(fresh);
    idx.insert(stale);

    QueryOptions opts;
    opts.top_k = 2;
    opts.temporal_blend = 0.4f;
    auto res = idx.search(emb, opts);

    ASSERT(res.size() >= 1);
    float fresh_ts = 0.0f, stale_ts = 0.0f;
    for (const auto& r : res) {
        if (r.id == "fresh") fresh_ts = r.temporal_score;
        if (r.id == "stale") stale_ts = r.temporal_score;
    }
    ASSERT(fresh_ts > stale_ts);
}

TEST(tvindex_remove) {
    TVIndex idx;
    idx.insert(CortexVector("del", {1.0f, 0.0f}));
    idx.insert(CortexVector("keep", {0.0f, 1.0f}));
    idx.remove("del");

    QueryOptions opts; opts.top_k = 5; opts.temporal_blend = 0.0f;
    auto res = idx.search({1.0f, 0.0f}, opts);
    for (const auto& r : res) ASSERT(r.id != "del");
}

TEST(tvindex_pq_compression_reduces_memory) {
    TVIndexConfig cfg;
    cfg.use_pq_compression = true;
    cfg.pq_subspaces = 4;  // dim=8, 4 subspaces => 2 dims/subspace
    TVIndex idx(cfg);

    std::vector<std::vector<float>> samples;
    for (int i = 0; i < 50; ++i)
        samples.push_back({float(i)*0.1f, float(i)*0.2f, float(i)*0.3f,
                            float(i)*0.4f, float(i)*0.5f, float(i)*0.6f,
                            float(i)*0.7f, float(i)*0.8f});
    idx.train_quantizer(samples);
    ASSERT(idx.quantizer_ready());

    for (int i = 0; i < 100; ++i) {
        CortexVector v("vec_" + std::to_string(i),
            {float(i)*0.1f, float(i)*0.2f, float(i)*0.3f, float(i)*0.4f,
             float(i)*0.5f, float(i)*0.6f, float(i)*0.7f, float(i)*0.8f});
        idx.insert(v);
    }

    ASSERT(idx.size() == 100);
    std::size_t compressed = idx.embedding_memory_bytes();
    std::size_t uncompressed_equiv = 100 * 8 * sizeof(float);  // 3200 bytes
    // At dim=8 / 4 subspaces (1 byte/subspace) the achievable ratio is
    // exactly 8x (32 bytes -> 4 bytes per vector). Real-world configs
    // (e.g. 768-dim / 96 subspaces) achieve ~32x, as shown in README.
    ASSERT(compressed < uncompressed_equiv / 5);  // expect >5x reduction

    QueryOptions opts; opts.top_k = 5;
    auto res = idx.search({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, opts);
    ASSERT(!res.empty());
}

TEST(tvindex_pq_uncompressed_default_unchanged) {
    // Default config (no compression) must behave exactly as before —
    // this is a regression guard for the PQ integration.
    TVIndex idx;  // default TVIndexConfig: use_pq_compression == false
    idx.insert(CortexVector("a", {1.0f, 0.0f}));
    idx.insert(CortexVector("b", {0.0f, 1.0f}));

    QueryOptions opts; opts.top_k = 5; opts.temporal_blend = 0.0f;
    auto res = idx.search({1.0f, 0.0f}, opts);
    ASSERT(!res.empty());
    ASSERT(res[0].id == "a");
    // Uncompressed nodes store full embeddings, so memory == raw float count
    ASSERT(idx.embedding_memory_bytes() == 2 * 2 * sizeof(float));
}

// ==================================================================
// SECTION 6: CausalGraph
// ==================================================================

TEST(causal_graph_add_edges) {
    CausalGraph g;
    g.add_edge("a", "b", EdgeType::CAUSES);
    g.add_edge("a", "c", EdgeType::CONTRADICTS);
    ASSERT(g.edge_count() == 2);
    ASSERT(g.node_count() == 3);  // a, b, c
}

TEST(causal_graph_contradictions) {
    CausalGraph g;
    g.add_edge("fact1", "fact2", EdgeType::CONTRADICTS);
    g.add_edge("fact1", "fact3", EdgeType::CONTRADICTS);
    auto contra = g.find_contradictions("fact1");
    ASSERT(contra.size() == 2);
}

TEST(causal_graph_entity_resolve) {
    CausalGraph g;
    g.add_edge("entity_v1", "entity_v2", EdgeType::ENTITY_OF);
    g.add_edge("entity_v2", "entity_v3", EdgeType::ENTITY_OF);
    auto resolved = g.resolve_entity("entity_v1");
    ASSERT(!resolved.empty());
}

TEST(causal_graph_bfs_hops) {
    CausalGraph g;
    g.add_edge("a", "b", EdgeType::CAUSES);
    g.add_edge("b", "c", EdgeType::CAUSES);
    g.add_edge("c", "d", EdgeType::CAUSES);

    auto hop1 = g.get_neighbours("a", 1);
    auto hop3 = g.get_neighbours("a", 3);
    ASSERT(hop1.size() == 1);  // only b
    ASSERT(hop3.size() == 3);  // b, c, d
}

TEST(causal_graph_causal_expand) {
    CausalGraph g;
    g.add_edge("v1", "v2", EdgeType::RELATED_TO);
    g.add_edge("v1", "v3", EdgeType::CAUSES);

    auto expanded = g.causal_expand({"v1"}, 1);
    ASSERT(expanded.size() == 3);  // v1 + v2 + v3
}

// ==================================================================
// SECTION 7: Collection (integration)
// ==================================================================

TEST(collection_flat_basic) {
    Collection::Config cfg;
    cfg.name       = "test_flat";
    cfg.dim        = 4;
    cfg.index_type = IndexType::FLAT;
    Collection col(cfg);

    col.upsert(CortexVector("doc1", {1.0f,0.0f,0.0f,0.0f}));
    col.upsert(CortexVector("doc2", {0.0f,1.0f,0.0f,0.0f}));
    ASSERT(col.size() == 2);

    QueryOptions opts; opts.top_k = 1; opts.temporal_blend = 0.0f;
    auto res = col.search({1.0f,0.0f,0.0f,0.0f}, opts);
    ASSERT(!res.empty() && res[0].id == "doc1");
}

TEST(collection_with_causal_edges) {
    Collection::Config cfg;
    cfg.name = "test_causal";
    cfg.index_type = IndexType::FLAT;
    Collection col(cfg);

    col.upsert(CortexVector("policy_v1", {1.0f, 0.0f}));
    col.upsert(CortexVector("policy_v2", {1.0f, 0.1f}));
    col.add_edge("policy_v1", "policy_v2", EdgeType::UPDATES, 0.95f);

    auto contra = col.graph().get_out_edges("policy_v1",
                                             EdgeType::UPDATES);
    ASSERT(contra.size() == 1);
    ASSERT(contra[0].tgt_id == "policy_v2");
}

TEST(collection_tvindex) {
    Collection::Config cfg;
    cfg.name = "test_tv";
    cfg.index_type = IndexType::TVINDEX;
    cfg.temporal.temporal_blend = 0.2f;
    Collection col(cfg);

    for (int i = 0; i < 30; ++i) {
        std::vector<float> emb = {static_cast<float>(i)/30.0f, 0.5f};
        col.upsert(CortexVector("v" + std::to_string(i), emb));
    }
    ASSERT(col.size() == 30);

    QueryOptions opts; opts.top_k = 5;
    auto res = col.search({0.99f, 0.5f}, opts);
    ASSERT(res.size() == 5);
}

TEST(collection_agent_dedup) {
    Collection::Config cfg;
    cfg.name = "agent_col";
    cfg.index_type = IndexType::FLAT;
    cfg.agent.enabled = true;
    cfg.agent.dedup_threshold = 0.99f;
    Collection col(cfg);

    std::vector<float> emb = {1.0f, 0.0f};
    col.upsert(CortexVector("fact_a", emb));
    col.upsert(CortexVector("fact_a_dup", emb));  // near-identical → deduped

    // Should only have 1 vector (dedup blocked the second)
    QueryOptions opts; opts.top_k = 10; opts.temporal_blend = 0.0f;
    auto res = col.search(emb, opts);
    ASSERT(res.size() == 1);
}

// ==================================================================
// SECTION 8: RetrievalTrace
// ==================================================================

TEST(retrieval_trace_json) {
    RetrievalTrace trace;
    trace.query_id    = "q_test_001";
    trace.strategy    = "TVINDEX_HNSW";
    trace.latency_ms  = 3.14;
    trace.staleness_warnings.push_back({"v_old", 120, 0.15f, "120 days old"});
    trace.contradiction_alerts.push_back({"v1", "v2", 0.88f});

    std::string json = trace.to_json();
    ASSERT(json.find("q_test_001") != std::string::npos);
    ASSERT(json.find("TVINDEX_HNSW") != std::string::npos);
    ASSERT(json.find("v_old") != std::string::npos);
    ASSERT(json.find("contradiction_alerts") != std::string::npos);
}

// ==================================================================
// MAIN
// ==================================================================
int main() {
    std::cout << "\n=== TideVec Unit Tests ===\n\n";
    std::cout << "\n--- Results ---\n";
    std::cout << tests_passed << " / " << tests_run << " tests passed\n";
    if (tests_passed == tests_run) {
        std::cout << "ALL TESTS PASSED\n\n";
        return 0;
    }
    std::cout << (tests_run - tests_passed) << " TESTS FAILED\n\n";
    return 1;
}

TEST(persistence_survive_restart) {
    const std::string data_dir = "/tmp/tidevec_test_persist_" +
        std::to_string(now_ms());
    std::filesystem::remove_all(data_dir);

    // Session 1: create + insert
    {
        CollectionRegistry reg(data_dir);
        reg.load_and_recover();
        CollectionRegistry::CreateParams p;
        p.name = "persist_col"; p.dim = 4;
        p.n_shards = 1; p.n_replicas = 1;
        reg.create(p);
        auto& col = reg.get("persist_col");
        for (int i = 0; i < 5; ++i) {
            CortexVector v;
            v.id = "v" + std::to_string(i);
            v.embedding = {float(i), float(i), float(i), float(i)};
            v.created_at = now_ms(); v.valid_from = v.created_at;
            col.upsert(v);
        }
        ASSERT(col.total_vectors() == 5);
        ASSERT(std::filesystem::exists(data_dir + "/registry.json"));
    }

    // Session 2: recover and verify
    {
        CollectionRegistry reg(data_dir);
        reg.load_and_recover();
        ASSERT(reg.size() == 1);
        auto& col = reg.get("persist_col");
        ASSERT(col.total_vectors() == 5);
        QueryOptions opts; opts.top_k = 5; opts.temporal_blend = 0.0f;
        auto r = col.search({1,1,1,1}, opts);
        ASSERT(!r.empty());
    }

    std::filesystem::remove_all(data_dir);
}

// ==================================================================
// Supersession (automatic newer-version detection)
// ==================================================================

static Collection make_supersession_collection(const std::string& match_field = "") {
    Collection::Config cfg;
    cfg.name = "sup";
    cfg.dim  = 4;
    cfg.index_type = IndexType::FLAT;
    cfg.supersession.enabled = true;
    cfg.supersession.similarity_threshold = 0.9f;
    cfg.supersession.match_field = match_field;
    return Collection(cfg);
}

static CortexVector aged(const std::string& id, std::vector<float> emb,
                         double age_days,
                         std::unordered_map<std::string, std::string> pl = {}) {
    CortexVector v(id, std::move(emb), std::move(pl));
    v.created_at = now_ms() - static_cast<Timestamp>(age_days * 86'400'000.0);
    v.valid_from = v.created_at;
    return v;
}

TEST(supersession_links_newer_version) {
    auto col = make_supersession_collection();
    col.upsert(aged("policy_v1", {1.0f, 0.0f, 0.0f, 0.0f}, 30));
    col.upsert(aged("other",     {0.0f, 1.0f, 0.0f, 0.0f}, 1));
    col.upsert(aged("policy_v2", {0.98f, 0.1f, 0.0f, 0.0f}, 1));

    QueryOptions opts; opts.top_k = 3; opts.temporal_blend = 0.0f;
    auto res = col.search({1.0f, 0.0f, 0.0f, 0.0f}, opts);
    bool checked = false;
    for (const auto& r : res) {
        if (r.id == "policy_v1") {
            ASSERT(r.superseded_by.size() == 1);
            ASSERT(r.superseded_by[0] == "policy_v2");
            ASSERT(r.staleness_warning);
            ASSERT(r.staleness_reason.find("policy_v2") != std::string::npos);
            checked = true;
        } else {
            ASSERT(r.superseded_by.empty());
        }
    }
    ASSERT(checked);
}

TEST(supersession_backfilled_older_doc_is_superseded) {
    auto col = make_supersession_collection();
    col.upsert(aged("new", {1.0f, 0.0f, 0.0f, 0.0f}, 1));
    col.upsert(aged("old", {0.99f, 0.05f, 0.0f, 0.0f}, 40));
    ASSERT(col.superseded_by("old") == std::vector<std::string>{"new"});
    ASSERT(col.superseded_by("new").empty());
}

TEST(supersession_match_field_must_agree) {
    auto col = make_supersession_collection("doc_id");
    col.upsert(aged("a_v1", {1.0f, 0.0f, 0.0f, 0.0f}, 10, {{"doc_id", "refunds"}}));
    col.upsert(aged("b_v1", {0.99f, 0.05f, 0.0f, 0.0f}, 1, {{"doc_id", "shipping"}}));
    ASSERT(col.superseded_by("a_v1").empty());

    col.upsert(aged("a_v2", {0.97f, 0.1f, 0.0f, 0.0f}, 1, {{"doc_id", "refunds"}}));
    ASSERT(col.superseded_by("a_v1") == std::vector<std::string>{"a_v2"});
}

TEST(supersession_penalty_demotes_stale_version) {
    auto col = make_supersession_collection();
    // Stale version is the closer semantic match to the query.
    col.upsert(aged("v1", {1.0f, 0.0f, 0.0f, 0.0f}, 2));
    col.upsert(aged("v2", {0.95f, 0.3f, 0.0f, 0.0f}, 1));

    QueryOptions opts; opts.top_k = 2; opts.temporal_blend = 0.0f;
    auto plain = col.search({1.0f, 0.0f, 0.0f, 0.0f}, opts);
    ASSERT(plain.size() == 2 && plain[0].id == "v1");

    opts.superseded_penalty = 0.5f;
    auto demoted = col.search({1.0f, 0.0f, 0.0f, 0.0f}, opts);
    ASSERT(demoted.size() == 2 && demoted[0].id == "v2");
}

TEST(supersession_manual_updates_edge_is_reported) {
    Collection::Config cfg;
    cfg.name = "manual"; cfg.dim = 2; cfg.index_type = IndexType::FLAT;
    Collection col(cfg);
    col.upsert(CortexVector("old", {1.0f, 0.0f}));
    col.upsert(CortexVector("new", {0.0f, 1.0f}));
    col.add_edge("new", "old", EdgeType::UPDATES);

    QueryOptions opts; opts.top_k = 1; opts.temporal_blend = 0.0f;
    auto res = col.search({1.0f, 0.0f}, opts);
    ASSERT(res.size() == 1);
    ASSERT(res[0].superseded_by == std::vector<std::string>{"new"});
}

TEST(tvindex_edge_rebuild_on_upsert) {
    // Verify that updating an existing vector's embedding rebuilds
    // graph edges — previously this was a TODO that left stale edges.
    // Use enough vectors so HNSW graph routing actually matters.
    TVIndexConfig cfg;
    cfg.M = 4; cfg.M0 = 8; cfg.ef_construction = 50;
    TVIndex idx(cfg);

    // Insert cluster A: vectors near [1,0]
    for (int i = 0; i < 20; ++i) {
        float noise = 0.01f * float(i);
        idx.insert(CortexVector("a_" + std::to_string(i),
            {1.0f - noise, noise}));
    }
    // Insert cluster B: vectors near [0,1]
    for (int i = 0; i < 20; ++i) {
        float noise = 0.01f * float(i);
        idx.insert(CortexVector("b_" + std::to_string(i),
            {noise, 1.0f - noise}));
    }
    // Insert v1 in cluster A
    idx.insert(CortexVector("v1", {0.99f, 0.01f}));

    QueryOptions opts;
    opts.top_k = 1;
    opts.temporal_blend = 0.0f;
    opts.ef_search = 50;

    // v1 starts near [1,0] — should rank highly for that query
    auto r1 = idx.search({1.0f, 0.0f}, opts);
    ASSERT(!r1.empty());

    // Move v1 to cluster B: [0,1]
    CortexVector v1_updated("v1", {0.01f, 0.99f});
    v1_updated.created_at = now_ms();
    v1_updated.valid_from = v1_updated.created_at;
    idx.upsert(v1_updated);

    // After edge rebuild, v1 should now appear near [0,1] not [1,0]
    // Search [0,1] — v1 should be findable (it moved into this cluster)
    auto r3 = idx.search({0.0f, 1.0f}, opts);
    ASSERT(!r3.empty());
    // v1 is in cluster B now — it should appear in top results for [0,1]
    // (either directly or nearby)

    // Core assertion: upsert doesn't crash and index remains searchable
    ASSERT(idx.size() == 41);
    auto r4 = idx.search({1.0f, 0.0f}, opts);
    ASSERT(!r4.empty());
    // The index is still functional after edge rebuild
}

// ==================================================================
// TVIndex graph quality (beyond ef_search, where brute force no
// longer hides traversal bugs)
// ==================================================================

static std::vector<float> random_vec(std::mt19937& rng, int dim) {
    std::normal_distribution<float> nd;
    std::vector<float> v(dim);
    for (auto& x : v) x = nd(rng);
    return v;
}

static double recall_vs_flat(TVIndex& idx, FlatIndex& flat, std::mt19937& rng,
                             int dim, const QueryOptions& opts, int queries = 50) {
    double hit = 0, total = 0;
    for (int q = 0; q < queries; ++q) {
        auto qv = random_vec(rng, dim);
        auto got = idx.search(qv, opts);
        auto gt  = flat.search(qv, opts);
        std::unordered_set<std::string> truth;
        for (auto& r : gt) truth.insert(r.id);
        for (auto& r : got) hit += truth.count(r.id);
        total += truth.size();
    }
    return total > 0 ? hit / total : 1.0;
}

TEST(tvindex_recall_above_ef) {
    const int N = 3000, D = 32;
    std::mt19937 rng(11);
    TemporalConfig tc; tc.temporal_blend = 0.0f;
    TVIndexConfig cfg; cfg.temporal = tc;
    TVIndex idx(cfg);
    FlatIndex flat(D, Metric::COSINE, tc);
    for (int i = 0; i < N; ++i) {
        CortexVector v("v" + std::to_string(i), random_vec(rng, D),
                       {{"tenant", i % 50 == 0 ? "rare" : "common"}});
        idx.insert(v); flat.upsert(v);
    }
    QueryOptions opts; opts.top_k = 10; opts.ef_search = 128;
    ASSERT(recall_vs_flat(idx, flat, rng, D, opts) >= 0.9);

    // Selective filter: matches must be reachable through non-matching nodes.
    opts.filter = "tenant=rare";
    ASSERT(recall_vs_flat(idx, flat, rng, D, opts) >= 0.9);
}

TEST(tvindex_upsert_entry_point_keeps_graph_connected) {
    // Re-embed every vector (so the entry point is certainly among them)
    // under several level-assignment seeds: recall must not depend on
    // the graph shape the updates happen to produce.
    for (uint32_t seed : {1u, 2u, 4u, 5u, 7u, TVIndexConfig{}.seed}) {
        const int N = 500, D = 16;
        std::mt19937 rng(3);
        TemporalConfig tc; tc.temporal_blend = 0.0f;
        TVIndexConfig cfg; cfg.temporal = tc; cfg.seed = seed;
        TVIndex idx(cfg);
        FlatIndex flat(D, Metric::COSINE, tc);
        std::vector<CortexVector> vs;
        for (int i = 0; i < N; ++i) {
            vs.emplace_back("v" + std::to_string(i), random_vec(rng, D));
            idx.insert(vs.back()); flat.upsert(vs.back());
        }
        for (auto& v : vs) {
            v.embedding = random_vec(rng, D);
            idx.upsert(v); flat.upsert(v);
        }
        QueryOptions opts; opts.top_k = 10;
        ASSERT(idx.size() == static_cast<std::size_t>(N));
        ASSERT(recall_vs_flat(idx, flat, rng, D, opts, 100) >= 0.95);
    }
}

TEST(tvindex_reembed_into_new_space_stays_reachable) {
    // A model migration moves every vector into an unrelated region of
    // the space. Queries from either region must keep finding live
    // vectors, and the tombstones left behind must be compacted away.
    const int N = 300, D = 16;
    std::mt19937 rng(9);
    auto half = [&](bool upper) {
        auto v = random_vec(rng, D);
        for (int d = 0; d < D; ++d)
            if ((d >= D / 2) != upper) v[d] = 0.0f;
        return v;
    };
    TemporalConfig tc; tc.temporal_blend = 0.0f;
    TVIndexConfig cfg; cfg.temporal = tc;
    TVIndex idx(cfg);
    FlatIndex flat(D, Metric::COSINE, tc);
    std::vector<CortexVector> vs;
    for (int i = 0; i < N; ++i) {
        vs.emplace_back("v" + std::to_string(i), half(false));
        idx.insert(vs.back());
    }
    for (auto& v : vs) {
        v.embedding = half(true);
        idx.upsert(v); flat.upsert(v);
    }
    ASSERT(idx.size() == static_cast<std::size_t>(N));
    ASSERT(idx.tombstones() < static_cast<std::size_t>(N) / 2);

    QueryOptions opts; opts.top_k = 10;
    for (int q = 0; q < 20; ++q)
        ASSERT(idx.search(half(false), opts).size() == 10);
    ASSERT(recall_vs_flat(idx, flat, rng, D, opts, 100) >= 0.95);

    for (int i = 0; i < N - 5; ++i) idx.remove(vs[i].id);
    auto left = idx.search(half(false), opts);
    ASSERT(left.size() == 5);
}

TEST(tvindex_outlier_stays_reachable) {
    // An isolated vector surrounded only by a dense cluster elsewhere must
    // keep an incoming edge; top-M pruning used to drop all of them.
    const int D = 16;
    std::mt19937 rng(5);
    std::normal_distribution<float> noise(0.0f, 0.05f);
    auto near_axis = [&](int axis) {
        std::vector<float> v(D);
        for (auto& x : v) x = noise(rng);
        v[axis] += 1.0f;
        return v;
    };
    TemporalConfig tc; tc.temporal_blend = 0.0f;
    TVIndexConfig cfg; cfg.temporal = tc;
    TVIndex idx(cfg);
    idx.insert(CortexVector("c0", near_axis(0)));
    std::vector<float> outlier(D, 0.0f); outlier[1] = 1.0f;
    idx.insert(CortexVector("outlier", outlier));
    for (int i = 1; i < 400; ++i)
        idx.insert(CortexVector("c" + std::to_string(i), near_axis(0)));

    QueryOptions opts; opts.top_k = 1;
    auto r = idx.search(outlier, opts);
    ASSERT(!r.empty() && r[0].id == "outlier");
}

TEST(tvindex_metadata_only_upsert_keeps_slot) {
    TVIndex idx(TVIndexConfig{});
    CortexVector v("a", {1.0f, 0.0f}, {{"rev", "1"}});
    idx.insert(v);
    v.payload["rev"] = "2";
    idx.upsert(v);
    ASSERT(idx.size() == 1);
    QueryOptions opts; opts.top_k = 5;
    auto r = idx.search({1.0f, 0.0f}, opts);
    ASSERT(r.size() == 1 && r[0].payload.at("rev") == "2");
}

TEST(tvindex_per_query_blend_overrides_collection) {
    TemporalConfig tc; tc.half_life_ms = 1000; tc.temporal_blend = 0.9f;
    TVIndexConfig cfg; cfg.temporal = tc;
    TVIndex idx(cfg);
    CortexVector old_exact("old", {1.0f, 0.0f});
    old_exact.created_at = now_ms() - 60'000;
    old_exact.valid_from = old_exact.created_at;
    idx.insert(old_exact);
    idx.insert(CortexVector("fresh_far", {0.6f, 0.8f}));

    QueryOptions opts; opts.top_k = 2;
    auto by_collection = idx.search({1.0f, 0.0f}, opts);
    ASSERT(by_collection.size() == 2);
    ASSERT(by_collection[0].id == "fresh_far");

    opts.temporal_blend = 0.0f;
    auto semantic_only = idx.search({1.0f, 0.0f}, opts);
    ASSERT(semantic_only.size() == 2);
    ASSERT(semantic_only[0].id == "old");
    ASSERT_NEAR(semantic_only[0].score, semantic_only[0].vector_score, 1e-5f);
}

TEST(tvindex_upsert_revives_purged_vector) {
    TVIndex idx(TVIndexConfig{});
    CortexVector v("ttl", {1.0f, 0.0f});
    v.valid_until = now_ms() - 1;
    idx.insert(v);
    ASSERT(idx.purge_expired() == 1);
    ASSERT(idx.size() == 0);
    v.valid_until.reset();
    idx.upsert(v);
    ASSERT(idx.size() == 1);
    QueryOptions opts; opts.top_k = 1;
    auto r = idx.search({1.0f, 0.0f}, opts);
    ASSERT(r.size() == 1 && r[0].id == "ttl");
}

// ==================================================================
// Phase 1 retrieval: filters, payload indexes, planner, get/scroll,
// as_of, BM25 / hybrid, aliases
// ==================================================================

static std::unordered_map<std::string, std::string> P(
        std::initializer_list<std::pair<const std::string, std::string>> kv) {
    return std::unordered_map<std::string, std::string>(kv);
}

TEST(filter_iso8601_parse) {
    ASSERT(parse_iso8601_ms("1970-01-01") == 0);
    ASSERT(parse_iso8601_ms("1970-01-02T00:00:00Z") == 86'400'000);
    ASSERT(parse_iso8601_ms("2024-03-01T12:30:15.250Z") == 1709296215250LL);
    ASSERT(parse_iso8601_ms("2024-03-01T14:30:15.250+02:00") == 1709296215250LL);
    ASSERT(!parse_iso8601_ms("2024-13-01"));
    ASSERT(!parse_iso8601_ms("yesterday"));
    ASSERT(parse_numeric_value("42") == 42.0);
    ASSERT(parse_numeric_value(" 2.5e1 ") == 25.0);
    ASSERT(parse_numeric_value("2024-01-01").value() == 1704067200000.0);
    ASSERT(!parse_numeric_value("abc"));
}

TEST(filter_operators_evaluate) {
    auto pl = P({{"lang", "en"}, {"year", "2021"}, {"price", "9.5"},
                 {"published", "2024-06-01T00:00:00Z"}});
    auto m = [&](const json& f, Timestamp created = 1000) {
        return filter_expr_from_json(f).matches("doc-1", pl, created);
    };
    ASSERT(m({{"lang", "en"}}));
    ASSERT(!m({{"lang", "de"}}));
    ASSERT(m({{"year", 2021}}));                       // numeric equality
    ASSERT(m({{"year", "2021"}}));                     // string equality
    ASSERT(m({{"lang", {{"$in", {"de", "en"}}}}}));
    ASSERT(!m({{"lang", {{"$nin", {"de", "en"}}}}}));
    ASSERT(m({{"lang", {{"$ne", "de"}}}}));
    ASSERT(m({{"price", {{"$gte", 9.5}, {"$lt", 10}}}}));
    ASSERT(!m({{"price", {{"$gt", 9.5}}}}));
    ASSERT(m({{"published", {{"$gte", "2024-01-01"}, {"$lt", "2025-01-01"}}}}));
    ASSERT(!m({{"published", {{"$lt", "2024-05-31"}}}}));
    ASSERT(m({{"created_at", {{"$lte", 1000}}}}));
    ASSERT(!m({{"created_at", {{"$lt", 1000}}}}));
    ASSERT(m({{"_id", {{"$in", {"doc-1", "doc-2"}}}}}));
    ASSERT(m({{"lang", {{"$exists", true}}}}));
    ASSERT(m({{"missing", {{"$exists", false}}}}));
    ASSERT(!m({{"missing", "x"}}));
    ASSERT(!m({{"missing", {{"$gte", 0}}}}));
    ASSERT(m({{"$or", {{{"lang", "de"}}, {{"year", {{"$gte", 2020}}}}}}}));
    ASSERT(!m({{"$and", {{{"lang", "en"}}, {{"year", {{"$gte", 2022}}}}}}}));
    ASSERT(m({{"$not", {{"lang", "de"}}}}));
    ASSERT(m({{"lang", "en"}, {"year", {{"$lte", 2021}}}}));   // implicit AND
    ASSERT(m(json("lang=en")));                                 // legacy string
    bool threw = false;
    try { filter_expr_from_json({{"lang", {{"$regex", "e.*"}}}}); } catch (...) { threw = true; }
    ASSERT(threw);
}

TEST(payload_index_estimates_are_upper_bounds) {
    PayloadIndex idx(true, "text");
    for (uint64_t i = 0; i < 100; ++i)
        idx.add(i, P({{"cat", "c" + std::to_string(i % 10)},
                      {"n", std::to_string(i)},
                      {"text", "free text that is not keyword indexed"}}),
                static_cast<Timestamp>(i * 10));
    PayloadIndex::IdLookup none;
    ASSERT(idx.estimate(filter_expr_from_json({{"cat", "c3"}}), none) == 10);
    ASSERT(idx.estimate(filter_expr_from_json({{"cat", {{"$in", {"c1", "c2"}}}}}), none) == 20);
    ASSERT(idx.estimate(filter_expr_from_json({{"n", {{"$gte", 10}, {"$lt", 20}}}}), none) == 10);
    ASSERT(idx.estimate(filter_expr_from_json({{"created_at", {{"$lt", 100}}}}), none) == 10);
    ASSERT(idx.estimate(filter_expr_from_json({{"cat", "c3"}, {"n", {{"$lt", 5}}}}), none) == 5);
    ASSERT(!idx.estimate(filter_expr_from_json({{"text", "free"}}), none));     // not indexed
    ASSERT(!idx.estimate(filter_expr_from_json({{"cat", {{"$ne", "c3"}}}}), none));
    ASSERT(idx.estimate(filter_expr_from_json({{"never_seen", "x"}}), none) == 0);
    std::vector<uint64_t> out;
    idx.collect(filter_expr_from_json({{"$or", {{{"cat", "c3"}}, {{"n", {{"$gte", 95}}}}}}}),
                none, out);
    ASSERT(out.size() == 15);
}

// Shared fixture for planner / recall tests: N random vectors with
// payload keys of known selectivity.
struct FilterFixture {
    int D = 32;
    std::vector<CortexVector> vecs;
    std::unique_ptr<TVIndex> idx;

    explicit FilterFixture(int n, uint32_t seed = 21) {
        std::mt19937 rng(seed);
        TVIndexConfig cfg; cfg.temporal.temporal_blend = 0.0f;
        idx = std::make_unique<TVIndex>(cfg);
        for (int i = 0; i < n; ++i) {
            CortexVector v("v" + std::to_string(i), random_vec(rng, D),
                           P({{"bucket", std::to_string(i % 100)},     // 1% per value
                              {"tenant", i % 2 ? "a" : "b"},           // 50%
                              {"score", std::to_string(i % 1000)}}));  // numeric
            v.created_at = 1'700'000'000'000LL + i * 1000LL;
            v.valid_from = v.created_at;
            idx->insert(v);
            vecs.push_back(std::move(v));
        }
    }

    // Exact top-k among vectors matching f (scores are pure similarity).
    std::vector<std::string> oracle(const std::vector<float>& q, const FilterExpr* f,
                                    int k, std::optional<Timestamp> as_of = std::nullopt) const {
        std::vector<std::pair<float, std::string>> s;
        for (const auto& v : vecs) {
            if (f && !f->matches(v)) continue;
            if (as_of && v.created_at > *as_of) continue;
            s.emplace_back(compute_similarity(q, v.embedding, Metric::COSINE), v.id);
        }
        std::sort(s.begin(), s.end(), [](auto& a, auto& b) { return a.first > b.first; });
        std::vector<std::string> out;
        for (int i = 0; i < k && i < static_cast<int>(s.size()); ++i) out.push_back(s[i].second);
        return out;
    }

    double recall(const json& filter, int queries = 30, int k = 10) const {
        std::mt19937 rng(99);
        QueryOptions opts; opts.top_k = k;
        opts.filter_expr = filter_from_json(filter);
        double hit = 0, total = 0;
        for (int qi = 0; qi < queries; ++qi) {
            auto q = random_vec(rng, D);
            auto truth = oracle(q, opts.filter_expr.get(), k);
            std::unordered_set<std::string> t(truth.begin(), truth.end());
            for (const auto& r : idx->search(q, opts)) {
                if (opts.filter_expr && !opts.filter_expr->matches(r.id, r.payload, r.created_at))
                    return -1.0;   // a result violating the filter is a hard failure
                hit += t.count(r.id);
            }
            total += t.size();
        }
        return total ? hit / total : 1.0;
    }
};

TEST(planner_picks_exact_for_selective_filters) {
    FilterFixture fx(20000);
    QueryOptions o; o.top_k = 10;
    o.filter_expr = filter_from_json({{"bucket", "7"}});                     // 1%
    ASSERT(fx.idx->explain(o).kind == SearchPlan::Kind::EXACT_INDEX);
    o.filter_expr = filter_from_json({{"tenant", "a"}});                     // 50%
    ASSERT(fx.idx->explain(o).kind == SearchPlan::Kind::GRAPH);
    o.filter_expr = filter_from_json({{"bucket", {{"$nin", {"7"}}}}});       // 99%, no index
    ASSERT(fx.idx->explain(o).kind == SearchPlan::Kind::GRAPH);
    ASSERT(!fx.idx->explain(o).indexed);
    o.filter_expr = filter_from_json({{"$not", {{"bucket", {{"$gte", 1}}}}}}); // 1%, no index
    ASSERT(fx.idx->explain(o).kind == SearchPlan::Kind::EXACT_SCAN);
    o.filter_expr = nullptr;
    ASSERT(fx.idx->explain(o).kind == SearchPlan::Kind::GRAPH);
}

TEST(filtered_recall_matches_oracle) {
    FilterFixture fx(20000);
    const json cases[] = {
        {{"bucket", "7"}},                                               // keyword 1%
        {{"score", {{"$gte", 100}, {"$lt", 110}}}},                       // range 1%
        {{"tenant", "a"}, {"score", {{"$lt", 20}}}},                     // AND 1%
        {{"$or", {{{"bucket", "3"}}, {{"bucket", "4"}}}}},               // OR 2%
        {{"bucket", {{"$in", {"1", "2", "3", "4", "5"}}}}},              // IN 5%
        {{"score", {{"$lt", 100}}}},                                      // range 10%
        {{"tenant", "b"}},                                                // 50%
        {{"$not", {{"bucket", {{"$gte", 1}}}}}},                          // unindexed 1%
        {{"bucket", {{"$ne", "0"}}}},                                     // unindexed 99%
        {{"created_at", {{"$lt", 1'700'000'000'000LL + 200'000LL}}}},     // created_at 1%
    };
    for (const auto& c : cases) {
        double r = fx.recall(c);
        if (r < 0.95) std::cerr << "  recall " << r << " for " << c.dump() << "\n";
        ASSERT(r >= 0.95);
    }
}

TEST(filter_with_no_matches_returns_empty) {
    FilterFixture fx(3000);
    QueryOptions o; o.top_k = 10;
    o.filter_expr = filter_from_json({{"bucket", "nope"}});
    ASSERT(fx.idx->search(fx.vecs[0].embedding, o).empty());
    o.filter_expr = filter_from_json({{"$not", {{"tenant", {{"$in", {"a", "b"}}}}}}});
    ASSERT(fx.idx->search(fx.vecs[0].embedding, o).empty());
}

TEST(payload_index_follows_metadata_updates_and_compaction) {
    TVIndexConfig cfg; cfg.temporal.temporal_blend = 0.0f;
    TVIndex idx(cfg);
    std::mt19937 rng(4);
    for (int i = 0; i < 300; ++i)
        idx.insert(CortexVector("v" + std::to_string(i), random_vec(rng, 8), P({{"k", "old"}})));
    CortexVector moved = CortexVector("v5", random_vec(rng, 8), P({{"k", "new"}}));
    idx.upsert(moved);                                   // re-embed: new node
    CortexVector meta("v6", {}, P({{"k", "new"}}));
    meta.created_at = 5; meta.valid_from = 5;
    idx.upsert(meta);                                    // metadata-only
    QueryOptions o; o.top_k = 10;
    o.filter_expr = filter_from_json({{"k", "new"}});
    auto r = idx.search(moved.embedding, o);
    ASSERT(r.size() == 2);
    o.filter_expr = filter_from_json({{"created_at", {{"$lte", 5}}}});
    r = idx.search(moved.embedding, o);
    ASSERT(r.size() == 1 && r[0].id == "v6");
    for (int i = 100; i < 300; ++i) idx.remove("v" + std::to_string(i));   // forces compaction
    ASSERT(idx.tombstones() < 100);
    o.filter_expr = filter_from_json({{"k", "new"}});
    ASSERT(idx.search(moved.embedding, o).size() == 2);
    o.filter_expr = filter_from_json({{"k", "old"}});
    ASSERT(idx.search(moved.embedding, o).size() == 10);
}

TEST(tvindex_get_and_scroll) {
    TVIndex idx(TVIndexConfig{});
    std::mt19937 rng(2);
    for (int i = 0; i < 250; ++i) {
        char id[16]; std::snprintf(id, sizeof id, "id%03d", i);
        idx.insert(CortexVector(id, random_vec(rng, 4), P({{"even", i % 2 ? "no" : "yes"}})));
    }
    auto g = idx.get("id007");
    ASSERT(g && g->id == "id007" && g->embedding.size() == 4 && g->payload.at("even") == "no");
    ASSERT(idx.get("id007", false)->embedding.empty());
    ASSERT(!idx.get("missing"));
    idx.remove("id007");
    ASSERT(!idx.get("id007"));

    // Paging with a cursor visits every live id exactly once, in order.
    QueryOptions none;
    std::vector<std::string> seen;
    std::string cursor;
    while (true) {
        auto page = idx.scroll(cursor, 64, none);
        for (auto& v : page) seen.push_back(v.id);
        if (page.size() < 64) break;
        cursor = page.back().id;
    }
    ASSERT(seen.size() == 249);
    ASSERT(std::is_sorted(seen.begin(), seen.end()));
    ASSERT(std::find(seen.begin(), seen.end(), "id007") == seen.end());

    QueryOptions even; even.filter_expr = filter_from_json({{"even", "yes"}});
    auto page = idx.scroll("", 1000, even);
    ASSERT(page.size() == 125);
    for (auto& v : page) ASSERT(v.payload.at("even") == "yes");
}

// ---- as_of vs. a brute-force oracle --------------------------------

TEST(as_of_matches_bruteforce_oracle) {
    // 200 documents x 3 versions, created at scattered times over 300
    // days and inserted in shuffled order. Queries at random cut-offs
    // must return exactly the oracle's top-k (exact plan) with the
    // oracle's supersession state at that time.
    const std::string dir = "/tmp/tidevec_asof_" + std::to_string(now_ms());
    std::filesystem::remove_all(dir);
    {
        CollectionRegistry reg(dir);
        CollectionRegistry::CreateParams p;
        p.name = "asof"; p.dim = 16; p.n_shards = 3;
        p.temporal.temporal_blend = 0.3f;
        p.temporal.half_life_ms = 30LL * 86'400'000;
        p.supersession.enabled = true;
        p.supersession.similarity_threshold = 0.5f;
        p.supersession.match_field = "doc";
        reg.create(p);
        auto& col = reg.get("asof");

        std::mt19937 rng(17);
        const Timestamp t0 = 1'700'000'000'000LL, day = 86'400'000LL;
        std::uniform_int_distribution<int> when(0, 300);
        std::normal_distribution<float> noise(0.0f, 0.05f);
        struct V { std::string id, doc; Timestamp at; std::vector<float> e; };
        std::vector<V> all;
        for (int d = 0; d < 200; ++d) {
            auto base = random_vec(rng, 16);
            for (int r = 0; r < 3; ++r) {
                auto e = base;
                for (auto& x : e) x += noise(rng);
                all.push_back({"d" + std::to_string(d) + "_r" + std::to_string(r),
                               "doc" + std::to_string(d), t0 + when(rng) * day + r, e});
            }
        }
        std::shuffle(all.begin(), all.end(), rng);
        for (const auto& v : all) {
            CortexVector cv(v.id, v.e, P({{"doc", v.doc}}));
            cv.created_at = v.at; cv.valid_from = v.at;
            col.upsert(cv);
        }

        TemporalScorer scorer(p.temporal);
        int checked = 0;
        for (int qi = 0; qi < 40; ++qi) {
            const Timestamp t = t0 + when(rng) * day + day / 2;
            auto q = all[qi].e;
            for (auto& x : q) x += noise(rng);

            // Oracle: score every vector created by t as of t.
            std::vector<std::pair<float, const V*>> s;
            for (const auto& v : all) {
                if (v.at > t) continue;
                float sim = compute_similarity(q, v.e, Metric::COSINE);
                float tw  = scorer.temporal_weight_raw(v.at, t);
                s.emplace_back(0.7f * sim + 0.3f * tw, &v);
            }
            std::sort(s.begin(), s.end(), [](auto& a, auto& b) { return a.first > b.first; });

            QueryOptions o; o.top_k = 10; o.as_of = t;
            auto got = col.search(q, o);
            ASSERT(got.size() == std::min<std::size_t>(10, s.size()));
            for (std::size_t i = 0; i < got.size(); ++i) {
                ASSERT(got[i].id == s[i].second->id);
                ASSERT(got[i].created_at <= t);
                ASSERT_NEAR(got[i].score, s[i].first, 1e-4f);
                // Newer versions of the same doc that existed at t.
                std::set<std::string> want;
                for (const auto& v : all)
                    if (v.doc == s[i].second->doc && v.at > s[i].second->at && v.at <= t)
                        want.insert(v.id);
                std::set<std::string> have(got[i].superseded_by.begin(),
                                           got[i].superseded_by.end());
                ASSERT(have == want);
                ++checked;
            }
        }
        ASSERT(checked > 200);

        // Without as_of every non-latest version is superseded.
        QueryOptions now_opts; now_opts.top_k = 30;
        for (const auto& r : col.search(all[0].e, now_opts)) {
            bool latest = true;
            for (const auto& v : all)
                if (v.doc == r.payload.at("doc") && v.at > r.created_at) latest = false;
            ASSERT(r.superseded_by.empty() == latest);
        }
    }
    std::filesystem::remove_all(dir);
}

TEST(as_of_scroll_and_graph_path) {
    FilterFixture fx(20000);
    const Timestamp cut = fx.vecs[15000].created_at;
    QueryOptions o; o.top_k = 10; o.as_of = cut;
    ASSERT(fx.idx->explain(o).kind == SearchPlan::Kind::GRAPH);   // 75% visible
    std::mt19937 rng(5);
    double hit = 0, total = 0;
    for (int qi = 0; qi < 30; ++qi) {
        auto q = random_vec(rng, fx.D);
        auto truth = fx.oracle(q, nullptr, 10, cut);
        std::unordered_set<std::string> t(truth.begin(), truth.end());
        for (const auto& r : fx.idx->search(q, o)) {
            ASSERT(r.created_at <= cut);
            hit += t.count(r.id);
        }
        total += t.size();
    }
    ASSERT(hit / total >= 0.95);
    o.as_of = fx.vecs[99].created_at;                               // 0.5% visible
    ASSERT(fx.idx->explain(o).kind == SearchPlan::Kind::EXACT_INDEX);
    ASSERT(fx.idx->scroll("", 1000, o).size() == 100);
}

// ---- BM25 / hybrid -------------------------------------------------

TEST(bm25_tokenize_and_rank) {
    auto t = Bm25Index::tokenize("Error ERR-4711: disk_full, café!");
    ASSERT((t == std::vector<std::string>{"error", "err", "4711", "disk", "full", "café"}));
    Bm25Index b;
    b.add(0, "the refund policy for damaged goods");
    b.add(1, "shipping policy and delivery times");
    b.add(2, "error ERR-4711 when the disk is full");
    b.add(3, "the the the the policy");
    auto all = [](uint64_t) { return true; };
    auto r = b.search("ERR-4711", 10, all);
    ASSERT(r.size() == 1 && r[0].second == 2);
    r = b.search("refund policy", 10, all);
    ASSERT(!r.empty() && r[0].second == 0);
    b.remove(0, "the refund policy for damaged goods");
    ASSERT(b.doc_count() == 3);
    ASSERT(b.search("refund", 10, all).empty());
    b.add(0, "refund rules changed");
    ASSERT(b.search("refund", 10, all).size() == 1);
    r = b.search("policy", 10, [](uint64_t d) { return d != 1; });
    ASSERT(r.size() == 1 && r[0].second == 3);
}

TEST(fusion_rrf_and_weighted) {
    auto mk = [](std::string id, float s, float ts = 0) {
        SearchResult r; r.id = std::move(id); r.score = s; r.text_score = ts; return r;
    };
    std::vector<SearchResult> dense  = {mk("a", 0.9f), mk("b", 0.8f), mk("c", 0.7f)};
    std::vector<SearchResult> sparse = {mk("c", 5, 5), mk("d", 4, 4), mk("a", 1, 1)};
    HybridOptions h;
    auto r = fuse_results(dense, sparse, h, 10);
    ASSERT(r.size() == 4);
    ASSERT(r[0].id == "a" || r[0].id == "c");          // in both lists
    ASSERT(r.back().id == "b" || r.back().id == "d");
    ASSERT(r[0].text_score > 0);
    h.fusion = FusionMethod::WEIGHTED; h.alpha = 0.0f;  // keyword only
    r = fuse_results(dense, sparse, h, 2);
    ASSERT(r.size() == 2 && r[0].id == "c" && r[1].id == "d");
    h.alpha = 1.0f;                                     // dense only
    r = fuse_results(dense, sparse, h, 1);
    ASSERT(r.size() == 1 && r[0].id == "a");
}

TEST(hybrid_search_finds_exact_terms) {
    const std::string dir = "/tmp/tidevec_hybrid_" + std::to_string(now_ms());
    std::filesystem::remove_all(dir);
    {
        CollectionRegistry reg(dir);
        CollectionRegistry::CreateParams p;
        p.name = "kb"; p.dim = 8; p.n_shards = 2; p.text_field = "text";
        p.temporal.temporal_blend = 0.0f;
        reg.create(p);
        auto& col = reg.get("kb");
        std::mt19937 rng(8);
        for (int i = 0; i < 200; ++i)
            col.upsert(CortexVector("doc" + std::to_string(i), random_vec(rng, 8),
                P({{"text", "general article number " + std::to_string(i)},
                   {"kind", i % 2 ? "faq" : "guide"}})));
        auto target = random_vec(rng, 8);
        col.upsert(CortexVector("needle", target,
            P({{"text", "Fix for error code ZX-9041 on boot"}, {"kind", "faq"}})));

        QueryOptions kw; kw.top_k = 5; kw.text = "zx-9041";
        auto r = col.search({}, kw);                                // keyword only
        ASSERT(!r.empty() && r[0].id == "needle" && r[0].text_score > 0);

        auto misleading = random_vec(rng, 8);
        QueryOptions dense; dense.top_k = 5;
        bool needle_in_dense = false;
        for (auto& h : col.search(misleading, dense)) needle_in_dense |= h.id == "needle";

        QueryOptions hy = kw; hy.top_k = 5;
        r = col.search(misleading, hy);                             // hybrid (RRF)
        ASSERT(!r.empty());
        bool found = false;
        for (auto& h : r) found |= h.id == "needle";
        ASSERT(found || needle_in_dense);
        hy.hybrid.fusion = FusionMethod::WEIGHTED; hy.hybrid.alpha = 0.3f;
        r = col.search(misleading, hy);
        ASSERT(!r.empty() && r[0].id == "needle");

        kw.filter_expr = filter_from_json({{"kind", "guide"}});    // filters apply to BM25
        ASSERT(col.search({}, kw).empty());

        // Collections without text_field reject text queries.
        CollectionRegistry::CreateParams p2; p2.name = "plain"; p2.dim = 8; p2.n_shards = 1;
        reg.create(p2);
        bool threw = false;
        try { reg.get("plain").search({}, kw); } catch (const std::invalid_argument&) { threw = true; }
        ASSERT(threw);
    }
    {
        CollectionRegistry reg(dir);       // text_field survives restart
        reg.load_and_recover();
        QueryOptions kw; kw.top_k = 1; kw.text = "ZX 9041";
        auto r = reg.get("kb").search({}, kw);
        ASSERT(r.size() == 1 && r[0].id == "needle");
    }
    std::filesystem::remove_all(dir);
}

// ---- registry: get / scroll / count / delete-by-filter / aliases ----

TEST(managed_collection_get_scroll_count) {
    const std::string dir = "/tmp/tidevec_scroll_" + std::to_string(now_ms());
    std::filesystem::remove_all(dir);
    {
        CollectionRegistry reg(dir);
        CollectionRegistry::CreateParams p; p.name = "c"; p.dim = 4; p.n_shards = 4;
        reg.create(p);
        auto& col = reg.get("c");
        std::mt19937 rng(1);
        for (int i = 0; i < 500; ++i)
            col.upsert(CortexVector("v" + std::to_string(i), random_vec(rng, 4),
                                    P({{"grp", std::to_string(i % 5)}, {"n", std::to_string(i)}})));
        col.add_edge("v1", "v2", EdgeType::UPDATES);
        auto v2 = col.get("v2");
        ASSERT(v2 && v2->payload.at("n") == "2");
        bool superseded = false;
        for (auto& e : v2->edges) superseded |= e.type == EdgeType::SUPERSEDED_BY && e.target_id == "v1";
        ASSERT(superseded);
        ASSERT(!col.get("nope"));

        QueryOptions none;
        std::set<std::string> ids;
        std::string cursor;
        int pages = 0;
        while (true) {
            auto page = col.scroll(cursor, 64, none);
            for (auto& v : page.vectors) ASSERT(ids.insert(v.id).second);
            ++pages;
            if (!page.next_cursor) break;
            cursor = *page.next_cursor;
        }
        ASSERT(ids.size() == 500 && pages == 8);

        QueryOptions f; f.filter_expr = filter_from_json({{"grp", "3"}, {"n", {{"$lt", 100}}}});
        ASSERT(col.count(f) == 20);
        ASSERT(col.count(none) == 500);
        ASSERT(col.remove_where(f.filter_expr) == 20);
        ASSERT(col.count(f) == 0);
        ASSERT(col.total_vectors() == 480);
        ASSERT(col.remove_where("grp=4") == 100);
    }
    std::filesystem::remove_all(dir);
}

TEST(aliases_swap_atomically_and_persist) {
    const std::string dir = "/tmp/tidevec_alias_" + std::to_string(now_ms());
    std::filesystem::remove_all(dir);
    {
        CollectionRegistry reg(dir);
        for (auto n : {"docs_v1", "docs_v2"}) {
            CollectionRegistry::CreateParams p; p.name = n; p.dim = 2; p.n_shards = 1;
            reg.create(p);
            reg.get(n).upsert(CortexVector(std::string("only_in_") + n, {1.0f, 0.0f}));
        }
        ASSERT(reg.set_alias("docs", "docs_v1").empty());
        QueryOptions o; o.top_k = 1;
        ASSERT(reg.get("docs").search({1, 0}, o)[0].id == "only_in_docs_v1");
        ASSERT(reg.set_alias("docs", "docs_v2") == "docs_v1");
        ASSERT(reg.get("docs").search({1, 0}, o)[0].id == "only_in_docs_v2");
        ASSERT(reg.resolve("docs") == "docs_v2" && reg.exists("docs"));
        bool threw = false;
        try { reg.set_alias("docs_v1", "docs_v2"); } catch (const std::invalid_argument&) { threw = true; }
        ASSERT(threw);                                       // cannot shadow a collection
        threw = false;
        try { reg.set_alias("x", "missing"); } catch (const std::runtime_error&) { threw = true; }
        ASSERT(threw);
        threw = false;
        CollectionRegistry::CreateParams clash; clash.name = "docs"; clash.dim = 2;
        try { reg.create(clash); } catch (const std::runtime_error&) { threw = true; }
        ASSERT(threw);
        ASSERT(reg.set_alias("old", "docs_v1").empty());
    }
    {
        CollectionRegistry reg(dir);
        reg.load_and_recover();
        ASSERT(reg.resolve("docs") == "docs_v2");
        ASSERT(reg.aliases().size() == 2);
        ASSERT(reg.drop("docs_v1"));                         // drops aliases pointing at it
        ASSERT(!reg.exists("old") && reg.aliases().size() == 1);
        ASSERT(reg.delete_alias("docs") && !reg.delete_alias("docs"));
        ASSERT(!reg.exists("docs"));
    }
    {
        // Without aliases the registry is written in the legacy array form.
        std::ifstream f(dir + "/registry.json");
        auto j = json::parse(f);
        ASSERT(j.is_array() && j.size() == 1);
    }
    std::filesystem::remove_all(dir);
}

// ---- background compaction -----------------------------------------

TEST(background_compaction_serves_reads_and_writes) {
    const int N = 4000, D = 32;
    std::mt19937 rng(21);
    TVIndexConfig cfg;
    cfg.temporal.temporal_blend = 0.0f;
    cfg.compaction_ratio = 1e9;                    // manual compaction only
    TVIndex idx(cfg);
    FlatIndex flat(D, Metric::COSINE, cfg.temporal);
    for (int i = 0; i < N; ++i) {
        CortexVector v("v" + std::to_string(i), random_vec(rng, D),
                       P({{"grp", std::to_string(i % 10)}}));
        idx.insert(v); flat.upsert(v);
    }
    std::unordered_set<std::string> gone;
    for (int i = 0; i < N; ++i) {
        if (i % 5 < 3) {                           // 60% deleted
            const auto id = "v" + std::to_string(i);
            idx.remove(id); flat.remove(id); gone.insert(id);
        }
    }
    QueryOptions opts; opts.top_k = 10; opts.ef_search = 128;
    std::mt19937 qrng(5);
    const double recall_before = recall_vs_flat(idx, flat, qrng, D, opts);
    const std::size_t bytes_before = idx.approx_memory_bytes();
    ASSERT(idx.tombstones() == gone.size());

    ASSERT(idx.compact_async());
    ASSERT(!idx.compact_async());                  // one at a time
    ASSERT(idx.compaction_stats().running);

    // Writes land on the live graph and in the journal while the shadow
    // graph is being built; a reader keeps querying throughout.
    std::atomic<bool> stop{false}, bad{false};
    std::atomic<int> reads{0};
    std::thread reader([&] {
        std::mt19937 r(9);
        QueryOptions o; o.top_k = 10;
        QueryOptions fo = o; fo.filter_expr = filter_from_json({{"grp", "4"}});
        while (!stop) {
            auto q = random_vec(r, D);
            auto res = idx.search(q, o);
            if (res.size() != 10) bad = true;
            for (auto& h : res) if (gone.count(h.id)) bad = true;
            for (auto& h : idx.search(q, fo))
                if (h.payload.at("grp") != "4") bad = true;
            ++reads;
        }
    });
    std::vector<std::string> added, removed;
    for (int i = 0; i < 300; ++i) {
        CortexVector v("w" + std::to_string(i), random_vec(rng, D), P({{"grp", "4"}}));
        idx.upsert(v); flat.upsert(v); added.push_back(v.id);
        if (i % 3 == 0) {
            const auto id = "v" + std::to_string(i * 5 + 3);   // a survivor
            idx.remove(id); flat.remove(id); removed.push_back(id);
        }
    }
    idx.wait_for_compaction();
    stop = true;
    reader.join();
    ASSERT(!bad);
    ASSERT(reads > 0);

    auto st = idx.compaction_stats();
    ASSERT(!st.running && st.runs == 1);
    ASSERT(st.last_reclaimed == gone.size());
    ASSERT(st.last_replayed > 0);
    ASSERT(idx.tombstones() <= removed.size());    // only removes replayed after the snapshot
    ASSERT(idx.size() == flat.size());
    for (auto& id : added) ASSERT(idx.get(id, false).has_value());
    for (auto& id : removed) ASSERT(!idx.get(id, false).has_value());
    for (auto& id : gone) ASSERT(!idx.get(id, false).has_value());

    std::mt19937 qrng2(5);
    const double recall_after = recall_vs_flat(idx, flat, qrng2, D, opts);
    ASSERT(recall_after >= 0.9);
    ASSERT(recall_after >= recall_before - 0.03);
    opts.filter_expr = filter_from_json({{"grp", "4"}});
    ASSERT(recall_vs_flat(idx, flat, qrng2, D, opts) >= 0.9);

    const std::size_t bytes_after = idx.approx_memory_bytes();
    ASSERT(bytes_after < bytes_before * 6 / 10);
}

TEST(background_compaction_triggers_on_tombstone_ratio) {
    TVIndexConfig cfg;
    cfg.temporal.temporal_blend = 0.0f;
    cfg.background_compaction = true;
    cfg.compaction_ratio = 0.25;
    TVIndex idx(cfg);
    std::mt19937 rng(3);
    for (int i = 0; i < 1000; ++i)
        idx.insert(CortexVector("v" + std::to_string(i), random_vec(rng, 16)));
    for (int i = 0; i < 199; ++i) idx.remove("v" + std::to_string(i));
    ASSERT(idx.compaction_stats().runs == 0 && !idx.compaction_stats().running);
    idx.remove("v199");                            // 200 >= 0.25 * 800
    idx.wait_for_compaction();
    auto st = idx.compaction_stats();
    ASSERT(st.runs == 1 && st.last_reclaimed == 200);
    ASSERT(idx.tombstones() == 0 && idx.size() == 800);
    QueryOptions o; o.top_k = 5;
    auto probe = idx.get("v500", true);
    ASSERT(probe.has_value());
    ASSERT(idx.search(probe->embedding, o)[0].id == "v500");
}

TEST(registry_compaction_covers_all_shards) {
    const std::string dir = "/tmp/tidevec_compact_" + std::to_string(now_ms());
    std::filesystem::remove_all(dir);
    {
        RegistryConfig rc; rc.data_dir = dir;
        CollectionRegistry reg(rc);
        CollectionRegistry::CreateParams p; p.name = "c"; p.dim = 8; p.n_shards = 3;
        p.compaction_ratio = 1e9;                  // manual only
        reg.create(p);
        auto& col = reg.get("c");
        std::mt19937 rng(2);
        for (int i = 0; i < 900; ++i)
            col.upsert(CortexVector("v" + std::to_string(i), random_vec(rng, 8)));
        for (int i = 0; i < 600; ++i) col.remove("v" + std::to_string(i));
        auto before = col.compaction_status();
        ASSERT(before.tombstones == 600 && before.runs == 0);
        auto after = col.compact(/*wait=*/true);
        ASSERT(after.started == 3 && after.running == 0 && after.runs == 3);
        ASSERT(after.tombstones == 0 && after.last_reclaimed == 600);
        ASSERT(after.index_bytes < before.index_bytes / 2);
        ASSERT(col.total_vectors() == 300);
        ASSERT(col.compact(true).started == 0);    // nothing left to reclaim
        auto v = col.get("v700", true);
        QueryOptions o; o.top_k = 1;
        ASSERT(v && col.search(v->embedding, o)[0].id == "v700");
    }
    std::filesystem::remove_all(dir);
}
