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

#include <cassert>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <thread>
#include <chrono>

// ---- minimal test harness ----------------------------------------
static int tests_run = 0, tests_passed = 0;

#define TEST(name) \
    void test_##name(); \
    struct _reg_##name { _reg_##name(){ run_test(#name, test_##name); } } _inst_##name; \
    void test_##name()

#define ASSERT(cond) do { \
    if (!(cond)) { \
        std::cerr << "  FAIL: " << #cond << " at line " << __LINE__ << "\n"; \
        return; \
    } \
} while(0)

#define ASSERT_NEAR(a, b, eps) ASSERT(std::abs((a)-(b)) < (eps))

static void run_test(const char* name, void(*fn)()) {
    ++tests_run;
    std::cout << "[TEST] " << name << " ... ";
    fn();
    ++tests_passed;
    std::cout << "PASS\n";
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
