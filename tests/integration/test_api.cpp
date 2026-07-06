// ================================================================
// test_api.cpp — REST API integration tests
// ================================================================

#include <tidevec/api/rest_server.hpp>
#include <thread>
#include <chrono>
#include <iostream>
#include <cassert>
#include <filesystem>

using namespace tidevec;
namespace fs = std::filesystem;

static int tests_run = 0, tests_passed = 0;
static const int TEST_PORT = 16399;
static RestServer* g_server = nullptr;
static httplib::Client* g_client = nullptr;

#define ASSERT(cond) do { \
    if (!(cond)) { std::cerr << "  FAIL: " #cond " line " << __LINE__ << "\n"; return false; } \
} while(0)

static json parse_body(const httplib::Result& r) { return json::parse(r->body); }

static std::string make_upsert_body(const std::string& id, std::vector<float> emb) {
    json j; j["id"] = id; j["embedding"] = emb; return j.dump();
}

static void run(const char* name, bool(*fn)()) {
    ++tests_run;
    std::cout << "[TEST] " << name << " ... ";
    std::cout.flush();
    if (fn()) { ++tests_passed; std::cout << "PASS\n"; }
    else         std::cout << "FAIL\n";
}

// ================================================================
// Individual test functions — return true = pass
// ================================================================
static bool t_health() {
    auto r = g_client->Get("/health");
    ASSERT(r && r->status == 200);
    auto j = parse_body(r);
    ASSERT(j["status"] == "ok");
    ASSERT(j.contains("version"));
    return true;
}

static bool t_info() {
    auto r = g_client->Get("/v1/info");
    ASSERT(r && r->status == 200);
    auto j = parse_body(r);
    // /v1/info returns flat JSON without "data" wrapper
    bool has_data = j.contains("data");
    if (has_data) ASSERT(j["data"]["name"] == "TideVec");
    else ASSERT(j["name"] == "TideVec");
    return true;
}

static bool t_metrics() {
    auto r = g_client->Get("/metrics");
    ASSERT(r && r->status == 200);
    ASSERT(r->body.find("tidevec_requests_total") != std::string::npos);
    return true;
}

static bool t_create_collection() {
    json body = {{"name","test_col"},{"dim",4},{"index_type","flat"},
                 {"metric","cosine"},{"n_shards",1},{"n_replicas",1},{"write_quorum",1}};
    auto r = g_client->Post("/v1/collections", body.dump(), "application/json");
    ASSERT(r && r->status == 201);
    ASSERT(parse_body(r)["data"]["name"] == "test_col");
    return true;
}

static bool t_create_duplicate_fails() {
    json body = {{"name","test_col"},{"dim",4}};
    auto r = g_client->Post("/v1/collections", body.dump(), "application/json");
    ASSERT(r && r->status == 400);
    return true;
}

static bool t_list_collections() {
    auto r = g_client->Get("/v1/collections");
    ASSERT(r && r->status == 200);
    auto j = parse_body(r);
    ASSERT(j["data"].is_array() && j["data"].size() >= 1);
    return true;
}

static bool t_get_collection_stats() {
    auto r = g_client->Get("/v1/collections/test_col");
    ASSERT(r && r->status == 200);
    auto j = parse_body(r);
    ASSERT(j["data"]["name"] == "test_col");
    ASSERT(j["data"].contains("n_vectors"));
    return true;
}

static bool t_get_404() {
    auto r = g_client->Get("/v1/collections/does_not_exist");
    ASSERT(r && r->status == 404);
    return true;
}

static bool t_upsert_single() {
    auto r = g_client->Post("/v1/collections/test_col/upsert",
        make_upsert_body("v1",{1.f,0.f,0.f,0.f}), "application/json");
    ASSERT(r && r->status == 201);
    ASSERT(parse_body(r)["data"]["inserted"] == 1);
    return true;
}

static bool t_upsert_batch() {
    json body; body["vectors"] = json::array();
    body["vectors"].push_back({{"id","v2"},{"embedding",{0.f,1.f,0.f,0.f}}});
    body["vectors"].push_back({{"id","v3"},{"embedding",{0.f,0.f,1.f,0.f}}});
    body["vectors"].push_back({{"id","v4"},{"embedding",{0.f,0.f,0.f,1.f}}});
    auto r = g_client->Post("/v1/collections/test_col/upsert",
                             body.dump(), "application/json");
    ASSERT(r && r->status == 201);
    ASSERT(parse_body(r)["data"]["inserted"] == 3);
    return true;
}

static bool t_upsert_with_payload() {
    json v = {{"id","v_payload"},{"embedding",{0.5f,0.5f,0.f,0.f}},
              {"payload",{{"source","wiki"},{"year","2025"}}},
              {"ttl_seconds",3600}};
    auto r = g_client->Post("/v1/collections/test_col/upsert",
                             v.dump(), "application/json");
    ASSERT(r && r->status == 201);
    return true;
}

static bool t_search_basic() {
    json body = {{"vector",{1.f,0.f,0.f,0.f}},{"top_k",3},{"temporal_blend",0.f}};
    auto r = g_client->Post("/v1/collections/test_col/search",
                             body.dump(), "application/json");
    ASSERT(r && r->status == 200);
    auto j = parse_body(r);
    ASSERT(j["data"]["results"].size() == 3);
    ASSERT(j["data"]["results"][0]["id"] == "v1");
    return true;
}

static bool t_search_with_filter() {
    json body = {{"vector",{0.5f,0.5f,0.f,0.f}},{"top_k",5},
                 {"temporal_blend",0.f},{"filter","source=wiki"}};
    auto r = g_client->Post("/v1/collections/test_col/search",
                             body.dump(), "application/json");
    ASSERT(r && r->status == 200);
    auto j = parse_body(r);
    ASSERT(j["data"]["results"].size() >= 1);
    ASSERT(j["data"]["results"][0]["id"] == "v_payload");
    return true;
}

static bool t_search_with_trace() {
    json body = {{"vector",{1.f,0.f,0.f,0.f}},{"top_k",2},
                 {"temporal_blend",0.f},{"include_trace",true}};
    auto r = g_client->Post("/v1/collections/test_col/search",
                             body.dump(), "application/json");
    ASSERT(r && r->status == 200);
    auto j = parse_body(r);
    ASSERT(j["data"].contains("trace"));
    ASSERT(j["data"]["trace"]["latency_ms"].get<double>() > 0.0);
    return true;
}

static bool t_search_causal_mode() {
    json body = {{"vector",{1.f,0.f,0.f,0.f}},{"top_k",2},
                 {"mode","causal_expand"},{"causal_hops",1},{"temporal_blend",0.f}};
    auto r = g_client->Post("/v1/collections/test_col/search",
                             body.dump(), "application/json");
    ASSERT(r && r->status == 200);
    ASSERT(parse_body(r)["data"]["results"].size() >= 1);
    return true;
}

static bool t_add_edges() {
    json body; body["edges"] = json::array({
        {{"src","v2"},{"tgt","v3"},{"type","CAUSES"},{"weight",0.8f}},
        {{"src","v3"},{"tgt","v4"},{"type","RELATED_TO"},{"weight",0.6f}}
    });
    auto r = g_client->Post("/v1/collections/test_col/edges",
                             body.dump(), "application/json");
    ASSERT(r && r->status == 201);
    ASSERT(parse_body(r)["data"]["added"] == 2);
    return true;
}

static bool t_update_temporal() {
    json body = {{"half_life_ms",86400000},{"temporal_blend",0.2f}};
    auto r = g_client->Put("/v1/collections/test_col/temporal",
                            body.dump(), "application/json");
    ASSERT(r && r->status == 200);
    return true;
}

static bool t_delete_vectors() {
    json body = {{"ids",{"v4"}}};
    auto r = g_client->Post("/v1/collections/test_col/delete",
                             body.dump(), "application/json");
    ASSERT(r && r->status == 200);
    ASSERT(parse_body(r)["data"]["deleted"] == 1);
    return true;
}

static bool t_search_after_delete() {
    json body = {{"vector",{0.f,0.f,0.f,1.f}},{"top_k",5},{"temporal_blend",0.f}};
    auto r = g_client->Post("/v1/collections/test_col/search",
                             body.dump(), "application/json");
    ASSERT(r && r->status == 200);
    for (const auto& res : parse_body(r)["data"]["results"])
        ASSERT(res["id"] != "v4");
    return true;
}

static bool t_bad_body_400() {
    auto r = g_client->Post("/v1/collections/test_col/upsert",
                             "not json", "application/json");
    ASSERT(r && r->status == 400);
    return true;
}

static bool t_drop_collection() {
    json body = {{"name","to_drop"},{"dim",4},{"n_shards",1},
                 {"n_replicas",1},{"write_quorum",1}};
    g_client->Post("/v1/collections", body.dump(), "application/json");
    auto r = g_client->Delete("/v1/collections/to_drop");
    ASSERT(r && r->status == 200);
    auto r2 = g_client->Get("/v1/collections/to_drop");
    ASSERT(r2 && r2->status == 404);
    return true;
}

static bool t_tvindex_collection() {
    json body = {{"name","tv_col"},{"dim",8},{"index_type","tvindex"},
                 {"n_shards",2},{"n_replicas",1},{"write_quorum",1},
                 {"temporal",{{"half_life_ms",3600000},{"temporal_blend",0.3f}}}};
    auto r = g_client->Post("/v1/collections", body.dump(), "application/json");
    ASSERT(r && r->status == 201);

    for (int i = 0; i < 20; ++i) {
        std::vector<float> emb = {float(i)/20.f,0.5f,0.1f,0.2f,0.3f,0.4f,0.5f,0.6f};
        json v = {{"id","tv_"+std::to_string(i)},{"embedding",emb}};
        g_client->Post("/v1/collections/tv_col/upsert", v.dump(), "application/json");
    }

    json sq = {{"vector",{0.99f,0.5f,0.1f,0.2f,0.3f,0.4f,0.5f,0.6f}},
               {"top_k",3},{"temporal_blend",0.3f}};
    auto sr = g_client->Post("/v1/collections/tv_col/search",
                              sq.dump(), "application/json");
    ASSERT(sr && sr->status == 200);
    auto j = parse_body(sr);
    ASSERT(j["data"]["results"].size() == 3);
    ASSERT(j["data"]["results"][0].contains("temporal_score"));
    return true;
}

static bool t_metrics_after_requests() {
    auto r = g_client->Get("/metrics");
    ASSERT(r && r->status == 200);
    ASSERT(r->body.find("tidevec_vectors_inserted_total") != std::string::npos);
    return true;
}

// ================================================================
// MAIN
// ================================================================
int main() {
    std::string data_dir = "/tmp/tidevec_api_" + std::to_string(tidevec::now_ms());

    RestServerConfig scfg;
    scfg.port              = TEST_PORT;
    scfg.threads           = 4;
    scfg.data_dir          = data_dir;
    scfg.log_requests      = false;
    scfg.require_auth      = false;
    scfg.ultra_durable     = false;
    scfg.use_segment_store = false;
    scfg.backup_enabled    = false;
    scfg.device            = "cpu";

    RestServer server(scfg);
    g_server = &server;

    std::thread server_thread([&server]() { server.listen(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    httplib::Client client("localhost", TEST_PORT);
    client.set_connection_timeout(5, 0);
    client.set_read_timeout(10, 0);
    g_client = &client;

    std::cout << "\n=== TideVec API Tests (port " << TEST_PORT << ") ===\n\n";

    run("health_check",             t_health);
    run("info_endpoint",            t_info);
    run("metrics_endpoint",         t_metrics);
    run("create_collection",        t_create_collection);
    run("create_duplicate_fails",   t_create_duplicate_fails);
    run("list_collections",         t_list_collections);
    run("get_collection_stats",     t_get_collection_stats);
    run("get_nonexistent_404",      t_get_404);
    run("upsert_single",            t_upsert_single);
    run("upsert_batch",             t_upsert_batch);
    run("upsert_with_payload",      t_upsert_with_payload);
    run("search_basic",             t_search_basic);
    run("search_with_filter",       t_search_with_filter);
    run("search_with_trace",        t_search_with_trace);
    run("search_causal_mode",       t_search_causal_mode);
    run("add_edges",                t_add_edges);
    run("update_temporal_config",   t_update_temporal);
    run("delete_vectors",           t_delete_vectors);
    run("search_after_delete",      t_search_after_delete);
    run("bad_body_returns_400",     t_bad_body_400);
    run("drop_collection",          t_drop_collection);
    run("tvindex_collection",       t_tvindex_collection);
    run("metrics_after_requests",   t_metrics_after_requests);

    std::cout << "\n--- Results ---\n";
    std::cout << tests_passed << " / " << tests_run << " tests passed\n";
    bool all_pass = (tests_passed == tests_run);
    if (all_pass) std::cout << "ALL TESTS PASSED\n\n";
    else          std::cout << (tests_run - tests_passed) << " TESTS FAILED\n\n";

    server.stop();
    if (server_thread.joinable()) server_thread.join();
    fs::remove_all(data_dir);
    return all_pass ? 0 : 1;
}
