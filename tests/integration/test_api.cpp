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
static bool t_supersession_rest() {
    json create = {{"name","sup_col"},{"dim",4},{"n_shards",2},
                   {"supersession",{{"similarity_threshold",0.9},
                                    {"match_field","doc_id"}}}};
    auto r = g_client->Post("/v1/collections", create.dump(), "application/json");
    ASSERT(r && r->status == 201);

    auto info = parse_body(g_client->Get("/v1/collections/sup_col"));
    ASSERT(info["data"]["supersession"]["enabled"] == true);
    ASSERT(info["data"]["supersession"]["match_field"] == "doc_id");

    long long day = 86'400'000LL;
    long long now = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    json up; up["vectors"] = json::array({
        {{"id","refund_v1"},{"embedding",{1.f,0.f,0.f,0.f}},
         {"created_at", now - 30*day},{"payload",{{"doc_id","refunds"}}}},
        {{"id","refund_v2"},{"embedding",{0.97f,0.2f,0.f,0.f}},
         {"created_at", now - 1*day},{"payload",{{"doc_id","refunds"}}}},
    });
    r = g_client->Post("/v1/collections/sup_col/upsert", up.dump(), "application/json");
    ASSERT(r && r->status == 201);

    json q = {{"vector",{1.f,0.f,0.f,0.f}},{"top_k",2},{"temporal_blend",0.f},
              {"superseded_penalty",0.5}};
    r = g_client->Post("/v1/collections/sup_col/search", q.dump(), "application/json");
    ASSERT(r && r->status == 200);
    auto res = parse_body(r)["data"]["results"];
    ASSERT(res.size() == 2);
    ASSERT(res[0]["id"] == "refund_v2");
    ASSERT(res[1]["id"] == "refund_v1");
    ASSERT(res[1]["superseded_by"] == json::array({"refund_v2"}));
    ASSERT(res[1]["staleness_warning"] == true);
    return true;
}

static bool t_delete_by_filter() {
    json create = {{"name","del_col"},{"dim",2},{"n_shards",2}};
    auto r = g_client->Post("/v1/collections", create.dump(), "application/json");
    ASSERT(r && r->status == 201);
    json up; up["vectors"] = json::array();
    for (int i = 0; i < 10; ++i)
        up["vectors"].push_back({{"id","d" + std::to_string(i)},
                                 {"embedding",{1.f, float(i)}},
                                 {"payload",{{"doc", i < 4 ? "a" : "b"}}}});
    r = g_client->Post("/v1/collections/del_col/upsert", up.dump(), "application/json");
    ASSERT(r && r->status == 201);

    r = g_client->Post("/v1/collections/del_col/delete",
                       json{{"filter","doc=a"}}.dump(), "application/json");
    ASSERT(r && r->status == 200);
    ASSERT(parse_body(r)["data"]["deleted"] == 4);

    r = g_client->Post("/v1/collections/del_col/delete",
                       json{{"ids",{"d4"}},{"filter","doc=a"}}.dump(), "application/json");
    ASSERT(r && r->status == 200);
    ASSERT(parse_body(r)["data"]["deleted"] == 1);

    r = g_client->Post("/v1/collections/del_col/delete", "{}", "application/json");
    ASSERT(r && r->status >= 400);

    auto info = parse_body(g_client->Get("/v1/collections/del_col"));
    ASSERT(info["data"]["n_vectors"] == 5);
    return true;
}

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

static bool t_reembed_url_allowlist() {
    UrlAllowlist open_list;
    ASSERT(open_list.is_allowed("https://embed.example.com/v1"));
    ASSERT(!open_list.is_allowed("http://127.0.0.1:9000/embed"));
    ASSERT(!open_list.is_allowed("http://10.0.0.5/embed"));
    ASSERT(!open_list.is_allowed("http://169.254.169.254/latest"));
    ASSERT(!open_list.is_allowed("file:///etc/passwd"));

    UrlAllowlist strict({"example.com"});
    ASSERT(strict.is_allowed("https://embed.example.com/v1"));
    ASSERT(!strict.is_allowed("https://evil.com/v1"));
    ASSERT(!strict.is_allowed("http://127.0.0.1/embed"));

    UrlAllowlist priv({"127.0.0.1", "10.0.0.5", "169.254.169.254"}, true);
    ASSERT(priv.is_allowed("http://127.0.0.1:9000/embed"));
    ASSERT(priv.is_allowed("http://10.0.0.5/embed"));
    ASSERT(!priv.is_allowed("http://10.0.0.6/embed"));
    ASSERT(!priv.is_allowed("http://169.254.169.254/latest"));
    ASSERT(!priv.is_allowed("http://localhost/embed"));
    return true;
}

static httplib::Result post(const std::string& path, const json& body) {
    return g_client->Post(path, body.dump(), "application/json");
}

static bool t_retrieval_setup() {
    json create = {{"name","ret_col"},{"dim",4},{"n_shards",2},{"text_field","text"}};
    auto r = post("/v1/collections", create);
    ASSERT(r && r->status == 201);
    json up; up["vectors"] = json::array();
    for (int i = 0; i < 50; ++i) {
        std::string text = "plain note " + std::to_string(i);
        if (i == 17) text = "error code TV-4711 on shutdown";
        up["vectors"].push_back({{"id","r" + std::string(i < 10 ? "0" : "") + std::to_string(i)},
                                 {"embedding",{1.f, float(i) / 50.f, 0.f, 0.f}},
                                 {"created_at", 1'700'000'000'000LL + i * 1000LL},
                                 {"payload",{{"lang", i % 2 ? "en" : "de"},
                                             {"year", std::to_string(2000 + i)},
                                             {"text", text}}}});
    }
    r = post("/v1/collections/ret_col/upsert", up);
    ASSERT(r && r->status == 201);
    auto info = parse_body(g_client->Get("/v1/collections/ret_col"));
    ASSERT(info["data"]["text_field"] == "text");
    return true;
}

static bool t_get_vectors() {
    auto r = g_client->Get("/v1/collections/ret_col/vectors/r07");
    ASSERT(r && r->status == 200);
    auto v = parse_body(r)["data"];
    ASSERT(v["id"] == "r07");
    ASSERT(v["payload"]["lang"] == "en");
    ASSERT(v["embedding"].size() == 4);
    ASSERT(v["created_at"] == 1'700'000'007'000LL);

    r = g_client->Get("/v1/collections/ret_col/vectors/nope");
    ASSERT(r && r->status == 404);
    r = g_client->Get("/v1/collections/no_such_col/vectors/r07");
    ASSERT(r && r->status == 404);

    r = post("/v1/collections/ret_col/get", {{"ids",{"r03","missing","r01"}}});
    ASSERT(r && r->status == 200);
    auto d = parse_body(r)["data"];
    ASSERT(d["vectors"].size() == 2);
    ASSERT(d["vectors"][0]["id"] == "r03" && d["vectors"][1]["id"] == "r01");
    ASSERT(!d["vectors"][0].contains("embedding"));
    ASSERT(d["missing"] == json::array({"missing"}));
    return true;
}

static bool t_scroll_and_ids() {
    std::vector<std::string> seen;
    json cursor = nullptr;
    int pages = 0;
    do {
        json body = {{"limit", 16}};
        if (!cursor.is_null()) body["cursor"] = cursor;
        auto r = post("/v1/collections/ret_col/scroll", body);
        ASSERT(r && r->status == 200);
        auto d = parse_body(r)["data"];
        for (const auto& v : d["vectors"]) seen.push_back(v["id"]);
        cursor = d["next_cursor"];
        ASSERT(++pages <= 10);
    } while (!cursor.is_null());
    ASSERT(seen.size() == 50 && pages == 4);
    ASSERT(std::is_sorted(seen.begin(), seen.end()));

    auto r = post("/v1/collections/ret_col/scroll",
                  {{"limit", 100}, {"filter", {{"lang","en"}}}, {"with_vector", true}});
    auto d = parse_body(r)["data"];
    ASSERT(d["count"] == 25 && d["next_cursor"].is_null());
    ASSERT(d["vectors"][0]["embedding"].size() == 4);

    r = g_client->Get("/v1/collections/ret_col/ids?limit=20");
    ASSERT(r && r->status == 200);
    d = parse_body(r)["data"];
    ASSERT(d["ids"].size() == 20 && d["ids"][0] == "r00" && d["next_cursor"] == "r19");
    r = g_client->Get("/v1/collections/ret_col/ids?limit=20&cursor=r19");
    d = parse_body(r)["data"];
    ASSERT(d["ids"][0] == "r20");

    r = post("/v1/collections/ret_col/ids", {{"filter","lang=de"}, {"limit", 5}});
    d = parse_body(r)["data"];
    ASSERT(d["ids"] == json::array({"r00","r02","r04","r06","r08"}));

    r = post("/v1/collections/ret_col/count", json::object());
    ASSERT(parse_body(r)["data"]["count"] == 50);
    r = post("/v1/collections/ret_col/count",
             {{"filter", {{"$or", {{{"lang","de"}}, {{"year",{{"$gte", 2045}}}}}}}}});
    ASSERT(parse_body(r)["data"]["count"] == 25 + 3);   // r45, r47, r49
    return true;
}

static bool t_rich_filter_search() {
    json q = {{"vector",{1.f,0.f,0.f,0.f}},{"top_k",50},{"temporal_blend",0.f},
              {"filter",{{"lang",{{"$in",{"en"}}}},
                         {"year",{{"$gte",2010},{"$lt",2020}}}}}};
    auto r = post("/v1/collections/ret_col/search", q);
    ASSERT(r && r->status == 200);
    auto res = parse_body(r)["data"]["results"];
    ASSERT(res.size() == 5);   // r11 r13 r15 r17 r19
    for (const auto& x : res) ASSERT(x["payload"]["lang"] == "en");

    q["filter"] = {{"created_at",{{"$lte","2023-11-14T22:13:25Z"}}}};   // 1700000005000
    res = parse_body(post("/v1/collections/ret_col/search", q))["data"]["results"];
    ASSERT(res.size() == 6);

    q["filter"] = {{"lang",{{"$bogus",1}}}};
    r = post("/v1/collections/ret_col/search", q);
    ASSERT(r && r->status == 400);
    return true;
}

static bool t_as_of_search() {
    json q = {{"vector",{1.f,0.f,0.f,0.f}},{"top_k",50},{"temporal_blend",0.f},
              {"as_of", 1'700'000'009'000LL}};
    auto res = parse_body(post("/v1/collections/ret_col/search", q))["data"]["results"];
    ASSERT(res.size() == 10);
    for (const auto& x : res) ASSERT(x["created_at"].get<long long>() <= 1'700'000'009'000LL);

    auto r = post("/v1/collections/ret_col/count", {{"as_of","2023-11-14T22:13:29Z"}});
    ASSERT(parse_body(r)["data"]["count"] == 10);
    return true;
}

static bool t_text_and_hybrid_search() {
    json q = {{"text","TV-4711"},{"top_k",3}};
    auto r = post("/v1/collections/ret_col/search", q);
    ASSERT(r && r->status == 200);
    auto res = parse_body(r)["data"]["results"];
    ASSERT(res.size() == 1 && res[0]["id"] == "r17");
    ASSERT(res[0]["text_score"].get<double>() > 0);

    q = {{"vector",{1.f,0.f,0.f,0.f}},{"text","tv-4711 shutdown"},{"top_k",3},
         {"temporal_blend",0.f},{"hybrid",{{"fusion","rrf"}}}};
    res = parse_body(post("/v1/collections/ret_col/search", q))["data"]["results"];
    ASSERT(res.size() == 3);
    bool found = false;
    for (const auto& x : res) found |= x["id"] == "r17";
    ASSERT(found);

    q["hybrid"] = {{"fusion","weighted"},{"alpha",0.2}};
    res = parse_body(post("/v1/collections/ret_col/search", q))["data"]["results"];
    ASSERT(res[0]["id"] == "r17");

    // text search needs a text_field on the collection
    r = post("/v1/collections/del_col/search", {{"text","anything"},{"top_k",3}});
    ASSERT(r && r->status == 400);
    return true;
}

static bool t_delete_by_filter_object() {
    json create = {{"name","del_obj"},{"dim",2},{"n_shards",2}};
    ASSERT(post("/v1/collections", create)->status == 201);
    json up; up["vectors"] = json::array();
    for (int i = 0; i < 10; ++i)
        up["vectors"].push_back({{"id","x" + std::to_string(i)},{"embedding",{1.f, float(i)}},
                                 {"payload",{{"n", std::to_string(i)}}}});
    post("/v1/collections/del_obj/upsert", up);
    auto r = post("/v1/collections/del_obj/delete", {{"filter",{{"n",{{"$gte",6}}}}}});
    ASSERT(r && r->status == 200);
    ASSERT(parse_body(r)["data"]["deleted"] == 4);
    ASSERT(parse_body(post("/v1/collections/del_obj/count", json::object()))["data"]["count"] == 6);
    return true;
}

static bool t_aliases() {
    for (const char* n : {"kb_v1", "kb_v2"}) {
        json create = {{"name", n},{"dim",2},{"n_shards",1}};
        ASSERT(post("/v1/collections", create)->status == 201);
        json v = {{"id", std::string(n) + "_doc"},{"embedding",{1.f,0.f}}};
        ASSERT(post(std::string("/v1/collections/") + n + "/upsert", v)->status == 201);
    }
    auto r = g_client->Put("/v1/aliases/kb", json{{"collection","kb_v1"}}.dump(), "application/json");
    ASSERT(r && r->status == 200);
    ASSERT(parse_body(r)["data"]["previous"].is_null());

    json q = {{"vector",{1.f,0.f}},{"top_k",1}};
    auto res = parse_body(post("/v1/collections/kb/search", q))["data"]["results"];
    ASSERT(res[0]["id"] == "kb_v1_doc");
    auto info = parse_body(g_client->Get("/v1/collections/kb"))["data"];
    ASSERT(info["name"] == "kb_v1" && info["alias"] == "kb");

    r = g_client->Put("/v1/aliases/kb", json{{"collection","kb_v2"}}.dump(), "application/json");
    ASSERT(r && r->status == 200);
    ASSERT(parse_body(r)["data"]["previous"] == "kb_v1");
    res = parse_body(post("/v1/collections/kb/search", q))["data"]["results"];
    ASSERT(res[0]["id"] == "kb_v2_doc");
    ASSERT(parse_body(g_client->Get("/v1/collections/kb/vectors/kb_v2_doc"))["data"]["id"] == "kb_v2_doc");

    auto list = parse_body(g_client->Get("/v1/aliases"))["data"]["aliases"];
    ASSERT(list == json::array({{{"alias","kb"},{"collection","kb_v2"}}}));

    // an alias can't shadow a collection or point at a missing one
    r = g_client->Put("/v1/aliases/kb_v1", json{{"collection","kb_v2"}}.dump(), "application/json");
    ASSERT(r && r->status == 400);
    r = g_client->Put("/v1/aliases/x", json{{"collection","nope"}}.dump(), "application/json");
    ASSERT(r && r->status == 404);

    r = g_client->Delete("/v1/aliases/kb");
    ASSERT(r && r->status == 200);
    r = g_client->Delete("/v1/aliases/kb");
    ASSERT(r && r->status == 404);
    r = g_client->Get("/v1/collections/kb");
    ASSERT(r && r->status == 404);
    return true;
}

static bool t_compact_endpoint() {
    json create = {{"name","cmp"},{"dim",4},{"n_shards",2},{"compaction_ratio",1000.0}};
    ASSERT(post("/v1/collections", create)->status == 201);
    json up; up["vectors"] = json::array();
    for (int i = 0; i < 120; ++i)
        up["vectors"].push_back({{"id","c" + std::to_string(i)},
                                 {"embedding",{1.f, float(i % 7), float(i % 11), float(i)}},
                                 {"payload",{{"grp", std::to_string(i % 4)}}}});
    auto r = post("/v1/collections/cmp/upsert", up);
    ASSERT(r && r->status == 201);
    r = post("/v1/collections/cmp/delete", {{"filter",{{"grp",{{"$in",{"0","1","2"}}}}}}});
    ASSERT(r && parse_body(r)["data"]["deleted"] == 90);

    auto status = parse_body(g_client->Get("/v1/collections/cmp/compact"))["data"];
    ASSERT(status["tombstones"] == 90 && status["running"] == false);
    auto info = parse_body(g_client->Get("/v1/collections/cmp"))["data"];
    ASSERT(info["compaction_ratio"] == 1000.0 && info["compaction"]["tombstones"] == 90);

    ASSERT(g_client->Put("/v1/aliases/cmp_live", json{{"collection","cmp"}}.dump(),
                         "application/json")->status == 200);
    r = post("/v1/collections/cmp_live/compact", {{"wait", true}});
    ASSERT(r && r->status == 200);
    auto d = parse_body(r)["data"];
    ASSERT(d["started"] == 2 && d["running"] == false);
    ASSERT(d["tombstones"] == 0 && d["last_reclaimed"] == 90);
    ASSERT(d["index_bytes"].get<std::size_t>() < status["index_bytes"].get<std::size_t>());

    r = post("/v1/collections/cmp/compact", json::object());      // nothing left
    ASSERT(r && r->status == 200 && parse_body(r)["data"]["started"] == 0);
    ASSERT(parse_body(post("/v1/collections/cmp/count", json::object()))["data"]["count"] == 30);
    json q = {{"vector",{1.f, 3.f, 3.f, 3.f}},{"top_k",5},{"filter",{{"grp","3"}}}};
    ASSERT(parse_body(post("/v1/collections/cmp/search", q))["data"]["results"].size() == 5);
    ASSERT(g_client->Get("/v1/collections/nope/compact")->status >= 400);
    ASSERT(post("/v1/collections/nope/compact", json::object())->status >= 400);
    ASSERT(post("/v1/collections", {{"name","bad_ratio"},{"dim",2},{"compaction_ratio",0}})->status == 400);

    // Drift routes resolve aliases like the other collection routes.
    auto drift = parse_body(g_client->Get("/v1/collections/cmp_live/drift/status"))["data"];
    ASSERT(drift["collection"] == "cmp" && drift["phase"] == "IDLE");
    r = post("/v1/collections/cmp_live/drift/abort", json::object());
    ASSERT(r && r->status == 400 &&
           parse_body(r)["error"].get<std::string>().find("for: cmp") != std::string::npos);
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
    run("supersession_rest",        t_supersession_rest);
    run("delete_by_filter",         t_delete_by_filter);
    run("reembed_url_allowlist",    t_reembed_url_allowlist);
    run("retrieval_setup",          t_retrieval_setup);
    run("get_vectors",              t_get_vectors);
    run("scroll_and_ids",           t_scroll_and_ids);
    run("rich_filter_search",       t_rich_filter_search);
    run("as_of_search",             t_as_of_search);
    run("text_and_hybrid_search",   t_text_and_hybrid_search);
    run("delete_by_filter_object",  t_delete_by_filter_object);
    run("aliases",                  t_aliases);
    run("compact_endpoint",         t_compact_endpoint);

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
