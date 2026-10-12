// ================================================================
// test_cpp_sdk.cpp — C++ SDK (sdk/cpp) against a live tidevec-server
//
// Starts the tidevec-server binary as a child process, plus an httplib
// stub that records request bodies for checks the server cannot show.
// ================================================================

#include <tidevec/tidevec.hpp>
#include "httplib.h"

#include <chrono>
#include <csignal>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <thread>

#include <fcntl.h>
#include <netinet/in.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;
namespace fs = std::filesystem;

#ifndef TIDEVEC_SERVER_BIN
#error "TIDEVEC_SERVER_BIN must point at the tidevec-server binary"
#endif

static int tests_run = 0, tests_passed = 0;
static std::string g_addr;
static const std::string kKey = "sdk_test_key";

#define ASSERT(cond) do { \
    if (!(cond)) { std::cerr << "  FAIL: " #cond " line " << __LINE__ << "\n"; return false; } \
} while(0)

template <typename E, typename F>
static bool throws(F&& f) {
    try { f(); } catch (const E&) { return true; } catch (...) { return false; }
    return false;
}

static void run(const char* name, bool(*fn)()) {
    ++tests_run;
    std::cout << "[TEST] " << name << " ... ";
    std::cout.flush();
    bool ok = false;
    try { ok = fn(); }
    catch (const std::exception& e) { std::cerr << "  EXCEPTION: " << e.what() << "\n"; }
    if (ok) { ++tests_passed; std::cout << "PASS\n"; }
    else    std::cout << "FAIL\n";
}

static int free_port() {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    bind(fd, reinterpret_cast<sockaddr*>(&a), sizeof(a));
    socklen_t len = sizeof(a);
    getsockname(fd, reinterpret_cast<sockaddr*>(&a), &len);
    int port = ntohs(a.sin_port);
    close(fd);
    return port;
}

static int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

static tidevec::TideVec client() { return tidevec::TideVec(g_addr, {.api_key = kKey}); }

// ================================================================
static bool t_tls_flag_rejected() {
    ASSERT(throws<tidevec::TideVecError>([] { tidevec::TideVec(g_addr, {.tls = true}); }));
    ASSERT(throws<tidevec::TideVecError>([] { tidevec::TideVec(g_addr, {.api_key = "a\r\nX: y"}); }));
    return true;
}

static bool t_auth_error() {
    tidevec::TideVec bad(g_addr, {.api_key = "wrong"});
    ASSERT(throws<tidevec::AuthError>([&] { bad.get_stats("anything"); }));
    return true;
}

static bool t_supersession_settings() {
    auto db = client();
    db.create_collection({
        .name = "sdk_sup", .dim = 4, .n_shards = 1,
        .supersession = tidevec::SupersessionConfig{
            .similarity_threshold = 0.9f, .match_field = "doc_id"},
    });
    auto st = db.get_stats("sdk_sup");
    ASSERT(st.dim == 4 && st.metric == "cosine");
    ASSERT(st.supersession.has_value());
    ASSERT(st.supersession->match_field == "doc_id");
    ASSERT(std::abs(st.supersession->similarity_threshold - 0.9f) < 1e-6f);

    const int64_t day = 86'400'000LL, now = now_ms();
    db.upsert("sdk_sup", {
        {.id = "refund_v1", .embedding = {1, 0, 0, 0},
         .payload = {{"doc_id", "refunds"}}, .timestamp_ms = now - 30 * day},
        {.id = "refund_v2", .embedding = {0.97f, 0.2f, 0, 0},
         .payload = {{"doc_id", "refunds"}}, .timestamp_ms = now - day},
    });
    auto resp = db.search("sdk_sup", {1, 0, 0, 0},
                          {.top_k = 2, .temporal_blend = 0.0f, .include_trace = true,
                           .superseded_penalty = 0.5f});
    ASSERT(resp.hits.size() == 2);
    ASSERT(resp.hits[0].id == "refund_v2");
    ASSERT(resp.hits[1].id == "refund_v1");
    ASSERT(resp.hits[1].superseded_by == std::vector<std::string>{"refund_v2"});
    ASSERT(resp.hits[1].created_at == now - 30 * day);   // timestamp_ms was sent
    ASSERT(resp.hits[0].payload.at("doc_id") == "refunds");
    ASSERT(!resp.trace.query_id.empty());

    // A collection without supersession reports none.
    db.create_collection({.name = "sdk_plain", .dim = 2, .n_shards = 1});
    ASSERT(!db.get_stats("sdk_plain").supersession.has_value());
    return true;
}

static bool t_drop_collection_drops() {
    auto db = client();
    db.create_collection({.name = "sdk_drop", .dim = 2, .n_shards = 1});
    db.upsert("sdk_drop", {{.id = "a", .embedding = {1, 0}}});
    ASSERT(db.get_stats("sdk_drop").vector_count == 1);
    db.drop_collection("sdk_drop");
    ASSERT(throws<tidevec::CollectionNotFound>([&] { db.get_stats("sdk_drop"); }));
    ASSERT(throws<tidevec::CollectionNotFound>([&] { db.drop_collection("sdk_drop"); }));
    ASSERT(throws<tidevec::CollectionNotFound>([&] { db.search("sdk_drop", {1, 0}); }));
    // The name is free again and starts empty.
    db.create_collection({.name = "sdk_drop", .dim = 2, .n_shards = 1});
    ASSERT(db.get_stats("sdk_drop").vector_count == 0);
    tidevec::Collection(db, "sdk_drop").drop();
    ASSERT(throws<tidevec::CollectionNotFound>([&] { db.get_stats("sdk_drop"); }));
    return true;
}

static bool t_edges_temporal_delete() {
    auto db = client();
    db.create_collection({.name = "sdk_ops", .dim = 2, .n_shards = 1});
    db.upsert("sdk_ops", {{.id = "a", .embedding = {1, 0}}, {.id = "b", .embedding = {0, 1}}});
    db.add_edges("sdk_ops", {{.target_id = "b", .type = tidevec::EdgeType::CAUSES, .weight = 0.8f}}, "a");
    auto resp = db.search("sdk_ops", {1, 0},
                          {.top_k = 2, .mode = tidevec::QueryMode::CAUSAL_EXPAND});
    ASSERT(!resp.hits.empty() && resp.hits[0].id == "a");
    db.set_temporal("sdk_ops", 3'600'000);
    tidevec::Collection(db, "sdk_ops").set_temporal(7'200'000, 0.1f);
    db.delete_vectors("sdk_ops", {"b"});
    ASSERT(db.get_stats("sdk_ops").vector_count == 1);
    return true;
}

// The server treats an absent temporal_blend as "use the collection's
// default", so the SDK must not send one unless asked to.
static bool t_blend_only_sent_when_set() {
    httplib::Server stub;
    std::mutex mu;
    std::vector<std::string> bodies;
    stub.Post(".*", [&](const httplib::Request& req, httplib::Response& res) {
        { std::lock_guard lock(mu); bodies.push_back(req.body); }
        res.status = 200;
        res.set_content(R"({"status":"ok","data":{"results":[],"count":0}})", "application/json");
    });
    stub.Put(".*", [&](const httplib::Request& req, httplib::Response& res) {
        { std::lock_guard lock(mu); bodies.push_back(req.body); }
        res.set_content(R"({"status":"ok"})", "application/json");
    });
    int port = free_port();
    std::thread th([&] { stub.listen("127.0.0.1", port); });
    stub.wait_until_ready();

    tidevec::TideVec db("127.0.0.1:" + std::to_string(port));
    db.search("c", {1, 0});
    db.search("c", {1, 0}, {.temporal_blend = 0.25f});
    db.create_collection({.name = "c", .dim = 2});
    db.create_collection({.name = "c", .dim = 2, .temporal_blend = 0.5f});
    db.set_temporal("c", 1000);
    stub.stop();
    th.join();

    ASSERT(bodies.size() == 5);
    ASSERT(bodies[0].find("temporal_blend") == std::string::npos);
    ASSERT(bodies[1].find("\"temporal_blend\":0.25") != std::string::npos);
    ASSERT(bodies[2].find("temporal_blend") == std::string::npos);
    ASSERT(bodies[2].find("supersession") == std::string::npos);
    ASSERT(bodies[3].find("\"temporal_blend\":0.5") != std::string::npos);
    ASSERT(bodies[4].find("temporal_blend") == std::string::npos);
    ASSERT(bodies[0].find("\"mode\":\"vector_only\"") != std::string::npos);
    return true;
}

// ================================================================
int main() {
    const int port = free_port();
    g_addr = "127.0.0.1:" + std::to_string(port);
    const fs::path dir = fs::temp_directory_path() / ("tidevec_sdk_" + std::to_string(now_ms()));
    fs::create_directories(dir);

    std::vector<std::string> args = {
        TIDEVEC_SERVER_BIN, "--host", "127.0.0.1", "--port", std::to_string(port),
        "--data-dir", (dir / "data").string(), "--backup-dir", (dir / "backups").string(),
        "--api-key", kKey, "--no-ultra-durable", "--no-segment-store", "--device", "cpu",
        "--quiet"};
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());
    argv.push_back(nullptr);
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    const std::string log = (dir / "server.log").string();
    posix_spawn_file_actions_addopen(&fa, 1, log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    posix_spawn_file_actions_adddup2(&fa, 1, 2);
    pid_t pid = 0;
    if (posix_spawn(&pid, argv[0], &fa, nullptr, argv.data(), environ) != 0) {
        std::cerr << "could not start " << argv[0] << "\n";
        return 1;
    }
    posix_spawn_file_actions_destroy(&fa);

    bool up = false;
    for (int i = 0; i < 200 && !up; ++i) {
        up = tidevec::TideVec(g_addr).ping();
        if (!up) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    std::cout << "\n=== TideVec C++ SDK tests (server pid " << pid << ", " << g_addr << ") ===\n\n";
    if (up) {
        run("tls_flag_rejected",          t_tls_flag_rejected);
        run("auth_error",                 t_auth_error);
        run("supersession_settings",      t_supersession_settings);
        run("drop_collection_drops",      t_drop_collection_drops);
        run("edges_temporal_delete",      t_edges_temporal_delete);
        run("blend_only_sent_when_set",   t_blend_only_sent_when_set);
    } else {
        std::cerr << "server did not start; log: " << log << "\n";
        ++tests_run;
    }

    kill(pid, SIGTERM);
    int status = 0;
    waitpid(pid, &status, 0);

    std::cout << "\n--- Results ---\n";
    std::cout << tests_passed << " / " << tests_run << " tests passed\n";
    bool all_pass = tests_passed == tests_run;
    if (all_pass) std::cout << "ALL TESTS PASSED\n\n";
    else          std::cout << (tests_run - tests_passed) << " TESTS FAILED\n\n";
    if (all_pass) fs::remove_all(dir);
    return all_pass ? 0 : 1;
}
