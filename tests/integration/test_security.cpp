// ================================================================
// test_security.cpp — auth, tenancy, backup/restore, OTLP, TLS
//
// Each test starts its own in-process RestServer on a free port with a
// fresh data directory under /tmp.
// ================================================================

#include <tidevec/api/rest_server.hpp>

#include <atomic>
#include <cmath>
#include <map>
#include <mutex>
#include <thread>
#include <chrono>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <regex>
#include <cstdlib>

#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace tidevec;
namespace fs = std::filesystem;

static int tests_run = 0, tests_passed = 0;
static fs::path g_root;
static std::string g_filter;  // argv[1]: run only tests whose name contains it

#define ASSERT(cond) do { \
    if (!(cond)) { std::cerr << "  FAIL: " #cond " line " << __LINE__ << "\n"; return false; } \
} while(0)

static void run(const char* name, bool(*fn)()) {
    if (!g_filter.empty() && std::string(name).find(g_filter) == std::string::npos) return;
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
    a.sin_port = 0;
    bind(fd, reinterpret_cast<sockaddr*>(&a), sizeof(a));
    socklen_t len = sizeof(a);
    getsockname(fd, reinterpret_cast<sockaddr*>(&a), &len);
    int port = ntohs(a.sin_port);
    close(fd);
    return port;
}

static fs::path fresh_dir(const std::string& name) {
    auto d = g_root / name;
    fs::remove_all(d);
    fs::create_directories(d);
    return d;
}

static RestServerConfig base_config(const fs::path& dir) {
    RestServerConfig c;
    c.host              = "127.0.0.1";
    c.port              = free_port();
    c.threads           = 4;
    c.data_dir          = (dir / "data").string();
    c.backup_dir        = (dir / "backups").string();
    c.log_requests      = false;
    c.ultra_durable     = false;
    c.use_segment_store = false;
    c.device            = "cpu";
    c.require_auth      = true;
    return c;
}

// RestServer::stop() is only effective once listen() has started
// serving, so the constructor waits for /health.
struct LiveServer {
    RestServer  server;
    std::thread thread;
    int         port;

    explicit LiveServer(const RestServerConfig& cfg, bool tls_client = false)
        : server(cfg), port(cfg.port) {
        thread = std::thread([this] { server.listen(); });
        for (int i = 0; i < 200; ++i) {
            if (_healthy(tls_client)) return;
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
        }
        throw std::runtime_error("server did not come up");
    }
    ~LiveServer() {
        server.stop();
        if (thread.joinable()) thread.join();
    }

    bool _healthy(bool tls_client) {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
        if (tls_client) {
            httplib::SSLClient c("127.0.0.1", port);
            c.enable_server_certificate_verification(false);
            auto r = c.Get("/health");
            return r && r->status == 200;
        }
#endif
        (void)tls_client;
        httplib::Client c("127.0.0.1", port);
        c.set_connection_timeout(1, 0);
        auto r = c.Get("/health");
        return r && r->status == 200;
    }
};

// Small client bound to one API key.
struct Api {
    httplib::Client cli;
    httplib::Headers h;
    Api(int port, const std::string& key) : cli("127.0.0.1", port) {
        cli.set_connection_timeout(5, 0);
        cli.set_read_timeout(20, 0);
        if (!key.empty()) h.emplace("X-Api-Key", key);
    }
    httplib::Result get(const std::string& p) { return cli.Get(p, h); }
    httplib::Result del(const std::string& p) { return cli.Delete(p, h); }
    httplib::Result post(const std::string& p, const json& b) {
        return cli.Post(p, h, b.dump(), "application/json");
    }
    httplib::Result put(const std::string& p, const json& b) {
        return cli.Put(p, h, b.dump(), "application/json");
    }
    int status_get(const std::string& p) { auto r = get(p); return r ? r->status : -1; }
    int status_post(const std::string& p, const json& b) { auto r = post(p, b); return r ? r->status : -1; }
    int status_put(const std::string& p, const json& b) { auto r = put(p, b); return r ? r->status : -1; }
    int status_del(const std::string& p) { auto r = del(p); return r ? r->status : -1; }
    int create(const std::string& name, int dim = 2) {
        return status_post("/v1/collections", {{"name", name}, {"dim", dim}, {"n_shards", 1}});
    }
    int upsert(const std::string& col, const std::vector<std::string>& ids) {
        json vs = json::array();
        for (const auto& id : ids) vs.push_back({{"id", id}, {"embedding", {1.0f, 0.5f}}});
        return status_post("/v1/collections/" + col + "/upsert", {{"vectors", vs}});
    }
};

static void write_file(const fs::path& p, const std::string& s) {
    std::ofstream f(p, std::ios::trunc);
    f << s;
}

static std::string read_file(const fs::path& p) {
    std::ifstream f(p);
    return std::string(std::istreambuf_iterator<char>(f), {});
}

// ================================================================
// 1. Key rotation: the configured key is authoritative
// ================================================================
static bool t_key_rotation_revokes_old_key() {
    auto dir = fresh_dir("rotation");
    auto cfg = base_config(dir);
    fs::create_directories(cfg.data_dir);
    // tenants.json as written by a pre-0.3.0 server that started with
    // "legacy_key", plus one explicitly configured tenant.
    write_file(fs::path(cfg.data_dir) / "tenants.json", R"([
        {"id":"default","api_key":"legacy_key","name":"Default Tenant"},
        {"id":"acme","api_key":"acme_key","collection_prefixes":["acme_"]}
    ])");

    cfg.api_key = "key_one";
    {
        LiveServer s(cfg);
        ASSERT(Api(s.port, "key_one").status_get("/v1/collections") == 200);
        ASSERT(Api(s.port, "legacy_key").status_get("/v1/collections") == 401);
        // Tenant keys only count in --multi-tenant mode.
        ASSERT(Api(s.port, "acme_key").status_get("/v1/collections") == 401);
        ASSERT(Api(s.port, "").status_get("/v1/collections") == 401);
    }
    auto file = read_file(fs::path(cfg.data_dir) / "tenants.json");
    ASSERT(file.find("legacy_key") == std::string::npos);   // secret removed from disk
    ASSERT(file.find("key_one") == std::string::npos);      // admin key never persisted
    ASSERT(file.find("acme_key") != std::string::npos);     // explicit tenant kept

    cfg.api_key = "key_two";
    cfg.port = free_port();
    {
        LiveServer s(cfg);
        ASSERT(Api(s.port, "key_one").status_get("/v1/collections") == 401);
        ASSERT(Api(s.port, "key_two").status_get("/v1/collections") == 200);
    }

    cfg.multi_tenant = true;
    cfg.port = free_port();
    {
        LiveServer s(cfg);
        ASSERT(Api(s.port, "key_one").status_get("/v1/collections") == 401);
        ASSERT(Api(s.port, "key_two").status_get("/v1/collections") == 200);
        ASSERT(Api(s.port, "acme_key").status_get("/v1/collections") == 200);
    }
    return true;
}

// ================================================================
// 2. Multi-tenant isolation, admin endpoints, per-tenant quotas
// ================================================================
// --no-require-auth wins over a configured key (with a warning), and is
// refused together with --multi-tenant.
static bool t_no_require_auth() {
    auto dir = fresh_dir("noauth");
    auto cfg = base_config(dir);
    cfg.api_key = "k";
    cfg.require_auth = false;
    {
        LiveServer s(cfg);
        ASSERT(Api(s.port, "").status_get("/v1/collections") == 200);
        ASSERT(Api(s.port, "wrong").status_get("/v1/collections") == 200);
        ASSERT(Api(s.port, "").status_get("/v1/admin/backups") == 200);
        auto r = Api(s.port, "").get("/health");
        ASSERT(r && json::parse(r->body)["auth_required"] == false);
    }
    cfg.multi_tenant = true;
    bool refused = false;
    try { RestServer s(cfg); } catch (const std::invalid_argument&) { refused = true; }
    ASSERT(refused);

    cfg.multi_tenant = false;
    cfg.require_auth = true;
    cfg.port = free_port();
    LiveServer s(cfg);
    ASSERT(Api(s.port, "").status_get("/v1/collections") == 401);
    ASSERT(Api(s.port, "k").status_get("/v1/collections") == 200);
    return true;
}

static RestServerConfig tenant_config(const fs::path& dir) {
    auto cfg = base_config(dir);
    fs::create_directories(cfg.data_dir);
    write_file(fs::path(cfg.data_dir) / "tenants.json", R"([
        {"id":"acme","api_key":"acme_key","collection_prefixes":["acme_"],
         "max_collections":2,"max_vectors":3},
        {"id":"beta","api_key":"beta_key","max_collections":1}
    ])");
    cfg.api_key      = "admin_key";
    cfg.multi_tenant = true;
    return cfg;
}

static bool t_tenant_isolation() {
    auto dir = fresh_dir("tenants");
    LiveServer s(tenant_config(dir));
    Api admin(s.port, "admin_key"), acme(s.port, "acme_key"), beta(s.port, "beta_key");

    ASSERT(admin.create("shared") == 201);
    ASSERT(beta.create("beta_a") == 201);       // default prefix is "<id>_"
    ASSERT(acme.create("acme_1") == 201);
    ASSERT(acme.create("beta_z") == 403);
    ASSERT(acme.create("zzz") == 403);
    ASSERT(beta.create("shared2") == 403);

    // Listing shows only the tenant's own collections.
    auto r = acme.get("/v1/collections");
    ASSERT(r && r->status == 200);
    auto list = json::parse(r->body)["data"];
    ASSERT(list.size() == 1 && list[0]["name"] == "acme_1");
    r = admin.get("/v1/collections");
    ASSERT(r && json::parse(r->body)["data"].size() == 3);

    // Every collection route refuses another tenant's (or the admin's) collection.
    for (const std::string col : {"beta_a", "shared"}) {
        const std::string base = "/v1/collections/" + col;
        ASSERT(acme.status_get(base) == 403);
        ASSERT(acme.status_post(base + "/search", {{"vector", {1, 0}}}) == 403);
        ASSERT(acme.upsert(col, {"x"}) == 403);
        ASSERT(acme.status_post(base + "/delete", {{"ids", {"x"}}}) == 403);
        ASSERT(acme.status_post(base + "/edges",
            {{"edges", {{{"src", "a"}, {"tgt", "b"}, {"type", "CAUSES"}}}}}) == 403);
        ASSERT(acme.status_get(base + "/drift/status") == 403);
        ASSERT(acme.status_post(base + "/drift/abort", json::object()) == 403);
        ASSERT(acme.status_post(base + "/count", json::object()) == 403);
        ASSERT(acme.status_post(base + "/get", {{"ids", {"x"}}}) == 403);
        ASSERT(acme.status_get(base + "/vectors/x") == 403);
        ASSERT(acme.status_put(base + "/temporal", {{"half_life_ms", 1000}}) == 403);
        ASSERT(acme.status_del(base) == 403);
    }
    ASSERT(admin.status_get("/v1/collections/beta_a") == 200);   // still there

    // Aliases cannot be used to reach another tenant's collection.
    ASSERT(admin.status_put("/v1/aliases/acme_evil", {{"collection", "beta_a"}}) == 200);
    ASSERT(acme.status_post("/v1/collections/acme_evil/search", {{"vector", {1, 0}}}) == 403);
    ASSERT(acme.status_del("/v1/aliases/acme_evil") == 403);
    ASSERT(acme.status_put("/v1/aliases/acme_al", {{"collection", "acme_1"}}) == 200);
    ASSERT(acme.status_put("/v1/aliases/zzz", {{"collection", "acme_1"}}) == 403);
    ASSERT(acme.status_put("/v1/aliases/acme_al2", {{"collection", "beta_a"}}) == 403);
    r = acme.get("/v1/aliases");
    ASSERT(r && r->status == 200);
    auto aliases = json::parse(r->body)["data"]["aliases"];
    ASSERT(aliases.size() == 1 && aliases[0]["alias"] == "acme_al");

    // Admin-only endpoints.
    ASSERT(acme.status_post("/v1/admin/backup", json::object()) == 403);
    ASSERT(acme.status_get("/v1/admin/backups") == 403);
    ASSERT(acme.status_get("/v1/admin/backups/manifests") == 403);
    ASSERT(acme.status_post("/v1/admin/restore",
        {{"snapshot", "tidevec_1.tar.gz"}, {"confirm", true}}) == 403);
    ASSERT(acme.status_get("/v1/cluster/status") == 403);
    ASSERT(acme.status_post("/v1/admin/cluster/add_node", {{"id", 9}, {"url", "http://x"}}) == 403);
    ASSERT(acme.status_get("/v1/admin/fault") == 403);
    ASSERT(acme.status_get("/v1/admin/anything-new") == 403);
    ASSERT(admin.status_get("/v1/admin/backups") == 200);
    ASSERT(admin.status_get("/v1/cluster/status") == 200);

    ASSERT(Api(s.port, "wrong_key").status_get("/v1/collections") == 401);
    return true;
}

static bool t_tenant_quotas_are_per_tenant() {
    auto dir = fresh_dir("quotas");
    LiveServer s(tenant_config(dir));
    Api admin(s.port, "admin_key"), acme(s.port, "acme_key"), beta(s.port, "beta_key");

    // The server already holds more collections than beta's quota of 1.
    ASSERT(admin.create("shared1") == 201);
    ASSERT(admin.create("shared2") == 201);
    ASSERT(admin.upsert("shared1", {"a", "b", "c", "d", "e"}) == 201);
    ASSERT(beta.create("beta_a") == 201);
    ASSERT(beta.create("beta_b") == 403);

    ASSERT(acme.create("acme_1") == 201);
    ASSERT(acme.create("acme_2") == 201);
    ASSERT(acme.create("acme_3") == 403);

    // acme: max 3 vectors across its own collections; the server holds more.
    ASSERT(acme.upsert("acme_1", {"v1", "v2"}) == 201);
    ASSERT(acme.upsert("acme_2", {"v3"}) == 201);
    ASSERT(acme.upsert("acme_2", {"v4"}) == 403);
    ASSERT(beta.upsert("beta_a", {"w1", "w2"}) == 201);   // unaffected by acme
    return true;
}

// ================================================================
// 3. Error bodies, name validation, rate-limit keying
// ================================================================
static bool t_missing_collection_404_body() {
    auto dir = fresh_dir("notfound");
    auto cfg = base_config(dir);
    cfg.api_key = "k";
    LiveServer s(cfg);
    Api a(s.port, "k");

    auto check = [](const httplib::Result& r, const std::string& want) {
        if (!r || r->status != 404) return false;
        auto j = json::parse(r->body);
        return j["error"].get<std::string>().find(want) != std::string::npos;
    };
    ASSERT(check(a.get("/v1/collections/nope"), "Collection not found: nope"));
    ASSERT(check(a.post("/v1/collections/nope/search", {{"vector", {1, 0}}}), "Collection not found"));
    ASSERT(check(a.post("/v1/collections/nope/upsert", {{"id", "x"}, {"embedding", {1, 0}}}),
                 "Collection not found"));
    ASSERT(check(a.del("/v1/collections/nope"), "Collection not found"));
    ASSERT(check(a.get("/v1/no-such-route"), "Endpoint not found"));
    return true;
}

static bool t_collection_name_validation() {
    auto dir = fresh_dir("names");
    auto cfg = base_config(dir);
    cfg.api_key = "k";
    LiveServer s(cfg);
    Api a(s.port, "k");

    for (const std::string bad : {"../evil", "a/b", "..", ".hidden", "raft", "registry.json",
                                  "tenants.json", "sp ace", "semi;colon", ""})
        ASSERT(a.create(bad) == 400);
    ASSERT(!fs::exists(dir / "evil"));
    ASSERT(a.create("ok-name_1.v2") == 201);
    ASSERT(a.status_put("/v1/aliases/.al", {{"collection", "ok-name_1.v2"}}) == 400);
    ASSERT(a.status_put("/v1/aliases/al.ok", {{"collection", "ok-name_1.v2"}}) == 200);
    return true;
}

static bool t_rate_limit_ignores_unknown_keys() {
    auto dir = fresh_dir("ratelimit");
    auto cfg = base_config(dir);
    cfg.api_key = "k";
    cfg.rate_limit_rps = 1;
    cfg.rate_limit_burst = 2;
    LiveServer s(cfg);

    // Made-up keys share the client address's bucket.
    std::vector<int> codes;
    for (int i = 0; i < 4; ++i)
        codes.push_back(Api(s.port, "fake_" + std::to_string(i))
                            .status_post("/v1/collections/x/count", json::object()));
    ASSERT(codes[0] == 401 && codes[1] == 401);
    ASSERT(codes[2] == 429 && codes[3] == 429);
    // The real key has its own bucket.
    ASSERT(Api(s.port, "k").status_post("/v1/collections/x/count", json::object()) == 404);
    return true;
}

// DriftBridge reembed_url: the allowlist must see the host the HTTP
// client will actually connect to.
static bool t_reembed_ssrf_bypasses_blocked() {
    UrlAllowlist open_list;
    for (const std::string url : {
            "http://x@127.0.0.1/", "http://evil.com%2f@127.0.0.1/", "http://2130706433/",
            "http://0x7f.1/", "http://127.1:80/", "http://0177.0.0.1/", "http://[::1]/",
            "http://[::ffff:127.0.0.1]:80/", "http://LOCALHOST/", "http://localhost./",
            "http://foo.localhost/", "http://169.254.169.254.:80/", "http://0/",
            "http://100.64.0.1/", "http://172.20.0.1/", "http://127.0.0.1:80:90/",
            "http://evil.com?@127.0.0.1/", "http://evil.com\\@127.0.0.1/", "ftp://example.com/",
            "http:///path", "http://224.0.0.1/"})
        if (open_list.is_allowed(url)) {
            std::cerr << "  allowed: " << url << "\n";
            return false;
        }
    ASSERT(open_list.is_allowed("http://172.2.3.4/embed"));          // public, not 172.16/12
    ASSERT(open_list.is_allowed("https://EMBED.example.com:8443/v1"));

    UrlAllowlist priv({"Embed.Internal", "10.0.0.5"}, true);
    ASSERT(priv.is_allowed("http://embed.internal/x"));
    ASSERT(priv.is_allowed("http://10.0.0.5:9000/x"));
    ASSERT(!priv.is_allowed("http://0x0a000005/x"));   // same address, not the listed host
    ASSERT(!priv.is_allowed("http://169.254.169.254/x"));
    return true;
}

// A host name from /etc/hosts that resolves only to loopback but that the
// name-based allowlist check accepts, e.g. the machine's own host name
// (127.0.1.1 on Debian/Ubuntu) or ip6-localhost. "" if there is none.
static std::string loopback_alias() {
    char hn[256] = {};
    ::gethostname(hn, sizeof(hn) - 1);
    for (const std::string& name : {std::string(hn), std::string("ip6-localhost"),
                                   std::string("ip6-loopback"),
                                   std::string("localhost.localdomain")}) {
        if (name.empty() || !UrlAllowlist().is_allowed("http://" + name + "/")) continue;
        try {
            auto u = parse_http_url("http://" + name + "/");
            pin_http_url(u, true);
            in6_addr a6{};
            in_addr a4{};
            if ((inet_pton(AF_INET, u.pinned_ip.c_str(), &a4) == 1 &&
                 (ntohl(a4.s_addr) >> 24) == 127) ||
                (inet_pton(AF_INET6, u.pinned_ip.c_str(), &a6) == 1 &&
                 IN6_IS_ADDR_LOOPBACK(&a6))) {
                bool private_only = false;
                try { pin_http_url(u, false); } catch (const ReembedUrlBlocked&) { private_only = true; }
                if (private_only) return name;
            }
        } catch (const std::exception&) {}
    }
    return {};
}

// Re-embedding endpoint stub bound to one address.
struct EmbedStub {
    httplib::Server svr;
    std::thread thread;
    int port;
    std::mutex mu;
    std::vector<std::string> hosts;  // Host header of each request

    EmbedStub(const std::string& ip, int p) : port(p) {
        svr.Post("/embed", [this](const httplib::Request& req, httplib::Response& res) {
            std::lock_guard lock(mu);
            hosts.push_back(req.get_header_value("Host"));
            res.set_content(R"({"embedding":[1.0,0.0]})", "application/json");
        });
        thread = std::thread([this, ip] { svr.listen(ip, port); });
        svr.wait_until_ready();
    }
    ~EmbedStub() { svr.stop(); if (thread.joinable()) thread.join(); }
    std::size_t hits() { std::lock_guard lock(mu); return hosts.size(); }
};

// Resolve-and-pin: a public-looking host name that resolves to a private
// address is refused, and allowed callbacks go to the pinned address.
static bool t_reembed_dns_pinning() {
    auto v6 = [](const char* s) {
        in6_addr a{};
        inet_pton(AF_INET6, s, &a);
        return classify_ipv6(a.s6_addr);
    };
    ASSERT(classify_ipv4(0x7F000001u) == AddressClass::Private);
    ASSERT(classify_ipv4(0xA9FEA9FEu) == AddressClass::Forbidden);
    ASSERT(classify_ipv4(0x08080808u) == AddressClass::Public);
    ASSERT(v6("::1") == AddressClass::Private);
    ASSERT(v6("::") == AddressClass::Forbidden);
    ASSERT(v6("fe80::1") == AddressClass::Forbidden);
    ASSERT(v6("fd00:ec2::254") == AddressClass::Forbidden);
    ASSERT(v6("fd12::1") == AddressClass::Private);
    ASSERT(v6("::ffff:10.0.0.1") == AddressClass::Private);
    ASSERT(v6("::ffff:169.254.169.254") == AddressClass::Forbidden);
    ASSERT(v6("64:ff9b::7f00:1") == AddressClass::Private);
    ASSERT(v6("2002:7f00:1::") == AddressClass::Private);
    ASSERT(v6("2606:4700::1111") == AddressClass::Public);

    auto lh = parse_http_url("http://localhost:1/");
    bool blocked = false;
    try { pin_http_url(lh, false); } catch (const ReembedUrlBlocked&) { blocked = true; }
    ASSERT(blocked);

    const std::string alias = loopback_alias();
    if (alias.empty()) { std::cout << "(no loopback host alias; partial) "; return true; }

    const int port = free_port();
    const std::string url = "http://" + alias + ":" + std::to_string(port) + "/embed";
    auto pinned = parse_http_url(url);
    blocked = false;
    try { pin_http_url(pinned, false); } catch (const ReembedUrlBlocked&) { blocked = true; }
    ASSERT(blocked);
    pin_http_url(pinned, true);

    EmbedStub stub(pinned.pinned_ip, port);
    std::string out;
    // Unpinned URLs are pinned with private addresses refused.
    ASSERT(!http_post_json(parse_http_url(url), "{}", out, 2));
    ASSERT(stub.hits() == 0);
    ASSERT(http_post_json(pinned, "{}", out, 2));
    ASSERT(out.find("embedding") != std::string::npos);
    std::lock_guard lock(stub.mu);
    ASSERT(stub.hosts == std::vector<std::string>{alias + ":" + std::to_string(port)});
    return true;
}

static bool t_reembed_dns_live() {
    const std::string alias = loopback_alias();
    if (alias.empty()) { std::cout << "(skipped: no loopback host alias) "; return true; }
    const int port = free_port();
    const std::string url = "http://" + alias + ":" + std::to_string(port) + "/embed";
    auto probe = parse_http_url(url);
    pin_http_url(probe, true);
    EmbedStub stub(probe.pinned_ip, port);

    auto start = [&](std::vector<std::string> allow, bool allow_private) {
        auto dir = fresh_dir("dns_live");
        auto cfg = base_config(dir);
        cfg.api_key = "k";
        cfg.reembed_allowed_hosts = std::move(allow);
        cfg.reembed_allow_private = allow_private;
        LiveServer s(cfg);
        Api a(s.port, "k");
        if (a.create("docs") != 201 || a.upsert("docs", {"v1", "v2"}) != 201) return -1;
        int st = a.status_post("/v1/collections/docs/drift/start", {{"reembed_url", url}});
        // The migration thread calls back into the server: let it finish.
        for (int i = 0; st == 200 && i < 400; ++i) {
            auto r = a.get("/v1/collections/docs/drift/status");
            if (r && json::parse(r->body)["data"].value("phase", "") == "COMPLETE") break;
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
        }
        return st;
    };
    ASSERT(start({}, false) == 403);
    ASSERT(start({alias}, false) == 403);
    ASSERT(start({alias}, true) == 200);
    std::lock_guard lock(stub.mu);
    ASSERT(stub.hosts.size() == 2);
    for (const auto& h : stub.hosts) ASSERT(h == alias + ":" + std::to_string(port));
    return true;
}

// ================================================================
// 4. Backups: no shell, validated names, restore into a fixed dir
// ================================================================
static bool t_snapshot_names_validated() {
    for (const std::string bad : {"../x.tar.gz", "a/b.tar.gz", "x.tar.gz; rm -rf /",
                                  "$(id).tar.gz", ".hidden.tar.gz", "x..tar.gz",
                                  "x.tar", "", "tidevec_1.tar.gz\n"})
        ASSERT(!is_valid_snapshot_name(bad));
    ASSERT(is_valid_snapshot_name("tidevec_1791755538.tar.gz"));
    ASSERT(is_valid_snapshot_name("tidevec_1791755538_2.tar.gz"));

    auto dir = fresh_dir("snapnames");
    auto cfg = base_config(dir);
    cfg.api_key = "k";
    LiveServer s(cfg);
    Api a(s.port, "k");
    for (const std::string bad : {"../../etc/passwd", "a/b.tar.gz", "x.tar.gz; touch PWNED",
                                  "$(touch PWNED).tar.gz", ""})
        ASSERT(a.status_post("/v1/admin/restore", {{"snapshot", bad}, {"confirm", true}}) == 400);
    ASSERT(a.status_post("/v1/admin/restore",
        {{"snapshot", "tidevec_1.tar.gz"}, {"confirm", true}}) == 404);
    ASSERT(!fs::exists("PWNED"));
    return true;
}

// Paths that a shell would interpret must be passed through untouched.
static bool t_backup_without_shell() {
    auto dir = fresh_dir("noshell");
    const fs::path data = dir / "d a't $(touch PWNED) `touch PWNED2`";
    fs::create_directories(data / "col");
    write_file(data / "col" / "file.bin", "payload");
    write_file(data / "registry.json", "{}");

    BackupConfig bc;
    bc.data_dir   = data.string();
    bc.backup_dir = (dir / "back ups").string();
    BackupManager bm(bc);
    auto snap = bm.snapshot_now();
    ASSERT(is_valid_snapshot_name(snap));
    ASSERT(bm.list_snapshots() == std::vector<std::string>{snap});

    write_file(data / "col" / "file.bin", "changed");
    write_file(data / "new.txt", "new");
    bm.restore_snapshot(snap);
    ASSERT(read_file(data / "col" / "file.bin") == "payload");
    ASSERT(!fs::exists(data / "new.txt"));
    ASSERT(!fs::exists("PWNED") && !fs::exists("PWNED2"));
    ASSERT(!fs::exists(dir / "PWNED") && !fs::exists(data / "PWNED"));
    // No staging or pre-restore leftovers.
    for (const auto& e : fs::directory_iterator(data))
        ASSERT(!e.path().filename().string().starts_with(".tidevec_"));
    return true;
}

// The data directory is a mount point in Docker: it can be neither
// renamed nor replaced. A read-only parent reproduces that.
static bool t_restore_into_fixed_data_dir() {
    if (geteuid() == 0) { std::cout << "(skipped as root) "; return true; }
    auto dir = fresh_dir("restore");
    const fs::path parent = dir / "mnt";
    fs::create_directories(parent);
    auto cfg = base_config(dir);
    cfg.data_dir = (parent / "data").string();
    cfg.api_key = "k";

    std::string snap;
    {
        LiveServer s(cfg);
        Api a(s.port, "k");
        ASSERT(a.create("docs") == 201);
        ASSERT(a.upsert("docs", {"v1"}) == 201);
        auto r = a.post("/v1/admin/backup", json::object());
        ASSERT(r && r->status == 200);
        snap = json::parse(r->body)["data"]["snapshot"];
        r = a.get("/v1/admin/backups");
        ASSERT(r && json::parse(r->body)["data"]["snapshots"] == json::array({snap}));

        ASSERT(a.upsert("docs", {"v2"}) == 201);
        ASSERT(a.create("later") == 201);

        ::chmod(parent.c_str(), 0555);
        r = a.post("/v1/admin/restore", {{"snapshot", snap}, {"confirm", true}});
        ::chmod(parent.c_str(), 0755);
        ASSERT(r && r->status == 200);
        ASSERT(json::parse(r->body)["data"]["restart_required"] == false);

        // Reloaded in place.
        ASSERT(a.status_get("/v1/collections/later") == 404);
        r = a.post("/v1/collections/docs/get", {{"ids", {"v1", "v2"}}});
        ASSERT(r && r->status == 200);
        auto got = json::parse(r->body)["data"];
        ASSERT(got["vectors"].size() == 1 && got["missing"] == json::array({"v2"}));
        ASSERT(a.upsert("docs", {"v3"}) == 201);

        std::vector<std::string> siblings;
        for (const auto& e : fs::directory_iterator(parent))
            siblings.push_back(e.path().filename().string());
        ASSERT(siblings == std::vector<std::string>{"data"});
    }
    // The restored state (plus the write after it) survives a restart.
    cfg.port = free_port();
    LiveServer s(cfg);
    Api a(s.port, "k");
    auto r = a.post("/v1/collections/docs/get", {{"ids", {"v1", "v2", "v3"}}});
    ASSERT(r && r->status == 200);
    ASSERT(json::parse(r->body)["data"]["missing"] == json::array({"v2"}));
    return true;
}

// Searches, upserts and reads keep running while the collection is
// dropped, re-created and restored: requests in flight must keep their
// collection alive, and the server must neither crash nor fail them with
// anything but "Collection not found".
static bool t_drop_and_restore_under_load() {
    auto dir = fresh_dir("lifetimes");
    auto cfg = base_config(dir);
    cfg.api_key          = "k";
    cfg.threads          = 16;
    cfg.rate_limit_rps   = 1'000'000;
    cfg.rate_limit_burst = 1'000'000;
    LiveServer s(cfg);
    Api admin(s.port, "k");

    constexpr int dim = 16;
    auto vec = [](int i) {
        json e = json::array();
        for (int d = 0; d < dim; ++d) e.push_back(std::sin(i * 0.7 + d));
        return e;
    };
    auto fill = [&](Api& a, const std::string& prefix, int from, int n) {
        json vs = json::array();
        for (int i = from; i < from + n; ++i)
            vs.push_back({{"id", prefix + std::to_string(i)}, {"embedding", vec(i)}});
        return a.status_post("/v1/collections/docs/upsert", {{"vectors", vs}});
    };
    auto create = [&] {
        return admin.status_post("/v1/collections",
            {{"name", "docs"}, {"dim", dim}, {"n_shards", 2}});
    };
    ASSERT(create() == 201);
    ASSERT(fill(admin, "v", 0, 500) == 201);
    auto r = admin.post("/v1/admin/backup", json::object());
    ASSERT(r && r->status == 200);
    const std::string snap = json::parse(r->body)["data"]["snapshot"];

    std::atomic<bool> stop{false};
    std::atomic<int> requests{0}, conn_failures{0};
    std::mutex mu;
    std::map<int, std::string> unexpected;  // status -> body
    std::vector<std::thread> workers;
    for (int t = 0; t < 6; ++t) {
        workers.emplace_back([&, t] {
            Api a(s.port, "k");
            for (int i = 0; !stop; ++i) {
                httplib::Result res;
                switch (i % 4) {
                case 0: res = a.post("/v1/collections/docs/search",
                                     {{"vector", vec(i)}, {"top_k", 5}}); break;
                case 1: {
                    json vs = json::array();
                    for (int k = 0; k < 20; ++k)
                        vs.push_back({{"id", "w" + std::to_string(t) + "_" + std::to_string(k)},
                                      {"embedding", vec(i + k)}});
                    res = a.post("/v1/collections/docs/upsert", {{"vectors", vs}});
                    break;
                }
                case 2: res = a.post("/v1/collections/docs/get", {{"ids", {"v1", "v2"}}}); break;
                default: res = a.post("/v1/collections/docs/count", json::object()); break;
                }
                ++requests;
                if (!res) { ++conn_failures; continue; }
                bool not_found = res->status == 404 &&
                    res->body.find("Collection not found") != std::string::npos;
                if (res->status != 200 && res->status != 201 && !not_found) {
                    std::lock_guard lock(mu);
                    unexpected.emplace(res->status, res->body);
                }
            }
        });
    }

    bool ok = true;
    for (int round = 0; round < 6 && ok; ++round) {
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
        ok = admin.status_del("/v1/collections/docs") == 200 && create() == 201 &&
             fill(admin, "v", 0, 100) == 201;
    }
    for (int round = 0; round < 3 && ok; ++round) {
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
        ok = admin.status_post("/v1/admin/restore", {{"snapshot", snap}, {"confirm", true}}) == 200;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
    stop = true;
    for (auto& w : workers) w.join();

    for (const auto& [st, body] : unexpected)
        std::cerr << "  unexpected " << st << ": " << body.substr(0, 200) << "\n";
    ASSERT(ok);
    ASSERT(requests > 100);
    ASSERT(conn_failures == 0);
    ASSERT(unexpected.empty());
    ASSERT(admin.status_get("/health") == 200);
    // The last restore brought back all 500 snapshot vectors.
    r = admin.post("/v1/collections/docs/get", {{"ids", {"v0", "v250", "v499"}}});
    ASSERT(r && r->status == 200);
    ASSERT(json::parse(r->body)["data"]["missing"].empty());
    return true;
}

// ================================================================
// 5. OTLP export against a local collector stub
// ================================================================
struct CollectorStub {
    httplib::Server svr;
    std::thread thread;
    int port = free_port();
    std::mutex mu;
    std::vector<std::pair<std::string, std::string>> received;  // path, body

    CollectorStub() {
        svr.Post(".*", [this](const httplib::Request& req, httplib::Response& res) {
            std::lock_guard lock(mu);
            received.emplace_back(req.path, req.body);
            res.set_content("{}", "application/json");
        });
        thread = std::thread([this] { svr.listen("127.0.0.1", port); });
        svr.wait_until_ready();
    }
    ~CollectorStub() { svr.stop(); thread.join(); }

    std::size_t count() { std::lock_guard lock(mu); return received.size(); }
    bool wait_for(std::size_t n) {
        for (int i = 0; i < 200 && count() < n; ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
        return count() >= n;
    }
};

static bool valid_span(const json& payload, const std::string& service, const std::string& collection) {
    static const std::regex trace_re("^[0-9a-f]{32}$"), span_re("^[0-9a-f]{16}$");
    const auto& rs = payload.at("resourceSpans").at(0);
    bool svc = false;
    for (const auto& a : rs.at("resource").at("attributes"))
        if (a["key"] == "service.name" && a["value"]["stringValue"] == service) svc = true;
    const auto& span = rs.at("scopeSpans").at(0).at("spans").at(0);
    auto tid = span.at("traceId").get<std::string>();
    auto sid = span.at("spanId").get<std::string>();
    auto start = std::stoll(span.at("startTimeUnixNano").get<std::string>());
    auto end   = std::stoll(span.at("endTimeUnixNano").get<std::string>());
    auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    bool col = false;
    for (const auto& a : span.at("attributes"))
        if (a["key"] == "tidevec.collection" && a["value"]["stringValue"] == collection) col = true;
    return svc && col && std::regex_match(tid, trace_re) && std::regex_match(sid, span_re) &&
           span.at("name") == "tidevec.search" && start <= end && end <= now &&
           now - end < 60'000'000'000LL;
}

static bool t_otlp_export() {
    CollectorStub stub;
    RetrievalTrace t;
    t.query_id = "q_42";
    t.strategy = "FLAT";
    t.collection_name = "docs";
    t.latency_ms = 2.5;

    {
        OtelExporter ex(OtelConfig{"http://127.0.0.1:" + std::to_string(stub.port) + "/v1/traces",
                                   "svc-a", false});
        ASSERT(ex.enabled());
        ex.export_trace(t);
        ex.export_trace(t);
        ASSERT(ex.flush());
        ASSERT(ex.exported() == 2 && ex.failed() == 0);
    }
    ASSERT(stub.count() == 2);
    {
        std::lock_guard lock(stub.mu);
        ASSERT(stub.received[0].first == "/v1/traces");
        auto p0 = json::parse(stub.received[0].second), p1 = json::parse(stub.received[1].second);
        ASSERT(valid_span(p0, "svc-a", "docs"));
        ASSERT(p0["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["traceId"] !=
               p1["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["traceId"]);
    }

    // Endpoint without a path defaults to /v1/traces.
    {
        OtelExporter ex(OtelConfig{"http://127.0.0.1:" + std::to_string(stub.port), "svc-b", false});
        ex.export_trace(t);
        ASSERT(ex.flush() && ex.exported() == 1);
    }
    {
        std::lock_guard lock(stub.mu);
        ASSERT(stub.received.back().first == "/v1/traces");
    }

    // An unreachable collector costs the caller nothing and is counted.
    {
        OtelExporter ex(OtelConfig{"http://127.0.0.1:" + std::to_string(free_port()), "svc", false});
        auto t0 = std::chrono::steady_clock::now();
        ex.export_trace(t);
        ASSERT(std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(100));
        ASSERT(ex.flush() && ex.failed() == 1 && ex.exported() == 0);
    }

    // End to end: a traced search on a server with --otel-endpoint.
    auto dir = fresh_dir("otel");
    auto cfg = base_config(dir);
    cfg.api_key = "k";
    cfg.otel_endpoint = "http://127.0.0.1:" + std::to_string(stub.port) + "/v1/traces";
    LiveServer s(cfg);
    Api a(s.port, "k");
    ASSERT(a.create("traced") == 201);
    ASSERT(a.upsert("traced", {"v1"}) == 201);
    std::size_t before = stub.count();
    ASSERT(a.status_post("/v1/collections/traced/search",
                         {{"vector", {1, 0}}, {"include_trace", true}}) == 200);
    ASSERT(stub.wait_for(before + 1));
    std::lock_guard lock(stub.mu);
    ASSERT(valid_span(json::parse(stub.received.back().second), "tidevec", "traced"));
    return true;
}

// ================================================================
// 6. TLS
// ================================================================
static bool t_tls_flags_validated() {
    auto dir = fresh_dir("tlsflags");
    auto cfg = base_config(dir);
    cfg.tls_cert = (dir / "cert.pem").string();   // key missing
    bool threw = false;
    try { RestServer s(cfg); } catch (const std::invalid_argument&) { threw = true; }
    ASSERT(threw);
#ifndef CPPHTTPLIB_OPENSSL_SUPPORT
    // Without TLS support, asking for TLS must fail instead of serving HTTP.
    cfg.tls_key = (dir / "key.pem").string();
    threw = false;
    try { RestServer s(cfg); } catch (const std::invalid_argument&) { threw = true; }
    ASSERT(threw);
#endif
    return true;
}

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
static bool t_tls_smoke() {
    if (std::system("openssl version > /dev/null 2>&1") != 0) {
        std::cout << "(skipped: no openssl CLI) ";
        return true;
    }
    auto dir = fresh_dir("tls");
    auto cert = (dir / "cert.pem").string(), key = (dir / "key.pem").string();
    std::string cmd = "openssl req -x509 -newkey rsa:2048 -nodes -days 1 -subj /CN=localhost"
                      " -addext subjectAltName=DNS:localhost,IP:127.0.0.1"
                      " -keyout " + key + " -out " + cert + " > /dev/null 2>&1";
    ASSERT(std::system(cmd.c_str()) == 0);

    auto cfg = base_config(dir);
    cfg.api_key = "k";
    cfg.tls_cert = cert;
    cfg.tls_key = key;
    LiveServer s(cfg, /*tls_client=*/true);

    httplib::SSLClient c("127.0.0.1", s.port);
    c.set_ca_cert_path(cert.c_str());
    c.enable_server_certificate_verification(true);
    auto r = c.Get("/health");
    ASSERT(r && r->status == 200);
    r = c.Get("/v1/collections", {{"X-Api-Key", "k"}});
    ASSERT(r && r->status == 200);
    r = c.Get("/v1/collections");
    ASSERT(r && r->status == 401);

    // Verification is real: a client that does not trust the cert fails.
    httplib::SSLClient untrusted("127.0.0.1", s.port);
    untrusted.set_ca_cert_path((dir / "missing.pem").string().c_str());
    untrusted.enable_server_certificate_verification(true);
    ASSERT(!untrusted.Get("/health"));

    // Plain HTTP to the TLS port gets no HTTP answer.
    httplib::Client plain("127.0.0.1", s.port);
    plain.set_read_timeout(2, 0);
    auto pr = plain.Get("/health");
    ASSERT(!pr || pr->status != 200);
    return true;
}
#endif

// ================================================================
// MAIN
// ================================================================
int main(int argc, char** argv) {
    if (argc > 1) g_filter = argv[1];
    g_root = fs::temp_directory_path() / ("tidevec_sec_" + std::to_string(tidevec::now_ms()));
    fs::create_directories(g_root);
    std::cout << "\n=== TideVec security tests (" << g_root.string() << ") ===\n\n";

    run("key_rotation_revokes_old_key",   t_key_rotation_revokes_old_key);
    run("no_require_auth",                t_no_require_auth);
    run("tenant_isolation",               t_tenant_isolation);
    run("tenant_quotas_are_per_tenant",   t_tenant_quotas_are_per_tenant);
    run("missing_collection_404_body",    t_missing_collection_404_body);
    run("collection_name_validation",     t_collection_name_validation);
    run("rate_limit_ignores_unknown_keys", t_rate_limit_ignores_unknown_keys);
    run("reembed_ssrf_bypasses_blocked",  t_reembed_ssrf_bypasses_blocked);
    run("reembed_dns_pinning",            t_reembed_dns_pinning);
    run("reembed_dns_live",               t_reembed_dns_live);
    run("snapshot_names_validated",       t_snapshot_names_validated);
    run("backup_without_shell",           t_backup_without_shell);
    run("restore_into_fixed_data_dir",    t_restore_into_fixed_data_dir);
    run("drop_and_restore_under_load",    t_drop_and_restore_under_load);
    run("otlp_export",                    t_otlp_export);
    run("tls_flags_validated",            t_tls_flags_validated);
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    run("tls_smoke",                      t_tls_smoke);
#endif

    std::cout << "\n--- Results ---\n";
    std::cout << tests_passed << " / " << tests_run << " tests passed\n";
    bool all_pass = tests_passed == tests_run;
    if (all_pass) std::cout << "ALL TESTS PASSED\n\n";
    else          std::cout << (tests_run - tests_passed) << " TESTS FAILED\n\n";
    std::error_code ec;
    fs::permissions(g_root, fs::perms::owner_all, fs::perm_options::add, ec);
    fs::remove_all(g_root, ec);
    return all_pass ? 0 : 1;
}
