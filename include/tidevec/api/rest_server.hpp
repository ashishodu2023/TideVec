#pragma once
// ================================================================
// RestServer — TideVec HTTP/1.1 REST API
//
// Endpoints:
//
//   Health / Info
//     GET  /health
//     GET  /v1/info
//     GET  /metrics             (Prometheus text format)
//
//   Collections
//     GET  /v1/collections                   list all
//     POST /v1/collections                   create
//     GET  /v1/collections/{name}            get stats
//     DELETE /v1/collections/{name}          drop
//     PUT  /v1/collections/{name}/temporal   update temporal config
//
//   Vectors
//     POST /v1/collections/{name}/upsert     upsert 1..N vectors
//     POST /v1/collections/{name}/delete     delete by ids and/or filter
//     GET  /v1/collections/{name}/vectors/{id}  fetch one vector
//     POST /v1/collections/{name}/get        fetch vectors by id
//     POST /v1/collections/{name}/scroll     page through vectors (filter, as_of)
//     GET|POST /v1/collections/{name}/ids    page through ids
//     POST /v1/collections/{name}/count      count (filter, as_of)
//
//   Aliases
//     GET    /v1/aliases
//     PUT    /v1/aliases/{alias}             create / atomically repoint
//     DELETE /v1/aliases/{alias}
//
//   Search
//     POST /v1/collections/{name}/search            ANN / temporal
//     POST /v1/collections/{name}/search/causal     causal expand
//     POST /v1/collections/{name}/search/contradict contradiction
//
//   Graph
//     POST /v1/collections/{name}/edges      add causal edge(s)
//
//   Trace
//     GET  /v1/trace/{query_id}              retrieve stored trace
// ================================================================

#include <tidevec/api/httplib.h>
#include <tidevec/api/json_serializers.hpp>
#include <tidevec/api/collection_registry.hpp>
#include <tidevec/api/server_security.hpp>
#include <tidevec/drift/drift_bridge.hpp>
#include <tidevec/drift/drift_http.hpp>
#include <tidevec/observability/otel_exporter.hpp>
#include <tidevec/ops/backup_manager.hpp>
#include <tidevec/consensus/raft_http.hpp>
#include <tidevec/cluster/state_snapshot.hpp>

#include <filesystem>
#include <fstream>
#include <string>
#include <atomic>
#include <chrono>
#include <mutex>
#include <unordered_map>
#include <sstream>
#include <iostream>

#ifndef TIDEVEC_SERVER_VERSION
#define TIDEVEC_SERVER_VERSION "0.3.0"
#endif

namespace tidevec {

// ================================================================
// Metrics — simple atomic counters for Prometheus endpoint
// ================================================================
struct ServerMetrics {
    std::atomic<uint64_t> requests_total{0};
    std::atomic<uint64_t> search_requests{0};
    std::atomic<uint64_t> upsert_requests{0};
    std::atomic<uint64_t> errors_total{0};
    std::atomic<uint64_t> vectors_inserted{0};
    std::chrono::steady_clock::time_point start_time{
        std::chrono::steady_clock::now()};

    std::string to_prometheus() const {
        auto uptime = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - start_time).count();
        std::ostringstream ss;
        ss << "# HELP tidevec_requests_total Total HTTP requests\n"
           << "# TYPE tidevec_requests_total counter\n"
           << "tidevec_requests_total " << requests_total.load() << "\n\n"
           << "# HELP tidevec_search_requests_total Search requests\n"
           << "# TYPE tidevec_search_requests_total counter\n"
           << "tidevec_search_requests_total " << search_requests.load() << "\n\n"
           << "# HELP tidevec_upsert_requests_total Upsert requests\n"
           << "# TYPE tidevec_upsert_requests_total counter\n"
           << "tidevec_upsert_requests_total " << upsert_requests.load() << "\n\n"
           << "# HELP tidevec_errors_total Total errors\n"
           << "# TYPE tidevec_errors_total counter\n"
           << "tidevec_errors_total " << errors_total.load() << "\n\n"
           << "# HELP tidevec_vectors_inserted_total Vectors upserted\n"
           << "# TYPE tidevec_vectors_inserted_total counter\n"
           << "tidevec_vectors_inserted_total " << vectors_inserted.load() << "\n\n"
           << "# HELP tidevec_uptime_seconds Server uptime\n"
           << "# TYPE tidevec_uptime_seconds gauge\n"
           << "tidevec_uptime_seconds " << uptime << "\n";
        return ss.str();
    }
};

// ================================================================
// RestServer
// ================================================================
struct RestServerConfig {
    std::string host         = "0.0.0.0";
    int         port         = 6399;
    int         threads      = 8;
    std::string api_key      = "";
    std::string data_dir     = "./tidevec_data";
    bool        log_requests = true;
    bool        require_auth = true;    // reject requests without valid API key

    // TLS (requires TIDEVEC_TLS=ON at build time)
    std::string tls_cert     = "";
    std::string tls_key      = "";

    // Durability / performance
    bool ultra_durable       = true;
    bool use_segment_store   = true;
    std::string device       = "auto";  // cpu | gpu | tpu | auto

    // Rate limiting
    int rate_limit_rps       = 100;
    int rate_limit_burst     = 200;

    // SSRF allowlist (comma-separated hosts; empty = block private IPs only)
    std::vector<std::string> reembed_allowed_hosts;
    bool reembed_allow_private = false;

    // OTel
    std::string otel_endpoint = "";
    bool otel_enabled         = false;

    // Backup
    bool backup_enabled       = false;
    int  backup_interval_hours= 6;
    std::string backup_dir    = "./backups";
    std::string backup_s3_uri = "";
    std::string backup_gcs_uri= "";

    // Multi-tenancy
    bool multi_tenant         = false;

    // EXPERIMENTAL multi-process Raft cluster. raft_node_id >= 0 enables
    // it; raft_peers[i] is the base URL of node i (including this one;
    // empty entries are ids that are not members). Every write is
    // replicated through the Raft log. Reads are served locally and may
    // lag the leader, unless the request asks for
    // ?consistency=linearizable (ReadIndex) or ?consistency=lease.
    int raft_node_id          = -1;
    std::vector<std::string> raft_peers;
    std::string cluster_secret;
    bool raft_verbose         = false;
    bool raft_join            = false;   // start empty; wait for /v1/admin/cluster/add_node
    uint64_t raft_snapshot_threshold = 4096;   // applied entries per snapshot (0 = never)
    int  raft_election_timeout_ms = 400;  // minimum; the maximum is twice this
    int  raft_heartbeat_ms    = 100;
    bool enable_fault_injection = false;  // exposes /v1/admin/fault (tests only)
};

class RestServer {
public:
    using Config = RestServerConfig;

    explicit RestServer(Config cfg = RestServerConfig{})
        : cfg_(std::move(cfg))
        , registry_(_make_registry_config(cfg_))
        , rate_limiter_(RateLimiter::Config{cfg_.rate_limit_rps, cfg_.rate_limit_burst})
        , url_allowlist_(cfg_.reembed_allowed_hosts, cfg_.reembed_allow_private)
        , tenant_store_(cfg_.data_dir)
        , otel_(OtelConfig{cfg_.otel_endpoint, "tidevec", cfg_.otel_enabled})
        , backup_(_make_backup_config(cfg_))
    {
        if (!cfg_.require_auth && cfg_.multi_tenant)
            throw std::invalid_argument("--multi-tenant needs authentication; it cannot be "
                                        "combined with --no-require-auth / TIDEVEC_REQUIRE_AUTH=0");
        if (!cfg_.require_auth && !cfg_.api_key.empty())
            std::cerr << "\n*** WARNING: authentication is DISABLED (--no-require-auth / "
                         "TIDEVEC_REQUIRE_AUTH=0). The configured API key is ignored and every "
                         "caller has admin access. ***\n\n";
        if (cfg_.multi_tenant && tenant_store_.size() == 0)
            std::cerr << "Warning: --multi-tenant set but " << cfg_.data_dir
                      << "/tenants.json lists no tenants; only the admin key is accepted.\n";
        _register_routes();
    }

    // Blocking — call from main thread
    void listen() {
        std::cout << "\n╔══════════════════════════════════════╗\n";
        std::cout << "║   TideVec REST API v" TIDEVEC_SERVER_VERSION "           ║\n";
        std::cout << "║   " << (_tls_enabled() ? "https" : "http") << "://" << cfg_.host
                  << ":" << cfg_.port << "          ║\n";
        std::cout << "╚══════════════════════════════════════╝\n\n";
        std::cout << "Data directory: " << cfg_.data_dir << "\n";
        std::cout << "Ultra-durable:  " << (cfg_.ultra_durable ? "ON" : "OFF") << "\n";
        std::cout << "Segment store:  " << (cfg_.use_segment_store ? "ON" : "OFF") << "\n";
        std::cout << "Device:         " << cfg_.device << "\n";
        std::cout << "Auth required:  " << (cfg_.require_auth ? "YES" : "NO") << "\n";

        std::cout << "Recovering persisted collections...\n";
        registry_.load_and_recover();
        std::cout << "Ready — " << registry_.size() << " collection(s) loaded\n\n";

        if (cfg_.raft_node_id >= 0) _start_raft();

        backup_.start();

        svr_.new_task_queue = [this] {
            return new httplib::ThreadPool(cfg_.threads);
        };

        if (_tls_enabled()) std::cout << "TLS enabled:    " << cfg_.tls_cert << "\n";
        bool clean = svr_.listen(cfg_.host, cfg_.port);
        if (!clean && !stopping_) {
            if (raft_) raft_->stop();
            throw std::runtime_error("could not listen on " + cfg_.host + ":" +
                                     std::to_string(cfg_.port) + " (port in use?)");
        }
    }

    void stop() {
        stopping_ = true;
        backup_.stop();
        svr_.stop();
        if (raft_) raft_->stop();
    }

    CollectionRegistry& registry() { return registry_; }
    const ServerMetrics& metrics() const { return metrics_; }

private:
    static RegistryConfig _make_registry_config(const Config& cfg) {
        RegistryConfig rcfg;
        rcfg.data_dir          = cfg.data_dir;
        // In cluster mode replication happens across processes; the
        // in-process Raft group of the ultra-durable backend is redundant.
        rcfg.ultra_durable     = cfg.ultra_durable && cfg.raft_node_id < 0;
        rcfg.use_segment_store = cfg.use_segment_store;
        rcfg.use_accel         = (cfg.device == "gpu" || cfg.device == "tpu" ||
                                  cfg.device == "auto");
        rcfg.device            = _parse_device(cfg.device);
        // Background rebuild timing differs per node; cluster replicas
        // compact inline so they keep building identical graphs.
        rcfg.background_compaction = cfg.raft_node_id < 0;
        return rcfg;
    }

    static BackupConfig _make_backup_config(const Config& cfg) {
        BackupConfig bcfg;
        bcfg.data_dir       = cfg.data_dir;
        bcfg.backup_dir     = cfg.backup_dir;
        bcfg.interval_hours = cfg.backup_interval_hours;
        bcfg.s3_uri         = cfg.backup_s3_uri;
        bcfg.gcs_uri        = cfg.backup_gcs_uri;
        bcfg.enabled        = cfg.backup_enabled;
        return bcfg;
    }

    static accel::DeviceType _parse_device(const std::string& d) {
        if (d == "gpu") return accel::DeviceType::GPU;
        if (d == "tpu") return accel::DeviceType::TPU;
        if (d == "cpu") return accel::DeviceType::CPU;
        return accel::DeviceType::AUTO;
    }

    // TLS needs a build with -DTIDEVEC_TLS=ON (CPPHTTPLIB_OPENSSL_SUPPORT).
    // Asking for it in a build without it is an error rather than a silent
    // fallback to plain HTTP.
    static std::unique_ptr<httplib::Server> _make_http_server(const Config& cfg) {
        if (cfg.tls_cert.empty() != cfg.tls_key.empty())
            throw std::invalid_argument("--tls-cert and --tls-key must be given together");
        if (cfg.tls_cert.empty()) return std::make_unique<httplib::Server>();
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
        auto s = std::make_unique<httplib::SSLServer>(cfg.tls_cert.c_str(), cfg.tls_key.c_str());
        if (!s->is_valid())
            throw std::runtime_error("TLS: could not load certificate " + cfg.tls_cert +
                                     " / key " + cfg.tls_key);
        return s;
#else
        throw std::invalid_argument("TLS requested but this server was built without TLS "
                                    "support (rebuild with -DTIDEVEC_TLS=ON or terminate "
                                    "TLS at a proxy)");
#endif
    }

    bool _tls_enabled() const { return !cfg_.tls_cert.empty(); }

    // Tenant id for an API key: TenantStore::kAdminId for the configured
    // --api-key, a tenants.json id in --multi-tenant mode, "" otherwise.
    std::string _identify(const std::string& key) const {
        if (key.empty()) return {};
        if (!cfg_.api_key.empty() && constant_time_equals(key, cfg_.api_key))
            return TenantStore::kAdminId;
        if (cfg_.multi_tenant) return tenant_store_.authenticate(key);
        return {};
    }

    // Rate-limit bucket: the authenticated tenant, else the client address
    // (made-up keys must not each get a fresh bucket).
    std::string _client_key(const httplib::Request& req) const {
        auto it = req.headers.find("X-Api-Key");
        if (it != req.headers.end()) {
            auto who = _identify(it->second);
            if (!who.empty()) return "tenant:" + who;
        }
        return "ip:" + req.remote_addr;
    }

    bool _rate_limit(const httplib::Request& req, httplib::Response& res) {
        if (!rate_limiter_.allow(_client_key(req))) {
            res.status = 429;
            res.set_content(err("Rate limit exceeded", 429).dump(), "application/json");
            ++metrics_.errors_total;
            return false;
        }
        return true;
    }

    // ---- middleware helpers ------------------------------------
    bool _auth(const httplib::Request& req, httplib::Response& res) {
        tls_current_tenant.clear();
        // Auth disabled: every caller is the admin.
        if (!cfg_.require_auth) return true;

        auto it = req.headers.find("X-Api-Key");
        if (it == req.headers.end() || it->second.empty()) {
            res.status = 401;
            res.set_content(err("Unauthorized — X-Api-Key required", 401).dump(),
                            "application/json");
            ++metrics_.errors_total;
            return false;
        }

        auto who = _identify(it->second);
        if (!who.empty()) {
            tls_current_tenant = who;
            return true;
        }

        res.status = 401;
        res.set_content(err("Unauthorized", 401).dump(), "application/json");
        ++metrics_.errors_total;
        return false;
    }

    // The admin is the --api-key holder (or anyone when auth is off);
    // tenants.json keys are tenants.
    bool _is_admin() const {
        return tls_current_tenant.empty() || tls_current_tenant == TenantStore::kAdminId;
    }

    bool _require_admin(httplib::Response& res) {
        if (_is_admin()) return true;
        _error(res, "This endpoint requires the admin API key", 403);
        return false;
    }

    // A tenant may use a name (collection or alias) only if both the name
    // and the collection it resolves to are inside its prefixes.
    bool _tenant_allows(const std::string& name) const {
        if (_is_admin()) return true;
        if (!tenant_store_.can_access_collection(tls_current_tenant, name)) return false;
        auto real = registry_.resolve(name);
        return real == name || tenant_store_.can_access_collection(tls_current_tenant, real);
    }

    bool _tenant_can_access(const std::string& collection,
                            httplib::Response& res) {
        if (_tenant_allows(collection)) return true;
        _error(res, "Tenant not authorized for collection: " + collection, 403);
        return false;
    }

    // {collections, vectors} owned by the current tenant.
    std::pair<std::size_t, std::size_t> _tenant_usage() const {
        std::size_t cols = 0, vecs = 0;
        for (const auto& info : registry_.list()) {
            if (!tenant_store_.can_access_collection(tls_current_tenant, info.name)) continue;
            ++cols;
            vecs += info.n_vectors;
        }
        return {cols, vecs};
    }

    bool _tenant_check_collection_quota(httplib::Response& res) {
        if (_is_admin()) return true;
        if (!tenant_store_.check_collection_quota(tls_current_tenant,
                                                   _tenant_usage().first)) {
            _error(res, "Collection quota exceeded for tenant", 403);
            return false;
        }
        return true;
    }

    bool _tenant_check_vector_quota(std::size_t adding,
                                    httplib::Response& res) {
        if (_is_admin()) return true;
        if (!tenant_store_.check_vector_quota(tls_current_tenant,
                                               _tenant_usage().second,
                                               adding)) {
            _error(res, "Vector quota exceeded for tenant", 403);
            return false;
        }
        return true;
    }

    // Defense in depth for routes that forget a tenant check: before
    // routing, a tenant key is refused on admin routes and on
    // /v1/collections/{name}/... and /v1/aliases/{name} outside its
    // prefixes. Requests without a valid key fall through to the route's
    // own 401.
    bool _tenant_route_guard(const httplib::Request& req, httplib::Response& res) {
        tls_current_tenant.clear();
        if (!cfg_.multi_tenant) return true;
        auto it = req.headers.find("X-Api-Key");
        if (it == req.headers.end()) return true;
        auto who = _identify(it->second);
        if (who.empty() || who == TenantStore::kAdminId) return true;
        tls_current_tenant = who;

        const std::string& p = req.path;
        if (p.starts_with("/v1/admin/") || p.starts_with("/v1/cluster/") ||
            p.starts_with("/raft/"))
            return _require_admin(res);
        for (std::string prefix : {"/v1/collections/", "/v1/aliases/"}) {
            if (!p.starts_with(prefix)) continue;
            auto end = p.find('/', prefix.size());
            auto name = p.substr(prefix.size(), end == std::string::npos
                                                    ? std::string::npos : end - prefix.size());
            if (!name.empty() && !_tenant_allows(name)) {
                _error(res, "Tenant not authorized for collection: " + name, 403);
                return false;
            }
        }
        return true;
    }

    static void _set_cors(httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin",  "*");
        res.set_header("Access-Control-Allow-Methods", "GET,POST,DELETE,PUT,OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type,X-Api-Key");
    }

    void _json_response(httplib::Response& res, const json& j, int status = 200) {
        res.status = status;
        res.set_content(j.dump(2), "application/json");
        _set_cors(res);
    }

    void _error(httplib::Response& res, const std::string& msg, int status = 400) {
        ++metrics_.errors_total;
        if (status == 400 && msg.rfind("Collection not found", 0) == 0) status = 404;
        _json_response(res, err(msg, status), status);
    }

    // Read-only collection route: auth, rate limit, tenant check, then
    // fn(collection) -> data. std::out_of_range maps to 404.
    template <typename F>
    void _read(const httplib::Request& req, httplib::Response& res, F&& fn) {
        ++metrics_.requests_total;
        if (!_rate_limit(req, res)) return;
        if (!_auth(req, res)) return;
        std::string name = req.matches[1];
        if (!_tenant_can_access(name, res)) return;
        if (!_read_barrier(req, res)) return;
        try {
            auto col_ref = registry_.acquire(name);
            auto& col = *col_ref;
            _json_response(res, ok(fn(col)));
        } catch (const std::out_of_range& e) {
            _error(res, e.what(), 404);
        } catch (const std::exception& e) {
            _error(res, e.what());
        }
    }

    static std::string _cursor(const json& j) {
        if (!j.contains("cursor") || j["cursor"].is_null()) return "";
        return j["cursor"].get<std::string>();
    }

    static std::size_t _limit(const json& j, std::size_t def) {
        std::size_t n = j.value("limit", def);
        if (n == 0 || n > 10000)
            throw std::invalid_argument("limit must be between 1 and 10000");
        return n;
    }

    void _drift_shadow_write(const std::string& collection,
                             const CortexVector& vec) {
        std::lock_guard<std::mutex> lg(drift_mutex_);
        auto it = drift_states_.find(collection);
        if (it == drift_states_.end()) return;
        if (it->second.bridge->phase() == DriftPhase::MIGRATING)
            it->second.bridge->shadow_write(vec, it->second.re_embed);
    }

    void _drift_apply_shadow(const std::string& collection,
                             const DriftBridge* bridge) {
        std::vector<CortexVector> migrated;
        {
            std::lock_guard<std::mutex> lg(drift_mutex_);
            auto it = drift_states_.find(collection);
            if (it == drift_states_.end() || it->second.bridge.get() != bridge ||
                it->second.bridge->phase() == DriftPhase::FAILED)
                return;
            if (auto shadow = it->second.bridge->take_shadow()) {
                shadow->each_vector([&](const CortexVector& v) {
                    migrated.push_back(v);
                });
            }
        }
        if (migrated.empty()) return;
        auto col_ref = registry_.acquire(collection);
        auto& col = *col_ref;

        // Only swap in vectors whose live version is the one that was
        // re-embedded: anything deleted or rewritten while the migration
        // ran must not be resurrected or rolled back.
        std::unordered_map<std::string, CortexVector> live;
        for (auto& v : col.snapshot_vectors()) live.emplace(v.id, std::move(v));
        for (const auto& v : migrated) {
            auto it = live.find(v.id);
            if (it == live.end() || it->second.created_at != v.created_at ||
                it->second.payload != v.payload)
                continue;
            // Runs on the migration thread: a re-embedder returning the
            // wrong dimension must not take the server down.
            try { col.upsert(v); }
            catch (const std::exception&) { ++metrics_.errors_total; }
        }
    }

    // ---- write path ---------------------------------------------
    //
    // Every mutating endpoint funnels through _apply_write(op, name,
    // body). Standalone servers call it directly; in cluster mode the
    // leader appends {op,name,body} to the Raft log and each node calls
    // it from the apply thread, so all replicas execute the same
    // deterministic sequence of writes.

    json _apply_write(const std::string& op, const std::string& name,
                      const json& j) {
        if (op == "create") {
            CollectionRegistry::CreateParams p;
            p.name        = name;
            p.dim         = j.value("dim",         768UL);
            p.index_type  = j.value("index_type",  std::string("tvindex"));
            p.metric      = j.value("metric",      std::string("cosine"));
            p.n_shards    = j.value("n_shards",    4UL);
            if (p.n_shards == 0) p.n_shards = 4;
            p.n_replicas  = j.value("n_replicas",  1);
            p.write_quorum= j.value("write_quorum",1);
            p.data_dir    = cfg_.data_dir;
            if (j.contains("temporal"))
                p.temporal = temporal_cfg_from_json(j["temporal"]);
            if (j.contains("agent"))
                p.agent = agent_cfg_from_json(j["agent"]);
            if (j.contains("supersession"))
                p.supersession = supersession_cfg_from_json(j["supersession"]);
            p.text_field    = j.value("text_field", std::string(""));
            p.payload_index = j.value("payload_index", true);
            p.compaction_ratio = j.value("compaction_ratio", 1.0);
            if (!(p.compaction_ratio > 0))
                throw std::invalid_argument("compaction_ratio must be > 0");
            registry_.create(p);
            otel_.export_event("info", "Collection created",
                {{"name", p.name}, {"tenant", tls_current_tenant}});
            return {{"name", p.name}, {"message", "Collection created"}};
        }
        if (op == "drop") {
            if (!registry_.drop(name))
                throw std::runtime_error("Collection not found: " + name);
            return {{"message", "Deleted"}};
        }
        if (op == "alias_put") {
            auto target = j.at("collection").get<std::string>();
            auto prev = registry_.set_alias(name, target);
            json out = {{"alias", name}, {"collection", target}};
            out["previous"] = prev.empty() ? json(nullptr) : json(prev);
            return out;
        }
        if (op == "alias_delete") {
            if (!registry_.delete_alias(name))
                throw std::runtime_error("Alias not found: " + name);
            return {{"message", "Alias deleted"}, {"alias", name}};
        }
        auto col_ref = registry_.acquire(name);
        auto& col = *col_ref;
        if (op == "upsert") {
            std::vector<std::string> inserted_ids;
            const std::string target = registry_.resolve(name);
            auto process = [&](const json& item) {
                auto v = vector_from_json(item);
                col.upsert(v);
                _drift_shadow_write(target, v);
                inserted_ids.push_back(v.id);
                ++metrics_.vectors_inserted;
            };
            if (j.contains("vectors")) {
                for (const auto& item : j["vectors"]) process(item);
            } else {
                process(j);  // single vector body
            }
            return {{"inserted", inserted_ids.size()}, {"ids", inserted_ids}};
        }
        if (op == "delete") {
            if (!j.contains("ids") && !j.contains("filter"))
                throw std::invalid_argument("body needs \"ids\" or \"filter\"");
            std::size_t removed = 0;
            if (j.contains("ids"))
                for (const auto& id : j["ids"].get<std::vector<std::string>>())
                    if (col.remove(id)) ++removed;
            if (j.contains("filter")) {
                if (j["filter"].is_string())
                    removed += col.remove_where(j["filter"].get<std::string>());
                else
                    removed += col.remove_where(filter_from_json(j["filter"]));
            }
            return {{"deleted", removed}};
        }
        if (op == "edges") {
            int added = 0;
            for (const auto& e : j.at("edges")) {
                col.add_edge(
                    e.at("src").get<std::string>(),
                    e.at("tgt").get<std::string>(),
                    edge_type_from_str(e.at("type").get<std::string>()),
                    e.value("weight", 1.0f));
                ++added;
            }
            return {{"added", added}};
        }
        if (op == "temporal") {
            col.set_temporal_config(temporal_cfg_from_json(j));
            return {{"message", "Temporal config updated"}};
        }
        if (op == "compact") {
            // Cluster nodes compact inline at the same log position so
            // their graphs stay identical.
            return _compaction_json(col.compact(raft_ ? true : j.value("wait", false)));
        }
        throw std::invalid_argument("unknown write op: " + op);
    }

    static json _compaction_json(const CompactionStatus& s) {
        return compaction_status_to_json(s);
    }

    // Wall-clock defaults must be fixed once, on the leader, or each
    // replica would stamp a different created_at.
    void _normalise_for_log(const std::string& op, json& j) const {
        if (op != "upsert") return;
        auto now = now_ms() + (fault_ ? fault_->clock_offset_ms.load() : 0);
        auto fix = [now](json& item) {
            if (item.is_object() && !item.contains("created_at"))
                item["created_at"] = now;
        };
        if (j.contains("vectors")) for (auto& item : j["vectors"]) fix(item);
        else fix(j);
    }

    void _write(httplib::Response& res, const std::string& op,
                const std::string& name, json body, int ok_status = 200) {
        if (!raft_) {
            _json_response(res, ok(_apply_write(op, name, body)), ok_status);
            return;
        }
        if (!raft_->is_leader()) { _not_leader(res); return; }

        _normalise_for_log(op, body);
        json cmd = {{"op", op}, {"name", name}, {"body", body}};
        auto bytes = cmd.dump();
        std::vector<uint8_t> payload(bytes.begin(), bytes.end());

        std::unique_lock lock(raft_apply_mu_);
        consensus::Term term = raft_->current_term();
        std::optional<consensus::Index> idx;
        try { idx = raft_->submit("CMD", payload); }
        catch (const std::exception& e) {
            lock.unlock();
            _error(res, std::string("Write not accepted: ") + e.what(), 503);
            return;
        }
        if (!idx) { lock.unlock(); _not_leader(res); return; }
        // Our entry is (idx, term) only if no election slipped in between
        // reading the term and appending; otherwise the outcome is unknown.
        if (raft_->term_at(*idx) != term) {
            lock.unlock();
            _error(res, "Write not confirmed: leadership changed or quorum "
                        "unavailable; it may or may not have been applied", 503);
            return;
        }
        raft_waiting_[*idx] = std::nullopt;

        bool done = raft_apply_cv_.wait_for(lock, std::chrono::seconds(10), [&]{
            return raft_waiting_[*idx].has_value();
        });
        auto outcome = std::move(raft_waiting_[*idx]);
        raft_waiting_.erase(*idx);
        lock.unlock();

        // The entry applied at idx must be the one we appended (same term).
        if (!done || outcome->indeterminate || outcome->term != term) {
            _error(res, "Write not confirmed: leadership changed or quorum "
                        "unavailable; it may or may not have been applied", 503);
            return;
        }
        if (!outcome->error.empty()) { _error(res, outcome->error); return; }
        _json_response(res, ok(outcome->data), ok_status);
    }

    // Cluster mode: followers bounce writes to the leader before doing any
    // local validation (their state may lag the leader's).
    bool _follower_redirect(httplib::Response& res) {
        if (!raft_ || raft_->is_leader()) return false;
        _not_leader(res);
        return true;
    }

    void _not_leader(httplib::Response& res) {
        ++metrics_.errors_total;
        auto leader = raft_->leader_id();
        json body = err("Not the Raft leader; retry the write on the leader", 421);
        std::string url = leader >= 0 ? raft_->address_of(leader) : "";
        if (!url.empty()) {
            body["leader"] = url;
            res.set_header("X-TideVec-Leader", url);
        } else {
            body["leader"] = nullptr;
        }
        _json_response(res, body, 421);
    }

    std::filesystem::path _raft_dir() const {
        return std::filesystem::path(cfg_.data_dir) / "raft";
    }

    consensus::Index _load_applied_index() const {
        std::ifstream f(_raft_dir() / "applied");
        consensus::Index idx = 0;
        if (f >> idx) return idx;
        return 0;
    }

    void _store_applied_index(consensus::Index idx) {
        auto tmp = (_raft_dir() / "applied.tmp").string();
        { std::ofstream f(tmp, std::ios::trunc); f << idx << "\n"; }
        std::filesystem::rename(tmp, _raft_dir() / "applied");
    }

    void _start_raft() {
        int n = static_cast<int>(cfg_.raft_peers.size());
        if (cfg_.raft_node_id >= n || cfg_.raft_peers[cfg_.raft_node_id].empty())
            throw std::invalid_argument("--node-id must index into --raft-peers");
        std::filesystem::create_directories(_raft_dir());

        consensus::RaftConfig rc;
        rc.node_id = cfg_.raft_node_id;
        for (int i = 0; i < n; ++i)
            if (!cfg_.raft_peers[i].empty()) rc.members.push_back({i, cfg_.raft_peers[i]});
        rc.join = cfg_.raft_join;
        rc.election_timeout_min_ms = cfg_.raft_election_timeout_ms;
        rc.election_timeout_max_ms = 2 * cfg_.raft_election_timeout_ms;
        rc.heartbeat_interval_ms   = cfg_.raft_heartbeat_ms;
        rc.max_log_entries_per_rpc = 32;
        rc.snapshot_threshold      = cfg_.raft_snapshot_threshold;
        rc.persist_dir     = _raft_dir().string();
        rc.initial_applied = _load_applied_index();
        rc.verbose         = cfg_.raft_verbose;
        rc.fault           = fault_;

        raft_ = std::make_unique<consensus::RaftNode>(rc,
            [this](const consensus::LogEntry& e) { _raft_apply(e); });
        raft_->set_snapshot_handlers(
            [this] { return _raft_snapshot(); },
            [this](const std::vector<uint8_t>& data, consensus::Index idx) {
                _raft_restore(data, idx);
            });
        auto secret = cfg_.cluster_secret;
        auto fault  = fault_;
        raft_->set_peer_factory([secret, fault](const consensus::Member& m)
                -> std::shared_ptr<consensus::RaftPeer> {
            if (m.addr.empty()) return nullptr;
            std::shared_ptr<consensus::RaftPeer> p =
                std::make_shared<consensus::HttpPeer>(m.addr, secret);
            if (fault) p = std::make_shared<consensus::FaultyPeer>(p, m.id, fault);
            return p;
        });
        raft_->start();
        std::cout << "Raft cluster:   node " << rc.node_id
                  << (rc.join ? " (joining)" : "") << ", " << rc.members.size()
                  << " configured peer(s) (EXPERIMENTAL), resuming after log index "
                  << rc.initial_applied << ", snapshot every "
                  << rc.snapshot_threshold << " entries\n";
        if (fault_) std::cout << "Fault injection: ENABLED (/v1/admin/fault) — test use only\n";
    }

    // Raft snapshot of the state machine: the files restart recovery reads.
    // Runs on the apply thread, the only writer, so the cut is consistent.
    std::vector<uint8_t> _raft_snapshot() {
        std::vector<std::string> entries{"registry.json"};
        for (const auto& info : registry_.list()) entries.push_back(info.name);
        return state_snapshot::pack(cfg_.data_dir, entries);
    }

    // Replaces the whole state machine with a snapshot taken at idx.
    void _raft_restore(const std::vector<uint8_t>& data, consensus::Index idx) {
        auto files = state_snapshot::unpack(data);
        {
            std::lock_guard lg(drift_mutex_);
            for (auto& [name, st] : drift_states_) st.bridge->abort();
            drift_states_.clear();
        }
        registry_.replace_from_disk([&](const std::vector<std::string>& old_names) {
            std::error_code ec;
            for (const auto& name : old_names)
                std::filesystem::remove_all(std::filesystem::path(cfg_.data_dir) / name, ec);
            std::filesystem::remove(std::filesystem::path(cfg_.data_dir) / "registry.json", ec);
            for (const auto& f : files) {
                auto top = std::filesystem::path(f.path).begin()->string();
                if (top != "registry.json")
                    std::filesystem::remove_all(std::filesystem::path(cfg_.data_dir) / top, ec);
            }
            state_snapshot::write_files(cfg_.data_dir, files);
        });
        std::lock_guard lock(raft_apply_mu_);
        _store_applied_index(idx);
        // Writes waiting on entries the snapshot covered can no longer be
        // matched to their outcome.
        for (auto& [i, out] : raft_waiting_)
            if (i <= idx && !out) { out = RaftOutcome{}; out->indeterminate = true; }
        raft_apply_cv_.notify_all();
        std::cout << "Raft: restored state snapshot at index " << idx << " ("
                  << files.size() << " files, " << registry_.size() << " collections)\n";
    }

    // Cluster read consistency, from ?consistency= or X-TideVec-Consistency:
    //   local (default) — serve this node's state, may be stale
    //   linearizable    — leader only; ReadIndex round before serving
    //   lease           — leader only; skips the round while its lease holds
    // Returns false after writing a 421/503 response.
    bool _read_barrier(const httplib::Request& req, httplib::Response& res) {
        if (!raft_) return true;
        std::string mode = req.get_param_value("consistency");
        if (mode.empty()) {
            auto it = req.headers.find("X-TideVec-Consistency");
            if (it != req.headers.end()) mode = it->second;
        }
        if (mode.empty() || mode == "local" || mode == "stale") return true;
        if (mode != "linearizable" && mode != "lease") {
            _error(res, "consistency must be local, linearizable or lease");
            return false;
        }
        if (!raft_->is_leader()) { _not_leader(res); return false; }
        auto ri = mode == "lease" ? raft_->lease_read(3000) : raft_->read_index(3000);
        if (!ri) {
            _error(res, "Read not served: could not confirm leadership with a quorum", 503);
            return false;
        }
        return true;
    }

    // Runs on the Raft apply thread, once per committed entry, in order.
    // A crash between applying and recording the index re-applies that
    // one entry on restart (writes are upserts/deletes, so this is benign
    // apart from possibly duplicated edges).
    void _raft_apply(const consensus::LogEntry& e) {
        RaftOutcome out;
        out.term = e.term;
        if (e.op == "CMD") {
            try {
                auto cmd = json::parse(e.payload.begin(), e.payload.end());
                out.data = _apply_write(cmd.at("op").get<std::string>(),
                                        cmd.at("name").get<std::string>(),
                                        cmd.at("body"));
            } catch (const std::exception& ex) {
                out.error = ex.what();
            }
        }
        std::lock_guard lock(raft_apply_mu_);
        _store_applied_index(e.index);
        auto it = raft_waiting_.find(e.index);
        if (it != raft_waiting_.end()) {
            it->second = std::move(out);
            raft_apply_cv_.notify_all();
        }
    }

    bool _cluster_auth(const httplib::Request& req, httplib::Response& res) {
        if (!raft_) {
            _error(res, "Raft not running on this node", 503);
            return false;
        }
        auto it = req.headers.find(consensus::kClusterSecretHeader);
        std::string got = it == req.headers.end() ? "" : it->second;
        if (!consensus::secret_equals(got, cfg_.cluster_secret)) {
            _error(res, "Bad cluster secret", 403);
            return false;
        }
        return true;
    }

    // Backup restore: closes every collection, runs swap (which replaces
    // the data directory's contents), then reloads from disk, also when
    // swap throws.
    void _reload_around(const std::function<void()>& swap) {
        {
            std::lock_guard lg(drift_mutex_);
            for (auto& [name, st] : drift_states_) st.bridge->abort();
            drift_states_.clear();
        }
        std::exception_ptr failure;
        registry_.replace_from_disk([&](const std::vector<std::string>&) {
            try { swap(); } catch (...) { failure = std::current_exception(); }
        });
        if (failure) std::rethrow_exception(failure);
    }

    // ---- route registration ------------------------------------
    void _register_routes() {
        // CORS preflight
        svr_.Options(".*", [](const httplib::Request&, httplib::Response& res) {
            _set_cors(res);
            res.status = 204;
        });

        // --------------------------------------------------------
        // GET /health
        // --------------------------------------------------------
        svr_.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
            ++metrics_.requests_total;
            _json_response(res, {
                {"status", "ok"},
                {"version", TIDEVEC_SERVER_VERSION},
                {"collections", registry_.size()},
                {"ultra_durable", cfg_.ultra_durable},
                {"segment_store", cfg_.use_segment_store},
                {"device", cfg_.device},
                {"auth_required", cfg_.require_auth},
                {"multi_tenant", cfg_.multi_tenant},
                {"backup_enabled", cfg_.backup_enabled},
                {"timestamp_ms", now_ms()}
            });
        });

        // --------------------------------------------------------
        // GET /v1/info
        // --------------------------------------------------------
        svr_.Get("/v1/info", [this](const httplib::Request&, httplib::Response& res) {
            ++metrics_.requests_total;
            _json_response(res, {
                {"name", "TideVec"},
                {"version", TIDEVEC_SERVER_VERSION},
                {"description", "Temporally-aware causal vector database"},
                {"features", json::array({
                    "TVIndex", "CausalEdge", "Supersession", "DriftBridge",
                    "RetrievalTrace", "UltraDurable", "SegmentStore",
                    "MultiTenant", "AutomatedBackup", "RateLimiting", "OTel"
                })},
                {"production", {
                    {"ultra_durable",    cfg_.ultra_durable},
                    {"segment_store",    cfg_.use_segment_store},
                    {"device",           cfg_.device},
                    {"auth_required",    cfg_.require_auth},
                    {"multi_tenant",     cfg_.multi_tenant},
                    {"backup_enabled",   cfg_.backup_enabled},
                    {"otel_enabled",     cfg_.otel_enabled},
                    {"rate_limit_rps",   cfg_.rate_limit_rps},
                }},
                {"collections", registry_.size()}
            });
        });

        // --------------------------------------------------------
        // GET /metrics   (Prometheus)
        // --------------------------------------------------------
        svr_.Get("/metrics", [this](const httplib::Request&, httplib::Response& res) {
            res.status = 200;
            res.set_content(metrics_.to_prometheus(), "text/plain; version=0.0.4");
        });

        // --------------------------------------------------------
        // GET /v1/collections
        // --------------------------------------------------------
        svr_.Get("/v1/collections", [this](const httplib::Request& req,
                                           httplib::Response& res) {
            ++metrics_.requests_total;
            if (!_auth(req, res)) return;
            auto list = registry_.list();
            json arr = json::array();
            for (const auto& info : list) {
                if (!_tenant_allows(info.name)) continue;
                arr.push_back({
                    {"name",       info.name},
                    {"n_vectors",  info.n_vectors},
                    {"n_shards",   info.n_shards},
                    {"dim",        info.dim},
                    {"index_type", info.index_type},
                    {"metric",     info.metric},
                    {"backend",    info.backend}
                });
            }
            _json_response(res, ok(arr));
        });

        // --------------------------------------------------------
        // POST /v1/collections    — create collection
        // Body: {"name":"docs","dim":768,"n_shards":4,"metric":"cosine"}
        // --------------------------------------------------------
        svr_.Post("/v1/collections", [this](const httplib::Request& req,
                                            httplib::Response& res) {
            ++metrics_.requests_total;
            if (!_auth(req, res)) return;
            if (_follower_redirect(res)) return;
            try {
                auto j = json::parse(req.body);
                auto cname = j.at("name").get<std::string>();
                validate_collection_name(cname);
                if (!_tenant_can_access(cname, res)) return;
                if (!_tenant_check_collection_quota(res)) return;
                _write(res, "create", cname, std::move(j), 201);
            } catch (const std::exception& e) {
                _error(res, e.what());
            }
        });

        // --------------------------------------------------------
        // GET /v1/collections/{name}
        // --------------------------------------------------------
        svr_.Get(R"(/v1/collections/([^/]+)$)",
            [this](const httplib::Request& req, httplib::Response& res) {
            ++metrics_.requests_total;
            if (!_auth(req, res)) return;
            std::string name = req.matches[1];
            if (!_tenant_can_access(name, res)) return;
            if (!_read_barrier(req, res)) return;
            try {
                auto col_ref = registry_.acquire(name);
                auto& col = *col_ref;
                const std::string real = registry_.resolve(name);
                json info = {
                    {"name",          real},
                    {"n_vectors",     col.total_vectors()},
                    {"n_shards",      col.n_shards()},
                    {"total_writes",  col.total_writes()},
                    {"total_queries", col.total_queries()},
                    {"backend",       col.backend_name()},
                    {"supersession",  supersession_cfg_to_json(col.supersession_config())},
                    {"text_field",    col.text_field()},
                    {"aliases",       registry_.aliases_of(real)},
                };
                if (auto p = registry_.params(real)) {
                    info["dim"]           = p->dim;
                    info["index_type"]    = p->index_type;
                    info["metric"]        = p->metric;
                    info["payload_index"] = p->payload_index;
                    info["compaction_ratio"] = p->compaction_ratio;
                }
                info["compaction"] = _compaction_json(col.compaction_status());
                if (real != name) info["alias"] = name;
                _json_response(res, ok(info));
            } catch (const std::exception& e) {
                _error(res, e.what(), 404);
            }
        });

        // --------------------------------------------------------
        // DELETE /v1/collections/{name}
        // --------------------------------------------------------
        svr_.Delete(R"(/v1/collections/([^/]+)$)",
            [this](const httplib::Request& req, httplib::Response& res) {
            ++metrics_.requests_total;
            if (!_auth(req, res)) return;
            if (_follower_redirect(res)) return;
            std::string name = req.matches[1];
            if (!_tenant_can_access(name, res)) return;
            try {
                _write(res, "drop", name, json::object());
            } catch (const std::exception& e) {
                _error(res, e.what());
            }
        });

        // --------------------------------------------------------
        // POST /v1/collections/{name}/upsert
        // Body: {"vectors":[{"id":"v1","embedding":[...],"payload":{}},...]}
        //    or single vector: {"id":"v1","embedding":[...]}
        // --------------------------------------------------------
        svr_.Post(R"(/v1/collections/([^/]+)/upsert)",
            [this](const httplib::Request& req, httplib::Response& res) {
            ++metrics_.requests_total;
            ++metrics_.upsert_requests;
            if (!_rate_limit(req, res)) return;
            if (!_auth(req, res)) return;
            if (_follower_redirect(res)) return;
            std::string name = req.matches[1];
            if (!_tenant_can_access(name, res)) return;
            try {
                registry_.get(name);
                auto j = json::parse(req.body);
                std::size_t n_vectors = j.contains("vectors")
                    ? j["vectors"].size() : 1;
                if (!_tenant_check_vector_quota(n_vectors, res)) return;
                _write(res, "upsert", name, std::move(j), 201);
            } catch (const std::exception& e) {
                _error(res, e.what());
            }
        });

        // --------------------------------------------------------
        // POST /v1/collections/{name}/delete
        // Body: {"ids":["v1","v2"]} and/or {"filter":"key=value"}
        // --------------------------------------------------------
        svr_.Post(R"(/v1/collections/([^/]+)/delete)",
            [this](const httplib::Request& req, httplib::Response& res) {
            ++metrics_.requests_total;
            if (!_auth(req, res)) return;
            if (_follower_redirect(res)) return;
            std::string name = req.matches[1];
            if (!_tenant_can_access(name, res)) return;
            try {
                registry_.get(name);
                _write(res, "delete", name, json::parse(req.body));
            } catch (const std::exception& e) {
                _error(res, e.what());
            }
        });

        // --------------------------------------------------------
        // POST /v1/collections/{name}/search
        // Body: {
        //   "vector": [...],
        //   "top_k": 10,
        //   "temporal_blend": 0.3,
        //   "mode": "vector_only"|"causal_expand"|"contradiction_check",
        //   "filter": "key=value" | {"lang":"en","year":{"$gte":2020}},
        //   "as_of": 1700000000000 | "2025-01-01T00:00:00Z",
        //   "text": "keyword query",          (BM25; hybrid with "vector")
        //   "hybrid": {"fusion":"rrf"|"weighted","alpha":0.5,"rrf_k":60},
        //   "include_trace": false,
        //   "metric": "cosine"
        // }
        // "vector" may be omitted when "text" is given (keyword-only).
        // --------------------------------------------------------
        svr_.Post(R"(/v1/collections/([^/]+)/search)",
            [this](const httplib::Request& req, httplib::Response& res) {
            ++metrics_.requests_total;
            ++metrics_.search_requests;
            if (!_rate_limit(req, res)) return;
            if (!_auth(req, res)) return;
            std::string name = req.matches[1];
            if (!_tenant_can_access(name, res)) return;
            if (!_read_barrier(req, res)) return;
            try {
                auto col_ref = registry_.acquire(name);
                auto& col = *col_ref;
                auto j = json::parse(req.body);

                auto opts  = query_opts_from_json(j);
                std::vector<float> query;
                if (j.contains("vector") && !j["vector"].is_null())
                    query = j["vector"].get<std::vector<float>>();
                else if (opts.text.empty())
                    throw std::invalid_argument("search needs \"vector\" and/or \"text\"");

                RetrievalTrace trace;
                auto results = col.search(query, opts,
                    opts.include_trace ? &trace : nullptr);

                // Build response
                json r_arr = json::array();
                for (const auto& r : results)
                    r_arr.push_back(result_to_json(r));

                json resp = ok({{"results", r_arr},
                               {"count",   results.size()}});
                if (opts.include_trace)
                    resp["data"]["trace"] = trace_to_json(trace);

                // Always emit structured log; export OTel when trace requested
                if (opts.include_trace) {
                    if (otel_.enabled()) otel_.export_trace(trace);
                } else if (otel_.enabled()) {
                    otel_.export_event("info", "search",
                        {{"collection", name},
                         {"count", results.size()},
                         {"tenant", tls_current_tenant}});
                }

                _json_response(res, resp);
            } catch (const std::exception& e) {
                _error(res, e.what());
            }
        });

        // --------------------------------------------------------
        // GET /v1/collections/{name}/vectors/{id}?with_vector=true
        // One stored vector: payload, timestamps, edges, superseded_by.
        // --------------------------------------------------------
        svr_.Get(R"(/v1/collections/([^/]+)/vectors/(.+))",
            [this](const httplib::Request& req, httplib::Response& res) {
            _read(req, res, [&](ManagedCollection& col) {
                const std::string id = req.matches[2];
                bool with_vector = req.get_param_value("with_vector") != "false";
                auto v = col.get(id, with_vector);
                if (!v) throw std::out_of_range("Vector not found: " + id);
                return stored_vector_to_json(*v, with_vector);
            });
        });

        // --------------------------------------------------------
        // POST /v1/collections/{name}/get
        // Body: {"ids":["a","b"],"with_vector":false,"with_payload":true}
        // → {"vectors":[...], "missing":["b"]}  (vectors in request order)
        // --------------------------------------------------------
        svr_.Post(R"(/v1/collections/([^/]+)/get)",
            [this](const httplib::Request& req, httplib::Response& res) {
            _read(req, res, [&](ManagedCollection& col) {
                auto j = json::parse(req.body);
                bool with_vector  = j.value("with_vector", false);
                bool with_payload = j.value("with_payload", true);
                json vecs = json::array(), missing = json::array();
                for (const auto& id : j.at("ids").get<std::vector<std::string>>()) {
                    if (auto v = col.get(id, with_vector))
                        vecs.push_back(stored_vector_to_json(*v, with_vector, with_payload));
                    else
                        missing.push_back(id);
                }
                return json{{"vectors", vecs}, {"missing", missing}};
            });
        });

        // --------------------------------------------------------
        // POST /v1/collections/{name}/scroll
        // Body: {"limit":100,"cursor":null,"filter":{...},"as_of":...,
        //        "with_vector":false,"with_payload":true}
        // Pages through vectors in id order. Pass back next_cursor until
        // it is null.
        // --------------------------------------------------------
        svr_.Post(R"(/v1/collections/([^/]+)/scroll)",
            [this](const httplib::Request& req, httplib::Response& res) {
            _read(req, res, [&](ManagedCollection& col) {
                auto j = req.body.empty() ? json::object() : json::parse(req.body);
                QueryOptions opts;
                apply_filter_fields(j, opts);
                bool with_vector  = j.value("with_vector", false);
                bool with_payload = j.value("with_payload", true);
                auto page = col.scroll(_cursor(j), _limit(j, 100), opts, with_vector);
                json vecs = json::array();
                for (const auto& v : page.vectors)
                    vecs.push_back(stored_vector_to_json(v, with_vector, with_payload));
                return json{{"vectors", vecs}, {"count", page.vectors.size()},
                            {"next_cursor", page.next_cursor ? json(*page.next_cursor)
                                                             : json(nullptr)}};
            });
        });

        // --------------------------------------------------------
        // GET  /v1/collections/{name}/ids?limit=1000&cursor=...
        // POST /v1/collections/{name}/ids  {"limit":..,"cursor":..,"filter":..,"as_of":..}
        // → {"ids":[...], "next_cursor": "..." | null}
        // --------------------------------------------------------
        auto ids_handler = [this](const httplib::Request& req, httplib::Response& res) {
            _read(req, res, [&](ManagedCollection& col) {
                json j = json::object();
                if (req.method == "POST" && !req.body.empty()) j = json::parse(req.body);
                if (req.has_param("limit"))  j["limit"]  = std::stoul(req.get_param_value("limit"));
                if (req.has_param("cursor")) j["cursor"] = req.get_param_value("cursor");
                if (req.has_param("filter")) j["filter"] = req.get_param_value("filter");
                QueryOptions opts;
                apply_filter_fields(j, opts);
                auto page = col.scroll(_cursor(j), _limit(j, 1000), opts, false);
                json ids = json::array();
                for (const auto& v : page.vectors) ids.push_back(v.id);
                return json{{"ids", ids},
                            {"next_cursor", page.next_cursor ? json(*page.next_cursor)
                                                             : json(nullptr)}};
            });
        };
        svr_.Get(R"(/v1/collections/([^/]+)/ids)", ids_handler);
        svr_.Post(R"(/v1/collections/([^/]+)/ids)", ids_handler);

        // --------------------------------------------------------
        // POST /v1/collections/{name}/count  {"filter":{...},"as_of":...}
        // --------------------------------------------------------
        svr_.Post(R"(/v1/collections/([^/]+)/count)",
            [this](const httplib::Request& req, httplib::Response& res) {
            _read(req, res, [&](ManagedCollection& col) {
                auto j = req.body.empty() ? json::object() : json::parse(req.body);
                QueryOptions opts;
                apply_filter_fields(j, opts);
                return json{{"count", col.count(opts)}};
            });
        });

        // --------------------------------------------------------
        // Aliases — a second name for a collection; repointing one is
        // atomic (zero-downtime cut-over to a re-built collection).
        //   GET    /v1/aliases                 → {"aliases":[{"alias","collection"}]}
        //   PUT    /v1/aliases/{alias}  {"collection":"docs_v2"}
        //   DELETE /v1/aliases/{alias}
        // Every /v1/collections/{name}/... route accepts an alias.
        // --------------------------------------------------------
        svr_.Get("/v1/aliases", [this](const httplib::Request& req, httplib::Response& res) {
            ++metrics_.requests_total;
            if (!_auth(req, res)) return;
            json arr = json::array();
            for (const auto& [a, c] : registry_.aliases())
                if (_tenant_allows(a))
                    arr.push_back({{"alias", a}, {"collection", c}});
            _json_response(res, ok({{"aliases", arr}}));
        });

        svr_.Put(R"(/v1/aliases/([^/]+))",
            [this](const httplib::Request& req, httplib::Response& res) {
            ++metrics_.requests_total;
            if (!_auth(req, res)) return;
            if (_follower_redirect(res)) return;
            std::string alias = req.matches[1];
            try {
                validate_collection_name(alias, "alias");
                if (!_tenant_can_access(alias, res)) return;
                auto j = json::parse(req.body);
                auto target = j.at("collection").get<std::string>();
                if (!_tenant_can_access(target, res)) return;
                registry_.get(target);
                _write(res, "alias_put", alias, std::move(j));
            } catch (const std::exception& e) {
                _error(res, e.what());
            }
        });

        svr_.Delete(R"(/v1/aliases/([^/]+))",
            [this](const httplib::Request& req, httplib::Response& res) {
            ++metrics_.requests_total;
            if (!_auth(req, res)) return;
            if (_follower_redirect(res)) return;
            std::string alias = req.matches[1];
            if (!_tenant_can_access(alias, res)) return;
            try {
                if (!registry_.aliases().count(alias)) {
                    _error(res, "Alias not found: " + alias, 404);
                    return;
                }
                _write(res, "alias_delete", alias, json::object());
            } catch (const std::exception& e) {
                _error(res, e.what());
            }
        });

        // --------------------------------------------------------
        // POST /v1/collections/{name}/edges
        // Body: {"edges":[{"src":"v1","tgt":"v2","type":"CAUSES","weight":0.9}]}
        // --------------------------------------------------------
        svr_.Post(R"(/v1/collections/([^/]+)/edges)",
            [this](const httplib::Request& req, httplib::Response& res) {
            ++metrics_.requests_total;
            if (!_auth(req, res)) return;
            if (_follower_redirect(res)) return;
            std::string name = req.matches[1];
            if (!_tenant_can_access(name, res)) return;
            try {
                registry_.get(name);
                _write(res, "edges", name, json::parse(req.body), 201);
            } catch (const std::exception& e) {
                _error(res, e.what());
            }
        });

        // --------------------------------------------------------
        // PUT /v1/collections/{name}/temporal
        // Body: {"half_life_ms":86400000,"temporal_blend":0.3}
        // --------------------------------------------------------
        svr_.Put(R"(/v1/collections/([^/]+)/temporal)",
            [this](const httplib::Request& req, httplib::Response& res) {
            ++metrics_.requests_total;
            if (!_auth(req, res)) return;
            if (_follower_redirect(res)) return;
            std::string name = req.matches[1];
            if (!_tenant_can_access(name, res)) return;
            try {
                registry_.get(name);
                _write(res, "temporal", name, json::parse(req.body));
            } catch (const std::exception& e) {
                _error(res, e.what());
            }
        });

        // --------------------------------------------------------
        // POST /v1/collections/{name}/compact   Body (optional): {"wait":true}
        // GET  /v1/collections/{name}/compact   → status only
        //
        // Rebuilds the vector graph and payload/text indexes without the
        // tombstones left by deletes and re-embeds. The rebuild runs in the
        // background (reads and writes continue; writes made meanwhile are
        // replayed before the swap); "wait":true returns once it is done.
        // Compaction also starts by itself once tombstones reach
        // compaction_ratio x live vectors (collection create option).
        // --------------------------------------------------------
        svr_.Post(R"(/v1/collections/([^/]+)/compact)",
            [this](const httplib::Request& req, httplib::Response& res) {
            ++metrics_.requests_total;
            if (!_auth(req, res)) return;
            if (_follower_redirect(res)) return;
            std::string name = req.matches[1];
            if (!_tenant_can_access(name, res)) return;
            try {
                registry_.get(name);
                json body = req.body.empty() ? json::object() : json::parse(req.body);
                _write(res, "compact", name, std::move(body));
            } catch (const std::exception& e) {
                _error(res, e.what());
            }
        });
        svr_.Get(R"(/v1/collections/([^/]+)/compact)",
            [this](const httplib::Request& req, httplib::Response& res) {
            _read(req, res, [&](ManagedCollection& col) {
                return _compaction_json(col.compaction_status());
            });
        });

        // --------------------------------------------------------
        // POST /v1/collections/{name}/drift/start
        // Body: {"reembed_url":"http://...","new_dim":1536,"M":16,
        //        "ef_construction":200,"temporal":{...}}
        // new_dim is optional and must equal the collection dim.
        //
        // Starts a zero-downtime model migration on the named collection.
        // TideVec builds a shadow index in the background while the live
        // index continues serving queries. The caller supplies a
        // reembed_url — an HTTP endpoint that accepts:
        //   POST {"id":"v1","payload":{...}} → {"embedding":[...]}
        // and returns the new-model embedding for each vector.
        //
        // Returns immediately; poll GET /drift/status for progress.
        // --------------------------------------------------------
        svr_.Post(R"(/v1/collections/([^/]+)/drift/start)",
            [this](const httplib::Request& req, httplib::Response& res) {
            ++metrics_.requests_total;
            if (!_auth(req, res)) return;
            std::string name = req.matches[1];
            if (!_tenant_can_access(name, res)) return;
            name = registry_.resolve(name);
            if (raft_) {
                _error(res, "DriftBridge migrations are not replicated; "
                            "unsupported in cluster mode", 501);
                return;
            }
            try {
                auto col_ref = registry_.acquire(name);
                auto& col = *col_ref;
                auto j    = json::parse(req.body);

                std::string reembed_url =
                    j.value("reembed_url", "");
                if (reembed_url.empty()) {
                    _error(res, "reembed_url is required — "
                           "POST {\"id\":...,\"payload\":...} → {\"embedding\":[...]}");
                    return;
                }

                // SSRF protection
                if (!url_allowlist_.is_allowed(reembed_url)) {
                    _error(res, "reembed_url not allowed — must be a public "
                           "http(s) URL on the allowlist", 403);
                    return;
                }

                // Build new TVIndex config from request
                TVIndexConfig new_cfg;
                new_cfg.M               = j.value("M",               16);
                new_cfg.ef_construction = j.value("ef_construction", 200);
                if (j.contains("temporal"))
                    new_cfg.temporal = temporal_cfg_from_json(j["temporal"]);
                else
                    new_cfg.temporal = col.temporal_config();

                // Get current config for the old index
                TVIndexConfig old_cfg;
                old_cfg.temporal = col.temporal_config();

                // Check if a migration is already running
                {
                    std::lock_guard<std::mutex> lg(drift_mutex_);
                    if (drift_states_.count(name)) {
                        auto phase = drift_states_[name].bridge->phase();
                        if (phase == DriftPhase::MIGRATING ||
                            phase == DriftPhase::SWAPPING) {
                            _error(res, "Migration already in progress for: " + name);
                            return;
                        }
                        drift_states_.erase(name);
                    }
                }

                // Snapshot current vectors for migration
                auto live_vectors = col.snapshot_vectors();
                std::size_t total = live_vectors.size();
                std::size_t new_dim = j.value("new_dim", 0UL);
                std::size_t col_dim = 0;
                for (const auto& info : registry_.list())
                    if (info.name == name) col_dim = info.dim;
                if (new_dim > 0 && col_dim > 0 && new_dim != col_dim) {
                    _error(res, "DriftBridge swaps the new embeddings into the "
                           "same collection, so new_dim must equal the "
                           "collection dim (" + std::to_string(col_dim) +
                           "); create a new collection to change dimension");
                    return;
                }
                if (new_dim == 0) new_dim = col_dim;

                ParsedHttpUrl parsed_url;
                try {
                    parsed_url = parse_http_url(reembed_url);
                } catch (const std::exception& e) {
                    _error(res, std::string("Invalid reembed_url: ") + e.what());
                    return;
                }
                try {
                    pin_http_url(parsed_url,
                                 url_allowlist_.may_resolve_private(reembed_url));
                } catch (const ReembedUrlBlocked& e) {
                    _error(res, e.what(), 403);
                    return;
                } catch (const std::exception& e) {
                    _error(res, e.what());
                    return;
                }

                // Create bridge
                auto bridge = std::make_shared<DriftBridge>(old_cfg, new_cfg);
                auto* bridge_ptr = bridge.get();

                // Re-embed callback: HTTP POST to caller's endpoint
                ReEmbedFn re_embed = [parsed_url, new_dim](
                    const std::string& id,
                    const std::unordered_map<std::string,std::string>& payload)
                    -> std::vector<float>
                {
                    json body = {{"id", id}, {"payload", payload}};
                    std::string body_str = body.dump();
                    std::string resp_body;
                    if (!http_post_json(parsed_url, body_str, resp_body))
                        return {};

                    try {
                        auto rj = json::parse(resp_body);
                        if (!rj.contains("embedding")) return {};
                        auto emb = rj["embedding"].get<std::vector<float>>();
                        if (new_dim > 0 && emb.size() != new_dim) return {};
                        return emb;
                    } catch (...) { return {}; }
                };

                // on_complete — bulk-upsert shadow vectors into all shards
                auto on_complete = [this, name, bridge_ptr](CortexVector) {
                    _drift_apply_shadow(name, bridge_ptr);
                };

                {
                    std::lock_guard<std::mutex> lg(drift_mutex_);
                    drift_states_[name] = DriftState{bridge, re_embed};
                }

                bridge->start(live_vectors, re_embed, on_complete);

                _json_response(res, ok({
                    {"message",       "Migration started"},
                    {"collection",    name},
                    {"total_vectors", total},
                    {"status_url",
                        "/v1/collections/" + name + "/drift/status"},
                }));
            } catch (const std::exception& e) {
                _error(res, e.what());
            }
        });

        // --------------------------------------------------------
        // GET /v1/collections/{name}/drift/status
        // Returns current migration progress.
        // --------------------------------------------------------
        svr_.Get(R"(/v1/collections/([^/]+)/drift/status)",
            [this](const httplib::Request& req, httplib::Response& res) {
            ++metrics_.requests_total;
            if (!_auth(req, res)) return;
            std::string name = req.matches[1];
            if (!_tenant_can_access(name, res)) return;
            name = registry_.resolve(name);

            std::lock_guard<std::mutex> lg(drift_mutex_);
            auto it = drift_states_.find(name);
            if (it == drift_states_.end()) {
                _json_response(res, ok({
                    {"collection", name},
                    {"phase",      "IDLE"},
                    {"message",    "No migration in progress"},
                }));
                return;
            }

            auto prog  = it->second.bridge->progress();
            std::string phase_str;
            switch (prog.phase) {
                case DriftPhase::IDLE:      phase_str = "IDLE";      break;
                case DriftPhase::MIGRATING: phase_str = "MIGRATING"; break;
                case DriftPhase::SWAPPING:  phase_str = "SWAPPING";  break;
                case DriftPhase::COMPLETE:  phase_str = "COMPLETE";  break;
                case DriftPhase::FAILED:    phase_str = "FAILED";    break;
                default:                    phase_str = "UNKNOWN";   break;
            }

            _json_response(res, ok({
                {"collection",     name},
                {"phase",          phase_str},
                {"total_vectors",  prog.total_vectors},
                {"migrated",       prog.migrated},
                {"skipped",        prog.skipped},
                {"pct_complete",   prog.pct()},
                {"error",          prog.error},
            }));
        });

        // --------------------------------------------------------
        // POST /v1/collections/{name}/drift/abort
        // Aborts an in-progress migration, keeps the live index.
        // --------------------------------------------------------
        svr_.Post(R"(/v1/collections/([^/]+)/drift/abort)",
            [this](const httplib::Request& req, httplib::Response& res) {
            ++metrics_.requests_total;
            if (!_auth(req, res)) return;
            std::string name = req.matches[1];
            if (!_tenant_can_access(name, res)) return;
            name = registry_.resolve(name);

            std::lock_guard<std::mutex> lg(drift_mutex_);
            auto it = drift_states_.find(name);
            if (it == drift_states_.end()) {
                _error(res, "No migration in progress for: " + name);
                return;
            }
            it->second.bridge->abort();
            drift_states_.erase(it);
            _json_response(res, ok({{"message", "Migration aborted"}, {"collection", name}}));
        });

        // --------------------------------------------------------
        // POST /v1/admin/backup — trigger manual snapshot
        // --------------------------------------------------------
        svr_.Post("/v1/admin/backup", [this](const httplib::Request& req,
                                              httplib::Response& res) {
            ++metrics_.requests_total;
            if (!_auth(req, res)) return;
            if (!_require_admin(res)) return;
            try {
                auto name = backup_.snapshot_now();
                otel_.export_event("info", "Manual backup triggered",
                    {{"snapshot", name}});
                _json_response(res, ok({{"snapshot", name},
                    {"message", "Backup created"}}));
            } catch (const std::exception& e) {
                _error(res, e.what(), 500);
            }
        });

        // --------------------------------------------------------
        // GET /v1/admin/backups — list snapshots
        // --------------------------------------------------------
        svr_.Get("/v1/admin/backups", [this](const httplib::Request& req,
                                             httplib::Response& res) {
            ++metrics_.requests_total;
            if (!_auth(req, res)) return;
            if (!_require_admin(res)) return;
            auto snaps = backup_.list_snapshots();
            _json_response(res, ok({{"snapshots", snaps}}));
        });

        // --------------------------------------------------------
        // GET /v1/admin/backups/manifests — PITR manifest history
        // --------------------------------------------------------
        svr_.Get("/v1/admin/backups/manifests", [this](const httplib::Request& req,
                                                        httplib::Response& res) {
            ++metrics_.requests_total;
            if (!_auth(req, res)) return;
            if (!_require_admin(res)) return;
            json arr = json::array();
            for (const auto& m : backup_.list_manifests()) {
                json entry = {
                    {"snapshot", m.snapshot},
                    {"created_at", m.created_at},
                    {"data_dir", m.data_dir},
                };
                if (!m.s3_uri.empty()) entry["s3_uri"] = m.s3_uri;
                if (!m.gcs_uri.empty()) entry["gcs_uri"] = m.gcs_uri;
                arr.push_back(entry);
            }
            _json_response(res, ok({{"manifests", arr}}));
        });

        // --------------------------------------------------------
        // POST /v1/admin/restore — point-in-time recovery
        // Body: {"snapshot":"tidevec_123.tar.gz","confirm":true}
        // Replaces the data directory's contents and reloads every
        // collection in place; no restart needed.
        // --------------------------------------------------------
        svr_.Post("/v1/admin/restore", [this](const httplib::Request& req,
                                               httplib::Response& res) {
            ++metrics_.requests_total;
            if (!_auth(req, res)) return;
            if (!_require_admin(res)) return;
            if (raft_) {
                _error(res, "Restore is per node and would diverge the cluster; "
                            "unsupported in cluster mode", 501);
                return;
            }
            try {
                auto j = json::parse(req.body);
                if (!j.value("confirm", false)) {
                    _error(res, "Restore requires {\"confirm\": true}", 400);
                    return;
                }
                std::string snap = j.at("snapshot").get<std::string>();
                auto msg = backup_.restore_snapshot(snap,
                    [this](const std::function<void()>& swap) { _reload_around(swap); });
                otel_.export_event("warn", "Backup restored",
                    {{"snapshot", snap}});
                _json_response(res, ok({
                    {"snapshot", snap},
                    {"message", msg},
                    {"restart_required", false},
                }));
            } catch (const SnapshotNotFound& e) {
                _error(res, e.what(), 404);
            } catch (const std::invalid_argument& e) {
                _error(res, e.what(), 400);
            } catch (const std::exception& e) {
                _error(res, e.what(), 500);
            }
        });

        // --------------------------------------------------------
        // Raft RPCs (cluster mode only, authenticated by shared secret)
        // --------------------------------------------------------
        // With fault injection, RPCs from a "partitioned" sender fail as if
        // the network dropped them.
        auto partitioned_from = [this](consensus::NodeId sender, httplib::Response& res) {
            if (!fault_ || !(fault_->blocks_from(sender) || fault_->should_drop())) return false;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            res.status = 503;
            return true;
        };

        svr_.Post("/raft/append_entries", [this, partitioned_from](const httplib::Request& req,
                                                                   httplib::Response& res) {
            if (!_cluster_auth(req, res)) return;
            try {
                auto args = consensus::append_args_from_json(json::parse(req.body));
                if (partitioned_from(args.leader_id, res)) return;
                res.set_content(consensus::to_json(raft_->on_append_entries(args)).dump(),
                                "application/json");
            } catch (const std::exception& e) {
                _error(res, e.what());
            }
        });

        svr_.Post("/raft/request_vote", [this, partitioned_from](const httplib::Request& req,
                                                                 httplib::Response& res) {
            if (!_cluster_auth(req, res)) return;
            try {
                auto args = consensus::vote_args_from_json(json::parse(req.body));
                if (partitioned_from(args.candidate_id, res)) return;
                res.set_content(consensus::to_json(raft_->on_request_vote(args)).dump(),
                                "application/json");
            } catch (const std::exception& e) {
                _error(res, e.what());
            }
        });

        svr_.Post("/raft/install_snapshot", [this, partitioned_from](const httplib::Request& req,
                                                                     httplib::Response& res) {
            if (!_cluster_auth(req, res)) return;
            try {
                auto args = consensus::snapshot_args_from_json(json::parse(req.body));
                if (partitioned_from(args.leader_id, res)) return;
                res.set_content(consensus::to_json(raft_->on_install_snapshot(args)).dump(),
                                "application/json");
            } catch (const std::exception& e) {
                _error(res, e.what());
            }
        });

        // --------------------------------------------------------
        // POST /v1/admin/cluster/add_node     {"id":3,"url":"http://h3:6399"}
        // POST /v1/admin/cluster/remove_node  {"id":1}
        // Leader only (421 + leader hint elsewhere). The new node must be
        // running with --raft-join; it replicates as a learner until caught
        // up, then a CONFIG entry makes it a voter. One change at a time.
        // --------------------------------------------------------
        svr_.Post("/v1/admin/cluster/add_node", [this](const httplib::Request& req,
                                                       httplib::Response& res) {
            ++metrics_.requests_total;
            if (!_auth(req, res)) return;
            if (!_require_admin(res)) return;
            if (!raft_) { _error(res, "Not running in cluster mode", 400); return; }
            if (_follower_redirect(res)) return;
            try {
                auto j = json::parse(req.body);
                consensus::Member m{j.at("id").get<int>(), j.at("url").get<std::string>()};
                auto r = raft_->add_server(m, j.value("timeout_ms", 30000));
                if (!r.success) { _error(res, r.error, 409); return; }
                _json_response(res, ok({{"message", "Node added"}, {"id", m.id},
                    {"config_index", r.index}, {"members", consensus::to_json(raft_->members())}}));
            } catch (const std::exception& e) {
                _error(res, e.what());
            }
        });

        svr_.Post("/v1/admin/cluster/remove_node", [this](const httplib::Request& req,
                                                          httplib::Response& res) {
            ++metrics_.requests_total;
            if (!_auth(req, res)) return;
            if (!_require_admin(res)) return;
            if (!raft_) { _error(res, "Not running in cluster mode", 400); return; }
            if (_follower_redirect(res)) return;
            try {
                auto j = json::parse(req.body);
                int id = j.at("id").get<int>();
                auto r = raft_->remove_server(id, j.value("timeout_ms", 30000));
                if (!r.success) { _error(res, r.error, 409); return; }
                _json_response(res, ok({{"message", "Node removed"}, {"id", id},
                    {"config_index", r.index}, {"members", consensus::to_json(raft_->members())}}));
            } catch (const std::exception& e) {
                _error(res, e.what());
            }
        });

        // --------------------------------------------------------
        // GET/POST /v1/admin/fault — TEST ONLY, needs --enable-fault-injection
        // Body: {"partition":[ids]} (both directions) or "drop_to"/"drop_from",
        //       "drop_prob":0.1, "rpc_delay_ms":50, "fsync_delay_ms":200,
        //       "disk_full":true, "timer_scale":0.5, "clock_offset_ms":-5000,
        //       or {"clear":true}. Fields not given are left unchanged.
        // --------------------------------------------------------
        auto fault_state = [this] {
            return json{{"drop_to", fault_->blocked_to()}, {"drop_from", fault_->blocked_from()},
                        {"drop_prob", fault_->drop_prob.load()},
                        {"rpc_delay_ms", fault_->rpc_delay_ms.load()},
                        {"fsync_delay_ms", fault_->fsync_delay_ms.load()},
                        {"disk_full", fault_->disk_full.load()},
                        {"timer_scale", fault_->timer_scale()},
                        {"clock_offset_ms", fault_->clock_offset_ms.load()}};
        };
        svr_.Get("/v1/admin/fault", [this, fault_state](const httplib::Request& req,
                                                        httplib::Response& res) {
            if (!_auth(req, res)) return;
            if (!_require_admin(res)) return;
            if (!fault_) { _error(res, "Fault injection disabled", 404); return; }
            _json_response(res, ok(fault_state()));
        });
        svr_.Post("/v1/admin/fault", [this, fault_state](const httplib::Request& req,
                                                         httplib::Response& res) {
            if (!_auth(req, res)) return;
            if (!_require_admin(res)) return;
            if (!fault_) { _error(res, "Fault injection disabled", 404); return; }
            try {
                auto j = json::parse(req.body.empty() ? "{}" : req.body);
                if (j.value("clear", false)) fault_->clear();
                if (j.contains("partition")) {
                    auto ids = j["partition"].get<std::set<int>>();
                    fault_->set_blocked(ids, ids);
                }
                if (j.contains("drop_to") || j.contains("drop_from"))
                    fault_->set_blocked(j.value("drop_to", fault_->blocked_to()),
                                        j.value("drop_from", fault_->blocked_from()));
                if (j.contains("drop_prob"))      fault_->drop_prob = j["drop_prob"].get<double>();
                if (j.contains("rpc_delay_ms"))   fault_->rpc_delay_ms = j["rpc_delay_ms"].get<int>();
                if (j.contains("fsync_delay_ms")) fault_->fsync_delay_ms = j["fsync_delay_ms"].get<int>();
                if (j.contains("disk_full"))      fault_->disk_full = j["disk_full"].get<bool>();
                if (j.contains("timer_scale"))    fault_->set_timer_scale(j["timer_scale"].get<double>());
                if (j.contains("clock_offset_ms"))
                    fault_->clock_offset_ms = j["clock_offset_ms"].get<int64_t>();
                _json_response(res, ok(fault_state()));
            } catch (const std::exception& e) {
                _error(res, e.what());
            }
        });

        // --------------------------------------------------------
        // GET /v1/cluster/status
        // --------------------------------------------------------
        svr_.Get("/v1/cluster/status", [this](const httplib::Request& req,
                                              httplib::Response& res) {
            ++metrics_.requests_total;
            if (!_auth(req, res)) return;
            if (!_require_admin(res)) return;
            if (!raft_) {
                _json_response(res, ok({{"mode", "standalone"}}));
                return;
            }
            auto s = raft_->status();
            json leader_url = nullptr;
            if (s.leader_id >= 0) {
                auto url = raft_->address_of(s.leader_id);
                if (!url.empty()) leader_url = url;
            }
            _json_response(res, ok({
                {"mode",           "raft (experimental)"},
                {"node_id",        s.node_id},
                {"role",           consensus::role_str(s.role)},
                {"term",           s.term},
                {"leader_id",      s.leader_id},
                {"leader",         leader_url},
                {"commit_index",   s.commit_index},
                {"last_applied",   s.last_applied},
                {"last_log_index", s.last_log_index},
                {"snapshot_index", s.snapshot_index},
                {"snapshot_term",  s.snapshot_term},
                {"members",        consensus::to_json(s.members)},
                {"config_index",   s.config_index},
                {"config_pending", s.config_pending},
                {"learners",       s.learners},
                {"peers",          cfg_.raft_peers},
                {"fault_injection", fault_ != nullptr},
            }));
        });

        // --------------------------------------------------------
        // Catch-all 404
        // --------------------------------------------------------
        svr_.set_error_handler([this](const httplib::Request&,
                                      httplib::Response& res) {
            if (res.status == 404 && res.body.empty())
                _json_response(res,
                    err("Endpoint not found", 404), 404);
        });

        svr_.set_pre_routing_handler([this](const httplib::Request& req,
                                            httplib::Response& res) {
            return _tenant_route_guard(req, res) ? httplib::Server::HandlerResponse::Unhandled
                                                 : httplib::Server::HandlerResponse::Handled;
        });

        // Request logger
        if (cfg_.log_requests) {
            svr_.set_logger([](const httplib::Request& req,
                               const httplib::Response& res) {
                std::cout << "[" << req.method << "] " << req.path
                          << " → " << res.status << "\n";
            });
        }
    }

    Config              cfg_;
    std::atomic<bool>   stopping_{false};
    CollectionRegistry  registry_;
    ServerMetrics       metrics_;
    std::unique_ptr<httplib::Server> svr_holder_ = _make_http_server(cfg_);  // SSLServer with TLS
    httplib::Server&    svr_ = *svr_holder_;
    RateLimiter         rate_limiter_;
    UrlAllowlist        url_allowlist_;
    TenantStore         tenant_store_;
    OtelExporter        otel_;
    BackupManager       backup_;

    inline static thread_local std::string tls_current_tenant;

    // DriftBridge state — one per collection currently migrating
    struct DriftState {
        std::shared_ptr<DriftBridge> bridge;
        ReEmbedFn                      re_embed;
    };

    std::mutex drift_mutex_;
    std::unordered_map<std::string, DriftState> drift_states_;

    // Cluster mode. raft_ is declared last so it is destroyed (and its
    // apply thread joined) before the registry it writes into.
    struct RaftOutcome {
        json        data;
        std::string error;
        consensus::Term term = 0;        // term of the entry that was applied
        bool indeterminate   = false;    // covered by a snapshot install
    };
    std::shared_ptr<consensus::FaultInjector> fault_ =
        cfg_.enable_fault_injection ? std::make_shared<consensus::FaultInjector>() : nullptr;
    std::mutex              raft_apply_mu_;
    std::condition_variable raft_apply_cv_;
    std::unordered_map<consensus::Index, std::optional<RaftOutcome>> raft_waiting_;
    std::unique_ptr<consensus::RaftNode> raft_;
};

} // namespace tidevec
