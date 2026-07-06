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
//     POST /v1/collections/{name}/delete     delete by id list
//     GET  /v1/collections/{name}/get/{id}   fetch one vector
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
#include <tidevec/observability/otel_exporter.hpp>
#include <tidevec/ops/backup_manager.hpp>

#include <string>
#include <atomic>
#include <chrono>
#include <mutex>
#include <unordered_map>
#include <sstream>
#include <iostream>

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
};

class RestServer {
public:
    using Config = RestServerConfig;

    explicit RestServer(Config cfg = RestServerConfig{})
        : cfg_(std::move(cfg))
        , registry_(_make_registry_config(cfg_))
        , rate_limiter_(RateLimiter::Config{cfg_.rate_limit_rps, cfg_.rate_limit_burst})
        , url_allowlist_(cfg_.reembed_allowed_hosts)
        , tenant_store_(cfg_.data_dir)
        , otel_(OtelConfig{cfg_.otel_endpoint, "tidevec", cfg_.otel_enabled})
        , backup_(_make_backup_config(cfg_))
    {
        if (!cfg_.api_key.empty())
            tenant_store_.ensure_default_tenant(cfg_.api_key);
        _register_routes();
    }

    // Blocking — call from main thread
    void listen() {
        std::cout << "\n╔══════════════════════════════════════╗\n";
        std::cout << "║   TideVec REST API v0.2.0           ║\n";
        std::cout << "║   http://" << cfg_.host << ":" << cfg_.port << "          ║\n";
        std::cout << "╚══════════════════════════════════════╝\n\n";
        std::cout << "Data directory: " << cfg_.data_dir << "\n";
        std::cout << "Ultra-durable:  " << (cfg_.ultra_durable ? "ON" : "OFF") << "\n";
        std::cout << "Segment store:  " << (cfg_.use_segment_store ? "ON" : "OFF") << "\n";
        std::cout << "Device:         " << cfg_.device << "\n";
        std::cout << "Auth required:  " << (cfg_.require_auth ? "YES" : "NO") << "\n";

        std::cout << "Recovering persisted collections...\n";
        registry_.load_and_recover();
        std::cout << "Ready — " << registry_.size() << " collection(s) loaded\n\n";

        backup_.start();

        svr_.new_task_queue = [this] {
            return new httplib::ThreadPool(cfg_.threads);
        };

#ifdef TIDEVEC_TLS
        if (!cfg_.tls_cert.empty() && !cfg_.tls_key.empty()) {
            std::cout << "TLS enabled: " << cfg_.tls_cert << "\n";
            svr_.listen(cfg_.host, cfg_.port, cfg_.tls_cert, cfg_.tls_key);
        } else {
            svr_.listen(cfg_.host, cfg_.port);
        }
#else
        svr_.listen(cfg_.host, cfg_.port);
#endif
    }

    void stop() {
        backup_.stop();
        svr_.stop();
    }

    CollectionRegistry& registry() { return registry_; }
    const ServerMetrics& metrics() const { return metrics_; }

private:
    static RegistryConfig _make_registry_config(const Config& cfg) {
        RegistryConfig rcfg;
        rcfg.data_dir          = cfg.data_dir;
        rcfg.ultra_durable     = cfg.ultra_durable;
        rcfg.use_segment_store = cfg.use_segment_store;
        rcfg.use_accel         = (cfg.device == "gpu" || cfg.device == "tpu" ||
                                  cfg.device == "auto");
        rcfg.device            = _parse_device(cfg.device);
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

    std::string _client_key(const httplib::Request& req) const {
        auto it = req.headers.find("X-Api-Key");
        if (it != req.headers.end()) return it->second;
        return req.remote_addr;
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
        // Legacy single-key mode
        if (!cfg_.require_auth && cfg_.api_key.empty()) return true;

        auto it = req.headers.find("X-Api-Key");
        if (it == req.headers.end() || it->second.empty()) {
            res.status = 401;
            res.set_content(err("Unauthorized — X-Api-Key required", 401).dump(),
                            "application/json");
            ++metrics_.errors_total;
            return false;
        }

        // Check tenant store first (multi-tenant)
        if (cfg_.multi_tenant || !cfg_.api_key.empty()) {
            std::string tenant = tenant_store_.authenticate(it->second);
            if (!tenant.empty()) {
                tls_current_tenant = tenant;
                return true;
            }
        }

        // Fallback: global API key
        if (!cfg_.api_key.empty() && it->second == cfg_.api_key) {
            tls_current_tenant = "default";
            return true;
        }

        res.status = 401;
        res.set_content(err("Unauthorized", 401).dump(), "application/json");
        ++metrics_.errors_total;
        return false;
    }

    bool _tenant_can_access(const std::string& collection,
                            httplib::Response& res) {
        if (tls_current_tenant.empty() || tls_current_tenant == "default")
            return true;
        if (!tenant_store_.can_access_collection(tls_current_tenant, collection)) {
            _error(res, "Tenant not authorized for collection: " + collection, 403);
            return false;
        }
        return true;
    }

    bool _tenant_check_collection_quota(httplib::Response& res) {
        if (tls_current_tenant.empty() || tls_current_tenant == "default")
            return true;
        if (!tenant_store_.check_collection_quota(tls_current_tenant,
                                                   registry_.size())) {
            _error(res, "Collection quota exceeded for tenant", 403);
            return false;
        }
        return true;
    }

    bool _tenant_check_vector_quota(std::size_t adding,
                                    httplib::Response& res) {
        if (tls_current_tenant.empty() || tls_current_tenant == "default")
            return true;
        if (!tenant_store_.check_vector_quota(tls_current_tenant,
                                               registry_.total_vectors(),
                                               adding)) {
            _error(res, "Vector quota exceeded for tenant", 403);
            return false;
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
        _json_response(res, err(msg, status), status);
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
                {"version", "0.2.0"},
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
                {"version", "0.2.0"},
                {"description", "Temporally-aware causal vector database"},
                {"features", json::array({
                    "TVIndex", "CausalEdge", "DriftBridge",
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
            try {
                auto j = json::parse(req.body);
                CollectionRegistry::CreateParams p;
                p.name        = j.at("name").get<std::string>();
                if (!_tenant_check_collection_quota(res)) return;
                if (!_tenant_can_access(p.name, res)) return;
                p.dim         = j.value("dim",         768UL);
                p.index_type  = j.value("index_type",  std::string("tvindex"));
                p.metric      = j.value("metric",      std::string("cosine"));
                p.n_shards    = j.value("n_shards",    4UL);
                if (p.n_shards == 0) p.n_shards = 4;  // guard against zero
                p.n_replicas  = j.value("n_replicas",  1);
                p.write_quorum= j.value("write_quorum",1);
                p.data_dir    = cfg_.data_dir;
                if (j.contains("temporal"))
                    p.temporal = temporal_cfg_from_json(j["temporal"]);

                registry_.create(p);
                otel_.export_event("info", "Collection created",
                    {{"name", p.name}, {"tenant", tls_current_tenant}});
                _json_response(res, ok({{"name", p.name},
                    {"message","Collection created"}}), 201);
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
            try {
                auto& col = registry_.get(name);
                _json_response(res, ok({
                    {"name",          name},
                    {"n_vectors",     col.total_vectors()},
                    {"n_shards",      col.n_shards()},
                    {"total_writes",  col.total_writes()},
                    {"total_queries", col.total_queries()},
                    {"backend",       col.backend_name()}
                }));
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
            std::string name = req.matches[1];
            if (registry_.drop(name))
                _json_response(res, ok({{"message","Deleted"}}));
            else
                _error(res, "Collection not found: " + name, 404);
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
            std::string name = req.matches[1];
            if (!_tenant_can_access(name, res)) return;
            try {
                auto& col = registry_.get(name);
                auto j = json::parse(req.body);

                std::size_t n_vectors = j.contains("vectors")
                    ? j["vectors"].size() : 1;
                if (!_tenant_check_vector_quota(n_vectors, res)) return;

                std::vector<std::string> inserted_ids;
                auto process = [&](const json& item) {
                    auto v = vector_from_json(item);
                    col.upsert(v);
                    inserted_ids.push_back(v.id);
                    ++metrics_.vectors_inserted;
                };

                if (j.contains("vectors")) {
                    for (const auto& item : j["vectors"]) process(item);
                } else {
                    process(j);  // single vector body
                }

                _json_response(res, ok({
                    {"inserted", inserted_ids.size()},
                    {"ids",      inserted_ids}
                }), 201);
            } catch (const std::exception& e) {
                _error(res, e.what());
            }
        });

        // --------------------------------------------------------
        // POST /v1/collections/{name}/delete
        // Body: {"ids":["v1","v2"]}
        // --------------------------------------------------------
        svr_.Post(R"(/v1/collections/([^/]+)/delete)",
            [this](const httplib::Request& req, httplib::Response& res) {
            ++metrics_.requests_total;
            if (!_auth(req, res)) return;
            std::string name = req.matches[1];
            try {
                auto& col = registry_.get(name);
                auto j = json::parse(req.body);
                auto ids = j.at("ids").get<std::vector<std::string>>();
                int removed = 0;
                for (const auto& id : ids)
                    if (col.remove(id)) ++removed;
                _json_response(res, ok({{"deleted", removed}}));
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
        //   "filter": "key=value",
        //   "include_trace": false,
        //   "metric": "cosine"
        // }
        // --------------------------------------------------------
        svr_.Post(R"(/v1/collections/([^/]+)/search)",
            [this](const httplib::Request& req, httplib::Response& res) {
            ++metrics_.requests_total;
            ++metrics_.search_requests;
            if (!_rate_limit(req, res)) return;
            if (!_auth(req, res)) return;
            std::string name = req.matches[1];
            if (!_tenant_can_access(name, res)) return;
            try {
                auto& col = registry_.get(name);
                auto j = json::parse(req.body);

                auto query = j.at("vector").get<std::vector<float>>();
                auto opts  = query_opts_from_json(j);

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
        // POST /v1/collections/{name}/edges
        // Body: {"edges":[{"src":"v1","tgt":"v2","type":"CAUSES","weight":0.9}]}
        // --------------------------------------------------------
        svr_.Post(R"(/v1/collections/([^/]+)/edges)",
            [this](const httplib::Request& req, httplib::Response& res) {
            ++metrics_.requests_total;
            if (!_auth(req, res)) return;
            std::string name = req.matches[1];
            try {
                auto& col = registry_.get(name);
                auto j = json::parse(req.body);
                int added = 0;
                for (const auto& e : j.at("edges")) {
                    col.add_edge(
                        e.at("src").get<std::string>(),
                        e.at("tgt").get<std::string>(),
                        edge_type_from_str(e.at("type").get<std::string>()),
                        e.value("weight", 1.0f)
                    );
                    ++added;
                }
                _json_response(res, ok({{"added", added}}), 201);
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
            std::string name = req.matches[1];
            try {
                auto& col = registry_.get(name);
                auto j = json::parse(req.body);
                auto cfg = temporal_cfg_from_json(j);
                col.set_temporal_config(cfg);
                _json_response(res, ok({{"message","Temporal config updated"}}));
            } catch (const std::exception& e) {
                _error(res, e.what());
            }
        });

        // --------------------------------------------------------
        // POST /v1/collections/{name}/drift/start
        // Body: {"new_dim":1536,"new_metric":"cosine","reembed_url":"http://..."}
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
            try {
                auto& col = registry_.get(name);
                auto j    = json::parse(req.body);

                std::string reembed_url =
                    j.value("reembed_url", "");
                if (reembed_url.empty()) {
                    _error(res, "reembed_url is required — "
                           "POST {\"id\":...,\"payload\":...} → {\"embedding\":[...]}");
                    return;
                }

                // SSRF protection
                if (UrlAllowlist::_is_blocked_url(reembed_url) ||
                    !url_allowlist_.is_allowed(reembed_url)) {
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
                    if (drift_bridges_.count(name)) {
                        auto phase = drift_bridges_[name]->phase();
                        if (phase == DriftPhase::MIGRATING ||
                            phase == DriftPhase::SWAPPING) {
                            _error(res, "Migration already in progress for: " + name);
                            return;
                        }
                        drift_bridges_.erase(name);
                    }
                }

                // Snapshot current vectors for migration
                auto live_vectors = col.snapshot_vectors();
                std::size_t total = live_vectors.size();

                // Create bridge
                auto bridge = std::make_unique<DriftBridge>(old_cfg, new_cfg);
                auto* bridge_ptr = bridge.get();

                {
                    std::lock_guard<std::mutex> lg(drift_mutex_);
                    drift_bridges_[name] = std::move(bridge);
                }

                // Re-embed callback: HTTP POST to caller's endpoint
                std::string url = reembed_url;
                ReEmbedFn re_embed = [url](
                    const std::string& id,
                    const std::unordered_map<std::string,std::string>& payload)
                    -> std::vector<float>
                {
                    // Build request JSON
                    json body = {{"id", id}, {"payload", payload}};
                    std::string body_str = body.dump();

                    httplib::Client cli(url);
                    auto r = cli.Post("/", body_str, "application/json");
                    if (!r || r->status != 200) return {};

                    try {
                        auto rj = json::parse(r->body);
                        if (!rj.contains("embedding")) return {};
                        return rj["embedding"].get<std::vector<float>>();
                    } catch (...) { return {}; }
                };

                // on_complete callback — swap the live index
                auto on_complete = [this, name, &col](CortexVector) {
                    std::lock_guard<std::mutex> lg(drift_mutex_);
                    auto it = drift_bridges_.find(name);
                    if (it == drift_bridges_.end()) return;
                    auto shadow = it->second->take_shadow();
                    if (shadow) col.swap_index(std::move(shadow));
                };

                bridge_ptr->start(live_vectors, re_embed, on_complete);

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

            std::lock_guard<std::mutex> lg(drift_mutex_);
            auto it = drift_bridges_.find(name);
            if (it == drift_bridges_.end()) {
                _json_response(res, ok({
                    {"collection", name},
                    {"phase",      "IDLE"},
                    {"message",    "No migration in progress"},
                }));
                return;
            }

            auto prog  = it->second->progress();
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

            std::lock_guard<std::mutex> lg(drift_mutex_);
            auto it = drift_bridges_.find(name);
            if (it == drift_bridges_.end()) {
                _error(res, "No migration in progress for: " + name);
                return;
            }
            it->second->abort();
            drift_bridges_.erase(it);
            _json_response(res, ok({{"message", "Migration aborted"}, {"collection", name}}));
        });

        // --------------------------------------------------------
        // POST /v1/admin/backup — trigger manual snapshot
        // --------------------------------------------------------
        svr_.Post("/v1/admin/backup", [this](const httplib::Request& req,
                                              httplib::Response& res) {
            ++metrics_.requests_total;
            if (!_auth(req, res)) return;
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
        // --------------------------------------------------------
        svr_.Post("/v1/admin/restore", [this](const httplib::Request& req,
                                               httplib::Response& res) {
            ++metrics_.requests_total;
            if (!_auth(req, res)) return;
            try {
                auto j = json::parse(req.body);
                if (!j.value("confirm", false)) {
                    _error(res, "Restore requires {\"confirm\": true}", 400);
                    return;
                }
                std::string snap = j.at("snapshot").get<std::string>();
                auto msg = backup_.restore_snapshot(snap);
                otel_.export_event("warn", "Backup restored",
                    {{"snapshot", snap}});
                _json_response(res, ok({
                    {"snapshot", snap},
                    {"message", msg},
                    {"restart_required", true},
                }));
            } catch (const std::exception& e) {
                _error(res, e.what(), 500);
            }
        });

        // --------------------------------------------------------
        // Catch-all 404
        // --------------------------------------------------------
        svr_.set_error_handler([this](const httplib::Request&,
                                      httplib::Response& res) {
            if (res.status == 404)
                _json_response(res,
                    err("Endpoint not found", 404), 404);
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
    CollectionRegistry  registry_;
    ServerMetrics       metrics_;
    httplib::Server     svr_;
    RateLimiter         rate_limiter_;
    UrlAllowlist        url_allowlist_;
    TenantStore         tenant_store_;
    OtelExporter        otel_;
    BackupManager       backup_;

    inline static thread_local std::string tls_current_tenant;

    // DriftBridge state — one per collection currently migrating
    std::mutex drift_mutex_;
    std::unordered_map<std::string, std::unique_ptr<DriftBridge>> drift_bridges_;
};

} // namespace tidevec
