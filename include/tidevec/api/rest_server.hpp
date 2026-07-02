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
#include <tidevec/drift/drift_bridge.hpp>

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
};

class RestServer {
public:
    using Config = RestServerConfig;

    explicit RestServer(Config cfg = RestServerConfig{})
        : cfg_(std::move(cfg))
        , registry_(cfg_.data_dir)
    {
        _register_routes();
    }

    // Blocking — call from main thread
    void listen() {
        std::cout << "\n╔══════════════════════════════════════╗\n";
        std::cout << "║   TideVec REST API v0.1.1           ║\n";
        std::cout << "║   http://" << cfg_.host << ":" << cfg_.port << "          ║\n";
        std::cout << "╚══════════════════════════════════════╝\n\n";
        std::cout << "Data directory: " << cfg_.data_dir << "\n";

        // Recover persisted collections from WAL before accepting requests
        std::cout << "Recovering persisted collections...\n";
        registry_.load_and_recover();
        std::cout << "Ready — " << registry_.size() << " collection(s) loaded\n\n";

        std::cout << "Endpoints:\n";
        std::cout << "  GET  /health\n";
        std::cout << "  GET  /v1/info\n";
        std::cout << "  GET  /metrics\n";
        std::cout << "  GET  /v1/collections\n";
        std::cout << "  POST /v1/collections\n";
        std::cout << "  POST /v1/collections/{name}/upsert\n";
        std::cout << "  POST /v1/collections/{name}/search\n";
        std::cout << "  POST /v1/collections/{name}/edges\n";
        std::cout << "  DELETE /v1/collections/{name}\n\n";

        svr_.new_task_queue = [this] {
            return new httplib::ThreadPool(cfg_.threads);
        };
        svr_.listen(cfg_.host, cfg_.port);
    }

    void stop() { svr_.stop(); }

    CollectionRegistry& registry() { return registry_; }
    const ServerMetrics& metrics() const { return metrics_; }

private:
    // ---- middleware helpers ------------------------------------
    bool _auth(const httplib::Request& req, httplib::Response& res) {
        if (cfg_.api_key.empty()) return true;
        auto it = req.headers.find("X-Api-Key");
        if (it == req.headers.end() || it->second != cfg_.api_key) {
            res.status = 401;
            res.set_content(err("Unauthorized", 401).dump(), "application/json");
            ++metrics_.errors_total;
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
                {"version", "0.1.0"},
                {"collections", registry_.size()},
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
                {"version", "0.1.0"},
                {"description", "Temporally-aware causal vector database"},
                {"features", json::array({"TVIndex", "CausalEdge",
                    "DriftBridge", "AgentContext", "RetrievalTrace"})},
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
                    {"metric",     info.metric}
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
                    {"total_queries", col.total_queries()}
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
            if (!_auth(req, res)) return;
            std::string name = req.matches[1];
            try {
                auto& col = registry_.get(name);
                auto j = json::parse(req.body);

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
            if (!_auth(req, res)) return;
            std::string name = req.matches[1];
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

    // DriftBridge state — one per collection currently migrating
    std::mutex drift_mutex_;
    std::unordered_map<std::string, std::unique_ptr<DriftBridge>> drift_bridges_;
};

} // namespace tidevec
