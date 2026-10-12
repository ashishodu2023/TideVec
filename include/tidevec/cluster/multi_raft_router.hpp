#pragma once
// ============================================================
// TideVec — multi-Raft shard router (EXPERIMENTAL)
//
// Stateless HTTP front end over several independent Raft groups
// (each a normal 3- or 5-node `tidevec-server --raft-peers ...`
// cluster). Every vector id hashes (shard_hash) into a range of the
// 64-bit space owned by exactly one group; the placement map lists
// the ranges and each group's replica URLs.
//
//   writes   upsert / delete-by-id are split by owner and forwarded
//            to each group's leader (421 redirects followed, retries
//            are safe because both are idempotent)
//   reads    GET vectors/{id} goes to the owner; search and count
//            scatter to every group and merge (top_k by score, sums)
//   DDL      create / drop collection are broadcast to every group
//
// Not atomic across groups: a multi-group upsert can partially
// succeed (the response lists per-group results; retrying is safe).
// Rebalancing (moving ranges between groups) is design-only, see
// include/tidevec/cluster/DESIGN_MULTI_RAFT.md.
// ============================================================

#include <tidevec/api/httplib.h>
#include <tidevec/api/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <future>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace tidevec {
namespace cluster {

using rjson = nlohmann::json;

inline uint64_t fnv1a64(std::string_view s) {
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ULL; }
    return h;
}

// FNV-1a barely changes the high bits for ids that differ only at the
// end ("doc-1", "doc-2"); the murmur3 finalizer spreads them across the
// ranges.
inline uint64_t shard_hash(std::string_view id) {
    uint64_t h = fnv1a64(id);
    h ^= h >> 33; h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}

struct ShardGroup {
    std::string id;
    std::vector<std::string> replicas;   // base URLs, e.g. http://10.0.0.1:8080
    uint64_t lo = 0, hi = 0;             // inclusive hash range
};

inline std::string hex64(uint64_t v) {
    char buf[19];
    std::snprintf(buf, sizeof buf, "0x%016llx", static_cast<unsigned long long>(v));
    return buf;
}

inline uint64_t parse_u64(const rjson& j) {
    if (j.is_number_unsigned()) return j.get<uint64_t>();
    if (j.is_number_integer() && j.get<int64_t>() >= 0) return static_cast<uint64_t>(j.get<int64_t>());
    if (j.is_string()) return std::stoull(j.get<std::string>(), nullptr, 0);
    throw std::invalid_argument("range bound must be a non-negative integer or hex string");
}

// Placement map: which group owns which hash range. Groups must cover
// [0, 2^64-1] with contiguous, non-overlapping ranges, in order.
struct Placement {
    uint64_t version = 1;
    std::vector<ShardGroup> groups;

    // Equal split of the hash space over the given replica sets.
    static Placement uniform(const std::vector<std::vector<std::string>>& replica_sets) {
        if (replica_sets.empty()) throw std::invalid_argument("placement needs at least one group");
        Placement p;
        const uint64_t n = replica_sets.size();
        const uint64_t step = UINT64_MAX / n;
        for (uint64_t i = 0; i < n; ++i) {
            ShardGroup g;
            g.id = "g" + std::to_string(i);
            g.replicas = replica_sets[i];
            g.lo = i == 0 ? 0 : i * step + 1;
            g.hi = i + 1 == n ? UINT64_MAX : (i + 1) * step;
            p.groups.push_back(std::move(g));
        }
        p.validate();
        return p;
    }

    // {"version":1,"groups":[{"id":"g0","replicas":[...],"range":["0x0","0x7fff..."]}, ...]}
    // "range" may be omitted on every group for an equal split.
    static Placement from_json(const rjson& j) {
        const auto& gs = j.at("groups");
        bool any_range = false;
        for (const auto& g : gs) any_range |= g.contains("range");
        if (!any_range) {
            std::vector<std::vector<std::string>> sets;
            for (const auto& g : gs) sets.push_back(g.at("replicas").get<std::vector<std::string>>());
            Placement p = uniform(sets);
            for (std::size_t i = 0; i < gs.size(); ++i)
                if (gs[i].contains("id")) p.groups[i].id = gs[i]["id"].get<std::string>();
            p.version = j.value("version", uint64_t{1});
            return p;
        }
        Placement p;
        p.version = j.value("version", uint64_t{1});
        for (const auto& g : gs) {
            ShardGroup s;
            s.id = g.at("id").get<std::string>();
            s.replicas = g.at("replicas").get<std::vector<std::string>>();
            const auto& r = g.at("range");
            s.lo = parse_u64(r.at(0));
            s.hi = parse_u64(r.at(1));
            p.groups.push_back(std::move(s));
        }
        p.validate();
        return p;
    }

    rjson to_json() const {
        rjson gs = rjson::array();
        for (const auto& g : groups)
            gs.push_back({{"id", g.id}, {"replicas", g.replicas},
                          {"range", {hex64(g.lo), hex64(g.hi)}}});
        return {{"version", version}, {"hash", "fmix64(fnv1a64(id))"}, {"groups", gs}};
    }

    void validate() const {
        if (groups.empty()) throw std::invalid_argument("placement needs at least one group");
        uint64_t next = 0;
        for (std::size_t i = 0; i < groups.size(); ++i) {
            const auto& g = groups[i];
            if (g.replicas.empty())
                throw std::invalid_argument("group " + g.id + " has no replicas");
            if (g.lo != next || g.hi < g.lo)
                throw std::invalid_argument("group " + g.id + ": ranges must be contiguous, "
                                            "ordered and start at 0");
            if (i + 1 == groups.size()) {
                if (g.hi != UINT64_MAX)
                    throw std::invalid_argument("last group must end at 0xffffffffffffffff");
            } else {
                next = g.hi + 1;
            }
        }
    }

    std::size_t owner(std::string_view id) const {
        const uint64_t h = shard_hash(id);
        auto it = std::lower_bound(groups.begin(), groups.end(), h,
            [](const ShardGroup& g, uint64_t v) { return g.hi < v; });
        return static_cast<std::size_t>(it - groups.begin());
    }
};

class MultiRaftRouter {
public:
    struct Config {
        std::string host = "0.0.0.0";
        int port = 8090;
        int threads = 16;
        int timeout_ms = 15000;    // per forwarded request
        Placement placement;
    };

    explicit MultiRaftRouter(Config cfg) : cfg_(std::move(cfg)), leaders_(cfg_.placement.groups.size()) {
        cfg_.placement.validate();
        _routes();
    }

    bool listen() {
        svr_.new_task_queue = [n = cfg_.threads] { return new httplib::ThreadPool(n); };
        return svr_.listen(cfg_.host, cfg_.port);
    }
    void stop() { svr_.stop(); }
    const Placement& placement() const { return cfg_.placement; }

    struct Reply { int status = 0; rjson body; };

    // Sends to the group's leader, following 421 hints and trying other
    // replicas on connection errors. Only for idempotent requests.
    Reply call_group(std::size_t g, const std::string& method, const std::string& path,
                     const rjson* body, const httplib::Headers& headers) {
        const auto& reps = cfg_.placement.groups[g].replicas;
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(cfg_.timeout_ms);
        std::string target = _leader(g);
        std::size_t rr = 0;
        Reply last{503, {{"status", "error"}, {"error", "group unavailable"}, {"code", 503}}};
        while (std::chrono::steady_clock::now() < deadline) {
            if (target.empty()) target = reps[rr++ % reps.size()];
            Reply r = _send(target, method, path, body, headers);
            if (r.status == 421) {
                std::string hint = r.body.is_object() && r.body.contains("leader") &&
                                   r.body["leader"].is_string() ? r.body["leader"].get<std::string>() : "";
                _set_leader(g, hint);
                target = hint;
                if (hint.empty()) std::this_thread::sleep_for(std::chrono::milliseconds(50));
                last = r;
                continue;
            }
            // 503 includes "Write not confirmed": retried, forwarded writes are idempotent.
            if (r.status == 0 || r.status == 503) {
                _set_leader(g, "");
                target.clear();
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                last = r.status ? r : Reply{503, {{"status", "error"},
                    {"error", "group " + cfg_.placement.groups[g].id + " unreachable"}, {"code", 503}}};
                continue;
            }
            _set_leader(g, target);
            return r;
        }
        return last;
    }

private:
    Config cfg_;
    httplib::Server svr_;
    std::mutex mu_;
    std::vector<std::string> leaders_;

    std::string _leader(std::size_t g) { std::lock_guard l(mu_); return leaders_[g]; }
    void _set_leader(std::size_t g, const std::string& url) { std::lock_guard l(mu_); leaders_[g] = url; }

    Reply _send(const std::string& base, const std::string& method, const std::string& path,
                const rjson* body, const httplib::Headers& headers) {
        httplib::Client cli(base);
        cli.set_connection_timeout(1, 0);
        cli.set_path_encode(false);   // paths are encoded by the router
        const int t = cfg_.timeout_ms;
        cli.set_read_timeout(t / 1000, (t % 1000) * 1000);
        cli.set_write_timeout(t / 1000, (t % 1000) * 1000);
        const std::string data = body ? body->dump() : "";
        auto res = [&] {
            if (method == "GET")    return cli.Get(path, headers);
            if (method == "DELETE") return cli.Delete(path, headers);
            if (method == "PUT")    return cli.Put(path, headers, data, "application/json");
            return cli.Post(path, headers, data, "application/json");
        }();
        if (!res) return {0, nullptr};
        Reply r;
        r.status = res->status;
        try { r.body = res->body.empty() ? rjson() : rjson::parse(res->body); }
        catch (...) { r.body = res->body; }
        return r;
    }

    static httplib::Headers _fwd_headers(const httplib::Request& req) {
        httplib::Headers h;
        for (const char* k : {"Authorization", "X-API-Key", "X-TideVec-Consistency", "X-Tenant-ID"}) {
            auto it = req.headers.find(k);
            if (it != req.headers.end()) h.emplace(k, it->second);
        }
        return h;
    }

    static std::string _query(const httplib::Request& req) {
        std::string q;
        for (const auto& [k, v] : req.params)
            q += (q.empty() ? "?" : "&") + httplib::encode_query_component(k) + "=" +
                 httplib::encode_query_component(v);
        return q;
    }

    static void _reply(httplib::Response& res, int status, const rjson& body) {
        res.status = status;
        res.set_content(body.dump(), "application/json");
    }
    static void _error(httplib::Response& res, int status, const std::string& msg) {
        _reply(res, status, {{"status", "error"}, {"error", msg}, {"code", status}});
    }

    // Runs fn(g) for every group in parallel.
    std::vector<Reply> _all(const std::function<Reply(std::size_t)>& fn) {
        std::vector<std::future<Reply>> fs;
        for (std::size_t g = 0; g < cfg_.placement.groups.size(); ++g)
            fs.push_back(std::async(std::launch::async, fn, g));
        std::vector<Reply> out;
        for (auto& f : fs) out.push_back(f.get());
        return out;
    }

    rjson _group_summary(const std::vector<Reply>& rs) const {
        rjson arr = rjson::array();
        for (std::size_t g = 0; g < rs.size(); ++g)
            arr.push_back({{"group", cfg_.placement.groups[g].id}, {"status", rs[g].status},
                           {"body", rs[g].body}});
        return arr;
    }

    static std::string _new_id() {
        static thread_local std::mt19937_64 rng{std::random_device{}()};
        std::ostringstream s;
        s << std::hex << rng() << rng();
        return s.str();
    }

    void _routes() {
        svr_.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
            _reply(res, 200, {{"status", "ok"}, {"role", "router"},
                              {"groups", cfg_.placement.groups.size()}, {"experimental", true}});
        });

        svr_.Get("/v1/cluster/placement", [this](const httplib::Request&, httplib::Response& res) {
            rjson p = cfg_.placement.to_json();
            for (std::size_t g = 0; g < p["groups"].size(); ++g)
                p["groups"][g]["leader"] = _leader(g);
            _reply(res, 200, {{"status", "ok"}, {"data", p}});
        });

        svr_.Get("/v1/cluster/status", [this](const httplib::Request& req, httplib::Response& res) {
            auto h = _fwd_headers(req);
            auto rs = _all([&](std::size_t g) { return call_group(g, "GET", "/v1/cluster/status", nullptr, h); });
            _reply(res, 200, {{"status", "ok"}, {"data", {{"groups", _group_summary(rs)}}}});
        });

        svr_.Get("/v1/collections", [this](const httplib::Request& req, httplib::Response& res) {
            auto r = call_group(0, "GET", "/v1/collections" + _query(req), nullptr, _fwd_headers(req));
            _reply(res, r.status, r.body);
        });

        // DDL: broadcast; "already exists" / "not found" on retry count as done.
        svr_.Post("/v1/collections", [this](const httplib::Request& req, httplib::Response& res) {
            rjson body;
            try { body = rjson::parse(req.body); } catch (const std::exception& e) { return _error(res, 400, e.what()); }
            auto h = _fwd_headers(req);
            auto rs = _all([&](std::size_t g) { return call_group(g, "POST", "/v1/collections", &body, h); });
            bool ok = true, all_exist = true;
            for (auto& r : rs) {
                bool exists = r.status >= 400 && r.body.is_object() &&
                              r.body.value("error", "").find("already exists") != std::string::npos;
                ok &= r.status == 201 || r.status == 200 || exists;
                all_exist &= exists;
            }
            if (ok && all_exist) return _reply(res, 409, rs[0].body);
            if (!ok) return _reply(res, 502, {{"status", "error"}, {"code", 502},
                {"error", "collection not created on every shard group (retry is safe)"},
                {"groups", _group_summary(rs)}});
            _reply(res, 201, {{"status", "ok"}, {"data", {{"name", body.value("name", "")},
                              {"shard_groups", rs.size()}}}});
        });

        svr_.Delete(R"(/v1/collections/([^/]+)$)", [this](const httplib::Request& req, httplib::Response& res) {
            std::string path = "/v1/collections/" + std::string(req.matches[1]);
            auto h = _fwd_headers(req);
            auto rs = _all([&](std::size_t g) { return call_group(g, "DELETE", path, nullptr, h); });
            bool any_ok = false, ok = true;
            for (auto& r : rs) { any_ok |= r.status == 200; ok &= r.status == 200 || r.status == 404; }
            if (!any_ok && ok) return _reply(res, 404, rs[0].body);
            if (!ok) return _reply(res, 502, {{"status", "error"}, {"code", 502},
                {"error", "collection not dropped on every shard group (retry is safe)"},
                {"groups", _group_summary(rs)}});
            _reply(res, 200, {{"status", "ok"}, {"data", {{"message", "Collection deleted"}}}});
        });

        svr_.Get(R"(/v1/collections/([^/]+)$)", [this](const httplib::Request& req, httplib::Response& res) {
            const std::string name = req.matches[1];
            auto h = _fwd_headers(req);
            auto q = _query(req);
            auto rs = _all([&](std::size_t g) {
                return call_group(g, "GET", "/v1/collections/" + name + q, nullptr, h); });
            for (auto& r : rs) if (r.status != 200) return _reply(res, r.status, r.body);
            rjson out = rs[0].body;
            if (out.contains("data") && out["data"].is_object()) {
                uint64_t total = 0;
                for (auto& r : rs)
                    if (r.body["data"].contains("count") && r.body["data"]["count"].is_number())
                        total += r.body["data"]["count"].get<uint64_t>();
                out["data"]["count"] = total;
                out["data"]["shard_groups"] = rs.size();
            }
            _reply(res, 200, out);
        });

        svr_.Post(R"(/v1/collections/([^/]+)/upsert)", [this](const httplib::Request& req, httplib::Response& res) {
            const std::string name = req.matches[1];
            rjson body;
            try { body = rjson::parse(req.body); } catch (const std::exception& e) { return _error(res, 400, e.what()); }
            rjson items = body.contains("vectors") ? body["vectors"] : rjson::array({body});
            if (!items.is_array()) return _error(res, 400, "\"vectors\" must be an array");
            std::vector<rjson> per(cfg_.placement.groups.size(), rjson::array());
            for (auto& v : items) {
                if (!v.is_object()) return _error(res, 400, "each vector must be an object");
                if (!v.contains("id") || !v["id"].is_string() || v["id"].get<std::string>().empty())
                    v["id"] = _new_id();
                per[cfg_.placement.owner(v["id"].get<std::string>())].push_back(v);
            }
            _split_write(req, res, "/v1/collections/" + name + "/upsert", body, per, "vectors",
                         [](const rjson& data, rjson& acc) {
                acc["inserted"] = acc.value("inserted", uint64_t{0}) + data.value("inserted", uint64_t{0});
                for (const auto& id : data.value("ids", rjson::array())) acc["ids"].push_back(id);
            }, 201);
        });

        svr_.Post(R"(/v1/collections/([^/]+)/delete)", [this](const httplib::Request& req, httplib::Response& res) {
            const std::string name = req.matches[1];
            rjson body;
            try { body = rjson::parse(req.body); } catch (const std::exception& e) { return _error(res, 400, e.what()); }
            auto sum = [](const rjson& data, rjson& acc) {
                acc["deleted"] = acc.value("deleted", uint64_t{0}) + data.value("deleted", uint64_t{0});
            };
            std::vector<rjson> per(cfg_.placement.groups.size(), rjson::array());
            if (body.contains("filter")) {
                // A filter can match anywhere: every group gets the full request.
                for (auto& p : per) p = body.value("ids", rjson::array());
                return _split_write(req, res, "/v1/collections/" + name + "/delete", body, per, "ids", sum, 200, true);
            }
            if (!body.contains("ids") || !body["ids"].is_array())
                return _error(res, 400, "body needs \"ids\" or \"filter\"");
            for (const auto& id : body["ids"])
                per[cfg_.placement.owner(id.get<std::string>())].push_back(id);
            _split_write(req, res, "/v1/collections/" + name + "/delete", body, per, "ids", sum, 200);
        });

        svr_.Get(R"(/v1/collections/([^/]+)/vectors/(.+))", [this](const httplib::Request& req, httplib::Response& res) {
            const std::string name = req.matches[1], id = req.matches[2];
            auto g = cfg_.placement.owner(id);
            auto r = call_group(g, "GET", "/v1/collections/" + name + "/vectors/" +
                                httplib::encode_path_component(id) + _query(req), nullptr, _fwd_headers(req));
            _reply(res, r.status, r.body);
        });

        svr_.Post(R"(/v1/collections/([^/]+)/search)", [this](const httplib::Request& req, httplib::Response& res) {
            const std::string name = req.matches[1];
            rjson body;
            try { body = rjson::parse(req.body); } catch (const std::exception& e) { return _error(res, 400, e.what()); }
            const std::size_t top_k = body.value("top_k", std::size_t{10});
            auto h = _fwd_headers(req);
            auto path = "/v1/collections/" + name + "/search" + _query(req);
            auto rs = _all([&](std::size_t g) { return call_group(g, "POST", path, &body, h); });
            rjson merged = rjson::array();
            for (auto& r : rs) {
                if (r.status != 200) return _reply(res, r.status == 0 ? 503 : r.status, r.body);
                for (auto& x : r.body["data"]["results"]) merged.push_back(x);
            }
            std::stable_sort(merged.begin(), merged.end(), [](const rjson& a, const rjson& b) {
                return a.value("score", 0.0) > b.value("score", 0.0);
            });
            if (merged.size() > top_k) merged.erase(merged.begin() + static_cast<std::ptrdiff_t>(top_k), merged.end());
            _reply(res, 200, {{"status", "ok"}, {"data", {{"results", merged}, {"count", merged.size()},
                              {"shard_groups", rs.size()}}}});
        });

        svr_.Post(R"(/v1/collections/([^/]+)/count)", [this](const httplib::Request& req, httplib::Response& res) {
            const std::string name = req.matches[1];
            rjson body = req.body.empty() ? rjson::object() : rjson::parse(req.body, nullptr, false);
            if (body.is_discarded()) return _error(res, 400, "invalid JSON");
            auto h = _fwd_headers(req);
            auto path = "/v1/collections/" + name + "/count" + _query(req);
            auto rs = _all([&](std::size_t g) { return call_group(g, "POST", path, &body, h); });
            uint64_t total = 0;
            for (auto& r : rs) {
                if (r.status != 200) return _reply(res, r.status == 0 ? 503 : r.status, r.body);
                total += r.body["data"].value("count", uint64_t{0});
            }
            _reply(res, 200, {{"status", "ok"}, {"data", {{"count", total}}}});
        });

        svr_.set_error_handler([](const httplib::Request&, httplib::Response& res) {
            if (res.status == 404 && res.body.empty())
                _error(res, 404, "Endpoint not supported by tidevec-router (experimental): "
                                 "collections create/list/get/drop, upsert, delete, "
                                 "vectors/{id}, search, count");
        });
    }

    // Forwards per-group sub-requests (key -> per[g]) in parallel and
    // merges each group's "data" with merge(data, acc).
    void _split_write(const httplib::Request& req, httplib::Response& res, const std::string& path,
                      const rjson& body, const std::vector<rjson>& per, const char* key,
                      const std::function<void(const rjson&, rjson&)>& merge, int ok_status,
                      bool broadcast = false) {
        auto h = _fwd_headers(req);
        std::vector<Reply> rs(per.size());
        std::vector<std::future<Reply>> fs(per.size());
        for (std::size_t g = 0; g < per.size(); ++g) {
            if (!broadcast && per[g].empty()) continue;
            rjson sub = body;
            sub[key] = per[g];
            fs[g] = std::async(std::launch::async, [this, g, sub, &path, &h] {
                return call_group(g, "POST", path, &sub, h);
            });
        }
        rjson acc = {{"ids", rjson::array()}};
        bool ok = true;
        for (std::size_t g = 0; g < per.size(); ++g) {
            if (!fs[g].valid()) { rs[g] = {ok_status, nullptr}; continue; }
            rs[g] = fs[g].get();
            if (rs[g].status == 200 || rs[g].status == 201) {
                if (rs[g].body.is_object() && rs[g].body.contains("data")) merge(rs[g].body["data"], acc);
            } else {
                ok = false;
            }
        }
        if (acc["ids"].empty()) acc.erase("ids");
        if (!ok)
            return _reply(res, 503, {{"status", "error"}, {"code", 503},
                {"error", "write failed or unconfirmed on some shard groups; it is not atomic "
                          "across groups (retrying is safe)"},
                {"groups", _group_summary(rs)}});
        _reply(res, ok_status, {{"status", "ok"}, {"data", acc}});
    }
};

} // namespace cluster
} // namespace tidevec
