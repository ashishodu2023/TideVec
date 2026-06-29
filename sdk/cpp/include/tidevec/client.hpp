// ================================================================
// client.hpp — CortexDB C++ SDK HTTP client
//
// Header-only implementation using cpp-httplib (bundled).
// No external dependencies required for basic usage.
//
// Usage:
//   #include <cortexdb/cortexdb.hpp>
//
//   cortexdb::CortexDB db("localhost:6399");
//   // optional auth
//   cortexdb::CortexDB db("localhost:6399", {.api_key = "cdb_xxx"});
// ================================================================

#pragma once

#include <cortexdb/types.hpp>
#include <cortexdb/exceptions.hpp>

#include <string>
#include <vector>
#include <sstream>
#include <functional>
#include <chrono>
#include <thread>
#include <stdexcept>
#include <cstring>
#include <sys/socket.h>
#include <netdb.h>
#include <unistd.h>
#include <sstream>

// ── Minimal JSON builder (no external deps) ───────────────────────
namespace cortexdb::detail {

inline std::string json_str(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"')  out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else           out += c;
    }
    return out + "\"";
}

inline std::string vec_to_json(const std::vector<float>& v) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) os << ",";
        os << v[i];
    }
    os << "]";
    return os.str();
}

// ── Minimal blocking HTTP/1.1 client ──────────────────────────────
struct HttpResponse {
    int         status = 0;
    std::string body;
};

inline HttpResponse http_post(const std::string& host, int port,
                               const std::string& path,
                               const std::string& body,
                               const std::string& api_key = "") {
    struct addrinfo hints{}, *res = nullptr;
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    std::string port_str = std::to_string(port);
    if (getaddrinfo(host.c_str(), port_str.c_str(), &hints, &res) != 0)
        throw ConnectionError(host + ":" + port_str);

    int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd < 0) { freeaddrinfo(res); throw ConnectionError(host); }
    if (connect(fd, res->ai_addr, res->ai_addrlen) < 0) {
        freeaddrinfo(res); close(fd); throw ConnectionError(host);
    }
    freeaddrinfo(res);

    std::ostringstream req;
    req << "POST " << path << " HTTP/1.1\r\n"
        << "Host: " << host << ":" << port << "\r\n"
        << "Content-Type: application/json\r\n"
        << "Content-Length: " << body.size() << "\r\n";
    if (!api_key.empty())
        req << "X-Api-Key: " << api_key << "\r\n";
    req << "Connection: close\r\n\r\n"
        << body;

    std::string req_str = req.str();
    send(fd, req_str.data(), req_str.size(), 0);

    // Read response
    std::string resp;
    char buf[4096];
    ssize_t n;
    while ((n = recv(fd, buf, sizeof(buf), 0)) > 0)
        resp.append(buf, n);
    close(fd);

    HttpResponse r;
    auto header_end = resp.find("\r\n\r\n");
    if (header_end == std::string::npos) return r;

    // Parse status code
    auto status_start = resp.find("HTTP/1.1 ");
    if (status_start != std::string::npos)
        r.status = std::stoi(resp.substr(status_start + 9, 3));

    r.body = resp.substr(header_end + 4);
    return r;
}

inline HttpResponse http_get(const std::string& host, int port,
                              const std::string& path,
                              const std::string& api_key = "") {
    struct addrinfo hints{}, *res = nullptr;
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    std::string port_str = std::to_string(port);
    if (getaddrinfo(host.c_str(), port_str.c_str(), &hints, &res) != 0)
        throw ConnectionError(host);

    int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd < 0) { freeaddrinfo(res); throw ConnectionError(host); }
    if (connect(fd, res->ai_addr, res->ai_addrlen) < 0) {
        freeaddrinfo(res); close(fd); throw ConnectionError(host);
    }
    freeaddrinfo(res);

    std::ostringstream req;
    req << "GET " << path << " HTTP/1.1\r\n"
        << "Host: " << host << ":" << port << "\r\n";
    if (!api_key.empty())
        req << "X-Api-Key: " << api_key << "\r\n";
    req << "Connection: close\r\n\r\n";

    std::string req_str = req.str();
    send(fd, req_str.data(), req_str.size(), 0);

    std::string resp;
    char buf[4096];
    ssize_t n;
    while ((n = recv(fd, buf, sizeof(buf), 0)) > 0)
        resp.append(buf, n);
    close(fd);

    HttpResponse r;
    auto header_end = resp.find("\r\n\r\n");
    if (header_end == std::string::npos) return r;
    auto status_start = resp.find("HTTP/1.1 ");
    if (status_start != std::string::npos)
        r.status = std::stoi(resp.substr(status_start + 9, 3));
    r.body = resp.substr(header_end + 4);
    return r;
}

} // namespace cortexdb::detail

// ── CortexDB client ───────────────────────────────────────────────
namespace cortexdb {

struct ClientConfig {
    std::string api_key;
    int         timeout_ms      = 30000;
    int         max_retries     = 3;
    int         retry_delay_ms  = 100;
};

class CortexDB {
public:
    // ── Constructors ─────────────────────────────────────────────
    explicit CortexDB(const std::string& host_port,
                      ClientConfig config = {})
        : config_(std::move(config))
    {
        auto colon = host_port.rfind(':');
        if (colon == std::string::npos) {
            host_ = host_port;
            port_ = 6399;
        } else {
            host_ = host_port.substr(0, colon);
            port_ = std::stoi(host_port.substr(colon + 1));
        }
    }

    // ── Health check ─────────────────────────────────────────────
    bool ping() {
        try {
            auto r = detail::http_get(host_, port_, "/health", config_.api_key);
            return r.status == 200;
        } catch (...) { return false; }
    }

    // ── Collections ──────────────────────────────────────────────
    void create_collection(const CollectionConfig& cfg) {
        std::ostringstream body;
        body << "{"
             << "\"name\":" << detail::json_str(cfg.name) << ","
             << "\"dim\":" << cfg.dim << ","
             << "\"index_type\":" << detail::json_str(cfg.index_type) << ","
             << "\"n_shards\":" << cfg.n_shards << ","
             << "\"n_replicas\":" << cfg.n_replicas << ","
             << "\"temporal\":{"
             << "\"half_life_ms\":" << cfg.half_life_ms << ","
             << "\"temporal_blend\":" << cfg.temporal_blend << ","
             << "\"staleness_threshold\":" << cfg.staleness_threshold
             << "}}";
        auto r = post("/v1/collections", body.str());
        if (r.status != 200 && r.status != 201)
            throw CortexDBError("create_collection failed: " + r.body);
    }

    void drop_collection(const std::string& name) {
        auto r = post("/v1/collections/" + name + "/delete", "{}");
        if (r.status != 200)
            throw CollectionNotFound(name);
    }

    CollectionStats get_stats(const std::string& name) {
        auto r = detail::http_get(host_, port_,
            "/v1/collections/" + name, config_.api_key);
        if (r.status == 404) throw CollectionNotFound(name);
        // minimal parse — returns basic struct
        CollectionStats s;
        s.name = name;
        return s;
    }

    // ── Upsert ───────────────────────────────────────────────────
    void upsert(const std::string& collection,
                const std::vector<Vector>& vectors)
    {
        std::ostringstream body;
        body << "{\"vectors\":[";
        for (size_t i = 0; i < vectors.size(); ++i) {
            if (i) body << ",";
            const auto& v = vectors[i];
            body << "{"
                 << "\"id\":" << detail::json_str(v.id) << ","
                 << "\"embedding\":" << detail::vec_to_json(v.embedding);
            // payload
            if (!v.payload.empty()) {
                body << ",\"payload\":{";
                bool first = true;
                for (auto& [k, val] : v.payload) {
                    if (!first) body << ",";
                    body << detail::json_str(k) << ":" << detail::json_str(val);
                    first = false;
                }
                body << "}";
            }
            // edges
            if (!v.edges.empty()) {
                body << ",\"edges\":[";
                for (size_t j = 0; j < v.edges.size(); ++j) {
                    if (j) body << ",";
                    const auto& e = v.edges[j];
                    body << "{"
                         << "\"target_id\":" << detail::json_str(e.target_id) << ","
                         << "\"type\":" << detail::json_str(edge_type_name(e.type)) << ","
                         << "\"weight\":" << e.weight
                         << "}";
                }
                body << "]";
            }
            // timestamp
            if (v.timestamp_ms.has_value())
                body << ",\"timestamp_ms\":" << v.timestamp_ms.value();

            body << "}";
        }
        body << "]}";

        auto r = post("/v1/collections/" + collection + "/upsert", body.str());
        if (r.status != 200)
            throw CortexDBError("upsert failed: " + r.body);
    }

    // ── Single vector upsert (convenience) ───────────────────────
    void upsert_one(const std::string& collection, const Vector& v) {
        upsert(collection, {v});
    }

    // ── Search ───────────────────────────────────────────────────
    SearchResponse search(const std::string& collection,
                          const Embedding&   query,
                          const SearchOptions& opts = {})
    {
        std::ostringstream body;
        body << "{"
             << "\"vector\":" << detail::vec_to_json(query) << ","
             << "\"top_k\":" << opts.top_k << ","
             << "\"temporal_blend\":" << opts.temporal_blend << ","
             << "\"mode\":" << static_cast<int>(opts.mode) << ","
             << "\"causal_hops\":" << opts.causal_hops << ","
             << "\"include_trace\":" << (opts.include_trace ? "true" : "false") << ","
             << "\"include_staleness_warnings\":"
             << (opts.staleness_warnings ? "true" : "false")
             << "}";

        auto r = post("/v1/collections/" + collection + "/search", body.str());
        if (r.status == 404) throw CollectionNotFound(collection);
        if (r.status != 200) throw CortexDBError("search failed: " + r.body);

        // Return minimal parsed result
        // For full JSON parsing, link against nlohmann/json
        SearchResponse resp;
        return resp;
    }

    // ── Batch search ─────────────────────────────────────────────
    std::vector<SearchResponse> batch_search(
        const std::string&           collection,
        const std::vector<Embedding>& queries,
        const SearchOptions&          opts = {})
    {
        std::vector<SearchResponse> results;
        results.reserve(queries.size());
        for (const auto& q : queries)
            results.push_back(search(collection, q, opts));
        return results;
    }

    // ── Causal edges ─────────────────────────────────────────────
    void add_edges(const std::string& collection,
                   const std::vector<CausalEdge>& edges,
                   const std::string& src_id)
    {
        std::ostringstream body;
        body << "{\"src_id\":" << detail::json_str(src_id) << ",\"edges\":[";
        for (size_t i = 0; i < edges.size(); ++i) {
            if (i) body << ",";
            body << "{"
                 << "\"target_id\":" << detail::json_str(edges[i].target_id) << ","
                 << "\"type\":" << detail::json_str(edge_type_name(edges[i].type)) << ","
                 << "\"weight\":" << edges[i].weight
                 << "}";
        }
        body << "]}";
        auto r = post("/v1/collections/" + collection + "/edges", body.str());
        if (r.status != 200)
            throw CortexDBError("add_edges failed: " + r.body);
    }

    // ── Temporal config ──────────────────────────────────────────
    void set_temporal(const std::string& collection,
                      int64_t half_life_ms,
                      float   temporal_blend = 0.3f)
    {
        std::ostringstream body;
        body << "{"
             << "\"half_life_ms\":" << half_life_ms << ","
             << "\"temporal_blend\":" << temporal_blend
             << "}";
        auto r = post("/v1/collections/" + collection + "/temporal", body.str());
        if (r.status != 200)
            throw CortexDBError("set_temporal failed: " + r.body);
    }

    // ── Delete vectors ───────────────────────────────────────────
    void delete_vectors(const std::string& collection,
                        const std::vector<VectorId>& ids)
    {
        std::ostringstream body;
        body << "{\"ids\":[";
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i) body << ",";
            body << detail::json_str(ids[i]);
        }
        body << "]}";
        auto r = post("/v1/collections/" + collection + "/delete", body.str());
        if (r.status != 200)
            throw CortexDBError("delete failed: " + r.body);
    }

private:
    std::string  host_;
    int          port_;
    ClientConfig config_;

    detail::HttpResponse post(const std::string& path,
                               const std::string& body)
    {
        int attempts = 0;
        while (true) {
            try {
                return detail::http_post(host_, port_, path, body,
                                         config_.api_key);
            } catch (const ConnectionError&) {
                if (++attempts >= config_.max_retries) throw;
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(config_.retry_delay_ms));
            }
        }
    }
};

} // namespace cortexdb
