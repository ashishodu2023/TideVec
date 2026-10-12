// ================================================================
// client.hpp — TideVec C++ SDK HTTP client
//
// Header-only, POSIX sockets, no external dependencies. Plain HTTP
// only: for a TLS endpoint, connect through a local TLS proxy
// (ClientConfig::tls = true is rejected rather than silently sending
// the API key in clear text).
//
// Usage:
//   #include <tidevec/tidevec.hpp>
//
//   tidevec::TideVec db("localhost:6399");
//   // optional auth
//   tidevec::TideVec db("localhost:6399", {.api_key = "tv_xxx"});
// ================================================================

#pragma once

#include <tidevec/types.hpp>
#include <tidevec/exceptions.hpp>

#include <string>
#include <string_view>
#include <vector>
#include <sstream>
#include <functional>
#include <chrono>
#include <thread>
#include <stdexcept>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <optional>
#include <cmath>
#include <limits>
#include <sys/socket.h>
#include <sys/time.h>
#include <netdb.h>
#include <unistd.h>

// ── Minimal JSON (no external deps) ───────────────────────────────
namespace tidevec::detail {

inline std::string json_str(const std::string& s) {
    static const char* hex = "0123456789abcdef";
    std::string out = "\"";
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            default:
                if (c < 0x20) {
                    out += "\\u00";
                    out += hex[c >> 4];
                    out += hex[c & 0xf];
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out + "\"";
}

inline std::string json_num(double x) {
    if (!std::isfinite(x)) throw TideVecError("cannot send NaN or infinity as JSON");
    std::ostringstream os;
    os.precision(std::numeric_limits<float>::max_digits10);
    os << x;
    return os.str();
}

inline std::string vec_to_json(const std::vector<float>& v) {
    std::string out = "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) out += ",";
        out += json_num(v[i]);
    }
    return out + "]";
}

struct JsonValue {
    enum class Type { Null, Bool, Number, String, Array, Object };
    Type type = Type::Null;
    bool        b = false;
    double      n = 0.0;
    std::string s;
    std::vector<JsonValue>   items;   // array elements or object values
    std::vector<std::string> keys;    // object keys (parallel to items)

    bool is_null()   const { return type == Type::Null; }
    bool is_array()  const { return type == Type::Array; }
    bool is_object() const { return type == Type::Object; }

    const JsonValue& operator[](std::string_view key) const {
        static const JsonValue null_value;
        if (type != Type::Object) return null_value;
        for (size_t i = 0; i < keys.size(); ++i)
            if (keys[i] == key) return items[i];
        return null_value;
    }
    std::string str(const std::string& def = "") const { return type == Type::String ? s : def; }
    double num(double def = 0.0) const { return type == Type::Number ? n : def; }
    bool boolean(bool def = false) const { return type == Type::Bool ? b : def; }
    std::vector<std::string> strings() const {
        std::vector<std::string> out;
        if (type == Type::Array)
            for (const auto& v : items) if (v.type == Type::String) out.push_back(v.s);
        return out;
    }
};

class JsonParser {
public:
    explicit JsonParser(std::string_view text) : t_(text) {}

    JsonValue parse() {
        auto v = value();
        ws();
        if (i_ != t_.size()) fail("trailing characters");
        return v;
    }

private:
    std::string_view t_;
    size_t i_ = 0;
    int depth_ = 0;

    [[noreturn]] void fail(const char* what) const {
        throw TideVecError(std::string("invalid JSON from server: ") + what);
    }
    void ws() { while (i_ < t_.size() && std::isspace(static_cast<unsigned char>(t_[i_]))) ++i_; }
    bool eat(char c) { ws(); if (i_ < t_.size() && t_[i_] == c) { ++i_; return true; } return false; }
    void expect(char c) { if (!eat(c)) fail("unexpected character"); }
    bool literal(std::string_view w) {
        if (t_.substr(i_, w.size()) != w) return false;
        i_ += w.size();
        return true;
    }

    JsonValue value() {
        if (++depth_ > 256) fail("nesting too deep");
        ws();
        if (i_ >= t_.size()) fail("unexpected end");
        JsonValue v;
        char c = t_[i_];
        if (c == '{') {
            ++i_;
            v.type = JsonValue::Type::Object;
            if (!eat('}')) {
                do {
                    ws();
                    v.keys.push_back(string());
                    expect(':');
                    v.items.push_back(value());
                } while (eat(','));
                expect('}');
            }
        } else if (c == '[') {
            ++i_;
            v.type = JsonValue::Type::Array;
            if (!eat(']')) {
                do { v.items.push_back(value()); } while (eat(','));
                expect(']');
            }
        } else if (c == '"') {
            v.type = JsonValue::Type::String;
            v.s = string();
        } else if (literal("true"))  { v.type = JsonValue::Type::Bool; v.b = true;
        } else if (literal("false")) { v.type = JsonValue::Type::Bool; v.b = false;
        } else if (literal("null"))  { v.type = JsonValue::Type::Null;
        } else {
            size_t start = i_;
            while (i_ < t_.size() && std::strchr("+-0123456789.eE", t_[i_])) ++i_;
            if (start == i_) fail("unexpected character");
            v.type = JsonValue::Type::Number;
            v.n = std::strtod(std::string(t_.substr(start, i_ - start)).c_str(), nullptr);
        }
        --depth_;
        return v;
    }

    static void utf8(std::string& out, uint32_t cp) {
        if (cp < 0x80) out += static_cast<char>(cp);
        else if (cp < 0x800) { out += static_cast<char>(0xC0 | (cp >> 6)); out += static_cast<char>(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) { out += static_cast<char>(0xE0 | (cp >> 12)); out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); out += static_cast<char>(0x80 | (cp & 0x3F)); }
        else { out += static_cast<char>(0xF0 | (cp >> 18)); out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F)); out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); out += static_cast<char>(0x80 | (cp & 0x3F)); }
    }

    uint32_t hex4() {
        if (i_ + 4 > t_.size()) fail("bad \\u escape");
        uint32_t cp = 0;
        for (int k = 0; k < 4; ++k) {
            char h = t_[i_++];
            cp <<= 4;
            if (h >= '0' && h <= '9') cp |= h - '0';
            else if (h >= 'a' && h <= 'f') cp |= h - 'a' + 10;
            else if (h >= 'A' && h <= 'F') cp |= h - 'A' + 10;
            else fail("bad \\u escape");
        }
        return cp;
    }

    std::string string() {
        if (i_ >= t_.size() || t_[i_] != '"') fail("expected string");
        ++i_;
        std::string out;
        while (i_ < t_.size() && t_[i_] != '"') {
            char c = t_[i_++];
            if (c != '\\') { out += c; continue; }
            if (i_ >= t_.size()) fail("bad escape");
            char e = t_[i_++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    uint32_t cp = hex4();
                    if (cp >= 0xD800 && cp < 0xDC00 && t_.substr(i_, 2) == "\\u") {
                        i_ += 2;
                        uint32_t lo = hex4();
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    utf8(out, cp);
                    break;
                }
                default: fail("bad escape");
            }
        }
        if (i_ >= t_.size()) fail("unterminated string");
        ++i_;
        return out;
    }
};

inline JsonValue parse_json(const std::string& text) { return JsonParser(text).parse(); }

// Percent-encodes one URL path segment (collection names, ids).
inline std::string url_segment(const std::string& s) {
    static const char* hex = "0123456789ABCDEF";
    std::string out;
    for (unsigned char c : s) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') out += static_cast<char>(c);
        else { out += '%'; out += hex[c >> 4]; out += hex[c & 0xf]; }
    }
    return out;
}

// ── Minimal blocking HTTP/1.1 client ──────────────────────────────
struct HttpResponse {
    int         status = 0;
    std::string body;

    // The server's {"error": "..."} message, or the raw body.
    std::string error_message() const {
        try {
            auto j = parse_json(body);
            if (auto m = j["error"].str(); !m.empty()) return m;
        } catch (...) {}
        return body;
    }
};

inline HttpResponse http_request(const std::string& method,
                                 const std::string& host, int port,
                                 const std::string& path,
                                 const std::string& body,
                                 const std::string& api_key,
                                 int timeout_ms) {
    struct addrinfo hints{}, *res = nullptr;
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    std::string port_str = std::to_string(port);
    if (getaddrinfo(host.c_str(), port_str.c_str(), &hints, &res) != 0)
        throw ConnectionError(host + ":" + port_str);

    int fd = -1;
    for (auto* ai = res; ai; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        if (timeout_ms > 0) {
            timeval tv{timeout_ms / 1000, (timeout_ms % 1000) * 1000};
            setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
            setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
        }
        if (connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    if (fd < 0) throw ConnectionError(host + ":" + port_str);

    std::ostringstream req;
    req << method << " " << path << " HTTP/1.1\r\n"
        << "Host: " << host << ":" << port << "\r\n";
    if (!body.empty() || method == "POST" || method == "PUT")
        req << "Content-Type: application/json\r\n"
            << "Content-Length: " << body.size() << "\r\n";
    if (!api_key.empty())
        req << "X-Api-Key: " << api_key << "\r\n";
    req << "Connection: close\r\n\r\n" << body;

    std::string req_str = req.str();
    size_t sent = 0;
    while (sent < req_str.size()) {
        ssize_t n = send(fd, req_str.data() + sent, req_str.size() - sent, MSG_NOSIGNAL);
        if (n <= 0) { close(fd); throw ConnectionError(host + ":" + port_str); }
        sent += static_cast<size_t>(n);
    }

    std::string resp;
    char buf[4096];
    ssize_t n;
    while ((n = recv(fd, buf, sizeof(buf), 0)) > 0)
        resp.append(buf, static_cast<size_t>(n));
    close(fd);

    HttpResponse r;
    auto header_end = resp.find("\r\n\r\n");
    if (header_end == std::string::npos || resp.compare(0, 5, "HTTP/") != 0)
        throw ConnectionError(host + ":" + port_str + " (incomplete response)");
    auto sp = resp.find(' ');
    r.status = std::atoi(resp.c_str() + sp + 1);
    r.body = resp.substr(header_end + 4);
    return r;
}

// Kept for source compatibility.
inline HttpResponse http_post(const std::string& host, int port, const std::string& path,
                              const std::string& body, const std::string& api_key = "") {
    return http_request("POST", host, port, path, body, api_key, 30000);
}
inline HttpResponse http_get(const std::string& host, int port, const std::string& path,
                             const std::string& api_key = "") {
    return http_request("GET", host, port, path, "", api_key, 30000);
}

} // namespace tidevec::detail

// ── TideVec client ───────────────────────────────────────────────
namespace tidevec {

struct ClientConfig {
    std::string api_key{};
    bool        tls             = false;  // not supported; see the header comment
    int         timeout_ms      = 30000;
    int         max_retries     = 3;
    int         retry_delay_ms  = 100;
};

class TideVec {
public:
    // ── Constructors ─────────────────────────────────────────────
    explicit TideVec(const std::string& host_port,
                      ClientConfig config = {})
        : config_(std::move(config))
    {
        if (config_.tls)
            throw TideVecError("ClientConfig::tls is not supported by the C++ SDK; "
                               "connect through a local TLS proxy");
        if (config_.api_key.find_first_of("\r\n") != std::string::npos)
            throw TideVecError("API key must not contain line breaks");
        auto colon = host_port.rfind(':');
        if (colon == std::string::npos) {
            host_ = host_port;
            port_ = 6399;
        } else {
            host_ = host_port.substr(0, colon);
            port_ = std::stoi(host_port.substr(colon + 1));
        }
        if (host_.find_first_of("\r\n /") != std::string::npos)
            throw TideVecError("invalid host: " + host_);
    }

    // ── Health check ─────────────────────────────────────────────
    bool ping() {
        try {
            return request("GET", "/health").status == 200;
        } catch (...) { return false; }
    }

    // ── Collections ──────────────────────────────────────────────
    void create_collection(const CollectionConfig& cfg) {
        std::ostringstream body;
        body << "{"
             << "\"name\":" << detail::json_str(cfg.name) << ","
             << "\"dim\":" << cfg.dim << ","
             << "\"index_type\":" << detail::json_str(cfg.index_type) << ","
             << "\"metric\":" << detail::json_str(cfg.metric) << ","
             << "\"n_shards\":" << cfg.n_shards << ","
             << "\"n_replicas\":" << cfg.n_replicas << ","
             << "\"temporal\":{"
             << "\"half_life_ms\":" << cfg.half_life_ms << ","
             << "\"staleness_threshold\":" << detail::json_num(cfg.staleness_threshold);
        if (cfg.temporal_blend)
            body << ",\"temporal_blend\":" << detail::json_num(*cfg.temporal_blend);
        body << "}";
        if (cfg.supersession) {
            const auto& s = *cfg.supersession;
            body << ",\"supersession\":{"
                 << "\"enabled\":" << (s.enabled ? "true" : "false") << ","
                 << "\"similarity_threshold\":" << detail::json_num(s.similarity_threshold) << ","
                 << "\"match_field\":" << detail::json_str(s.match_field) << ","
                 << "\"candidates\":" << s.candidates
                 << "}";
        }
        body << "}";
        auto r = request("POST", "/v1/collections", body.str());
        check(r, "create_collection");
    }

    void drop_collection(const std::string& name) {
        auto r = request("DELETE", "/v1/collections/" + detail::url_segment(name));
        check(r, "drop_collection", name);
    }

    CollectionStats get_stats(const std::string& name) {
        auto r = request("GET", "/v1/collections/" + detail::url_segment(name));
        check(r, "get_stats", name);
        auto d = detail::parse_json(r.body)["data"];
        CollectionStats s;
        s.name         = d["name"].str(name);
        s.vector_count = static_cast<int64_t>(d["n_vectors"].num());
        s.n_shards     = static_cast<int>(d["n_shards"].num());
        s.index_type   = d["index_type"].str();
        s.dim          = static_cast<int>(d["dim"].num());
        s.metric       = d["metric"].str();
        const auto& sup = d["supersession"];
        if (sup.is_object() && sup["enabled"].boolean()) {
            SupersessionConfig c;
            c.enabled              = true;
            c.similarity_threshold = static_cast<float>(sup["similarity_threshold"].num(c.similarity_threshold));
            c.match_field          = sup["match_field"].str();
            c.candidates           = static_cast<int>(sup["candidates"].num(c.candidates));
            s.supersession = c;
        }
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
            if (!v.edges.empty()) {
                body << ",\"edges\":[";
                for (size_t j = 0; j < v.edges.size(); ++j) {
                    if (j) body << ",";
                    const auto& e = v.edges[j];
                    body << "{"
                         << "\"target_id\":" << detail::json_str(e.target_id) << ","
                         << "\"type\":" << detail::json_str(edge_type_name(e.type)) << ","
                         << "\"weight\":" << detail::json_num(e.weight)
                         << "}";
                }
                body << "]";
            }
            if (v.timestamp_ms.has_value())
                body << ",\"created_at\":" << v.timestamp_ms.value();
            body << "}";
        }
        body << "]}";

        auto r = request("POST", col_path(collection, "/upsert"), body.str());
        check(r, "upsert", collection);
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
             << "\"mode\":" << detail::json_str(mode_name(opts.mode)) << ","
             << "\"causal_hops\":" << opts.causal_hops << ","
             << "\"include_trace\":" << (opts.include_trace ? "true" : "false") << ","
             << "\"include_staleness_warnings\":"
             << (opts.staleness_warnings ? "true" : "false");
        if (opts.temporal_blend)
            body << ",\"temporal_blend\":" << detail::json_num(*opts.temporal_blend);
        if (opts.superseded_penalty)
            body << ",\"superseded_penalty\":" << detail::json_num(*opts.superseded_penalty);
        body << "}";

        auto r = request("POST", col_path(collection, "/search"), body.str());
        check(r, "search", collection);

        auto data = detail::parse_json(r.body)["data"];
        SearchResponse resp;
        for (const auto& h : data["results"].items) {
            SearchHit hit;
            hit.id                = h["id"].str();
            hit.score             = static_cast<float>(h["score"].num());
            hit.semantic_score    = static_cast<float>(h["vector_score"].num());
            hit.temporal_score    = static_cast<float>(h["temporal_score"].num());
            hit.created_at        = static_cast<int64_t>(h["created_at"].num());
            hit.staleness_warning = h["staleness_warning"].boolean();
            hit.staleness_reason  = h["staleness_reason"].str();
            hit.causal_neighbors  = h["causal_neighbors"].strings();
            hit.contradicted_by   = h["contradicted_by"].strings();
            hit.superseded_by     = h["superseded_by"].strings();
            const auto& p = h["payload"];
            for (size_t i = 0; i < p.keys.size(); ++i)
                hit.payload[p.keys[i]] = p.items[i].str();
            resp.hits.push_back(std::move(hit));
        }
        const auto& t = data["trace"];
        if (t.is_object()) {
            resp.trace.algorithm       = t["strategy"].str();
            resp.trace.latency_ms      = t["latency_ms"].num();
            resp.trace.recall_estimate = static_cast<float>(t["estimated_recall"].num());
            resp.trace.vectors_scanned = static_cast<int>(t["distance_computations"].num());
            resp.trace.query_id        = t["query_id"].str();
        }
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

    // ── Causal edges (src_id → each edge's target_id) ────────────
    void add_edges(const std::string& collection,
                   const std::vector<CausalEdge>& edges,
                   const std::string& src_id)
    {
        std::ostringstream body;
        body << "{\"edges\":[";
        for (size_t i = 0; i < edges.size(); ++i) {
            if (i) body << ",";
            body << "{"
                 << "\"src\":" << detail::json_str(src_id) << ","
                 << "\"tgt\":" << detail::json_str(edges[i].target_id) << ","
                 << "\"type\":" << detail::json_str(edge_type_name(edges[i].type)) << ","
                 << "\"weight\":" << detail::json_num(edges[i].weight)
                 << "}";
        }
        body << "]}";
        auto r = request("POST", col_path(collection, "/edges"), body.str());
        check(r, "add_edges", collection);
    }

    // ── Temporal config ──────────────────────────────────────────
    void set_temporal(const std::string& collection,
                      int64_t half_life_ms,
                      std::optional<float> temporal_blend = std::nullopt)
    {
        std::ostringstream body;
        body << "{\"half_life_ms\":" << half_life_ms;
        if (temporal_blend)
            body << ",\"temporal_blend\":" << detail::json_num(*temporal_blend);
        body << "}";
        auto r = request("PUT", col_path(collection, "/temporal"), body.str());
        check(r, "set_temporal", collection);
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
        auto r = request("POST", col_path(collection, "/delete"), body.str());
        check(r, "delete", collection);
    }

    // ── DriftBridge (zero-downtime model migration) ───────────────
    void start_drift(const std::string& collection,
                     const std::string& reembed_url)
    {
        std::ostringstream body;
        body << "{\"reembed_url\":" << detail::json_str(reembed_url) << "}";
        auto r = request("POST", col_path(collection, "/drift/start"), body.str());
        check(r, "start_drift", collection);
    }

    std::string drift_status(const std::string& collection)
    {
        auto r = request("GET", col_path(collection, "/drift/status"));
        check(r, "drift_status", collection);
        return r.body;
    }

    void abort_drift(const std::string& collection)
    {
        auto r = request("POST", col_path(collection, "/drift/abort"), "{}");
        check(r, "abort_drift", collection);
    }

    // ── Admin (backups; needs the admin API key) ─────────────────
    std::string trigger_backup()
    {
        auto r = request("POST", "/v1/admin/backup", "{}");
        check(r, "trigger_backup");
        return r.body;
    }

    std::string list_backups()
    {
        auto r = request("GET", "/v1/admin/backups");
        check(r, "list_backups");
        return r.body;
    }

    std::string restore_backup(const std::string& snapshot)
    {
        std::ostringstream body;
        body << "{\"snapshot\":" << detail::json_str(snapshot)
             << ",\"confirm\":true}";
        auto r = request("POST", "/v1/admin/restore", body.str());
        check(r, "restore_backup");
        return r.body;
    }

private:
    std::string  host_;
    int          port_;
    ClientConfig config_;

    static const char* mode_name(QueryMode m) {
        switch (m) {
            case QueryMode::CAUSAL_EXPAND:       return "causal_expand";
            case QueryMode::CONTRADICTION_CHECK: return "contradiction_check";
            default:                             return "vector_only";
        }
    }

    static std::string col_path(const std::string& collection, const char* suffix) {
        return "/v1/collections/" + detail::url_segment(collection) + suffix;
    }

    // Throws on any non-2xx status; 404 on a collection route means the
    // collection does not exist.
    static void check(const detail::HttpResponse& r, const char* op,
                      const std::string& collection = "") {
        if (r.status >= 200 && r.status < 300) return;
        if (r.status == 401) throw AuthError();
        auto msg = r.error_message();
        if (r.status == 404 && !collection.empty() &&
            msg.find("Collection not found") != std::string::npos)
            throw CollectionNotFound(collection);
        throw TideVecError(std::string(op) + " failed (HTTP " +
                           std::to_string(r.status) + "): " + msg);
    }

    // Retries connection failures and 429s.
    detail::HttpResponse request(const std::string& method, const std::string& path,
                                 const std::string& body = "")
    {
        int attempts = 0;
        while (true) {
            try {
                auto r = detail::http_request(method, host_, port_, path, body,
                                              config_.api_key, config_.timeout_ms);
                if (r.status != 429 || ++attempts >= config_.max_retries) return r;
            } catch (const ConnectionError&) {
                if (++attempts >= config_.max_retries) throw;
            }
            std::this_thread::sleep_for(
                std::chrono::milliseconds(config_.retry_delay_ms * attempts));
        }
    }
};

} // namespace tidevec
