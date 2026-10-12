#pragma once
// ================================================================
// OtelExporter — export RetrievalTrace to an OTLP/HTTP (JSON) endpoint
//
// Spans are queued and posted by a background thread, so a slow or
// unreachable collector never delays a search. The queue is bounded;
// spans that do not fit are dropped and counted.
// ================================================================

#include <tidevec/observability/retrieval_trace.hpp>
#include <tidevec/api/json_serializers.hpp>
#include <tidevec/api/httplib.h>

#include <string>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <random>
#include <iostream>

namespace tidevec {

struct OtelConfig {
    std::string endpoint = "";          // e.g. http://localhost:4318/v1/traces
    std::string service_name = "tidevec";
    bool enabled = false;               // JSON log lines on stdout
    std::size_t max_queue = 1024;
};

class OtelExporter {
public:
    explicit OtelExporter(OtelConfig cfg) : cfg_(std::move(cfg)) {
        if (cfg_.endpoint.empty()) return;
        _parse_endpoint();
        if (base_url_.empty()) {
            std::cerr << "[otel] Ignoring invalid --otel-endpoint '" << cfg_.endpoint
                      << "' (expected http(s)://host[:port][/path])\n";
            return;
        }
        worker_ = std::thread([this] { _run(); });
    }

    ~OtelExporter() {
        {
            std::lock_guard lock(q_mu_);
            stopping_ = true;
        }
        q_cv_.notify_all();
        if (worker_.joinable()) worker_.join();
    }

    OtelExporter(const OtelExporter&) = delete;
    OtelExporter& operator=(const OtelExporter&) = delete;

    void export_trace(const RetrievalTrace& trace) {
        if (cfg_.enabled) _log_structured(trace);
        if (!worker_.joinable()) return;
        auto payload = _otlp_payload(trace).dump();
        {
            std::lock_guard lock(q_mu_);
            if (queue_.size() >= cfg_.max_queue) { ++dropped_; return; }
            queue_.push_back(std::move(payload));
        }
        q_cv_.notify_one();
    }

    void export_event(const std::string& level,
                      const std::string& message,
                      const nlohmann::json& fields = {}) {
        if (!cfg_.enabled) return;
        auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        nlohmann::json log = {
            {"timestamp_ms", ts},
            {"level", level},
            {"service", cfg_.service_name},
            {"message", message},
        };
        if (!fields.empty()) log["fields"] = fields;
        std::lock_guard lock(mu_);
        std::cout << log.dump() << "\n";
    }

    // True when traces are logged or exported.
    bool enabled() const { return cfg_.enabled || worker_.joinable(); }

    // Waits until every queued span has been posted (or timeout).
    bool flush(std::chrono::milliseconds timeout = std::chrono::seconds(5)) {
        std::unique_lock lock(q_mu_);
        return idle_cv_.wait_for(lock, timeout, [this] { return queue_.empty() && !in_flight_; });
    }

    uint64_t exported() const { return exported_.load(); }
    uint64_t failed()   const { return failed_.load(); }
    uint64_t dropped()  const { return dropped_.load(); }

private:
    void _parse_endpoint() {
        const std::string& url = cfg_.endpoint;
        auto scheme_end = url.find("://");
        if (scheme_end == std::string::npos) return;
        std::string scheme = url.substr(0, scheme_end);
        if (scheme != "http" && scheme != "https") return;
        auto path_start = url.find('/', scheme_end + 3);
        std::string host_port = url.substr(scheme_end + 3,
            path_start == std::string::npos ? std::string::npos : path_start - scheme_end - 3);
        if (host_port.empty()) return;
        path_ = path_start == std::string::npos ? "" : url.substr(path_start);
        if (path_.empty() || path_ == "/") path_ = "/v1/traces";
        base_url_ = scheme + "://" + host_port;
    }

    std::string _random_hex(std::size_t bytes) {
        static const char* digits = "0123456789abcdef";
        std::lock_guard lock(rng_mu_);
        std::string out;
        bool all_zero = true;
        while (all_zero) {
            out.clear();
            for (std::size_t i = 0; i < bytes; ++i) {
                auto b = static_cast<unsigned>(rng_() & 0xff);
                if (b) all_zero = false;
                out += digits[b >> 4];
                out += digits[b & 0xf];
            }
        }
        return out;
    }

    nlohmann::json _otlp_payload(const RetrievalTrace& trace) {
        auto end_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        auto start_ns = end_ns - static_cast<int64_t>(trace.latency_ms * 1e6);
        auto str_attr = [](const char* k, const std::string& v) {
            return nlohmann::json{{"key", k}, {"value", {{"stringValue", v}}}};
        };
        auto int_attr = [](const char* k, int64_t v) {
            return nlohmann::json{{"key", k}, {"value", {{"intValue", std::to_string(v)}}}};
        };
        auto dbl_attr = [](const char* k, double v) {
            return nlohmann::json{{"key", k}, {"value", {{"doubleValue", v}}}};
        };
        nlohmann::json span = {
            {"traceId", _random_hex(16)},
            {"spanId",  _random_hex(8)},
            {"name",    "tidevec.search"},
            {"kind",    1},  // SPAN_KIND_INTERNAL
            {"startTimeUnixNano", std::to_string(start_ns)},
            {"endTimeUnixNano",   std::to_string(end_ns)},
            {"attributes", nlohmann::json::array({
                str_attr("tidevec.query_id", trace.query_id),
                str_attr("tidevec.strategy", trace.strategy),
                str_attr("tidevec.collection", trace.collection_name),
                int_attr("tidevec.ef_search", static_cast<int64_t>(trace.ef_search)),
                int_attr("tidevec.distance_computations",
                         static_cast<int64_t>(trace.distance_computations)),
                dbl_attr("tidevec.estimated_recall", trace.estimated_recall),
                dbl_attr("tidevec.latency_ms", trace.latency_ms),
            })},
            {"status", {{"code", 1}}},  // STATUS_CODE_OK
        };
        return {
            {"resourceSpans", nlohmann::json::array({{
                {"resource", {{"attributes", nlohmann::json::array({
                    str_attr("service.name", cfg_.service_name),
                })}}},
                {"scopeSpans", nlohmann::json::array({{
                    {"scope", {{"name", "tidevec"}}},
                    {"spans", nlohmann::json::array({span})},
                }})},
            }})},
        };
    }

    void _run() {
        httplib::Client cli(base_url_);
        cli.set_connection_timeout(2, 0);
        cli.set_read_timeout(5, 0);
        cli.set_write_timeout(5, 0);
        bool warned = false;
        std::unique_lock lock(q_mu_);
        while (true) {
            q_cv_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
            if (stopping_) break;
            auto body = std::move(queue_.front());
            queue_.pop_front();
            in_flight_ = true;
            lock.unlock();

            auto res = cli.Post(path_, body, "application/json");
            bool ok = res && res->status >= 200 && res->status < 300;
            if (ok) ++exported_;
            else {
                ++failed_;
                if (!warned) {
                    warned = true;
                    std::cerr << "[otel] Export to " << base_url_ << path_ << " failed: "
                              << (res ? "HTTP " + std::to_string(res->status)
                                      : httplib::to_string(res.error()))
                              << " (further failures are counted, not logged)\n";
                }
            }

            lock.lock();
            in_flight_ = false;
            if (queue_.empty()) idle_cv_.notify_all();
        }
        in_flight_ = false;
        idle_cv_.notify_all();
    }

    void _log_structured(const RetrievalTrace& trace) {
        nlohmann::json j = trace_to_json(trace);
        j["service"] = cfg_.service_name;
        j["type"]    = "retrieval_trace";
        auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        j["exported_at_ms"] = ts;
        std::lock_guard lock(mu_);
        std::cout << j.dump() << "\n";
    }

    OtelConfig cfg_;
    std::string base_url_;   // scheme://host[:port]
    std::string path_;       // /v1/traces
    std::mutex mu_;          // stdout

    std::mutex rng_mu_;
    std::mt19937_64 rng_{std::random_device{}()};

    std::mutex q_mu_;
    std::condition_variable q_cv_, idle_cv_;
    std::deque<std::string> queue_;
    bool stopping_  = false;
    bool in_flight_ = false;
    std::atomic<uint64_t> exported_{0}, failed_{0}, dropped_{0};
    std::thread worker_;
};

} // namespace tidevec
