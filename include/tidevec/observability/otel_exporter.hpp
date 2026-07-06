#pragma once
// ================================================================
// OTelExporter — export RetrievalTrace to OTLP HTTP endpoint
// ================================================================

#include <tidevec/observability/retrieval_trace.hpp>
#include <tidevec/api/json_serializers.hpp>
#include <tidevec/api/httplib.h>

#include <string>

namespace tidevec {

struct OtelConfig {
    std::string endpoint = "";          // e.g. http://localhost:4318/v1/traces
    std::string service_name = "tidevec";
    bool enabled = false;
};

class OtelExporter {
public:
    explicit OtelExporter(OtelConfig cfg) : cfg_(std::move(cfg)) {}

    void export_trace(const RetrievalTrace& trace) {
        if (!cfg_.enabled || cfg_.endpoint.empty()) return;

        // Structured JSON log (always emitted when enabled)
        _log_structured(trace);
        _send_otlp(trace);
    }

    void export_event(const std::string& level,
                      const std::string& message,
                      const nlohmann::json& fields = {}) {
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

    bool enabled() const { return cfg_.enabled; }

private:
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

    void _send_otlp(const RetrievalTrace& trace) {
        nlohmann::json span = {
            {"traceId", trace.query_id},
            {"spanId",  trace.query_id.substr(0, 16)},
            {"name",    "tidevec.search"},
            {"kind",    1},
            {"startTimeUnixNano",
                static_cast<int64_t>(trace.latency_ms * 1'000'000)},
            {"endTimeUnixNano",
                static_cast<int64_t>(trace.latency_ms * 1'000'000)},
            {"attributes", nlohmann::json::array({
                {{"key","strategy"},{"value",{{"stringValue",trace.strategy}}}},
                {{"key","collection"},{"value",{{"stringValue",trace.collection_name}}}},
                {{"key","latency_ms"},{"value",{{"doubleValue",trace.latency_ms}}}},
            })},
        };
        nlohmann::json payload = {
            {"resourceSpans", nlohmann::json::array({
                {{"resource",{{"attributes",nlohmann::json::array({
                    {{"key","service.name"},
                     {"value",{{"stringValue",cfg_.service_name}}}}
                })}}},
                 {"scopeSpans", nlohmann::json::array({
                     {{"spans", nlohmann::json::array({span})}}
                 })}}
            })}
        };

        try {
            httplib::Client cli(cfg_.endpoint.c_str());
            cli.set_connection_timeout(2, 0);
            cli.Post("/", payload.dump(), "application/json");
        } catch (...) {}
    }

    OtelConfig cfg_;
    std::mutex mu_;
};

} // namespace tidevec
