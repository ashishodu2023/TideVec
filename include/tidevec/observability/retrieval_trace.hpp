#pragma once

#include <cortexdb/core/cortex_vector.hpp>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>

namespace cortexdb {

// ------------------------------------------------------------------
// RetrievalTrace — emitted for every query when include_trace=true
//
// Novel: no other open-source vector DB ships this.
// Compatible with OpenTelemetry span attributes.
// ------------------------------------------------------------------
struct StalenessWarning {
    std::string id;
    int64_t     age_days;
    float       temporal_score;
    std::string reason;
};

struct ContradictionAlert {
    std::string result_id;
    std::string contradicted_by;
    float       edge_weight;
};

struct RetrievalTrace {
    std::string query_id;
    std::string strategy;          // "FLAT" | "TVINDEX_HNSW" | "CAUSAL_EXPAND"
    std::string collection_name;

    // TVIndex internals
    int   ef_search              = 0;
    int   distance_computations  = 0;
    int   filter_pruned          = 0;
    int   temporal_excluded      = 0;
    float estimated_recall       = 1.0f;

    // Temporal info
    float avg_temporal_score     = 0.0f;
    float min_temporal_score     = 1.0f;

    // Causal
    int   causal_hops            = 0;
    int   causal_nodes_visited   = 0;

    // Alerts
    std::vector<StalenessWarning>    staleness_warnings;
    std::vector<ContradictionAlert>  contradiction_alerts;

    // Timing
    Timestamp  query_time_ms     = 0;
    double     latency_ms        = 0.0;

    // OTel trace/span IDs (populated when OTel exporter is attached)
    std::string otel_trace_id;
    std::string otel_span_id;

    // ------ serialisation ----------------------------------------

    std::string to_json() const {
        std::ostringstream j;
        j << std::fixed << std::setprecision(4);
        j << "{\n";
        j << "  \"query_id\": \""          << query_id          << "\",\n";
        j << "  \"strategy\": \""          << strategy          << "\",\n";
        j << "  \"collection\": \""        << collection_name   << "\",\n";
        j << "  \"ef_search\": "           << ef_search         << ",\n";
        j << "  \"distance_computations\": "<< distance_computations << ",\n";
        j << "  \"filter_pruned\": "       << filter_pruned     << ",\n";
        j << "  \"temporal_excluded\": "   << temporal_excluded << ",\n";
        j << "  \"estimated_recall\": "    << estimated_recall  << ",\n";
        j << "  \"avg_temporal_score\": "  << avg_temporal_score<< ",\n";
        j << "  \"min_temporal_score\": "  << min_temporal_score<< ",\n";
        j << "  \"causal_hops\": "         << causal_hops       << ",\n";
        j << "  \"causal_nodes_visited\": "<< causal_nodes_visited << ",\n";
        j << "  \"latency_ms\": "          << latency_ms        << ",\n";

        // staleness warnings
        j << "  \"staleness_warnings\": [";
        for (std::size_t i = 0; i < staleness_warnings.size(); ++i) {
            const auto& w = staleness_warnings[i];
            j << (i ? "," : "") << "\n    {\"id\":\"" << w.id
              << "\",\"age_days\":" << w.age_days
              << ",\"temporal_score\":" << w.temporal_score
              << ",\"reason\":\"" << w.reason << "\"}";
        }
        j << "\n  ],\n";

        // contradiction alerts
        j << "  \"contradiction_alerts\": [";
        for (std::size_t i = 0; i < contradiction_alerts.size(); ++i) {
            const auto& a = contradiction_alerts[i];
            j << (i ? "," : "") << "\n    {\"result_id\":\"" << a.result_id
              << "\",\"contradicted_by\":\"" << a.contradicted_by
              << "\",\"edge_weight\":" << a.edge_weight << "}";
        }
        j << "\n  ]";

        if (!otel_trace_id.empty())
            j << ",\n  \"otel_trace_id\": \"" << otel_trace_id << "\"";
        if (!otel_span_id.empty())
            j << ",\n  \"otel_span_id\": \""  << otel_span_id  << "\"";

        j << "\n}";
        return j.str();
    }
};

} // namespace cortexdb
