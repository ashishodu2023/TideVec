#pragma once
// ================================================================
// json_serializers.hpp — TideVec ↔ JSON conversion helpers
// Uses nlohmann/json (single header, no external deps)
// ================================================================

#include <tidevec/api/json.hpp>
#include <tidevec/core/cortex_vector.hpp>
#include <tidevec/core/collection.hpp>
#include <tidevec/observability/retrieval_trace.hpp>

namespace tidevec {
using json = nlohmann::json;

// ------ EdgeType --------------------------------------------------
inline std::string edge_type_to_str(EdgeType t) { return edge_type_str(t); }

inline EdgeType edge_type_from_str(const std::string& s) {
    if (s == "CAUSES")      return EdgeType::CAUSES;
    if (s == "CONTRADICTS") return EdgeType::CONTRADICTS;
    if (s == "UPDATES")     return EdgeType::UPDATES;
    if (s == "RELATED_TO")  return EdgeType::RELATED_TO;
    if (s == "ENTITY_OF")   return EdgeType::ENTITY_OF;
    if (s == "SUPPORTS")    return EdgeType::SUPPORTS;
    throw std::invalid_argument("Unknown EdgeType: " + s);
}

// ------ CortexVector → JSON ---------------------------------------
inline json vector_to_json(const CortexVector& v) {
    json j;
    j["id"]         = v.id;
    j["embedding"]  = v.embedding;
    j["created_at"] = v.created_at;
    j["valid_from"] = v.valid_from;
    if (v.valid_until) j["valid_until"] = *v.valid_until;
    j["payload"]    = v.payload;

    json edges = json::array();
    for (const auto& e : v.edges) {
        edges.push_back({
            {"target_id",  e.target_id},
            {"type",       edge_type_to_str(e.type)},
            {"weight",     e.weight},
            {"created_at", e.created_at}
        });
    }
    j["edges"] = edges;
    return j;
}

// ------ JSON → CortexVector ---------------------------------------
inline CortexVector vector_from_json(const json& j) {
    CortexVector v;
    v.id        = j.at("id").get<std::string>();
    v.embedding = j.at("embedding").get<std::vector<float>>();

    if (j.contains("created_at"))
        v.created_at = j["created_at"].get<Timestamp>();
    else
        v.created_at = now_ms();

    v.valid_from = j.value("valid_from", v.created_at);

    if (j.contains("valid_until"))
        v.valid_until = j["valid_until"].get<Timestamp>();

    if (j.contains("ttl_seconds"))
        v.set_ttl_seconds(j["ttl_seconds"].get<int64_t>());

    if (j.contains("payload"))
        v.payload = j["payload"].get<std::unordered_map<std::string,std::string>>();

    if (j.contains("edges")) {
        for (const auto& e : j["edges"]) {
            v.edges.emplace_back(
                e.at("target_id").get<std::string>(),
                edge_type_from_str(e.at("type").get<std::string>()),
                e.value("weight", 1.0f)
            );
        }
    }
    return v;
}

// ------ SearchResult → JSON ---------------------------------------
inline json result_to_json(const SearchResult& r) {
    json j;
    j["id"]              = r.id;
    j["score"]           = r.score;
    j["vector_score"]    = r.vector_score;
    j["temporal_score"]  = r.temporal_score;
    j["created_at"]      = r.created_at;
    j["payload"]         = r.payload;

    if (r.staleness_warning) {
        j["staleness_warning"] = true;
        j["staleness_reason"]  = r.staleness_reason;
    }
    if (!r.causal_neighbors.empty())
        j["causal_neighbors"] = r.causal_neighbors;
    if (!r.contradicted_by.empty())
        j["contradicted_by"] = r.contradicted_by;
    return j;
}

// ------ QueryOptions from JSON -----------------------------------
inline QueryOptions query_opts_from_json(const json& j) {
    QueryOptions opts;
    opts.top_k           = j.value("top_k", 10);
    opts.temporal_blend  = j.value("temporal_blend", 0.3f);
    opts.filter          = j.value("filter", "");
    opts.metric          = j.value("metric", "cosine");
    opts.ef_search       = j.value("ef_search", 128);
    opts.include_trace   = j.value("include_trace", false);
    opts.include_staleness_warnings = j.value("include_staleness_warnings", true);

    std::string mode_str = j.value("mode", "vector_only");
    if (mode_str == "causal_expand")        opts.mode = QueryMode::CAUSAL_EXPAND;
    else if (mode_str == "contradiction_check") opts.mode = QueryMode::CONTRADICTION_CHECK;
    else if (mode_str == "entity_resolve")  opts.mode = QueryMode::ENTITY_RESOLVE;
    else                                    opts.mode = QueryMode::VECTOR_ONLY;

    opts.causal_hops = j.value("causal_hops", 1);
    return opts;
}

// ------ RetrievalTrace → JSON ------------------------------------
inline json trace_to_json(const RetrievalTrace& t) {
    json j;
    j["query_id"]              = t.query_id;
    j["strategy"]              = t.strategy;
    j["collection"]            = t.collection_name;
    j["ef_search"]             = t.ef_search;
    j["distance_computations"] = t.distance_computations;
    j["filter_pruned"]         = t.filter_pruned;
    j["estimated_recall"]      = t.estimated_recall;
    j["avg_temporal_score"]    = t.avg_temporal_score;
    j["min_temporal_score"]    = t.min_temporal_score;
    j["causal_hops"]           = t.causal_hops;
    j["latency_ms"]            = t.latency_ms;

    json sw = json::array();
    for (const auto& w : t.staleness_warnings)
        sw.push_back({{"id", w.id}, {"age_days", w.age_days},
                      {"temporal_score", w.temporal_score},
                      {"reason", w.reason}});
    j["staleness_warnings"] = sw;

    json ca = json::array();
    for (const auto& a : t.contradiction_alerts)
        ca.push_back({{"result_id", a.result_id},
                      {"contradicted_by", a.contradicted_by},
                      {"edge_weight", a.edge_weight}});
    j["contradiction_alerts"] = ca;

    if (!t.otel_trace_id.empty()) j["otel_trace_id"] = t.otel_trace_id;
    return j;
}

// ------ TemporalConfig from JSON ---------------------------------
inline TemporalConfig temporal_cfg_from_json(const json& j) {
    TemporalConfig cfg;
    cfg.half_life_ms       = j.value("half_life_ms",        cfg.half_life_ms);
    cfg.temporal_blend     = j.value("temporal_blend",       cfg.temporal_blend);
    cfg.staleness_threshold= j.value("staleness_threshold",  cfg.staleness_threshold);
    cfg.temporal_cutoff    = j.value("temporal_cutoff",      cfg.temporal_cutoff);
    return cfg;
}

// ------ Standard API responses -----------------------------------
inline json ok(json data = nullptr) {
    json r; r["status"] = "ok";
    if (!data.is_null()) r["data"] = data;
    return r;
}

inline json err(const std::string& msg, int code = 400) {
    return {{"status","error"}, {"error", msg}, {"code", code}};
}

} // namespace tidevec
