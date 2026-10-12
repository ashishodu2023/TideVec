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
    if (s == "SUPERSEDED_BY") return EdgeType::SUPERSEDED_BY;
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
    if (!r.superseded_by.empty())
        j["superseded_by"] = r.superseded_by;
    if (r.text_score != 0.0f)
        j["text_score"] = r.text_score;
    return j;
}

// ------ Filters from JSON ------------------------------------------
//
//   {"lang": "en"}                          equality (implicit $eq)
//   {"page": 3}                             numeric equality
//   {"lang": {"$in": ["en", "de"]}}         $in / $nin
//   {"price": {"$gte": 10, "$lt": 20}}      $gt / $gte / $lt / $lte
//   {"published": {"$gte": "2024-01-01"}}   ISO-8601 dates compare as time
//   {"created_at": {"$lt": 1700000000000}}  vector timestamp (ms or ISO)
//   {"_id": {"$in": ["a", "b"]}}            vector ids
//   {"tag": {"$exists": true}}              key present / absent
//   {"$and": [...]}, {"$or": [...]}, {"$not": {...}}
// Several keys in one object are ANDed. A plain "key=value" string is
// the legacy single-equality filter.
inline FilterValue filter_value_from_json(const json& v) {
    if (v.is_string())  return {v.get<std::string>(), std::nullopt};
    if (v.is_boolean()) return {v.get<bool>() ? "true" : "false", std::nullopt};
    if (v.is_number())  {
        double d = v.get<double>();
        return {FilterExpr::_num_str(d), d};
    }
    throw std::invalid_argument("filter values must be strings, numbers or booleans");
}

inline double filter_bound_from_json(const json& v, const std::string& key) {
    if (v.is_number()) return v.get<double>();
    if (v.is_string()) {
        if (auto n = parse_numeric_value(v.get<std::string>())) return *n;
    }
    throw std::invalid_argument("range bound for '" + key +
                                "' must be a number or an ISO-8601 date");
}

inline FilterExpr filter_expr_from_json(const json& j);

inline FilterExpr filter_field_from_json(const std::string& key, const json& v) {
    if (!v.is_object()) {
        if (v.is_array()) {
            FilterExpr f; f.op = FilterExpr::Op::IN; f.key = key;
            for (const auto& x : v) f.values.push_back(filter_value_from_json(x));
            return f;
        }
        FilterExpr f; f.op = FilterExpr::Op::EQ; f.key = key;
        f.values.push_back(filter_value_from_json(v));
        return f;
    }
    std::vector<FilterExpr> parts;
    std::optional<double> gt, gte, lt, lte;
    for (const auto& [op, arg] : v.items()) {
        if (op == "$eq" || op == "$ne") {
            FilterExpr f; f.op = op == "$eq" ? FilterExpr::Op::EQ : FilterExpr::Op::NE;
            f.key = key;
            f.values.push_back(filter_value_from_json(arg));
            parts.push_back(std::move(f));
        } else if (op == "$in" || op == "$nin") {
            if (!arg.is_array())
                throw std::invalid_argument(op + " for '" + key + "' needs an array");
            FilterExpr f; f.op = op == "$in" ? FilterExpr::Op::IN : FilterExpr::Op::NIN;
            f.key = key;
            for (const auto& x : arg) f.values.push_back(filter_value_from_json(x));
            parts.push_back(std::move(f));
        } else if (op == "$gt")  { gt  = filter_bound_from_json(arg, key);
        } else if (op == "$gte") { gte = filter_bound_from_json(arg, key);
        } else if (op == "$lt")  { lt  = filter_bound_from_json(arg, key);
        } else if (op == "$lte") { lte = filter_bound_from_json(arg, key);
        } else if (op == "$exists") {
            parts.push_back(FilterExpr::has(key, arg.get<bool>()));
        } else {
            throw std::invalid_argument("unknown filter operator '" + op + "' for '" + key + "'");
        }
    }
    if (gt || gte || lt || lte) parts.push_back(FilterExpr::range(key, gte, lte, gt, lt));
    if (parts.empty()) throw std::invalid_argument("empty condition for '" + key + "'");
    if (parts.size() == 1) return std::move(parts[0]);
    return FilterExpr::all_of(std::move(parts));
}

inline FilterExpr filter_expr_from_json(const json& j) {
    if (j.is_string()) {
        auto f = parse_legacy_filter(j.get<std::string>());
        if (!f) throw std::invalid_argument("filter string must look like key=value");
        return *f;
    }
    if (!j.is_object()) throw std::invalid_argument("filter must be an object or key=value");
    std::vector<FilterExpr> parts;
    for (const auto& [key, v] : j.items()) {
        if (key == "$and" || key == "$or") {
            if (!v.is_array()) throw std::invalid_argument(key + " needs an array");
            std::vector<FilterExpr> cs;
            for (const auto& c : v) cs.push_back(filter_expr_from_json(c));
            parts.push_back(key == "$and" ? FilterExpr::all_of(std::move(cs))
                                          : FilterExpr::any_of(std::move(cs)));
        } else if (key == "$not") {
            parts.push_back(FilterExpr::negate(filter_expr_from_json(v)));
        } else if (!key.empty() && key[0] == '$') {
            throw std::invalid_argument("unknown filter operator '" + key + "'");
        } else {
            parts.push_back(filter_field_from_json(key, v));
        }
    }
    if (parts.size() == 1) return std::move(parts[0]);
    return FilterExpr::all_of(std::move(parts));
}

// Null / empty object / empty string = no filter.
inline FilterPtr filter_from_json(const json& j) {
    if (j.is_null()) return nullptr;
    if (j.is_object() && j.empty()) return nullptr;
    if (j.is_string() && j.get<std::string>().empty()) return nullptr;
    return std::make_shared<FilterExpr>(filter_expr_from_json(j));
}

// as_of: epoch milliseconds or an ISO-8601 string.
inline std::optional<Timestamp> timestamp_from_json(const json& j) {
    if (j.is_null()) return std::nullopt;
    if (j.is_number()) return j.get<Timestamp>();
    if (j.is_string()) {
        const auto s = j.get<std::string>();
        if (auto ms = parse_iso8601_ms(s)) return *ms;
        if (auto n = parse_numeric_value(s)) return static_cast<Timestamp>(*n);
    }
    throw std::invalid_argument("timestamp must be epoch milliseconds or ISO-8601");
}

inline HybridOptions hybrid_opts_from_json(const json& j) {
    HybridOptions h;
    if (!j.is_object()) return h;
    std::string fusion = j.value("fusion", std::string("rrf"));
    if (fusion == "rrf") h.fusion = FusionMethod::RRF;
    else if (fusion == "weighted") h.fusion = FusionMethod::WEIGHTED;
    else throw std::invalid_argument("hybrid.fusion must be \"rrf\" or \"weighted\"");
    h.alpha    = j.value("alpha", h.alpha);
    h.rrf_k    = j.value("rrf_k", h.rrf_k);
    h.prefetch = j.value("prefetch", h.prefetch);
    return h;
}

// Filter / as_of fields shared by search, scroll, ids, count and delete.
inline void apply_filter_fields(const json& j, QueryOptions& opts) {
    if (j.contains("filter")) {
        const auto& f = j["filter"];
        if (f.is_string()) opts.filter = f.get<std::string>();
        else opts.filter_expr = filter_from_json(f);
    }
    if (j.contains("as_of")) opts.as_of = timestamp_from_json(j["as_of"]);
}

// ------ Stored vector → JSON (get / scroll) ------------------------
inline json stored_vector_to_json(const CortexVector& v, bool with_vector,
                                  bool with_payload = true) {
    json j;
    j["id"]         = v.id;
    j["created_at"] = v.created_at;
    j["valid_from"] = v.valid_from;
    if (v.valid_until) j["valid_until"] = *v.valid_until;
    if (with_payload) j["payload"] = v.payload;
    if (with_vector) j["embedding"] = v.embedding;
    if (!v.edges.empty()) {
        json edges = json::array();
        std::vector<std::string> newer, older;
        for (const auto& e : v.edges) {
            edges.push_back({{"target_id", e.target_id}, {"type", edge_type_to_str(e.type)},
                             {"weight", e.weight}, {"created_at", e.created_at}});
            if (e.type == EdgeType::SUPERSEDED_BY) newer.push_back(e.target_id);
            if (e.type == EdgeType::UPDATES) older.push_back(e.target_id);
        }
        j["edges"] = edges;
        if (!newer.empty()) j["superseded_by"] = newer;
        if (!older.empty()) j["supersedes"] = older;
    }
    return j;
}

// ------ QueryOptions from JSON -----------------------------------
inline QueryOptions query_opts_from_json(const json& j) {
    QueryOptions opts;
    opts.top_k           = j.value("top_k", 10);
    opts.temporal_blend  = j.value("temporal_blend", -1.0f);
    apply_filter_fields(j, opts);
    opts.text            = j.value("text", std::string(""));
    if (j.contains("hybrid")) opts.hybrid = hybrid_opts_from_json(j["hybrid"]);
    opts.metric          = j.value("metric", "cosine");
    opts.ef_search       = j.value("ef_search", 128);
    opts.include_trace   = j.value("include_trace", false);
    opts.include_staleness_warnings = j.value("include_staleness_warnings", true);
    opts.superseded_penalty = j.value("superseded_penalty", 0.0f);

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

// ------ SupersessionConfig ↔ JSON --------------------------------
inline SupersessionConfig supersession_cfg_from_json(const json& j) {
    SupersessionConfig cfg;
    cfg.enabled              = j.value("enabled", true);
    cfg.similarity_threshold = j.value("similarity_threshold", cfg.similarity_threshold);
    cfg.match_field          = j.value("match_field", std::string(""));
    cfg.candidates           = j.value("candidates", cfg.candidates);
    return cfg;
}

inline json supersession_cfg_to_json(const SupersessionConfig& cfg) {
    return {
        {"enabled",              cfg.enabled},
        {"similarity_threshold", cfg.similarity_threshold},
        {"match_field",          cfg.match_field},
        {"candidates",           cfg.candidates},
    };
}

// ------ AgentContext from JSON -----------------------------------
inline AgentContext agent_cfg_from_json(const json& j) {
    AgentContext cfg;
    cfg.agent_id    = j.value("agent_id", std::string(""));
    cfg.memory_ttl_seconds = j.value("memory_ttl_seconds", int64_t{0});
    cfg.dedup_threshold    = j.value("dedup_threshold", 0.0f);
    cfg.enabled            = j.value("enabled", false);
    if (j.contains("cross_agent_readable"))
        cfg.cross_agent_readable =
            j["cross_agent_readable"].get<std::vector<std::string>>();
    return cfg;
}

inline json agent_cfg_to_json(const AgentContext& cfg) {
    return {
        {"agent_id",             cfg.agent_id},
        {"memory_ttl_seconds",   cfg.memory_ttl_seconds},
        {"dedup_threshold",      cfg.dedup_threshold},
        {"cross_agent_readable", cfg.cross_agent_readable},
        {"enabled",              cfg.enabled},
    };
}

// ------ Compaction status (POST/GET /v1/collections/{c}/compact) ---
inline json compaction_status_to_json(const CompactionStatus& s) {
    return {
        {"started",        s.started},
        {"running",        s.running > 0},
        {"running_shards", s.running},
        {"runs",           s.runs},
        {"tombstones",     s.tombstones},
        {"index_bytes",    s.index_bytes},
        {"last_reclaimed", s.last_reclaimed},
        {"last_replayed",  s.last_replayed},
        {"last_ms",        s.last_ms},
    };
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
