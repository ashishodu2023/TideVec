// ================================================================
// types.hpp — TideVec C++ SDK core types
// ================================================================

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <cstdint>
#include <functional>

namespace tidevec {

// ── Vector types ─────────────────────────────────────────────────
using Embedding   = std::vector<float>;
using VectorId    = std::string;
using Payload     = std::unordered_map<std::string, std::string>;

// ── Causal edge types ─────────────────────────────────────────────
enum class EdgeType : uint8_t {
    CAUSES       = 0,
    CONTRADICTS  = 1,
    UPDATES      = 2,
    RELATED_TO   = 3,
    ENTITY_OF    = 4,
    SUPPORTS     = 5,
};

inline const char* edge_type_name(EdgeType t) {
    switch (t) {
        case EdgeType::CAUSES:      return "CAUSES";
        case EdgeType::CONTRADICTS: return "CONTRADICTS";
        case EdgeType::UPDATES:     return "UPDATES";
        case EdgeType::RELATED_TO:  return "RELATED_TO";
        case EdgeType::ENTITY_OF:   return "ENTITY_OF";
        case EdgeType::SUPPORTS:    return "SUPPORTS";
        default:                    return "UNKNOWN";
    }
}

struct CausalEdge {
    VectorId  target_id;
    EdgeType  type   = EdgeType::RELATED_TO;
    float     weight = 1.0f;
};

// ── Vector record ────────────────────────────────────────────────
struct Vector {
    VectorId                 id;
    Embedding                embedding;
    Payload                  payload{};
    std::vector<CausalEdge>  edges{};
    std::optional<int64_t>   timestamp_ms{};  // sent as created_at; defaults to now()
};

// ── Search options ───────────────────────────────────────────────
enum class QueryMode : uint8_t {
    DEFAULT            = 0,
    CAUSAL_EXPAND      = 1,
    CONTRADICTION_CHECK = 2,
    EXACT              = 3,
};

enum class Device : uint8_t {
    AUTO = 0,
    CPU  = 1,
    GPU  = 2,
    TPU  = 3,
};

struct SearchOptions {
    int         top_k            = 10;
    // 0 = pure semantic, 1 = pure recency. Unset = the collection's default.
    std::optional<float> temporal_blend{};
    QueryMode   mode             = QueryMode::DEFAULT;
    Device      device           = Device::AUTO;
    int         causal_hops      = 1;
    bool        include_trace    = false;
    bool        staleness_warnings = true;
    float       staleness_threshold = 0.2f;
    // Score multiplier penalty for results superseded by a newer version.
    std::optional<float> superseded_penalty{};
};

// ── Search result ────────────────────────────────────────────────
struct SearchHit {
    VectorId    id;
    float       score;           // final blended score
    float       semantic_score;  // raw cosine similarity
    float       temporal_score;  // Ebbinghaus decay score
    Payload     payload;
    bool        staleness_warning = false;
    std::string staleness_reason;
    std::vector<VectorId> causal_neighbors;
    std::vector<VectorId> contradicted_by;
    std::vector<VectorId> superseded_by;
    int64_t     created_at = 0;
};

struct RetrievalTrace {
    std::string algorithm;       // server "strategy"
    double      latency_ms      = 0.0;
    float       recall_estimate = 0.0f;
    std::string device_used;
    int         vectors_scanned = 0;   // server "distance_computations"
    std::string query_id;
};

struct SearchResponse {
    std::vector<SearchHit> hits;
    RetrievalTrace         trace;
};

// ── Supersession (newer versions of a document mark older ones) ─
struct SupersessionConfig {
    bool        enabled              = true;
    float       similarity_threshold = 0.92f;
    std::string match_field;      // payload key identifying the same document
    int         candidates           = 5;
};

// ── Collection config ────────────────────────────────────────────
struct CollectionConfig {
    std::string name;
    int         dim              = 768;
    int64_t     half_life_ms     = 2592000000LL;  // 30 days default
    std::optional<float> temporal_blend{};        // unset = server default
    int         n_shards         = 4;
    int         n_replicas       = 1;
    std::string index_type       = "tvindex";      // tvindex | hnsw | flat | ivf
    float       staleness_threshold = 0.2f;
    std::string metric           = "cosine";
    std::optional<SupersessionConfig> supersession{};  // unset = disabled
};

// ── Collection stats ─────────────────────────────────────────────
struct CollectionStats {
    std::string name;
    int64_t     vector_count       = 0;
    int64_t     edge_count         = 0;
    int64_t     disk_bytes         = 0;
    double      avg_temporal_score = 0.0;
    int         n_shards           = 0;
    int         n_replicas         = 0;
    std::string index_type;
    int         dim                = 0;
    std::string metric;
    std::optional<SupersessionConfig> supersession;  // set when enabled
};

} // namespace tidevec
