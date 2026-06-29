// ================================================================
// types.hpp — CortexDB C++ SDK core types
// ================================================================

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <cstdint>
#include <functional>

namespace cortexdb {

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
    Payload                  payload;
    std::vector<CausalEdge>  edges;
    std::optional<int64_t>   timestamp_ms;  // defaults to now()
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
    float       temporal_blend   = 0.3f;   // 0=pure semantic, 1=pure recency
    QueryMode   mode             = QueryMode::DEFAULT;
    Device      device           = Device::AUTO;
    int         causal_hops      = 1;
    bool        include_trace    = false;
    bool        staleness_warnings = true;
    float       staleness_threshold = 0.2f;
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
};

struct RetrievalTrace {
    std::string algorithm;       // "TVIndex" | "CAGRA" | "XLA_MATMUL" | "IVF"
    double      latency_ms;
    float       recall_estimate;
    std::string device_used;
    int         vectors_scanned;
};

struct SearchResponse {
    std::vector<SearchHit> hits;
    RetrievalTrace         trace;
};

// ── Collection config ────────────────────────────────────────────
struct CollectionConfig {
    std::string name;
    int         dim              = 768;
    int64_t     half_life_ms     = 2592000000LL;  // 30 days default
    float       temporal_blend   = 0.3f;
    int         n_shards         = 4;
    int         n_replicas       = 1;
    std::string index_type       = "tvindex";      // tvindex | hnsw | flat | ivf
    float       staleness_threshold = 0.2f;
};

// ── Collection stats ─────────────────────────────────────────────
struct CollectionStats {
    std::string name;
    int64_t     vector_count;
    int64_t     edge_count;
    int64_t     disk_bytes;
    double      avg_temporal_score;
    int         n_shards;
    int         n_replicas;
    std::string index_type;
};

} // namespace cortexdb
