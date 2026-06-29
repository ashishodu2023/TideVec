#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <cstdint>
#include <optional>

namespace cortexdb {

// ------------------------------------------------------------------
// Timestamp — milliseconds since epoch
// ------------------------------------------------------------------
using Timestamp = int64_t;

inline Timestamp now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(
        system_clock::now().time_since_epoch()).count();
}

// ------------------------------------------------------------------
// EdgeType — semantic relationship between two vectors
// ------------------------------------------------------------------
enum class EdgeType : uint8_t {
    CAUSES      = 0,
    CONTRADICTS = 1,
    UPDATES     = 2,
    RELATED_TO  = 3,
    ENTITY_OF   = 4,
    SUPPORTS    = 5
};

inline std::string edge_type_str(EdgeType t) {
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

// ------------------------------------------------------------------
// CausalEdge — directed typed link between vectors
// ------------------------------------------------------------------
struct CausalEdge {
    std::string  target_id;
    EdgeType     type;
    float        weight    = 1.0f;
    Timestamp    created_at;

    CausalEdge(std::string tid, EdgeType et, float w = 1.0f)
        : target_id(std::move(tid)), type(et), weight(w), created_at(now_ms()) {}
};

// ------------------------------------------------------------------
// CortexVector — the core unit stored in CortexDB
//
// Extends a standard embedding vector with:
//   · created_at  — insertion timestamp (TVIndex temporal scoring)
//   · valid_from  — start of validity window
//   · valid_until — end of validity window (optional TTL)
//   · edges       — CausalEdge graph layer
//   · payload     — arbitrary metadata (filterable)
// ------------------------------------------------------------------
struct CortexVector {
    std::string  id;
    std::vector<float> embedding;
    std::unordered_map<std::string, std::string> payload;

    Timestamp created_at;
    Timestamp valid_from;
    std::optional<Timestamp> valid_until;  // nullopt = never expires

    std::vector<CausalEdge> edges;

    // ------ constructors ------------------------------------------

    CortexVector() = default;

    CortexVector(std::string id_,
                 std::vector<float> emb,
                 std::unordered_map<std::string, std::string> pl = {})
        : id(std::move(id_))
        , embedding(std::move(emb))
        , payload(std::move(pl))
        , created_at(now_ms())
        , valid_from(created_at)
        , valid_until(std::nullopt)
    {}

    // ------ helpers -----------------------------------------------

    std::size_t dim() const { return embedding.size(); }

    bool is_valid_at(Timestamp t) const {
        if (t < valid_from) return false;
        if (valid_until.has_value() && t > valid_until.value()) return false;
        return true;
    }

    bool is_currently_valid() const { return is_valid_at(now_ms()); }

    void add_edge(const std::string& target, EdgeType type, float w = 1.0f) {
        edges.emplace_back(target, type, w);
    }

    void set_ttl_seconds(int64_t seconds) {
        valid_until = created_at + seconds * 1000LL;
    }
};

// ------------------------------------------------------------------
// SearchResult — returned from any query
// ------------------------------------------------------------------
struct SearchResult {
    std::string  id;
    float        score;           // final blended score
    float        vector_score;    // pure cosine/l2 similarity
    float        temporal_score;  // TVIndex temporal weight
    std::unordered_map<std::string, std::string> payload;
    Timestamp    created_at;
    bool         staleness_warning = false;
    std::string  staleness_reason;

    // causal expansion
    std::vector<std::string> causal_neighbors;
    std::vector<std::string> contradicted_by;
};

// ------------------------------------------------------------------
// QueryOptions — controls all search behaviour
// ------------------------------------------------------------------
enum class QueryMode {
    VECTOR_ONLY,          // standard ANN
    CAUSAL_EXPAND,        // ANN + causal neighbor traversal
    CONTRADICTION_CHECK,  // ANN + return contradicting vectors
    ENTITY_RESOLVE        // find all vectors sharing an entity
};

struct QueryOptions {
    int        top_k            = 10;
    float      temporal_blend   = 0.3f;   // α=1-blend (semantic), β=blend (temporal)
    QueryMode  mode             = QueryMode::VECTOR_ONLY;
    int        causal_hops      = 1;
    std::string filter          = "";      // simple key=value for now
    std::string metric          = "cosine";
    bool       include_trace    = false;
    bool       include_staleness_warnings = true;
    int        ef_search        = 128;    // HNSW search width
};

} // namespace cortexdb
