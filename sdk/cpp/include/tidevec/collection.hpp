// ================================================================
// collection.hpp — CortexDB collection handle
// ================================================================

#pragma once

#include <cortexdb/client.hpp>

namespace cortexdb {

// Convenience wrapper — binds a client + collection name together
class Collection {
public:
    Collection(CortexDB& db, std::string name)
        : db_(db), name_(std::move(name)) {}

    void upsert(const std::vector<Vector>& vecs) {
        db_.upsert(name_, vecs);
    }

    void upsert_one(const Vector& v) {
        db_.upsert_one(name_, v);
    }

    SearchResponse search(const Embedding& query,
                          const SearchOptions& opts = {}) {
        return db_.search(name_, query, opts);
    }

    void add_edges(const std::string& src_id,
                   const std::vector<CausalEdge>& edges) {
        db_.add_edges(name_, edges, src_id);
    }

    void set_temporal(int64_t half_life_ms, float blend = 0.3f) {
        db_.set_temporal(name_, half_life_ms, blend);
    }

    void drop() { db_.drop_collection(name_); }

    const std::string& name() const { return name_; }

private:
    CortexDB&   db_;
    std::string name_;
};

} // namespace cortexdb
