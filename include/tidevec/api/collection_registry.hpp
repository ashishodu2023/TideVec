#pragma once
// ================================================================
// CollectionRegistry — manages multiple named DurableCollections
//
// The API server holds one registry. Each POST /v1/collections
// creates a new entry; all subsequent calls look up by name.
// Thread-safe: shared_mutex (many readers, exclusive writer).
// ================================================================

#include <tidevec/cluster/durable_collection.hpp>

#include <string>
#include <memory>
#include <unordered_map>
#include <shared_mutex>
#include <stdexcept>
#include <vector>

namespace tidevec {

class CollectionRegistry {
public:
    struct CreateParams {
        std::string name;
        std::size_t dim         = 768;
        std::string index_type  = "tvindex";  // "flat" | "tvindex"
        std::string metric      = "cosine";   // "cosine" | "l2" | "dot"
        std::size_t n_shards    = 4;
        int         n_replicas  = 1;
        int         write_quorum= 1;
        std::string data_dir    = "./tidevec_data";
        TemporalConfig temporal;
    };

    // Create a new collection; throws if name already exists
    void create(const CreateParams& p) {
        std::unique_lock lock(mutex_);
        if (collections_.count(p.name))
            throw std::runtime_error("Collection already exists: " + p.name);

        DurableCollection::Config cfg;
        cfg.name         = p.name;
        cfg.dim          = p.dim;
        cfg.n_shards     = p.n_shards;
        cfg.n_replicas   = p.n_replicas;
        cfg.write_quorum = p.write_quorum;
        cfg.data_dir     = p.data_dir;
        cfg.temporal     = p.temporal;
        cfg.parallel_search = true;

        // Index type + metric go into the TVIndex config
        cfg.tvindex.metric   = parse_metric(p.metric);
        cfg.tvindex.temporal = p.temporal;

        if (p.index_type == "flat") {
            // For flat index, still wrap in DurableCollection
            // but signal via a smaller shard count
            cfg.n_shards = 1;
        }

        collections_[p.name] = std::make_unique<DurableCollection>(cfg);
        metadata_[p.name] = p;
    }

    // Delete a collection
    bool drop(const std::string& name) {
        std::unique_lock lock(mutex_);
        return collections_.erase(name) > 0;
    }

    // Get a collection (throws if not found)
    DurableCollection& get(const std::string& name) {
        std::shared_lock lock(mutex_);
        auto it = collections_.find(name);
        if (it == collections_.end())
            throw std::runtime_error("Collection not found: " + name);
        return *it->second;
    }

    bool exists(const std::string& name) const {
        std::shared_lock lock(mutex_);
        return collections_.count(name) > 0;
    }

    // List all collection names + basic stats
    struct CollectionInfo {
        std::string name;
        std::size_t n_vectors;
        std::size_t n_shards;
        std::size_t dim;
        std::string index_type;
        std::string metric;
    };

    std::vector<CollectionInfo> list() const {
        std::shared_lock lock(mutex_);
        std::vector<CollectionInfo> out;
        for (const auto& [name, col] : collections_) {
            CollectionInfo info;
            info.name       = name;
            info.n_vectors  = col->total_vectors();
            info.n_shards   = col->n_shards();
            auto it = metadata_.find(name);
            if (it != metadata_.end()) {
                info.dim        = it->second.dim;
                info.index_type = it->second.index_type;
                info.metric     = it->second.metric;
            }
            out.push_back(info);
        }
        return out;
    }

    std::size_t size() const {
        std::shared_lock lock(mutex_);
        return collections_.size();
    }

private:
    mutable std::shared_mutex mutex_;
    std::unordered_map<std::string, std::unique_ptr<DurableCollection>> collections_;
    std::unordered_map<std::string, CreateParams> metadata_;
};

} // namespace tidevec
