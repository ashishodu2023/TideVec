#pragma once
// ================================================================
// CollectionRegistry — manages multiple named DurableCollections
//
// The API server holds one registry. Each POST /v1/collections
// creates a new entry; all subsequent calls look up by name.
// Thread-safe: shared_mutex (many readers, exclusive writer).
//
// PERSISTENCE: On create/drop, collection metadata is written to
// {data_dir}/registry.json. On startup, load_and_recover() reads
// this file, recreates all collections, and replays their WALs.
// This gives full crash-recovery without any external database.
// ================================================================

#include <tidevec/cluster/durable_collection.hpp>
#include <tidevec/api/json_serializers.hpp>

#include <string>
#include <memory>
#include <unordered_map>
#include <shared_mutex>
#include <stdexcept>
#include <vector>
#include <fstream>
#include <filesystem>

namespace tidevec {
namespace fs = std::filesystem;

class CollectionRegistry {
public:
    struct CreateParams {
        std::string name;
        std::size_t dim         = 768;
        std::string index_type  = "tvindex";
        std::string metric      = "cosine";
        std::size_t n_shards    = 4;
        int         n_replicas  = 1;
        int         write_quorum= 1;
        std::string data_dir    = "./tidevec_data";
        TemporalConfig temporal;
    };

    explicit CollectionRegistry(std::string data_dir = "./tidevec_data")
        : data_dir_(std::move(data_dir)) {
        fs::create_directories(data_dir_);
    }

    // ---- Startup: load metadata + replay WALs -------------------
    // Call once at server startup before accepting requests.
    void load_and_recover() {
        auto path = _registry_path();
        if (!fs::exists(path)) return;

        try {
            std::ifstream f(path);
            auto j = json::parse(f);
            for (const auto& item : j) {
                CreateParams p;
                p.name         = item.at("name").get<std::string>();
                p.dim          = item.value("dim", 768UL);
                p.index_type   = item.value("index_type", "tvindex");
                p.metric       = item.value("metric", "cosine");
                p.n_shards     = item.value("n_shards", 4UL);
                p.n_replicas   = item.value("n_replicas", 1);
                p.write_quorum = item.value("write_quorum", 1);
                p.data_dir     = data_dir_;
                if (item.contains("temporal"))
                    p.temporal = temporal_cfg_from_json(item["temporal"]);

                std::cout << "[registry] Recovering collection '"
                          << p.name << "'...\n";
                _create_internal(p, /*recover=*/true);
            }
        } catch (const std::exception& e) {
            std::cerr << "[registry] Warning: failed to load registry: "
                      << e.what() << "\n";
        }
    }

    // ---- Create a new collection --------------------------------
    void create(const CreateParams& p) {
        std::unique_lock lock(mutex_);
        if (collections_.count(p.name))
            throw std::runtime_error("Collection already exists: " + p.name);
        CreateParams mp = p;
        mp.data_dir = data_dir_;
        _create_internal(mp, /*recover=*/false);
        _save_registry();
    }

    // ---- Delete a collection ------------------------------------
    bool drop(const std::string& name) {
        std::unique_lock lock(mutex_);
        bool erased = collections_.erase(name) > 0;
        metadata_.erase(name);
        if (erased) _save_registry();
        return erased;
    }

    // ---- Get a collection (throws if not found) -----------------
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
            info.name      = name;
            info.n_vectors = col->total_vectors();
            info.n_shards  = col->n_shards();
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
    std::string data_dir_;
    mutable std::shared_mutex mutex_;
    std::unordered_map<std::string, std::unique_ptr<DurableCollection>> collections_;
    std::unordered_map<std::string, CreateParams> metadata_;

    std::string _registry_path() const {
        return data_dir_ + "/registry.json";
    }

    // Internal create — optionally replays WAL for crash recovery.
    // Caller must hold write lock (or be in load_and_recover before
    // the server accepts connections, where locking isn't needed).
    void _create_internal(const CreateParams& p, bool recover) {
        DurableCollection::Config cfg;
        cfg.name          = p.name;
        cfg.dim           = p.dim;
        cfg.n_shards      = p.n_shards;
        cfg.n_replicas    = p.n_replicas;
        cfg.write_quorum  = p.write_quorum;
        cfg.data_dir      = data_dir_ + "/" + p.name;
        cfg.temporal      = p.temporal;
        cfg.tvindex.metric   = parse_metric(p.metric);
        cfg.tvindex.temporal = p.temporal;
        cfg.parallel_search  = true;

        if (p.index_type == "flat")
            cfg.n_shards = 1;

        fs::create_directories(cfg.data_dir);

        auto col = std::make_unique<DurableCollection>(cfg);

        if (recover) {
            std::size_t n = col->recover();
            std::cout << "[registry]   -> " << n << " records replayed\n";
        }

        collections_[p.name] = std::move(col);
        metadata_[p.name] = p;
    }

    // Persist collection metadata to disk as JSON.
    // Called under write lock.
    void _save_registry() {
        json j = json::array();
        for (const auto& [name, p] : metadata_) {
            j.push_back({
                {"name",         p.name},
                {"dim",          p.dim},
                {"index_type",   p.index_type},
                {"metric",       p.metric},
                {"n_shards",     p.n_shards},
                {"n_replicas",   p.n_replicas},
                {"write_quorum", p.write_quorum},
                {"temporal", {
                    {"half_life_ms",        p.temporal.half_life_ms},
                    {"temporal_blend",      p.temporal.temporal_blend},
                    {"staleness_threshold", p.temporal.staleness_threshold},
                }},
            });
        }
        std::ofstream f(_registry_path());
        f << j.dump(2);
    }
};

} // namespace tidevec
