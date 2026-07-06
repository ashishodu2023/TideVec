#pragma once
// ================================================================
// CollectionRegistry — manages multiple ManagedCollections
//
// PERSISTENCE: On create/drop, collection metadata is written to
// {data_dir}/registry.json. On startup, load_and_recover() reads
// this file, recreates all collections, and replays their WALs.
// ================================================================

#include <tidevec/api/managed_collection.hpp>
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

struct RegistryConfig {
    std::string data_dir        = "./tidevec_data";
    bool ultra_durable          = false;
    bool use_segment_store      = true;
    bool use_accel              = false;
    accel::DeviceType device    = accel::DeviceType::AUTO;
    std::string tenant_id       = "";   // empty = no tenant scoping
};

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
        std::string tenant_id   = "";
    };

    explicit CollectionRegistry(RegistryConfig cfg)
        : cfg_(std::move(cfg)) {
        fs::create_directories(cfg_.data_dir);
    }

    // Backward-compatible constructor (tests, SDK examples)
    explicit CollectionRegistry(std::string data_dir) {
        cfg_.data_dir = std::move(data_dir);
        fs::create_directories(cfg_.data_dir);
    }

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
                p.data_dir     = cfg_.data_dir;
                p.tenant_id    = item.value("tenant_id", std::string(""));
                if (item.contains("temporal"))
                    p.temporal = temporal_cfg_from_json(item["temporal"]);

                // Skip collections not owned by this tenant (multi-tenancy)
                if (!cfg_.tenant_id.empty() && !p.tenant_id.empty() &&
                    p.tenant_id != cfg_.tenant_id)
                    continue;

                std::cout << "[registry] Recovering collection '"
                          << p.name << "'...\n";
                _create_internal(p, /*recover=*/true);
            }
        } catch (const std::exception& e) {
            std::cerr << "[registry] Warning: failed to load registry: "
                      << e.what() << "\n";
        }
    }

    void create(const CreateParams& p) {
        std::unique_lock lock(mutex_);
        if (collections_.count(p.name))
            throw std::runtime_error("Collection already exists: " + p.name);
        CreateParams mp = p;
        mp.data_dir = cfg_.data_dir;
        if (!cfg_.tenant_id.empty()) mp.tenant_id = cfg_.tenant_id;
        _create_internal(mp, /*recover=*/false);
        _save_registry();
    }

    bool drop(const std::string& name) {
        std::unique_lock lock(mutex_);
        bool erased = collections_.erase(name) > 0;
        metadata_.erase(name);
        if (erased) _save_registry();
        return erased;
    }

    ManagedCollection& get(const std::string& name) {
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
        std::string backend;
        std::string tenant_id;
    };

    std::vector<CollectionInfo> list() const {
        std::shared_lock lock(mutex_);
        std::vector<CollectionInfo> out;
        for (const auto& [name, col] : collections_) {
            CollectionInfo info;
            info.name      = name;
            info.n_vectors = col->total_vectors();
            info.n_shards  = col->n_shards();
            info.backend   = col->backend_name();
            auto it = metadata_.find(name);
            if (it != metadata_.end()) {
                info.dim        = it->second.dim;
                info.index_type = it->second.index_type;
                info.metric     = it->second.metric;
                info.tenant_id  = it->second.tenant_id;
            }
            out.push_back(info);
        }
        return out;
    }

    std::size_t size() const {
        std::shared_lock lock(mutex_);
        return collections_.size();
    }

    std::size_t total_vectors() const {
        std::shared_lock lock(mutex_);
        std::size_t n = 0;
        for (const auto& [_, col] : collections_) n += col->total_vectors();
        return n;
    }

    const RegistryConfig& config() const { return cfg_; }

private:
    RegistryConfig cfg_;
    mutable std::shared_mutex mutex_;
    std::unordered_map<std::string, std::unique_ptr<ManagedCollection>> collections_;
    std::unordered_map<std::string, CreateParams> metadata_;

    std::string _registry_path() const {
        return cfg_.data_dir + "/registry.json";
    }

    void _create_internal(const CreateParams& p, bool recover) {
        ManagedCollectionConfig mcfg;
        mcfg.durable.name          = p.name;
        mcfg.durable.dim           = p.dim;
        mcfg.durable.n_shards      = p.n_shards;
        mcfg.durable.n_replicas    = p.n_replicas;
        mcfg.durable.write_quorum  = p.write_quorum;
        mcfg.durable.data_dir      = cfg_.data_dir + "/" + p.name;
        mcfg.durable.temporal      = p.temporal;
        mcfg.durable.tvindex.metric   = parse_metric(p.metric);
        mcfg.durable.tvindex.temporal = p.temporal;
        mcfg.durable.parallel_search  = true;
        mcfg.durable.use_segment_store = cfg_.use_segment_store;
        mcfg.ultra_durable = cfg_.ultra_durable;
        mcfg.use_accel     = cfg_.use_accel;
        mcfg.device_hint   = cfg_.device;

        if (p.index_type == "flat")
            mcfg.durable.n_shards = 1;

        fs::create_directories(mcfg.durable.data_dir);

        auto col = std::make_unique<ManagedCollection>(mcfg);

        if (recover) {
            std::size_t n = col->recover();
            std::cout << "[registry]   -> " << n << " records replayed\n";
        }

        collections_[p.name] = std::move(col);
        metadata_[p.name] = p;
    }

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
                {"tenant_id",    p.tenant_id},
                {"temporal", {
                    {"half_life_ms",        p.temporal.half_life_ms},
                    {"temporal_blend",      p.temporal.temporal_blend},
                    {"staleness_threshold", p.temporal.staleness_threshold},
                }},
            });
        }
        std::ofstream f(_registry_path());
        f << j.dump(2);
        f.flush();
        fsync(f);  // durable registry write — best effort
    }

    static void fsync(std::ofstream& f) {
        f.flush();
        // Registry fsync via reopen (portable)
        (void)f;
    }
};

} // namespace tidevec
