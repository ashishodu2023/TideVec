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
#include <condition_variable>
#include <chrono>
#include <thread>
#include <stdexcept>
#include <vector>
#include <fstream>
#include <filesystem>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <fcntl.h>
#include <unistd.h>

namespace tidevec {
namespace fs = std::filesystem;

// A collection kept alive for as long as the holder needs it, even if it is
// dropped or replaced (restore) in the meantime.
using CollectionPtr = std::shared_ptr<ManagedCollection>;

struct RegistryConfig {
    std::string data_dir        = "./tidevec_data";
    bool ultra_durable          = false;
    bool use_segment_store      = true;
    bool use_accel              = false;
    accel::DeviceType device    = accel::DeviceType::AUTO;
    std::string tenant_id       = "";   // empty = no tenant scoping
    // Rebuild indexes off the write path when the tombstone threshold is
    // crossed. Turn off where replicas must build identical graphs.
    bool background_compaction  = true;
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
        AgentContext   agent;
        SupersessionConfig supersession;
        std::string tenant_id   = "";
        std::string text_field  = "";     // BM25 / hybrid search field
        bool        payload_index = true; // index payload keys for filters
        double      compaction_ratio = 1.0;  // compact at tombstones >= ratio * live
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
            auto doc = json::parse(f);
            // Legacy files are a bare array of collections.
            const json j = doc.is_object() ? doc.value("collections", json::array()) : doc;
            if (doc.is_object() && doc.contains("aliases"))
                for (const auto& [a, c] : doc["aliases"].items())
                    aliases_[a] = c.get<std::string>();
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
                if (item.contains("agent"))
                    p.agent = agent_cfg_from_json(item["agent"]);
                if (item.contains("supersession"))
                    p.supersession = supersession_cfg_from_json(item["supersession"]);
                p.text_field    = item.value("text_field", std::string(""));
                p.payload_index = item.value("payload_index", true);
                p.compaction_ratio = item.value("compaction_ratio", 1.0);

                // Skip collections not owned by this tenant (multi-tenancy)
                if (!cfg_.tenant_id.empty() && !p.tenant_id.empty() &&
                    p.tenant_id != cfg_.tenant_id)
                    continue;

                std::cout << "[registry] Recovering collection '"
                          << p.name << "'...\n";
                _create_internal(p, /*recover=*/true);
            }
            for (auto it = aliases_.begin(); it != aliases_.end();)
                it = collections_.count(it->second) ? std::next(it) : aliases_.erase(it);
        } catch (const std::exception& e) {
            std::cerr << "[registry] Warning: failed to load registry: "
                      << e.what() << "\n";
        }
    }

    void create(const CreateParams& p) {
        std::lock_guard lifecycle(lifecycle_mu_);
        std::unique_lock lock(mutex_);
        if (collections_.count(p.name))
            throw std::runtime_error("Collection already exists: " + p.name);
        if (aliases_.count(p.name))
            throw std::runtime_error("An alias with this name already exists: " + p.name);
        CreateParams mp = p;
        mp.data_dir = cfg_.data_dir;
        if (!cfg_.tenant_id.empty()) mp.tenant_id = cfg_.tenant_id;
        _create_internal(mp, /*recover=*/false);
        _save_registry();
    }

    // Removes the collection, waits for requests still using it, then
    // closes it and deletes its files. Holding lifecycle_mu_ throughout
    // keeps a same-named create() from starting while the old instance
    // could still write into the directory.
    bool drop(const std::string& name) {
        std::lock_guard lifecycle(lifecycle_mu_);
        CollectionPtr old;
        {
            std::unique_lock lock(mutex_);
            auto it = collections_.find(name);
            if (it == collections_.end()) return false;
            old = std::move(it->second);
            collections_.erase(it);
            metadata_.erase(name);
            std::erase_if(aliases_, [&](const auto& a) { return a.second == name; });
            _save_registry();
        }
        _drain(old, name);
        old.reset();
        // Otherwise a re-created collection with the same name would
        // replay the dropped collection's WAL on the next restart.
        std::error_code ec;
        fs::remove_all(cfg_.data_dir + "/" + name, ec);
        return true;
    }

    // `name` may be a collection or an alias. The returned pointer keeps
    // the collection alive across a concurrent drop or restore.
    CollectionPtr acquire(const std::string& name) {
        std::shared_lock lock(mutex_);
        _wait_not_replacing(lock);
        auto it = collections_.find(_resolve_locked(name));
        if (it == collections_.end())
            throw std::runtime_error("Collection not found: " + name);
        return it->second;
    }

    // The reference is only valid while the collection is neither dropped
    // nor replaced; use acquire() where that can happen concurrently.
    ManagedCollection& get(const std::string& name) {
        return *acquire(name);
    }

    bool exists(const std::string& name) const {
        std::shared_lock lock(mutex_);
        return collections_.count(_resolve_locked(name)) > 0;
    }

    // The collection an alias points to (or `name` itself).
    std::string resolve(const std::string& name) const {
        std::shared_lock lock(mutex_);
        return _resolve_locked(name);
    }

    // ------ aliases --------------------------------------------------
    //
    // An alias is a second name for a collection. Repointing one is a
    // single registry write, so readers switch from the old collection
    // to the new one atomically (zero-downtime re-index / model change:
    // build "docs_v2", then point alias "docs" at it).

    // Creates or repoints `alias`; returns the previous target ("" if new).
    std::string set_alias(const std::string& alias, const std::string& collection) {
        std::unique_lock lock(mutex_);
        if (alias.empty()) throw std::invalid_argument("alias name must not be empty");
        if (collections_.count(alias))
            throw std::invalid_argument("A collection named '" + alias +
                                        "' exists; aliases cannot shadow collections");
        if (!collections_.count(collection))
            throw std::runtime_error("Collection not found: " + collection);
        std::string prev;
        if (auto it = aliases_.find(alias); it != aliases_.end()) prev = it->second;
        aliases_[alias] = collection;
        _save_registry();
        return prev;
    }

    bool delete_alias(const std::string& alias) {
        std::unique_lock lock(mutex_);
        if (!aliases_.erase(alias)) return false;
        _save_registry();
        return true;
    }

    std::map<std::string, std::string> aliases() const {
        std::shared_lock lock(mutex_);
        return aliases_;
    }

    std::vector<std::string> aliases_of(const std::string& collection) const {
        std::shared_lock lock(mutex_);
        std::vector<std::string> out;
        for (const auto& [a, c] : aliases_) if (c == collection) out.push_back(a);
        return out;
    }

    // Creation parameters of a collection (or of an alias's target).
    std::optional<CreateParams> params(const std::string& name) const {
        std::shared_lock lock(mutex_);
        auto it = metadata_.find(_resolve_locked(name));
        if (it == metadata_.end()) return std::nullopt;
        return it->second;
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

    // Restore and Raft InstallSnapshot: closes every collection, lets
    // `replace_files` rewrite registry.json and the collection directories,
    // then reloads them. Requests that already hold a collection finish
    // first (their writes land before the files are replaced); new ones
    // wait in acquire() until the reload is done.
    void replace_from_disk(const std::function<void(const std::vector<std::string>& old_names)>& replace_files) {
        std::lock_guard lifecycle(lifecycle_mu_);
        std::vector<std::string> names;
        std::vector<std::pair<std::string, CollectionPtr>> old;
        {
            std::unique_lock lock(mutex_);
            replacing_ = true;
            for (auto& [name, col] : collections_) {
                names.push_back(name);
                old.emplace_back(name, std::move(col));
            }
            collections_.clear();
            metadata_.clear();
            aliases_.clear();
        }
        for (auto& [name, col] : old) _drain(col, name);
        old.clear();

        std::unique_lock lock(mutex_);
        struct Done {
            CollectionRegistry* r;
            ~Done() { r->replacing_ = false; r->replaced_cv_.notify_all(); }
        } done{this};
        replace_files(names);
        load_and_recover();
    }

private:
    RegistryConfig cfg_;
    mutable std::shared_mutex mutex_;
    // Serializes create/drop/replace_from_disk (held while draining).
    std::mutex lifecycle_mu_;
    bool replacing_ = false;  // guarded by mutex_
    mutable std::condition_variable_any replaced_cv_;
    std::unordered_map<std::string, CollectionPtr> collections_;
    std::unordered_map<std::string, CreateParams> metadata_;
    std::map<std::string, std::string> aliases_;   // alias -> collection

    std::string _registry_path() const {
        return cfg_.data_dir + "/registry.json";
    }

    // Bounded, so a caller blocked inside a request that itself waits on
    // the registry cannot hang a restore forever.
    void _wait_not_replacing(std::shared_lock<std::shared_mutex>& lock) const {
        if (replacing_)
            replaced_cv_.wait_for(lock, std::chrono::seconds(60),
                                  [this] { return !replacing_; });
    }

    // Waits until `col` is held only by the caller, i.e. no request is
    // still using it.
    static void _drain(const CollectionPtr& col, const std::string& name) {
        using clock = std::chrono::steady_clock;
        const auto deadline = clock::now() + std::chrono::seconds(30);
        while (col && col.use_count() > 1) {
            if (clock::now() > deadline) {
                std::cerr << "[registry] Collection '" << name << "' still in use after 30 s; "
                             "closing it when the last request finishes\n";
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
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
        mcfg.durable.agent         = p.agent;
        mcfg.durable.supersession  = p.supersession;
        mcfg.durable.tvindex.metric   = parse_metric(p.metric);
        mcfg.durable.tvindex.temporal = p.temporal;
        mcfg.durable.tvindex.text_field    = p.text_field;
        mcfg.durable.tvindex.payload_index = p.payload_index;
        mcfg.durable.tvindex.compaction_ratio      = p.compaction_ratio;
        mcfg.durable.tvindex.background_compaction = cfg_.background_compaction;
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
            if (p.agent.enabled)
                j.back()["agent"] = agent_cfg_to_json(p.agent);
            if (p.supersession.enabled)
                j.back()["supersession"] = supersession_cfg_to_json(p.supersession);
            if (!p.text_field.empty()) j.back()["text_field"] = p.text_field;
            if (!p.payload_index) j.back()["payload_index"] = false;
            if (p.compaction_ratio != 1.0) j.back()["compaction_ratio"] = p.compaction_ratio;
        }
        // Without aliases keep the legacy bare-array layout, so older
        // servers can still open the data directory.
        json doc = j;
        if (!aliases_.empty()) doc = {{"collections", j}, {"aliases", aliases_}};

        // Write-then-rename so a crash never leaves a torn registry, and
        // an alias swap is all-or-nothing on disk too.
        const std::string path = _registry_path(), tmp = path + ".tmp";
        {
            std::ofstream f(tmp, std::ios::trunc);
            f << doc.dump(2);
            f.flush();
            if (!f) throw std::runtime_error("failed to write " + tmp);
        }
        _fsync_path(tmp);
        fs::rename(tmp, path);
        _fsync_path(cfg_.data_dir);
    }

    static void _fsync_path(const std::string& p) {
        int fd = ::open(p.c_str(), O_RDONLY);
        if (fd < 0) return;
        ::fsync(fd);
        ::close(fd);
    }

    std::string _resolve_locked(const std::string& name) const {
        if (collections_.count(name)) return name;
        auto it = aliases_.find(name);
        return it == aliases_.end() ? name : it->second;
    }
};

} // namespace tidevec
