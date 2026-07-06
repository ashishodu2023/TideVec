#pragma once
// ================================================================
// ManagedCollection — unified collection backend for the REST server
//
// Selects the appropriate storage stack based on server config:
//   · ultra_durable → UltraDurableCollection (Raft + RS + health monitor)
//   · accelerated   → AcceleratedCollection (GPU/TPU dispatch)
//   · default       → DurableCollection (WAL + sharding + SegmentStore)
// ================================================================

#include <tidevec/cluster/durable_collection.hpp>
#include <tidevec/cluster/ultra_durable_collection.hpp>
#include <tidevec/accelerator/accelerated_collection.hpp>
#include <tidevec/accelerator/device.hpp>

#include <memory>
#include <string>
#include <vector>

namespace tidevec {

struct ManagedCollectionConfig {
    DurableCollection::Config durable;
    bool ultra_durable = false;
    bool use_accel     = false;
    accel::DeviceType device_hint = accel::DeviceType::AUTO;
};

class ManagedCollection {
public:
    enum class Backend { DURABLE, ULTRA_DURABLE, ACCELERATED };

    explicit ManagedCollection(ManagedCollectionConfig cfg)
        : cfg_(std::move(cfg))
        , backend_(_select_backend())
    {
        switch (backend_) {
        case Backend::ULTRA_DURABLE: {
            UltraDurableCollection::Config ucfg;
            ucfg.name         = cfg_.durable.name;
            ucfg.dim          = cfg_.durable.dim;
            ucfg.n_shards     = cfg_.durable.n_shards;
            ucfg.temporal     = cfg_.durable.temporal;
            ucfg.tvindex      = cfg_.durable.tvindex;
            ucfg.data_dir     = cfg_.durable.data_dir;
            ultra_ = std::make_unique<UltraDurableCollection>(ucfg);
            break;
        }
        case Backend::ACCELERATED: {
            AcceleratedCollection::Config acfg;
            acfg.durable = cfg_.durable;
            acfg.accel.preferred = cfg_.device_hint;
            accel_ = std::make_unique<AcceleratedCollection>(acfg);
            break;
        }
        default:
            durable_ = std::make_unique<DurableCollection>(cfg_.durable);
            break;
        }
    }

    std::size_t recover() {
        switch (backend_) {
        case Backend::ULTRA_DURABLE: return ultra_->recover();
        case Backend::ACCELERATED:   return accel_->recover();
        default:                     return durable_->recover();
        }
    }

    void upsert(const CortexVector& vec) {
        switch (backend_) {
        case Backend::ULTRA_DURABLE: ultra_->upsert(vec); break;
        case Backend::ACCELERATED:   accel_->upsert(vec); break;
        default:                     durable_->upsert(vec); break;
        }
    }

    bool remove(const std::string& id) {
        switch (backend_) {
        case Backend::ULTRA_DURABLE: return ultra_->remove(id);
        case Backend::ACCELERATED:   return accel_->remove(id);
        default:                     return durable_->remove(id);
        }
    }

    void add_edge(const std::string& src, const std::string& tgt,
                  EdgeType type, float weight = 1.0f) {
        switch (backend_) {
        case Backend::ULTRA_DURABLE: ultra_->add_edge(src, tgt, type, weight); break;
        case Backend::ACCELERATED:   accel_->add_edge(src, tgt, type, weight); break;
        default:                     durable_->add_edge(src, tgt, type, weight); break;
        }
    }

    std::vector<SearchResult> search(const std::vector<float>& query,
                                     QueryOptions opts,
                                     RetrievalTrace* trace = nullptr) {
        switch (backend_) {
        case Backend::ULTRA_DURABLE:
            return ultra_->search(query, opts);
        case Backend::ACCELERATED:
            return accel_->search(query, opts, trace, cfg_.device_hint);
        default:
            return durable_->search(query, opts, trace);
        }
    }

    std::vector<CortexVector> snapshot_vectors() const {
        switch (backend_) {
        case Backend::ULTRA_DURABLE: return ultra_->snapshot_vectors();
        case Backend::ACCELERATED:   return accel_->snapshot_vectors();
        default:                     return durable_->snapshot_vectors();
        }
    }

    void swap_index(std::unique_ptr<TVIndex> new_index) {
        switch (backend_) {
        case Backend::ULTRA_DURABLE: ultra_->swap_index(std::move(new_index)); break;
        case Backend::ACCELERATED:   accel_->swap_index(std::move(new_index)); break;
        default:                     durable_->swap_index(std::move(new_index)); break;
        }
    }

    TemporalConfig temporal_config() const {
        switch (backend_) {
        case Backend::ULTRA_DURABLE: return ultra_->temporal_config();
        case Backend::ACCELERATED:   return accel_->temporal_config();
        default:                     return durable_->temporal_config();
        }
    }

    void set_temporal_config(const TemporalConfig& cfg) {
        switch (backend_) {
        case Backend::ULTRA_DURABLE: ultra_->set_temporal_config(cfg); break;
        case Backend::ACCELERATED:   accel_->set_temporal_config(cfg); break;
        default:                     durable_->set_temporal_config(cfg); break;
        }
    }

    std::size_t total_vectors() const {
        switch (backend_) {
        case Backend::ULTRA_DURABLE: return ultra_->total_vectors();
        case Backend::ACCELERATED:   return accel_->total_vectors();
        default:                     return durable_->total_vectors();
        }
    }

    std::size_t n_shards() const {
        switch (backend_) {
        case Backend::ULTRA_DURABLE: return ultra_->n_shards();
        case Backend::ACCELERATED:   return accel_->n_shards();
        default:                     return durable_->n_shards();
        }
    }

    uint64_t total_writes() const {
        switch (backend_) {
        case Backend::ULTRA_DURABLE: return ultra_->total_writes();
        case Backend::ACCELERATED:   return accel_->total_writes();
        default:                     return durable_->total_writes();
        }
    }

    uint64_t total_queries() const {
        switch (backend_) {
        case Backend::ULTRA_DURABLE: return ultra_->total_queries();
        case Backend::ACCELERATED:   return accel_->total_queries();
        default:                     return durable_->total_queries();
        }
    }

    Backend backend() const { return backend_; }
    std::string backend_name() const {
        switch (backend_) {
        case Backend::ULTRA_DURABLE: return "ultra_durable";
        case Backend::ACCELERATED:   return "accelerated";
        default:                     return "durable";
        }
    }

    bool gpu_available() const {
        return backend_ == Backend::ACCELERATED && accel_ &&
               accel_->gpu_available();
    }

private:
    Backend _select_backend() const {
        if (cfg_.ultra_durable) return Backend::ULTRA_DURABLE;
        if (cfg_.use_accel)     return Backend::ACCELERATED;
        return Backend::DURABLE;
    }

    ManagedCollectionConfig cfg_;
    Backend backend_;

    std::unique_ptr<DurableCollection>      durable_;
    std::unique_ptr<UltraDurableCollection>   ultra_;
    std::unique_ptr<AcceleratedCollection>    accel_;
};

} // namespace tidevec
