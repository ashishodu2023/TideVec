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
#include <tidevec/core/filter.hpp>
#include <tidevec/core/hybrid.hpp>

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
            ucfg.supersession = cfg_.durable.supersession;
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
        _check_dim(vec.embedding.size(), "vector " + vec.id);
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

    // Removes every vector whose payload matches a "key=value" filter.
    std::size_t remove_where(const std::string& filter) {
        auto eq = filter.find('=');
        if (eq == std::string::npos || eq == 0)
            throw std::invalid_argument("filter must be key=value");
        return remove_where(parse_legacy_filter(filter));
    }

    // Removes every vector matching a structured filter.
    std::size_t remove_where(const FilterPtr& filter) {
        if (!filter) throw std::invalid_argument("delete needs a non-empty filter");
        QueryOptions opts;
        opts.filter_expr = filter;
        std::vector<std::string> ids;
        std::string cursor;
        while (true) {
            auto page = _durable().scroll(cursor, 1024, opts, false);
            for (const auto& v : page) ids.push_back(v.id);
            if (page.size() < 1024) break;
            cursor = page.back().id;
        }
        std::size_t n = 0;
        for (const auto& id : ids) if (remove(id)) ++n;
        return n;
    }

    // ------ point reads / iteration ------------------------------

    std::optional<CortexVector> get(const std::string& id, bool with_vector = true) const {
        return _durable().get(id, with_vector);
    }

    struct ScrollPage {
        std::vector<CortexVector> vectors;
        std::optional<std::string> next_cursor;   // nullopt = no more pages
    };

    // Vectors with id > cursor in id order, filtered by opts.filter /
    // filter_expr / as_of.
    ScrollPage scroll(const std::string& cursor, std::size_t limit,
                      const QueryOptions& opts, bool with_vector = false) const {
        if (limit == 0) throw std::invalid_argument("limit must be > 0");
        ScrollPage page;
        page.vectors = _durable().scroll(cursor, limit + 1, opts, with_vector);
        if (page.vectors.size() > limit) {
            page.vectors.resize(limit);
            page.next_cursor = page.vectors.back().id;
        }
        return page;
    }

    std::size_t count(const QueryOptions& opts) const {
        if (!effective_filter(opts)) return total_vectors();
        std::size_t n = 0;
        std::string cursor;
        while (true) {
            auto page = _durable().scroll(cursor, 4096, opts, false);
            n += page.size();
            if (page.size() < 4096) break;
            cursor = page.back().id;
        }
        return n;
    }

    const std::string& text_field() const { return cfg_.durable.tvindex.text_field; }

    CompactionStatus compact(bool wait = false) {
        return _durable().compact(wait);
    }
    CompactionStatus compaction_status() const {
        return _durable().compaction_status();
    }

    void add_edge(const std::string& src, const std::string& tgt,
                  EdgeType type, float weight = 1.0f) {
        switch (backend_) {
        case Backend::ULTRA_DURABLE: ultra_->add_edge(src, tgt, type, weight); break;
        case Backend::ACCELERATED:   accel_->add_edge(src, tgt, type, weight); break;
        default:                     durable_->add_edge(src, tgt, type, weight); break;
        }
    }

    // Dense search; keyword (BM25) search when only opts.text is given;
    // hybrid fusion of both when the query has a vector and text.
    std::vector<SearchResult> search(const std::vector<float>& query,
                                     QueryOptions opts,
                                     RetrievalTrace* trace = nullptr) {
        if (opts.as_of && !opts.created_at_of) {
            const DurableCollection* d = &_durable();
            opts.created_at_of = [d](const std::string& id) { return d->created_at_of(id); };
        }
        if (!opts.text.empty()) {
            if (text_field().empty())
                throw std::invalid_argument(
                    "text search needs a collection created with text_field");
            if (query.empty()) return _durable().text_search(opts.text, opts);
            _check_dim(query.size(), "query");
            QueryOptions sub = opts;
            sub.top_k = hybrid_prefetch(opts);
            auto sparse = _durable().text_search(opts.text, sub);
            sub.text.clear();
            auto dense = _dense_search(query, sub, trace);
            return fuse_results(dense, sparse, opts.hybrid, opts.top_k);
        }
        _check_dim(query.size(), "query");
        return _dense_search(query, opts, trace);
    }

private:
    std::vector<SearchResult> _dense_search(const std::vector<float>& query,
                                            const QueryOptions& opts,
                                            RetrievalTrace* trace) {
        switch (backend_) {
        case Backend::ULTRA_DURABLE:
            return ultra_->search(query, opts);
        case Backend::ACCELERATED:
            return accel_->search(query, opts, trace, cfg_.device_hint);
        default:
            return durable_->search(query, opts, trace);
        }
    }

    const DurableCollection& _durable() const {
        switch (backend_) {
        case Backend::ULTRA_DURABLE: return ultra_->underlying();
        case Backend::ACCELERATED:   return accel_->durable();
        default:                     return *durable_;
        }
    }

public:
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

    const SupersessionConfig& supersession_config() const {
        return cfg_.durable.supersession;
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

    void _check_dim(std::size_t got, const std::string& what) const {
        const std::size_t want = cfg_.durable.dim;
        if (want > 0 && got != want)
            throw std::invalid_argument(what + " has dimension " + std::to_string(got) +
                                        ", collection expects " + std::to_string(want));
    }

    ManagedCollectionConfig cfg_;
    Backend backend_;

    std::unique_ptr<DurableCollection>      durable_;
    std::unique_ptr<UltraDurableCollection>   ultra_;
    std::unique_ptr<AcceleratedCollection>    accel_;
};

} // namespace tidevec
