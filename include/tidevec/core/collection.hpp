#pragma once

#include <tidevec/core/cortex_vector.hpp>
#include <tidevec/core/temporal_scorer.hpp>
#include <tidevec/index/flat_index.hpp>
#include <tidevec/index/tv_index.hpp>
#include <tidevec/graph/causal_graph.hpp>

#include <string>
#include <memory>
#include <stdexcept>

namespace tidevec {

// ------------------------------------------------------------------
// IndexType — which ANN index backs this collection
// ------------------------------------------------------------------
enum class IndexType {
    FLAT,    // exact search — small collections / ground truth
    TVINDEX  // HNSW + temporal scoring — production
};

// ------------------------------------------------------------------
// AgentContext — per-agent lifecycle settings
// ------------------------------------------------------------------
struct AgentContext {
    std::string agent_id;
    int64_t     memory_ttl_seconds  = 0;     // 0 = no expiry
    float       dedup_threshold     = 0.0f;  // 0 = no dedup; 0.97 = aggressive
    std::vector<std::string> cross_agent_readable;
    bool        enabled             = false;
};

// ------------------------------------------------------------------
// Collection — logical namespace in TideVec
//
// Owns:
//   · An ANN index (FlatIndex or TVIndex)
//   · A CausalGraph
//   · AgentContext lifecycle settings
// ------------------------------------------------------------------
class Collection {
public:
    struct Config {
        std::string   name;
        std::size_t   dim        = 0;
        IndexType     index_type = IndexType::TVINDEX;
        Metric        metric     = Metric::COSINE;
        TemporalConfig temporal;
        AgentContext  agent;
        TVIndexConfig tvindex_cfg;
    };

    explicit Collection(Config cfg)
        : cfg_(std::move(cfg))
        , graph_(std::make_unique<CausalGraph>())
    {
        cfg_.tvindex_cfg.metric  = cfg_.metric;
        cfg_.tvindex_cfg.temporal = cfg_.temporal;

        if (cfg_.index_type == IndexType::FLAT) {
            flat_ = std::make_unique<FlatIndex>(cfg_.dim, cfg_.metric, cfg_.temporal);
        } else {
            tv_ = std::make_unique<TVIndex>(cfg_.tvindex_cfg);
        }
    }

    // ------ write -------------------------------------------------

    void upsert(CortexVector vec) {
        // Apply agent TTL
        if (cfg_.agent.enabled && cfg_.agent.memory_ttl_seconds > 0)
            vec.set_ttl_seconds(cfg_.agent.memory_ttl_seconds);

        // Deduplication check (agent mode)
        if (cfg_.agent.enabled && cfg_.agent.dedup_threshold > 0.0f) {
            QueryOptions dedup_opts;
            dedup_opts.top_k = 1;
            dedup_opts.temporal_blend = 0.0f;  // pure similarity for dedup
            dedup_opts.include_staleness_warnings = false;
            auto hits = search(vec.embedding, dedup_opts);
            if (!hits.empty() && hits[0].vector_score >= cfg_.agent.dedup_threshold) {
                // Near-duplicate: skip insert, optionally merge payload
                return;
            }
        }

        if (flat_) flat_->upsert(vec);
        else       tv_->upsert(vec);
    }

    bool remove(const std::string& id) {
        graph_->remove_node(id);
        if (flat_) return flat_->remove(id);
        return tv_->remove(id);
    }

    void add_edge(const std::string& src, const std::string& tgt,
                  EdgeType type, float weight = 1.0f) {
        graph_->add_edge(src, tgt, type, weight);
    }

    // ------ search ------------------------------------------------

    std::vector<SearchResult> search(const std::vector<float>& query,
                                     const QueryOptions& opts) const {
        std::vector<SearchResult> results;

        if (flat_) results = flat_->search(query, opts);
        else       results = tv_->search(query, opts);

        // Causal expansion
        if (opts.mode == QueryMode::CAUSAL_EXPAND && !results.empty()) {
            std::vector<std::string> seed_ids;
            for (const auto& r : results) seed_ids.push_back(r.id);
            auto expanded = graph_->causal_expand(seed_ids, opts.causal_hops);
            for (auto& r : results) {
                r.causal_neighbors = graph_->get_neighbours(
                    r.id, opts.causal_hops, std::nullopt) ;
            }
        }

        // Contradiction detection
        if (opts.mode == QueryMode::CONTRADICTION_CHECK) {
            for (auto& r : results) {
                r.contradicted_by = graph_->find_contradictions(r.id);
            }
        }

        return results;
    }

    // ------ accessors ---------------------------------------------

    const std::string& name()       const { return cfg_.name; }
    IndexType          index_type() const { return cfg_.index_type; }
    const CausalGraph& graph()      const { return *graph_; }
    CausalGraph&       graph()            { return *graph_; }

    std::size_t size() const {
        if (flat_) return flat_->size();
        return tv_->size();
    }

    void set_temporal_config(const TemporalConfig& cfg) {
        cfg_.temporal = cfg;
        if (flat_) flat_->set_temporal_config(cfg);
        if (tv_)   tv_->set_temporal_config(cfg);
    }

    // Iterate every stored vector — used by DriftBridge for snapshotting.
    // Only TVIndex supports full vector iteration; FlatIndex iteration
    // is a future enhancement.
    void each_vector(std::function<void(const CortexVector&)> fn) const {
        if (tv_) tv_->each_vector(fn);
        // FlatIndex iteration not yet implemented — skipped
    }

    // Atomically swap the TVIndex — called after DriftBridge migration completes.
    void swap_index(std::unique_ptr<TVIndex> new_idx) {
        tv_ = std::move(new_idx);
    }

    // Wire disk-backed SegmentStore for PQ exact rescoring
    void set_vector_store(VectorStore store) {
        if (tv_) tv_->set_vector_store(std::move(store));
    }

private:
    Config cfg_;
    std::unique_ptr<FlatIndex>   flat_;
    std::unique_ptr<TVIndex>     tv_;
    std::unique_ptr<CausalGraph> graph_;
};

} // namespace tidevec
