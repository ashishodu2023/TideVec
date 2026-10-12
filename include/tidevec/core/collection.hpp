#pragma once

#include <tidevec/core/cortex_vector.hpp>
#include <tidevec/core/temporal_scorer.hpp>
#include <tidevec/core/supersession.hpp>
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

// Index compaction state of a sharded collection. Counts are summed over
// shards and taken from each shard's primary (replicas hold the same data),
// except `running`, which covers every copy.
struct CompactionStatus {
    std::size_t started = 0;       // shards a compact() call started rebuilding
    std::size_t running = 0;       // shards with a rebuild in progress
    std::size_t runs = 0;          // completed compactions
    std::size_t tombstones = 0;
    std::size_t index_bytes = 0;   // approximate graph memory
    std::size_t last_reclaimed = 0;
    std::size_t last_replayed = 0;
    double      last_ms = 0;       // slowest shard's last compaction
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
        SupersessionConfig supersession;
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

        std::vector<SupersessionLink> links;
        if (cfg_.supersession.enabled && size() > 0)
            links = plan_supersession(
                vec, search(vec.embedding, supersession_query_opts(cfg_.supersession)),
                cfg_.supersession);

        if (flat_) flat_->upsert(vec);
        else       tv_->upsert(vec);

        for (const auto& l : links)
            if (!graph_->has_edge(l.newer, l.older, EdgeType::UPDATES))
                graph_->add_edge(l.newer, l.older, EdgeType::UPDATES);
    }

    bool contains(const std::string& id) const {
        return flat_ ? flat_->contains(id) : tv_->contains(id);
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

    void purge_expired() {
        if (tv_) tv_->purge_expired();
    }

    // Start a background rebuild of the vector, payload and text indexes
    // (see TVIndex::compact_async). Flat collections have nothing to compact.
    bool compact_async() { return tv_ ? tv_->compact_async() : false; }
    void wait_for_compaction() { if (tv_) tv_->wait_for_compaction(); }
    CompactionStats compaction_stats() const {
        return tv_ ? tv_->compaction_stats() : CompactionStats{};
    }
    std::size_t tombstones() const { return tv_ ? tv_->tombstones() : 0; }
    std::size_t index_memory_bytes() const { return tv_ ? tv_->approx_memory_bytes() : 0; }

    const AgentContext& agent_config() const { return cfg_.agent; }

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

        _apply_supersession(results, opts);
        return results;
    }

    // BM25 keyword search over the configured text field.
    std::vector<SearchResult> text_search(const std::string& text,
                                          const QueryOptions& opts) const {
        if (!tv_) return {};
        auto results = tv_->text_search(text, opts);
        _apply_supersession(results, opts);
        return results;
    }

    std::vector<std::string> superseded_by(const std::string& id) const {
        return graph_->find_superseding(id);
    }

    // ------ point reads -------------------------------------------

    // A live vector by id, with its outgoing edges (UPDATES,
    // SUPERSEDED_BY, ...) attached.
    std::optional<CortexVector> get(const std::string& id, bool with_vector = true) const {
        std::optional<CortexVector> v;
        if (tv_) v = tv_->get(id, with_vector);
        else if (const auto* p = flat_->get(id)) {
            v = *p;
            if (!with_vector) v->embedding.clear();
        }
        if (!v) return v;
        v->edges.clear();
        for (const auto& e : graph_->get_out_edges(id)) {
            CausalEdge ce(e.tgt_id, e.type, e.weight);
            ce.created_at = e.created_at;
            v->edges.push_back(std::move(ce));
        }
        // Single-collection supersession stores only newer -UPDATES-> older.
        for (const auto& newer : graph_->find_superseding(id)) {
            bool present = std::any_of(v->edges.begin(), v->edges.end(), [&](const CausalEdge& e) {
                return e.type == EdgeType::SUPERSEDED_BY && e.target_id == newer;
            });
            if (!present) v->edges.emplace_back(newer, EdgeType::SUPERSEDED_BY);
        }
        return v;
    }

    std::optional<Timestamp> created_at_of(const std::string& id) const {
        if (tv_) return tv_->created_at_of(id);
        if (const auto* p = flat_->get(id)) return p->created_at;
        return std::nullopt;
    }

    std::vector<CortexVector> scroll(const std::string& after, std::size_t limit,
                                     const QueryOptions& opts,
                                     bool with_vector = false) const {
        if (tv_) return tv_->scroll(after, limit, opts, with_vector);
        return flat_->scroll(after, limit, opts, with_vector);
    }

    SearchPlan explain(const QueryOptions& opts) const {
        return tv_ ? tv_->explain(opts) : SearchPlan{};
    }

    const SupersessionConfig& supersession_config() const { return cfg_.supersession; }
    void set_supersession_config(const SupersessionConfig& cfg) { cfg_.supersession = cfg; }

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
        else for (const auto& v : flat_->all_vectors()) fn(v);
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
    // For as_of queries only links whose newer version existed at as_of
    // count; the newer version may live on another shard, hence the
    // optional cross-shard created_at_of hook.
    void _apply_supersession(std::vector<SearchResult>& results,
                             const QueryOptions& opts) const {
        apply_supersession(results, opts, [this, &opts](const std::string& id) {
            auto newer = graph_->find_superseding(id);
            if (!opts.as_of) return newer;
            std::erase_if(newer, [&](const std::string& n) {
                auto t = opts.created_at_of ? opts.created_at_of(n) : created_at_of(n);
                return !t || *t > *opts.as_of;
            });
            return newer;
        });
    }

    Config cfg_;
    std::unique_ptr<FlatIndex>   flat_;
    std::unique_ptr<TVIndex>     tv_;
    std::unique_ptr<CausalGraph> graph_;
};

} // namespace tidevec
