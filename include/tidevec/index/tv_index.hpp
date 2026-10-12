#pragma once

#include <tidevec/core/cortex_vector.hpp>
#include <tidevec/core/metrics.hpp>
#include <tidevec/core/temporal_scorer.hpp>
#include <tidevec/core/filter.hpp>
#include <tidevec/index/payload_index.hpp>
#include <tidevec/index/bm25_index.hpp>
#include <tidevec/quantization/product_quantizer.hpp>

#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <random>
#include <shared_mutex>
#include <tuple>
#include <cmath>
#include <limits>
#include <algorithm>
#include <functional>
#include <mutex>
#include <memory>
#include <thread>
#include <chrono>

namespace tidevec {

// ------------------------------------------------------------------
// TVIndex — Temporal Vector Index
//
// HNSW (Hierarchical Navigable Small World) extended with temporal scoring.
//
// Novel contribution vs all existing vector DBs:
//   · Graph traversal uses blended_score = α·cosine + β·temporal_weight
//   · Temporally fresh vectors are preferred during greedy search
//   · Staleness warnings emitted when temporal_score < threshold
//   · Hard exclusion of expired validity windows before traversal
//
// Memory profile (vs vanilla HNSW):
//   Vanilla HNSW must keep every full-precision vector resident in RAM
//   alongside O(M) neighbour pointers per node — at billion-vector
//   scale this is the dominant cost (e.g. 768-dim float32 = 3072 bytes
//   per vector before graph overhead).
//
//   When TVIndexConfig::use_pq_compression is enabled, TVIndex stores
//   an 8-bit Product Quantization code (default 96 bytes for 768-dim,
//   M=96 subspaces) instead of the raw embedding — roughly a 32x
//   reduction in resident vector memory. Graph traversal scores
//   candidates via asymmetric distance computation (ADC) against
//   these codes directly, so the coarse HNSW walk never needs the
//   full-precision vector. Final top-K results can optionally be
//   exact-rescored by supplying a VectorStore callback that fetches
//   full vectors from an external backing store (disk, object store,
//   etc) — the standard DiskANN pattern. If no VectorStore is wired
//   in, results are returned using the PQ-approximated score directly
//   (still correctly temporal-blended, just approximate on the
//   semantic-similarity component).
//
// Parameters:
//   M              — max edges per node per layer (default 16)
//   ef_construction — candidate pool size during build (default 200)
//   ml             — level multiplier = 1/ln(M) (standard HNSW)
// ------------------------------------------------------------------

// Optional callback for fetching full-precision vectors from an
// external store, used only to exact-rescore the final top-K when
// PQ compression is active. Returns empty vector if not found.
using VectorStore = std::function<std::vector<float>(const std::string& external_id)>;

struct HNSWNode {
    uint64_t    internal_id;
    std::string external_id;

    // Full-precision embedding. When PQ compression is enabled this
    // is cleared after encoding (empty() == true) to free RAM —
    // only pq_code is kept resident. When compression is disabled
    // this holds the full vector as before (unchanged behaviour).
    std::vector<float> embedding;

    // 8-bit Product Quantization code. Populated only when
    // TVIndexConfig::use_pq_compression is true. Size == M subspaces
    // (e.g. 96 bytes for a 768-dim vector at M=96), vs 3072 bytes
    // for the raw float32 embedding — ~32x smaller.
    std::vector<uint8_t> pq_code;

    std::unordered_map<std::string, std::string> payload;
    Timestamp   created_at;
    Timestamp   valid_from;
    std::optional<Timestamp> valid_until;

    // Adjacency lists per layer: neighbours[layer] = {node_id, ...}
    std::vector<std::vector<uint64_t>> neighbours;

    int level() const { return static_cast<int>(neighbours.size()) - 1; }

    // True once the node is storing only the compressed PQ code
    // (embedding has been freed).
    bool is_compressed() const { return embedding.empty() && !pq_code.empty(); }
};

struct TVIndexConfig {
    int   M               = 16;
    int   M0              = 32;
    int   ef_construction = 200;
    Metric metric         = Metric::COSINE;
    TemporalConfig temporal{};

    // ---- Memory-compressed storage (opt-in, default off) --------
    // When true, vectors are stored as Product Quantization codes
    // instead of full-precision floats, trading a small amount of
    // recall for ~32x lower resident memory at billion-vector scale.
    // The quantizer MUST be trained via TVIndex::train_quantizer()
    // before the first insert when this is enabled.
    bool   use_pq_compression = false;
    int    pq_subspaces       = 96;   // must divide embedding dim evenly

    // Level-assignment seed. Fixed so that replicas applying the same
    // write sequence build identical graphs (and therefore make
    // identical supersession decisions).
    uint32_t seed = 0x7469'6465u;

    // Payload field indexed for BM25 keyword / hybrid search (empty = off).
    std::string text_field;
    // Index every payload key (keyword + numeric) for filtered search.
    // created_at is always indexed.
    bool payload_index = true;

    // Automatic compaction runs once tombstones >= compaction_ratio * live
    // (and at least 64). With background_compaction the rebuild happens on
    // a separate thread while reads and writes continue; otherwise it runs
    // inline on the write that crossed the threshold. Background timing is
    // nondeterministic, so leave it off where replicas must build identical
    // graphs from the same write sequence.
    bool   background_compaction = false;
    double compaction_ratio      = 1.0;
};

struct CompactionStats {
    bool        running = false;
    std::size_t runs = 0;             // completed compactions (sync + background)
    std::size_t last_reclaimed = 0;   // tombstones dropped by the last run
    std::size_t last_replayed = 0;    // writes replayed onto the last rebuilt graph
    double      last_ms = 0;
};

// How a filtered search is executed (see TVIndex::explain).
struct SearchPlan {
    enum class Kind { GRAPH, EXACT_INDEX, EXACT_SCAN };
    Kind        kind = Kind::GRAPH;
    std::size_t live = 0;             // live vectors in this index
    double      estimated_matches = 0;
    double      max_matches = 0;      // upper bound on matches
    bool        indexed = false;      // estimate came from payload indexes

    std::string name() const {
        switch (kind) {
        case Kind::EXACT_INDEX: return "exact_index";
        case Kind::EXACT_SCAN:  return "exact_scan";
        default:                return "graph";
        }
    }
};

class TVIndex {
public:
    using Config = TVIndexConfig;

public:
    explicit TVIndex(Config cfg = Config{})
        : cfg_(cfg)
        , scorer_(cfg.temporal)
        , ml_(1.0 / std::log(static_cast<double>(cfg.M)))
        , rng_(cfg.seed)
        , pindex_(cfg.payload_index, cfg.text_field)
    {}

    ~TVIndex() { wait_for_compaction(); }

    // ------ memory-compressed storage (opt-in) ---------------------

    // Train the Product Quantizer on a representative sample of
    // vectors. MUST be called before the first insert() if
    // cfg.use_pq_compression is true. samples.size() of a few
    // thousand to ~100K is typically sufficient.
    void train_quantizer(const std::vector<std::vector<float>>& samples) {
        if (!cfg_.use_pq_compression) return;
        if (samples.empty())
            throw std::invalid_argument(
                "train_quantizer: samples must not be empty");
        quantizer_ = std::make_unique<ProductQuantizer>(
            samples[0].size(), static_cast<std::size_t>(cfg_.pq_subspaces));
        quantizer_->train(samples);
    }

    bool quantizer_ready() const {
        return quantizer_ && quantizer_->is_trained();
    }

    // Optional: wire in a callback to fetch full-precision vectors
    // from an external store (disk/object-store) for exact top-K
    // re-ranking when compression is active. Without this, search()
    // returns PQ-approximated scores directly.
    void set_vector_store(VectorStore store) {
        vector_store_ = std::move(store);
    }

    // Approximate resident memory for vector storage only (excludes
    // graph adjacency lists, payload, and per-node bookkeeping).
    std::size_t embedding_memory_bytes() const {
        std::shared_lock lock(mutex_);
        std::size_t total = 0;
        for (const auto& n : nodes_) {
            total += n.embedding.size() * sizeof(float);
            total += n.pq_code.size();
        }
        return total;
    }

    // ------ write -------------------------------------------------

    void insert(const CortexVector& vec) {
        std::unique_lock lock(mutex_);
        if (journal_on_) journal_.push_back({JournalOp::INSERT, vec});
        _insert_locked(vec);
    }

    void upsert(const CortexVector& vec) {
        std::unique_lock lock(mutex_);
        if (journal_on_) journal_.push_back({JournalOp::UPSERT, vec});
        auto it = ext_to_int_.find(vec.id);
        if (it != ext_to_int_.end()) {
            uint64_t iid = it->second;
            auto& node = nodes_[iid];
            bool same_position = vec.embedding.empty() ||
                (!node.is_compressed() && node.embedding == vec.embedding);
            if (same_position) {
                // Metadata-only update: the graph position is unchanged.
                deleted_.erase(iid);
                _unindex_text_locked(iid);
                node.created_at  = vec.created_at;
                node.valid_from  = vec.valid_from;
                node.valid_until = vec.valid_until;
                node.payload     = vec.payload;
                pindex_.add(iid, node.payload, node.created_at);
                _index_text_locked(iid);
                return;
            }
            // The embedding moved. Relinking the slot in place leaves
            // neighbours holding edges chosen for the old position, and
            // recall decays as updates accumulate. Instead tombstone the
            // old slot (it stays navigable at its original, consistent
            // position) and insert the new embedding as a fresh node.
            deleted_.insert(iid);
            _unindex_text_locked(iid);
            ext_to_int_.erase(it);
            _insert_locked(vec);
            _maybe_compact_locked();
        } else {
            _insert_locked(vec);
        }
    }

    bool remove(const std::string& ext_id) {
        std::unique_lock lock(mutex_);
        auto it = ext_to_int_.find(ext_id);
        if (it == ext_to_int_.end()) return false;
        if (journal_on_) _journal_remove_locked(ext_id);
        uint64_t iid = it->second;
        bool was_live = deleted_.insert(iid).second;
        _unindex_text_locked(iid);
        ext_to_int_.erase(it);
        _maybe_compact_locked();
        return was_live;
    }

    // Rebuild the graph from live vectors, dropping every tombstone.
    // Blocks writers for the whole rebuild; see compact_async().
    void compact() {
        wait_for_compaction();
        std::unique_lock lock(mutex_);
        _compact_locked();
    }

    // Rebuild the graph and payload/text indexes off the write path:
    // snapshot the live vectors, build a fresh index without holding the
    // lock (reads and writes continue against the current graph), replay
    // writes that arrived meanwhile, then swap it in under a short
    // exclusive lock. Returns false when there is nothing to reclaim or a
    // compaction is already running. PQ-compressed indexes compact inline.
    bool compact_async() {
        std::unique_lock lock(mutex_);
        if (compacting_ || deleted_.empty()) return false;
        if (cfg_.use_pq_compression) {
            _compact_locked();
            return true;
        }
        _start_compaction_locked();
        return true;
    }

    void wait_for_compaction() {
        std::thread t;
        {
            std::unique_lock lock(mutex_);
            t = std::move(compactor_);
        }
        if (t.joinable()) t.join();
    }

    CompactionStats compaction_stats() const {
        std::shared_lock lock(mutex_);
        CompactionStats s = cstats_;
        s.running = compacting_;
        return s;
    }

    std::size_t tombstones() const {
        std::shared_lock lock(mutex_);
        return deleted_.size();
    }

    // Approximate resident bytes of the graph: node slots (including
    // tombstones), embeddings/codes, adjacency lists, payloads and the
    // id map. Payload/text indexes are not counted.
    std::size_t approx_memory_bytes() const {
        std::shared_lock lock(mutex_);
        std::size_t total = nodes_.capacity() * sizeof(HNSWNode);
        for (const auto& n : nodes_) {
            total += n.external_id.capacity();
            total += n.embedding.capacity() * sizeof(float);
            total += n.pq_code.capacity();
            for (const auto& layer : n.neighbours)
                total += sizeof(layer) + layer.capacity() * sizeof(uint64_t);
            for (const auto& [k, v] : n.payload)
                total += 64 + k.capacity() + v.capacity();
        }
        for (const auto& [k, _] : ext_to_int_) total += 48 + k.capacity();
        total += deleted_.size() * 24;
        return total;
    }

    // ------ search ------------------------------------------------

    // Remove vectors whose validity window has passed.
    std::size_t purge_expired(Timestamp qt = now_ms()) {
        std::unique_lock lock(mutex_);
        std::size_t n = 0;
        for (const auto& node : nodes_) {
            if (deleted_.count(node.internal_id)) continue;
            if (node.valid_until.has_value() && qt > node.valid_until.value()) {
                if (journal_on_) _journal_remove_locked(node.external_id);
                deleted_.insert(node.internal_id);
                _unindex_text_locked(node.internal_id);
                ++n;
            }
        }
        if (n) _maybe_compact_locked();
        return n;
    }

    std::vector<SearchResult> search(const std::vector<float>& query,
                                     const QueryOptions& opts) const {
        std::shared_lock lock(mutex_);
        if (nodes_.empty() || entry_point_ < 0) return {};

        Timestamp qt = opts.as_of ? *opts.as_of : now_ms();
        Metric m = parse_metric(opts.metric);
        int ef = std::max(opts.ef_search, opts.top_k);
        float blend = opts.blend_or(cfg_.temporal.temporal_blend);
        const FilterPtr filter = effective_filter(opts);

        // Precompute the ADC distance table once per query if
        // compression is active — turns each candidate scoring from
        // O(D) full-precision distance into O(M) table lookups.
        std::unique_ptr<ProductQuantizer::DistanceTable> adc_table;
        if (cfg_.use_pq_compression && quantizer_ready())
            adc_table = std::make_unique<ProductQuantizer::DistanceTable>(
                quantizer_->precompute_adc(query));

        // Filtered queries: scan the matching vectors exactly when that
        // is cheaper than walking the graph past the non-matching ones.
        SearchPlan plan;
        std::vector<uint64_t> index_cands;
        if (filter) {
            plan = _plan_locked(*filter, ef);
            if (plan.kind == SearchPlan::Kind::EXACT_INDEX)
                pindex_.collect(*filter, _id_lookup(), index_cands);
        }

        std::vector<std::pair<float, uint64_t>> candidates;
        if (plan.kind == SearchPlan::Kind::GRAPH) {
            // Start greedy search from top layer down to layer 1
            uint64_t ep = static_cast<uint64_t>(entry_point_);
            int top_layer = nodes_[ep].level();

            for (int layer = top_layer; layer >= 1; --layer) {
                ep = _greedy_closest(query, ep, layer, qt, m, blend, adc_table.get());
            }

            // Layer 0: beam search with ef candidates
            candidates = _beam_search(query, ep, 0, ef, qt, m, blend,
                                      /*for_search=*/true, filter.get(),
                                      adc_table.get());
            // A filter whose matches the beam could not reach (e.g. they
            // sit in a region the walk never entered): fall back to exact.
            if (filter && candidates.size() < static_cast<std::size_t>(opts.top_k) &&
                plan.max_matches > static_cast<double>(candidates.size()))
                candidates = _exact_search(query, opts.top_k, qt, m, blend, filter.get(),
                                           nullptr, adc_table.get());
        } else {
            candidates = _exact_search(query, opts.top_k, qt, m, blend, filter.get(),
                plan.kind == SearchPlan::Kind::EXACT_INDEX ? &index_cands : nullptr,
                adc_table.get());
        }

        // Take top_k
        int k = std::min(opts.top_k, static_cast<int>(candidates.size()));
        std::vector<SearchResult> results;
        results.reserve(k);

        for (int i = 0; i < k; ++i) {
            auto [score, iid] = candidates[i];
            const auto& node = nodes_[iid];
            if (node.valid_until.has_value() && qt > node.valid_until.value())
                continue;

            float raw_sim;
            bool  approximate = false;

            if (node.is_compressed() && vector_store_) {
                // Exact rescore: fetch the full-precision vector
                // from the backing store for the final top-K only —
                // this is the DiskANN pattern. The expensive fetch
                // only happens for the handful of results actually
                // returned, not the whole candidate pool scanned
                // during graph traversal.
                auto full_vec = vector_store_(node.external_id);
                if (!full_vec.empty()) {
                    raw_sim = compute_similarity(query, full_vec, m);
                } else {
                    raw_sim = compute_similarity(query,
                        quantizer_->decode(node.pq_code), m);
                    approximate = true;
                }
            } else if (node.is_compressed()) {
                // No backing store wired in — approximate via decode.
                raw_sim = compute_similarity(query,
                    quantizer_->decode(node.pq_code), m);
                approximate = true;
            } else {
                raw_sim = compute_similarity(query, node.embedding, m);
            }

            auto ts = scorer_.score_raw(node.created_at, raw_sim, qt);
            ts.final_score = (1.0f - blend) * raw_sim + blend * ts.temporal_weight;

            SearchResult r;
            r.id             = node.external_id;
            r.score          = ts.final_score;
            r.vector_score   = ts.vector_score;
            r.temporal_score = ts.temporal_weight;
            r.payload        = node.payload;
            r.created_at     = node.created_at;
            if (opts.include_staleness_warnings) {
                r.staleness_warning = ts.staleness_warning;
                r.staleness_reason  = ts.staleness_reason;
            }
            if (approximate) {
                // Surface that this score came from a decoded PQ
                // approximation rather than the exact vector, so
                // callers relying on precise ranking can tell.
                r.payload["_tidevec_approx_score"] = "true";
            }
            results.push_back(std::move(r));
        }
        std::stable_sort(results.begin(), results.end(),
            [](const SearchResult& a, const SearchResult& b) {
                return a.score > b.score;
            });
        return results;
    }

    // ------ stats -------------------------------------------------

    std::size_t size() const {
        std::shared_lock lock(mutex_);
        return nodes_.size() - deleted_.size();
    }

    bool contains(const std::string& ext_id) const {
        std::shared_lock lock(mutex_);
        auto it = ext_to_int_.find(ext_id);
        return it != ext_to_int_.end() && !deleted_.count(it->second);
    }

    int max_layer() const {
        std::shared_lock lock(mutex_);
        if (entry_point_ < 0) return -1;
        return nodes_[entry_point_].level();
    }

    // Iterate every non-deleted vector — used by DriftBridge to
    // snapshot the live index before migration begins.
    void each_vector(std::function<void(const CortexVector&)> fn) const {
        std::shared_lock lock(mutex_);
        for (const auto& node : nodes_) {
            if (deleted_.count(node.internal_id)) continue;
            CortexVector v;
            v.id         = node.external_id;
            v.embedding  = node.is_compressed()
                               ? quantizer_->decode(node.pq_code)
                               : node.embedding;
            v.payload    = node.payload;
            v.created_at = node.created_at;
            v.valid_from = node.valid_from;
            v.valid_until= node.valid_until;
            fn(v);
        }
    }

    // ------ point reads / iteration --------------------------------

    // A live (not deleted, not expired) vector by id.
    std::optional<CortexVector> get(const std::string& ext_id,
                                    bool with_vector = true) const {
        std::shared_lock lock(mutex_);
        auto it = ext_to_int_.find(ext_id);
        if (it == ext_to_int_.end() || !_live_locked(it->second, now_ms()))
            return std::nullopt;
        return _to_vector_locked(it->second, with_vector);
    }

    std::optional<Timestamp> created_at_of(const std::string& ext_id) const {
        std::shared_lock lock(mutex_);
        auto it = ext_to_int_.find(ext_id);
        if (it == ext_to_int_.end() || deleted_.count(it->second)) return std::nullopt;
        return nodes_[it->second].created_at;
    }

    // Up to `limit` live vectors with id > `after`, in id order, that
    // match the query's filter / as_of.
    std::vector<CortexVector> scroll(const std::string& after, std::size_t limit,
                                     const QueryOptions& opts,
                                     bool with_vector = false) const {
        std::shared_lock lock(mutex_);
        const FilterPtr filter = effective_filter(opts);
        const Timestamp now = now_ms();
        std::vector<CortexVector> out;
        for (auto it = ext_to_int_.upper_bound(after);
             it != ext_to_int_.end() && out.size() < limit; ++it) {
            const uint64_t iid = it->second;
            if (!_live_locked(iid, now)) continue;
            const auto& n = nodes_[iid];
            if (filter && !filter->matches(n.external_id, n.payload, n.created_at)) continue;
            out.push_back(_to_vector_locked(iid, with_vector));
        }
        return out;
    }

    // BM25 keyword search over cfg.text_field (empty when not configured).
    std::vector<SearchResult> text_search(const std::string& text,
                                          const QueryOptions& opts) const {
        std::shared_lock lock(mutex_);
        if (cfg_.text_field.empty() || text.empty()) return {};
        const Timestamp qt = opts.as_of ? *opts.as_of : now_ms();
        const FilterPtr filter = effective_filter(opts);
        auto hits = bm25_.search(text, static_cast<std::size_t>(std::max(0, opts.top_k)),
            [&](uint64_t iid) { return _eligible_locked(iid, qt, filter.get()); });
        std::vector<SearchResult> results;
        results.reserve(hits.size());
        for (const auto& [score, iid] : hits) {
            const auto& node = nodes_[iid];
            auto ts = scorer_.score_raw(node.created_at, 0.0f, qt);
            SearchResult r;
            r.id             = node.external_id;
            r.score          = score;
            r.text_score     = score;
            r.vector_score   = 0.0f;
            r.temporal_score = ts.temporal_weight;
            r.payload        = node.payload;
            r.created_at     = node.created_at;
            if (opts.include_staleness_warnings) {
                r.staleness_warning = ts.staleness_warning;
                r.staleness_reason  = ts.staleness_reason;
            }
            results.push_back(std::move(r));
        }
        return results;
    }

    const std::string& text_field() const { return cfg_.text_field; }

    // The plan search() would use for these options.
    SearchPlan explain(const QueryOptions& opts) const {
        std::shared_lock lock(mutex_);
        const FilterPtr filter = effective_filter(opts);
        SearchPlan plan;
        plan.live = nodes_.size() - deleted_.size();
        if (!filter) return plan;
        return _plan_locked(*filter, std::max(opts.ef_search, opts.top_k));
    }

    void set_temporal_config(const TemporalConfig& cfg) {
        scorer_ = TemporalScorer(cfg);
        cfg_.temporal = cfg;
    }

private:
    // ------ internals (called under write lock) -------------------

    void _insert_locked(const CortexVector& vec) {
        if (vec.embedding.empty()) return;
        if (cfg_.use_pq_compression && !quantizer_ready())
            throw std::runtime_error(
                "TVIndex: use_pq_compression is enabled but "
                "train_quantizer() was not called before insert");

        uint64_t new_id = static_cast<uint64_t>(nodes_.size());
        int new_level   = _random_level();

        HNSWNode node;
        node.internal_id  = new_id;
        node.external_id  = vec.id;
        node.payload      = vec.payload;
        node.created_at   = vec.created_at;
        node.valid_from   = vec.valid_from;
        node.valid_until  = vec.valid_until;
        node.neighbours.resize(new_level + 1);

        if (cfg_.use_pq_compression) {
            // Store the compressed code only — full embedding is
            // used transiently below for graph linking, then freed.
            node.pq_code = quantizer_->encode(vec.embedding);
        } else {
            node.embedding = vec.embedding;
        }

        nodes_.push_back(std::move(node));
        ext_to_int_[vec.id] = new_id;
        pindex_.add(new_id, vec.payload, vec.created_at);
        _index_text_locked(new_id);

        if (entry_point_ < 0) {
            entry_point_ = static_cast<int64_t>(new_id);
            return;
        }

        // Graph linking still needs a query vector to score against.
        // Use the original full-precision embedding for build-time
        // accuracy (it's only transiently in scope here, not stored).
        const std::vector<float>& link_query = vec.embedding;

        Timestamp qt = vec.created_at;
        Metric m = cfg_.metric;
        uint64_t ep = static_cast<uint64_t>(entry_point_);
        int top_layer = nodes_[ep].level();

        // Descend to new_level+1
        for (int layer = top_layer; layer > new_level; --layer)
            ep = _greedy_closest(link_query, ep, layer, qt, m, 0.0f);

        // Connect at each layer from new_level down to 0
        for (int layer = std::min(new_level, top_layer); layer >= 0; --layer) {
            int M_layer = (layer == 0) ? cfg_.M0 : cfg_.M;
            auto cands = _beam_search(link_query, ep, layer,
                                      cfg_.ef_construction, qt, m,
                                      0.0f, /*for_search=*/false, nullptr);

            for (uint64_t nb : _select_neighbours(cands, M_layer, m)) {
                nodes_[new_id].neighbours[layer].push_back(nb);
                nodes_[nb].neighbours[layer].push_back(new_id);
                // Prune neighbours to M_layer if overfull
                _prune_neighbours(nb, layer, M_layer, m);
            }
            if (!cands.empty()) ep = cands[0].second;
        }

        // Update entry point if new node has higher level
        if (new_level > top_layer)
            entry_point_ = static_cast<int64_t>(new_id);
    }

    // Tombstones stay navigable, but a graph made mostly of them loses
    // the links into the live region (new nodes only link to live ones),
    // so whole-collection re-embeds can strand queries. Rebuilding when
    // tombstones reach a fixed fraction of the live count keeps the cost
    // amortised O(1) rebuilt node per delete.
    void _maybe_compact_locked() {
        if (compacting_) return;
        const std::size_t dead = deleted_.size();
        const double live = static_cast<double>(nodes_.size() - dead);
        if (dead < kCompactMinTombstones ||
            static_cast<double>(dead) < cfg_.compaction_ratio * live)
            return;
        if (cfg_.background_compaction && !cfg_.use_pq_compression)
            _start_compaction_locked();
        else
            _compact_locked();
    }

    void _compact_locked() {
        if (deleted_.empty()) return;
        const auto t0 = std::chrono::steady_clock::now();
        const std::size_t reclaimed = deleted_.size();
        std::vector<CortexVector> live = _live_vectors_locked();
        nodes_.clear();
        ext_to_int_.clear();
        deleted_.clear();
        pindex_.clear();
        bm25_.clear();
        entry_point_ = -1;
        for (const auto& v : live) _insert_locked(v);
        _record_compaction_locked(t0, reclaimed, 0);
    }

    struct JournalOp {
        enum Kind { INSERT, UPSERT, REMOVE } kind;
        CortexVector vec;   // REMOVE: only id is set
    };

    void _journal_remove_locked(const std::string& id) {
        CortexVector v;
        v.id = id;
        journal_.push_back({JournalOp::REMOVE, std::move(v)});
    }

    void _record_compaction_locked(std::chrono::steady_clock::time_point t0,
                                   std::size_t reclaimed, std::size_t replayed) {
        cstats_.runs += 1;
        cstats_.last_reclaimed = reclaimed;
        cstats_.last_replayed  = replayed;
        cstats_.last_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
    }

    void _start_compaction_locked() {
        // The previous worker clears compacting_ in its final critical
        // section, so joining here (under the lock) never waits on it.
        if (compactor_.joinable()) compactor_.join();
        compacting_  = true;
        journal_on_  = true;
        journal_.clear();
        Config shadow_cfg = cfg_;
        shadow_cfg.background_compaction = false;
        compactor_ = std::thread(
            [this, shadow_cfg, snap = _live_vectors_locked(),
             reclaimed = deleted_.size(),
             t0 = std::chrono::steady_clock::now()]() mutable {
                _run_compaction(shadow_cfg, std::move(snap), reclaimed, t0);
            });
    }

    static void _replay(TVIndex& shadow, std::vector<JournalOp>& ops) {
        for (auto& op : ops) {
            switch (op.kind) {
            case JournalOp::INSERT: shadow.insert(op.vec); break;
            case JournalOp::UPSERT: shadow.upsert(op.vec); break;
            case JournalOp::REMOVE: shadow.remove(op.vec.id); break;
            }
        }
    }

    void _run_compaction(const Config& shadow_cfg, std::vector<CortexVector> snap,
                         std::size_t reclaimed,
                         std::chrono::steady_clock::time_point t0) {
        std::size_t replayed = 0;
        try {
            TVIndex shadow(shadow_cfg);
            for (const auto& v : snap) shadow._insert_locked(v);
            snap.clear();
            snap.shrink_to_fit();

            // Catch up on concurrent writes without blocking them; the last
            // short tail is applied under the exclusive lock with the swap.
            for (int round = 0; round < kCompactCatchUpRounds; ++round) {
                std::vector<JournalOp> batch;
                {
                    std::unique_lock lock(mutex_);
                    if (journal_.size() <= kCompactFinalReplay) break;
                    batch.swap(journal_);
                }
                replayed += batch.size();
                _replay(shadow, batch);
            }

            std::unique_lock lock(mutex_);
            replayed += journal_.size();
            _replay(shadow, journal_);
            journal_.clear();
            journal_.shrink_to_fit();
            journal_on_ = false;
            nodes_       = std::move(shadow.nodes_);
            ext_to_int_  = std::move(shadow.ext_to_int_);
            deleted_     = std::move(shadow.deleted_);
            entry_point_ = shadow.entry_point_;
            pindex_      = std::move(shadow.pindex_);
            bm25_        = std::move(shadow.bm25_);
            rng_         = shadow.rng_;
            _record_compaction_locked(t0, reclaimed, replayed);
            compacting_ = false;
        } catch (...) {
            std::unique_lock lock(mutex_);
            journal_.clear();
            journal_on_ = false;
            compacting_ = false;
        }
    }

    std::vector<CortexVector> _live_vectors_locked() const {
        std::vector<CortexVector> live;
        live.reserve(nodes_.size() - deleted_.size());
        for (const auto& node : nodes_) {
            if (deleted_.count(node.internal_id)) continue;
            CortexVector v;
            v.id = node.external_id;
            if (!node.is_compressed()) {
                v.embedding = node.embedding;
            } else {
                if (vector_store_) v.embedding = vector_store_(node.external_id);
                if (v.embedding.empty()) v.embedding = quantizer_->decode(node.pq_code);
            }
            v.payload     = node.payload;
            v.created_at  = node.created_at;
            v.valid_from  = node.valid_from;
            v.valid_until = node.valid_until;
            live.push_back(std::move(v));
        }
        return live;
    }

    // ------ filtered-search internals (called under a lock) ---------

    bool _live_locked(uint64_t iid, Timestamp now) const {
        if (deleted_.count(iid)) return false;
        const auto& n = nodes_[iid];
        return !(n.valid_until.has_value() && now > n.valid_until.value());
    }

    bool _eligible_locked(uint64_t iid, Timestamp qt, const FilterExpr* filter) const {
        if (deleted_.count(iid)) return false;
        const auto& n = nodes_[iid];
        if (n.valid_until.has_value() && qt > n.valid_until.value()) return false;
        if (filter && !filter->matches(n.external_id, n.payload, n.created_at)) return false;
        if (scorer_.is_hard_excluded_raw(n.created_at, qt)) return false;
        return true;
    }

    CortexVector _to_vector_locked(uint64_t iid, bool with_vector) const {
        const auto& node = nodes_[iid];
        CortexVector v;
        v.id          = node.external_id;
        v.payload     = node.payload;
        v.created_at  = node.created_at;
        v.valid_from  = node.valid_from;
        v.valid_until = node.valid_until;
        if (with_vector) {
            if (!node.is_compressed()) v.embedding = node.embedding;
            else {
                if (vector_store_) v.embedding = vector_store_(node.external_id);
                if (v.embedding.empty()) v.embedding = quantizer_->decode(node.pq_code);
            }
        }
        return v;
    }

    PayloadIndex::IdLookup _id_lookup() const {
        return [this](const std::string& id) -> std::optional<uint64_t> {
            auto it = ext_to_int_.find(id);
            if (it == ext_to_int_.end()) return std::nullopt;
            return it->second;
        };
    }

    // Cost model, in units of one distance computation. A filtered beam
    // search has to expand roughly ef / selectivity nodes (each costing
    // ~kGraphFanout distances, most neighbours being already visited) to
    // collect ef matches; an exact scan costs one distance per match plus
    // a cheap predicate check per vector it has to look at (all of them,
    // unless the payload indexes narrow it). kGraphFanout is calibrated
    // against measured latencies on 100k-vector shards.
    static constexpr std::size_t kExactBelow    = 2048;
    static constexpr double      kGraphFanout   = 6.0;
    static constexpr double      kPredicateCost = 0.1;
    static constexpr std::size_t kSelectivitySample = 1024;

    SearchPlan _plan_locked(const FilterExpr& filter, int ef) const {
        SearchPlan plan;
        plan.live = nodes_.size() - deleted_.size();
        const double live = static_cast<double>(std::max<std::size_t>(plan.live, 1));

        // Candidates an indexed exact search walks (an upper bound on matches).
        double candidates = 0.0;
        const auto ids = _id_lookup();
        if (auto est = pindex_.estimate(filter, ids)) {
            plan.indexed = true;
            candidates = std::min(static_cast<double>(*est), live);
            plan.estimated_matches =
                std::min(pindex_.expected(filter, ids, live).value_or(candidates), candidates);
        } else {
            // Sample evenly spaced slots to estimate selectivity.
            const std::size_t n = nodes_.size();
            const std::size_t step = std::max<std::size_t>(1, n / kSelectivitySample);
            std::size_t seen = 0, hit = 0;
            for (std::size_t i = 0; i < n; i += step) {
                if (deleted_.count(i)) continue;
                const auto& node = nodes_[i];
                ++seen;
                if (filter.matches(node.external_id, node.payload, node.created_at)) ++hit;
            }
            plan.estimated_matches = seen ? live * static_cast<double>(hit) / seen : 0.0;
        }
        plan.max_matches = plan.indexed ? candidates : static_cast<double>(plan.live);

        if (plan.live <= kExactBelow) {
            plan.kind = plan.indexed ? SearchPlan::Kind::EXACT_INDEX
                                     : SearchPlan::Kind::EXACT_SCAN;
            return plan;
        }
        // Sampling can miss a rare value entirely; assume half a hit.
        const double matches = std::max(plan.estimated_matches,
                                         plan.indexed ? 0.0 : live / (2.0 * kSelectivitySample));
        const double sel = std::max(matches / live, 1e-9);
        const double graph_cost = std::min(live, ef / sel) * kGraphFanout;
        const double exact_cost = plan.indexed
            ? plan.estimated_matches + candidates * kPredicateCost
            : live * kPredicateCost + matches;
        if (exact_cost <= graph_cost)
            plan.kind = plan.indexed ? SearchPlan::Kind::EXACT_INDEX
                                     : SearchPlan::Kind::EXACT_SCAN;
        return plan;
    }

    // Brute-force top-k over `cands` (or every node) that pass the filter.
    std::vector<std::pair<float, uint64_t>>
    _exact_search(const std::vector<float>& query, int k, Timestamp qt, Metric m,
                  float blend, const FilterExpr* filter,
                  std::vector<uint64_t>* cands,
                  const ProductQuantizer::DistanceTable* adc) const {
        using P = std::pair<float, uint64_t>;
        std::priority_queue<P, std::vector<P>, std::greater<P>> top;
        const std::size_t kk = static_cast<std::size_t>(std::max(k, 1));
        auto consider = [&](uint64_t iid) {
            if (!_eligible_locked(iid, qt, filter)) return;
            float s = _blended_score(query, iid, qt, m, blend, adc);
            if (top.size() < kk) top.push({s, iid});
            else if (s > top.top().first) { top.pop(); top.push({s, iid}); }
        };
        if (cands) {
            std::sort(cands->begin(), cands->end());
            cands->erase(std::unique(cands->begin(), cands->end()), cands->end());
            for (uint64_t iid : *cands) if (iid < nodes_.size()) consider(iid);
        } else {
            for (uint64_t iid = 0; iid < nodes_.size(); ++iid) consider(iid);
        }
        std::vector<P> out;
        out.reserve(top.size());
        while (!top.empty()) { out.push_back(top.top()); top.pop(); }
        std::reverse(out.begin(), out.end());
        return out;
    }

    const std::string* _text_of_locked(uint64_t iid) const {
        if (cfg_.text_field.empty()) return nullptr;
        const auto& p = nodes_[iid].payload;
        auto it = p.find(cfg_.text_field);
        return it == p.end() ? nullptr : &it->second;
    }

    void _index_text_locked(uint64_t iid) {
        if (const auto* t = _text_of_locked(iid)) bm25_.add(iid, *t);
    }

    void _unindex_text_locked(uint64_t iid) {
        if (const auto* t = _text_of_locked(iid)) bm25_.remove(iid, *t);
    }

    static constexpr std::size_t kCompactMinTombstones = 64;
    static constexpr int         kCompactCatchUpRounds = 8;
    static constexpr std::size_t kCompactFinalReplay   = 256;

    int _random_level() {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        int level = 0;
        while (dist(rng_) < (1.0 / static_cast<double>(cfg_.M)) && level < 32)
            ++level;
        return level;
    }

    // Greedy single-step closest at given layer
    uint64_t _greedy_closest(const std::vector<float>& query,
                              uint64_t start, int layer,
                              Timestamp qt, Metric m, float blend,
                              const ProductQuantizer::DistanceTable* adc = nullptr) const {
        uint64_t cur = start;
        float cur_score = _blended_score(query, cur, qt, m, blend, adc);
        bool improved = true;
        while (improved) {
            improved = false;
            for (uint64_t nb : nodes_[cur].neighbours[layer]) {
                float s = _blended_score(query, nb, qt, m, blend, adc);
                if (s > cur_score) { cur_score = s; cur = nb; improved = true; }
            }
        }
        return cur;
    }

    // Beam search — returns sorted (score, id) pairs, best first
    std::vector<std::pair<float, uint64_t>>
    _beam_search(const std::vector<float>& query,
                 uint64_t ep, int layer, int ef,
                 Timestamp qt, Metric m, float blend, bool for_search,
                 const FilterExpr* filter,
                 const ProductQuantizer::DistanceTable* adc = nullptr) const {

        using P = std::pair<float, uint64_t>;

        // Deleted, expired, filtered-out and hard-excluded nodes stay
        // navigable (they hold graph connectivity) but never enter W.
        auto eligible = [&](uint64_t id) {
            if (!for_search) return !deleted_.count(id);
            return _eligible_locked(id, qt, filter);
        };

        std::priority_queue<P> C;                                        // best candidate on top
        std::priority_queue<P, std::vector<P>, std::greater<P>> W;       // worst result on top
        std::unordered_set<uint64_t> visited;

        float ep_score = _blended_score(query, ep, qt, m, blend, adc);
        C.push({ep_score, ep});
        if (eligible(ep)) W.push({ep_score, ep});
        visited.insert(ep);

        const auto full = [&] { return W.size() >= static_cast<std::size_t>(ef); };

        while (!C.empty()) {
            auto [c_score, c] = C.top(); C.pop();
            if (full() && c_score < W.top().first) break;

            for (uint64_t nb : nodes_[c].neighbours[layer]) {
                if (!visited.insert(nb).second) continue;
                float s = _blended_score(query, nb, qt, m, blend, adc);
                if (full() && s <= W.top().first) continue;
                C.push({s, nb});
                if (eligible(nb)) {
                    W.push({s, nb});
                    if (W.size() > static_cast<std::size_t>(ef)) W.pop();
                }
            }
        }

        // Drain heap into sorted vector
        std::vector<P> result;
        result.reserve(W.size());
        while (!W.empty()) { result.push_back(W.top()); W.pop(); }
        std::sort(result.begin(), result.end(),
                  [](const P& a, const P& b){ return a.first > b.first; });
        return result;
    }

    // Computes the blended (semantic + temporal) score for a node.
    // If the node is PQ-compressed and a DistanceTable is supplied,
    // uses fast O(M) asymmetric distance computation instead of full
    // O(D) distance — this is what makes compressed search fast.
    float _blended_score(const std::vector<float>& query,
                         uint64_t iid, Timestamp qt, Metric m, float blend,
                         const ProductQuantizer::DistanceTable* adc = nullptr) const {
        const auto& node = nodes_[iid];
        float sim;
        if (node.is_compressed() && adc != nullptr) {
            // ADC gives squared L2 distance over the quantized code;
            // convert to the same [0,1] similarity space as
            // compute_similarity() for consistent blending.
            float d2 = quantizer_->adc_distance(*adc, node.pq_code);
            sim = 1.0f / (1.0f + d2);
        } else if (node.is_compressed()) {
            // No distance table supplied (e.g. called from a context
            // without precompute_adc) — decode and score exactly.
            // Slower path, used rarely (e.g. single ad-hoc lookups).
            auto decoded = quantizer_->decode(node.pq_code);
            sim = compute_similarity(query, decoded, m);
        } else {
            sim = compute_similarity(query, node.embedding, m);
        }
        if (blend <= 0.0f) return sim;
        float tw = scorer_.temporal_weight_raw(node.created_at, qt);
        return (1.0f - blend) * sim + blend * tw;
    }

    std::vector<float> _embedding_of(uint64_t iid) const {
        const auto& n = nodes_[iid];
        return n.is_compressed() ? quantizer_->decode(n.pq_code) : n.embedding;
    }

    float _sim_between(uint64_t a, uint64_t b, Metric m) const {
        const auto& na = nodes_[a];
        const auto& nb = nodes_[b];
        if (!na.is_compressed() && !nb.is_compressed())
            return compute_similarity(na.embedding, nb.embedding, m);
        return compute_similarity(_embedding_of(a), _embedding_of(b), m);
    }

    // HNSW neighbour-selection heuristic (Malkov & Yashunin, Alg. 4).
    // cands holds (similarity to the base node, id), best first. A
    // candidate is kept only if it is closer to the base than to every
    // neighbour already kept, so links point in diverse directions;
    // plain top-M selection drops every edge into outlying nodes and
    // leaves them unreachable. Remaining slots are filled best-first.
    std::vector<uint64_t> _select_neighbours(
            const std::vector<std::pair<float, uint64_t>>& cands,
            int M_max, Metric m) const {
        std::vector<uint64_t> kept, skipped;
        kept.reserve(M_max);
        for (const auto& [sim, c] : cands) {
            if (static_cast<int>(kept.size()) >= M_max) break;
            bool diverse = true;
            for (uint64_t r : kept)
                if (_sim_between(c, r, m) > sim) { diverse = false; break; }
            (diverse ? kept : skipped).push_back(c);
        }
        for (uint64_t c : skipped) {
            if (static_cast<int>(kept.size()) >= M_max) break;
            kept.push_back(c);
        }
        return kept;
    }

    void _prune_neighbours(uint64_t node_id, int layer, int M_max, Metric m) {
        auto& nbs = nodes_[node_id].neighbours[layer];
        if (static_cast<int>(nbs.size()) <= M_max) return;

        // Tombstoned neighbours are considered last: keeping links to
        // live nodes is what keeps the live part of the graph reachable.
        const std::vector<float> self = _embedding_of(node_id);
        std::vector<std::tuple<bool, float, uint64_t>> scored;
        scored.reserve(nbs.size());
        for (uint64_t nb : nbs)
            scored.emplace_back(!deleted_.count(nb),
                                _blended_score(self, nb, 0, m, 0.0f), nb);
        std::sort(scored.begin(), scored.end(),
                  [](const auto& a, const auto& b) { return a > b; });
        std::vector<std::pair<float, uint64_t>> live, dead;
        for (const auto& [is_live, sim, id] : scored)
            (is_live ? live : dead).emplace_back(sim, id);
        nbs = _select_neighbours(live, M_max, m);
        for (const auto& [sim, id] : dead) {
            if (static_cast<int>(nbs.size()) >= M_max) break;
            nbs.push_back(id);
        }
    }

    Config                cfg_;
    TemporalScorer        scorer_;
    double                ml_;
    mutable std::mt19937  rng_;

    std::unique_ptr<ProductQuantizer> quantizer_;  // null until trained
    VectorStore                       vector_store_;  // optional, for exact rescore

    std::vector<HNSWNode>                    nodes_;
    std::map<std::string, uint64_t>          ext_to_int_;   // ordered: scroll
    std::unordered_set<uint64_t>              deleted_;
    int64_t                                   entry_point_ = -1;
    PayloadIndex                              pindex_;
    Bm25Index                                 bm25_;

    mutable std::shared_mutex mutex_;

    // Background compaction (all guarded by mutex_ except the worker's
    // unlocked shadow build).
    bool                   compacting_ = false;
    bool                   journal_on_ = false;
    std::vector<JournalOp> journal_;
    CompactionStats        cstats_;
    std::thread            compactor_;
};

} // namespace tidevec
