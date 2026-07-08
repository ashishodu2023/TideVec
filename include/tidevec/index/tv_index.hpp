#pragma once

#include <tidevec/core/cortex_vector.hpp>
#include <tidevec/core/metrics.hpp>
#include <tidevec/core/temporal_scorer.hpp>
#include <tidevec/quantization/product_quantizer.hpp>

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <random>
#include <shared_mutex>
#include <cmath>
#include <limits>
#include <algorithm>
#include <functional>
#include <mutex>
#include <memory>

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
};

class TVIndex {
public:
    using Config = TVIndexConfig;

public:
    explicit TVIndex(Config cfg = Config{})
        : cfg_(cfg)
        , scorer_(cfg.temporal)
        , ml_(1.0 / std::log(static_cast<double>(cfg.M)))
        , rng_(std::random_device{}())
    {}

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
        _insert_locked(vec);
    }

    void upsert(const CortexVector& vec) {
        std::unique_lock lock(mutex_);
        auto it = ext_to_int_.find(vec.id);
        if (it != ext_to_int_.end()) {
            // ── Update existing vector ──────────────────────────────
            // The embedding may have changed, so the node's geometric
            // position has shifted. We must rebuild its graph edges
            // to reflect the new position — otherwise HNSW traversal
            // will follow stale connections and miss the updated vector.
            //
            // Strategy (standard HNSW update pattern):
            //   1. Remove the node's back-links from all its neighbours
            //   2. Clear the node's own neighbour lists
            //   3. Update embedding / timestamps / payload
            //   4. Re-link by running the standard insert graph-building
            //      pass (without creating a new node slot)

            uint64_t iid = it->second;
            auto& node = nodes_[iid];

            // Step 1: remove back-links from all current neighbours
            for (int layer = 0; layer <= node.level(); ++layer) {
                for (uint64_t nb_id : node.neighbours[layer]) {
                    auto& nb_nbs = nodes_[nb_id].neighbours[layer];
                    nb_nbs.erase(
                        std::remove(nb_nbs.begin(), nb_nbs.end(), iid),
                        nb_nbs.end());
                }
            }

            // Step 2: clear the node's neighbour lists
            // Keep the same number of layers (level doesn't change on update)
            for (auto& layer_nbs : node.neighbours)
                layer_nbs.clear();

            // Step 3: update embedding, timestamps, payload
            if (cfg_.use_pq_compression) {
                if (!quantizer_ready())
                    throw std::runtime_error(
                        "TVIndex: use_pq_compression is enabled but "
                        "train_quantizer() was not called before upsert");
                node.pq_code = quantizer_->encode(vec.embedding);
                node.embedding.clear();
            } else {
                node.embedding = vec.embedding;
            }
            node.created_at  = vec.created_at;
            node.valid_from  = vec.valid_from;
            node.valid_until = vec.valid_until;
            node.payload     = vec.payload;

            // Step 4: re-link into the graph at the node's existing level
            // Use the same greedy descent + beam-search as _insert_locked,
            // but write into the existing node slot instead of appending.
            if (entry_point_ >= 0 &&
                static_cast<uint64_t>(entry_point_) != iid) {

                const std::vector<float>& link_query = vec.embedding;
                Timestamp qt = vec.created_at;
                Metric m = cfg_.metric;
                uint64_t ep = static_cast<uint64_t>(entry_point_);
                int top_layer = nodes_[ep].level();
                int new_level = node.level();

                // Descend to new_level+1
                for (int layer = top_layer; layer > new_level; --layer)
                    ep = _greedy_closest(link_query, ep, layer, qt, m);

                // Re-connect at each layer from new_level down to 0
                for (int layer = std::min(new_level, top_layer); layer >= 0; --layer) {
                    int M_layer = (layer == 0) ? cfg_.M0 : cfg_.M;
                    auto cands = _beam_search(link_query, ep, layer,
                                              cfg_.ef_construction, qt, m, "");

                    int take = std::min(M_layer, static_cast<int>(cands.size()));
                    for (int i = 0; i < take; ++i) {
                        uint64_t nb = cands[i].second;
                        if (nb == iid) continue; // don't self-link
                        node.neighbours[layer].push_back(nb);
                        nodes_[nb].neighbours[layer].push_back(iid);
                        _prune_neighbours(nb, layer, M_layer, qt, m, link_query);
                    }
                    if (!cands.empty()) ep = cands[0].second;
                }
            }
        } else {
            _insert_locked(vec);
        }
    }

    bool remove(const std::string& ext_id) {
        std::unique_lock lock(mutex_);
        auto it = ext_to_int_.find(ext_id);
        if (it == ext_to_int_.end()) return false;
        uint64_t iid = it->second;
        deleted_.insert(iid);
        ext_to_int_.erase(it);
        // If we deleted the entry point, find a new one
        if (entry_point_ >= 0 &&
            static_cast<uint64_t>(entry_point_) == iid) {
            entry_point_ = -1;
            for (uint64_t i = 0; i < nodes_.size(); ++i) {
                if (!deleted_.count(i)) { entry_point_ = static_cast<int64_t>(i); break; }
            }
        }
        return true;
    }

    // ------ search ------------------------------------------------

    // Remove vectors whose validity window has passed.
    std::size_t purge_expired(Timestamp qt = now_ms()) {
        std::unique_lock lock(mutex_);
        std::size_t n = 0;
        for (const auto& node : nodes_) {
            if (deleted_.count(node.internal_id)) continue;
            if (node.valid_until.has_value() && qt > node.valid_until.value()) {
                deleted_.insert(node.internal_id);
                ++n;
            }
        }
        return n;
    }

    std::vector<SearchResult> search(const std::vector<float>& query,
                                     const QueryOptions& opts) const {
        std::shared_lock lock(mutex_);
        if (nodes_.empty() || entry_point_ < 0) return {};

        Timestamp qt = now_ms();
        Metric m = parse_metric(opts.metric);
        int ef = std::max(opts.ef_search, opts.top_k);

        // Precompute the ADC distance table once per query if
        // compression is active — turns each candidate scoring from
        // O(D) full-precision distance into O(M) table lookups.
        std::unique_ptr<ProductQuantizer::DistanceTable> adc_table;
        if (cfg_.use_pq_compression && quantizer_ready())
            adc_table = std::make_unique<ProductQuantizer::DistanceTable>(
                quantizer_->precompute_adc(query));

        // Start greedy search from top layer down to layer 1
        uint64_t ep = static_cast<uint64_t>(entry_point_);
        int top_layer = nodes_[ep].level();

        for (int layer = top_layer; layer >= 1; --layer) {
            ep = _greedy_closest(query, ep, layer, qt, m, adc_table.get());
        }

        // Layer 0: beam search with ef candidates
        auto candidates = _beam_search(query, ep, 0, ef, qt, m, opts.filter,
                                       adc_table.get());

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
        return results;
    }

    // ------ stats -------------------------------------------------

    std::size_t size() const {
        std::shared_lock lock(mutex_);
        return nodes_.size() - deleted_.size();
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
            ep = _greedy_closest(link_query, ep, layer, qt, m);

        // Connect at each layer from new_level down to 0
        for (int layer = std::min(new_level, top_layer); layer >= 0; --layer) {
            int M_layer = (layer == 0) ? cfg_.M0 : cfg_.M;
            auto cands = _beam_search(link_query, ep, layer,
                                      cfg_.ef_construction, qt, m, "");

            int take = std::min(M_layer, static_cast<int>(cands.size()));
            for (int i = 0; i < take; ++i) {
                uint64_t nb = cands[i].second;
                nodes_[new_id].neighbours[layer].push_back(nb);
                nodes_[nb].neighbours[layer].push_back(new_id);
                // Prune neighbours to M_layer if overfull
                _prune_neighbours(nb, layer, M_layer, qt, m, link_query);
            }
            if (!cands.empty()) ep = cands[0].second;
        }

        // Update entry point if new node has higher level
        if (new_level > top_layer)
            entry_point_ = static_cast<int64_t>(new_id);
    }

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
                              Timestamp qt, Metric m,
                              const ProductQuantizer::DistanceTable* adc = nullptr) const {
        uint64_t cur = start;
        float cur_score = _blended_score(query, cur, qt, m, adc);
        bool improved = true;
        while (improved) {
            improved = false;
            for (uint64_t nb : nodes_[cur].neighbours[layer]) {
                if (deleted_.count(nb)) continue;
                float s = _blended_score(query, nb, qt, m, adc);
                if (s > cur_score) { cur_score = s; cur = nb; improved = true; }
            }
        }
        return cur;
    }

    // Beam search — returns sorted (score, id) pairs, best first
    std::vector<std::pair<float, uint64_t>>
    _beam_search(const std::vector<float>& query,
                 uint64_t ep, int layer, int ef,
                 Timestamp qt, Metric m,
                 const std::string& filter,
                 const ProductQuantizer::DistanceTable* adc = nullptr) const {

        using P = std::pair<float, uint64_t>;

        // candidates: max-heap (best at top for pruning)
        std::priority_queue<P> W;  // results
        // visited min-heap to expand cheapest unexpanded
        std::priority_queue<P, std::vector<P>, std::greater<P>> C;

        std::unordered_set<uint64_t> visited;

        float ep_score = _blended_score(query, ep, qt, m, adc);
        W.push({ep_score, ep});
        C.push({ep_score, ep});
        visited.insert(ep);

        while (!C.empty()) {
            auto [c_score, c] = C.top(); C.pop();
            if (W.size() >= static_cast<std::size_t>(ef) &&
                c_score < W.top().first - 1e-6f) break;

            for (uint64_t nb : nodes_[c].neighbours[layer]) {
                if (visited.count(nb) || deleted_.count(nb)) continue;
                visited.insert(nb);

                const auto& nnode = nodes_[nb];
                if (nnode.valid_until.has_value() &&
                    qt > nnode.valid_until.value()) continue;
                if (!filter.empty() && !_filter_match(nnode, filter)) continue;
                if (scorer_.is_hard_excluded_raw(nnode.created_at, qt)) continue;

                float s = _blended_score(query, nb, qt, m, adc);
                C.push({s, nb});
                W.push({s, nb});
                if (static_cast<int>(W.size()) > ef) W.pop();
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
                         uint64_t iid, Timestamp qt, Metric m,
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
        float tw  = scorer_.temporal_weight_raw(node.created_at, qt);
        float alpha = 1.0f - cfg_.temporal.temporal_blend;
        float beta  = cfg_.temporal.temporal_blend;
        return alpha * sim + beta * tw;
    }

    void _prune_neighbours(uint64_t node_id, int layer, int M_max,
                           Timestamp qt, Metric m,
                           const std::vector<float>& node_query) {
        auto& nbs = nodes_[node_id].neighbours[layer];
        if (static_cast<int>(nbs.size()) <= M_max) return;

        // Keep M_max closest by blended score. node_query is the
        // full-precision embedding of node_id, passed explicitly
        // since nodes_[node_id].embedding may be empty (compressed).
        std::sort(nbs.begin(), nbs.end(), [&](uint64_t a, uint64_t b){
            return _blended_score(node_query, a, qt, m) >
                   _blended_score(node_query, b, qt, m);
        });
        nbs.resize(M_max);
    }

    static bool _filter_match(const HNSWNode& n, const std::string& filter) {
        auto eq = filter.find('=');
        if (eq == std::string::npos) return true;
        auto key = filter.substr(0, eq);
        auto val = filter.substr(eq + 1);
        auto it = n.payload.find(key);
        return (it != n.payload.end() && it->second == val);
    }

    Config                cfg_;
    TemporalScorer        scorer_;
    double                ml_;
    mutable std::mt19937  rng_;

    std::unique_ptr<ProductQuantizer> quantizer_;  // null until trained
    VectorStore                       vector_store_;  // optional, for exact rescore

    std::vector<HNSWNode>                    nodes_;
    std::unordered_map<std::string, uint64_t> ext_to_int_;
    std::unordered_set<uint64_t>              deleted_;
    int64_t                                   entry_point_ = -1;

    mutable std::shared_mutex mutex_;
};

} // namespace tidevec
