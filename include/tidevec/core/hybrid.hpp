#pragma once
// ================================================================
// Hybrid search fusion — combine a dense (vector) ranking with a
// sparse (BM25) ranking.
//
//   RRF       score = alpha / (k + rank_dense) + (1 - alpha) / (k + rank_sparse)
//             (ranks are 1-based; a list the hit is missing from adds 0)
//   WEIGHTED  score = alpha * norm(dense) + (1 - alpha) * norm(bm25)
//             with each list min-max normalised to [0, 1]
//
// The fused hit keeps the dense result's fields when it has one (vector
// and temporal scores, supersession annotations) and carries the BM25
// score in text_score.
// ================================================================

#include <tidevec/core/cortex_vector.hpp>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

namespace tidevec {

inline int hybrid_prefetch(const QueryOptions& opts) {
    if (opts.hybrid.prefetch > 0) return std::max(opts.hybrid.prefetch, opts.top_k);
    return std::max(4 * opts.top_k, 50);
}

inline std::vector<SearchResult> fuse_results(const std::vector<SearchResult>& dense,
                                              const std::vector<SearchResult>& sparse,
                                              const HybridOptions& h, int top_k) {
    const float alpha = std::clamp(h.alpha, 0.0f, 1.0f);
    struct Acc {
        SearchResult r;
        float fused = 0.0f;
    };
    std::unordered_map<std::string, Acc> acc;
    std::vector<std::string> order;

    auto norm = [](const std::vector<SearchResult>& xs, bool text) {
        float lo = 0, hi = 0;
        bool first = true;
        for (const auto& x : xs) {
            float s = text ? x.text_score : x.score;
            if (first) { lo = hi = s; first = false; }
            lo = std::min(lo, s);
            hi = std::max(hi, s);
        }
        return std::pair<float, float>{lo, hi};
    };
    auto scaled = [](float s, std::pair<float, float> r) {
        return r.second > r.first ? (s - r.first) / (r.second - r.first) : 1.0f;
    };

    const auto dense_range  = norm(dense, false);
    const auto sparse_range = norm(sparse, true);
    const float k = static_cast<float>(std::max(1, h.rrf_k));

    for (std::size_t i = 0; i < dense.size(); ++i) {
        const auto& d = dense[i];
        auto [it, fresh] = acc.try_emplace(d.id);
        if (fresh) order.push_back(d.id);
        it->second.r = d;
        it->second.fused += h.fusion == FusionMethod::RRF
            ? alpha / (k + static_cast<float>(i + 1))
            : alpha * scaled(d.score, dense_range);
    }
    for (std::size_t i = 0; i < sparse.size(); ++i) {
        const auto& s = sparse[i];
        auto [it, fresh] = acc.try_emplace(s.id);
        if (fresh) {
            order.push_back(s.id);
            it->second.r = s;
        }
        it->second.r.text_score = s.text_score;
        it->second.fused += h.fusion == FusionMethod::RRF
            ? (1.0f - alpha) / (k + static_cast<float>(i + 1))
            : (1.0f - alpha) * scaled(s.text_score, sparse_range);
    }

    std::vector<SearchResult> out;
    out.reserve(order.size());
    for (const auto& id : order) {
        auto& a = acc[id];
        a.r.score = a.fused;
        out.push_back(std::move(a.r));
    }
    std::stable_sort(out.begin(), out.end(),
        [](const SearchResult& a, const SearchResult& b) { return a.score > b.score; });
    if (out.size() > static_cast<std::size_t>(std::max(0, top_k)))
        out.resize(static_cast<std::size_t>(std::max(0, top_k)));
    return out;
}

} // namespace tidevec
