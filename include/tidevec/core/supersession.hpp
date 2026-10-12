#pragma once
// ================================================================
// Supersession — automatic "newer version of the same document"
// detection.
//
// On upsert, the incoming vector is compared with its nearest
// neighbours (pure similarity, no temporal blend). A neighbour is
// treated as another version of the same document when:
//   · vector_score >= similarity_threshold (for the cosine metric
//     vector_score is (1 + cos) / 2, so 0.92 means cos >= 0.84), and
//   · if match_field is set, payload[match_field] is non-empty and
//     equal on both vectors (e.g. "doc_id", "url", "policy_key").
// The newer vector (by created_at) is linked to the older one with
//   newer —UPDATES→ older   and   older —SUPERSEDED_BY→ newer.
//
// At query time, results with a newer version get `superseded_by`
// populated, a staleness warning, and an optional score penalty.
// ================================================================

#include <tidevec/core/cortex_vector.hpp>

#include <algorithm>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace tidevec {

struct SupersessionConfig {
    bool        enabled              = false;
    float       similarity_threshold = 0.92f;
    std::string match_field;            // empty = similarity only
    int         candidates           = 5;
};

struct SupersessionLink {
    std::string newer;
    std::string older;
};

inline QueryOptions supersession_query_opts(const SupersessionConfig& cfg) {
    QueryOptions q;
    q.top_k = std::max(1, cfg.candidates);
    q.temporal_blend = 0.0f;
    q.include_staleness_warnings = false;
    return q;
}

inline std::vector<SupersessionLink> plan_supersession(
        const CortexVector& incoming,
        const std::vector<SearchResult>& neighbours,
        const SupersessionConfig& cfg) {
    std::vector<SupersessionLink> links;
    if (!cfg.enabled) return links;

    std::string key;
    if (!cfg.match_field.empty()) {
        auto it = incoming.payload.find(cfg.match_field);
        if (it == incoming.payload.end() || it->second.empty()) return links;
        key = it->second;
    }

    for (const auto& n : neighbours) {
        if (n.id == incoming.id) continue;
        if (n.vector_score < cfg.similarity_threshold) continue;
        if (!key.empty()) {
            auto it = n.payload.find(cfg.match_field);
            if (it == n.payload.end() || it->second != key) continue;
        }
        if (n.created_at <= incoming.created_at)
            links.push_back({incoming.id, n.id});
        else
            links.push_back({n.id, incoming.id});
    }
    return links;
}

// Annotate results with newer versions, add a staleness warning, apply
// the score penalty and re-sort. `lookup(id)` returns the superseding ids.
inline void apply_supersession(
        std::vector<SearchResult>& results,
        const QueryOptions& opts,
        const std::function<std::vector<std::string>(const std::string&)>& lookup) {
    bool penalised = false;
    for (auto& r : results) {
        r.superseded_by = lookup(r.id);
        if (r.superseded_by.empty()) continue;

        if (opts.include_staleness_warnings) {
            r.staleness_warning = true;
            std::string ids;
            for (const auto& s : r.superseded_by)
                ids += (ids.empty() ? "" : ", ") + s;
            r.staleness_reason = "Superseded by " + ids;
        }
        if (opts.superseded_penalty > 0.0f) {
            float p = std::clamp(opts.superseded_penalty, 0.0f, 1.0f);
            r.score *= (1.0f - p);
            penalised = true;
        }
    }
    if (penalised)
        std::stable_sort(results.begin(), results.end(),
            [](const SearchResult& a, const SearchResult& b) {
                return a.score > b.score;
            });
}

} // namespace tidevec
