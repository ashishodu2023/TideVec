#pragma once

#include <tidevec/core/cortex_vector.hpp>
#include <mutex>

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <shared_mutex>
#include <optional>
#include <functional>

namespace tidevec {

// ------------------------------------------------------------------
// CausalGraph — native directed typed-edge graph
//
// Stored in-memory (persisted to .cdb segment by StorageEngine).
// Supports:
//   · add_edge(src, tgt, type, weight)
//   · get_neighbours(id, type_filter, hops)
//   · find_contradictions(id)  — all CONTRADICTS edges from id
//   · resolve_entity(id)       — all ENTITY_OF neighbours (same entity)
//   · path_exists(src, dst)    — BFS reachability
// ------------------------------------------------------------------

struct GraphEdge {
    std::string src_id;
    std::string tgt_id;
    EdgeType    type;
    float       weight;
    Timestamp   created_at;
};

class CausalGraph {
public:
    CausalGraph() = default;

    // ------ write -------------------------------------------------

    void add_edge(const std::string& src,
                  const std::string& tgt,
                  EdgeType type,
                  float weight = 1.0f) {
        std::unique_lock lock(mutex_);
        GraphEdge e{src, tgt, type, weight, now_ms()};
        out_edges_[src].push_back(e);
        in_edges_[tgt].push_back(e);
        all_edges_.push_back(e);
    }

    bool remove_edges(const std::string& src, const std::string& tgt) {
        std::unique_lock lock(mutex_);
        bool removed = false;
        auto& out = out_edges_[src];
        std::size_t before = out.size();
        out.erase(std::remove_if(out.begin(), out.end(),
            [&](const GraphEdge& e){ return e.tgt_id == tgt; }), out.end());
        removed = out.size() < before;

        auto& in = in_edges_[tgt];
        in.erase(std::remove_if(in.begin(), in.end(),
            [&](const GraphEdge& e){ return e.src_id == src; }), in.end());
        return removed;
    }

    void remove_node(const std::string& id) {
        std::unique_lock lock(mutex_);
        out_edges_.erase(id);
        in_edges_.erase(id);
        for (auto& [k, v] : out_edges_)
            v.erase(std::remove_if(v.begin(), v.end(),
                [&](const GraphEdge& e){ return e.tgt_id == id; }), v.end());
        for (auto& [k, v] : in_edges_)
            v.erase(std::remove_if(v.begin(), v.end(),
                [&](const GraphEdge& e){ return e.src_id == id; }), v.end());
    }

    // ------ read --------------------------------------------------

    // Get all outgoing edges from a node (optional type filter)
    std::vector<GraphEdge> get_out_edges(
            const std::string& id,
            std::optional<EdgeType> type_filter = std::nullopt) const {
        std::shared_lock lock(mutex_);
        auto it = out_edges_.find(id);
        if (it == out_edges_.end()) return {};
        if (!type_filter) return it->second;
        std::vector<GraphEdge> result;
        for (const auto& e : it->second)
            if (e.type == *type_filter) result.push_back(e);
        return result;
    }

    // BFS up to `hops` depth — returns all reachable node IDs
    std::vector<std::string> get_neighbours(
            const std::string& start,
            int hops = 1,
            std::optional<EdgeType> type_filter = std::nullopt) const {
        std::shared_lock lock(mutex_);
        std::unordered_set<std::string> visited;
        std::vector<std::string> frontier = {start};
        visited.insert(start);

        for (int h = 0; h < hops; ++h) {
            std::vector<std::string> next;
            for (const auto& node : frontier) {
                auto it = out_edges_.find(node);
                if (it == out_edges_.end()) continue;
                for (const auto& e : it->second) {
                    if (type_filter && e.type != *type_filter) continue;
                    if (!visited.count(e.tgt_id)) {
                        visited.insert(e.tgt_id);
                        next.push_back(e.tgt_id);
                    }
                }
            }
            frontier = std::move(next);
        }

        visited.erase(start);
        return std::vector<std::string>(visited.begin(), visited.end());
    }

    // Find all vectors that CONTRADICT the given id
    std::vector<std::string> find_contradictions(const std::string& id) const {
        return get_neighbours(id, 1, EdgeType::CONTRADICTS);
    }

    // Find all vectors sharing the same entity (ENTITY_OF links)
    std::vector<std::string> resolve_entity(const std::string& id) const {
        return get_neighbours(id, 2, EdgeType::ENTITY_OF);
    }

    // BFS reachability check
    bool path_exists(const std::string& src, const std::string& dst,
                     int max_hops = 5) const {
        auto reachable = get_neighbours(src, max_hops, std::nullopt);
        for (const auto& n : reachable)
            if (n == dst) return true;
        return false;
    }

    // Causal expansion: start nodes + their causal neighbours up to hops
    std::vector<std::string> causal_expand(
            const std::vector<std::string>& seed_ids,
            int hops = 1) const {
        std::unordered_set<std::string> result(seed_ids.begin(), seed_ids.end());
        for (const auto& id : seed_ids) {
            auto nbs = get_neighbours(id, hops, std::nullopt);
            for (const auto& nb : nbs) result.insert(nb);
        }
        return std::vector<std::string>(result.begin(), result.end());
    }

    std::size_t edge_count() const {
        std::shared_lock lock(mutex_);
        return all_edges_.size();
    }

    std::size_t node_count() const {
        std::shared_lock lock(mutex_);
        // count unique nodes (both src and tgt side)
        std::unordered_set<std::string> nodes;
        for (const auto& e : all_edges_) {
            nodes.insert(e.src_id);
            nodes.insert(e.tgt_id);
        }
        return nodes.size();
    }

private:
    std::unordered_map<std::string, std::vector<GraphEdge>> out_edges_;
    std::unordered_map<std::string, std::vector<GraphEdge>> in_edges_;
    std::vector<GraphEdge> all_edges_;
    mutable std::shared_mutex mutex_;
};

} // namespace tidevec
