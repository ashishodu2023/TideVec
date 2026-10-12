#pragma once
// ================================================================
// PayloadIndex — secondary indexes over vector payloads
//
//   keyword  value -> ids          (equality / IN on string values)
//   numeric  sorted (value, id)    (ranges and numeric equality; values
//                                   that parse as numbers or ISO dates)
//   created_at sorted (ts, id)     (always on; as_of and date ranges)
//
// Every payload key is indexed automatically (keyword values longer
// than kMaxKeywordBytes are not, which marks that key's keyword index
// incomplete so equality on it falls back to scanning).
//
// The index is append-only between rebuilds: a payload change adds
// the new entries and leaves the old ones behind, so lookups return a
// superset of the matching ids. Callers always re-check the filter on
// each candidate; the owning index rebuilds this one on compaction.
// ================================================================

#include <tidevec/core/filter.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tidevec {

class NumericIndex {
public:
    using Entry = std::pair<double, uint64_t>;

    void add(double v, uint64_t id) {
        tail_.emplace_back(v, id);
        if (tail_.size() > std::max<std::size_t>(kMinTail, sorted_.size() / 16)) _merge();
    }

    void clear() { sorted_.clear(); tail_.clear(); }
    std::size_t size() const { return sorted_.size() + tail_.size(); }

    std::size_t count(double lo, bool lo_incl, double hi, bool hi_incl) const {
        auto [b, e] = _bounds(lo, lo_incl, hi, hi_incl);
        std::size_t n = static_cast<std::size_t>(e - b);
        for (const auto& t : tail_) if (_in(t.first, lo, lo_incl, hi, hi_incl)) ++n;
        return n;
    }

    void collect(double lo, bool lo_incl, double hi, bool hi_incl,
                 std::vector<uint64_t>& out) const {
        auto [b, e] = _bounds(lo, lo_incl, hi, hi_incl);
        for (auto it = b; it != e; ++it) out.push_back(it->second);
        for (const auto& t : tail_)
            if (_in(t.first, lo, lo_incl, hi, hi_incl)) out.push_back(t.second);
    }

private:
    static constexpr std::size_t kMinTail = 1024;

    static bool _in(double v, double lo, bool lo_incl, double hi, bool hi_incl) {
        if (lo_incl ? v < lo : v <= lo) return false;
        if (hi_incl ? v > hi : v >= hi) return false;
        return true;
    }

    std::pair<std::vector<Entry>::const_iterator, std::vector<Entry>::const_iterator>
    _bounds(double lo, bool lo_incl, double hi, bool hi_incl) const {
        auto b = lo_incl
            ? std::lower_bound(sorted_.begin(), sorted_.end(), lo,
                  [](const Entry& e, double v) { return e.first < v; })
            : std::upper_bound(sorted_.begin(), sorted_.end(), lo,
                  [](double v, const Entry& e) { return v < e.first; });
        auto e = hi_incl
            ? std::upper_bound(b, sorted_.end(), hi,
                  [](double v, const Entry& x) { return v < x.first; })
            : std::lower_bound(b, sorted_.end(), hi,
                  [](const Entry& x, double v) { return x.first < v; });
        if (e < b) e = b;
        return {b, e};
    }

    void _merge() {
        std::sort(tail_.begin(), tail_.end());
        std::size_t mid = sorted_.size();
        sorted_.insert(sorted_.end(), tail_.begin(), tail_.end());
        std::inplace_merge(sorted_.begin(), sorted_.begin() + mid, sorted_.end());
        tail_.clear();
    }

    std::vector<Entry> sorted_;
    std::vector<Entry> tail_;
};

class PayloadIndex {
public:
    // Resolves a vector id to an internal id (for "_id" filters).
    using IdLookup = std::function<std::optional<uint64_t>(const std::string&)>;

    static constexpr std::size_t kMaxKeywordBytes = 128;
    static constexpr std::size_t kMaxIntersectRatio = 64;

    PayloadIndex() = default;
    PayloadIndex(bool payload_keys, std::string text_field)
        : enabled_(payload_keys), text_field_(std::move(text_field)) {}

    void add(uint64_t id, const std::unordered_map<std::string, std::string>& payload,
             Timestamp created_at) {
        created_at_.add(static_cast<double>(created_at), id);
        if (!enabled_) return;
        for (const auto& [k, v] : payload) {
            auto& kw = keyword_[k];
            if (k == text_field_ || v.size() > kMaxKeywordBytes) kw.complete = false;
            else kw.postings[v].push_back(id);
            if (auto n = parse_numeric_value(v)) numeric_[k].add(*n, id);
        }
    }

    void clear() {
        keyword_.clear();
        numeric_.clear();
        created_at_.clear();
    }

    // Upper bound on the number of matching ids, or nullopt when the
    // filter cannot be answered from the indexes.
    std::optional<std::size_t> estimate(const FilterExpr& f, const IdLookup& ids) const {
        using Op = FilterExpr::Op;
        switch (f.op) {
        case Op::AND: {
            std::optional<std::size_t> best;
            for (const auto& c : f.children)
                if (auto e = estimate(c, ids); e && (!best || *e < *best)) best = e;
            return best;
        }
        case Op::OR: {
            std::size_t sum = 0;
            for (const auto& c : f.children) {
                auto e = estimate(c, ids);
                if (!e) return std::nullopt;
                sum += *e;
            }
            return sum;
        }
        case Op::EQ:
        case Op::IN: {
            std::size_t sum = 0;
            for (const auto& v : f.values) {
                auto e = _estimate_value(f.key, v, ids);
                if (!e) return std::nullopt;
                sum += *e;
            }
            return sum;
        }
        case Op::RANGE: {
            if (f.key == kFilterCreatedAt)
                return created_at_.count(f.lo, f.lo_incl, f.hi, f.hi_incl);
            if (f.key == kFilterId || !enabled_) return std::nullopt;
            auto it = numeric_.find(f.key);
            return it == numeric_.end() ? 0 : it->second.count(f.lo, f.lo_incl, f.hi, f.hi_incl);
        }
        default:
            return std::nullopt;
        }
    }

    // Expected number of matches out of `total`, treating the children of
    // an AND as independent. Unlike estimate() this is not an upper bound;
    // it is only used to pick a plan. nullopt exactly when estimate() is.
    std::optional<double> expected(const FilterExpr& f, const IdLookup& ids, double total) const {
        using Op = FilterExpr::Op;
        if (f.op == Op::AND) {
            std::optional<double> frac;
            for (const auto& c : f.children)
                if (auto e = expected(c, ids, total))
                    frac = frac.value_or(1.0) * std::min(1.0, *e / std::max(total, 1.0));
            if (!frac) return std::nullopt;
            return *frac * total;
        }
        if (f.op == Op::OR) {
            double sum = 0;
            for (const auto& c : f.children) {
                auto e = expected(c, ids, total);
                if (!e) return std::nullopt;
                sum += *e;
            }
            return std::min(sum, total);
        }
        if (auto e = estimate(f, ids)) return static_cast<double>(*e);
        return std::nullopt;
    }

    // Candidate ids for a filter estimate() could answer. May contain
    // duplicates and stale ids.
    void collect(const FilterExpr& f, const IdLookup& ids, std::vector<uint64_t>& out) const {
        using Op = FilterExpr::Op;
        switch (f.op) {
        case Op::AND: {
            // Intersect the indexable children, most selective first. Each
            // child's candidates are a superset of its matches, so the
            // intersection is a superset of the AND's matches.
            std::vector<std::pair<std::size_t, const FilterExpr*>> parts;
            for (const auto& c : f.children)
                if (auto e = estimate(c, ids)) parts.emplace_back(*e, &c);
            if (parts.empty()) return;
            std::sort(parts.begin(), parts.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });
            std::vector<uint64_t> acc, next, both;
            collect(*parts[0].second, ids, acc);
            std::sort(acc.begin(), acc.end());
            acc.erase(std::unique(acc.begin(), acc.end()), acc.end());
            for (std::size_t i = 1; i < parts.size() && !acc.empty(); ++i) {
                // Not worth walking a much larger posting list.
                if (parts[i].first > acc.size() * kMaxIntersectRatio) break;
                next.clear();
                collect(*parts[i].second, ids, next);
                std::sort(next.begin(), next.end());
                both.clear();
                std::set_intersection(acc.begin(), acc.end(), next.begin(), next.end(),
                                      std::back_inserter(both));
                acc.swap(both);
            }
            out.insert(out.end(), acc.begin(), acc.end());
            return;
        }
        case Op::OR:
            for (const auto& c : f.children) collect(c, ids, out);
            return;
        case Op::EQ:
        case Op::IN:
            for (const auto& v : f.values) _collect_value(f.key, v, ids, out);
            return;
        case Op::RANGE:
            if (f.key == kFilterCreatedAt) {
                created_at_.collect(f.lo, f.lo_incl, f.hi, f.hi_incl, out);
            } else if (auto it = numeric_.find(f.key); it != numeric_.end()) {
                it->second.collect(f.lo, f.lo_incl, f.hi, f.hi_incl, out);
            }
            return;
        default:
            return;
        }
    }

private:
    struct KeywordIndex {
        std::unordered_map<std::string, std::vector<uint64_t>> postings;
        bool complete = true;
    };

    std::optional<std::size_t> _estimate_value(const std::string& key, const FilterValue& v,
                                               const IdLookup& ids) const {
        if (key == kFilterId) return ids && ids(v.str) ? 1 : 0;
        if (key == kFilterCreatedAt) {
            auto n = v.num ? v.num : parse_numeric_value(v.str);
            return n ? created_at_.count(*n, true, *n, true) : 0;
        }
        if (!enabled_) return std::nullopt;
        if (v.num) {
            auto it = numeric_.find(key);
            return it == numeric_.end() ? 0 : it->second.count(*v.num, true, *v.num, true);
        }
        auto it = keyword_.find(key);
        if (it == keyword_.end()) return 0;
        if (!it->second.complete) return std::nullopt;
        auto p = it->second.postings.find(v.str);
        return p == it->second.postings.end() ? 0 : p->second.size();
    }

    void _collect_value(const std::string& key, const FilterValue& v, const IdLookup& ids,
                        std::vector<uint64_t>& out) const {
        if (key == kFilterId) {
            if (ids) if (auto iid = ids(v.str)) out.push_back(*iid);
            return;
        }
        if (key == kFilterCreatedAt) {
            if (auto n = v.num ? v.num : parse_numeric_value(v.str))
                created_at_.collect(*n, true, *n, true, out);
            return;
        }
        if (v.num) {
            if (auto it = numeric_.find(key); it != numeric_.end())
                it->second.collect(*v.num, true, *v.num, true, out);
            return;
        }
        auto it = keyword_.find(key);
        if (it == keyword_.end()) return;
        auto p = it->second.postings.find(v.str);
        if (p != it->second.postings.end())
            out.insert(out.end(), p->second.begin(), p->second.end());
    }

    bool enabled_ = true;
    std::string text_field_;
    std::unordered_map<std::string, KeywordIndex> keyword_;
    std::unordered_map<std::string, NumericIndex> numeric_;
    NumericIndex created_at_;
};

} // namespace tidevec
