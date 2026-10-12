#pragma once
// ================================================================
// Filter — payload / metadata predicates for search, scroll, delete
//
// A FilterExpr is a small boolean tree:
//   AND / OR / NOT over leaves
//   EQ, NE        key == value (string, or numeric when the value is a number)
//   IN, NIN       key in {values}
//   RANGE         lo <(=) key <(=) hi, numbers or ISO-8601 dates
//   EXISTS        key present (or absent)
//
// Payload values are strings. Numeric comparisons parse the payload
// value as a number, or as an ISO-8601 date (epoch milliseconds).
// Two keys address vector fields instead of the payload:
//   "created_at"  the vector timestamp (ms since epoch)
//   "_id"         the vector id
//
// The JSON form ({"genre": {"$in": [...]}, "$or": [...]}) is parsed in
// api/json_serializers.hpp; the legacy "key=value" string is parsed by
// parse_legacy_filter() below.
// ================================================================

#include <tidevec/core/cortex_vector.hpp>

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace tidevec {

inline constexpr const char* kFilterCreatedAt = "created_at";
inline constexpr const char* kFilterId        = "_id";

// Days since 1970-01-01 for a proleptic Gregorian date (H. Hinnant).
inline int64_t days_from_civil(int64_t y, unsigned m, unsigned d) {
    y -= m <= 2;
    const int64_t era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(y - era * 400);
    const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + static_cast<int64_t>(doe) - 719468;
}

// Parses "YYYY-MM-DD", "YYYY-MM-DDTHH:MM[:SS[.fff]]" with an optional
// "Z" or "+HH:MM"/"-HH:MM" suffix (a space may replace the "T").
// Returns epoch milliseconds (UTC when no offset is given).
inline std::optional<int64_t> parse_iso8601_ms(const std::string& s) {
    const std::size_t n = s.size();
    if (n < 10) return std::nullopt;
    auto digits = [&](std::size_t pos, std::size_t len, int& out) {
        if (pos + len > n) return false;
        int v = 0;
        for (std::size_t i = pos; i < pos + len; ++i) {
            if (!std::isdigit(static_cast<unsigned char>(s[i]))) return false;
            v = v * 10 + (s[i] - '0');
        }
        out = v;
        return true;
    };
    int Y, M, D;
    if (!digits(0, 4, Y) || s[4] != '-' || !digits(5, 2, M) || s[7] != '-' ||
        !digits(8, 2, D))
        return std::nullopt;
    if (M < 1 || M > 12 || D < 1 || D > 31) return std::nullopt;
    int64_t ms = days_from_civil(Y, static_cast<unsigned>(M), static_cast<unsigned>(D))
                 * 86'400'000LL;
    std::size_t p = 10;
    if (p == n) return ms;
    if (s[p] != 'T' && s[p] != 't' && s[p] != ' ') return std::nullopt;
    ++p;
    int h, mi, sec = 0;
    if (!digits(p, 2, h) || p + 2 >= n || s[p + 2] != ':' || !digits(p + 3, 2, mi))
        return std::nullopt;
    p += 5;
    if (p < n && s[p] == ':') {
        if (!digits(p + 1, 2, sec)) return std::nullopt;
        p += 3;
    }
    int frac_ms = 0;
    if (p < n && (s[p] == '.' || s[p] == ',')) {
        ++p;
        int scale = 100;
        while (p < n && std::isdigit(static_cast<unsigned char>(s[p]))) {
            frac_ms += (s[p] - '0') * scale;
            scale /= 10;
            ++p;
        }
    }
    if (h > 23 || mi > 59 || sec > 60) return std::nullopt;
    ms += (h * 3600LL + mi * 60LL + sec) * 1000LL + frac_ms;
    if (p == n) return ms;
    if (s[p] == 'Z' || s[p] == 'z') return p + 1 == n ? std::optional<int64_t>(ms)
                                                      : std::nullopt;
    if (s[p] == '+' || s[p] == '-') {
        int oh, om = 0;
        if (!digits(p + 1, 2, oh)) return std::nullopt;
        std::size_t q = p + 3;
        if (q < n && s[q] == ':') ++q;
        if (q < n && !digits(q, 2, om)) return std::nullopt;
        if (q < n) q += 2;
        if (q != n) return std::nullopt;
        int64_t off = (oh * 60LL + om) * 60'000LL;
        return s[p] == '+' ? ms - off : ms + off;
    }
    return std::nullopt;
}

// A payload value as a number: decimal/scientific, else an ISO-8601 date.
inline std::optional<double> parse_numeric_value(const std::string& s) {
    if (s.empty()) return std::nullopt;
    const char* b = s.c_str();
    char* end = nullptr;
    double v = std::strtod(b, &end);
    if (end != b) {
        while (*end && std::isspace(static_cast<unsigned char>(*end))) ++end;
        if (*end == '\0' && std::isfinite(v)) return v;
    }
    if (auto ms = parse_iso8601_ms(s)) return static_cast<double>(*ms);
    return std::nullopt;
}

struct FilterValue {
    std::string str;                 // exact string form
    std::optional<double> num;       // set when the value was a JSON number
};

struct FilterExpr {
    enum class Op { AND, OR, NOT, EQ, NE, IN, NIN, RANGE, EXISTS };

    Op op = Op::AND;
    std::string key;
    std::vector<FilterValue> values;   // EQ/NE (one), IN/NIN (many)
    double lo = -std::numeric_limits<double>::infinity();
    double hi =  std::numeric_limits<double>::infinity();
    bool lo_incl = true, hi_incl = true;
    bool exists = true;                // EXISTS: true = present, false = absent
    std::vector<FilterExpr> children;  // AND / OR / NOT (one child)

    // ---- builders -------------------------------------------------
    static FilterExpr eq(std::string k, std::string v) {
        FilterExpr f; f.op = Op::EQ; f.key = std::move(k);
        f.values.push_back({std::move(v), std::nullopt});
        return f;
    }
    static FilterExpr eq_num(std::string k, double v) {
        FilterExpr f; f.op = Op::EQ; f.key = std::move(k);
        f.values.push_back({_num_str(v), v});
        return f;
    }
    static FilterExpr in(std::string k, std::vector<std::string> vs) {
        FilterExpr f; f.op = Op::IN; f.key = std::move(k);
        for (auto& v : vs) f.values.push_back({std::move(v), std::nullopt});
        return f;
    }
    static FilterExpr range(std::string k,
                            std::optional<double> gte, std::optional<double> lte,
                            std::optional<double> gt = std::nullopt,
                            std::optional<double> lt = std::nullopt) {
        FilterExpr f; f.op = Op::RANGE; f.key = std::move(k);
        if (gte) { f.lo = *gte; f.lo_incl = true; }
        if (gt && (!gte || *gt >= *gte)) { f.lo = *gt; f.lo_incl = false; }
        if (lte) { f.hi = *lte; f.hi_incl = true; }
        if (lt && (!lte || *lt <= *lte)) { f.hi = *lt; f.hi_incl = false; }
        return f;
    }
    static FilterExpr all_of(std::vector<FilterExpr> cs) {
        FilterExpr f; f.op = Op::AND; f.children = std::move(cs); return f;
    }
    static FilterExpr any_of(std::vector<FilterExpr> cs) {
        FilterExpr f; f.op = Op::OR; f.children = std::move(cs); return f;
    }
    static FilterExpr negate(FilterExpr c) {
        FilterExpr f; f.op = Op::NOT; f.children.push_back(std::move(c)); return f;
    }
    static FilterExpr has(std::string k, bool present = true) {
        FilterExpr f; f.op = Op::EXISTS; f.key = std::move(k); f.exists = present;
        return f;
    }

    bool in_range(double v) const {
        if (lo_incl ? v < lo : v <= lo) return false;
        if (hi_incl ? v > hi : v >= hi) return false;
        return true;
    }

    // ---- evaluation ----------------------------------------------
    bool matches(const std::string& id,
                 const std::unordered_map<std::string, std::string>& payload,
                 Timestamp created_at) const {
        switch (op) {
        case Op::AND:
            for (const auto& c : children)
                if (!c.matches(id, payload, created_at)) return false;
            return true;
        case Op::OR:
            for (const auto& c : children)
                if (c.matches(id, payload, created_at)) return true;
            return children.empty();
        case Op::NOT:
            return children.empty() || !children[0].matches(id, payload, created_at);
        default: break;
        }

        const std::string* sval = nullptr;
        std::string tmp;
        std::optional<double> nval;
        bool have_num = false;
        if (key == kFilterCreatedAt) {
            nval = static_cast<double>(created_at);
            have_num = true;
            tmp = std::to_string(created_at);
            sval = &tmp;
        } else if (key == kFilterId) {
            sval = &id;
        } else {
            auto it = payload.find(key);
            if (it != payload.end()) sval = &it->second;
        }
        auto num = [&]() -> std::optional<double> {
            if (!have_num) { nval = sval ? parse_numeric_value(*sval) : std::nullopt;
                             have_num = true; }
            return nval;
        };
        auto value_eq = [&](const FilterValue& fv) {
            if (!sval) return false;
            if (fv.num) {
                auto n = num();
                return n && *n == *fv.num;
            }
            if (key == kFilterCreatedAt) {
                auto n = parse_numeric_value(fv.str);
                return n && *n == static_cast<double>(created_at);
            }
            return *sval == fv.str;
        };

        switch (op) {
        case Op::EXISTS: return (sval != nullptr) == exists;
        case Op::EQ:
        case Op::IN:
            for (const auto& v : values) if (value_eq(v)) return true;
            return false;
        case Op::NE:
        case Op::NIN:
            for (const auto& v : values) if (value_eq(v)) return false;
            return true;
        case Op::RANGE: {
            if (!sval) return false;
            auto n = num();
            return n && in_range(*n);
        }
        default: return true;
        }
    }

    bool matches(const CortexVector& v) const {
        return matches(v.id, v.payload, v.created_at);
    }

    static std::string _num_str(double v) {
        if (std::floor(v) == v && std::fabs(v) < 9.0e15)
            return std::to_string(static_cast<long long>(v));
        return std::to_string(v);
    }
};

using FilterPtr = std::shared_ptr<const FilterExpr>;

// Legacy "key=value" filter; anything without '=' matches everything.
inline FilterPtr parse_legacy_filter(const std::string& s) {
    auto eq = s.find('=');
    if (eq == std::string::npos) return nullptr;
    return std::make_shared<FilterExpr>(FilterExpr::eq(s.substr(0, eq), s.substr(eq + 1)));
}

// The filter a query actually applies: the structured filter, the legacy
// string filter and the as_of cut-off (created_at <= as_of), ANDed.
inline FilterPtr effective_filter(const QueryOptions& opts) {
    std::vector<FilterExpr> parts;
    if (opts.filter_expr) parts.push_back(*opts.filter_expr);
    if (!opts.filter.empty())
        if (auto legacy = parse_legacy_filter(opts.filter)) parts.push_back(*legacy);
    if (opts.as_of)
        parts.push_back(FilterExpr::range(kFilterCreatedAt, std::nullopt,
                                          static_cast<double>(*opts.as_of)));
    if (parts.empty()) return nullptr;
    if (parts.size() == 1) return std::make_shared<FilterExpr>(std::move(parts[0]));
    return std::make_shared<FilterExpr>(FilterExpr::all_of(std::move(parts)));
}

} // namespace tidevec
