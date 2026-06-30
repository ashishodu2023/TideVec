// ================================================================
// half_life.hpp — Temporal decay half-life presets
// ================================================================

#pragma once

#include <cstdint>

namespace tidevec {

struct HalfLife {
    static constexpr int64_t ONE_HOUR   =     3'600'000LL; // agent sessions
    static constexpr int64_t ONE_DAY    =    86'400'000LL; // news / RSS feeds
    static constexpr int64_t ONE_WEEK   =   604'800'000LL; // support tickets
    static constexpr int64_t ONE_MONTH  = 2'592'000'000LL; // documents (default)
    static constexpr int64_t ONE_YEAR   = 31'536'000'000LL;// long-term knowledge
};

} // namespace tidevec
