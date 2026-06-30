#pragma once

#include <tidevec/core/cortex_vector.hpp>
#include <cmath>
#include <stdexcept>

namespace tidevec {

// ------------------------------------------------------------------
// TemporalConfig — per-collection decay settings
// ------------------------------------------------------------------
struct TemporalConfig {
    // Ebbinghaus exponential decay half-life in milliseconds
    // Presets:
    //   News / feeds      → 1 day   (86400000 ms)
    //   Agent session     → 1 hour  (3600000  ms)
    //   Support tickets   → 30 days (2592000000 ms)
    //   Documents / wiki  → 1 year  (31536000000 ms)
    int64_t half_life_ms = 30LL * 24 * 3600 * 1000;  // default: 30 days

    // Blend: final_score = (1-β)·vector_score + β·temporal_score
    float temporal_blend = 0.3f;

    // Staleness warning threshold: warn if temporal_score < this
    float staleness_threshold = 0.20f;

    // Hard exclude vectors whose temporal_score < cutoff (0 = never exclude)
    float temporal_cutoff = 0.0f;
};

// ------------------------------------------------------------------
// TemporalScorer — computes time-aware retrieval scores
// ------------------------------------------------------------------
class TemporalScorer {
public:
    explicit TemporalScorer(const TemporalConfig& cfg = {}) : cfg_(cfg) {}

    // Ebbinghaus decay: weight = 2^(−Δt / half_life)
    // Returns value in (0, 1]: 1.0 = just inserted, → 0 as age → ∞
    float temporal_weight(Timestamp created_at,
                          Timestamp query_time = 0) const {
        if (query_time == 0) query_time = now_ms();
        int64_t delta_ms = query_time - created_at;
        if (delta_ms <= 0) return 1.0f;  // future-dated or fresh
        double exponent = -static_cast<double>(delta_ms)
                        / static_cast<double>(cfg_.half_life_ms);
        return static_cast<float>(std::pow(2.0, exponent));
    }

    // Blended score: α·vector + β·temporal
    float blend(float vector_score, float t_weight) const {
        float alpha = 1.0f - cfg_.temporal_blend;
        float beta  = cfg_.temporal_blend;
        return alpha * vector_score + beta * t_weight;
    }

    // Full scoring for a retrieved candidate
    struct Score {
        float vector_score;
        float temporal_weight;
        float final_score;
        bool  staleness_warning;
        std::string staleness_reason;
    };

    Score score(const CortexVector& vec,
                float raw_similarity,
                Timestamp query_time = 0) const {
        if (query_time == 0) query_time = now_ms();

        float tw = temporal_weight(vec.created_at, query_time);
        float fs = blend(raw_similarity, tw);

        bool warn = (tw < cfg_.staleness_threshold);
        std::string reason;
        if (warn) {
            int64_t age_days = (query_time - vec.created_at)
                             / (1000LL * 3600 * 24);
            reason = "Vector is " + std::to_string(age_days)
                   + " days old (temporal_score=" +
                   std::to_string(tw).substr(0,5) + ")";
        }

        return Score{raw_similarity, tw, fs, warn, reason};
    }

    bool is_hard_excluded(const CortexVector& vec,
                          Timestamp query_time = 0) const {
        if (cfg_.temporal_cutoff <= 0.0f) return false;
        if (query_time == 0) query_time = now_ms();
        return temporal_weight(vec.created_at, query_time) < cfg_.temporal_cutoff;
    }

    // Raw versions (no CortexVector needed) — used by TVIndex internals
    float temporal_weight_raw(Timestamp created_at, Timestamp qt) const {
        return temporal_weight(created_at, qt);
    }

    bool is_hard_excluded_raw(Timestamp created_at, Timestamp qt) const {
        if (cfg_.temporal_cutoff <= 0.0f) return false;
        return temporal_weight(created_at, qt) < cfg_.temporal_cutoff;
    }

    Score score_raw(Timestamp created_at, float raw_sim, Timestamp qt) const {
        float tw = temporal_weight(created_at, qt);
        float fs = blend(raw_sim, tw);
        bool warn = (tw < cfg_.staleness_threshold);
        std::string reason;
        if (warn) {
            int64_t age_days = (qt - created_at) / (1000LL * 3600 * 24);
            reason = "Vector is " + std::to_string(age_days) + " days old";
        }
        return Score{raw_sim, tw, fs, warn, reason};
    }

    const TemporalConfig& config() const { return cfg_; }
    void set_config(const TemporalConfig& cfg) { cfg_ = cfg; }

private:
    TemporalConfig cfg_;
};

} // namespace tidevec
