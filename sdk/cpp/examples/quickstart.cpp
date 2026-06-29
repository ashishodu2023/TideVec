// ================================================================
// quickstart.cpp — TideVec C++ SDK quickstart
//
// Build:
//   g++ -std=c++20 -I../include quickstart.cpp -o quickstart
//   ./quickstart
//
// Or with CMake FetchContent — see README.md
// ================================================================

#include <tidevec/tidevec.hpp>
#include <iostream>
#include <random>

int main() {
    // ── Connect ───────────────────────────────────────────────────
    tidevec::TideVec db("localhost:6399", {
        .api_key = "",   // set if auth enabled
    });

    if (!db.ping()) {
        std::cerr << "Cannot connect to TideVec at localhost:6399\n";
        std::cerr << "Start with: docker run -p 6399:6399 averm004/tidevec:latest\n";
        return 1;
    }
    std::cout << "Connected to TideVec ✓\n";

    // ── Create collection ─────────────────────────────────────────
    db.create_collection({
        .name          = "docs",
        .dim           = 4,                            // use 768 in production
        .half_life_ms  = tidevec::HalfLife::ONE_WEEK,
        .temporal_blend = 0.3f,
    });
    std::cout << "Collection 'docs' created ✓\n";

    // ── Upsert vectors ────────────────────────────────────────────
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    auto rand_vec = [&](int dim) {
        tidevec::Embedding v(dim);
        for (auto& x : v) x = dist(rng);
        return v;
    };

    db.upsert("docs", {
        {
            .id        = "policy_v2",
            .embedding = rand_vec(4),
            .payload   = {{"source","confluence"},{"team","platform"}},
            .edges     = {{
                .target_id = "policy_v1",
                .type      = tidevec::EdgeType::UPDATES,
                .weight    = 0.95f,
            }},
        },
        {
            .id        = "policy_v1",
            .embedding = rand_vec(4),
            .payload   = {{"source","confluence"},{"team","platform"}},
        },
        {
            .id        = "compliance_doc",
            .embedding = rand_vec(4),
            .payload   = {{"source","gdrive"},{"team","legal"}},
        },
    });
    std::cout << "3 vectors upserted ✓\n";

    // ── Search ────────────────────────────────────────────────────
    auto query = rand_vec(4);
    auto resp = db.search("docs", query, {
        .top_k          = 5,
        .temporal_blend = 0.3f,
        .mode           = tidevec::QueryMode::CAUSAL_EXPAND,
        .include_trace  = true,
    });

    std::cout << "\nSearch results (" << resp.hits.size() << " hits):\n";
    for (const auto& hit : resp.hits) {
        std::cout << "  " << hit.id
                  << "  score=" << hit.score
                  << "  temporal=" << hit.temporal_score;
        if (hit.staleness_warning)
            std::cout << "  ⚠ " << hit.staleness_reason;
        std::cout << "\n";
    }

    // ── Collection handle (convenience) ──────────────────────────
    tidevec::Collection col(db, "docs");
    auto resp2 = col.search(query, {.top_k = 3});
    std::cout << "\nCollection handle search: " << resp2.hits.size() << " hits ✓\n";

    std::cout << "\nTideVec C++ SDK quickstart complete ✓\n";
    std::cout << "Docs: https://tidevec.com/docs\n";
    return 0;
}
