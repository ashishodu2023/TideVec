#pragma once
// ================================================================
// DriftBridge — zero-downtime embedding model migration
//
// Problem: when you upgrade your embedding model (e.g. OpenAI
// text-embedding-ada-002 → text-embedding-3-small), every vector
// in the collection is now in the wrong space. Naive approach:
// drop collection, re-embed everything, re-insert. This means
// downtime — the collection is empty during migration.
//
// DriftBridge solves this with a dual-index approach:
//   1. "Shadow" index receives all new writes with new model
//   2. "Live" index continues serving reads from old model
//   3. Background migration moves old vectors to shadow index
//      by calling a user-supplied re-embedding callback
//   4. Once migration is complete, atomically swap live → shadow
//   5. Old index is dropped. Zero downtime. No empty period.
//
// Usage (server-side, called via REST):
//   POST /v1/collections/{name}/drift/start
//     Body: {"old_dim": 1536, "new_dim": 1536, "model": "3-small"}
//
//   The client is responsible for providing a migration callback
//   that re-embeds a text chunk with the new model. TideVec calls
//   this callback via a registered HTTP endpoint during migration.
// ================================================================

#include <tidevec/index/tv_index.hpp>
#include <tidevec/core/cortex_vector.hpp>

#include <string>
#include <functional>
#include <atomic>
#include <thread>
#include <mutex>
#include <vector>
#include <iostream>

namespace tidevec {

// Callback type: given a vector ID and its payload, return the
// new embedding in the target model's space.
// Return empty vector to skip this ID (e.g. source doc deleted).
using ReEmbedFn = std::function<std::vector<float>(
    const std::string& id,
    const std::unordered_map<std::string, std::string>& payload)>;

enum class DriftPhase {
    IDLE,         // no migration in progress
    MIGRATING,    // background migration running
    SWAPPING,     // atomic swap in progress
    COMPLETE,     // migration done, old index dropped
    FAILED,       // migration failed
};

struct DriftProgress {
    DriftPhase  phase          = DriftPhase::IDLE;
    std::size_t total_vectors  = 0;
    std::size_t migrated       = 0;
    std::size_t skipped        = 0;
    std::string error;

    float pct() const {
        if (total_vectors == 0) return 0.0f;
        return 100.0f * float(migrated + skipped) / float(total_vectors);
    }
};

class DriftBridge {
public:
    explicit DriftBridge(TVIndexConfig old_cfg, TVIndexConfig new_cfg)
        : old_cfg_(std::move(old_cfg))
        , new_cfg_(std::move(new_cfg))
        , phase_(DriftPhase::IDLE)
    {}

    DriftPhase phase() const { return phase_.load(); }

    DriftProgress progress() const {
        std::lock_guard<std::mutex> lg(progress_mutex_);
        return progress_;
    }

    // Start background migration. re_embed_fn is called for each
    // vector in the live index to produce a new-model embedding.
    // new_vectors receives the write stream during migration —
    // both live AND shadow indexes are updated on every write.
    bool start(const std::vector<CortexVector>& live_vectors,
               ReEmbedFn re_embed_fn,
               std::function<void(CortexVector)> on_complete) {

        if (phase_.load() != DriftPhase::IDLE) return false;

        phase_.store(DriftPhase::MIGRATING);
        {
            std::lock_guard<std::mutex> lg(progress_mutex_);
            progress_.phase         = DriftPhase::MIGRATING;
            progress_.total_vectors = live_vectors.size();
            progress_.migrated      = 0;
            progress_.skipped       = 0;
        }

        // Build shadow index
        shadow_index_ = std::make_unique<TVIndex>(new_cfg_);

        // Background migration thread
        migration_thread_ = std::thread([this, live_vectors,
                                         re_embed_fn, on_complete]() {
            std::cout << "[DriftBridge] Starting migration of "
                      << live_vectors.size() << " vectors\n";

            for (const auto& v : live_vectors) {
                if (phase_.load() == DriftPhase::FAILED) return;

                try {
                    auto new_emb = re_embed_fn(v.id, v.payload);
                    if (new_emb.empty()) {
                        std::lock_guard<std::mutex> lg(progress_mutex_);
                        ++progress_.skipped;
                        continue;
                    }

                    CortexVector nv = v;
                    nv.embedding = std::move(new_emb);
                    shadow_index_->insert(nv);

                    std::lock_guard<std::mutex> lg(progress_mutex_);
                    ++progress_.migrated;

                    if (progress_.migrated % 1000 == 0)
                        std::cout << "[DriftBridge] " << progress_.pct()
                                  << "% complete\n";

                } catch (const std::exception& e) {
                    std::cerr << "[DriftBridge] Error migrating "
                              << v.id << ": " << e.what() << "\n";
                    std::lock_guard<std::mutex> lg(progress_mutex_);
                    ++progress_.skipped;
                }
            }

            // Migration complete — signal caller to atomic swap
            phase_.store(DriftPhase::SWAPPING);
            {
                std::lock_guard<std::mutex> lg(progress_mutex_);
                progress_.phase = DriftPhase::SWAPPING;
            }
            std::cout << "[DriftBridge] Migration complete. "
                      << "Swapping live → shadow index.\n";

            // Caller does the actual swap (needs collection lock)
            on_complete(CortexVector{});  // signal: swap ready

            phase_.store(DriftPhase::COMPLETE);
            {
                std::lock_guard<std::mutex> lg(progress_mutex_);
                progress_.phase = DriftPhase::COMPLETE;
            }
            std::cout << "[DriftBridge] Swap complete. "
                      << "Zero downtime migration finished.\n";
        });
        migration_thread_.detach();
        return true;
    }

    // Forward new writes to BOTH live and shadow during migration
    void shadow_write(const CortexVector& v, ReEmbedFn re_embed_fn) {
        if (phase_.load() != DriftPhase::MIGRATING || !shadow_index_)
            return;
        try {
            auto new_emb = re_embed_fn(v.id, v.payload);
            if (new_emb.empty()) return;
            CortexVector nv = v;
            nv.embedding = std::move(new_emb);
            shadow_index_->insert(nv);
        } catch (...) {}
    }

    // Take ownership of the shadow index after migration completes
    std::unique_ptr<TVIndex> take_shadow() {
        return std::move(shadow_index_);
    }

    void abort() {
        phase_.store(DriftPhase::FAILED);
        std::lock_guard<std::mutex> lg(progress_mutex_);
        progress_.phase = DriftPhase::FAILED;
        progress_.error = "Aborted by user";
    }

private:
    TVIndexConfig old_cfg_;
    TVIndexConfig new_cfg_;
    std::atomic<DriftPhase> phase_;
    std::unique_ptr<TVIndex> shadow_index_;
    std::thread migration_thread_;
    mutable std::mutex progress_mutex_;
    DriftProgress progress_;
};

} // namespace tidevec
