// ================================================================
// tpu_kernels.cc — XLA computation graphs for TPU vector search
//
// TPU ARCHITECTURE for 1B vector search:
//
//   TPU v5e MXU: 256×256 systolic array
//   Peak throughput: 2e14 bf16 FLOPS/s per chip
//
//   For B=1000 queries, D=768, N=100M vectors per TPU chip:
//   FLOPs needed: B × D × N = 1000 × 768 × 100M = 76.8 TFLOPs
//   Time on TPU v5e: 76.8T / 200T = 0.38 seconds
//
//   With quantisation (int8/bf16):
//   int8: 400 TOPS → 0.19 seconds for same workload
//
//   For 1B total: 10 TPU chips, each handling 100M shard
//   Parallel time: 0.19 seconds at int8 across 10 chips
//   Network: scatter queries (1000×768×4=3MB) + gather top-K (negligible)
//
// XLA COMPILATION STRATEGY:
//   1. XlaBuilder constructs computation graph
//   2. XlaBuilder::Build() → XlaComputation (portable HLO)
//   3. LocalClient::Compile() → XlaExecutable (GPU/TPU machine code)
//   4. LocalClient::Execute() → result literals
//
//   Key: compilation happens ONCE per (batch_size, N, D) shape.
//   Cache compiled executables keyed by shape.
//
// OPERATIONS:
//   cosine_search: normalise queries + database, then matmul
//   exact_l2:      expand L2 = ||q||² - 2(q·d) + ||d||²
//   top_k:         XLA built-in TopK operation
//   bf16_cast:     convert f32→bf16 for TPU MXU efficiency
// ================================================================

#ifdef TIDEVEC_XLA_ENABLED

#include "xla/client/xla_builder.h"
#include "xla/client/client_library.h"
#include "xla/client/local_client.h"
#include "xla/literal.h"
#include "xla/shape_util.h"
#include "xla/client/lib/math.h"
#include "xla/client/lib/matrix.h"

#include <map>
#include <tuple>
#include <mutex>
#include <memory>
#include <string>
#include <stdexcept>
#include <iostream>

namespace tidevec {
namespace tpu {

// ================================================================
// Shape key for compilation cache
// ================================================================
using ShapeKey = std::tuple<int64_t, int64_t, int64_t, int>;
// (n_queries, n_db, dim, top_k)

// ================================================================
// XlaComputationCache
// Caches compiled executables per (shape, device_type)
// Avoids recompilation on every call — critical for latency
// ================================================================
class XlaComputationCache {
public:
    static XlaComputationCache& instance() {
        static XlaComputationCache cache;
        return cache;
    }

    xla::LocalExecutable* get_or_compile(
        xla::LocalClient* client,
        const ShapeKey& key,
        bool use_bf16 = true)
    {
        std::lock_guard lock(mutex_);
        auto it = cache_.find(key);
        if (it != cache_.end()) return it->second.get();

        auto [n_q, n_db, dim, top_k] = key;

        // Build XLA computation
        xla::XlaBuilder builder("tidevec_search_" +
            std::to_string(n_q) + "x" + std::to_string(n_db));

        // Types
        xla::PrimitiveType float_type = use_bf16 ? xla::BF16 : xla::F32;

        // Parameters
        auto q_shape  = xla::ShapeUtil::MakeShape(float_type, {n_q, dim});
        auto db_shape = xla::ShapeUtil::MakeShape(float_type, {n_db, dim});

        xla::XlaOp q  = xla::Parameter(&builder, 0, q_shape,  "queries");
        xla::XlaOp db = xla::Parameter(&builder, 1, db_shape, "database");

        // ---- Cosine search computation -------------------------
        // Step 1: L2-normalise queries (for cosine similarity)
        // norm = sqrt(sum(q^2, axis=1, keepdims=True))
        xla::XlaOp q_sq   = xla::Mul(q, q);
        xla::XlaOp q_norm = xla::Sqrt(
            xla::ReduceSum(q_sq, {1}));  // [n_q]
        // Reshape for broadcasting: [n_q, 1]
        xla::XlaOp q_norm_bc = xla::Reshape(q_norm, {n_q, 1});
        xla::XlaOp eps = xla::ConstantR0<float>(&builder, 1e-8f);
        // q_unit = q / (norm + eps)
        xla::XlaOp q_unit = xla::Div(q,
            xla::Add(xla::BroadcastInDim(q_norm_bc, {n_q, dim}, {0, 1}),
                     xla::Broadcast(eps, {n_q, dim})));

        // Step 2: Same for database vectors
        xla::XlaOp db_sq   = xla::Mul(db, db);
        xla::XlaOp db_norm = xla::Sqrt(xla::ReduceSum(db_sq, {1}));  // [n_db]
        xla::XlaOp db_norm_bc = xla::Reshape(db_norm, {n_db, 1});
        xla::XlaOp db_unit = xla::Div(db,
            xla::Add(xla::BroadcastInDim(db_norm_bc, {n_db, dim}, {0, 1}),
                     xla::Broadcast(eps, {n_db, dim})));

        // Step 3: Similarity matrix S[n_q, n_db] = q_unit @ db_unit.T
        // XLA DotGeneral handles batched and transposed matmul
        xla::DotDimensionNumbers dot_dims;
        dot_dims.add_lhs_contracting_dimensions(1);  // contract over dim
        dot_dims.add_rhs_contracting_dimensions(1);  // contract over dim (transposed)
        xla::XlaOp similarity = xla::DotGeneral(q_unit, db_unit, dot_dims);
        // similarity shape: [n_q, n_db]

        // Step 4: Top-K (largest cosine similarities = nearest neighbours)
        // XLA's TopK returns (values, indices) tuple
        xla::XlaOp topk_result = xla::TopK(similarity, top_k, /*largest=*/true);

        // Compile
        auto comp_or = builder.Build(topk_result);
        if (!comp_or.ok())
            throw std::runtime_error("XLA Build failed: " +
                comp_or.status().ToString());

        // Device-specific compilation options
        xla::ExecutableBuildOptions build_opts;
        build_opts.set_num_replicas(1);
        build_opts.set_num_partitions(1);
        // On TPU: enable bf16 arithmetic
        if (use_bf16) {
            build_opts.mutable_debug_options()->set_xla_gpu_use_runtime_fusion(true);
        }

        auto exec_or = client->Compile(comp_or.value(), {}, build_opts);
        if (!exec_or.ok())
            throw std::runtime_error("XLA Compile failed: " +
                exec_or.status().ToString());

        auto* ptr = exec_or.value().get();
        cache_[key] = std::move(exec_or.value());
        std::cout << "[XLA] Compiled cosine search for shape ("
                  << n_q << "," << n_db << "," << dim << ") top_k=" << top_k << "\n";
        return ptr;
    }

private:
    std::mutex mutex_;
    std::map<ShapeKey, std::unique_ptr<xla::LocalExecutable>> cache_;
};

// ================================================================
// TPU IVF-PQ computation graph
//
// For 1B+ vectors that don't fit in HBM:
//   Use Product Quantization codes (1 byte per subspace)
//   ADC (Asymmetric Distance Computation) via XLA:
//     distances[n_q, N] = sum_m(adc_table[n_q, m, code[n, m]])
//
// This is a gather + reduce operation — maps well to XLA
// ================================================================
xla::XlaOp build_adc_computation(
    xla::XlaBuilder* builder,
    xla::XlaOp adc_tables,   // [n_q, M, K] precomputed lookup tables
    xla::XlaOp pq_codes,     // [N, M]  uint8 codes
    int n_q, int N, int M, int K)
{
    // Flatten pq_codes to indices for gather
    // XLA Gather: for each (n, m), retrieve adc_tables[q, m, code[n,m]]

    // Compute ADC distances via einsum-like operation:
    // 1. one_hot encode codes: [N, M, K]
    auto codes_int = xla::ConvertElementType(pq_codes, xla::S32);
    auto codes_oh  = xla::OneHot(codes_int, K, xla::S32, xla::F32);
    // codes_oh: [N, M, K] one-hot

    // 2. Broadcast and multiply with ADC tables
    // tables: [n_q, M, K], codes_oh: [N, M, K]
    // Result: [n_q, N, M, K] → sum over K and M → [n_q, N]
    // Equivalent to: dist[q, n] = sum_m(sum_k(table[q,m,k] * onehot[n,m,k]))

    // Reshape for batched matmul:
    // tables_2d: [n_q*M, K]
    auto tables_2d = xla::Reshape(adc_tables, {n_q * M, K});
    // codes_2d:  [N*M, K]
    auto codes_2d  = xla::Reshape(codes_oh, {(int64_t)N * M, K});

    // Batched dot: [n_q*M, K] @ [N*M, K]^T → not quite right
    // Better: use Einsum
    // dist[q, n] = sum_{m,k} table[q,m,k] * onehot[n,m,k]
    // = Einsum("qmk,nmk->qn", tables, codes_oh)
    xla::XlaOp distances = xla::Einsum(adc_tables, codes_oh, "qmk,nmk->qn");
    // distances: [n_q, N]

    return distances;
}

// ================================================================
// Billion-scale search strategy:
//
// For 1B vectors across P TPU chips:
//   Each chip holds 1B/P vectors in HBM
//   All chips receive the same batch of queries
//   Each chip computes top-K candidates from its shard
//   Results merged on coordinator
//
// XLA SPMD (Single Program Multiple Data) handles this automatically
// when run on a TPU pod via PartitionedCall
// ================================================================
class TpuBillionScaleSearch {
public:
    explicit TpuBillionScaleSearch(xla::LocalClient* client,
                                   int n_chips = 8,
                                   bool use_bf16 = true)
        : client_(client)
        , n_chips_(n_chips)
        , use_bf16_(use_bf16)
    {}

    // Execute search on a single shard (run in parallel across chips)
    xla::StatusOr<std::vector<std::vector<int64_t>>>
    search_shard(
        const std::vector<float>& queries,  // [n_q, dim]
        const std::vector<float>& shard,    // [shard_n, dim]
        int n_q, int shard_n, int dim, int top_k)
    {
        auto key = std::make_tuple(
            static_cast<int64_t>(n_q),
            static_cast<int64_t>(shard_n),
            static_cast<int64_t>(dim),
            top_k);

        auto* exec = XlaComputationCache::instance()
            .get_or_compile(client_, key, use_bf16_);

        // Create input literals
        xla::Shape q_shape  = xla::ShapeUtil::MakeShape(xla::F32, {n_q, dim});
        xla::Shape db_shape = xla::ShapeUtil::MakeShape(xla::F32, {shard_n, dim});

        auto q_lit  = xla::LiteralUtil::CreateR2FromArray2D(
            xla::Array2D<float>(n_q, dim, queries.data()));
        auto db_lit = xla::LiteralUtil::CreateR2FromArray2D(
            xla::Array2D<float>(shard_n, dim, shard.data()));

        // Transfer to device
        TF_ASSIGN_OR_RETURN(auto q_buf,
            client_->LiteralToShapedBuffer(q_lit));
        TF_ASSIGN_OR_RETURN(auto db_buf,
            client_->LiteralToShapedBuffer(db_lit));

        // Execute
        xla::ExecutableRunOptions run_opts;
        TF_ASSIGN_OR_RETURN(auto result,
            exec->Run({&q_buf, &db_buf}, run_opts));

        // Extract indices from tuple result
        TF_ASSIGN_OR_RETURN(auto result_lit,
            client_->ShapedBufferToLiteral(result));

        // result is a tuple (values[n_q, top_k], indices[n_q, top_k])
        const auto& idx_lit = result_lit.tuple_element(1);

        std::vector<std::vector<int64_t>> out(n_q, std::vector<int64_t>(top_k));
        const int32_t* idx_data = idx_lit.data<int32_t>().data();
        for (int q = 0; q < n_q; ++q)
            for (int k = 0; k < top_k; ++k)
                out[q][k] = idx_data[q * top_k + k];

        return out;
    }

private:
    xla::LocalClient* client_;
    int  n_chips_;
    bool use_bf16_;
};

} // namespace tpu
} // namespace tidevec

#endif // TIDEVEC_XLA_ENABLED
