#pragma once
// ================================================================
// tpu_engine.hpp — XLA/TPU accelerated ANN engine
//
// TPU ARCHITECTURE for vector search:
//
//   TPU MXU (Matrix Multiply Unit) is purpose-built for exactly
//   the operation vector search needs: batched matmul.
//
//   search(queries[B,D], database[N,D]) → top_k
//   = argmax(queries @ database.T)     for cosine similarity
//   = [B,D] @ [D,N] → [B,N] matmul     ← MXU does this natively
//
//   TPU v5e: 256×256 systolic array, 2e14 bf16 FLOPS/s
//   For B=1000, D=768, N=1M: ~1.5T FLOPs → ~7.5ms on TPU v5e
//
// XLA C++ API:
//   - Builds a computation graph: XlaBuilder → XlaComputation
//   - XLA JIT-compiles to optimised TPU instructions
//   - Execution: LocalClient::Execute → literals in/out
//
// COMPILATION:
//   With XLA:    -DTIDEVEC_XLA_ENABLED, link against libxla_client
//   Without XLA: transparent CPU fallback
//
// TPU SEARCH ALGORITHM:
//   1. Load database shard into TPU HBM (~16GB per TPU chip)
//   2. For each query batch:
//      a. Transfer queries (B × D floats) via PCIe/ICI
//      b. XLA matmul: S[B,N] = Q[B,D] × DB^T[D,N]
//      c. XLA top-k: argmax along N axis → indices[B,top_k]
//      d. Transfer results back
//   3. Multi-chip (pod): scatter-gather across TPU chips
// ================================================================

#include <tidevec/accelerator/device.hpp>
#include <tidevec/accelerator/cpu_engine.hpp>

#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <stdexcept>
#include <cmath>

// ================================================================
// XLA path — compiled when XLA client library available
// ================================================================
#ifdef TIDEVEC_XLA_ENABLED
#include "xla/client/xla_builder.h"
#include "xla/client/client_library.h"
#include "xla/client/local_client.h"
#include "xla/literal.h"
#include "xla/shape_util.h"
#include "xla/statusor.h"

namespace tidevec {
namespace accel {

// ================================================================
// XlaBatchMatmulEngine — TPU-native batch cosine search via XLA
//
// Compiles an XLA computation once, then executes it repeatedly.
// The compiled program is JIT-specialised for (B, N, D) shapes,
// so use fixed batch sizes for best performance.
// ================================================================
class XlaBatchMatmulEngine : public AnnEngine {
public:
    struct Config {
        int  batch_size  = 256;     // queries per XLA execution
        bool bfloat16    = true;    // use bf16 on TPU (faster, less precise)
        int  top_k_max   = 100;
    };

    explicit XlaBatchMatmulEngine(Config cfg = {}) : cfg_(cfg) {
        // Get local XLA client (connects to TPU runtime or GPU)
        auto client_or = xla::ClientLibrary::GetOrCreateLocalClient();
        if (!client_or.ok())
            throw std::runtime_error("XLA client init failed: " +
                client_or.status().ToString());
        client_ = client_or.value();
    }

    void add(const float* data, int64_t n, int64_t dim) override {
        if (dim_ == 0) dim_ = dim;
        size_t old = db_.size();
        db_.resize(old + n * dim);
        // Normalise during insertion (cosine search = dot of unit vectors)
        for (int64_t i = 0; i < n; ++i) {
            float norm = 0.0f;
            for (int64_t d = 0; d < dim; ++d)
                norm += data[i*dim+d] * data[i*dim+d];
            norm = std::sqrt(norm) + 1e-10f;
            for (int64_t d = 0; d < dim; ++d)
                db_[old + i*dim+d] = data[i*dim+d] / norm;
        }
        n_ += n;
        compiled_computation_.reset();  // invalidate on new data
    }

    AccelSearchResult search(const float* queries, int64_t n_q,
                             int64_t dim, int top_k) override {
        auto t0 = std::chrono::steady_clock::now();
        _ensure_compiled(static_cast<int>(n_q), static_cast<int>(dim), top_k);

        // Normalise queries
        std::vector<float> nq(n_q * dim);
        for (int64_t i = 0; i < n_q; ++i) {
            float norm = 0.0f;
            for (int64_t d = 0; d < dim; ++d)
                norm += queries[i*dim+d] * queries[i*dim+d];
            norm = std::sqrt(norm) + 1e-10f;
            for (int64_t d = 0; d < dim; ++d)
                nq[i*dim+d] = queries[i*dim+d] / norm;
        }

        // Build query literal
        xla::Shape q_shape = xla::ShapeUtil::MakeShape(
            xla::F32, {static_cast<int64_t>(n_q), dim_});
        auto q_lit = xla::Literal::CreateFromArray(
            xla::Array2D<float>(nq.data(), n_q, dim));

        // Build database literal (for this execution)
        xla::Shape db_shape = xla::ShapeUtil::MakeShape(
            xla::F32, {n_, dim_});
        auto db_lit = xla::Literal::CreateFromArray(
            xla::Array2D<float>(db_.data(), n_, dim_));

        // Execute compiled computation
        auto result_or = client_->Execute(
            *compiled_computation_,
            {client_->LiteralToShapedBuffer(*q_lit).value(),
             client_->LiteralToShapedBuffer(*db_lit).value()},
            xla::ExecutionOptions{});

        if (!result_or.ok())
            throw std::runtime_error("XLA execute failed: " +
                result_or.status().ToString());

        // Parse result (top_k indices + distances)
        auto result_lit = client_->ShapedBufferToLiteral(
            result_or.value()).value();

        AccelSearchResult res;
        res.n_queries   = static_cast<int>(n_q);
        res.top_k       = top_k;
        res.device_used = DeviceType::TPU;
        res.indices.resize(n_q * top_k);
        res.distances.resize(n_q * top_k);

        // Extract indices from result literal
        // (XLA top_k returns (values, indices) tuple)
        const auto& indices_lit = result_lit.tuple_element(1);
        const int32_t* idx_data = indices_lit.data<int32_t>().data();
        for (int64_t i = 0; i < n_q * top_k; ++i)
            res.indices[i] = idx_data[i];

        const auto& values_lit = result_lit.tuple_element(0);
        const float* val_data = values_lit.data<float>().data();
        for (int64_t i = 0; i < n_q * top_k; ++i)
            res.distances[i] = 1.0f - val_data[i];  // cosine dist = 1 - sim

        auto t1 = std::chrono::steady_clock::now();
        res.latency_ms = std::chrono::duration<double,std::milli>(t1-t0).count();
        return res;
    }

    void reset() override {
        db_.clear(); n_ = 0; dim_ = 0;
        compiled_computation_.reset();
    }

    DeviceInfo device_info() const override {
        DeviceInfo info;
        info.type      = DeviceType::TPU;
        info.name      = "XLA/TPU (batch=" + std::to_string(cfg_.batch_size) + ")";
        info.available = true;
        return info;
    }
    DeviceType device_type() const override { return DeviceType::TPU; }

private:
    void _ensure_compiled(int n_q, int dim, int top_k) {
        if (compiled_computation_) return;

        // Build XLA computation: matmul + top_k
        xla::XlaBuilder builder("tidevec_search");

        // Params: queries[n_q, dim], database[n_, dim]
        auto q  = xla::Parameter(&builder, 0,
            xla::ShapeUtil::MakeShape(xla::F32, {n_q, dim_}), "q");
        auto db = xla::Parameter(&builder, 1,
            xla::ShapeUtil::MakeShape(xla::F32, {n_, dim_}), "db");

        // S[n_q, n_] = Q @ DB^T
        auto scores = xla::Dot(q, xla::Transpose(db, {1, 0}));

        // top-k along the N axis
        auto topk = xla::TopK(scores, top_k, /*largest=*/true);

        auto comp_or = builder.Build(topk);
        if (!comp_or.ok())
            throw std::runtime_error("XLA build failed: " +
                comp_or.status().ToString());

        auto exec_or = client_->Compile(comp_or.value(), {}, {});
        if (!exec_or.ok())
            throw std::runtime_error("XLA compile failed: " +
                exec_or.status().ToString());

        compiled_computation_ = std::move(exec_or.value());
    }

    Config              cfg_;
    xla::LocalClient*   client_ = nullptr;
    std::unique_ptr<xla::LocalExecutable> compiled_computation_;
    std::vector<float>  db_;
    int64_t             n_ = 0, dim_ = 0;
};

} // namespace accel
} // namespace tidevec

#else
// ================================================================
// NO XLA — CPU fallback stub with honest reporting
// ================================================================
namespace tidevec {
namespace accel {

struct XlaStubConfig { int batch_size=256; bool bfloat16=true; int top_k_max=100; };

class XlaBatchMatmulEngine : public AnnEngine {
public:
    using Config = XlaStubConfig;
    explicit XlaBatchMatmulEngine(Config = XlaStubConfig{}) : cpu_(true) {
        std::cout << "[TideVec] XLA/TPU not compiled in. "
                  << "Rebuild with -DTIDEVEC_XLA_ENABLED to enable TPU.\n"
                  << "          Using CPU fallback.\n";
    }

    void add(const float* d, int64_t n, int64_t dim) override { cpu_.add(d,n,dim); }

    AccelSearchResult search(const float* q, int64_t nq, int64_t dim, int k) override {
        auto r = cpu_.search(q, nq, dim, k);
        r.device_used = DeviceType::CPU;
        return r;
    }

    void reset() override { cpu_.reset(); }

    DeviceInfo device_info() const override {
        DeviceInfo info;
        info.type      = DeviceType::TPU;
        info.name      = "XLA/TPU stub (not compiled; -DTIDEVEC_XLA_ENABLED)";
        info.available = false;
        return info;
    }
    DeviceType device_type() const override { return DeviceType::TPU; }

private:
    CpuFlatEngine cpu_;
};

} // namespace accel
} // namespace tidevec
#endif // TIDEVEC_XLA_ENABLED
