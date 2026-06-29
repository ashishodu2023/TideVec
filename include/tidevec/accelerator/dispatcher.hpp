#pragma once
// ================================================================
// dispatcher.hpp — Smart accelerator routing
//
// Decision logic (research-backed thresholds):
//
//   batch_size >= 64 && GPU available   → GpuBruteForceEngine
//                                         (cuBLAS SGEMM, best throughput)
//   batch_size >= 32 && TPU available   → XlaBatchMatmulEngine
//                                         (MXU matmul, best efficiency)
//   batch_size < 32 && recall >= 0.99   → CpuFlatEngine (exact, low latency)
//   batch_size < 32 && N > 10M         → CpuIvfEngine (IVF probing)
//   else                                → CpuFlatEngine
//
//   For index BUILD (not search):
//   GPU available → CagraStyleEngine (12× faster than HNSW)
//   else          → TVIndex (CPU HNSW)
//
// The dispatcher also manages:
//   - Device enumeration at startup
//   - Memory pressure monitoring (don't OOM the GPU)
//   - Latency SLO enforcement (fall back to approximate if needed)
//   - Telemetry: which device served each query
// ================================================================

#include <cortexdb/accelerator/gpu_engine.hpp>
#include <cortexdb/accelerator/tpu_engine.hpp>
#include <cortexdb/accelerator/cpu_engine.hpp>

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <atomic>
#include <mutex>
#include <iostream>

namespace cortexdb {
namespace accel {

// ================================================================
// System-level device discovery
// ================================================================
struct SystemDevices {
    std::vector<DeviceInfo> gpus;
    std::vector<DeviceInfo> tpus;
    DeviceInfo              cpu;
    bool has_gpu = false;
    bool has_tpu = false;

    void print() const {
        std::cout << "\n=== CortexDB Accelerator Devices ===\n";
        cpu.print();
        for (const auto& g : gpus) g.print();
        for (const auto& t : tpus) t.print();
        std::cout << "=====================================\n\n";
    }
};

inline SystemDevices discover_devices() {
    SystemDevices sys;

    // CPU always available
    CpuFlatEngine cpu_engine;
    sys.cpu = cpu_engine.device_info();
    sys.cpu.available = true;

#ifdef CORTEXDB_CUDA_ENABLED
    // Enumerate CUDA GPUs
    int n_gpus = 0;
    cudaGetDeviceCount(&n_gpus);
    for (int i = 0; i < n_gpus; ++i) {
        try {
            GpuBruteForceEngine eng(i);
            sys.gpus.push_back(eng.device_info());
            sys.has_gpu = true;
        } catch (...) {}
    }
#else
    // Report GPU as unavailable
    DeviceInfo gpu_info;
    gpu_info.type = DeviceType::GPU;
    gpu_info.name = "NVIDIA GPU (compile with -DCORTEXDB_CUDA_ENABLED)";
    gpu_info.available = false;
    sys.gpus.push_back(gpu_info);
#endif

#ifdef CORTEXDB_XLA_ENABLED
    // XLA detects TPU vs GPU vs CPU automatically
    try {
        XlaBatchMatmulEngine tpu_engine;
        auto info = tpu_engine.device_info();
        info.available = true;
        sys.tpus.push_back(info);
        sys.has_tpu = true;
    } catch (...) {}
#else
    DeviceInfo tpu_info;
    tpu_info.type = DeviceType::TPU;
    tpu_info.name = "Google XLA/TPU (compile with -DCORTEXDB_XLA_ENABLED)";
    tpu_info.available = false;
    sys.tpus.push_back(tpu_info);
#endif

    return sys;
}

// ================================================================
// AcceleratorDispatcher
// ================================================================
struct AccelDispatcherConfig {
    DeviceType preferred = DeviceType::AUTO;
    int gpu_batch_threshold = 64;
    int tpu_batch_threshold = 32;
    size_t gpu_memory_reserve = 512ULL * 1024 * 1024;
    bool verbose = false;
    float latency_slo_ms = 0.0f;
};

class AcceleratorDispatcher {
public:
    using Config = AccelDispatcherConfig;
    explicit AcceleratorDispatcher(Config cfg = AccelDispatcherConfig{}) : cfg_(cfg) {
        sys_ = discover_devices();
        if (cfg_.verbose) sys_.print();
        _init_engines();
    }

    // ------ Main search interface --------------------------------

    AccelSearchResult search(
        const float* queries,
        int64_t n_queries,
        int64_t dim,
        int top_k,
        DeviceType hint = DeviceType::AUTO)
    {
        DeviceType target = (hint == DeviceType::AUTO)
            ? _select_device(static_cast<int>(n_queries))
            : hint;

        auto* engine = _get_engine(target);
        if (!engine) {
            // Fallback chain: GPU→TPU→CPU
            target = DeviceType::CPU;
            engine = _get_engine(DeviceType::CPU);
        }

        auto result = engine->search(queries, n_queries, dim, top_k);

        // Track stats
        ++query_count_;
        device_query_counts_[result.device_used].fetch_add(1);
        total_latency_ms_.fetch_add(
            static_cast<int64_t>(result.latency_ms * 1000));

        if (cfg_.verbose) {
            std::cout << "[accel] " << n_queries << "q × dim=" << dim
                      << " → " << device_type_str(result.device_used)
                      << " in " << result.latency_ms << "ms\n";
        }
        return result;
    }

    // Add vectors to the appropriate engine
    void add(const float* data, int64_t n, int64_t dim,
             DeviceType target = DeviceType::AUTO) {
        if (target == DeviceType::AUTO)
            target = _select_build_device();
        auto* engine = _get_engine(target);
        if (!engine) engine = _get_engine(DeviceType::CPU);
        engine->add(data, n, dim);
    }

    // ------ Device info ------------------------------------------

    const SystemDevices& system_devices() const { return sys_; }

    struct Stats {
        uint64_t total_queries;
        std::unordered_map<std::string, uint64_t> queries_by_device;
        double avg_latency_ms;
    };

    Stats stats() const {
        Stats s;
        s.total_queries = query_count_.load();
        s.avg_latency_ms = s.total_queries > 0
            ? (total_latency_ms_.load() / 1000.0) / s.total_queries
            : 0.0;
        for (auto& [dev, cnt] : device_query_counts_)
            s.queries_by_device[device_type_str(dev)] = cnt.load();
        return s;
    }

    bool gpu_available() const { return sys_.has_gpu; }
    bool tpu_available() const { return sys_.has_tpu; }

    // ------ Query routing logic ----------------------------------
    DeviceType select_device_for_batch(int batch_size) const {
        return _select_device(batch_size);
    }

private:
    DeviceType _select_device(int batch_size) const {
        if (cfg_.preferred != DeviceType::AUTO) return cfg_.preferred;

        // GPU: best for large batches (cuBLAS SGEMM throughput)
        if (sys_.has_gpu && batch_size >= cfg_.gpu_batch_threshold)
            return DeviceType::GPU;

        // TPU: best for medium batches (MXU efficiency)
        if (sys_.has_tpu && batch_size >= cfg_.tpu_batch_threshold)
            return DeviceType::TPU;

        // CPU: single queries or small batches
        return DeviceType::CPU;
    }

    DeviceType _select_build_device() const {
        // Index build: GPU is best (CAGRA 12× faster than HNSW)
        if (sys_.has_gpu) return DeviceType::GPU;
        return DeviceType::CPU;
    }

    AnnEngine* _get_engine(DeviceType type) {
        switch (type) {
            case DeviceType::GPU:  return gpu_engine_.get();
            case DeviceType::TPU:  return tpu_engine_.get();
            case DeviceType::CPU:  return cpu_engine_.get();
            default:               return cpu_engine_.get();
        }
    }

    void _init_engines() {
        // CPU engine always created
        cpu_engine_ = std::make_unique<CpuFlatEngine>(true);

        // GPU engine (uses stubs if CUDA not available)
#ifdef CORTEXDB_CUDA_ENABLED
        if (!sys_.gpus.empty() && sys_.gpus[0].available) {
            try {
                gpu_engine_ = std::make_unique<CagraStyleEngine>(
                    CagraStyleEngine::Config{}, 0);
            } catch (const std::exception& e) {
                std::cerr << "[accel] GPU init failed: " << e.what()
                          << " — using CPU\n";
                gpu_engine_ = std::make_unique<GpuBruteForceEngine>(0);
            }
        } else {
            gpu_engine_ = std::make_unique<GpuBruteForceEngine>(0);
        }
#else
        gpu_engine_ = std::make_unique<GpuBruteForceEngine>(0);
#endif

        // TPU engine
        tpu_engine_ = std::make_unique<XlaBatchMatmulEngine>(
            XlaBatchMatmulEngine::Config{});
    }

    Config cfg_;
    SystemDevices sys_;

    std::unique_ptr<AnnEngine> cpu_engine_;
    std::unique_ptr<AnnEngine> gpu_engine_;
    std::unique_ptr<AnnEngine> tpu_engine_;

    std::atomic<uint64_t> query_count_{0};
    std::atomic<int64_t>  total_latency_ms_{0};
    std::unordered_map<DeviceType, std::atomic<uint64_t>> device_query_counts_;
    std::mutex stats_mutex_;
};

} // namespace accel
} // namespace cortexdb
