// gpubench — validate TideVec's CUDA engines (include/tidevec/accelerator/
// gpu_engine.hpp + src/accelerator/gpu_kernels.cu) on real data.
//
// Reads the .fbin/.ibin files written by benchmarks/ann/prepare.py
// (use an angular dataset: both engines assume cosine on unit vectors).
//
//   gpubench --dir ../ann/data/glove-100-angular-100000 --engine brute
//   gpubench --dir ... --engine cagra --itopk 64,128,256
//
// Prints one JSON object: build time, and recall@10 / QPS per setting.

#include <tidevec/accelerator/gpu_engine.hpp>

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

using namespace tidevec::accel;
using Clock = std::chrono::steady_clock;

namespace {

struct Matrix {
    int32_t n = 0, d = 0;
    std::vector<float> data;
};

Matrix read_fbin(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    Matrix m;
    f.read(reinterpret_cast<char*>(&m.n), 4);
    f.read(reinterpret_cast<char*>(&m.d), 4);
    m.data.resize(static_cast<std::size_t>(m.n) * m.d);
    f.read(reinterpret_cast<char*>(m.data.data()), m.data.size() * sizeof(float));
    return m;
}

std::vector<int32_t> read_ibin(const std::string& path, int32_t& n, int32_t& k) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    f.read(reinterpret_cast<char*>(&n), 4);
    f.read(reinterpret_cast<char*>(&k), 4);
    std::vector<int32_t> out(static_cast<std::size_t>(n) * k);
    f.read(reinterpret_cast<char*>(out.data()), out.size() * 4);
    return out;
}

double recall_at(const AccelSearchResult& r, const std::vector<int32_t>& gt, int gt_k,
                 int nq, int k) {
    long hits = 0;
    for (int q = 0; q < nq; ++q) {
        std::unordered_set<int32_t> truth(gt.begin() + q * gt_k, gt.begin() + q * gt_k + k);
        for (int i = 0; i < k; ++i) hits += truth.count(static_cast<int32_t>(r.indices[q * k + i]));
    }
    return static_cast<double>(hits) / (static_cast<double>(nq) * k);
}

}  // namespace

int main(int argc, char** argv) {
    std::string dir, engine = "brute", itopks = "64";
    int k = 10, batch = 256, degree = 64, n_iter = 20;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&] { return std::string(argv[++i]); };
        if (a == "--dir") dir = next();
        else if (a == "--engine") engine = next();
        else if (a == "--itopk") itopks = next();
        else if (a == "--batch") batch = std::stoi(next());
        else if (a == "--degree") degree = std::stoi(next());
        else if (a == "--iter") n_iter = std::stoi(next());
        else { std::cerr << "unknown arg " << a << "\n"; return 2; }
    }
    Matrix base = read_fbin(dir + "/base.fbin");
    Matrix query = read_fbin(dir + "/query.fbin");
    int32_t gn, gk;
    auto gt = read_ibin(dir + "/gt.ibin", gn, gk);
    int nq = query.n;

    std::ostringstream js;
    js << "{\"engine\":\"" << engine << "\",\"n\":" << base.n << ",\"d\":" << base.d
       << ",\"nq\":" << nq << ",\"runs\":[";

    if (engine == "brute") {
        GpuBruteForceEngine eng;
        auto t0 = Clock::now();
        eng.add(base.data.data(), base.n, base.d);
        double add_s = std::chrono::duration<double>(Clock::now() - t0).count();
        // The engine allocates an nq x n distance matrix per call: batch queries.
        AccelSearchResult all;
        all.indices.resize(static_cast<std::size_t>(nq) * k);
        auto ts = Clock::now();
        for (int s = 0; s < nq; s += batch) {
            int b = std::min(batch, nq - s);
            auto r = eng.search(query.data.data() + static_cast<std::size_t>(s) * query.d, b,
                                query.d, k);
            std::copy(r.indices.begin(), r.indices.end(), all.indices.begin() + s * k);
        }
        double wall = std::chrono::duration<double>(Clock::now() - ts).count();
        js << "{\"batch\":" << batch << ",\"recall\":" << recall_at(all, gt, gk, nq, k)
           << ",\"qps\":" << nq / wall << "}],\"build_s\":" << add_s;
    } else {
        CagraStyleEngine::Config cfg;
        cfg.degree = degree;
        cfg.n_iter = n_iter;
        bool first = true;
        double build_s = 0;
        std::stringstream ss(itopks);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            cfg.itopk_size = std::stoi(tok);
            CagraStyleEngine eng(cfg);
            eng.add(base.data.data(), base.n, base.d);
            auto tb = Clock::now();
            eng.build_index();
            cudaDeviceSynchronize();
            build_s = std::chrono::duration<double>(Clock::now() - tb).count();
            auto ts = Clock::now();
            auto r = eng.search(query.data.data(), nq, query.d, k);
            double wall = std::chrono::duration<double>(Clock::now() - ts).count();
            js << (first ? "" : ",") << "{\"itopk\":" << cfg.itopk_size
               << ",\"recall\":" << recall_at(r, gt, gk, nq, k) << ",\"qps\":" << nq / wall
               << ",\"build_s\":" << build_s << "}";
            first = false;
            std::cerr << "  itopk=" << cfg.itopk_size << " build " << build_s << " s\n";
        }
        js << "],\"degree\":" << degree << ",\"nn_descent_iter\":" << n_iter;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    js << ",\"gpu\":\"" << prop.name << "\"}";
    std::cout << js.str() << std::endl;
    return 0;
}
