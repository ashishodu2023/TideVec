// tvbench — standard ANN benchmark for TideVec's TVIndex.
//
// Drives tidevec::TVIndex directly (no Python, JSON or WAL in the loop) so
// the numbers measure the index itself. Inputs are the .fbin/.ibin files
// written by prepare.py:
//
//   <dir>/base.fbin    int32 n, int32 d, n*d float32
//   <dir>/query.fbin   int32 nq, int32 d, nq*d float32
//   <dir>/gt.ibin      int32 nq, int32 k, nq*k int32 (exact neighbours)
//
// Output: one JSON object on stdout (build time, memory, and for every
// ef_search value recall@k, QPS, mean/p50/p95/p99 latency).

#include <tidevec/index/tv_index.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

using namespace tidevec;
using Clock = std::chrono::steady_clock;

namespace {

struct Matrix {
    int32_t n = 0, d = 0;
    std::vector<float> data;
    const float* row(std::size_t i) const { return data.data() + i * d; }
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

std::vector<std::vector<int32_t>> read_ibin(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    int32_t n, k;
    f.read(reinterpret_cast<char*>(&n), 4);
    f.read(reinterpret_cast<char*>(&k), 4);
    std::vector<std::vector<int32_t>> out(n, std::vector<int32_t>(k));
    for (auto& row : out) f.read(reinterpret_cast<char*>(row.data()), k * 4);
    return out;
}

long rss_kb(const char* field = "VmRSS:") {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line))
        if (line.rfind(field, 0) == 0) return std::stol(line.substr(std::string(field).size()));
    return -1;
}

std::vector<int> parse_ints(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) out.push_back(std::stoi(tok));
    return out;
}

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    std::size_t i = static_cast<std::size_t>(p / 100.0 * (v.size() - 1) + 0.5);
    return v[std::min(i, v.size() - 1)];
}

}  // namespace

int main(int argc, char** argv) {
    std::string dir, metric = "l2", efs = "10,16,32,64,128,256,512";
    int M = 16, efc = 200, k = 10, nq_limit = 0, threads = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&] { return std::string(argv[++i]); };
        if (a == "--dir") dir = next();
        else if (a == "--metric") metric = next();
        else if (a == "--M") M = std::stoi(next());
        else if (a == "--efc") efc = std::stoi(next());
        else if (a == "--ef") efs = next();
        else if (a == "--k") k = std::stoi(next());
        else if (a == "--nq") nq_limit = std::stoi(next());
        else if (a == "--threads") threads = std::stoi(next());
        else { std::cerr << "unknown arg " << a << "\n"; return 2; }
    }
    if (dir.empty()) { std::cerr << "usage: tvbench --dir DIR [--metric l2|cosine]\n"; return 2; }

    Matrix base = read_fbin(dir + "/base.fbin");
    Matrix query = read_fbin(dir + "/query.fbin");
    auto gt = read_ibin(dir + "/gt.ibin");
    int nq = nq_limit > 0 ? std::min(nq_limit, query.n) : query.n;

    TVIndexConfig cfg;
    cfg.M = M;
    cfg.M0 = 2 * M;
    cfg.ef_construction = efc;
    cfg.metric = parse_metric(metric);
    cfg.temporal.temporal_blend = 0.0f;
    TVIndex index(cfg);

    long rss_before = rss_kb();
    auto t0 = Clock::now();
    for (int32_t i = 0; i < base.n; ++i) {
        CortexVector v;
        v.id = std::to_string(i);
        v.embedding.assign(base.row(i), base.row(i) + base.d);
        v.created_at = v.valid_from = 0;
        index.insert(v);
        if ((i + 1) % 100000 == 0) {
            double s = std::chrono::duration<double>(Clock::now() - t0).count();
            std::cerr << "  inserted " << (i + 1) << " in " << s << " s\n";
        }
    }
    double build_s = std::chrono::duration<double>(Clock::now() - t0).count();
    long rss_after = rss_kb();

    std::ostringstream js;
    js << "{\"engine\":\"tidevec\",\"n\":" << base.n << ",\"d\":" << base.d
       << ",\"nq\":" << nq << ",\"metric\":\"" << metric << "\",\"M\":" << M
       << ",\"ef_construction\":" << efc << ",\"build_s\":" << build_s
       << ",\"index_rss_mb\":" << (rss_after - rss_before) / 1024.0
       << ",\"peak_rss_mb\":" << rss_kb("VmHWM:") / 1024.0
       << ",\"vector_bytes_mb\":" << index.embedding_memory_bytes() / 1048576.0
       << ",\"runs\":[";

    bool first = true;
    for (int ef : parse_ints(efs)) {
        QueryOptions opts;
        opts.top_k = k;
        opts.ef_search = ef;
        opts.temporal_blend = 0.0f;
        opts.metric = metric;
        opts.include_staleness_warnings = false;

        std::vector<double> lat(nq);
        std::vector<int> hits(nq);
        auto run_one = [&](int q) {
            std::vector<float> qv(query.row(q), query.row(q) + query.d);
            auto s = Clock::now();
            auto res = index.search(qv, opts);
            lat[q] = std::chrono::duration<double, std::micro>(Clock::now() - s).count();
            std::unordered_set<int32_t> truth(gt[q].begin(), gt[q].begin() + k);
            int h = 0;
            for (const auto& r : res) h += truth.count(std::stoi(r.id));
            hits[q] = h;
        };

        auto ts = Clock::now();
        for (int q = 0; q < nq; ++q) run_one(q);
        double wall = std::chrono::duration<double>(Clock::now() - ts).count();

        double mt_qps = 0;
        if (threads > 1) {
            std::atomic<int> next{0};
            auto tm = Clock::now();
            std::vector<std::thread> pool;
            for (int t = 0; t < threads; ++t)
                pool.emplace_back([&] { for (int q; (q = next++) < nq;) run_one(q); });
            for (auto& th : pool) th.join();
            mt_qps = nq / std::chrono::duration<double>(Clock::now() - tm).count();
        }

        long total = 0;
        for (int h : hits) total += h;
        double mean = 0;
        for (double l : lat) mean += l;
        mean /= nq;
        js << (first ? "" : ",") << "{\"ef_search\":" << ef
           << ",\"recall\":" << static_cast<double>(total) / (static_cast<double>(nq) * k)
           << ",\"qps\":" << nq / wall << ",\"mean_us\":" << mean
           << ",\"p50_us\":" << percentile(lat, 50) << ",\"p95_us\":" << percentile(lat, 95)
           << ",\"p99_us\":" << percentile(lat, 99);
        if (threads > 1) js << ",\"qps_threads\":" << threads << ",\"qps_mt\":" << mt_qps;
        js << "}";
        first = false;
        std::cerr << "  ef=" << ef << " recall=" << static_cast<double>(total) / (nq * k)
                  << " qps=" << nq / wall << "\n";
    }
    js << "]}";
    std::cout << js.str() << std::endl;
    return 0;
}
