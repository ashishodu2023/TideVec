// ============================================================
// tidevec-router — multi-Raft shard router (EXPERIMENTAL)
//
//   tidevec-router --port 8090 --placement placement.json
//   tidevec-router --port 8090 --groups "http://a:8080,http://b:8080,http://c:8080;http://d:8080,..."
//
// --groups splits the id-hash space equally over the ';'-separated
// replica sets (one Raft group each). See multi_raft_router.hpp.
// ============================================================

#include <tidevec/cluster/multi_raft_router.hpp>

#include <csignal>
#include <fstream>
#include <iostream>
#include <sstream>

using tidevec::cluster::Placement;
using tidevec::cluster::MultiRaftRouter;

static MultiRaftRouter* g_router = nullptr;

static std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, sep))
        if (!item.empty()) out.push_back(item);
    return out;
}

static void usage() {
    std::cerr << "usage: tidevec-router [--host H] [--port P] [--threads N] [--timeout-ms MS]\n"
                 "                      (--placement FILE.json | --groups \"u1,u2,u3;u4,u5,u6\")\n";
}

int main(int argc, char* argv[]) {
    MultiRaftRouter::Config cfg;
    std::string placement_file, groups;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage(); std::exit(2); }
            return argv[++i];
        };
        if (a == "--host") cfg.host = next();
        else if (a == "--port") cfg.port = std::stoi(next());
        else if (a == "--threads") cfg.threads = std::stoi(next());
        else if (a == "--timeout-ms") cfg.timeout_ms = std::stoi(next());
        else if (a == "--placement") placement_file = next();
        else if (a == "--groups") groups = next();
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { std::cerr << "unknown argument: " << a << "\n"; usage(); return 2; }
    }
    try {
        if (!placement_file.empty()) {
            std::ifstream f(placement_file);
            if (!f) throw std::runtime_error("cannot read " + placement_file);
            cfg.placement = Placement::from_json(tidevec::cluster::rjson::parse(f));
        } else if (!groups.empty()) {
            std::vector<std::vector<std::string>> sets;
            for (const auto& g : split(groups, ';')) sets.push_back(split(g, ','));
            cfg.placement = Placement::uniform(sets);
        } else {
            usage();
            return 2;
        }
    } catch (const std::exception& e) {
        std::cerr << "invalid placement: " << e.what() << "\n";
        return 2;
    }

    MultiRaftRouter router(cfg);
    g_router = &router;
    std::signal(SIGINT, [](int) { if (g_router) g_router->stop(); });
    std::signal(SIGTERM, [](int) { if (g_router) g_router->stop(); });

    std::cout << "tidevec-router (EXPERIMENTAL) on http://" << cfg.host << ":" << cfg.port
              << " — " << cfg.placement.groups.size() << " shard group(s)\n";
    for (const auto& g : cfg.placement.groups) {
        std::cout << "  " << g.id << " [" << tidevec::cluster::hex64(g.lo) << ", "
                  << tidevec::cluster::hex64(g.hi) << "]";
        for (const auto& r : g.replicas) std::cout << " " << r;
        std::cout << "\n";
    }
    std::cout << std::flush;
    if (!router.listen()) {
        std::cerr << "failed to listen on " << cfg.host << ":" << cfg.port << "\n";
        return 1;
    }
    return 0;
}
