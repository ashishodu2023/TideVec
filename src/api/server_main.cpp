// ================================================================
// tidevec-server — main entry point
//
// Usage:
//   ./tidevec-server [--host 0.0.0.0] [--port 6399]
//                     [--threads 8] [--data-dir ./data]
//                     [--api-key mykey] [--require-auth]
//                     [--device auto|cpu|gpu|tpu]
//                     [--ultra-durable] [--no-segment-store]
//                     [--tls-cert cert.pem] [--tls-key key.pem]
//                     [--rate-limit 100] [--quiet]
//                     [--backup [--backup-interval-hours 6]]
//                     [--reembed-allow host1,host2] [--reembed-allow-private]
//
// EXPERIMENTAL cluster mode (Raft over HTTP):
//   ./tidevec-server --port 7001 --data-dir ./n0 --node-id 0
//       --raft-peers 0=http://h0:7001,1=http://h1:7001,2=http://h2:7001
//       --cluster-secret s3cret        (or TIDEVEC_CLUSTER_SECRET)
//       [--raft-snapshot-threshold 4096] [--raft-election-timeout-ms 400]
//       [--raft-heartbeat-ms 100] [--raft-verbose]
//   Adding a node: start it with --raft-join (its own entry in
//   --raft-peers is enough), then POST /v1/admin/cluster/add_node
//   {"id":3,"url":"http://h3:7001"} to the leader.
//   --enable-fault-injection exposes POST /v1/admin/fault (tests only).
// ================================================================

#include <tidevec/api/rest_server.hpp>
#include <iostream>
#include <string>
#include <csignal>
#include <atomic>
#include <cstdlib>
#include <vector>

static std::atomic<tidevec::RestServer*> g_server{nullptr};

static void signal_handler(int) {
    std::cout << "\nShutting down TideVec...\n";
    if (auto* s = g_server.load()) s->stop();
}

static std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    std::size_t start = 0;
    while (start < s.size()) {
        auto pos = s.find(',', start);
        auto part = s.substr(start, pos == std::string::npos ? s.size() - start : pos - start);
        if (!part.empty()) out.push_back(part);
        if (pos == std::string::npos) break;
        start = pos + 1;
    }
    return out;
}

// "0=http://a:6399,1=http://b:6399,2=http://c:6399" -> URLs indexed by node id.
// Ids may have gaps (e.g. after node 1 was removed and node 3 added).
static std::vector<std::string> parse_peers(const std::string& s) {
    std::vector<std::string> urls;
    for (const auto& part : split_csv(s)) {
        auto eq = part.find('=');
        if (eq == std::string::npos)
            throw std::runtime_error("--raft-peers entries must look like id=url: " + part);
        int id = std::stoi(part.substr(0, eq));
        if (id < 0) throw std::runtime_error("--raft-peers: negative node id");
        if (static_cast<std::size_t>(id) >= urls.size()) urls.resize(id + 1);
        urls[id] = part.substr(eq + 1);
    }
    return urls;
}

int main(int argc, char* argv[]) {
    tidevec::RestServer::Config cfg;

    if (const char* s = std::getenv("TIDEVEC_CLUSTER_SECRET"); s && s[0])
        cfg.cluster_secret = s;

    // Auth on by default: generate key from env or random if not provided
    const char* env_key = std::getenv("TIDEVEC_API_KEY");
    if (env_key && env_key[0]) cfg.api_key = env_key;

    const char* env_auth = std::getenv("TIDEVEC_REQUIRE_AUTH");
    cfg.require_auth = (env_auth == nullptr) ? true : (std::string(env_auth) != "0");

    auto parse_int = [](const std::string& flag, const std::string& val) -> int {
        try { return std::stoi(val); }
        catch (const std::exception&) {
            std::cerr << "Error: " << flag << " expects an integer, got: \""
                      << val << "\"\n";
            std::exit(1);
        }
    };

    try {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 < argc) return argv[++i];
                throw std::runtime_error("Missing value for " + arg);
            };
            if      (arg == "--host")              cfg.host     = next();
            else if (arg == "--port")              cfg.port     = parse_int(arg, next());
            else if (arg == "--threads")           cfg.threads  = parse_int(arg, next());
            else if (arg == "--data-dir")          cfg.data_dir = next();
            else if (arg == "--api-key")           cfg.api_key  = next();
            else if (arg == "--device")            cfg.device   = next();
            else if (arg == "--tls-cert")          cfg.tls_cert = next();
            else if (arg == "--tls-key")           cfg.tls_key  = next();
            else if (arg == "--rate-limit")        cfg.rate_limit_rps = parse_int(arg, next());
            else if (arg == "--backup-dir")        cfg.backup_dir = next();
            else if (arg == "--backup-interval-hours") {
                cfg.backup_interval_hours = parse_int(arg, next());
                if (cfg.backup_interval_hours < 1)
                    throw std::runtime_error("--backup-interval-hours must be >= 1");
            }
            else if (arg == "--backup-s3")         cfg.backup_s3_uri = next();
            else if (arg == "--backup-gcs")        cfg.backup_gcs_uri = next();
            else if (arg == "--otel-endpoint")     cfg.otel_endpoint = next();
            else if (arg == "--reembed-allow")     cfg.reembed_allowed_hosts = split_csv(next());
            else if (arg == "--reembed-allow-private") cfg.reembed_allow_private = true;
            else if (arg == "--require-auth")      cfg.require_auth = true;
            else if (arg == "--no-require-auth")   cfg.require_auth = false;
            else if (arg == "--ultra-durable")     cfg.ultra_durable = true;
            else if (arg == "--no-ultra-durable")  cfg.ultra_durable = false;
            else if (arg == "--segment-store")     cfg.use_segment_store = true;
            else if (arg == "--no-segment-store")  cfg.use_segment_store = false;
            else if (arg == "--backup")            cfg.backup_enabled = true;
            else if (arg == "--otel")              cfg.otel_enabled = true;
            else if (arg == "--multi-tenant")      cfg.multi_tenant = true;
            else if (arg == "--quiet")             cfg.log_requests = false;
            else if (arg == "--node-id")           cfg.raft_node_id = parse_int(arg, next());
            else if (arg == "--raft-peers")        cfg.raft_peers = parse_peers(next());
            else if (arg == "--cluster-secret")    cfg.cluster_secret = next();
            else if (arg == "--raft-verbose")      cfg.raft_verbose = true;
            else if (arg == "--raft-join")         cfg.raft_join = true;
            else if (arg == "--raft-snapshot-threshold")
                cfg.raft_snapshot_threshold = static_cast<uint64_t>(parse_int(arg, next()));
            else if (arg == "--raft-election-timeout-ms")
                cfg.raft_election_timeout_ms = parse_int(arg, next());
            else if (arg == "--raft-heartbeat-ms") cfg.raft_heartbeat_ms = parse_int(arg, next());
            else if (arg == "--enable-fault-injection") cfg.enable_fault_injection = true;
            else {
                std::cerr << "Unknown argument: " << arg << "\n";
                return 1;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    if (cfg.raft_node_id >= 0) {
        if (cfg.raft_peers.size() < 1 ||
            cfg.raft_node_id >= static_cast<int>(cfg.raft_peers.size()) ||
            cfg.raft_peers[cfg.raft_node_id].empty()) {
            std::cerr << "Error: --node-id requires --raft-peers listing every node "
                         "(including this one)\n";
            return 1;
        }
        if (cfg.cluster_secret.empty())
            std::cerr << "Warning: no --cluster-secret / TIDEVEC_CLUSTER_SECRET; "
                         "Raft RPC endpoints accept any caller.\n";
    }

    // Warn if auth required but no key configured
    if (cfg.require_auth && cfg.api_key.empty()) {
        std::cerr << "Warning: --require-auth set but no API key configured.\n"
                  << "Set TIDEVEC_API_KEY or pass --api-key.\n";
    }

    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);

    try {
        tidevec::RestServer server(cfg);
        g_server.store(&server);
        server.listen();
        g_server.store(nullptr);
    } catch (const std::exception& e) {
        g_server.store(nullptr);
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
