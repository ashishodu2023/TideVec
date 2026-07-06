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

int main(int argc, char* argv[]) {
    tidevec::RestServer::Config cfg;

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
            else if (arg == "--backup-s3")         cfg.backup_s3_uri = next();
            else if (arg == "--backup-gcs")        cfg.backup_gcs_uri = next();
            else if (arg == "--otel-endpoint")     cfg.otel_endpoint = next();
            else if (arg == "--reembed-allow")     cfg.reembed_allowed_hosts = split_csv(next());
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
            else {
                std::cerr << "Unknown argument: " << arg << "\n";
                return 1;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    // Warn if auth required but no key configured
    if (cfg.require_auth && cfg.api_key.empty()) {
        std::cerr << "Warning: --require-auth set but no API key configured.\n"
                  << "Set TIDEVEC_API_KEY or pass --api-key.\n";
    }

    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);

    tidevec::RestServer server(cfg);
    g_server.store(&server);
    server.listen();
    return 0;
}
