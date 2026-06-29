// ================================================================
// tidevec-server — main entry point
//
// Usage:
//   ./tidevec-server [--host 0.0.0.0] [--port 6399]
//                     [--threads 8] [--data-dir ./data]
//                     [--api-key mykey] [--quiet]
//
// Quick start:
//   ./tidevec-server
//   curl http://localhost:6399/health
// ================================================================

#include <tidevec/api/rest_server.hpp>
#include <iostream>
#include <string>
#include <csignal>
#include <atomic>

static std::atomic<tidevec::RestServer*> g_server{nullptr};

static void signal_handler(int) {
    std::cout << "\nShutting down TideVec...\n";
    if (auto* s = g_server.load()) s->stop();
}

int main(int argc, char* argv[]) {
    tidevec::RestServer::Config cfg;

    // Parse CLI args
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 < argc) return argv[++i];
            throw std::runtime_error("Missing value for " + arg);
        };
        if      (arg == "--host")     cfg.host     = next();
        else if (arg == "--port")     cfg.port     = std::stoi(next());
        else if (arg == "--threads")  cfg.threads  = std::stoi(next());
        else if (arg == "--data-dir") cfg.data_dir = next();
        else if (arg == "--api-key")  cfg.api_key  = next();
        else if (arg == "--quiet")    cfg.log_requests = false;
        else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return 1;
        }
    }

    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);

    tidevec::RestServer server(cfg);
    g_server.store(&server);
    server.listen();   // blocks until stop()
    return 0;
}
