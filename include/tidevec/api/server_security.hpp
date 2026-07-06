#pragma once
// ================================================================
// Server security helpers — rate limiting, SSRF allowlist, tenants
// ================================================================

#include <tidevec/api/json.hpp>

#include <string>
#include <shared_mutex>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <cctype>

namespace tidevec {
namespace fs = std::filesystem;

// ================================================================
// Token-bucket rate limiter (per client key)
// ================================================================
class RateLimiter {
public:
    struct Config {
        int requests_per_second;
        int burst;
        Config(int rps = 100, int b = 200)
            : requests_per_second(rps), burst(b) {}
    };

    explicit RateLimiter(Config cfg) : cfg_(std::move(cfg)) {}
    RateLimiter() : RateLimiter(Config{}) {}

    // Returns true if request is allowed, false if rate-limited.
    bool allow(const std::string& client_key) {
        auto now = std::chrono::steady_clock::now();
        std::lock_guard lock(mu_);
        auto [it, inserted] = buckets_.try_emplace(client_key);
        auto& b = it->second;
        if (inserted)
            b.tokens = static_cast<double>(cfg_.burst);
        if (b.tokens < 1.0) {
            auto elapsed = std::chrono::duration<double>(now - b.last_refill).count();
            b.tokens = std::min(static_cast<double>(cfg_.burst),
                b.tokens + elapsed * cfg_.requests_per_second);
            b.last_refill = now;
        }
        if (b.tokens >= 1.0) {
            b.tokens -= 1.0;
            return true;
        }
        return false;
    }

private:
    struct Bucket {
        double tokens = 0;
        std::chrono::steady_clock::time_point last_refill{
            std::chrono::steady_clock::now()};
    };

    Config cfg_;
    std::mutex mu_;
    std::unordered_map<std::string, Bucket> buckets_;
};

// ================================================================
// SSRF allowlist for DriftBridge reembed_url
// ================================================================
class UrlAllowlist {
public:
    // Empty allowlist = allow any public http/https URL (legacy mode).
    // Non-empty = only matching host prefixes are permitted.
    explicit UrlAllowlist(std::vector<std::string> allowed_hosts = {})
        : allowed_hosts_(std::move(allowed_hosts)) {}

    bool is_allowed(const std::string& url) const {
        if (allowed_hosts_.empty()) return true;

        // Block private/link-local/metadata IPs regardless of allowlist
        if (_is_blocked_url(url)) return false;

        std::string host = _extract_host(url);
        if (host.empty()) return false;

        for (const auto& allowed : allowed_hosts_) {
            if (host == allowed || host.ends_with("." + allowed))
                return true;
        }
        return false;
    }

    static bool _is_blocked_url(const std::string& url) {
        std::string host = _extract_host(url);
        if (host.empty()) return true;

        // Block non-http(s) schemes
        if (url.find("http://") != 0 && url.find("https://") != 0)
            return true;

        // Block localhost variants
        if (host == "localhost" || host == "127.0.0.1" || host == "::1" ||
            host == "0.0.0.0" || host.starts_with("127."))
            return true;

        // Block cloud metadata endpoints
        if (host == "169.254.169.254" || host == "metadata.google.internal")
            return true;

        // Block private RFC1918 ranges (simple prefix check)
        if (host.starts_with("10.") || host.starts_with("192.168.") ||
            host.starts_with("172.16.") || host.starts_with("172.17.") ||
            host.starts_with("172.18.") || host.starts_with("172.19.") ||
            host.starts_with("172.2") || host.starts_with("172.30.") ||
            host.starts_with("172.31."))
            return true;

        return false;
    }

private:
    static std::string _extract_host(const std::string& url) {
        auto scheme_end = url.find("://");
        if (scheme_end == std::string::npos) return {};
        auto host_start = scheme_end + 3;
        auto host_end = url.find('/', host_start);
        if (host_end == std::string::npos) host_end = url.size();
        std::string host_port = url.substr(host_start, host_end - host_start);
        // Strip port
        auto colon = host_port.rfind(':');
        if (colon != std::string::npos && host_port.find(']') == std::string::npos)
            return host_port.substr(0, colon);
        return host_port;
    }

    std::vector<std::string> allowed_hosts_;
};

// ================================================================
// Multi-tenant API key store
// ================================================================
struct Tenant {
    std::string id;
    std::string api_key;
    std::string name;
    std::size_t max_collections = 100;
    std::size_t max_vectors     = 10'000'000;
    std::vector<std::string> collection_prefixes;  // empty = all
    bool enabled = true;
};

class TenantStore {
public:
    explicit TenantStore(std::string data_dir)
        : data_dir_(std::move(data_dir)) {
        fs::create_directories(data_dir_);
        _load();
    }

    // Validate API key; returns tenant id or empty string.
    std::string authenticate(const std::string& api_key) const {
        if (api_key.empty()) return {};
        std::shared_lock lock(mu_);
        for (const auto& [id, t] : tenants_) {
            if (t.enabled && t.api_key == api_key) return id;
        }
        return {};
    }

    const Tenant* get_tenant(const std::string& tenant_id) const {
        std::shared_lock lock(mu_);
        auto it = tenants_.find(tenant_id);
        return it != tenants_.end() ? &it->second : nullptr;
    }

    bool can_access_collection(const std::string& tenant_id,
                               const std::string& collection) const {
        std::shared_lock lock(mu_);
        auto it = tenants_.find(tenant_id);
        if (it == tenants_.end() || !it->second.enabled) return false;
        if (it->second.collection_prefixes.empty()) return true;
        for (const auto& prefix : it->second.collection_prefixes) {
            if (collection.starts_with(prefix)) return true;
        }
        return false;
    }

    bool check_collection_quota(const std::string& tenant_id,
                                std::size_t current_count) const {
        auto* t = get_tenant(tenant_id);
        return t && current_count < t->max_collections;
    }

    bool check_vector_quota(const std::string& tenant_id,
                            std::size_t current_vectors,
                            std::size_t adding) const {
        auto* t = get_tenant(tenant_id);
        return t && (current_vectors + adding) <= t->max_vectors;
    }

    void record_usage(const std::string& tenant_id,
                      std::size_t collections,
                      std::size_t vectors) {
        std::unique_lock lock(mu_);
        usage_[tenant_id] = {collections, vectors};
    }

    // Bootstrap a default tenant from a single API key (Docker / single-tenant mode)
    void ensure_default_tenant(const std::string& api_key) {
        if (api_key.empty()) return;
        std::unique_lock lock(mu_);
        if (!tenants_.empty()) return;
        Tenant t;
        t.id      = "default";
        t.api_key = api_key;
        t.name    = "Default Tenant";
        tenants_[t.id] = t;
        _save_locked();
    }

private:
    void _load() {
        auto path = data_dir_ + "/tenants.json";
        if (!fs::exists(path)) return;
        try {
            std::ifstream f(path);
            auto j = nlohmann::json::parse(f);
            std::unique_lock lock(mu_);
            for (const auto& item : j) {
                Tenant t;
                t.id              = item.at("id").get<std::string>();
                t.api_key         = item.at("api_key").get<std::string>();
                t.name            = item.value("name", t.id);
                t.max_collections = item.value("max_collections", 100UL);
                t.max_vectors     = item.value("max_vectors", 10'000'000UL);
                t.enabled         = item.value("enabled", true);
                if (item.contains("collection_prefixes"))
                    t.collection_prefixes =
                        item["collection_prefixes"].get<std::vector<std::string>>();
                tenants_[t.id] = t;
            }
        } catch (...) {}
    }

    void _save_locked() {
        nlohmann::json j = nlohmann::json::array();
        for (const auto& [id, t] : tenants_) {
            j.push_back({
                {"id", id},
                {"api_key", t.api_key},
                {"name", t.name},
                {"max_collections", t.max_collections},
                {"max_vectors", t.max_vectors},
                {"enabled", t.enabled},
                {"collection_prefixes", t.collection_prefixes},
            });
        }
        std::ofstream f(data_dir_ + "/tenants.json");
        f << j.dump(2);
    }

    std::string data_dir_;
    mutable std::shared_mutex mu_;
    std::unordered_map<std::string, Tenant> tenants_;
    std::unordered_map<std::string, std::pair<std::size_t, std::size_t>> usage_;
};

} // namespace tidevec
