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
#include <cstdint>
#include <iostream>
#include <optional>

#include <arpa/inet.h>
#include <netinet/in.h>

namespace tidevec {
namespace fs = std::filesystem;

// Compares secrets without an early exit on the first differing byte.
inline bool constant_time_equals(const std::string& a, const std::string& b) {
    unsigned char diff = a.size() == b.size() ? 0 : 1;
    const std::size_t n = std::max(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i) {
        unsigned char x = i < a.size() ? static_cast<unsigned char>(a[i]) : 0;
        unsigned char y = i < b.size() ? static_cast<unsigned char>(b[i]) : 0;
        diff |= static_cast<unsigned char>(x ^ y);
    }
    return diff == 0;
}

// Collection and alias names become directory names under the data
// directory, so only [A-Za-z0-9_.-] is allowed, without a leading '.'
// (no "..", no hidden files) and without the names of the server's own
// files in the data directory.
inline bool is_valid_collection_name(const std::string& name) {
    if (name.empty() || name.size() > 128 || name[0] == '.') return false;
    for (unsigned char c : name)
        if (!(std::isalnum(c) || c == '_' || c == '-' || c == '.')) return false;
    static const char* reserved[] = {"raft", "registry.json", "tenants.json"};
    for (const char* r : reserved)
        if (name == r) return false;
    return true;
}

inline void validate_collection_name(const std::string& name, const char* what = "collection") {
    if (!is_valid_collection_name(name))
        throw std::invalid_argument(
            std::string("invalid ") + what + " name '" + name +
            "': use 1-128 characters from [A-Za-z0-9_.-], not starting with '.'");
}

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
        if (buckets_.size() >= kMaxBuckets) _prune_locked(now);
        auto [it, inserted] = buckets_.try_emplace(client_key);
        auto& b = it->second;
        if (inserted) {
            b.tokens = static_cast<double>(cfg_.burst);
        } else {
            auto elapsed = std::chrono::duration<double>(now - b.last_refill).count();
            b.tokens = std::min(static_cast<double>(cfg_.burst),
                b.tokens + elapsed * cfg_.requests_per_second);
        }
        b.last_refill = now;
        if (b.tokens >= 1.0) {
            b.tokens -= 1.0;
            return true;
        }
        return false;
    }

private:
    static constexpr std::size_t kMaxBuckets = 100'000;

    struct Bucket {
        double tokens = 0;
        std::chrono::steady_clock::time_point last_refill{
            std::chrono::steady_clock::now()};
    };

    // Buckets that have refilled completely carry no state worth keeping.
    void _prune_locked(std::chrono::steady_clock::time_point now) {
        const double full_after = cfg_.requests_per_second > 0
            ? static_cast<double>(cfg_.burst) / cfg_.requests_per_second : 0.0;
        std::erase_if(buckets_, [&](const auto& kv) {
            return std::chrono::duration<double>(now - kv.second.last_refill).count() >= full_after;
        });
    }

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
    // allow_private: hosts listed *exactly* in allowed_hosts may resolve to
    // loopback/RFC1918 addresses (e.g. an in-VPC embedding service). Cloud
    // metadata endpoints and non-http(s) schemes stay blocked.
    explicit UrlAllowlist(std::vector<std::string> allowed_hosts = {},
                          bool allow_private = false)
        : allowed_hosts_(std::move(allowed_hosts)),
          allow_private_(allow_private) {
        for (auto& h : allowed_hosts_)
            for (auto& c : h) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    bool is_allowed(const std::string& url) const {
        std::string host = _extract_host(url);
        if (host.empty()) return false;

        if (_is_blocked_url(url)) {
            if (!allow_private_ || _is_always_blocked(url)) return false;
            for (const auto& allowed : allowed_hosts_)
                if (host == allowed) return true;
            return false;
        }

        if (allowed_hosts_.empty()) return true;
        for (const auto& allowed : allowed_hosts_) {
            if (host == allowed || host.ends_with("." + allowed))
                return true;
        }
        return false;
    }

    // Whether the URL's host may resolve to loopback/private addresses:
    // only hosts listed exactly, and only with allow_private.
    bool may_resolve_private(const std::string& url) const {
        if (!allow_private_) return false;
        std::string host = _extract_host(url);
        if (host.empty() || _is_always_blocked(url)) return false;
        for (const auto& allowed : allowed_hosts_)
            if (host == allowed) return true;
        return false;
    }

    // Metadata endpoints, link-local, "this network", multicast/reserved,
    // and anything that does not parse as a plain http(s) host.
    static bool _is_always_blocked(const std::string& url) {
        std::string host = _extract_host(url);
        if (host.empty()) return true;
        if (host == "metadata.google.internal" || host == "metadata") return true;
        if (auto a = _ipv4(host)) {
            return (*a >> 24) == 0 ||                    // 0.0.0.0/8
                   (*a >> 16) == 0xA9FE ||               // 169.254.0.0/16
                   (*a >> 28) >= 0xE;                    // 224.0.0.0/3
        }
        return false;
    }

    // Always-blocked, loopback and private addresses.
    static bool _is_blocked_url(const std::string& url) {
        if (_is_always_blocked(url)) return true;
        std::string host = _extract_host(url);
        if (host == "localhost" || host.ends_with(".localhost")) return true;
        if (auto a = _ipv4(host)) {
            return (*a >> 24) == 10 ||                   // 10.0.0.0/8
                   (*a >> 24) == 127 ||                  // 127.0.0.0/8
                   (*a >> 20) == 0xAC1 ||                // 172.16.0.0/12
                   (*a >> 16) == 0xC0A8 ||               // 192.168.0.0/16
                   (*a & 0xFFC00000u) == 0x64400000u ||  // 100.64.0.0/10
                   (*a & 0xFFFE0000u) == 0xC6120000u;    // 198.18.0.0/15
        }
        return false;
    }

private:
    // The host the DriftBridge client will connect to: the authority runs
    // to the first '/', the port follows the first ':' (as in
    // parse_http_url). Lower-cased, trailing dot removed. "" when the
    // scheme is not http(s) or the authority holds anything but a host
    // name / IPv4 literal and a numeric port (userinfo, IPv6 brackets,
    // percent-encoding, '?', '#', '\').
    static std::string _extract_host(const std::string& url) {
        auto scheme_end = url.find("://");
        if (scheme_end == std::string::npos) return {};
        auto scheme = url.substr(0, scheme_end);
        if (scheme != "http" && scheme != "https") return {};
        auto start = scheme_end + 3;
        auto end = url.find('/', start);
        std::string authority = url.substr(start, end == std::string::npos
                                                      ? std::string::npos : end - start);
        auto colon = authority.find(':');
        std::string host = authority.substr(0, colon);
        if (colon != std::string::npos) {
            auto port = authority.substr(colon + 1);
            if (port.empty() || port.size() > 5) return {};
            for (unsigned char c : port) if (!std::isdigit(c)) return {};
        }
        for (auto& c : host) {
            auto u = static_cast<unsigned char>(c);
            if (!(std::isalnum(u) || c == '-' || c == '.')) return {};
            c = static_cast<char>(std::tolower(u));
        }
        if (!host.empty() && host.back() == '.') host.pop_back();
        return host;
    }

    // Parses IPv4 literals the way the resolver does, including shorthand
    // such as "127.1", "0x7f.0.0.1" or "2130706433".
    static std::optional<uint32_t> _ipv4(const std::string& host) {
        in_addr a{};
        if (host.empty() || inet_aton(host.c_str(), &a) == 0) return std::nullopt;
        return ntohl(a.s_addr);
    }

    std::vector<std::string> allowed_hosts_;
    bool allow_private_ = false;
};

// ================================================================
// Multi-tenant API key store
//
// The configured --api-key is the admin key. It is held by RestServer and
// never written here, so restarting with a different key revokes the old
// one. tenants.json lists the additional tenant keys accepted in
// --multi-tenant mode. The id "default" is reserved for the admin key:
// servers before 0.3.0 persisted the first --api-key they saw under that
// id, and such entries are removed from the file on load.
//
// A tenant may only touch collections (and aliases) whose names start with
// one of its collection_prefixes; with no prefixes configured that is
// "<id>_". [""] grants every collection.
// ================================================================
struct Tenant {
    std::string id;
    std::string api_key;
    std::string name;
    std::size_t max_collections = 100;
    std::size_t max_vectors     = 10'000'000;
    std::vector<std::string> collection_prefixes;  // empty = {"<id>_"}
    bool enabled = true;
};

class TenantStore {
public:
    static constexpr const char* kAdminId = "default";

    explicit TenantStore(std::string data_dir)
        : data_dir_(std::move(data_dir)) {
        fs::create_directories(data_dir_);
        _load();
    }

    // Validate API key; returns tenant id or empty string.
    std::string authenticate(const std::string& api_key) const {
        if (api_key.empty()) return {};
        std::shared_lock lock(mu_);
        std::string found;
        for (const auto& [id, t] : tenants_)
            if (constant_time_equals(t.api_key, api_key) && t.enabled && found.empty())
                found = id;
        return found;
    }

    // Tenants are only loaded at construction, so the pointer stays valid.
    const Tenant* get_tenant(const std::string& tenant_id) const {
        std::shared_lock lock(mu_);
        auto it = tenants_.find(tenant_id);
        return it != tenants_.end() ? &it->second : nullptr;
    }

    bool can_access_collection(const std::string& tenant_id,
                               const std::string& collection) const {
        auto* t = get_tenant(tenant_id);
        if (!t || !t->enabled) return false;
        for (const auto& prefix : prefixes_of(*t))
            if (collection.starts_with(prefix)) return true;
        return false;
    }

    // owned_collections / owned_vectors: the tenant's own usage.
    bool check_collection_quota(const std::string& tenant_id,
                                std::size_t owned_collections) const {
        auto* t = get_tenant(tenant_id);
        return t && owned_collections < t->max_collections;
    }

    bool check_vector_quota(const std::string& tenant_id,
                            std::size_t owned_vectors,
                            std::size_t adding) const {
        auto* t = get_tenant(tenant_id);
        return t && owned_vectors + adding <= t->max_vectors;
    }

    std::size_t size() const {
        std::shared_lock lock(mu_);
        return tenants_.size();
    }

    static std::vector<std::string> prefixes_of(const Tenant& t) {
        if (t.collection_prefixes.empty()) return {t.id + "_"};
        return t.collection_prefixes;
    }

private:
    std::string _path() const { return data_dir_ + "/tenants.json"; }

    void _load() {
        auto path = _path();
        if (!fs::exists(path)) return;
        nlohmann::json j;
        try {
            std::ifstream f(path);
            j = nlohmann::json::parse(f);
            if (!j.is_array()) throw std::runtime_error("expected a JSON array");
        } catch (const std::exception& e) {
            std::cerr << "[tenants] Ignoring " << path << ": " << e.what() << "\n";
            return;
        }

        nlohmann::json kept = nlohmann::json::array();
        bool dropped_default = false;
        std::unique_lock lock(mu_);
        for (const auto& item : j) {
            try {
                Tenant t;
                t.id              = item.at("id").get<std::string>();
                if (t.id == kAdminId) {
                    dropped_default = true;
                    continue;
                }
                kept.push_back(item);
                t.api_key         = item.at("api_key").get<std::string>();
                t.name            = item.value("name", t.id);
                t.max_collections = item.value("max_collections", std::size_t{100});
                t.max_vectors     = item.value("max_vectors", std::size_t{10'000'000});
                t.enabled         = item.value("enabled", true);
                if (item.contains("collection_prefixes"))
                    t.collection_prefixes =
                        item["collection_prefixes"].get<std::vector<std::string>>();
                if (t.api_key.empty())
                    throw std::invalid_argument("empty api_key");
                bool dup = false;
                for (const auto& [_, other] : tenants_)
                    if (constant_time_equals(other.api_key, t.api_key)) dup = true;
                if (dup) throw std::invalid_argument("api_key already used by another tenant");
                if (tenants_.count(t.id)) throw std::invalid_argument("duplicate id");
                tenants_[t.id] = std::move(t);
            } catch (const std::exception& e) {
                std::cerr << "[tenants] Skipping entry in " << path << ": " << e.what() << "\n";
            }
        }

        if (dropped_default) {
            std::cerr << "[tenants] Removed the 'default' entry from " << path
                      << ": the admin key is --api-key / TIDEVEC_API_KEY\n";
            _write_file(kept);
        }
    }

    void _write_file(const nlohmann::json& j) const {
        auto tmp = _path() + ".tmp";
        {
            std::ofstream f(tmp, std::ios::trunc);
            f << j.dump(2) << "\n";
        }
        std::error_code ec;
        fs::permissions(tmp, fs::perms::owner_read | fs::perms::owner_write, ec);
        fs::rename(tmp, _path(), ec);
        if (ec) std::cerr << "[tenants] Could not rewrite " << _path() << ": " << ec.message() << "\n";
    }

    std::string data_dir_;
    mutable std::shared_mutex mu_;
    std::unordered_map<std::string, Tenant> tenants_;
};

} // namespace tidevec
