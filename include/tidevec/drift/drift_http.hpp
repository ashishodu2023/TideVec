#pragma once
// HTTP helper for DriftBridge re-embedding callbacks.
//
// The reembed_url host is resolved once (pin_http_url) and every callback
// connects to that pinned address, with the original host name in the Host
// header and TLS SNI/certificate checks. A host name that resolves to a
// loopback, private or link-local address is rejected unless the caller
// allows private addresses, so DNS cannot point the server at internal
// services, and re-resolving later (DNS rebinding) has no effect.

#include <tidevec/api/httplib.h>

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>

namespace tidevec {

struct ParsedHttpUrl {
    std::string scheme;   // "http" or "https"
    std::string host;
    int         port     = 0;
    std::string path     = "/";
    std::string pinned_ip;  // numeric address chosen by pin_http_url()
};

// Parse http(s)://host[:port]/path — used by DriftBridge reembed_url.
inline ParsedHttpUrl parse_http_url(const std::string& url) {
    ParsedHttpUrl out;
    std::string rest = url;

    auto scheme_end = rest.find("://");
    if (scheme_end == std::string::npos)
        throw std::runtime_error("reembed_url must start with http:// or https://");

    out.scheme = rest.substr(0, scheme_end);
    if (out.scheme != "http" && out.scheme != "https")
        throw std::runtime_error("reembed_url scheme must be http or https");

    rest = rest.substr(scheme_end + 3);
    auto path_start = rest.find('/');
    std::string authority = (path_start == std::string::npos)
        ? rest : rest.substr(0, path_start);
    if (path_start != std::string::npos)
        out.path = rest.substr(path_start);
    if (out.path.empty()) out.path = "/";

    auto colon = authority.find(':');
    if (colon == std::string::npos) {
        out.host = authority;
        out.port = (out.scheme == "https") ? 443 : 80;
    } else {
        out.host = authority.substr(0, colon);
        out.port = std::stoi(authority.substr(colon + 1));
    }

    if (out.host.empty())
        throw std::runtime_error("reembed_url missing host");
    if (out.port <= 0 || out.port > 65535)
        throw std::runtime_error("reembed_url port out of range");
    return out;
}

// ----------------------------------------------------------------
// Address classification
// ----------------------------------------------------------------
enum class AddressClass {
    Public,
    Private,    // loopback, RFC 1918, CGNAT, benchmarking, IPv6 ULA
    Forbidden,  // unspecified, link-local (cloud metadata), multicast, reserved
};

inline AddressClass classify_ipv4(uint32_t a) {
    if ((a >> 24) == 0 ||                       // 0.0.0.0/8
        (a >> 16) == 0xA9FE ||                  // 169.254.0.0/16 (metadata)
        a == 0x646464C8u ||                     // 100.100.100.200 (Alibaba metadata)
        (a >> 28) >= 0xE)                       // 224.0.0.0/3
        return AddressClass::Forbidden;
    if ((a >> 24) == 10 ||                      // 10.0.0.0/8
        (a >> 24) == 127 ||                     // 127.0.0.0/8
        (a >> 20) == 0xAC1 ||                   // 172.16.0.0/12
        (a >> 16) == 0xC0A8 ||                  // 192.168.0.0/16
        (a & 0xFFC00000u) == 0x64400000u ||     // 100.64.0.0/10
        (a & 0xFFFE0000u) == 0xC6120000u ||     // 198.18.0.0/15
        (a & 0xFFFFFF00u) == 0xC0000000u)       // 192.0.0.0/24
        return AddressClass::Private;
    return AddressClass::Public;
}

inline AddressClass classify_ipv6(const uint8_t b[16]) {
    auto v4_at = [&](int off) {
        return (uint32_t(b[off]) << 24) | (uint32_t(b[off + 1]) << 16) |
               (uint32_t(b[off + 2]) << 8) | uint32_t(b[off + 3]);
    };
    static const uint8_t aws_metadata[16] =
        {0xfd, 0x00, 0x0e, 0xc2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0x54};
    if (std::memcmp(b, aws_metadata, 16) == 0) return AddressClass::Forbidden;

    bool first10_zero = true;
    for (int i = 0; i < 10; ++i) if (b[i]) first10_zero = false;
    if (first10_zero && b[10] == 0xff && b[11] == 0xff)       // ::ffff:a.b.c.d
        return classify_ipv4(v4_at(12));
    if (first10_zero && b[10] == 0 && b[11] == 0) {
        bool loopback = b[15] == 1;
        for (int i = 12; i < 15; ++i) if (b[i]) loopback = false;
        // ::1 is loopback; :: and the deprecated ::a.b.c.d are not routable.
        return loopback ? AddressClass::Private : AddressClass::Forbidden;
    }
    if (b[0] == 0x00 && b[1] == 0x64 && b[2] == 0xff && b[3] == 0x9b) {   // 64:ff9b::/96 NAT64
        bool well_known = true;
        for (int i = 4; i < 12; ++i) if (b[i]) well_known = false;
        if (well_known) return classify_ipv4(v4_at(12));
    }
    if (b[0] == 0x20 && b[1] == 0x02)                                     // 2002::/16 6to4
        return classify_ipv4(v4_at(2));
    if (b[0] == 0x20 && b[1] == 0x01 && b[2] == 0 && b[3] == 0)           // 2001::/32 Teredo
        return AddressClass::Forbidden;
    if (b[0] == 0xff) return AddressClass::Forbidden;                     // multicast
    if (b[0] == 0xfe && (b[1] & 0xc0) == 0x80) return AddressClass::Forbidden;  // fe80::/10
    if (b[0] == 0xfe && (b[1] & 0xc0) == 0xc0) return AddressClass::Private;    // fec0::/10
    if ((b[0] & 0xfe) == 0xfc) return AddressClass::Private;                    // fc00::/7
    return AddressClass::Public;
}

inline AddressClass classify_address(const sockaddr* sa) {
    if (sa->sa_family == AF_INET) {
        const auto* in = reinterpret_cast<const sockaddr_in*>(sa);
        return classify_ipv4(ntohl(in->sin_addr.s_addr));
    }
    if (sa->sa_family == AF_INET6) {
        const auto* in6 = reinterpret_cast<const sockaddr_in6*>(sa);
        return classify_ipv6(in6->sin6_addr.s6_addr);
    }
    return AddressClass::Forbidden;
}

// The reembed_url host resolved to an address the server must not call.
class ReembedUrlBlocked : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Resolves url.host once and pins the first address in url.pinned_ip.
// Throws ReembedUrlBlocked if any resolved address is forbidden, or private
// while allow_private is false; std::runtime_error if it does not resolve.
inline void pin_http_url(ParsedHttpUrl& url, bool allow_private) {
    addrinfo hints{};
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    int rc = ::getaddrinfo(url.host.c_str(), nullptr, &hints, &res);
    if (rc != 0 || !res)
        throw std::runtime_error("cannot resolve reembed_url host '" + url.host +
                                 "': " + ::gai_strerror(rc));
    std::unique_ptr<addrinfo, decltype(&::freeaddrinfo)> guard(res, ::freeaddrinfo);

    std::string first;
    for (auto* ai = res; ai; ai = ai->ai_next) {
        if (ai->ai_family != AF_INET && ai->ai_family != AF_INET6) continue;
        char buf[INET6_ADDRSTRLEN] = {};
        const void* src = ai->ai_family == AF_INET6
            ? static_cast<const void*>(&reinterpret_cast<sockaddr_in6*>(ai->ai_addr)->sin6_addr)
            : static_cast<const void*>(&reinterpret_cast<sockaddr_in*>(ai->ai_addr)->sin_addr);
        ::inet_ntop(ai->ai_family, src, buf, sizeof(buf));

        auto cls = classify_address(ai->ai_addr);
        if (cls == AddressClass::Forbidden ||
            (cls == AddressClass::Private && !allow_private))
            throw ReembedUrlBlocked("reembed_url host '" + url.host +
                                    "' resolves to a non-public address (" + buf + ")");
        if (first.empty()) first = buf;
    }
    if (first.empty())
        throw std::runtime_error("reembed_url host '" + url.host + "' has no IP address");
    url.pinned_ip = first;
}

// POST JSON body to a parsed URL; returns response body on HTTP 200.
// Connects only to url.pinned_ip; an unpinned URL is pinned here with
// private addresses refused.
inline bool http_post_json(const ParsedHttpUrl& url_in,
                           const std::string& body,
                           std::string& out_body,
                           int timeout_sec = 60) {
    ParsedHttpUrl url = url_in;
    if (url.pinned_ip.empty()) {
        try { pin_http_url(url, /*allow_private=*/false); }
        catch (const std::exception&) { return false; }
    }
    std::map<std::string, std::string> pin{{url.host, url.pinned_ip}};

#if defined(CPPHTTPLIB_OPENSSL_SUPPORT)
    if (url.scheme == "https") {
        httplib::SSLClient cli(url.host, url.port);
        cli.set_hostname_addr_map(pin);
        cli.set_connection_timeout(timeout_sec, 0);
        cli.set_read_timeout(timeout_sec, 0);
        auto res = cli.Post(url.path.c_str(), body, "application/json");
        if (!res || res->status != 200) return false;
        out_body = res->body;
        return true;
    }
#else
    if (url.scheme == "https")
        return false;  // rebuild with -DTIDEVEC_TLS for https reembed_url
#endif

    httplib::Client cli(url.host, url.port);
    cli.set_hostname_addr_map(pin);
    cli.set_connection_timeout(timeout_sec, 0);
    cli.set_read_timeout(timeout_sec, 0);
    auto res = cli.Post(url.path.c_str(), body, "application/json");
    if (!res || res->status != 200) return false;
    out_body = res->body;
    return true;
}

} // namespace tidevec
