#pragma once
// HTTP helper for DriftBridge re-embedding callbacks.

#include <tidevec/api/httplib.h>

#include <string>
#include <stdexcept>

namespace tidevec {

struct ParsedHttpUrl {
    std::string scheme;   // "http" or "https"
    std::string host;
    int         port     = 0;
    std::string path     = "/";
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
    return out;
}

// POST JSON body to a parsed URL; returns response body on HTTP 200.
inline bool http_post_json(const ParsedHttpUrl& url,
                           const std::string& body,
                           std::string& out_body,
                           int timeout_sec = 60) {
#if defined(CPPHTTPLIB_OPENSSL_SUPPORT)
    if (url.scheme == "https") {
        httplib::SSLClient cli(url.host, url.port);
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
    cli.set_connection_timeout(timeout_sec, 0);
    cli.set_read_timeout(timeout_sec, 0);
    auto res = cli.Post(url.path.c_str(), body, "application/json");
    if (!res || res->status != 200) return false;
    out_body = res->body;
    return true;
}

} // namespace tidevec
