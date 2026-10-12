#pragma once
// ================================================================
// consensus/raft_http.hpp — HTTP transport for multi-process Raft
//
// EXPERIMENTAL. Raft RPCs are JSON over HTTP/1.1, posted to
//   POST {peer}/raft/append_entries
//   POST {peer}/raft/request_vote      (also pre-votes: "pre_vote": true)
//   POST {peer}/raft/install_snapshot  (chunked, see RaftConfig::snapshot_chunk_bytes)
// and authenticated with a shared secret in X-TideVec-Cluster-Secret.
// Log payloads are base64-encoded. Use TLS or a private network in
// any real deployment: the secret is sent in clear text over http://.
// ================================================================

#include <tidevec/consensus/raft.hpp>
#include <tidevec/api/httplib.h>
#include <tidevec/api/json.hpp>

#include <mutex>
#include <string>

namespace tidevec {
namespace consensus {

using rjson = nlohmann::json;

inline constexpr const char* kClusterSecretHeader = "X-TideVec-Cluster-Secret";

inline std::string base64_encode(const std::vector<uint8_t>& in) {
    static const char* tbl =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((in.size() + 2) / 3 * 4);
    std::size_t i = 0;
    for (; i + 2 < in.size(); i += 3) {
        uint32_t v = (in[i] << 16) | (in[i + 1] << 8) | in[i + 2];
        out += tbl[(v >> 18) & 63]; out += tbl[(v >> 12) & 63];
        out += tbl[(v >> 6) & 63];  out += tbl[v & 63];
    }
    if (i < in.size()) {
        uint32_t v = in[i] << 16;
        if (i + 1 < in.size()) v |= in[i + 1] << 8;
        out += tbl[(v >> 18) & 63]; out += tbl[(v >> 12) & 63];
        out += (i + 1 < in.size()) ? tbl[(v >> 6) & 63] : '=';
        out += '=';
    }
    return out;
}

inline std::vector<uint8_t> base64_decode(const std::string& in) {
    auto val = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    std::vector<uint8_t> out;
    out.reserve(in.size() / 4 * 3);
    uint32_t buf = 0;
    int bits = 0;
    for (char c : in) {
        if (c == '=') break;
        int v = val(c);
        if (v < 0) throw std::invalid_argument("invalid base64");
        buf = (buf << 6) | static_cast<uint32_t>(v);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<uint8_t>((buf >> bits) & 0xFF));
        }
    }
    return out;
}

// ---- JSON codecs ---------------------------------------------------

inline rjson to_json(const LogEntry& e) {
    return {{"term", e.term}, {"index", e.index}, {"op", e.op},
            {"payload", base64_encode(e.payload)}};
}

inline LogEntry log_entry_from_json(const rjson& j) {
    return {j.at("term").get<Term>(), j.at("index").get<Index>(),
            j.at("op").get<std::string>(),
            base64_decode(j.at("payload").get<std::string>())};
}

inline rjson to_json(const AppendEntriesArgs& a) {
    rjson entries = rjson::array();
    for (const auto& e : a.entries) entries.push_back(to_json(e));
    return {{"term", a.term}, {"leader_id", a.leader_id},
            {"prev_log_index", a.prev_log_index}, {"prev_log_term", a.prev_log_term},
            {"entries", entries}, {"leader_commit", a.leader_commit}};
}

inline AppendEntriesArgs append_args_from_json(const rjson& j) {
    AppendEntriesArgs a;
    a.term           = j.at("term").get<Term>();
    a.leader_id      = j.at("leader_id").get<NodeId>();
    a.prev_log_index = j.at("prev_log_index").get<Index>();
    a.prev_log_term  = j.at("prev_log_term").get<Term>();
    a.leader_commit  = j.at("leader_commit").get<Index>();
    for (const auto& e : j.at("entries")) a.entries.push_back(log_entry_from_json(e));
    return a;
}

inline rjson to_json(const AppendEntriesReply& r) {
    return {{"term", r.term}, {"success", r.success},
            {"conflict_index", r.conflict_index}, {"conflict_term", r.conflict_term}};
}

inline AppendEntriesReply append_reply_from_json(const rjson& j) {
    return {j.at("term").get<Term>(), j.at("success").get<bool>(),
            j.at("conflict_index").get<Index>(), j.at("conflict_term").get<Term>()};
}

inline rjson to_json(const RequestVoteArgs& a) {
    return {{"term", a.term}, {"candidate_id", a.candidate_id},
            {"last_log_index", a.last_log_index}, {"last_log_term", a.last_log_term},
            {"pre_vote", a.pre_vote}};
}

inline RequestVoteArgs vote_args_from_json(const rjson& j) {
    return {j.at("term").get<Term>(), j.at("candidate_id").get<NodeId>(),
            j.at("last_log_index").get<Index>(), j.at("last_log_term").get<Term>(),
            j.value("pre_vote", false)};
}

inline rjson to_json(const Membership& m) {
    rjson arr = rjson::array();
    for (const auto& x : m) arr.push_back({{"id", x.id}, {"addr", x.addr}});
    return arr;
}

inline Membership membership_from_json(const rjson& j) {
    Membership m;
    for (const auto& x : j)
        m.push_back({x.at("id").get<NodeId>(), x.value("addr", std::string())});
    return m;
}

inline rjson to_json(const InstallSnapshotArgs& a) {
    return {{"term", a.term}, {"leader_id", a.leader_id},
            {"last_included_index", a.last_included_index},
            {"last_included_term", a.last_included_term},
            {"config", to_json(a.config)}, {"offset", a.offset},
            {"data", base64_encode(a.data)}, {"done", a.done}};
}

inline InstallSnapshotArgs snapshot_args_from_json(const rjson& j) {
    return {j.at("term").get<Term>(), j.at("leader_id").get<NodeId>(),
            j.at("last_included_index").get<Index>(), j.at("last_included_term").get<Term>(),
            membership_from_json(j.at("config")), j.at("offset").get<uint64_t>(),
            base64_decode(j.at("data").get<std::string>()), j.at("done").get<bool>()};
}

inline rjson to_json(const InstallSnapshotReply& r) {
    return {{"term", r.term}, {"success", r.success}, {"next_offset", r.next_offset}};
}

inline InstallSnapshotReply snapshot_reply_from_json(const rjson& j) {
    return {j.at("term").get<Term>(), j.at("success").get<bool>(),
            j.at("next_offset").get<uint64_t>()};
}

inline rjson to_json(const RequestVoteReply& r) {
    return {{"term", r.term}, {"vote_granted", r.vote_granted}};
}

inline RequestVoteReply vote_reply_from_json(const rjson& j) {
    return {j.at("term").get<Term>(), j.at("vote_granted").get<bool>()};
}

// Constant-time comparison for the shared cluster secret.
inline bool secret_equals(const std::string& a, const std::string& b) {
    unsigned char diff = static_cast<unsigned char>(a.size() != b.size());
    for (std::size_t i = 0; i < std::max(a.size(), b.size()); ++i) {
        unsigned char x = i < a.size() ? a[i] : 0;
        unsigned char y = i < b.size() ? b[i] : 0;
        diff |= x ^ y;
    }
    return diff == 0;
}

// ---- HttpPeer --------------------------------------------------------

class HttpPeer : public RaftPeer {
public:
    // base_url like "http://10.0.0.2:6399"
    HttpPeer(std::string base_url, std::string secret,
             int connect_timeout_ms = 200, int read_timeout_ms = 2000)
        : base_url_(std::move(base_url))
        , secret_(std::move(secret))
        , client_(base_url_)
    {
        client_.set_connection_timeout(0, connect_timeout_ms * 1000);
        client_.set_read_timeout(read_timeout_ms / 1000, (read_timeout_ms % 1000) * 1000);
        client_.set_write_timeout(read_timeout_ms / 1000, (read_timeout_ms % 1000) * 1000);
        client_.set_keep_alive(true);
    }

    std::optional<AppendEntriesReply>
    append_entries(const AppendEntriesArgs& args) override {
        auto r = _post("/raft/append_entries", to_json(args));
        if (!r) return std::nullopt;
        try { return append_reply_from_json(*r); }
        catch (const std::exception&) { return std::nullopt; }
    }

    std::optional<RequestVoteReply>
    request_vote(const RequestVoteArgs& args) override {
        auto r = _post("/raft/request_vote", to_json(args));
        if (!r) return std::nullopt;
        try { return vote_reply_from_json(*r); }
        catch (const std::exception&) { return std::nullopt; }
    }

    std::optional<InstallSnapshotReply>
    install_snapshot(const InstallSnapshotArgs& args) override {
        auto r = _post("/raft/install_snapshot", to_json(args));
        if (!r) return std::nullopt;
        try { return snapshot_reply_from_json(*r); }
        catch (const std::exception&) { return std::nullopt; }
    }

    const std::string& url() const { return base_url_; }

private:
    std::optional<rjson> _post(const std::string& path, const rjson& body) {
        std::lock_guard lock(mu_);
        httplib::Headers h{{kClusterSecretHeader, secret_}};
        auto res = client_.Post(path, h, body.dump(), "application/json");
        if (!res || res->status != 200) return std::nullopt;
        try { return rjson::parse(res->body); }
        catch (const std::exception&) { return std::nullopt; }
    }

    std::string     base_url_;
    std::string     secret_;
    std::mutex      mu_;   // httplib::Client is not safe for concurrent use
    httplib::Client client_;
};

} // namespace consensus
} // namespace tidevec
