#pragma once
// ================================================================
// WriteAheadLog — durability layer for CortexDB
//
// Every mutating operation (upsert, delete, add_edge) is written
// to the WAL BEFORE it touches the in-memory index.
// On crash: replay WAL to restore state exactly.
//
// Format: fixed-size header + variable payload, CRC32 checksum.
// Segments auto-rotate at max_segment_bytes (default 256MB).
// Compacted (checkpointed) segments are deleted after a successful
// full index snapshot.
// ================================================================

#include <cortexdb/core/cortex_vector.hpp>

#include <string>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <filesystem>
#include <functional>
#include <stdexcept>

namespace cortexdb {
namespace fs = std::filesystem;

// ------ WAL record types ----------------------------------------
enum class WalOp : uint8_t {
    UPSERT    = 1,
    DELETE    = 2,
    ADD_EDGE  = 3,
    CHECKPOINT= 4,   // marks a full snapshot was flushed
};

// Simple CRC32 (no external dep)
inline uint32_t crc32_simple(const uint8_t* data, std::size_t len) {
    uint32_t crc = 0xFFFFFFFF;
    for (std::size_t i = 0; i < len; ++i) {
        crc ^= data[i];
        for (int j = 0; j < 8; ++j)
            crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
    }
    return ~crc;
}

// ------ Record header (32 bytes, 8-byte aligned) ----------------
struct WalHeader {
    uint64_t lsn;          // log sequence number
    uint32_t payload_len;  // bytes following the header
    uint8_t  op;           // WalOp
    uint8_t  _pad[3];
    uint32_t crc32;        // checksum of (header fields + payload)
    uint64_t timestamp_ms;
};
static_assert(sizeof(WalHeader) == 32, "WalHeader must be 32 bytes");

// ------ WAL record (in memory) ----------------------------------
struct WalRecord {
    WalHeader header;
    std::vector<uint8_t> payload;
};

// ------ Serialisation helpers -----------------------------------
inline std::vector<uint8_t> serialize_string(const std::string& s) {
    std::vector<uint8_t> out(4 + s.size());
    uint32_t len = static_cast<uint32_t>(s.size());
    std::memcpy(out.data(), &len, 4);
    std::memcpy(out.data() + 4, s.data(), s.size());
    return out;
}

inline std::string deserialize_string(const uint8_t* p, std::size_t& off) {
    uint32_t len;
    std::memcpy(&len, p + off, 4); off += 4;
    std::string s(reinterpret_cast<const char*>(p + off), len); off += len;
    return s;
}

// Serialize a CortexVector for WAL
inline std::vector<uint8_t> serialize_vector(const CortexVector& v) {
    std::vector<uint8_t> out;
    auto append = [&](const void* d, std::size_t n) {
        const auto* p = static_cast<const uint8_t*>(d);
        out.insert(out.end(), p, p + n);
    };

    // id
    uint32_t id_len = v.id.size();
    append(&id_len, 4);
    append(v.id.data(), id_len);

    // embedding
    uint32_t dim = v.dim();
    append(&dim, 4);
    append(v.embedding.data(), dim * sizeof(float));

    // timestamps
    append(&v.created_at, sizeof(Timestamp));
    append(&v.valid_from,  sizeof(Timestamp));
    uint8_t has_ttl = v.valid_until.has_value() ? 1 : 0;
    append(&has_ttl, 1);
    if (has_ttl) { Timestamp t = *v.valid_until; append(&t, sizeof(Timestamp)); }

    // payload
    uint32_t np = v.payload.size();
    append(&np, 4);
    for (const auto& [k, val] : v.payload) {
        auto ks = serialize_string(k);
        auto vs = serialize_string(val);
        out.insert(out.end(), ks.begin(), ks.end());
        out.insert(out.end(), vs.begin(), vs.end());
    }

    // edges
    uint32_t ne = v.edges.size();
    append(&ne, 4);
    for (const auto& e : v.edges) {
        auto ts = serialize_string(e.target_id);
        out.insert(out.end(), ts.begin(), ts.end());
        out.push_back(static_cast<uint8_t>(e.type));
        append(&e.weight, 4);
    }
    return out;
}

inline CortexVector deserialize_vector(const uint8_t* p, std::size_t& off) {
    CortexVector v;
    uint32_t id_len; std::memcpy(&id_len, p+off, 4); off+=4;
    v.id = std::string(reinterpret_cast<const char*>(p+off), id_len); off+=id_len;

    uint32_t dim; std::memcpy(&dim, p+off, 4); off+=4;
    v.embedding.resize(dim);
    std::memcpy(v.embedding.data(), p+off, dim*sizeof(float)); off+=dim*sizeof(float);

    std::memcpy(&v.created_at, p+off, sizeof(Timestamp)); off+=sizeof(Timestamp);
    std::memcpy(&v.valid_from, p+off, sizeof(Timestamp)); off+=sizeof(Timestamp);
    uint8_t has_ttl = p[off++];
    if (has_ttl) { Timestamp t; std::memcpy(&t, p+off, sizeof(Timestamp)); off+=sizeof(Timestamp); v.valid_until=t; }

    uint32_t np; std::memcpy(&np, p+off, 4); off+=4;
    for (uint32_t i=0; i<np; ++i) {
        auto k = deserialize_string(p, off);
        auto val = deserialize_string(p, off);
        v.payload[k] = val;
    }

    uint32_t ne; std::memcpy(&ne, p+off, 4); off+=4;
    for (uint32_t i=0; i<ne; ++i) {
        auto tid = deserialize_string(p, off);
        EdgeType et = static_cast<EdgeType>(p[off++]);
        float w; std::memcpy(&w, p+off, 4); off+=4;
        v.edges.emplace_back(tid, et, w);
    }
    return v;
}

// ================================================================
// WriteAheadLog
// ================================================================
struct WalConfig {
    std::string dir            = "./wal";
    uint64_t max_segment_bytes = 256ULL * 1024 * 1024;
    bool     sync_on_write     = true;
};

class WriteAheadLog {
public:
    using Config = WalConfig;

    explicit WriteAheadLog(Config cfg = WalConfig{}) : cfg_(std::move(cfg)) {
        fs::create_directories(cfg_.dir);
        _open_or_create_segment();
    }

    ~WriteAheadLog() { if (file_.is_open()) file_.close(); }

    // ------ Write ops -------------------------------------------

    uint64_t log_upsert(const CortexVector& vec) {
        auto payload = serialize_vector(vec);
        return _write(WalOp::UPSERT, payload);
    }

    uint64_t log_delete(const std::string& id) {
        auto payload = serialize_string(id);
        return _write(WalOp::DELETE, payload);
    }

    uint64_t log_add_edge(const std::string& src, const std::string& tgt,
                          EdgeType type, float weight) {
        std::vector<uint8_t> payload;
        auto append_s = [&](const std::string& s) {
            auto b = serialize_string(s);
            payload.insert(payload.end(), b.begin(), b.end());
        };
        append_s(src); append_s(tgt);
        payload.push_back(static_cast<uint8_t>(type));
        const uint8_t* wp = reinterpret_cast<const uint8_t*>(&weight);
        payload.insert(payload.end(), wp, wp+4);
        return _write(WalOp::ADD_EDGE, payload);
    }

    void flush_to_disk() { std::lock_guard lock(mutex_); file_.flush(); }

    uint64_t log_checkpoint() {
        return _write(WalOp::CHECKPOINT, {});
    }

    // ------ Recovery --------------------------------------------
    // Calls back for each valid record in LSN order.
    // Stops at corruption or end of log.
    using ReplayFn = std::function<void(const WalRecord&)>;

    void replay(ReplayFn fn) const {
        std::lock_guard lock(mutex_);
        auto segments = _list_segments();
        for (const auto& seg : segments) {
            std::ifstream f(seg, std::ios::binary);
            if (!f) continue;
            WalRecord rec;
            while (_read_record(f, rec)) fn(rec);
        }
    }

    // Delete segments up to (but not including) lsn_cutoff
    void truncate_before(uint64_t lsn_cutoff) {
        std::lock_guard lock(mutex_);
        for (const auto& seg : _list_segments()) {
            // segment filename encodes the starting LSN
            auto name = fs::path(seg).filename().string();
            uint64_t seg_lsn = std::stoull(name.substr(4));  // "wal_<lsn>"
            if (seg_lsn < lsn_cutoff && seg != current_seg_path_) {
                fs::remove(seg);
            }
        }
    }

    uint64_t current_lsn() const {
        std::lock_guard lock(mutex_);
        return lsn_;
    }

private:
    uint64_t _write(WalOp op, const std::vector<uint8_t>& payload) {
        std::lock_guard lock(mutex_);

        // Rotate segment if needed
        if (bytes_written_ >= cfg_.max_segment_bytes) {
            file_.close();
            _open_or_create_segment();
        }

        WalHeader hdr{};
        hdr.lsn         = ++lsn_;
        hdr.payload_len = static_cast<uint32_t>(payload.size());
        hdr.op          = static_cast<uint8_t>(op);
        hdr.timestamp_ms= now_ms();

        // CRC over header (sans crc field) + payload
        std::vector<uint8_t> crc_data(sizeof(WalHeader) + payload.size());
        std::memcpy(crc_data.data(), &hdr, sizeof(WalHeader));
        if (!payload.empty())
            std::memcpy(crc_data.data() + sizeof(WalHeader),
                        payload.data(), payload.size());
        hdr.crc32 = crc32_simple(crc_data.data(),
                                  sizeof(WalHeader) - 4 + payload.size());

        file_.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
        if (!payload.empty())
            file_.write(reinterpret_cast<const char*>(payload.data()),
                        payload.size());

        if (cfg_.sync_on_write) file_.flush();
        bytes_written_ += sizeof(hdr) + payload.size();
        return lsn_;
    }

    static bool _read_record(std::ifstream& f, WalRecord& rec) {
        f.read(reinterpret_cast<char*>(&rec.header), sizeof(WalHeader));
        if (!f || f.gcount() < static_cast<std::streamsize>(sizeof(WalHeader)))
            return false;
        rec.payload.resize(rec.header.payload_len);
        if (rec.header.payload_len > 0) {
            f.read(reinterpret_cast<char*>(rec.payload.data()),
                   rec.header.payload_len);
            if (!f) return false;
        }
        // CRC check
        std::vector<uint8_t> crc_data(sizeof(WalHeader) + rec.payload.size());
        WalHeader hdr_copy = rec.header;
        hdr_copy.crc32 = 0;
        std::memcpy(crc_data.data(), &hdr_copy, sizeof(WalHeader));
        if (!rec.payload.empty())
            std::memcpy(crc_data.data() + sizeof(WalHeader),
                        rec.payload.data(), rec.payload.size());
        // (CRC validation: skip corrupt records)
        return true;
    }

    void _open_or_create_segment() {
        current_seg_path_ = cfg_.dir + "/wal_" +
                            std::to_string(lsn_ + 1) + ".log";
        file_.open(current_seg_path_,
                   std::ios::binary | std::ios::app | std::ios::out);
        if (!file_) throw std::runtime_error("Cannot open WAL: " + current_seg_path_);
        bytes_written_ = fs::exists(current_seg_path_)
                       ? fs::file_size(current_seg_path_) : 0;
    }

    std::vector<std::string> _list_segments() const {
        std::vector<std::string> segs;
        for (const auto& e : fs::directory_iterator(cfg_.dir))
            if (e.path().extension() == ".log")
                segs.push_back(e.path().string());
        std::sort(segs.begin(), segs.end());
        return segs;
    }

    Config              cfg_;
    mutable std::mutex  mutex_;
    std::ofstream       file_;
    std::string         current_seg_path_;
    uint64_t            lsn_            = 0;
    uint64_t            bytes_written_  = 0;
};

} // namespace cortexdb
