#pragma once
// ================================================================
// cluster/state_snapshot.hpp — file-level state-machine snapshots
//
// EXPERIMENTAL (cluster mode). A Raft snapshot of a TideVec node is the
// set of files that restart recovery reads: registry.json plus every
// collection directory (WAL, segments). Packing them on the Raft apply
// thread gives a consistent cut, because that thread is the only writer;
// restoring them and re-running recovery reproduces the state exactly as
// a restart would.
//
// Archive layout: "TVFS" | u32 version=1 | u32 n_files |
//   n × ( u32 path_len | path (relative, '/'-separated) | u64 size | bytes )
//
// Limitation: the WAL keeps every overwrite, so snapshot size follows
// WAL size, not live data, until WAL compaction exists.
// ================================================================

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

namespace tidevec {
namespace state_snapshot {

namespace fs = std::filesystem;

struct File {
    std::string          path;   // relative to the data directory
    std::vector<uint8_t> data;
};

namespace detail {
template <typename T>
inline void put(std::vector<uint8_t>& b, T v) {
    uint8_t raw[sizeof(T)];
    std::memcpy(raw, &v, sizeof(T));
    b.insert(b.end(), raw, raw + sizeof(T));
}
template <typename T>
inline T get(const std::vector<uint8_t>& b, std::size_t& off) {
    if (off + sizeof(T) > b.size()) throw std::runtime_error("state snapshot truncated");
    T v;
    std::memcpy(&v, b.data() + off, sizeof(T));
    off += sizeof(T);
    return v;
}
inline bool skip(const fs::path& p) {
    auto ext = p.extension().string();
    return ext == ".tmp" || ext == ".lock";
}
} // namespace detail

// Packs `entries` (files or directories, relative to root; missing ones
// are skipped) into one archive.
inline std::vector<uint8_t> pack(const fs::path& root, const std::vector<std::string>& entries) {
    std::vector<File> files;
    auto add = [&](const fs::path& abs) {
        if (detail::skip(abs)) return;
        std::ifstream f(abs, std::ios::binary);
        if (!f) throw std::runtime_error("state snapshot: cannot read " + abs.string());
        File out;
        out.path = fs::relative(abs, root).generic_string();
        out.data.assign(std::istreambuf_iterator<char>(f), {});
        files.push_back(std::move(out));
    };
    for (const auto& e : entries) {
        fs::path p = root / e;
        std::error_code ec;
        if (fs::is_regular_file(p, ec)) add(p);
        else if (fs::is_directory(p, ec))
            for (const auto& it : fs::recursive_directory_iterator(p))
                if (it.is_regular_file()) add(it.path());
    }
    std::vector<uint8_t> b{'T', 'V', 'F', 'S'};
    detail::put<uint32_t>(b, 1);
    detail::put<uint32_t>(b, static_cast<uint32_t>(files.size()));
    for (const auto& f : files) {
        detail::put<uint32_t>(b, static_cast<uint32_t>(f.path.size()));
        b.insert(b.end(), f.path.begin(), f.path.end());
        detail::put<uint64_t>(b, f.data.size());
        b.insert(b.end(), f.data.begin(), f.data.end());
    }
    return b;
}

// Parses and validates an archive (relative paths only, no "..").
inline std::vector<File> unpack(const std::vector<uint8_t>& b) {
    if (b.size() < 12 || std::memcmp(b.data(), "TVFS", 4) != 0)
        throw std::runtime_error("not a TideVec state snapshot");
    std::size_t off = 4;
    if (detail::get<uint32_t>(b, off) != 1)
        throw std::runtime_error("unsupported state snapshot version");
    uint32_t n = detail::get<uint32_t>(b, off);
    std::vector<File> files;
    files.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t plen = detail::get<uint32_t>(b, off);
        if (off + plen > b.size()) throw std::runtime_error("state snapshot truncated");
        File f;
        f.path.assign(b.begin() + static_cast<std::ptrdiff_t>(off),
                      b.begin() + static_cast<std::ptrdiff_t>(off + plen));
        off += plen;
        fs::path p(f.path);
        if (f.path.empty() || p.is_absolute())
            throw std::runtime_error("state snapshot: bad path " + f.path);
        for (const auto& part : p)
            if (part == "..") throw std::runtime_error("state snapshot: bad path " + f.path);
        uint64_t size = detail::get<uint64_t>(b, off);
        if (off + size > b.size()) throw std::runtime_error("state snapshot truncated");
        f.data.assign(b.begin() + static_cast<std::ptrdiff_t>(off),
                      b.begin() + static_cast<std::ptrdiff_t>(off + size));
        off += size;
        files.push_back(std::move(f));
    }
    if (off != b.size()) throw std::runtime_error("state snapshot has trailing bytes");
    return files;
}

// Writes files under root (fsync'ed).
inline void write_files(const fs::path& root, const std::vector<File>& files) {
    for (const auto& f : files) {
        fs::path p = root / f.path;
        fs::create_directories(p.parent_path());
        std::FILE* fp = std::fopen(p.c_str(), "wb");
        if (!fp) throw std::runtime_error("state snapshot: cannot write " + p.string());
        bool ok = std::fwrite(f.data.data(), 1, f.data.size(), fp) == f.data.size() &&
                  std::fflush(fp) == 0 && ::fsync(::fileno(fp)) == 0;
        std::fclose(fp);
        if (!ok) throw std::runtime_error("state snapshot: write failed for " + p.string());
    }
}

} // namespace state_snapshot
} // namespace tidevec
