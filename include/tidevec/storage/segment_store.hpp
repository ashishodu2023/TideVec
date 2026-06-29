#pragma once
// ================================================================
// SegmentStore — disk-resident vector storage for billion-scale
//
// Architecture (inspired by DiskANN + LSM-VEC research):
//
//   · Vectors live in immutable SEGMENT FILES on SSD (.cvec)
//   · Each segment: fixed header + packed float32 rows
//   · Segments are mmap()'d for zero-copy page-cache reads
//   · New writes go to a WRITE BUFFER; flushed when full
//   · Background COMPACTION merges small segments into larger ones
//   · PQ codes (M bytes each) kept in a separate in-memory table
//     for fast ADC candidate ranking before SSD re-scoring
//
// Capacity math (1B vectors, dim=768):
//   Raw storage: 1B * 768 * 4 bytes = ~2.9 TB  (SSD)
//   PQ codes:    1B * 96 bytes      = ~89 GB    (RAM, M=96)
//   HNSW graph:  1B * 16 * 4 * 4   = ~256 GB   (SSD via DiskANN layout)
//
// This file implements the storage layer only (no index logic).
// ================================================================

#include <cortexdb/core/cortex_vector.hpp>

#include <string>
#include <vector>
#include <unordered_map>
#include <shared_mutex>
#include <fstream>
#include <filesystem>
#include <cstring>
#include <stdexcept>
#include <atomic>
#include <functional>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace cortexdb {
namespace fs = std::filesystem;

// ---- Segment file header (64 bytes) ----------------------------
struct SegmentHeader {
    uint8_t  magic[8]   = {'C','O','R','T','E','X','D','B'};
    uint32_t version    = 1;
    uint32_t dim;
    uint64_t n_vectors;
    uint64_t created_at;
    uint8_t  _pad[36];
};
static_assert(sizeof(SegmentHeader) == 64, "SegmentHeader must be 64 bytes");

// ---- Per-vector fixed-size record ------------------------------
// Stored as: id (64 bytes, null-padded) + float32[dim] + timestamps(16B)
struct VecRecord {
    char      id[64];
    int64_t   created_at;
    int64_t   valid_from;
    // float data follows: dim * sizeof(float) bytes
};

// ================================================================
// SegmentFile — a single immutable on-disk segment
// ================================================================
class SegmentFile {
public:
    SegmentFile(const std::string& path, std::size_t dim)
        : path_(path), dim_(dim)
    {
        _mmap_open();
    }

    ~SegmentFile() { _mmap_close(); }

    // Read the float embedding for vector at index i
    const float* get_embedding(uint64_t i) const {
        if (i >= header_->n_vectors)
            throw std::out_of_range("Segment index out of range");
        std::size_t rec_size = sizeof(VecRecord) + dim_ * sizeof(float);
        const uint8_t* base = mapped_ + sizeof(SegmentHeader);
        return reinterpret_cast<const float*>(
            base + i * rec_size + sizeof(VecRecord));
    }

    const VecRecord* get_record(uint64_t i) const {
        std::size_t rec_size = sizeof(VecRecord) + dim_ * sizeof(float);
        const uint8_t* base = mapped_ + sizeof(SegmentHeader);
        return reinterpret_cast<const VecRecord*>(base + i * rec_size);
    }

    uint64_t    n_vectors() const { return header_->n_vectors; }
    std::size_t dim()       const { return dim_; }
    std::string path()      const { return path_; }

private:
    void _mmap_open() {
        fd_ = open(path_.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("Cannot open segment: " + path_);
        struct stat st;
        fstat(fd_, &st);
        size_ = static_cast<std::size_t>(st.st_size);
        mapped_ = static_cast<uint8_t*>(
            mmap(nullptr, size_, PROT_READ, MAP_SHARED | MAP_POPULATE, fd_, 0));
        if (mapped_ == MAP_FAILED)
            throw std::runtime_error("mmap failed: " + path_);
        header_ = reinterpret_cast<const SegmentHeader*>(mapped_);
        // Advise sequential + random based on workload
        madvise(mapped_, size_, MADV_WILLNEED);
    }

    void _mmap_close() {
        if (mapped_ && mapped_ != MAP_FAILED) munmap(mapped_, size_);
        if (fd_ >= 0) close(fd_);
    }

    std::string path_;
    std::size_t dim_;
    int         fd_     = -1;
    uint8_t*    mapped_ = nullptr;
    std::size_t size_   = 0;
    const SegmentHeader* header_ = nullptr;
};

// ================================================================
// SegmentWriter — writes a new immutable segment to disk
// ================================================================
class SegmentWriter {
public:
    SegmentWriter(const std::string& path, std::size_t dim)
        : path_(path), dim_(dim)
    {
        file_.open(path_, std::ios::binary | std::ios::trunc | std::ios::out);
        if (!file_) throw std::runtime_error("Cannot create segment: " + path_);
        // Write placeholder header (will overwrite at close)
        SegmentHeader hdr{};
        hdr.dim        = static_cast<uint32_t>(dim_);
        hdr.created_at = now_ms();
        file_.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    }

    void write(const CortexVector& v) {
        VecRecord rec{};
        std::size_t id_copy = std::min(v.id.size(), sizeof(rec.id)-1);
        std::memcpy(rec.id, v.id.data(), id_copy);
        rec.created_at = v.created_at;
        rec.valid_from = v.valid_from;
        file_.write(reinterpret_cast<const char*>(&rec), sizeof(rec));
        file_.write(reinterpret_cast<const char*>(v.embedding.data()),
                    dim_ * sizeof(float));
        ++n_written_;
    }

    void close() {
        // Rewrite header with final n_vectors
        file_.seekp(0);
        SegmentHeader hdr{};
        hdr.dim        = static_cast<uint32_t>(dim_);
        hdr.n_vectors  = n_written_;
        hdr.created_at = now_ms();
        file_.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
        file_.close();
    }

    uint64_t    n_written() const { return n_written_; }
    std::string path()      const { return path_; }

private:
    std::string   path_;
    std::size_t   dim_;
    std::ofstream file_;
    uint64_t      n_written_ = 0;
};

// ================================================================
// SegmentStore — manages all segments for a collection
// ================================================================
class SegmentStore {
public:
    struct Config {
        std::string data_dir       = "./data";
        std::string collection_name;
        std::size_t dim            = 0;
        std::size_t write_buf_size = 100'000;   // vectors before flush
    };

    explicit SegmentStore(Config cfg) : cfg_(std::move(cfg)) {
        fs::create_directories(_seg_dir());
        _load_existing_segments();
    }

    // ------ Write path ------------------------------------------

    // Buffer a vector; auto-flushes when write_buf_size reached
    void put(const CortexVector& vec) {
        std::unique_lock lock(mutex_);
        write_buffer_.push_back(vec);
        id_to_loc_[vec.id] = {-1, 0};  // pending flush
        if (write_buffer_.size() >= cfg_.write_buf_size)
            _flush_locked();
    }

    // Force flush write buffer to a new segment
    void flush() {
        std::unique_lock lock(mutex_);
        _flush_locked();
    }

    // ------ Read path --------------------------------------------

    // Fetch a vector's float data (zero-copy from mmap'd segment)
    const float* get_embedding(const std::string& id) const {
        std::shared_lock lock(mutex_);
        // Check write buffer first
        for (const auto& v : write_buffer_)
            if (v.id == id) return v.embedding.data();
        auto it = id_to_loc_.find(id);
        if (it == id_to_loc_.end()) return nullptr;
        auto [seg_idx, rec_idx] = it->second;
        if (seg_idx < 0) return nullptr;  // still in buffer
        return segments_[seg_idx]->get_embedding(rec_idx);
    }

    // Materialise a full CortexVector from disk
    std::optional<CortexVector> get(const std::string& id) const {
        std::shared_lock lock(mutex_);
        // write buffer
        for (const auto& v : write_buffer_)
            if (v.id == id) return v;

        auto it = id_to_loc_.find(id);
        if (it == id_to_loc_.end()) return std::nullopt;
        auto [seg_idx, rec_idx] = it->second;
        if (seg_idx < 0) return std::nullopt;

        const auto* seg = segments_[seg_idx].get();
        const auto* rec = seg->get_record(rec_idx);
        CortexVector v;
        v.id         = std::string(rec->id);
        v.created_at = rec->created_at;
        v.valid_from = rec->valid_from;
        v.embedding.assign(seg->get_embedding(rec_idx),
                           seg->get_embedding(rec_idx) + cfg_.dim);
        return v;
    }

    bool contains(const std::string& id) const {
        std::shared_lock lock(mutex_);
        return id_to_loc_.count(id) > 0;
    }

    void mark_deleted(const std::string& id) {
        std::unique_lock lock(mutex_);
        deleted_.insert(id);
        id_to_loc_.erase(id);
    }

    // Iterate all non-deleted vector IDs
    void for_each_id(std::function<void(const std::string&)> fn) const {
        std::shared_lock lock(mutex_);
        for (const auto& [id, _] : id_to_loc_)
            if (!deleted_.count(id)) fn(id);
        for (const auto& v : write_buffer_)
            if (!deleted_.count(v.id)) fn(v.id);
    }

    std::size_t total_vectors() const {
        std::shared_lock lock(mutex_);
        return id_to_loc_.size() + write_buffer_.size();
    }

    std::size_t n_segments() const {
        std::shared_lock lock(mutex_);
        return segments_.size();
    }

private:
    std::string _seg_dir() const {
        return cfg_.data_dir + "/" + cfg_.collection_name;
    }

    std::string _new_seg_path() const {
        return _seg_dir() + "/seg_" + std::to_string(now_ms()) + ".cvec";
    }

    void _flush_locked() {
        if (write_buffer_.empty()) return;
        std::string path = _new_seg_path();
        SegmentWriter writer(path, cfg_.dim);
        for (const auto& v : write_buffer_) {
            if (!deleted_.count(v.id)) {
                writer.write(v);
                uint64_t rec_idx = writer.n_written() - 1;
                int seg_idx = static_cast<int>(segments_.size());
                id_to_loc_[v.id] = {seg_idx, rec_idx};
            }
        }
        writer.close();
        segments_.push_back(
            std::make_unique<SegmentFile>(path, cfg_.dim));
        write_buffer_.clear();
    }

    void _load_existing_segments() {
        if (!fs::exists(_seg_dir())) return;
        for (const auto& e : fs::directory_iterator(_seg_dir())) {
            if (e.path().extension() != ".cvec") continue;
            try {
                auto seg = std::make_unique<SegmentFile>(
                    e.path().string(), cfg_.dim);
                int seg_idx = static_cast<int>(segments_.size());
                for (uint64_t i = 0; i < seg->n_vectors(); ++i) {
                    const auto* rec = seg->get_record(i);
                    std::string id(rec->id);
                    if (!deleted_.count(id))
                        id_to_loc_[id] = {seg_idx, i};
                }
                segments_.push_back(std::move(seg));
            } catch (...) { /* skip corrupt segment */ }
        }
    }

    Config cfg_;
    mutable std::shared_mutex mutex_;

    std::vector<std::unique_ptr<SegmentFile>> segments_;
    std::vector<CortexVector>                 write_buffer_;
    // id → {segment_index, record_index_in_segment}
    std::unordered_map<std::string, std::pair<int,uint64_t>> id_to_loc_;
    std::unordered_set<std::string> deleted_;
};

} // namespace cortexdb
