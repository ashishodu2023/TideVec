#pragma once
// ================================================================
// BackupManager — periodic snapshots, S3/GCS upload, PITR restore
//
// A snapshot is `tar -czf` of the data directory's contents. External
// programs (tar, aws, gsutil) are started with posix_spawnp and an argv
// array, never through a shell.
// ================================================================

#include <tidevec/api/json.hpp>

#include <string>
#include <algorithm>
#include <thread>
#include <atomic>
#include <chrono>
#include <iostream>
#include <sstream>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <vector>
#include <cctype>
#include <cerrno>

#include <spawn.h>
#include <fcntl.h>
#include <sys/wait.h>

extern char** environ;

namespace tidevec {
namespace fs = std::filesystem;

struct BackupConfig {
    std::string data_dir          = "./tidevec_data";
    std::string backup_dir        = "./backups";
    int         interval_hours    = 6;
    int         retention_count   = 48;
    std::string s3_uri            = "";
    std::string gcs_uri           = "";
    bool        enabled           = false;
};

struct BackupManifest {
    std::string snapshot;
    int64_t     created_at = 0;
    std::string data_dir;
    std::string s3_uri;
    std::string gcs_uri;
};

struct SnapshotNotFound : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Snapshot names are file names inside the backup directory:
// [A-Za-z0-9_.-]+ ending in .tar.gz, no leading '.', no "..".
inline bool is_valid_snapshot_name(const std::string& name) {
    static const std::string ext = ".tar.gz";
    if (name.size() <= ext.size() || name.size() > 255) return false;
    if (!name.ends_with(ext) || name[0] == '.') return false;
    if (name.find("..") != std::string::npos) return false;
    for (unsigned char c : name)
        if (!(std::isalnum(c) || c == '_' || c == '-' || c == '.')) return false;
    return true;
}

// Runs argv[0] (looked up in PATH) with stdin/stdout/stderr on /dev/null
// and returns its exit status, or -1 if it could not be started.
inline int run_program(const std::vector<std::string>& args) {
    if (args.empty()) return -1;
    std::vector<char*> argv;
    for (const auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
    argv.push_back(nullptr);

    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_addopen(&fa, 1, "/dev/null", O_WRONLY, 0);
    posix_spawn_file_actions_addopen(&fa, 2, "/dev/null", O_WRONLY, 0);
    pid_t pid = 0;
    int rc = posix_spawnp(&pid, argv[0], &fa, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&fa);
    if (rc != 0) return -1;

    int status = 0;
    while (waitpid(pid, &status, 0) < 0)
        if (errno != EINTR) return -1;
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

class BackupManager {
public:
    // Called by restore_snapshot with the function that swaps the data
    // directory's contents; lets the caller close and reopen collections
    // around it. Must call swap exactly once and propagate its exceptions.
    using SwapHook = std::function<void(const std::function<void()>& swap)>;

    explicit BackupManager(BackupConfig cfg)
        : cfg_(std::move(cfg)) {
        fs::create_directories(cfg_.backup_dir);
    }

    ~BackupManager() { stop(); }

    void start() {
        if (!cfg_.enabled) return;
        running_ = true;
        thread_ = std::thread([this] { _loop(); });
        std::cout << "[backup] Started — interval "
                  << cfg_.interval_hours << "h, dir "
                  << cfg_.backup_dir << "\n";
    }

    void stop() {
        running_ = false;
        if (thread_.joinable()) thread_.join();
    }

    std::string snapshot_now() {
        std::lock_guard lock(op_mu_);
        return _create_snapshot();
    }

    std::vector<std::string> list_snapshots() const {
        std::vector<std::string> out;
        std::error_code ec;
        if (!fs::exists(cfg_.backup_dir, ec)) return out;
        for (const auto& e : fs::directory_iterator(cfg_.backup_dir, ec)) {
            auto name = e.path().filename().string();
            if (e.is_regular_file() && is_valid_snapshot_name(name))
                out.push_back(name);
        }
        std::sort(out.begin(), out.end());
        return out;
    }

    // Point-in-time recovery manifest history (newest last)
    std::vector<BackupManifest> list_manifests() const {
        std::vector<BackupManifest> out;
        auto path = cfg_.backup_dir + "/manifests.jsonl";
        if (!fs::exists(path)) {
            // Fall back to latest manifest.json
            auto latest = cfg_.backup_dir + "/manifest.json";
            if (fs::exists(latest)) {
                try {
                    std::ifstream f(latest);
                    auto m = _manifest_from_json(json::parse(f));
                    if (!m.snapshot.empty()) out.push_back(m);
                } catch (...) {}
            }
            return out;
        }
        std::ifstream f(path);
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            try { out.push_back(_manifest_from_json(json::parse(line))); }
            catch (...) {}
        }
        return out;
    }

    // Replaces the contents of data_dir with the snapshot. The data
    // directory itself is never renamed or removed, so it may be a mount
    // point. tenants.json (API keys) is left as it is.
    std::string restore_snapshot(const std::string& snapshot_name,
                                 const SwapHook& around_swap = {}) {
        if (!is_valid_snapshot_name(snapshot_name))
            throw std::invalid_argument("Invalid snapshot name: '" + snapshot_name +
                                        "' (expected a file name like tidevec_<ts>.tar.gz)");
        std::lock_guard lock(op_mu_);
        const fs::path archive = fs::path(cfg_.backup_dir) / snapshot_name;
        if (!fs::is_regular_file(archive))
            throw SnapshotNotFound("Snapshot not found: " + snapshot_name);

        std::string safety = _create_snapshot();
        std::cout << "[backup] Pre-restore safety snapshot: " << safety << "\n";

        const fs::path data = cfg_.data_dir;
        fs::create_directories(data);
        const std::string stamp = std::to_string(
            std::chrono::system_clock::now().time_since_epoch().count());
        const fs::path staging = data / (std::string(kInternalPrefix) + "restore_" + stamp);
        const fs::path aside   = data / (std::string(kInternalPrefix) + "pre_restore_" + stamp);
        fs::create_directory(staging);

        struct Cleanup {
            fs::path a, b;
            ~Cleanup() { std::error_code ec; fs::remove_all(a, ec); fs::remove_all(b, ec); }
        } cleanup{staging, aside};

        int rc = run_program({"tar", "-xzf", archive.string(), "-C", staging.string(),
                              "--no-same-owner"});
        if (rc != 0)
            throw std::runtime_error("Restore failed: could not extract " + snapshot_name +
                                     " (tar exit " + std::to_string(rc) + ")");
        const fs::path root = _snapshot_root(staging);

        auto swap = [&] { _swap_contents(data, root, aside); };
        if (around_swap) around_swap(swap);
        else swap();

        json restore_log = {
            {"restored_from", snapshot_name},
            {"restored_at",
                std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count()},
            {"safety_snapshot", safety},
        };
        std::ofstream rf(cfg_.backup_dir + "/last_restore.json");
        rf << restore_log.dump(2);

        return "Restored from " + snapshot_name + " (safety snapshot: " + safety + ")";
    }

private:
    // Top-level entries of the data directory that snapshots skip and
    // restores leave in place.
    static constexpr const char* kInternalPrefix = ".tidevec_";
    static constexpr const char* kTenantsFile    = "tenants.json";

    static BackupManifest _manifest_from_json(const json& j) {
        BackupManifest m;
        m.snapshot   = j.value("snapshot", "");
        m.created_at = j.value("created_at", int64_t{0});
        m.data_dir   = j.value("data_dir", "");
        m.s3_uri     = j.value("s3_uri", "");
        m.gcs_uri    = j.value("gcs_uri", "");
        return m;
    }

    void _loop() {
        while (running_) {
            try {
                std::string name;
                {
                    std::lock_guard lock(op_mu_);
                    name = _create_snapshot();
                    _prune_old();
                }
                std::cout << "[backup] Snapshot created: " << name << "\n";
            } catch (const std::exception& e) {
                std::cerr << "[backup] Error: " << e.what() << "\n";
            }
            for (int h = 0; h < cfg_.interval_hours * 3600 && running_; ++h)
                std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }

    // The backup directory's path relative to the data directory, if it
    // is inside it ("" otherwise).
    std::string _backup_dir_inside_data() const {
        std::error_code ec;
        auto data = fs::weakly_canonical(cfg_.data_dir, ec);
        auto back = fs::weakly_canonical(cfg_.backup_dir, ec);
        auto rel  = back.lexically_relative(data);
        if (rel.empty() || *rel.begin() == ".." || rel == ".") return "";
        return rel.begin()->string();
    }

    // Caller holds op_mu_.
    std::string _create_snapshot() {
        auto ts = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        std::string name = "tidevec_" + std::to_string(ts) + ".tar.gz";
        for (int n = 1; fs::exists(fs::path(cfg_.backup_dir) / name); ++n)
            name = "tidevec_" + std::to_string(ts) + "_" + std::to_string(n) + ".tar.gz";
        const fs::path path = fs::path(cfg_.backup_dir) / name;
        const fs::path partial = path.string() + ".partial";

        fs::create_directories(cfg_.data_dir);
        std::vector<std::string> args{"tar", "-czf", partial.string(), "-C", cfg_.data_dir,
                                      std::string("--exclude=./") + kInternalPrefix + "*"};
        if (auto inner = _backup_dir_inside_data(); !inner.empty())
            args.push_back("--exclude=./" + inner);
        args.push_back(".");
        int rc = run_program(args);
        // GNU tar exits 1 when a file changed while it was read, which is
        // expected for a live server.
        if (rc != 0 && rc != 1) {
            std::error_code ec;
            fs::remove(partial, ec);
            throw std::runtime_error("tar snapshot failed (exit " + std::to_string(rc) + ")");
        }
        fs::rename(partial, path);

        if (!cfg_.s3_uri.empty() &&
            run_program({"aws", "s3", "cp", path.string(), cfg_.s3_uri + name}) != 0)
            std::cerr << "[backup] Upload to " << cfg_.s3_uri << " failed\n";
        if (!cfg_.gcs_uri.empty() &&
            run_program({"gsutil", "cp", path.string(), cfg_.gcs_uri + name}) != 0)
            std::cerr << "[backup] Upload to " << cfg_.gcs_uri << " failed\n";

        json manifest = {
            {"snapshot", name},
            {"created_at", ts},
            {"data_dir", cfg_.data_dir},
        };
        if (!cfg_.s3_uri.empty())
            manifest["s3_uri"] = cfg_.s3_uri + name;
        if (!cfg_.gcs_uri.empty())
            manifest["gcs_uri"] = cfg_.gcs_uri + name;
        std::ofstream mf(cfg_.backup_dir + "/manifest.json");
        mf << manifest.dump(2);

        // Append to PITR history
        std::ofstream hist(cfg_.backup_dir + "/manifests.jsonl",
                           std::ios::app);
        hist << manifest.dump() << "\n";

        return name;
    }

    // Snapshots before 0.3.0 archived the data directory itself, so their
    // contents sit under a single top-level directory.
    static fs::path _snapshot_root(const fs::path& staging) {
        std::vector<fs::path> entries;
        for (const auto& e : fs::directory_iterator(staging)) entries.push_back(e.path());
        if (entries.size() == 1 && fs::is_directory(entries[0]) &&
            !fs::exists(staging / "registry.json") &&
            fs::exists(entries[0] / "registry.json"))
            return entries[0];
        return staging;
    }

    bool _keep_in_place(const std::string& name) const {
        return name.starts_with(kInternalPrefix) || name == kTenantsFile ||
               name == _backup_dir_inside_data();
    }

    // Moves the current contents of data into aside, then the snapshot's
    // contents into data. All renames stay inside data's file system. On
    // failure the original contents are moved back.
    void _swap_contents(const fs::path& data, const fs::path& root,
                        const fs::path& aside) const {
        auto entries = [this](const fs::path& dir) {
            std::vector<std::string> out;
            for (const auto& e : fs::directory_iterator(dir)) {
                auto name = e.path().filename().string();
                if (!_keep_in_place(name)) out.push_back(name);
            }
            return out;
        };
        fs::create_directory(aside);
        std::vector<std::string> moved_out, moved_in;
        try {
            for (const auto& name : entries(data)) {
                fs::rename(data / name, aside / name);
                moved_out.push_back(name);
            }
            for (const auto& name : entries(root)) {
                fs::rename(root / name, data / name);
                moved_in.push_back(name);
            }
        } catch (...) {
            std::error_code ec;
            for (const auto& name : moved_in) fs::remove_all(data / name, ec);
            for (const auto& name : moved_out) fs::rename(aside / name, data / name, ec);
            throw;
        }
    }

    // Caller holds op_mu_.
    void _prune_old() {
        auto snaps = list_snapshots();
        if (static_cast<int>(snaps.size()) <= cfg_.retention_count) return;
        int to_delete = static_cast<int>(snaps.size()) - cfg_.retention_count;
        std::error_code ec;
        for (int i = 0; i < to_delete; ++i)
            fs::remove(fs::path(cfg_.backup_dir) / snaps[i], ec);
    }

    BackupConfig cfg_;
    std::mutex op_mu_;     // one snapshot or restore at a time
    std::atomic<bool> running_{false};
    std::thread thread_;
};

} // namespace tidevec
