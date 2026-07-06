#pragma once
// ================================================================
// BackupManager — periodic snapshots, S3/GCS upload, PITR restore
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
#include <cstdlib>
#include <vector>

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

class BackupManager {
public:
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
        return _create_snapshot();
    }

    std::vector<std::string> list_snapshots() const {
        std::vector<std::string> out;
        if (!fs::exists(cfg_.backup_dir)) return out;
        for (const auto& e : fs::directory_iterator(cfg_.backup_dir)) {
            if (e.path().extension() == ".tar.gz")
                out.push_back(e.path().filename().string());
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
                    auto j = json::parse(f);
                    BackupManifest m;
                    m.snapshot    = j.value("snapshot", "");
                    m.created_at  = j.value("created_at", 0L);
                    m.data_dir    = j.value("data_dir", "");
                    m.s3_uri      = j.value("s3_uri", "");
                    m.gcs_uri     = j.value("gcs_uri", "");
                    if (!m.snapshot.empty()) out.push_back(m);
                } catch (...) {}
            }
            return out;
        }
        std::ifstream f(path);
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            try {
                auto j = json::parse(line);
                BackupManifest m;
                m.snapshot   = j.value("snapshot", "");
                m.created_at = j.value("created_at", 0L);
                m.data_dir   = j.value("data_dir", "");
                m.s3_uri     = j.value("s3_uri", "");
                m.gcs_uri    = j.value("gcs_uri", "");
                out.push_back(m);
            } catch (...) {}
        }
        return out;
    }

    // Restore data_dir from a snapshot tarball (server restart recommended).
    std::string restore_snapshot(const std::string& snapshot_name) {
        std::string path = cfg_.backup_dir + "/" + snapshot_name;
        if (!fs::exists(path))
            throw std::runtime_error("Snapshot not found: " + snapshot_name);

        // Safety snapshot before restore
        std::string safety = _create_snapshot();
        std::cout << "[backup] Pre-restore safety snapshot: " << safety << "\n";

        auto parent = fs::path(cfg_.data_dir).parent_path();
        auto name   = fs::path(cfg_.data_dir).filename().string();

        // Move current data aside
        std::string staging = cfg_.data_dir + "_pre_restore_" +
            std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
        if (fs::exists(cfg_.data_dir))
            fs::rename(cfg_.data_dir, staging);

        fs::create_directories(parent);

        std::ostringstream cmd;
        cmd << "tar -xzf \"" << path << "\" -C \""
            << parent.string() << "\" 2>/dev/null";
        int rc = std::system(cmd.str().c_str());
        if (rc != 0) {
            // Roll back
            if (fs::exists(staging))
                fs::rename(staging, cfg_.data_dir);
            throw std::runtime_error("Restore failed (tar rc=" +
                                     std::to_string(rc) + ")");
        }

        json restore_log = {
            {"restored_from", snapshot_name},
            {"restored_at",
                std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count()},
            {"safety_snapshot", safety},
            {"pre_restore_backup", staging},
            {"message", "Restore complete — restart tidevec-server to reload WAL"},
        };
        std::ofstream rf(cfg_.backup_dir + "/last_restore.json");
        rf << restore_log.dump(2);

        return "Restored from " + snapshot_name +
               " (safety snapshot: " + safety + "). Restart server.";
    }

private:
    void _loop() {
        while (running_) {
            try {
                auto name = _create_snapshot();
                std::cout << "[backup] Snapshot created: " << name << "\n";
                _prune_old();
            } catch (const std::exception& e) {
                std::cerr << "[backup] Error: " << e.what() << "\n";
            }
            for (int h = 0; h < cfg_.interval_hours * 3600 && running_; ++h)
                std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }

    std::string _create_snapshot() {
        auto ts = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        std::string name = "tidevec_" + std::to_string(ts) + ".tar.gz";
        std::string path = cfg_.backup_dir + "/" + name;

        std::ostringstream cmd;
        cmd << "tar -czf \"" << path << "\" -C \""
            << fs::path(cfg_.data_dir).parent_path().string()
            << "\" \"" << fs::path(cfg_.data_dir).filename().string()
            << "\" 2>/dev/null";
        int rc = std::system(cmd.str().c_str());
        if (rc != 0)
            throw std::runtime_error("tar snapshot failed (rc=" +
                                     std::to_string(rc) + ")");

        if (!cfg_.s3_uri.empty()) {
            std::string s3cmd = "aws s3 cp \"" + path + "\" \"" +
                cfg_.s3_uri + name + "\" 2>/dev/null";
            std::system(s3cmd.c_str());
        }
        if (!cfg_.gcs_uri.empty()) {
            std::string gscmd = "gsutil cp \"" + path + "\" \"" +
                cfg_.gcs_uri + name + "\" 2>/dev/null";
            std::system(gscmd.c_str());
        }

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

    void _prune_old() {
        auto snaps = list_snapshots();
        if (static_cast<int>(snaps.size()) <= cfg_.retention_count) return;
        int to_delete = static_cast<int>(snaps.size()) - cfg_.retention_count;
        for (int i = 0; i < to_delete; ++i)
            fs::remove(cfg_.backup_dir + "/" + snaps[i]);
    }

    BackupConfig cfg_;
    std::atomic<bool> running_{false};
    std::thread thread_;
};

} // namespace tidevec
