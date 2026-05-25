#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Daydream Handler – Checkpoint Manager
// Saves/loads safetensors-compatible JSON snapshots of engine state.
// Every checkpoint includes: step, P&L, positions, hparams, telemetry.
// ─────────────────────────────────────────────────────────────────────────────
#include <nlohmann/json.hpp>
#include <string>
#include <filesystem>
#include <fstream>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <iomanip>
#include <sstream>

namespace dh::storage {

using json = nlohmann::json;

class CheckpointManager {
public:
    explicit CheckpointManager(std::string dir = "checkpoints",
                                int max_keep = 5)
        : m_dir(std::move(dir)), m_max_keep(max_keep)
    {
        std::filesystem::create_directories(m_dir);
    }

    /// Save a checkpoint JSON. Returns the path written.
    std::string save(const json& state, int64_t step)
    {
        std::string path = checkpoint_path(step);
        std::ofstream f(path, std::ios::trunc);
        if (!f.is_open())
            throw std::runtime_error("Cannot write checkpoint: " + path);

        // Embed metadata header (safetensors-compatible convention)
        json doc = {
            {"__version__", "1.0"},
            {"__type__",    "daydream_checkpoint"},
            {"step",        step},
            {"state",       state}
        };
        f << doc.dump(2) << "\n";
        f.flush();

        rotate_old_checkpoints();
        return path;
    }

    /// Load the most recent checkpoint.
    json load_latest() const
    {
        auto files = list_checkpoints();
        if (files.empty())
            throw std::runtime_error("No checkpoints found in " + m_dir);
        return load_file(files.back());
    }

    /// Load a checkpoint at a specific step.
    json load(int64_t step) const
    {
        return load_file(checkpoint_path(step));
    }

    /// List all checkpoint paths sorted ascending by step.
    std::vector<std::string> list_checkpoints() const
    {
        std::vector<std::string> files;
        if (!std::filesystem::exists(m_dir)) return files;
        for (auto& e : std::filesystem::directory_iterator(m_dir)) {
            if (e.path().extension() == ".json"
                && e.path().stem().string().rfind("ckpt_", 0) == 0)
                files.push_back(e.path().string());
        }
        std::sort(files.begin(), files.end());
        return files;
    }

private:
    std::string checkpoint_path(int64_t step) const
    {
        std::ostringstream ss;
        ss << m_dir << "/ckpt_" << std::setw(10) << std::setfill('0')
           << step << ".json";
        return ss.str();
    }

    json load_file(const std::string& path) const
    {
        std::ifstream f(path);
        if (!f.is_open())
            throw std::runtime_error("Cannot open checkpoint: " + path);
        json doc;
        f >> doc;
        return doc;
    }

    void rotate_old_checkpoints()
    {
        auto files = list_checkpoints();
        while (static_cast<int>(files.size()) > m_max_keep) {
            std::filesystem::remove(files.front());
            files.erase(files.begin());
        }
    }

    std::string m_dir;
    int         m_max_keep;
};

} // namespace dh::storage
