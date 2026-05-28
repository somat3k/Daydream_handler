#pragma once

#include "../logging/logger.hpp"
#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace dh::storage {

using json = nlohmann::json;

class SafetensorsLoader {
public:
    explicit SafetensorsLoader(std::string path)
        : m_path(std::move(path))
    {
        parse();
    }

    const json& metadata() const { return m_header; }

    std::vector<float> load_tensor(const std::string& name) const
    {
        auto log = dh::logging::get("safetensors");
        auto it = m_tensors.find(name);
        if (it == m_tensors.end()) {
            log->error("Tensor '{}' not found in {}", name, m_path);
            throw std::runtime_error("Tensor not found: " + name);
        }

        const auto& info = it->second;
        if (info.dtype != "F32" && info.dtype != "I32") {
            log->error("Unsupported dtype '{}' for tensor '{}'", info.dtype, name);
            throw std::runtime_error("Unsupported safetensors dtype: " + info.dtype);
        }

        std::ifstream f(m_path, std::ios::binary);
        if (!f.is_open()) {
            throw std::runtime_error("Cannot open safetensors file: " + m_path);
        }

        const auto byte_count = info.end - info.begin;

        f.seekg(static_cast<std::streamoff>(m_data_start + info.begin), std::ios::beg);

        if (info.dtype == "I32") {
            if (byte_count % sizeof(int32_t) != 0) {
                throw std::runtime_error("Tensor byte count not aligned to int32 size: " + name);
            }
            const size_t n = byte_count / sizeof(int32_t);
            std::vector<int32_t> raw(n);
            f.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(byte_count));
            if (!f.good()) {
                throw std::runtime_error("Failed to read tensor bytes for: " + name);
            }
            std::vector<float> out(n);
            for (size_t i = 0; i < n; ++i)
                out[i] = static_cast<float>(raw[i]);
            return out;
        }

        if (byte_count % sizeof(float) != 0) {
            throw std::runtime_error("Tensor byte count not aligned to float size: " + name);
        }

        std::vector<float> out(byte_count / sizeof(float));
        f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(byte_count));
        if (!f.good()) {
            throw std::runtime_error("Failed to read tensor bytes for: " + name);
        }
        return out;
    }

private:
    struct TensorInfo {
        std::string dtype;
        uint64_t begin = 0;
        uint64_t end = 0;
    };

    void parse()
    {
        auto log = dh::logging::get("safetensors");
        if (!std::filesystem::exists(m_path)) {
            throw std::runtime_error("Safetensors file does not exist: " + m_path);
        }

        std::ifstream f(m_path, std::ios::binary);
        if (!f.is_open()) {
            throw std::runtime_error("Cannot open safetensors file: " + m_path);
        }

        uint64_t header_len = 0;
        f.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
        if (!f.good()) {
            throw std::runtime_error("Cannot read safetensors header length: " + m_path);
        }

        std::string header_text(header_len, '\0');
        f.read(header_text.data(), static_cast<std::streamsize>(header_len));
        if (!f.good()) {
            throw std::runtime_error("Cannot read safetensors header JSON: " + m_path);
        }

        try {
            m_header = json::parse(header_text);
        } catch (const std::exception& e) {
            log->error("Invalid safetensors header JSON in {}: {}", m_path, e.what());
            throw;
        }

        m_data_start = sizeof(header_len) + header_len;

        for (auto it = m_header.begin(); it != m_header.end(); ++it) {
            if (it.key() == "__metadata__") continue;
            if (!it->is_object()) continue;
            TensorInfo info;
            info.dtype = it->value("dtype", "");
            auto offsets = it->at("data_offsets");
            info.begin = offsets.at(0).get<uint64_t>();
            info.end = offsets.at(1).get<uint64_t>();
            m_tensors.emplace(it.key(), info);
        }

        log->info("Parsed safetensors: {} tensors from {}", m_tensors.size(), m_path);
    }

    std::string m_path;
    uint64_t m_data_start = 0;
    json m_header;
    std::unordered_map<std::string, TensorInfo> m_tensors;
};

} // namespace dh::storage
