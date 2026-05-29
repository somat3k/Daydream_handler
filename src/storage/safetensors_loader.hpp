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
        const auto& info = find_tensor(name, log);

        if (info.dtype == "I32") {
            const auto raw = load_tensor_i32(name);
            std::vector<float> out(raw.size());
            for (std::size_t i = 0; i < raw.size(); ++i)
                out[i] = static_cast<float>(raw[i]);
            return out;
        }

        if (info.dtype != "F32") {
            log->error("Unsupported dtype '{}' for tensor '{}'", info.dtype, name);
            throw std::runtime_error("Unsupported safetensors dtype: " + info.dtype);
        }

        const auto byte_count = info.end - info.begin;
        if (byte_count % sizeof(float) != 0) {
            throw std::runtime_error("Tensor byte count not aligned to float size: " + name);
        }

        std::vector<float> out(byte_count / sizeof(float));
        read_bytes(info.begin, reinterpret_cast<char*>(out.data()),
                   static_cast<std::streamsize>(byte_count), name);
        return out;
    }

    std::vector<int32_t> load_tensor_i32(const std::string& name) const
    {
        auto log = dh::logging::get("safetensors");
        const auto& info = find_tensor(name, log);

        if (info.dtype != "I32") {
            log->error("Expected I32 dtype for tensor '{}', got '{}'", name, info.dtype);
            throw std::runtime_error("Expected I32 dtype for tensor: " + name);
        }

        const auto byte_count = info.end - info.begin;
        if (byte_count % sizeof(int32_t) != 0) {
            throw std::runtime_error("Tensor byte count not aligned to int32 size: " + name);
        }

        std::vector<int32_t> out(byte_count / sizeof(int32_t));
        read_bytes(info.begin, reinterpret_cast<char*>(out.data()),
                   static_cast<std::streamsize>(byte_count), name);
        return out;
    }

private:
    struct TensorInfo {
        std::string dtype;
        uint64_t begin = 0;
        uint64_t end = 0;
    };

    const TensorInfo& find_tensor(
        const std::string& name,
        const std::shared_ptr<spdlog::logger>& log) const
    {
        auto it = m_tensors.find(name);
        if (it == m_tensors.end()) {
            log->error("Tensor '{}' not found in {}", name, m_path);
            throw std::runtime_error("Tensor not found: " + name);
        }
        return it->second;
    }

    void read_bytes(uint64_t offset, char* dest, std::streamsize size,
                    const std::string& name) const
    {
        std::ifstream f(m_path, std::ios::binary);
        if (!f.is_open()) {
            throw std::runtime_error("Cannot open safetensors file: " + m_path);
        }
        f.seekg(static_cast<std::streamoff>(m_data_start + offset), std::ios::beg);
        f.read(dest, size);
        if (!f.good()) {
            throw std::runtime_error("Failed to read tensor bytes for: " + name);
        }
    }

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
