#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Daydream Handler – SafeTensors Binary Reader
// Header-only C++ parser for the safetensors weight format.
//
// Format specification:
//   Bytes  0- 7 : uint64_t (LE) – JSON header length
//   Bytes  8  – 8+N : UTF-8 JSON metadata header
//   Bytes  8+N onwards : raw tensor data (concatenated, no padding)
//
// JSON header structure:
//   {
//     "__metadata__": { "format": "pt", ... },
//     "tensor_name": {
//       "dtype":        "F32"|"F16"|"BF16"|"I32"|"I64"|"U8"|"BOOL",
//       "shape":        [d0, d1, ...],
//       "data_offsets": [start_byte, end_byte]  // within data region
//     }
//   }
// ─────────────────────────────────────────────────────────────────────────────
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <cstdint>
#include <stdexcept>
#include <algorithm>
#include <cstring>

namespace dh::ml::loader {

using json = nlohmann::json;

// ── Tensor descriptor ─────────────────────────────────────────────────────────
struct TensorInfo {
    std::string          dtype;         // "F32", "F16", "BF16", "I32", "I64", etc.
    std::vector<int64_t> shape;
    uint64_t             offset_start = 0;  // byte offset within data region
    uint64_t             offset_end   = 0;

    /// Total element count
    size_t numel() const {
        if (shape.empty()) return 0;
        size_t n = 1;
        for (auto d : shape) n *= static_cast<size_t>(d);
        return n;
    }

    /// Bytes per element for this dtype
    size_t element_bytes() const {
        if (dtype == "F32" || dtype == "I32") return 4;
        if (dtype == "F16" || dtype == "BF16") return 2;
        if (dtype == "F64" || dtype == "I64") return 8;
        if (dtype == "U8" || dtype == "I8" || dtype == "BOOL") return 1;
        return 4;
    }

    /// Total byte span of this tensor
    size_t byte_size() const { return offset_end - offset_start; }

    bool is_2d() const { return shape.size() == 2; }
    bool is_1d() const { return shape.size() == 1; }
};

// ── Parsed safetensors file ───────────────────────────────────────────────────
struct SafeTensorsFile {
    std::string                             path;
    bool                                    parsed = false;
    json                                    metadata;    ///< __metadata__ block
    std::unordered_map<std::string, TensorInfo> tensors; ///< name → descriptor
    uint64_t                                data_offset = 0; ///< byte where data region starts

    // ── Factory: parse header only (no weight data read) ─────────────────────
    static SafeTensorsFile parse_header(const std::string& filepath)
    {
        SafeTensorsFile sf;
        sf.path = filepath;

        std::ifstream f(filepath, std::ios::binary);
        if (!f.is_open()) return sf;

        // Read 8-byte header length
        uint64_t header_len = 0;
        f.read(reinterpret_cast<char*>(&header_len), 8);
        if (!f || header_len == 0 || header_len > 100'000'000ULL) return sf;

        // Read JSON header
        std::string header_str(static_cast<size_t>(header_len), '\0');
        f.read(&header_str[0], static_cast<std::streamsize>(header_len));
        if (!f) return sf;

        sf.data_offset = 8 + header_len;

        auto j = json::parse(header_str, nullptr, /*allow_exceptions=*/false);
        if (j.is_discarded()) return sf;

        if (j.contains("__metadata__"))
            sf.metadata = j.at("__metadata__");

        for (auto& [name, info] : j.items()) {
            if (name == "__metadata__") continue;
            if (!info.is_object()) continue;

            TensorInfo ti;
            ti.dtype = info.value("dtype", "F32");

            if (info.contains("shape")) {
                for (auto& d : info.at("shape"))
                    ti.shape.push_back(d.get<int64_t>());
            }
            if (info.contains("data_offsets") && info.at("data_offsets").size() >= 2) {
                ti.offset_start = info.at("data_offsets")[0].get<uint64_t>();
                ti.offset_end   = info.at("data_offsets")[1].get<uint64_t>();
            }
            sf.tensors[name] = std::move(ti);
        }

        sf.parsed = true;
        return sf;
    }

    // ── Read a single float32 tensor from file ────────────────────────────────
    // Handles F32 directly; converts F16/BF16 to F32 on the fly.
    std::vector<float> read_tensor_f32(const std::string& name) const
    {
        auto it = tensors.find(name);
        if (it == tensors.end()) return {};

        const TensorInfo& ti = it->second;
        size_t n = ti.numel();
        if (n == 0) return {};

        std::ifstream f(path, std::ios::binary);
        if (!f.is_open()) return {};

        uint64_t abs_offset = data_offset + ti.offset_start;
        f.seekg(static_cast<std::streamoff>(abs_offset), std::ios::beg);
        if (!f) return {};

        std::vector<float> out(n);

        if (ti.dtype == "F32") {
            f.read(reinterpret_cast<char*>(out.data()),
                   static_cast<std::streamsize>(n * 4));
            if (!f) return {};
        } else if (ti.dtype == "F16") {
            // 16-bit IEEE half → float32
            std::vector<uint16_t> raw(n);
            f.read(reinterpret_cast<char*>(raw.data()),
                   static_cast<std::streamsize>(n * 2));
            if (!f) return {};
            for (size_t i = 0; i < n; ++i)
                out[i] = half_to_float(raw[i]);
        } else if (ti.dtype == "BF16") {
            // bfloat16 → float32 (upper 16 bits are the exponent/mantissa)
            std::vector<uint16_t> raw(n);
            f.read(reinterpret_cast<char*>(raw.data()),
                   static_cast<std::streamsize>(n * 2));
            if (!f) return {};
            for (size_t i = 0; i < n; ++i)
                out[i] = bf16_to_float(raw[i]);
        } else if (ti.dtype == "F64") {
            std::vector<double> raw(n);
            f.read(reinterpret_cast<char*>(raw.data()),
                   static_cast<std::streamsize>(n * 8));
            if (!f) return {};
            for (size_t i = 0; i < n; ++i)
                out[i] = static_cast<float>(raw[i]);
        } else {
            // Unsupported dtype – return empty
            return {};
        }

        return out;
    }

    /// Tensor names sorted alphabetically
    std::vector<std::string> tensor_names() const {
        std::vector<std::string> names;
        names.reserve(tensors.size());
        for (auto& [k, _] : tensors) names.push_back(k);
        std::sort(names.begin(), names.end());
        return names;
    }

    /// Summary string for logging
    std::string summary() const {
        if (!parsed) return "[not parsed]";
        std::string s = "[" + std::to_string(tensors.size()) + " tensors]";
        if (metadata.contains("format"))
            s += " format=" + metadata.at("format").get<std::string>();
        return s;
    }

private:
    // IEEE 754 half-precision → single-precision
    static float half_to_float(uint16_t h) {
        uint32_t sign     = (h >> 15) & 0x1;
        uint32_t exponent = (h >> 10) & 0x1f;
        uint32_t mantissa = h & 0x3ff;
        uint32_t f;
        if (exponent == 0) {
            if (mantissa == 0) {
                f = sign << 31;
            } else {
                exponent = 1;
                while (!(mantissa & 0x400)) { mantissa <<= 1; --exponent; }
                mantissa &= 0x3ff;
                f = (sign << 31) | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
            }
        } else if (exponent == 31) {
            f = (sign << 31) | (0xff << 23) | (mantissa << 13);
        } else {
            f = (sign << 31) | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
        }
        float result;
        std::memcpy(&result, &f, 4);
        return result;
    }

    // bfloat16 → float32
    static float bf16_to_float(uint16_t b) {
        uint32_t f = static_cast<uint32_t>(b) << 16;
        float result;
        std::memcpy(&result, &f, 4);
        return result;
    }
};

} // namespace dh::ml::loader
