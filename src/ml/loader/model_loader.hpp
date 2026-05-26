#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Daydream Handler – Model Loader
// Discovers and loads model artifacts from the models/ directory tree.
//
// Supported artifact formats (native C++ — no Python runtime needed):
//   *.safetensors   → SafeTensorsModel  (pure C++ weight loader + MLP runner)
//   *.onnx          → OnnxModel         (ONNX Runtime C++ API, USE_ONNXRUNTIME)
//   *_manifest.json → manifest-driven multi-tensor model loading
//
// Python-only formats (skipped at load time, logged as a warning):
//   *.joblib, *.keras, *.h5   — require scikit-learn / Keras Python runtime
//
// Discovery conventions (from models/ directory layout):
//   models/layer0_model_l0/layer0.safetensors          → "layer0_merged"
//   models/layer0_model_l0/grand_unified_model.safetensors → "grand_unified"
//   models/layer0_model_l0/grand_unified_manifest.json → architecture for grand_unified
//   models/advanced_model/model_merged.safetensors     → "model_merged"
//   models/advanced_model/layer0_rebuild/layer0_rebuild.onnx → "layer0_rebuild"
//
// Artifacts skipped (training artifacts, not inference models):
//   *replay_buffer*  *snapshot_epoch*  *train_advanced_weights*  *training_log*
// ─────────────────────────────────────────────────────────────────────────────
#include "../inference/inference_engine.hpp"
#include "../../logging/logger.hpp"
#include "../../logging/telemetry.hpp"
#include <string>
#include <memory>
#include <vector>
#include <filesystem>
#include <unordered_map>

namespace dh::ml::loader {

// ── Model manifest (parsed from *_manifest.json) ─────────────────────────────
struct LayerSpec {
    std::string type;         // "linear", "relu", "softmax"
    int64_t     in_features = 0;
    int64_t     out_features = 0;
    std::string weight_key;   // tensor name in the safetensors file
    std::string bias_key;
};

struct ModelManifest {
    std::string              model_key;
    std::string              version;
    std::string              description;
    std::vector<LayerSpec>   layers;
    std::string              weights_file;  // relative path to .safetensors
    bool                     valid = false;
};

// ── ModelLoader ───────────────────────────────────────────────────────────────
class ModelLoader {
public:
    explicit ModelLoader(std::string model_base_dir = "models");

    // ── Primary entry point ───────────────────────────────────────────────────
    /// Walk model_base_dir, discover all artifacts, register them.
    /// Returns number of successfully loaded models.
    int discover_and_load(ModelRegistry& registry);

    // ── Individual loaders ────────────────────────────────────────────────────
    /// Load a safetensors model by explicit key and path.
    std::shared_ptr<IModel> load_safetensors(const std::string& key,
                                              const std::string& path);

    /// Load an ONNX model by explicit key and path (requires USE_ONNXRUNTIME).
    std::shared_ptr<IModel> load_onnx(const std::string& key,
                                       const std::string& path);

    /// Load a model described by a manifest JSON file.
    std::shared_ptr<IModel> load_from_manifest(const std::string& manifest_path);

    // ── Training-artifact import ──────────────────────────────────────────────
    /// Parse a training_log.csv and emit all metrics into the Telemetry system.
    /// CSV format expected: epoch,loss,accuracy,... (with header row)
    void import_training_log(const std::string& csv_path);

    /// Parse training telemetry JSONL and replay scalars into Telemetry.
    void import_telemetry_jsonl(const std::string& jsonl_path);

private:
    // Determine the model key from a file path using naming conventions
    std::string key_from_path(const std::filesystem::path& p) const;

    // True if a filename represents a training artifact (not an inference model)
    bool is_training_artifact(const std::filesystem::path& p) const;

    // True if a filename is a Python-only format
    bool is_python_only(const std::filesystem::path& p) const;

    // Parse a manifest JSON file
    ModelManifest parse_manifest(const std::string& path) const;

    std::string m_base_dir;
    std::shared_ptr<spdlog::logger> m_log;
};

} // namespace dh::ml::loader
