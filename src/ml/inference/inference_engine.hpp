#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Daydream Handler – ML Inference Engine
// Abstract interface for running HuggingFace-style models.
// Backends (in priority order):
//   1. OnnxModel       – ONNX Runtime C++ API (USE_ONNXRUNTIME, .onnx files)
//   2. SafeTensorsModel – Pure C++ MLP runner loading .safetensors weights
//   3. TorchModel      – LibTorch TorchScript (USE_LIBTORCH, .pt files)
//   4. StubModel       – Heuristic fallback for CI / offline environments
//
// Models: Layer0, Grand Unified (from msomatothing/layer0 on HuggingFace)
//   Artifacts: models/layer0_model_l0/, models/advanced_model/layer0_rebuild/
// ─────────────────────────────────────────────────────────────────────────────
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <functional>
#include <nlohmann/json.hpp>

namespace dh::ml {

using json = nlohmann::json;

// ── Inference input/output ─────────────────────────────────────────────────
struct ModelInput {
    std::vector<float>  features;        // flattened feature tensor
    std::vector<int64_t> shape;          // [batch, seq, features]
    std::string          symbol;
    int64_t              bar_index = 0;
};

struct ModelOutput {
    std::vector<float>  logits;          // raw model outputs
    std::vector<float>  probabilities;   // softmax'd probabilities
    int                 predicted_class; // argmax class
    float               confidence;      // max probability
    std::string         label;           // human-readable
    json                metadata;
};

// ── Base model interface ──────────────────────────────────────────────────────
class IModel {
public:
    virtual ~IModel() = default;

    virtual std::string name() const = 0;
    virtual std::string version() const = 0;

    /// Load model from path (TorchScript .pt or safetensors directory).
    virtual bool load(const std::string& model_path) = 0;
    virtual bool is_loaded() const = 0;

    /// Run forward pass.
    virtual ModelOutput infer(const ModelInput& input) = 0;

    /// Batch inference.
    virtual std::vector<ModelOutput> infer_batch(
        const std::vector<ModelInput>& inputs);

    /// Export model hparams for telemetry.
    virtual json hparams() const = 0;
};

// ── Stub (heuristic) backend for offline / CI environments ───────────────────
class StubModel : public IModel {
public:
    explicit StubModel(std::string name,
                       std::string version = "0.0.1-stub",
                       int num_classes = 3)
        : m_name(std::move(name))
        , m_version(std::move(version))
        , m_num_classes(num_classes)
    {}

    std::string name() const override { return m_name; }
    std::string version() const override { return m_version; }
    bool load(const std::string&) override { m_loaded = true; return true; }
    bool is_loaded() const override { return m_loaded; }

    ModelOutput infer(const ModelInput& input) override;

    json hparams() const override {
        return {{"model", m_name}, {"version", m_version},
                {"backend", "stub"}, {"num_classes", m_num_classes}};
    }

private:
    std::string m_name;
    std::string m_version;
    int         m_num_classes;
    bool        m_loaded = false;
};

// ── LibTorch TorchScript backend ──────────────────────────────────────────────
#ifdef USE_LIBTORCH
class TorchModel : public IModel {
public:
    explicit TorchModel(std::string name, std::string version = "1.0.0")
        : m_name(std::move(name)), m_version(std::move(version)) {}

    std::string name() const override { return m_name; }
    std::string version() const override { return m_version; }
    bool load(const std::string& model_path) override;
    bool is_loaded() const override { return m_loaded; }
    ModelOutput infer(const ModelInput& input) override;
    json hparams() const override;

private:
    std::string m_name;
    std::string m_version;
    bool        m_loaded = false;
    // torch::jit::script::Module m_module;  // uncomment when linking LibTorch
};
#endif

// ── SafeTensors MLP backend (pure C++, no Python runtime needed) ──────────────
// Loads float32 weight tensors from a .safetensors file and runs a
// feed-forward MLP using the GEMM/ReLU/Softmax operations already present
// in the FeatureComposer module.
//
// Layer auto-detection heuristics (applied in order):
//   Pattern A – numbered: "0.weight"/"0.bias", "1.weight"/"1.bias", ...
//   Pattern B – named:    "fc.weight"/"fc.bias"
//   Pattern C – layered:  "layers.0.weight"/"layers.0.bias", ...
//   Pattern D – classifier: "classifier.weight"/"classifier.bias"
//   Fallback  – stub heuristic when no weight matrices can be detected
//
// The last layer's output is softmaxed to produce class probabilities.
class SafeTensorsModel : public IModel {
public:
    explicit SafeTensorsModel(std::string key,
                               std::string version   = "1.0.0-st",
                               int         num_classes = 3)
        : m_key(std::move(key))
        , m_version(std::move(version))
        , m_num_classes(num_classes)
    {}

    std::string name() const override { return m_key; }
    std::string version() const override { return m_version; }

    bool load(const std::string& model_path) override;
    bool is_loaded() const override { return m_loaded; }

    ModelOutput infer(const ModelInput& input) override;

    json hparams() const override {
        return {{"model", m_key}, {"version", m_version},
                {"backend", "safetensors"}, {"num_classes", m_num_classes},
                {"num_layers", static_cast<int>(m_weights.size())},
                {"path", m_path}};
    }

    /// Inject a pre-built layer stack (used by manifest-driven loader)
    struct LinearLayer {
        std::vector<float> W;      // [out × in] row-major
        std::vector<float> b;      // [out]
        int64_t            in_features  = 0;
        int64_t            out_features = 0;
        bool               relu_after   = true;
    };
    void set_layers(std::vector<LinearLayer> layers) { m_weights = std::move(layers); }

private:
    // Run MLP forward pass: input → [L0 → ReLU]* → L(last) → softmax
    ModelOutput run_mlp(const std::vector<float>& features) const;

    // Attempt to auto-detect layers from the loaded tensor name map
    bool auto_detect_layers(
        const std::unordered_map<std::string, std::vector<float>>& tensors,
        const std::unordered_map<std::string, std::vector<int64_t>>& shapes);

    std::string                  m_key;
    std::string                  m_version;
    int                          m_num_classes;
    bool                         m_loaded = false;
    std::string                  m_path;
    std::vector<LinearLayer>     m_weights;
};

// ── ONNX Runtime backend (native C++ API, requires USE_ONNXRUNTIME) ───────────
#ifdef USE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>

class OnnxModel : public IModel {
public:
    explicit OnnxModel(std::string key, std::string version = "1.0.0-onnx")
        : m_key(std::move(key)), m_version(std::move(version))
        , m_env(ORT_LOGGING_LEVEL_WARNING, "daydream_onnx")
    {}

    std::string name() const override { return m_key; }
    std::string version() const override { return m_version; }

    bool load(const std::string& model_path) override;
    bool is_loaded() const override { return m_loaded; }

    ModelOutput infer(const ModelInput& input) override;

    json hparams() const override {
        return {{"model", m_key}, {"version", m_version},
                {"backend", "onnxruntime"}, {"path", m_path},
                {"input_dim", m_input_dim}, {"num_classes", m_num_classes}};
    }

private:
    std::string                      m_key;
    std::string                      m_version;
    std::string                      m_path;
    bool                             m_loaded      = false;
    int64_t                          m_input_dim   = 0;
    int64_t                          m_num_classes = 3;
    Ort::Env                         m_env;
    std::unique_ptr<Ort::Session>    m_session;
    Ort::SessionOptions              m_session_opts;
    std::vector<std::string>         m_input_names_storage;
    std::vector<std::string>         m_output_names_storage;
    std::vector<const char*>         m_input_names;
    std::vector<const char*>         m_output_names;
    std::vector<std::vector<int64_t>> m_input_shapes;
};
#endif // USE_ONNXRUNTIME

// ── Model Registry ─────────────────────────────────────────────────────────
class ModelRegistry {
public:
    static ModelRegistry& instance();

    /// Register a model under a symbolic name.
    void register_model(const std::string& key,
                        std::shared_ptr<IModel> model);

    /// Retrieve a model by key.
    std::shared_ptr<IModel> get(const std::string& key) const;

    /// List all registered model keys.
    std::vector<std::string> keys() const;

    /// Load all registered models from a base directory.
    void load_all(const std::string& base_dir);

private:
    ModelRegistry() = default;
    std::unordered_map<std::string, std::shared_ptr<IModel>> m_models;
};

// ── Named model factory ───────────────────────────────────────────────────────
namespace models {

inline std::shared_ptr<IModel> make_layer0()
{
#ifdef USE_ONNXRUNTIME
    // OnnxModel preferred when ONNX Runtime is available (layer0_rebuild.onnx)
    return std::make_shared<OnnxModel>("layer0_merged", "1.0.0-onnx");
#else
    // SafeTensors MLP loader – loads weights from layer0.safetensors
    return std::make_shared<SafeTensorsModel>("layer0_merged", "1.0.0-st", 3);
#endif
}

inline std::shared_ptr<IModel> make_grand_unified()
{
#ifdef USE_ONNXRUNTIME
    return std::make_shared<OnnxModel>("grand_unified", "1.0.0-onnx");
#else
    return std::make_shared<SafeTensorsModel>("grand_unified", "1.0.0-st", 5);
#endif
}

inline std::shared_ptr<IModel> make_model_merged()
{
    // advanced_model/model_merged.safetensors (merged ensemble)
    return std::make_shared<SafeTensorsModel>("model_merged", "1.0.0-st", 3);
}

inline std::shared_ptr<IModel> make_layer0_rebuild()
{
#ifdef USE_ONNXRUNTIME
    return std::make_shared<OnnxModel>("layer0_rebuild", "1.0.0-onnx");
#else
    return std::make_shared<SafeTensorsModel>("layer0_rebuild", "1.0.0-st", 3);
#endif
}

} // namespace models

} // namespace dh::ml
