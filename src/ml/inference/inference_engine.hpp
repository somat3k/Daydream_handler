#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Daydream Handler – ML Inference Engine
// Abstract interface for running HuggingFace-style models.
// Backends: LibTorch (TorchScript), stub (heuristic fallback).
// Models: Layer0, Grand Unified (from msomatothing/layer0 on HuggingFace)
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
#ifdef USE_LIBTORCH
    return std::make_shared<TorchModel>("layer0_merged", "1.0.0");
#else
    return std::make_shared<StubModel>("layer0_merged", "1.0.0-stub", 3);
#endif
}

inline std::shared_ptr<IModel> make_grand_unified()
{
#ifdef USE_LIBTORCH
    return std::make_shared<TorchModel>("grand_unified", "1.0.0");
#else
    return std::make_shared<StubModel>("grand_unified", "1.0.0-stub", 5);
#endif
}

} // namespace models

} // namespace dh::ml
