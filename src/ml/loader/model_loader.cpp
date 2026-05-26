#include "model_loader.hpp"
#include "safetensors_reader.hpp"
#include "../inference/inference_engine.hpp"
#include "../../logging/logger.hpp"
#include "../../logging/telemetry.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>

namespace fs = std::filesystem;
using json   = nlohmann::json;

namespace dh::ml::loader {

// ── Name-to-key mappings from the models/ directory structure ─────────────────
//
// models/layer0_model_l0/layer0.safetensors          → "layer0_merged"
// models/layer0_model_l0/grand_unified_model.st      → "grand_unified"
// models/advanced_model/model_merged.safetensors     → "model_merged"
// models/advanced_model/layer0_rebuild/layer0_rebuild.onnx → "layer0_rebuild"
// Other *.safetensors whose stem == key              → stem-based key
static const std::unordered_map<std::string, std::string> KNOWN_STEMS = {
    {"layer0",              "layer0_merged"},
    {"layer0_merged",       "layer0_merged"},
    {"grand_unified_model", "grand_unified"},
    {"grand_unified",       "grand_unified"},
    {"model_merged",        "model_merged"},
    {"layer0_rebuild",      "layer0_rebuild"},
};

// ── Training artifacts to skip ────────────────────────────────────────────────
static bool contains_any(const std::string& s,
                          const std::vector<std::string>& needles) {
    for (auto& n : needles)
        if (s.find(n) != std::string::npos) return true;
    return false;
}

// ── ModelLoader ───────────────────────────────────────────────────────────────
ModelLoader::ModelLoader(std::string model_base_dir)
    : m_base_dir(std::move(model_base_dir))
    , m_log(dh::logging::get("model.loader"))
{}

// ── is_training_artifact ──────────────────────────────────────────────────────
bool ModelLoader::is_training_artifact(const fs::path& p) const
{
    std::string stem = p.stem().string();
    return contains_any(stem, {
        "replay_buffer", "snapshot_epoch", "train_advanced_weights",
        "training_log", "snapshot-final"
    });
}

// ── is_python_only ────────────────────────────────────────────────────────────
bool ModelLoader::is_python_only(const fs::path& p) const
{
    std::string ext = p.extension().string();
    return ext == ".joblib" || ext == ".keras" || ext == ".h5" || ext == ".pkl";
}

// ── key_from_path ─────────────────────────────────────────────────────────────
std::string ModelLoader::key_from_path(const fs::path& p) const
{
    std::string stem = p.stem().string();
    auto it = KNOWN_STEMS.find(stem);
    if (it != KNOWN_STEMS.end()) return it->second;
    // Fall back to stem as key
    return stem;
}

// ── parse_manifest ────────────────────────────────────────────────────────────
ModelManifest ModelLoader::parse_manifest(const std::string& path) const
{
    ModelManifest manifest;
    std::ifstream f(path);
    if (!f.is_open()) {
        m_log->warn("Cannot open manifest: {}", path);
        return manifest;
    }

    json j;
    try { f >> j; }
    catch (const json::exception& e) {
        m_log->error("JSON parse error in manifest {}: {}", path, e.what());
        return manifest;
    }

    manifest.model_key    = j.value("model_key", "unknown");
    manifest.version      = j.value("version",   "1.0.0");
    manifest.description  = j.value("description", "");
    manifest.weights_file = j.value("weights_file", "");

    // Parse layer specifications
    if (j.contains("layers") && j.at("layers").is_array()) {
        for (auto& lj : j.at("layers")) {
            LayerSpec ls;
            ls.type         = lj.value("type", "linear");
            ls.in_features  = lj.value("in_features",  0);
            ls.out_features = lj.value("out_features", 0);
            ls.weight_key   = lj.value("weight_key",   "");
            ls.bias_key     = lj.value("bias_key",     "");
            manifest.layers.push_back(ls);
        }
    }

    manifest.valid = true;
    return manifest;
}

// ── load_safetensors ──────────────────────────────────────────────────────────
std::shared_ptr<IModel> ModelLoader::load_safetensors(const std::string& key,
                                                        const std::string& path)
{
    m_log->debug("Creating SafeTensorsModel '{}' from '{}'", key, path);

    // Determine num_classes from key
    int num_classes = 3; // default: HOLD/BUY/SELL
    if (key == "grand_unified") num_classes = 5;

    auto model = std::make_shared<SafeTensorsModel>(key, "1.0.0-st", num_classes);
    bool ok = model->load(path);

    if (!ok) {
        m_log->warn("SafeTensors load failed for '{}' – stub active", key);
    }
    return model;
}

// ── load_onnx ─────────────────────────────────────────────────────────────────
std::shared_ptr<IModel> ModelLoader::load_onnx(const std::string& key,
                                                 const std::string& path)
{
#ifdef USE_ONNXRUNTIME
    m_log->debug("Creating OnnxModel '{}' from '{}'", key, path);
    auto model = std::make_shared<OnnxModel>(key, "1.0.0-onnx");
    bool ok = model->load(path);
    if (!ok) {
        m_log->warn("ONNX load failed for '{}' – falling back to SafeTensors", key);
        return load_safetensors(key, fs::path(path).replace_extension(".safetensors").string());
    }
    return model;
#else
    m_log->debug("USE_ONNXRUNTIME not set – loading '{}' as SafeTensors instead", key);
    return load_safetensors(key, fs::path(path).replace_extension(".safetensors").string());
#endif
}

// ── load_from_manifest ────────────────────────────────────────────────────────
std::shared_ptr<IModel> ModelLoader::load_from_manifest(const std::string& manifest_path)
{
    m_log->info("Parsing manifest: {}", manifest_path);
    auto mf = parse_manifest(manifest_path);
    if (!mf.valid) {
        m_log->warn("Invalid manifest: {}", manifest_path);
        return nullptr;
    }

    m_log->info("Manifest '{}' v{}: {} layers, weights='{}'",
                mf.model_key, mf.version, mf.layers.size(), mf.weights_file);

    // Resolve weights path relative to manifest directory
    fs::path manifest_dir = fs::path(manifest_path).parent_path();
    fs::path weights_path = mf.weights_file.empty()
        ? manifest_dir / (mf.model_key + ".safetensors")
        : (fs::path(mf.weights_file).is_absolute()
            ? fs::path(mf.weights_file)
            : manifest_dir / mf.weights_file);

    int num_classes = 3;
    for (auto& ls : mf.layers)
        if (ls.type == "linear" && ls.out_features > 0)
            num_classes = static_cast<int>(ls.out_features);

    auto model = std::make_shared<SafeTensorsModel>(
        mf.model_key, mf.version, num_classes);

    // Build layer stack from manifest
    if (!mf.layers.empty()) {
        SafeTensorsFile sf;
        if (fs::exists(weights_path)) {
            sf = SafeTensorsFile::parse_header(weights_path.string());
            m_log->info("Manifest weights loaded: {}", sf.summary());
        }

        std::vector<SafeTensorsModel::LinearLayer> layer_stack;
        bool has_weights = sf.parsed;

        for (size_t i = 0; i < mf.layers.size(); ++i) {
            auto& ls = mf.layers[i];
            if (ls.type != "linear") continue;

            SafeTensorsModel::LinearLayer ll;
            ll.in_features  = ls.in_features;
            ll.out_features = ls.out_features;
            ll.relu_after   = (i + 1 < mf.layers.size());

            if (has_weights && !ls.weight_key.empty()) {
                ll.W = sf.read_tensor_f32(ls.weight_key);
                if (!ls.bias_key.empty())
                    ll.b = sf.read_tensor_f32(ls.bias_key);
            }
            layer_stack.push_back(std::move(ll));
        }

        if (!layer_stack.empty()) {
            model->set_layers(std::move(layer_stack));
            m_log->info("Manifest injected {} layers into '{}'",
                        layer_stack.size(), mf.model_key);
        }
    }

    bool ok = model->load(weights_path.string());
    if (!ok)
        m_log->warn("Manifest model '{}' weights not found – layers may be empty",
                    mf.model_key);

    return model;
}

// ── discover_and_load ─────────────────────────────────────────────────────────
int ModelLoader::discover_and_load(ModelRegistry& registry)
{
    m_log->info("Discovering model artifacts in '{}'", m_base_dir);

    if (!fs::exists(m_base_dir)) {
        m_log->warn("Model directory '{}' does not exist – "
                    "all models will use stub inference", m_base_dir);
        // Still register primary model stubs so the registry is always valid
        auto ensure_stub = [&](const std::string& key, int n_classes) {
            try { registry.get(key); }
            catch (const std::out_of_range&) {
                auto stub = std::make_shared<StubModel>(key, "0.0.1-stub", n_classes);
                stub->load("");
                registry.register_model(key, stub);
                telemetry::hparam("loader/model_" + key + "_backend", "stub");
            }
        };
        ensure_stub("layer0_merged",  3);
        ensure_stub("grand_unified",  5);
        return 0;
    }

    int loaded = 0;

    // Track the best artifact per key (ONNX > safetensors)
    // Priority: higher number = preferred
    std::unordered_map<std::string, int>         key_priority;
    std::unordered_map<std::string, std::string> key_path;
    std::unordered_map<std::string, std::string> key_manifest;

    // Walk the directory tree
    for (auto& entry : fs::recursive_directory_iterator(m_base_dir,
            fs::directory_options::skip_permission_denied))
    {
        if (!entry.is_regular_file()) continue;
        auto p = entry.path();

        // Skip Python-only formats
        if (is_python_only(p)) {
            m_log->debug("Skipping Python-only artifact: {}", p.string());
            continue;
        }

        // Skip known training artifacts
        if (is_training_artifact(p)) {
            m_log->debug("Skipping training artifact: {}", p.string());
            continue;
        }

        std::string ext = p.extension().string();

        if (ext == ".json" && p.stem().string().find("manifest") != std::string::npos) {
            // Manifest file: strip "_manifest" suffix from stem to get the model key
            std::string stem = p.stem().string();
            auto pos = stem.find("_manifest");
            if (pos != std::string::npos) stem = stem.substr(0, pos);

            auto it = KNOWN_STEMS.find(stem);
            std::string mkey = (it != KNOWN_STEMS.end()) ? it->second : stem;
            key_manifest[mkey] = p.string();
            m_log->debug("Found manifest for '{}': {}", mkey, p.string());
            continue;
        }

        if (ext == ".onnx") {
            std::string key = key_from_path(p);
            if (key_priority[key] < 2) {
                key_priority[key] = 2;
                key_path[key]     = p.string();
                m_log->debug("Found ONNX artifact for '{}': {}", key, p.string());
            }
            continue;
        }

        if (ext == ".safetensors") {
            std::string key = key_from_path(p);
            if (key_priority[key] < 1) {
                key_priority[key] = 1;
                key_path[key]     = p.string();
                m_log->debug("Found safetensors artifact for '{}': {}", key, p.string());
            }
            continue;
        }
    }

    // Apply manifests: if a key has both a manifest and an artifact, combine
    for (auto& [mkey, mpath] : key_manifest) {
        m_log->info("Loading manifest-described model '{}' from '{}'", mkey, mpath);
        auto model = load_from_manifest(mpath);
        if (model) {
            registry.register_model(mkey, model);
            ++loaded;
            telemetry::hparam("loader/model_" + mkey + "_backend", "manifest");
        }
    }

    // Load remaining artifacts that didn't have a manifest
    for (auto& [key, path] : key_path) {
        if (key_manifest.count(key)) continue; // already loaded via manifest

        m_log->info("Loading model '{}' from '{}'", key, path);
        std::shared_ptr<IModel> model;

        if (key_priority[key] == 2) {
            model = load_onnx(key, path);
        } else {
            model = load_safetensors(key, path);
        }

        if (model) {
            registry.register_model(key, model);
            ++loaded;
            telemetry::hparam("loader/model_" + key + "_backend",
                              model->hparams().value("backend", "unknown"));
            telemetry::hparam("loader/model_" + key + "_path", path);
        }
    }

    // Ensure the two primary models are always in the registry (stub if missing)
    auto ensure = [&](const std::string& key, int n_classes) {
        try { registry.get(key); }
        catch (const std::out_of_range&) {
            m_log->warn("Primary model '{}' not discovered – "
                        "registering stub fallback", key);
            auto stub = std::make_shared<StubModel>(key, "0.0.1-stub", n_classes);
            stub->load("");
            registry.register_model(key, stub);
            telemetry::hparam("loader/model_" + key + "_backend", "stub");
        }
    };
    ensure("layer0_merged",  3);
    ensure("grand_unified",  5);

    m_log->info("Model discovery complete: {} models loaded from '{}'. "
                "Registry size: {}",
                loaded, m_base_dir, registry.keys().size());

    // Emit registry summary to telemetry
    for (auto& k : registry.keys()) {
        auto m = registry.get(k);
        m_log->info("  Registry: '{}' v{} loaded={} backend={}",
                    k, m->version(), m->is_loaded(),
                    m->hparams().value("backend", "?"));
        telemetry::hparam("loader/model_" + k + "_loaded",
                          m->is_loaded() ? 1.0 : 0.0);
    }

    return loaded;
}

// ── import_training_log ───────────────────────────────────────────────────────
void ModelLoader::import_training_log(const std::string& csv_path)
{
    m_log->info("Importing training log from CSV: {}", csv_path);

    std::ifstream f(csv_path);
    if (!f.is_open()) {
        m_log->warn("Training log not found: {}", csv_path);
        return;
    }

    std::string header_line;
    if (!std::getline(f, header_line)) return;

    // Parse column names
    std::vector<std::string> cols;
    std::istringstream hss(header_line);
    std::string col;
    while (std::getline(hss, col, ',')) {
        // Trim whitespace
        col.erase(0, col.find_first_not_of(" \t\r"));
        col.erase(col.find_last_not_of(" \t\r") + 1);
        cols.push_back(col);
    }

    if (cols.empty()) { m_log->warn("Empty header in {}", csv_path); return; }

    // Identify epoch column
    size_t epoch_col = 0;
    for (size_t i = 0; i < cols.size(); ++i)
        if (cols[i] == "epoch" || cols[i] == "step") { epoch_col = i; break; }

    int64_t row_idx = 0;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::vector<double> vals;
        std::istringstream lss(line);
        std::string cell;
        while (std::getline(lss, cell, ',')) {
            try { vals.push_back(std::stod(cell)); }
            catch (...) { vals.push_back(0.0); }
        }

        int64_t step = row_idx;
        if (epoch_col < vals.size())
            step = static_cast<int64_t>(vals[epoch_col]);

        for (size_t c = 0; c < cols.size() && c < vals.size(); ++c) {
            if (c == epoch_col) continue;
            telemetry::scalar("train/" + cols[c], vals[c], step);
        }
        ++row_idx;
    }

    m_log->info("Imported {} training log rows from '{}'", row_idx, csv_path);
}

// ── import_telemetry_jsonl ────────────────────────────────────────────────────
void ModelLoader::import_telemetry_jsonl(const std::string& jsonl_path)
{
    m_log->info("Importing telemetry JSONL: {}", jsonl_path);

    std::ifstream f(jsonl_path);
    if (!f.is_open()) {
        m_log->warn("Telemetry JSONL not found: {}", jsonl_path);
        return;
    }

    int64_t row_idx = 0;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto j = json::parse(line, nullptr, false);
        if (j.is_discarded()) continue;

        // Expected format: {"tag": "...", "value": 1.23, "step": 42}
        // or {"scalars": {"tag1": val1, "tag2": val2}, "step": 42}
        if (j.contains("tag") && j.contains("value")) {
            std::string tag = j["tag"].get<std::string>();
            double      val = j["value"].get<double>();
            int64_t     step = j.value("step", row_idx);
            telemetry::scalar("imported/" + tag, val, step);
        } else if (j.contains("scalars") && j.at("scalars").is_object()) {
            int64_t step = j.value("step", row_idx);
            for (auto& [tag, val] : j.at("scalars").items())
                telemetry::scalar("imported/" + tag, val.get<double>(), step);
        }
        ++row_idx;
    }

    m_log->info("Imported {} telemetry rows from '{}'", row_idx, jsonl_path);
}

} // namespace dh::ml::loader
