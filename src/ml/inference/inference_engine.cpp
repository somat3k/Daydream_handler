#include "inference_engine.hpp"
#include "../loader/safetensors_reader.hpp"
#include "../../logging/logger.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <filesystem>
#include <regex>

namespace dh::ml {

// ── IModel default batch ──────────────────────────────────────────────────────
std::vector<ModelOutput> IModel::infer_batch(
    const std::vector<ModelInput>& inputs)
{
    std::vector<ModelOutput> out;
    out.reserve(inputs.size());
    for (auto& in : inputs) out.push_back(infer(in));
    return out;
}

// ── StubModel ─────────────────────────────────────────────────────────────────
// Heuristic: trend = sign of mean(last 5 features) > 0 → BUY, < 0 → SELL.
ModelOutput StubModel::infer(const ModelInput& input)
{
    ModelOutput out;
    if (input.features.empty()) {
        out.predicted_class = 0;
        out.confidence      = 0.0f;
        out.label           = "HOLD";
        out.logits          = {0.33f, 0.33f, 0.34f};
        out.probabilities   = {0.33f, 0.33f, 0.34f};
        return out;
    }

    float sum = 0;
    size_t w  = std::min<size_t>(5, input.features.size());
    for (size_t i = input.features.size() - w;
         i < input.features.size(); ++i)
        sum += input.features[i];
    float mean = sum / static_cast<float>(w);

    // 3 classes: 0=HOLD, 1=BUY, 2=SELL
    std::vector<float> logits(static_cast<size_t>(m_num_classes), 0.0f);
    if (m_num_classes >= 3) {
        logits[0] = 0.5f;
        logits[1] = mean > 0 ? 1.0f : 0.2f;
        logits[2] = mean < 0 ? 1.0f : 0.2f;
    }

    // Softmax
    float max_l = *std::max_element(logits.begin(), logits.end());
    std::vector<float> probs(logits.size());
    float sum_exp = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_l);
        sum_exp += probs[i];
    }
    for (auto& p : probs) p /= sum_exp;

    int cls = static_cast<int>(
        std::max_element(probs.begin(), probs.end()) - probs.begin());

    out.logits        = logits;
    out.probabilities = probs;
    out.predicted_class = cls;
    out.confidence      = probs[static_cast<size_t>(cls)];

    static const char* LABELS[] = {"HOLD", "BUY", "SELL",
                                    "STRONG_BUY", "STRONG_SELL"};
    if (cls < static_cast<int>(sizeof(LABELS)/sizeof(LABELS[0])))
        out.label = LABELS[cls];
    else
        out.label = "HOLD";

    out.metadata = {{"backend","stub"}, {"bar_index", input.bar_index}};
    return out;
}

// ── SafeTensorsModel ──────────────────────────────────────────────────────────

namespace {

// Softmax helper
std::vector<float> softmax_vec(const std::vector<float>& v) {
    if (v.empty()) return {};
    float max_v = *std::max_element(v.begin(), v.end());
    std::vector<float> out(v.size());
    float sum = 0.f;
    for (size_t i = 0; i < v.size(); ++i) {
        out[i] = std::exp(v[i] - max_v);
        sum += out[i];
    }
    if (sum > 0.f) for (auto& x : out) x /= sum;
    return out;
}

// Dense GEMM: out[i] = sum_j(W[i*in + j] * x[j]) + b[i]  (in → out)
std::vector<float> dense_forward(const std::vector<float>& W,
                                  const std::vector<float>& b,
                                  int64_t in_dim, int64_t out_dim,
                                  const std::vector<float>& x)
{
    std::vector<float> out(static_cast<size_t>(out_dim), 0.f);
    for (int64_t i = 0; i < out_dim; ++i) {
        float acc = b.empty() ? 0.f : b[static_cast<size_t>(i)];
        for (int64_t j = 0; j < in_dim; ++j)
            acc += W[static_cast<size_t>(i * in_dim + j)]
                 * (j < static_cast<int64_t>(x.size()) ? x[static_cast<size_t>(j)] : 0.f);
        out[static_cast<size_t>(i)] = acc;
    }
    return out;
}

// ReLU in-place
void relu_inplace(std::vector<float>& v) {
    for (auto& x : v) if (x < 0.f) x = 0.f;
}

// Try to match a pattern across a set of tensor names
// Returns ordered list of (weight_key, bias_key, layer_index)
struct LayerEntry { std::string w_key; std::string b_key; int idx; };

std::vector<LayerEntry> detect_numbered(
    const std::unordered_map<std::string, std::vector<int64_t>>& shapes)
{
    // Pattern A: "N.weight" / "N.bias" where N is a non-negative integer
    std::map<int, LayerEntry> found;
    static const std::regex re_w(R"(^(\d+)\.weight$)");
    static const std::regex re_b(R"(^(\d+)\.bias$)");
    for (auto& [k, _] : shapes) {
        std::smatch m;
        if (std::regex_match(k, m, re_w)) {
            int idx = std::stoi(m[1]);
            found[idx].w_key = k;
            found[idx].idx   = idx;
        } else if (std::regex_match(k, m, re_b)) {
            int idx = std::stoi(m[1]);
            found[idx].b_key = k;
            found[idx].idx   = idx;
        }
    }
    std::vector<LayerEntry> layers;
    for (auto& [idx, le] : found)
        if (!le.w_key.empty()) layers.push_back(le);
    return layers;
}

std::vector<LayerEntry> detect_layers_prefix(
    const std::unordered_map<std::string, std::vector<int64_t>>& shapes,
    const std::string& prefix)
{
    // Pattern: "<prefix>.N.weight" / "<prefix>.N.bias"
    std::map<int, LayerEntry> found;
    std::regex re_w("^" + prefix + R"(\.(\d+)\.weight$)");
    std::regex re_b("^" + prefix + R"(\.(\d+)\.bias$)");
    for (auto& [k, _] : shapes) {
        std::smatch m;
        if (std::regex_match(k, m, re_w)) {
            int idx = std::stoi(m[1]);
            found[idx].w_key = k;
            found[idx].idx   = idx;
        } else if (std::regex_match(k, m, re_b)) {
            int idx = std::stoi(m[1]);
            found[idx].b_key = k;
            found[idx].idx   = idx;
        }
    }
    std::vector<LayerEntry> layers;
    for (auto& [idx, le] : found)
        if (!le.w_key.empty()) layers.push_back(le);
    return layers;
}

} // anonymous namespace

bool SafeTensorsModel::load(const std::string& model_path)
{
    auto log = dh::logging::get("model.safetensors");
    m_path = model_path;

    // Resolve path: try direct, then with .safetensors extension
    std::string resolved = model_path;
    if (!std::filesystem::exists(resolved))
        resolved = model_path + ".safetensors";
    if (!std::filesystem::exists(resolved)) {
        log->warn("[{}] safetensors not found at '{}' or '{}.safetensors'",
                  m_key, model_path, model_path);
        return false;
    }

    log->info("[{}] Loading safetensors: {}", m_key, resolved);

    auto sf = loader::SafeTensorsFile::parse_header(resolved);
    if (!sf.parsed) {
        log->error("[{}] Failed to parse safetensors header: {}", m_key, resolved);
        return false;
    }

    log->info("[{}] Header parsed – {}", m_key, sf.summary());

    // Log all discovered tensors at debug level
    for (auto& tname : sf.tensor_names()) {
        auto& ti = sf.tensors.at(tname);
        std::string shape_str;
        for (size_t i = 0; i < ti.shape.size(); ++i) {
            if (i) shape_str += "×";
            shape_str += std::to_string(ti.shape[i]);
        }
        log->debug("[{}]   tensor '{}' dtype={} shape=[{}] bytes={}",
                   m_key, tname, ti.dtype, shape_str, ti.byte_size());
    }

    // Build shape map for layer detection
    std::unordered_map<std::string, std::vector<float>> raw_tensors;
    std::unordered_map<std::string, std::vector<int64_t>> shapes;
    for (auto& [tname, ti] : sf.tensors) {
        shapes[tname] = ti.shape;
    }

    // Only load if layers not already injected (e.g. by manifest loader)
    if (m_weights.empty()) {
        if (!auto_detect_layers(raw_tensors, shapes)) {
            // lazy-load needed tensors after detection — retry with actual data
            // Re-run detection: first load all 2D/1D tensors then detect
            for (auto& [tname, ti] : sf.tensors) {
                if (ti.dtype != "F32" && ti.dtype != "F16" && ti.dtype != "BF16"
                    && ti.dtype != "F64") continue;
                if (ti.numel() == 0 || ti.numel() > 100'000'000ULL) continue;
                auto data = sf.read_tensor_f32(tname);
                if (!data.empty()) raw_tensors[tname] = std::move(data);
            }
            if (!auto_detect_layers(raw_tensors, shapes)) {
                log->warn("[{}] Could not detect MLP layers – will use stub inference",
                          m_key);
                // Still mark as loaded so registry shows it as available
                m_loaded = true;
                return true;
            }
        } else {
            // Load actual weight data for detected layers
            for (auto& layer : m_weights) {
                // Only load if not already loaded
                if (layer.W.empty() && !layer.in_features) continue;
            }
            // Reload with actual data
            for (auto& [tname, ti] : sf.tensors) {
                if (ti.dtype != "F32" && ti.dtype != "F16" && ti.dtype != "BF16"
                    && ti.dtype != "F64") continue;
                if (ti.numel() == 0 || ti.numel() > 100'000'000ULL) continue;
                raw_tensors[tname] = sf.read_tensor_f32(tname);
            }
            m_weights.clear();
            auto_detect_layers(raw_tensors, shapes);
        }
    }

    log->info("[{}] Loaded {} MLP layers", m_key, m_weights.size());
    for (size_t i = 0; i < m_weights.size(); ++i)
        log->debug("[{}]   layer[{}]: {}→{} relu={}",
                   m_key, i, m_weights[i].in_features, m_weights[i].out_features,
                   m_weights[i].relu_after);

    m_loaded = true;
    return true;
}

bool SafeTensorsModel::auto_detect_layers(
    const std::unordered_map<std::string, std::vector<float>>& tensors,
    const std::unordered_map<std::string, std::vector<int64_t>>& shapes)
{
    auto log = dh::logging::get("model.safetensors");

    // Try patterns in priority order
    std::vector<std::vector<LayerEntry>> candidates = {
        detect_numbered(shapes),
        detect_layers_prefix(shapes, "layers"),
        detect_layers_prefix(shapes, "model"),
        detect_layers_prefix(shapes, "net"),
        detect_layers_prefix(shapes, "mlp"),
    };

    // Also try single fc/classifier layer
    std::vector<std::string> single_w_keys = {
        "fc.weight", "classifier.weight", "linear.weight", "output.weight",
        "weight", "head.weight"
    };

    std::vector<LayerEntry> best;
    for (auto& cand : candidates) {
        if (cand.size() > best.size()) best = cand;
    }

    // Single-layer fallback
    if (best.empty()) {
        for (auto& wk : single_w_keys) {
            if (shapes.count(wk) && shapes.at(wk).size() == 2) {
                LayerEntry le;
                le.w_key = wk;
                le.idx   = 0;
                // Try matching bias key
                std::string bk = wk.substr(0, wk.size() - 6) + "bias"; // replace "weight"
                if (shapes.count(bk)) le.b_key = bk;
                best = {le};
                break;
            }
        }
    }

    if (best.empty()) {
        log->debug("No recognisable layer pattern found in tensor names");
        return false;
    }

    m_weights.clear();
    for (size_t i = 0; i < best.size(); ++i) {
        auto& le = best[i];
        if (!shapes.count(le.w_key)) continue;
        auto& wshape = shapes.at(le.w_key);
        if (wshape.size() != 2) continue;

        LinearLayer ll;
        ll.out_features = wshape[0];
        ll.in_features  = wshape[1];
        ll.relu_after   = (i + 1 < best.size()); // last layer: no relu

        // Copy weight data if available
        if (tensors.count(le.w_key))
            ll.W = tensors.at(le.w_key);
        if (!le.b_key.empty() && tensors.count(le.b_key))
            ll.b = tensors.at(le.b_key);

        m_weights.push_back(std::move(ll));
    }

    return !m_weights.empty();
}

ModelOutput SafeTensorsModel::run_mlp(const std::vector<float>& features) const
{
    if (m_weights.empty()) {
        // No layers loaded — use stub heuristic
        StubModel stub(m_key, m_version, m_num_classes);
        stub.load("");
        ModelInput in;
        in.features = features;
        return stub.infer(in);
    }

    std::vector<float> x = features;

    for (size_t li = 0; li < m_weights.size(); ++li) {
        auto& ll = m_weights[li];
        if (ll.W.empty()) continue;

        // Truncate or pad input to match layer's expected in_features
        std::vector<float> padded(static_cast<size_t>(ll.in_features), 0.f);
        size_t copy_n = std::min(x.size(), padded.size());
        std::copy(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(copy_n),
                  padded.begin());

        x = dense_forward(ll.W, ll.b, ll.in_features, ll.out_features, padded);
        if (ll.relu_after) relu_inplace(x);
    }

    // Final softmax
    // Resize to expected num_classes if necessary
    if (static_cast<int>(x.size()) != m_num_classes) {
        x.resize(static_cast<size_t>(m_num_classes), 0.f);
    }
    auto probs = softmax_vec(x);

    int cls = static_cast<int>(
        std::max_element(probs.begin(), probs.end()) - probs.begin());

    ModelOutput out;
    out.logits          = x;
    out.probabilities   = probs;
    out.predicted_class = cls;
    out.confidence      = probs[static_cast<size_t>(cls)];

    static const char* LABELS[] = {"HOLD", "BUY", "SELL",
                                    "STRONG_BUY", "STRONG_SELL"};
    if (cls < static_cast<int>(sizeof(LABELS)/sizeof(LABELS[0])))
        out.label = LABELS[cls];
    else
        out.label = "HOLD";

    out.metadata = {{"backend", "safetensors"}, {"path", m_path},
                    {"num_layers", static_cast<int>(m_weights.size())}};
    return out;
}

ModelOutput SafeTensorsModel::infer(const ModelInput& input)
{
    return run_mlp(input.features);
}

// ── ONNX Runtime backend ──────────────────────────────────────────────────────
#ifdef USE_ONNXRUNTIME

bool OnnxModel::load(const std::string& model_path)
{
    auto log = dh::logging::get("model.onnx");
    m_path = model_path;

    std::string resolved = model_path;
    if (!std::filesystem::exists(resolved))
        resolved = model_path + ".onnx";
    if (!std::filesystem::exists(resolved)) {
        log->warn("[{}] ONNX model not found at '{}' or '{}.onnx'",
                  m_key, model_path, model_path);
        return false;
    }

    log->info("[{}] Loading ONNX model: {}", m_key, resolved);

    try {
        m_session_opts.SetIntraOpNumThreads(1);
        m_session_opts.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        m_session = std::make_unique<Ort::Session>(
            m_env, resolved.c_str(), m_session_opts);

        Ort::AllocatorWithDefaultOptions alloc;

        // Collect input names and shapes
        size_t num_inputs = m_session->GetInputCount();
        log->debug("[{}] ONNX inputs: {}", m_key, num_inputs);
        for (size_t i = 0; i < num_inputs; ++i) {
            auto name = m_session->GetInputNameAllocated(i, alloc);
            m_input_names_storage.emplace_back(name.get());

            auto type_info = m_session->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto shape = tensor_info.GetShape();
            m_input_shapes.push_back(shape);

            if (shape.size() >= 2 && shape.back() > 0)
                m_input_dim = shape.back();

            log->debug("[{}]   input[{}] '{}' shape=[{}]", m_key, i,
                       m_input_names_storage.back(),
                       [&]() {
                           std::string s;
                           for (size_t k = 0; k < shape.size(); ++k) {
                               if (k) s += "×";
                               s += shape[k] < 0
                                    ? "?" : std::to_string(shape[k]);
                           }
                           return s;
                       }());
        }

        // Collect output names
        size_t num_outputs = m_session->GetOutputCount();
        log->debug("[{}] ONNX outputs: {}", m_key, num_outputs);
        for (size_t i = 0; i < num_outputs; ++i) {
            auto name = m_session->GetOutputNameAllocated(i, alloc);
            m_output_names_storage.emplace_back(name.get());

            auto type_info = m_session->GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto shape = tensor_info.GetShape();
            if (!shape.empty() && shape.back() > 0)
                m_num_classes = shape.back();

            log->debug("[{}]   output[{}] '{}' shape=[{}]", m_key, i,
                       m_output_names_storage.back(),
                       [&]() {
                           std::string s;
                           for (size_t k = 0; k < shape.size(); ++k) {
                               if (k) s += "×";
                               s += shape[k] < 0
                                    ? "?" : std::to_string(shape[k]);
                           }
                           return s;
                       }());
        }

        // Build raw C-string pointer vectors for Run()
        for (auto& s : m_input_names_storage)  m_input_names.push_back(s.c_str());
        for (auto& s : m_output_names_storage) m_output_names.push_back(s.c_str());

        log->info("[{}] ONNX session ready: input_dim={} num_classes={}",
                  m_key, m_input_dim, m_num_classes);
        m_loaded = true;
        return true;

    } catch (const Ort::Exception& e) {
        log->error("[{}] ONNX Runtime error during load: {}", m_key, e.what());
        return false;
    } catch (const std::exception& e) {
        log->error("[{}] Exception during ONNX load: {}", m_key, e.what());
        return false;
    }
}

ModelOutput OnnxModel::infer(const ModelInput& input)
{
    auto log = dh::logging::get("model.onnx");

    if (!m_loaded || !m_session) {
        log->warn("[{}] ONNX model not loaded, returning stub output", m_key);
        StubModel stub(m_key, m_version, static_cast<int>(m_num_classes));
        stub.load("");
        return stub.infer(input);
    }

    try {
        auto memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);

        // Build input tensor – pad/truncate features to expected input_dim
        int64_t feat_dim = (m_input_dim > 0)
                           ? m_input_dim
                           : static_cast<int64_t>(input.features.size());

        std::vector<float> feat_data(static_cast<size_t>(feat_dim), 0.f);
        size_t copy_n = std::min(input.features.size(),
                                 static_cast<size_t>(feat_dim));
        std::copy(input.features.begin(),
                  input.features.begin() + static_cast<std::ptrdiff_t>(copy_n),
                  feat_data.begin());

        std::vector<int64_t> input_shape = {1, feat_dim};
        auto input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, feat_data.data(), feat_data.size(),
            input_shape.data(), input_shape.size());

        auto outputs = m_session->Run(
            Ort::RunOptions{nullptr},
            m_input_names.data(),  &input_tensor,  m_input_names.size(),
            m_output_names.data(), m_output_names.size());

        if (outputs.empty()) {
            log->warn("[{}] ONNX Run returned no outputs", m_key);
            return {};
        }

        float* raw = outputs[0].GetTensorMutableData<float>();
        auto   info  = outputs[0].GetTensorTypeAndShapeInfo();
        auto   shape = info.GetShape();
        int64_t n_out = shape.empty() ? m_num_classes : shape.back();

        std::vector<float> logits(raw, raw + n_out);
        auto probs = softmax_vec(logits);

        int cls = static_cast<int>(
            std::max_element(probs.begin(), probs.end()) - probs.begin());

        ModelOutput out;
        out.logits          = logits;
        out.probabilities   = probs;
        out.predicted_class = cls;
        out.confidence      = probs[static_cast<size_t>(cls)];

        static const char* LABELS[] = {"HOLD", "BUY", "SELL",
                                        "STRONG_BUY", "STRONG_SELL"};
        if (cls < static_cast<int>(sizeof(LABELS)/sizeof(LABELS[0])))
            out.label = LABELS[cls];
        else
            out.label = "HOLD";

        out.metadata = {{"backend", "onnxruntime"}, {"path", m_path},
                        {"bar_index", input.bar_index}};
        return out;

    } catch (const Ort::Exception& e) {
        log->error("[{}] ONNX Runtime inference error: {}", m_key, e.what());
        return {};
    }
}

#endif // USE_ONNXRUNTIME

// ── ModelRegistry ─────────────────────────────────────────────────────────────
ModelRegistry& ModelRegistry::instance()
{
    static ModelRegistry reg;
    return reg;
}

void ModelRegistry::register_model(const std::string& key,
                                    std::shared_ptr<IModel> model)
{
    m_models[key] = std::move(model);
}

std::shared_ptr<IModel> ModelRegistry::get(const std::string& key) const
{
    auto it = m_models.find(key);
    if (it == m_models.end())
        throw std::out_of_range("Model not found: " + key);
    return it->second;
}

std::vector<std::string> ModelRegistry::keys() const
{
    std::vector<std::string> ks;
    ks.reserve(m_models.size());
    for (auto& [k,_] : m_models) ks.push_back(k);
    return ks;
}

void ModelRegistry::load_all(const std::string& base_dir)
{
    auto log = dh::logging::get("model.registry");
    log->info("Loading all registered models from '{}'", base_dir);
    for (auto& [k, m] : m_models) {
        std::string path = base_dir + "/" + k;
        log->info("  Loading model '{}' from '{}'", k, path);
        bool ok = m->load(path);
        log->log(ok ? spdlog::level::info : spdlog::level::warn,
                 "  Model '{}' load {}", k, ok ? "OK" : "FAILED (stub active)");
    }
}

} // namespace dh::ml

