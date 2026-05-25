#include "inference_engine.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>

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
    for (auto& [k, m] : m_models) {
        std::string path = base_dir + "/" + k;
        m->load(path);
    }
}

} // namespace dh::ml
