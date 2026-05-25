#include "feature_composer.hpp"
#include <stdexcept>
#include <random>
#include <cstring>

namespace dh::ml {

// ── GEMM ──────────────────────────────────────────────────────────────────────
Matrix gemm(const Matrix& A, const Matrix& B, float alpha, float beta)
{
    if (A.cols != B.rows)
        throw std::invalid_argument("gemm: dimension mismatch");
    Matrix C;
    C.rows = A.rows;
    C.cols = B.cols;
    C.data.assign(C.rows * C.cols, 0.0f);
    for (size_t i = 0; i < A.rows; ++i)
        for (size_t k = 0; k < A.cols; ++k)
            for (size_t j = 0; j < B.cols; ++j)
                C.at(i, j) += alpha * A.at(i, k) * B.at(k, j);
    if (beta != 0.0f)
        for (auto& v : C.data) v *= beta;
    return C;
}

// ── ReLU ─────────────────────────────────────────────────────────────────────
std::vector<float> relu(const std::vector<float>& v)
{
    std::vector<float> out(v.size());
    for (size_t i = 0; i < v.size(); ++i)
        out[i] = std::max(0.0f, v[i]);
    return out;
}

// ── Softmax ───────────────────────────────────────────────────────────────────
std::vector<float> softmax(const std::vector<float>& v)
{
    if (v.empty()) return {};
    float max_v = *std::max_element(v.begin(), v.end());
    std::vector<float> out(v.size());
    float sum = 0.0f;
    for (size_t i = 0; i < v.size(); ++i) {
        out[i] = std::exp(v[i] - max_v);
        sum   += out[i];
    }
    for (auto& x : out) x /= sum;
    return out;
}

// ── Concat ────────────────────────────────────────────────────────────────────
std::vector<float> concat(const std::vector<float>& a,
                           const std::vector<float>& b)
{
    std::vector<float> out;
    out.reserve(a.size() + b.size());
    out.insert(out.end(), a.begin(), a.end());
    out.insert(out.end(), b.begin(), b.end());
    return out;
}

// ── Shaper ────────────────────────────────────────────────────────────────────
Matrix shape_to(const std::vector<float>& v, size_t rows, size_t cols)
{
    if (v.size() != rows * cols)
        throw std::invalid_argument("shape_to: size mismatch");
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.data = v;
    return m;
}

// ── Splitter ──────────────────────────────────────────────────────────────────
std::pair<std::vector<float>, std::vector<float>>
split_at(const std::vector<float>& v, size_t index)
{
    if (index > v.size())
        throw std::out_of_range("split_at: index out of range");
    return {
        std::vector<float>(v.begin(), v.begin() + index),
        std::vector<float>(v.begin() + index, v.end())
    };
}

// ── FeatureComposer ───────────────────────────────────────────────────────────
std::vector<float> FeatureComposer::normalise_zscore(
    const std::vector<double>& prices)
{
    if (prices.empty()) return {};
    double mean = std::accumulate(prices.begin(), prices.end(), 0.0)
                  / prices.size();
    double sq   = 0;
    for (double p : prices) sq += (p - mean) * (p - mean);
    double std  = prices.size() > 1
                  ? std::sqrt(sq / (prices.size() - 1))
                  : 1.0;
    if (std < 1e-10) std = 1.0;

    std::vector<float> out(prices.size());
    for (size_t i = 0; i < prices.size(); ++i)
        out[i] = static_cast<float>((prices[i] - mean) / std);
    return out;
}

size_t FeatureComposer::base_dim() const
{
    size_t d = static_cast<size_t>(m_cfg.window) * 4; // O/H/L/C
    if (m_cfg.include_ich) d += 5;   // tenkan, kijun, senkA, senkB, chikou
    if (m_cfg.include_vol) d += 5;   // upper, lower, mid, atr, atr_pct
    if (m_cfg.include_fib) d += 7;   // 7 fib levels
    return d;
}

std::vector<float> FeatureComposer::compose(
    const data::TimeSeries& ts,
    const indicators::IchimokuValues& ich,
    const indicators::VolatilityCloudValues& vol,
    double swing_low,
    double swing_high) const
{
    size_t n = ts.size();
    size_t w = std::min(static_cast<size_t>(m_cfg.window), n);

    auto closes = ts.close_window(w);
    auto opens  = ts.open_window(w);
    auto highs  = ts.high_window(w);
    auto lows   = ts.low_window(w);

    std::vector<float> feat;
    feat.reserve(base_dim());

    auto add_norm = [&](const std::vector<double>& v) {
        auto nv = m_cfg.normalise ? normalise_zscore(v)
                                  : std::vector<float>(v.begin(), v.end());
        feat.insert(feat.end(), nv.begin(), nv.end());
    };

    add_norm(closes);
    add_norm(opens);
    add_norm(highs);
    add_norm(lows);

    if (m_cfg.include_ich && closes.back() > 0) {
        double ref = closes.back();
        feat.push_back(static_cast<float>((ich.tenkan_sen  - ref) / ref));
        feat.push_back(static_cast<float>((ich.kijun_sen   - ref) / ref));
        feat.push_back(static_cast<float>((ich.senkou_a    - ref) / ref));
        feat.push_back(static_cast<float>((ich.senkou_b    - ref) / ref));
        feat.push_back(static_cast<float>((ich.chikou_span - ref) / ref));
    }

    if (m_cfg.include_vol && closes.back() > 0) {
        double ref = closes.back();
        feat.push_back(static_cast<float>((vol.upper_band - ref) / ref));
        feat.push_back(static_cast<float>((vol.lower_band - ref) / ref));
        feat.push_back(static_cast<float>((vol.mid        - ref) / ref));
        feat.push_back(static_cast<float>(vol.atr_pct));
        feat.push_back(vol.expanding ? 1.0f : (vol.contracting ? -1.0f : 0.0f));
    }

    if (m_cfg.include_fib && swing_high > swing_low) {
        auto fibs = indicators::Fibonacci::retracement(swing_low, swing_high);
        double ref = closes.back() > 0 ? closes.back() : 1.0;
        for (size_t i = 0; i < std::min<size_t>(7, fibs.size()); ++i)
            feat.push_back(static_cast<float>((fibs[i].price - ref) / ref));
    }

    return project(feat);
}

std::vector<float> FeatureComposer::project(const std::vector<float>& feat) const
{
    if (m_cfg.project_dim <= 0 || feat.empty()) return feat;

    size_t in_dim  = feat.size();
    size_t out_dim = static_cast<size_t>(m_cfg.project_dim);

    // Lazy-init random projection weights (scaled Xavier)
    if (m_W.size() != in_dim * out_dim) {
        m_W.resize(in_dim * out_dim);
        std::mt19937 rng(2024);
        float scale = std::sqrt(2.0f / (in_dim + out_dim));
        std::normal_distribution<float> dist(0.0f, scale);
        for (auto& w : m_W) w = dist(rng);
    }

    // MatMul: out = feat (1 × in_dim) * W (in_dim × out_dim)
    Matrix F = shape_to(feat, 1, in_dim);
    Matrix W;
    W.rows = in_dim;
    W.cols = out_dim;
    W.data = m_W;

    Matrix out = gemm(F, W);
    return relu(out.data);
}

} // namespace dh::ml
