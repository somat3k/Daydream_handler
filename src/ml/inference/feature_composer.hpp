#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Daydream Handler – Feature Composer
// Builds multi-dimensional feature tensors from indicator values and
// timeseries windows. Applies GEMM-style projections, ReLU activations,
// concatenation (Concat), reshaping (Shapers) and splitting (Splitters).
// ─────────────────────────────────────────────────────────────────────────────
#include "../../data/timeseries.hpp"
#include "../../core/indicators/ichimoku.hpp"
#include "../../core/indicators/volatility_cloud.hpp"
#include "../../core/indicators/fibonacci.hpp"
#include <vector>
#include <cmath>
#include <string>
#include <numeric>
#include <algorithm>

namespace dh::ml {

// ── Simple matrix (row-major) ─────────────────────────────────────────────────
struct Matrix {
    std::vector<float> data;
    size_t rows = 0;
    size_t cols = 0;

    float& at(size_t r, size_t c) { return data[r * cols + c]; }
    float  at(size_t r, size_t c) const { return data[r * cols + c]; }
};

// ── Linear algebra primitives ─────────────────────────────────────────────────

/// GEMM: C = alpha * A * B + beta * C
Matrix gemm(const Matrix& A, const Matrix& B,
            float alpha = 1.0f, float beta = 0.0f);

/// ReLU activation element-wise
std::vector<float> relu(const std::vector<float>& v);

/// Softmax
std::vector<float> softmax(const std::vector<float>& v);

/// Concat two feature vectors
std::vector<float> concat(const std::vector<float>& a,
                           const std::vector<float>& b);

/// Shaper: reshape flat vector to matrix (rows × cols)
Matrix shape_to(const std::vector<float>& v, size_t rows, size_t cols);

/// Splitter: split flat vector at index
std::pair<std::vector<float>, std::vector<float>>
split_at(const std::vector<float>& v, size_t index);

// ── Feature extraction ────────────────────────────────────────────────────────

struct FeatureConfig {
    int    window       = 50;   // bars to look back for timeseries features
    bool   normalise    = true; // zero-mean / unit-std
    bool   include_ich  = true;
    bool   include_vol  = true;
    bool   include_fib  = true;
    int    project_dim  = 64;   // GEMM output dimension (0 = skip projection)
};

class FeatureComposer {
public:
    explicit FeatureComposer(FeatureConfig cfg = {}) : m_cfg(cfg) {}

    /// Build feature vector from a single timeseries snapshot.
    std::vector<float> compose(
        const data::TimeSeries& ts,
        const indicators::IchimokuValues& ich,
        const indicators::VolatilityCloudValues& vol,
        double swing_low  = 0.0,
        double swing_high = 0.0) const;

    /// Normalise a window of prices (z-score).
    static std::vector<float> normalise_zscore(
        const std::vector<double>& prices);

    /// Project features through a random-init weight matrix (GEMM + ReLU).
    std::vector<float> project(const std::vector<float>& feat) const;

    size_t output_dim() const {
        return m_cfg.project_dim > 0
               ? static_cast<size_t>(m_cfg.project_dim)
               : base_dim();
    }

private:
    size_t base_dim() const;
    mutable std::vector<float> m_W;  // lazy-init projection weights

    FeatureConfig m_cfg;
};

} // namespace dh::ml
