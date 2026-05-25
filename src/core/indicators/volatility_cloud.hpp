#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Daydream Handler – Volatility Cloud Indicator
// ATR-based adaptive volatility envelope forming a dynamic cloud.
// ─────────────────────────────────────────────────────────────────────────────
#include "../../data/timeseries.hpp"
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

namespace dh::indicators {

struct VolatilityCloudParams {
    int    atr_period    = 14;
    double multiplier    = 1.5;
    int    smooth_period = 5;
};

struct VolatilityCloudValues {
    double upper_band   = 0;
    double lower_band   = 0;
    double mid          = 0;
    double atr          = 0;
    double atr_pct      = 0;   // ATR as % of close
    bool   expanding    = false;
    bool   contracting  = false;
};

class VolatilityCloud {
public:
    using Params = VolatilityCloudParams;

    explicit VolatilityCloud(Params p = Params{}) : m_p(p) {}

    VolatilityCloudValues calculate(const data::TimeSeries& ts) const {
        size_t n = ts.size();
        if (n < static_cast<size_t>(m_p.atr_period + m_p.smooth_period))
            return {};

        // True Range
        auto highs  = ts.high_window(n);
        auto lows   = ts.low_window(n);
        auto closes = ts.close_window(n);

        std::vector<double> tr(n);
        tr[0] = highs[0] - lows[0];
        for (size_t i = 1; i < n; ++i) {
            tr[i] = std::max({
                highs[i]  - lows[i],
                std::abs(highs[i]  - closes[i-1]),
                std::abs(lows[i]   - closes[i-1])
            });
        }

        // ATR (simple MA of TR for last period)
        double atr = 0;
        for (size_t i = n - m_p.atr_period; i < n; ++i)
            atr += tr[i];
        atr /= m_p.atr_period;

        // EMA-smoothed midpoint
        double k   = 2.0 / (m_p.smooth_period + 1);
        double ema = closes[n - m_p.smooth_period];
        for (size_t i = n - m_p.smooth_period + 1; i < n; ++i)
            ema = closes[i] * k + ema * (1.0 - k);

        // Previous ATR for expanding/contracting detection
        double prev_atr = 0;
        if (n > static_cast<size_t>(m_p.atr_period + 1)) {
            for (size_t i = n - m_p.atr_period - 1; i < n - 1; ++i)
                prev_atr += tr[i];
            prev_atr /= m_p.atr_period;
        }

        VolatilityCloudValues v;
        v.atr         = atr;
        v.mid         = ema;
        v.upper_band  = ema + m_p.multiplier * atr;
        v.lower_band  = ema - m_p.multiplier * atr;
        v.atr_pct     = closes.back() > 0 ? atr / closes.back() : 0.0;
        v.expanding   = prev_atr > 0 && atr > prev_atr;
        v.contracting = prev_atr > 0 && atr < prev_atr;
        return v;
    }

private:
    Params m_p;
};

} // namespace dh::indicators
