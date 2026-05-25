#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Daydream Handler – Ichimoku Cloud Indicator
// Full Ichimoku Kinko Hyo: Tenkan, Kijun, Senkou A/B, Chikou.
// ─────────────────────────────────────────────────────────────────────────────
#include "../../data/market_data.hpp"
#include "../../data/timeseries.hpp"
#include <vector>
#include <algorithm>
#include <stdexcept>

namespace dh::indicators {

struct IchimokuParams {
    int tenkan_period   = 9;
    int kijun_period    = 26;
    int senkou_b_period = 52;
    int displacement    = 26;
};

struct IchimokuValues {
    double tenkan_sen   = 0;   // Conversion line
    double kijun_sen    = 0;   // Base line
    double senkou_a     = 0;   // Leading Span A
    double senkou_b     = 0;   // Leading Span B
    double chikou_span  = 0;   // Lagging Span
    bool   price_above_cloud = false;
    bool   price_below_cloud = false;
    bool   cloud_bullish     = false;  // Span A > Span B
};

class Ichimoku {
public:
    explicit Ichimoku(IchimokuParams p = {}) : m_p(p) {}

    IchimokuValues calculate(const data::TimeSeries& ts) const {
        size_t n = ts.size();
        size_t required = static_cast<size_t>(
            m_p.senkou_b_period + m_p.displacement);
        if (n < required) return {};

        auto highs  = ts.high_window(n);
        auto lows   = ts.low_window(n);
        auto closes = ts.close_window(n);

        double tenkan = midpoint(highs, lows, n, m_p.tenkan_period);
        double kijun  = midpoint(highs, lows, n, m_p.kijun_period);
        double senkA  = (tenkan + kijun) / 2.0;
        double senkB  = midpoint(highs, lows, n, m_p.senkou_b_period);
        double chikou = closes.back();
        double close  = closes.back();

        IchimokuValues v;
        v.tenkan_sen  = tenkan;
        v.kijun_sen   = kijun;
        v.senkou_a    = senkA;
        v.senkou_b    = senkB;
        v.chikou_span = chikou;
        v.cloud_bullish    = senkA > senkB;
        double cloud_top   = std::max(senkA, senkB);
        double cloud_bot   = std::min(senkA, senkB);
        v.price_above_cloud = close > cloud_top;
        v.price_below_cloud = close < cloud_bot;
        return v;
    }

private:
    static double midpoint(const std::vector<double>& highs,
                           const std::vector<double>& lows,
                           size_t n, int period)
    {
        if (static_cast<size_t>(period) > n) return 0.0;
        size_t start = n - static_cast<size_t>(period);
        double hmax  = *std::max_element(highs.begin() + start, highs.end());
        double lmin  = *std::min_element(lows.begin()  + start, lows.end());
        return (hmax + lmin) / 2.0;
    }

    IchimokuParams m_p;
};

} // namespace dh::indicators
