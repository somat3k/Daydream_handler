#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Daydream Handler – Supply & Demand Zone Detector
// Identifies institutional S/D zones from impulse moves.
// ─────────────────────────────────────────────────────────────────────────────
#include "../../data/timeseries.hpp"
#include <vector>
#include <algorithm>

namespace dh::indicators {

struct SDZoneParams {
    int    impulse_bars  = 3;
    double impulse_ratio = 0.003;
    int    base_bars     = 5;
    int    max_zones     = 20;
};

struct SDZone {
    double top;
    double bottom;
    bool   is_supply;   // true = supply, false = demand
    int64_t formed_at;  // bar index
    int     strength;   // number of tests without break
};

class SupplyDemand {
public:
    using Params = SDZoneParams;

    explicit SupplyDemand(Params p = Params{}) : m_p(p) {}

    std::vector<SDZone> detect(const data::TimeSeries& ts) const {
        std::vector<SDZone> zones;
        size_t n = ts.size();
        if (n < static_cast<size_t>(m_p.impulse_bars + m_p.base_bars + 2))
            return zones;

        auto highs  = ts.high_window(n);
        auto lows   = ts.low_window(n);
        auto closes = ts.close_window(n);

        for (size_t i = m_p.base_bars;
             i < n - m_p.impulse_bars;
             ++i)
        {
            // Demand: consolidation then bullish impulse
            double base_high = *std::max_element(
                highs.begin() + i - m_p.base_bars, highs.begin() + i);
            double base_low  = *std::min_element(
                lows.begin()  + i - m_p.base_bars, lows.begin()  + i);

            double imp_close_up = closes[i + m_p.impulse_bars - 1];
            double imp_close_dn = closes[i + m_p.impulse_bars - 1];
            double imp_high = *std::max_element(
                highs.begin() + i, highs.begin() + i + m_p.impulse_bars);
            double imp_low  = *std::min_element(
                lows.begin()  + i, lows.begin()  + i + m_p.impulse_bars);
            (void)imp_high; (void)imp_low;

            double price_ref = closes[i];
            if (price_ref <= 0) continue;

            // Bullish impulse → demand zone
            if (imp_close_up - base_high > m_p.impulse_ratio * price_ref) {
                SDZone z;
                z.top      = base_high;
                z.bottom   = base_low;
                z.is_supply = false;
                z.formed_at = static_cast<int64_t>(i);
                z.strength  = 1;
                zones.push_back(z);
            }
            // Bearish impulse → supply zone
            if (base_low - imp_close_dn > m_p.impulse_ratio * price_ref) {
                SDZone z;
                z.top       = base_high;
                z.bottom    = base_low;
                z.is_supply = true;
                z.formed_at = static_cast<int64_t>(i);
                z.strength  = 1;
                zones.push_back(z);
            }

            if (static_cast<int>(zones.size()) >= m_p.max_zones) break;
        }
        return zones;
    }

private:
    Params m_p;
};

} // namespace dh::indicators
