#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Daydream Handler – Liquidity Sweep Detector
// Detects equal highs/lows, stop-hunt runs and liquidity grab candles.
// ─────────────────────────────────────────────────────────────────────────────
#include "../../data/timeseries.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

namespace dh::indicators {

struct LiquiditySweepParams {
    double equal_tolerance  = 0.0002;
    double hunt_threshold   = 0.001;
    int    lookback         = 20;
};

enum class LiquidityEventType {
    EQUAL_HIGH,
    EQUAL_LOW,
    STOP_HUNT_BULLISH,
    STOP_HUNT_BEARISH,
    SPREAD_SPIKE,
};

struct LiquidityEvent {
    LiquidityEventType type;
    double             price_level;
    int64_t            bar_index;
    double             magnitude;  // pips equivalent
};

class LiquiditySweep {
public:
    using Params = LiquiditySweepParams;

    explicit LiquiditySweep(Params p = Params{}) : m_p(p) {}

    std::vector<LiquidityEvent> detect(const data::TimeSeries& ts) const {
        std::vector<LiquidityEvent> events;
        size_t n = ts.size();
        if (n < static_cast<size_t>(m_p.lookback + 1)) return events;

        auto highs  = ts.high_window(n);
        auto lows   = ts.low_window(n);
        auto closes = ts.close_window(n);

        for (size_t i = m_p.lookback; i < n; ++i) {
            double ref_h = highs[i - 1];
            double ref_l = lows[i - 1];
            double curr_h = highs[i];
            double curr_l = lows[i];
            double close  = closes[i];

            // Equal highs/lows within tolerance
            if (std::abs(curr_h - ref_h) / ref_h < m_p.equal_tolerance) {
                events.push_back({LiquidityEventType::EQUAL_HIGH,
                                  (curr_h + ref_h) / 2.0,
                                  static_cast<int64_t>(i), 0.0});
            }
            if (std::abs(curr_l - ref_l) / ref_l < m_p.equal_tolerance) {
                events.push_back({LiquidityEventType::EQUAL_LOW,
                                  (curr_l + ref_l) / 2.0,
                                  static_cast<int64_t>(i), 0.0});
            }

            // Stop hunt: wick beyond prior swing then reversal close
            double window_high = *std::max_element(
                highs.begin() + i - m_p.lookback, highs.begin() + i);
            double window_low  = *std::min_element(
                lows.begin()  + i - m_p.lookback, lows.begin()  + i);

            double hunt_h = window_high * (1.0 + m_p.hunt_threshold);
            double hunt_l = window_low  * (1.0 - m_p.hunt_threshold);

            if (curr_h > hunt_h && close < window_high) {
                events.push_back({LiquidityEventType::STOP_HUNT_BEARISH,
                                  curr_h,
                                  static_cast<int64_t>(i),
                                  curr_h - window_high});
            }
            if (curr_l < hunt_l && close > window_low) {
                events.push_back({LiquidityEventType::STOP_HUNT_BULLISH,
                                  curr_l,
                                  static_cast<int64_t>(i),
                                  window_low - curr_l});
            }
        }
        return events;
    }

private:
    Params m_p;
};

} // namespace dh::indicators
