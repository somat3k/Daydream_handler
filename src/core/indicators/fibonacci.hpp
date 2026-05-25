#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Daydream Handler – Fibonacci Levels
// Retracements, extensions and expansion zones.
// ─────────────────────────────────────────────────────────────────────────────
#include <vector>
#include <string>

namespace dh::indicators {

struct FibLevel {
    double ratio;
    double price;
    std::string label;
};

class Fibonacci {
public:
    // Standard retracement ratios
    static constexpr double RATIOS[] = {
        0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0,
        1.272, 1.414, 1.618, 2.0, 2.618
    };

    /// Retracement levels from swing_high down to swing_low.
    static std::vector<FibLevel> retracement(double swing_low,
                                              double swing_high)
    {
        std::vector<FibLevel> out;
        double range = swing_high - swing_low;
        for (double r : RATIOS) {
            out.push_back({r, swing_high - r * range,
                           "Fib " + fmt(r)});
        }
        return out;
    }

    /// Extension levels projected from swing_high above swing_low.
    static std::vector<FibLevel> extension(double swing_low,
                                            double swing_high,
                                            double retrace_low)
    {
        std::vector<FibLevel> out;
        double range = swing_high - swing_low;
        for (double r : RATIOS) {
            if (r < 1.0) continue;  // extensions only
            out.push_back({r, retrace_low + r * range,
                           "Ext " + fmt(r)});
        }
        return out;
    }

    /// Find the nearest Fibonacci level to a given price.
    static const FibLevel* nearest(const std::vector<FibLevel>& levels,
                                   double price)
    {
        const FibLevel* best = nullptr;
        double min_dist = 1e18;
        for (auto& lv : levels) {
            double d = std::abs(lv.price - price);
            if (d < min_dist) { min_dist = d; best = &lv; }
        }
        return best;
    }

private:
    static std::string fmt(double r) {
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%.3f", r);
        return {buf};
    }
};

} // namespace dh::indicators
