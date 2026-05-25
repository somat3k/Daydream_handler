#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Daydream Handler – TimeSeries + LFO Jitter Replay
// Stores ordered time-indexed series; supports LFO (Low-Frequency Oscillator)
// jitter injection for early-stage replay augmentation.
// ─────────────────────────────────────────────────────────────────────────────
#include "market_data.hpp"
#include <deque>
#include <vector>
#include <cmath>
#include <random>
#include <stdexcept>
#include <algorithm>
#include <functional>

namespace dh::data {

// ── LFO Jitter parameters ─────────────────────────────────────────────────────
struct LFOJitterParams {
    double  frequency   = 0.01;    // cycles per bar
    double  amplitude   = 0.0002;  // fraction of price
    double  phase       = 0.0;     // radians
    bool    add_noise   = true;    // add Gaussian noise on top of LFO
    double  noise_sigma = 0.0001;  // price fraction
    uint32_t seed       = 42;
};

// ── Sliding window timeseries ─────────────────────────────────────────────────
class TimeSeries {
public:
    explicit TimeSeries(size_t max_window = 500)
        : m_max_window(max_window) {}

    void push(const Bar& bar) {
        m_bars.push_back(bar);
        if (m_bars.size() > m_max_window)
            m_bars.pop_front();
        m_step++;
    }

    [[nodiscard]] size_t size()  const noexcept { return m_bars.size(); }
    [[nodiscard]] bool   empty() const noexcept { return m_bars.empty(); }
    [[nodiscard]] int64_t step() const noexcept { return m_step; }

    [[nodiscard]] const Bar& latest() const {
        if (m_bars.empty()) throw std::underflow_error("TimeSeries is empty");
        return m_bars.back();
    }

    [[nodiscard]] const Bar& at(size_t i) const { return m_bars.at(i); }

    // Collect the last N close prices into a vector (newest last)
    [[nodiscard]] std::vector<double> close_window(size_t n) const {
        n = std::min(n, m_bars.size());
        std::vector<double> out(n);
        auto it = m_bars.end();
        for (size_t i = n; i-- > 0; ) out[i] = (--it)->close;
        return out;
    }

    [[nodiscard]] std::vector<double> open_window(size_t n) const {
        n = std::min(n, m_bars.size());
        std::vector<double> out(n);
        auto it = m_bars.end();
        for (size_t i = n; i-- > 0; ) out[i] = (--it)->open;
        return out;
    }

    [[nodiscard]] std::vector<double> high_window(size_t n) const {
        n = std::min(n, m_bars.size());
        std::vector<double> out(n);
        auto it = m_bars.end();
        for (size_t i = n; i-- > 0; ) out[i] = (--it)->high;
        return out;
    }

    [[nodiscard]] std::vector<double> low_window(size_t n) const {
        n = std::min(n, m_bars.size());
        std::vector<double> out(n);
        auto it = m_bars.end();
        for (size_t i = n; i-- > 0; ) out[i] = (--it)->low;
        return out;
    }

    // Range iterator
    using const_iterator = std::deque<Bar>::const_iterator;
    [[nodiscard]] const_iterator begin() const { return m_bars.begin(); }
    [[nodiscard]] const_iterator end()   const { return m_bars.end();   }

private:
    std::deque<Bar> m_bars;
    size_t          m_max_window;
    int64_t         m_step = 0;
};

// ── LFO Jitter Replay engine ──────────────────────────────────────────────────
class LFOJitterReplay {
public:
    explicit LFOJitterReplay(LFOJitterParams params = {})
        : m_params(params)
        , m_rng(params.seed)
        , m_dist(0.0, params.noise_sigma)
    {}

    /// Apply LFO jitter to a bar (augmentation for replay).
    Bar augment(Bar bar, int64_t bar_index) const {
        double lfo = m_params.amplitude * bar.close *
                     std::sin(2.0 * M_PI * m_params.frequency * bar_index
                              + m_params.phase);
        double noise = m_params.add_noise
                       ? m_dist(m_rng) * bar.close
                       : 0.0;
        double delta = lfo + noise;
        bar.open  += delta;
        bar.high  += std::abs(delta);
        bar.low   -= std::abs(delta);
        bar.close += delta;
        return bar;
    }

    /// Replay a stored series with jitter injection, calling cb for each bar.
    void replay(const TimeSeries& src,
                const std::function<void(const Bar&, int64_t)>& cb,
                bool augment_bars = true) const
    {
        int64_t idx = 0;
        for (const auto& bar : src) {
            cb(augment_bars ? augment(bar, idx) : bar, idx);
            ++idx;
        }
    }

    void set_params(LFOJitterParams p) {
        m_params = p;
        m_rng.seed(p.seed);
        m_dist = std::normal_distribution<double>(0.0, p.noise_sigma);
    }

private:
    LFOJitterParams                       m_params;
    mutable std::mt19937                  m_rng;
    mutable std::normal_distribution<double> m_dist;
};

} // namespace dh::data
