#include "strategy.hpp"
#include <algorithm>

namespace dh::strategy {

// ── BreakoutStrategy ─────────────────────────────────────────────────────────
std::optional<Signal> BreakoutStrategy::evaluate(
    const data::TimeSeries& ts,
    const data::Instrument& instr,
    int64_t bar_index)
{
    size_t n = ts.size();
    if (n < static_cast<size_t>(m_p.lookback + 1)) return {};

    auto highs  = ts.high_window(n);
    auto lows   = ts.low_window(n);
    auto closes = ts.close_window(n);

    double range_high = *std::max_element(
        highs.begin(), highs.begin() + n - 1);
    double range_low  = *std::min_element(
        lows.begin(),  lows.begin()  + n - 1);

    double close = closes.back();

    // Bullish breakout: close above range_high + threshold
    if (close > range_high * (1.0 + m_p.threshold)) {
        Signal s;
        s.direction   = Direction::LONG;
        s.strategy    = name();
        s.entry_price = close;
        s.sl_ref      = range_low;
        s.tp_ref      = close + (close - range_low) * 0.618;
        s.confidence  = 0.65f;
        s.reason      = "Bullish breakout above " + std::to_string(range_high);
        s.bar_index   = bar_index;
        return s;
    }
    // Bearish breakout
    if (close < range_low * (1.0 - m_p.threshold)) {
        Signal s;
        s.direction   = Direction::SHORT;
        s.strategy    = name();
        s.entry_price = close;
        s.sl_ref      = range_high;
        s.tp_ref      = close - (range_high - close) * 0.618;
        s.confidence  = 0.65f;
        s.reason      = "Bearish breakout below " + std::to_string(range_low);
        s.bar_index   = bar_index;
        return s;
    }
    return {};
}

// ── ReversalStrategy ─────────────────────────────────────────────────────────
std::optional<Signal> ReversalStrategy::evaluate(
    const data::TimeSeries& ts,
    const data::Instrument& instr,
    int64_t bar_index)
{
    if (ts.size() < 52u) return {};

    indicators::Ichimoku ich_ind(m_p.ichi);
    indicators::VolatilityCloud vol_ind(m_p.vol);

    auto ich = ich_ind.calculate(ts);
    auto vol = vol_ind.calculate(ts);

    auto closes = ts.close_window(ts.size());
    double close  = closes.back();
    double prev   = closes[closes.size() - 2];

    // Reversal: price below cloud + vol contracting → mean reversion long
    if (ich.price_below_cloud && vol.contracting && close > prev) {
        Signal s;
        s.direction   = Direction::LONG;
        s.strategy    = name();
        s.entry_price = close;
        s.sl_ref      = vol.lower_band;
        s.tp_ref      = ich.kijun_sen;
        s.confidence  = 0.60f;
        s.reason      = "Reversal: below cloud, volatility contracting";
        s.bar_index   = bar_index;
        return s;
    }
    // Reversal: price above cloud + vol contracting → mean reversion short
    if (ich.price_above_cloud && vol.contracting && close < prev) {
        Signal s;
        s.direction   = Direction::SHORT;
        s.strategy    = name();
        s.entry_price = close;
        s.sl_ref      = vol.upper_band;
        s.tp_ref      = ich.kijun_sen;
        s.confidence  = 0.60f;
        s.reason      = "Reversal: above cloud, volatility contracting";
        s.bar_index   = bar_index;
        return s;
    }
    return {};
}

// ── SDZoneStrategy ───────────────────────────────────────────────────────────
std::optional<Signal> SDZoneStrategy::evaluate(
    const data::TimeSeries& ts,
    const data::Instrument& instr,
    int64_t bar_index)
{
    indicators::SupplyDemand sd(m_p.sd);
    auto zones = sd.detect(ts);
    if (zones.empty()) return {};

    auto closes = ts.close_window(ts.size());
    double close = closes.back();

    for (auto& z : zones) {
        // Demand zone: price touching zone bottom from above → LONG
        if (!z.is_supply && close >= z.bottom && close <= z.top) {
            Signal s;
            s.direction   = Direction::LONG;
            s.strategy    = name();
            s.entry_price = close;
            s.sl_ref      = z.bottom;
            s.tp_ref      = close + (z.top - z.bottom) * 2.0;
            s.confidence  = 0.62f + 0.02f * std::min(z.strength, 5);
            s.reason      = "Demand zone test";
            s.bar_index   = bar_index;
            return s;
        }
        // Supply zone: price touching zone top from below → SHORT
        if (z.is_supply && close <= z.top && close >= z.bottom) {
            Signal s;
            s.direction   = Direction::SHORT;
            s.strategy    = name();
            s.entry_price = close;
            s.sl_ref      = z.top;
            s.tp_ref      = close - (z.top - z.bottom) * 2.0;
            s.confidence  = 0.62f + 0.02f * std::min(z.strength, 5);
            s.reason      = "Supply zone test";
            s.bar_index   = bar_index;
            return s;
        }
    }
    return {};
}

// ── StrategyMultiplexer ───────────────────────────────────────────────────────
std::optional<Signal> StrategyMultiplexer::evaluate(
    const data::TimeSeries& ts,
    const data::Instrument& instr,
    int64_t bar_index,
    const ml::FeatureComposer& composer,
    const indicators::IchimokuValues& ich,
    const indicators::VolatilityCloudValues& vol)
{
    // Collect all strategy signals
    std::vector<Signal> candidates;
    for (auto& strat : m_strategies) {
        auto sig = strat->evaluate(ts, instr, bar_index);
        if (sig) candidates.push_back(*sig);
    }
    if (candidates.empty()) return {};

    // Sort by confidence descending
    std::sort(candidates.begin(), candidates.end(),
              [](const Signal& a, const Signal& b){
                  return a.confidence > b.confidence; });

    Signal& best = candidates.front();

    // ML gate: run model and check agreement
    if (m_cfg.require_ml_agree && m_model && m_model->is_loaded()) {
        auto features = composer.compose(ts, ich, vol);
        ml::ModelInput in;
        in.features   = features;
        in.shape      = {1, 1, static_cast<int64_t>(features.size())};
        in.symbol     = instr.symbol;
        in.bar_index  = bar_index;

        auto out = m_model->infer(in);

        // Classes: 0=HOLD,1=BUY,2=SELL
        bool ml_long  = (out.predicted_class == 1);
        bool ml_short = (out.predicted_class == 2);

        bool agreed = (best.direction == Direction::LONG  && ml_long)
                   || (best.direction == Direction::SHORT && ml_short);

        if (!agreed) {
            m_log->debug("ML gate rejected signal: strategy={} ml_class={}",
                         best.strategy, out.predicted_class);
            return {};
        }
        // Blend ML confidence into signal confidence
        best.confidence = best.confidence * 0.6f + out.confidence * 0.4f;
    }

    if (best.confidence < m_cfg.min_confidence) return {};

    telemetry::scalar("strategy/confidence", best.confidence, bar_index);
    telemetry::scalar("strategy/direction",
                      static_cast<double>(best.direction), bar_index);

    m_log->info("Signal: {} {} @ {:.5f} conf={:.2f} [{}]",
                best.strategy,
                best.direction == Direction::LONG ? "LONG" : "SHORT",
                best.entry_price, best.confidence, best.reason);
    return best;
}

} // namespace dh::strategy
