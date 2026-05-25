#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Daydream Handler – Strategy System
// Breakout, Reversal, Supply/Demand entry strategies with multiplexed MTF
// confirmation, decision tree gating, and ML signal fusion.
// ─────────────────────────────────────────────────────────────────────────────
#include "../../data/market_data.hpp"
#include "../../data/timeseries.hpp"
#include "../../core/indicators/ichimoku.hpp"
#include "../../core/indicators/volatility_cloud.hpp"
#include "../../core/indicators/supply_demand.hpp"
#include "../../core/indicators/liquidity.hpp"
#include "../../core/indicators/fibonacci.hpp"
#include "../../core/risk/risk_manager.hpp"
#include "../../ml/inference/inference_engine.hpp"
#include "../../ml/inference/feature_composer.hpp"
#include "../../logging/logger.hpp"
#include "../../logging/telemetry.hpp"
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <optional>

namespace dh::strategy {

// ── Signal types ──────────────────────────────────────────────────────────────
enum class Direction { NONE, LONG, SHORT };

struct Signal {
    Direction   direction   = Direction::NONE;
    std::string strategy;
    double      entry_price = 0;
    double      sl_ref      = 0;
    double      tp_ref      = 0;
    float       confidence  = 0;
    std::string reason;
    int64_t     bar_index   = 0;
};

// ── MTF Confirmation ──────────────────────────────────────────────────────────
struct MTFContext {
    data::TimeSeries* h4 = nullptr;
    data::TimeSeries* h1 = nullptr;
    data::TimeSeries* m15 = nullptr;
    data::TimeSeries* m5  = nullptr;
};

// ── Decision Tree Gate ────────────────────────────────────────────────────────
// A simple binary tree for pre-ML gating.
struct DTNode {
    int    feature_index = -1;  // -1 = leaf
    double threshold     = 0;
    int    left_class    = -1;
    int    right_class   = -1;
    std::unique_ptr<DTNode> left;
    std::unique_ptr<DTNode> right;

    int classify(const std::vector<float>& features) const {
        if (feature_index < 0) return left_class;
        if (static_cast<size_t>(feature_index) >= features.size())
            return left_class;
        if (features[static_cast<size_t>(feature_index)] <= threshold)
            return left  ? left->classify(features)  : left_class;
        return right ? right->classify(features) : right_class;
    }
};

// ── Base Strategy ─────────────────────────────────────────────────────────────
class IStrategy {
public:
    virtual ~IStrategy() = default;
    virtual std::string name() const = 0;
    virtual std::optional<Signal> evaluate(
        const data::TimeSeries& ts,
        const data::Instrument& instr,
        int64_t bar_index) = 0;
};

// ── Strategy params (forward-declared before their classes) ───────────────────
struct BreakoutParams {
    int    lookback  = 20;
    double threshold = 0.001;
};

struct ReversalParams {
    indicators::IchimokuParams      ichi;
    indicators::VolatilityCloudParams vol;
};

struct SDZoneParams {
    indicators::SDZoneParams sd;
};

struct MultiplexerConfig {
    float min_confidence   = 0.55f;
    bool  require_ml_agree = true;
    bool  require_dt_gate  = true;
};

// ── Breakout Strategy ────────────────────────────────────────────────────────
class BreakoutStrategy : public IStrategy {
public:
    using Params = BreakoutParams;
    explicit BreakoutStrategy(Params p = Params{}) : m_p(p) {}
    std::string name() const override { return "breakout"; }
    std::optional<Signal> evaluate(const data::TimeSeries& ts,
                                   const data::Instrument& instr,
                                   int64_t bar_index) override;
private:
    Params m_p;
};

// ── Reversal Strategy ─────────────────────────────────────────────────────────
class ReversalStrategy : public IStrategy {
public:
    using Params = ReversalParams;
    explicit ReversalStrategy(Params p = Params{}) : m_p(p) {}
    std::string name() const override { return "reversal"; }
    std::optional<Signal> evaluate(const data::TimeSeries& ts,
                                   const data::Instrument& instr,
                                   int64_t bar_index) override;
private:
    Params m_p;
};

// ── Supply/Demand Zone Strategy ───────────────────────────────────────────────
class SDZoneStrategy : public IStrategy {
public:
    using Params = SDZoneParams;
    explicit SDZoneStrategy(Params p = Params{}) : m_p(p) {}
    std::string name() const override { return "sd_zone"; }
    std::optional<Signal> evaluate(const data::TimeSeries& ts,
                                   const data::Instrument& instr,
                                   int64_t bar_index) override;
private:
    Params m_p;
};

// ── Strategy Multiplexer (combines all strategies + ML gate) ──────────────────
class StrategyMultiplexer {
public:
    using Config = MultiplexerConfig;

    explicit StrategyMultiplexer(Config cfg = Config{}) : m_cfg(cfg) {}

    void add_strategy(std::unique_ptr<IStrategy> s) {
        m_strategies.push_back(std::move(s));
    }

    void set_model(std::shared_ptr<ml::IModel> m) { m_model = m; }

    /// Returns the highest-confidence signal after gating.
    std::optional<Signal> evaluate(
        const data::TimeSeries& ts,
        const data::Instrument& instr,
        int64_t bar_index,
        const ml::FeatureComposer& composer,
        const indicators::IchimokuValues& ich,
        const indicators::VolatilityCloudValues& vol);

private:
    Config                                m_cfg;
    std::vector<std::unique_ptr<IStrategy>> m_strategies;
    std::shared_ptr<ml::IModel>           m_model;
    std::shared_ptr<spdlog::logger>       m_log
        = dh::logging::get("strategy");
};

} // namespace dh::strategy
