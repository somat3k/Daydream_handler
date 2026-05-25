#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Daydream Handler – Trading Engine
// Orchestrates the full workflow: data → features → strategy → risk → execution.
// Manages open positions, P&L tracking, checkpoint saving.
// ─────────────────────────────────────────────────────────────────────────────
#include "../../data/market_data.hpp"
#include "../../data/timeseries.hpp"
#include "../../data/data_pipeline.hpp"
#include "../../core/indicators/ichimoku.hpp"
#include "../../core/indicators/volatility_cloud.hpp"
#include "../../core/strategy/strategy.hpp"
#include "../../core/risk/risk_manager.hpp"
#include "../../ml/inference/inference_engine.hpp"
#include "../../ml/inference/feature_composer.hpp"
#include "../../logging/logger.hpp"
#include "../../logging/telemetry.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <functional>
#include <atomic>
#include <mutex>

namespace dh::engine {

// ── Open position ──────────────────────────────────────────────────────────
struct Position {
    std::string symbol;
    bool        is_long;
    double      entry_price;
    double      stop_loss;
    double      take_profit;
    double      lot_size;
    int64_t     open_bar;
    double      unrealised_pnl = 0;
};

// ── Engine configuration ──────────────────────────────────────────────────
struct EngineConfig {
    std::vector<std::string> symbols = {
        "XAUUSD","XAUEUR","XAGUSD","XAGEUR",
        "GBPUSD","EURUSD","BTCUSD","ETHUSD","HYPEUSD"
    };
    std::vector<data::Timeframe> timeframes = {
        data::Timeframe::M1, data::Timeframe::M5, data::Timeframe::M15,
        data::Timeframe::H1, data::Timeframe::H4, data::Timeframe::D1
    };
    std::string  model_dir       = "models";
    std::string  checkpoint_dir  = "checkpoints";
    std::string  data_dir        = "data";
    int          checkpoint_freq = 1000;  // bars between checkpoints
    bool         live_mode       = false;
    risk::RiskConfig risk;
    ml::FeatureConfig features;
};

// ── Trading Engine ─────────────────────────────────────────────────────────
class TradingEngine {
public:
    explicit TradingEngine(EngineConfig cfg = {});

    /// Initialise components and load models.
    void init();

    /// Process a new bar for a given symbol/timeframe.
    void on_bar(const std::string& symbol,
                data::Timeframe tf,
                const data::Bar& bar);

    /// Run full replay on loaded data.
    void run_replay();

    /// Access per-symbol timeseries.
    data::TimeSeries& series(const std::string& sym, data::Timeframe tf);

    /// Current open positions.
    const std::vector<Position>& positions() const { return m_positions; }

    /// Cumulative P&L.
    double total_pnl() const { return m_total_pnl; }

    /// Engine step counter.
    int64_t step() const { return m_step.load(); }

    /// Shutdown cleanly.
    void shutdown();

    using OnSignalCb = std::function<void(const strategy::Signal&,
                                          const risk::PositionSpec&)>;
    void on_signal(OnSignalCb cb) { m_signal_cb = std::move(cb); }

private:
    void process_signal(const std::string& symbol,
                        const strategy::Signal& sig,
                        const data::Instrument& instr);
    void update_positions(const std::string& symbol, double current_price);
    void save_checkpoint();
    data::Instrument get_instrument(const std::string& sym) const;

    EngineConfig                                m_cfg;
    std::unordered_map<std::string,data::TimeSeries> m_series;
    strategy::StrategyMultiplexer               m_mux;
    risk::RiskManager                           m_risk;
    ml::FeatureComposer                         m_composer;
    indicators::Ichimoku                        m_ich;
    indicators::VolatilityCloud                 m_vol;
    std::vector<Position>                       m_positions;
    std::atomic<int64_t>                        m_step{0};
    double                                      m_total_pnl = 0;
    mutable std::mutex                          m_mu;
    OnSignalCb                                  m_signal_cb;
    std::shared_ptr<spdlog::logger>             m_log;
};

} // namespace dh::engine
