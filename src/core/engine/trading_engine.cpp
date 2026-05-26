#include "trading_engine.hpp"
#include "../../storage/checkpoint_manager.hpp"
#include <filesystem>
#include <algorithm>

namespace dh::engine {

// ── Instrument lookup ─────────────────────────────────────────────────────────
data::Instrument TradingEngine::get_instrument(const std::string& sym) const
{
    if (sym == "XAUUSD") return data::instruments::XAUUSD();
    if (sym == "XAUEUR") return data::instruments::XAUEUR();
    if (sym == "XAGUSD") return data::instruments::XAGUSD();
    if (sym == "XAGEUR") return data::instruments::XAGEUR();
    if (sym == "GBPUSD") return data::instruments::GBPUSD();
    if (sym == "EURUSD") return data::instruments::EURUSD();
    if (sym == "BTCUSD") return data::instruments::BTCUSD();
    if (sym == "ETHUSD") return data::instruments::ETHUSD();
    if (sym == "HYPEUSD")return data::instruments::HYPEUSD();
    return {sym, 0.0001, 100000.0, 3.0, 1.0};
}

// ── Constructor ───────────────────────────────────────────────────────────────
TradingEngine::TradingEngine(EngineConfig cfg)
    : m_cfg(std::move(cfg))
    , m_risk(m_cfg.risk)
    , m_composer(m_cfg.features)
    , m_log(dh::logging::get("engine"))
{
    // Model registration is deferred to init() via ModelLoader
}

// ── init ──────────────────────────────────────────────────────────────────────
void TradingEngine::init()
{
    m_log->info("Initialising TradingEngine v1.0");

    // ── Step 1: Discover and load model artifacts via ModelLoader ─────────────
    // The loader walks model_dir, picks up .onnx / .safetensors / manifest files,
    // and populates the ModelRegistry with fully-logged load results.
    {
        ml::loader::ModelLoader loader(m_cfg.model_dir);
        int n = loader.discover_and_load(ml::ModelRegistry::instance());
        m_log->info("ModelLoader: {} model(s) loaded from '{}'",
                    n, m_cfg.model_dir);

        // Import training artifacts into telemetry if present
        std::string tlog = m_cfg.model_dir
                         + "/advanced_model/layer0_rebuild/training_log.csv";
        loader.import_training_log(tlog);

        std::string teljsonl = m_cfg.model_dir
                             + "/advanced_model/train_advanced_telemetry.jsonl";
        loader.import_telemetry_jsonl(teljsonl);
    }

    // ── Step 2: Wire strategies ───────────────────────────────────────────────
    m_mux.add_strategy(
        std::make_unique<strategy::BreakoutStrategy>());
    m_mux.add_strategy(
        std::make_unique<strategy::ReversalStrategy>());
    m_mux.add_strategy(
        std::make_unique<strategy::SDZoneStrategy>());

    // ── Step 3: Attach primary gating model ───────────────────────────────────
    // Prefer layer0_rebuild (ONNX/safetensors) if available, fall back to
    // layer0_merged, then grand_unified, then any loaded model.
    const std::vector<std::string> gate_priority = {
        "layer0_rebuild", "layer0_merged", "layer0", "grand_unified"
    };
    bool gate_attached = false;
    for (auto& key : gate_priority) {
        try {
            auto model = ml::ModelRegistry::instance().get(key);
            m_mux.set_model(model);
            m_log->info("ML gate attached: '{}' v{} loaded={} backend={}",
                        key, model->version(), model->is_loaded(),
                        model->hparams().value("backend", "?"));
            gate_attached = true;
            break;
        } catch (const std::out_of_range&) { /* try next */ }
    }
    if (!gate_attached)
        m_log->warn("No primary gating model found – running without ML gate");

    // ── Step 4: Telemetry hparams ─────────────────────────────────────────────
    telemetry::Telemetry::instance().log_hparams({
        {"model_dir",       m_cfg.model_dir},
        {"checkpoint_dir",  m_cfg.checkpoint_dir},
        {"risk_pct",        m_cfg.risk.risk_pct},
        {"max_dd_pct",      m_cfg.risk.max_dd_pct},
        {"feature_window",  m_cfg.features.window},
        {"project_dim",     m_cfg.features.project_dim},
        {"symbols",         m_cfg.symbols.size()},
        {"live_mode",       m_cfg.live_mode},
        {"registry_size",   ml::ModelRegistry::instance().keys().size()}
    });

    std::filesystem::create_directories(m_cfg.checkpoint_dir);
    m_log->info("Engine ready. Symbols={} Registry={}",
                m_cfg.symbols.size(),
                ml::ModelRegistry::instance().keys().size());
}

// ── on_bar ────────────────────────────────────────────────────────────────────
void TradingEngine::on_bar(const std::string& symbol,
                            data::Timeframe tf,
                            const data::Bar& bar)
{
    std::lock_guard lock(m_mu);
    int64_t step = m_step.fetch_add(1);

    auto& ts = series(symbol, tf);
    ts.push(bar);

    telemetry::Telemetry::instance().begin_event("on_bar");

    update_positions(symbol, bar.close);

    // Only evaluate strategy on H1+ timeframes to reduce noise
    if (tf == data::Timeframe::H1 || tf == data::Timeframe::H4
     || tf == data::Timeframe::D1)
    {
        auto ich = m_ich.calculate(ts);
        auto vol = m_vol.calculate(ts);
        auto instr = get_instrument(symbol);

        auto sig = m_mux.evaluate(ts, instr, step, m_composer, ich, vol);
        if (sig) process_signal(symbol, *sig, instr);
    }

    if (step % m_cfg.checkpoint_freq == 0) save_checkpoint();

    telemetry::Telemetry::instance().end_event("on_bar", step);
    telemetry::scalar("engine/total_pnl", m_total_pnl, step);
    telemetry::scalar("engine/open_positions",
                      static_cast<double>(m_positions.size()), step);

    if (step % 100 == 0) telemetry::Telemetry::instance().flush();
}

// ── run_replay ────────────────────────────────────────────────────────────────
void TradingEngine::run_replay()
{
    m_log->info("Starting replay across {} symbols", m_cfg.symbols.size());
    for (auto& sym : m_cfg.symbols) {
        for (auto tf : m_cfg.timeframes) {
            auto key = sym + ":" + std::to_string(static_cast<int>(tf));
            auto it  = m_series.find(key);
            if (it == m_series.end()) continue;

            data::LFOJitterReplay replayer;
            replayer.replay(it->second, [&](const data::Bar& bar, int64_t){
                on_bar(sym, tf, bar);
            });
        }
    }
    m_log->info("Replay complete. Total P&L: {:.2f}", m_total_pnl);
}

// ── process_signal ────────────────────────────────────────────────────────────
void TradingEngine::process_signal(const std::string& symbol,
                                    const strategy::Signal& sig,
                                    const data::Instrument& instr)
{
    if (m_positions.size() >= static_cast<size_t>(m_cfg.risk.max_positions))
        return;

    auto spec = m_risk.size_position(instr,
                                     sig.entry_price,
                                     sig.sl_ref,
                                     sig.tp_ref,
                                     sig.direction == strategy::Direction::LONG);

    if (spec.lot_size <= 0 || spec.rr_ratio < 1.0) {
        m_log->debug("Signal rejected – bad RR: {:.2f}", spec.rr_ratio);
        return;
    }

    Position pos;
    pos.symbol      = symbol;
    pos.is_long     = sig.direction == strategy::Direction::LONG;
    pos.entry_price = spec.entry_price;
    pos.stop_loss   = spec.stop_loss;
    pos.take_profit = spec.take_profit;
    pos.lot_size    = spec.lot_size;
    pos.open_bar    = sig.bar_index;
    m_positions.push_back(pos);

    if (m_signal_cb) m_signal_cb(sig, spec);

    m_log->info("Position opened: {} {} {:.2f}lots SL={:.5f} TP={:.5f}",
                symbol,
                pos.is_long ? "LONG" : "SHORT",
                pos.lot_size, pos.stop_loss, pos.take_profit);

    telemetry::scalar("trade/entry_price",  spec.entry_price, sig.bar_index);
    telemetry::scalar("trade/lot_size",     spec.lot_size,    sig.bar_index);
    telemetry::scalar("trade/rr_ratio",     spec.rr_ratio,    sig.bar_index);
}

// ── update_positions ──────────────────────────────────────────────────────────
void TradingEngine::update_positions(const std::string& symbol,
                                      double current_price)
{
    std::vector<Position> remaining;
    for (auto& pos : m_positions) {
        if (pos.symbol != symbol) { remaining.push_back(pos); continue; }

        double pnl_per_lot = pos.is_long
            ? (current_price - pos.entry_price)
            : (pos.entry_price - current_price);

        pos.unrealised_pnl = pnl_per_lot * pos.lot_size;

        // Check SL/TP hit
        bool sl_hit = pos.is_long
            ? current_price <= pos.stop_loss
            : current_price >= pos.stop_loss;
        bool tp_hit = pos.is_long
            ? current_price >= pos.take_profit
            : current_price <= pos.take_profit;

        if (sl_hit || tp_hit) {
            double closed_pnl = pnl_per_lot * pos.lot_size;
            m_total_pnl += closed_pnl;
            m_log->info("Position closed: {} {} PnL={:.2f} reason={}",
                         pos.symbol,
                         pos.is_long ? "LONG" : "SHORT",
                         closed_pnl,
                         tp_hit ? "TP" : "SL");
            telemetry::scalar("trade/closed_pnl", closed_pnl, m_step.load());
        } else {
            remaining.push_back(pos);
        }
    }
    m_positions = std::move(remaining);
}

// ── save_checkpoint ───────────────────────────────────────────────────────────
void TradingEngine::save_checkpoint()
{
    storage::CheckpointManager cm(m_cfg.checkpoint_dir);
    nlohmann::json meta = {
        {"step",            m_step.load()},
        {"total_pnl",       m_total_pnl},
        {"open_positions",  m_positions.size()},
        {"telemetry",       telemetry::Telemetry::instance().snapshot()}
    };
    cm.save(meta, m_step.load());
    m_log->info("Checkpoint saved at step {}", m_step.load());
}

// ── series ────────────────────────────────────────────────────────────────────
data::TimeSeries& TradingEngine::series(const std::string& sym,
                                         data::Timeframe tf)
{
    auto key = sym + ":" + std::to_string(static_cast<int>(tf));
    auto it  = m_series.find(key);
    if (it == m_series.end()) {
        m_series.emplace(key, data::TimeSeries(500));
        return m_series.at(key);
    }
    return it->second;
}

// ── shutdown ──────────────────────────────────────────────────────────────────
void TradingEngine::shutdown()
{
    save_checkpoint();
    telemetry::Telemetry::instance().flush();
    dh::logging::flush_all();
    m_log->info("Engine shutdown complete.");
}

} // namespace dh::engine
