#include "trading_engine.hpp"
#include "../../storage/checkpoint_manager.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>

namespace dh::engine {

namespace {
std::string mode_to_string(RuntimeMode mode)
{
    switch (mode) {
        case RuntimeMode::Replay: return "replay";
        case RuntimeMode::Paper: return "paper";
        case RuntimeMode::Live: return "live";
    }
    return "replay";
}
} // namespace

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
    if (sym == "HYPEUSD") return data::instruments::HYPEUSD();
    return {sym, 0.0001, 100000.0, 3.0, 1.0};
}

TradingEngine::TradingEngine(EngineConfig cfg)
    : m_cfg(std::move(cfg))
    , m_risk(m_cfg.risk)
    , m_composer(m_cfg.features)
    , m_log(dh::logging::get("engine"))
{
    if (m_cfg.report_dir.empty()) {
        m_cfg.report_dir = "logs/reports";
    }
}

void TradingEngine::validate_runtime_requirements()
{
    if (!m_cfg.require_model_artifacts) return;
    const std::vector<std::string> required_models = {"layer0_rebuild", "grand_unified"};
    for (const auto& key : required_models) {
        auto model = ml::ModelRegistry::instance().get(key);
        if (!model || !model->is_loaded()) {
            throw std::runtime_error("Required model is missing/unloaded: " + key);
        }
    }
    if (!m_manifest_composer.loaded()) {
        throw std::runtime_error("Manifest feature composer failed to load required manifest");
    }
}

void TradingEngine::init()
{
    std::filesystem::create_directories(m_cfg.checkpoint_dir);
    std::filesystem::create_directories(m_cfg.report_dir);

    m_peak_equity = m_cfg.risk.account_equity;
    const auto action_path = m_cfg.report_dir + "/actions.jsonl";
    m_action_file.open(action_path, std::ios::app);

    log_action("startup", {
        {"runtime_mode", mode_to_string(m_cfg.runtime_mode)},
        {"symbols", m_cfg.symbols},
        {"timeframes", m_cfg.timeframes.size()}
    });
    m_log->info("Initialising TradingEngine v1.0 mode={}", mode_to_string(m_cfg.runtime_mode));

    m_model_loader.load(m_cfg.model_dir);
    m_loaded_models = m_model_loader.loaded_keys();
    m_manifest_composer.load_manifest(
        m_cfg.model_dir + "/layer0_model_l0/grand_unified_manifest.json");
    log_action("model_load", {{"loaded_models", m_loaded_models}});

    m_mux.add_strategy(std::make_unique<strategy::BreakoutStrategy>());
    m_mux.add_strategy(std::make_unique<strategy::ReversalStrategy>());
    m_mux.add_strategy(std::make_unique<strategy::SDZoneStrategy>());

    try {
        auto model = ml::ModelRegistry::instance().get("layer0_rebuild");
        m_mux.set_model(model);
    } catch (...) {
        m_log->warn("layer0_rebuild model not available – running without primary ML gate");
    }

    try {
        auto model = ml::ModelRegistry::instance().get("grand_unified");
        m_mux.set_secondary_model(model);
    } catch (...) {
        m_log->warn("grand_unified model not available – running without secondary ML gate");
    }
    m_mux.set_manifest_composer(&m_manifest_composer);
    validate_runtime_requirements();

    telemetry::Telemetry::instance().log_hparams({
        {"model_dir",       m_cfg.model_dir},
        {"checkpoint_dir",  m_cfg.checkpoint_dir},
        {"risk_pct",        m_cfg.risk.risk_pct},
        {"max_dd_pct",      m_cfg.risk.max_dd_pct},
        {"feature_window",  m_cfg.features.window},
        {"project_dim",     m_cfg.features.project_dim},
        {"symbols",         m_cfg.symbols.size()},
        {"runtime_mode",    mode_to_string(m_cfg.runtime_mode)}
    });

#ifdef USE_BITGET
    if (m_cfg.runtime_mode == RuntimeMode::Paper || m_cfg.runtime_mode == RuntimeMode::Live) {
        init_live_exchange();
    }
#else
    if (m_cfg.runtime_mode == RuntimeMode::Paper || m_cfg.runtime_mode == RuntimeMode::Live) {
        throw std::runtime_error("Bitget runtime requested but build has USE_BITGET=OFF");
    }
#endif

    m_log->info("Engine ready. Symbols={} Models={}",
                m_cfg.symbols.size(),
                ml::ModelRegistry::instance().keys().size());
}

void TradingEngine::log_action(const std::string& action, const nlohmann::json& payload)
{
    nlohmann::json j = {
        {"action", action},
        {"step", m_step.load()},
        {"wall_ms", std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count()},
        {"mode", mode_to_string(m_cfg.runtime_mode)}
    };
    if (!payload.is_null() && !payload.empty()) j["data"] = payload;
    if (m_action_file.is_open()) {
        m_action_file << j.dump() << "\n";
        m_action_file.flush();
    }
    telemetry::scalar("action/" + action, 1.0, m_step.load());
}

std::string TradingEngine::next_correlation_id(const std::string& prefix)
{
    auto seq = m_corr_seq.fetch_add(1) + 1;
    return prefix + "_" + std::to_string(m_step.load()) + "_" + std::to_string(seq);
}

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

    if (tf == data::Timeframe::H1 || tf == data::Timeframe::H4 || tf == data::Timeframe::D1) {
        auto ich = m_ich.calculate(ts);
        auto vol = m_vol.calculate(ts);
        auto instr = get_instrument(symbol);
        auto sig = m_mux.evaluate(ts, instr, step, m_composer, &m_manifest_composer, ich, vol);
        if (sig) process_signal(symbol, *sig, instr);
    }

    if (step % m_cfg.checkpoint_freq == 0) save_checkpoint();
    if (m_cfg.report_freq > 0 && step % m_cfg.report_freq == 0) generate_reports(false);

    telemetry::Telemetry::instance().end_event("on_bar", step);
    telemetry::scalar("engine/total_pnl", m_total_pnl, step);
    telemetry::scalar("engine/open_positions", static_cast<double>(m_positions.size()), step);

    if (step % 100 == 0) telemetry::Telemetry::instance().flush();
}

void TradingEngine::run_replay()
{
    m_log->info("Starting replay across {} symbols", m_cfg.symbols.size());
    for (auto& sym : m_cfg.symbols) {
        for (auto tf : m_cfg.timeframes) {
            auto key = sym + ":" + std::to_string(static_cast<int>(tf));
            auto it  = m_series.find(key);
            if (it == m_series.end()) continue;

            data::LFOJitterReplay replayer;
            replayer.replay(it->second, [&](const data::Bar& bar, int64_t) {
                on_bar(sym, tf, bar);
            });
        }
    }
    m_log->info("Replay complete. Total P&L: {:.2f}", m_total_pnl);
}

void TradingEngine::process_signal(const std::string& symbol,
                                   const strategy::Signal& sig,
                                   const data::Instrument& instr)
{
    const auto corr = next_correlation_id("sig");
    log_action("signal_evaluated", {
        {"correlation_id", corr},
        {"symbol", symbol},
        {"direction", static_cast<int>(sig.direction)},
        {"confidence", sig.confidence},
        {"strategy", sig.strategy}
    });

    if (m_positions.size() >= static_cast<size_t>(m_cfg.risk.max_positions)) {
        log_action("risk_rejection", {
            {"correlation_id", corr},
            {"reason", "max_positions"},
            {"max_positions", m_cfg.risk.max_positions}
        });
        return;
    }

    if (m_cfg.max_daily_loss_abs > 0.0 && std::abs(std::min(0.0, m_total_pnl)) >= m_cfg.max_daily_loss_abs) {
        m_daily_loss_triggered = true;
        log_action("risk_rejection", {
            {"correlation_id", corr},
            {"reason", "max_daily_loss"},
            {"max_daily_loss_abs", m_cfg.max_daily_loss_abs}
        });
        return;
    }

#ifdef USE_BITGET
    if ((m_cfg.runtime_mode == RuntimeMode::Paper || m_cfg.runtime_mode == RuntimeMode::Live) && m_exchange) {
        sync_account_equity();
    }
#endif

    auto spec = m_risk.size_position(instr,
                                     sig.entry_price,
                                     sig.sl_ref,
                                     sig.tp_ref,
                                     sig.direction == strategy::Direction::LONG);

    if (spec.lot_size <= 0 || spec.rr_ratio < 1.0) {
        log_action("signal_rejected", {
            {"correlation_id", corr},
            {"reason", "bad_rr_or_lot"},
            {"rr_ratio", spec.rr_ratio},
            {"lot_size", spec.lot_size}
        });
        return;
    }

    Position pos;
    pos.id          = next_correlation_id("pos");
    pos.correlation_id = corr;
    pos.symbol      = symbol;
    pos.is_long     = sig.direction == strategy::Direction::LONG;
    pos.entry_price = spec.entry_price;
    pos.stop_loss   = spec.stop_loss;
    pos.take_profit = spec.take_profit;
    pos.lot_size    = spec.lot_size;
    pos.rr_ratio    = spec.rr_ratio;
    pos.open_bar    = sig.bar_index;
    m_positions.push_back(pos);

    if (m_signal_cb) m_signal_cb(sig, spec);

    log_action("signal_accepted", {
        {"position_id", pos.id},
        {"correlation_id", corr},
        {"symbol", symbol},
        {"lot_size", pos.lot_size},
        {"rr_ratio", pos.rr_ratio}
    });

    telemetry::scalar("trade/entry_price", spec.entry_price, sig.bar_index);
    telemetry::scalar("trade/lot_size", spec.lot_size, sig.bar_index);
    telemetry::scalar("trade/rr_ratio", spec.rr_ratio, sig.bar_index);

#ifdef USE_BITGET
    if ((m_cfg.runtime_mode == RuntimeMode::Paper || m_cfg.runtime_mode == RuntimeMode::Live) && m_exchange) {
        if (m_cfg.kill_switch) {
            log_action("order_rejected", {
                {"correlation_id", corr},
                {"reason", "kill_switch_enabled"}
            });
        } else {
            execute_order(symbol, pos.is_long, pos.lot_size);
        }
    }
#endif
}

void TradingEngine::update_positions(const std::string& symbol, double current_price)
{
    std::vector<Position> remaining;
    for (auto& pos : m_positions) {
        if (pos.symbol != symbol) {
            remaining.push_back(pos);
            continue;
        }

        double pnl_per_lot = pos.is_long
            ? (current_price - pos.entry_price)
            : (pos.entry_price - current_price);

        pos.unrealised_pnl = pnl_per_lot * pos.lot_size;

        bool sl_hit = pos.is_long ? current_price <= pos.stop_loss : current_price >= pos.stop_loss;
        bool tp_hit = pos.is_long ? current_price >= pos.take_profit : current_price <= pos.take_profit;

        if (sl_hit || tp_hit) {
            double closed_pnl = pnl_per_lot * pos.lot_size;
            m_total_pnl += closed_pnl;
            const auto equity = m_cfg.risk.account_equity + m_total_pnl;
            m_peak_equity = std::max(m_peak_equity, equity);
            m_max_drawdown_abs = std::max(m_max_drawdown_abs, m_peak_equity - equity);

            TradeLedgerEntry ledger;
            ledger.id = pos.id;
            ledger.correlation_id = pos.correlation_id;
            ledger.symbol = pos.symbol;
            ledger.is_long = pos.is_long;
            ledger.entry_price = pos.entry_price;
            ledger.exit_price = current_price;
            ledger.lot_size = pos.lot_size;
            ledger.pnl = closed_pnl;
            ledger.rr_ratio = pos.rr_ratio;
            ledger.open_step = pos.open_bar;
            ledger.close_step = m_step.load();
            ledger.close_reason = tp_hit ? "TP" : "SL";
            m_trade_ledger.push_back(ledger);

            log_action("position_closed", {
                {"position_id", pos.id},
                {"correlation_id", pos.correlation_id},
                {"symbol", pos.symbol},
                {"pnl", closed_pnl},
                {"reason", ledger.close_reason}
            });
            telemetry::scalar("trade/closed_pnl", closed_pnl, m_step.load());
        } else {
            remaining.push_back(pos);
        }
    }
    m_positions = std::move(remaining);
}

void TradingEngine::save_checkpoint()
{
    storage::CheckpointManager cm(m_cfg.checkpoint_dir);
    nlohmann::json meta = {
        {"step",            m_step.load()},
        {"total_pnl",       m_total_pnl},
        {"open_positions",  m_positions.size()},
        {"loaded_models",   m_loaded_models},
        {"snapshot", {
            {"max_drawdown_abs", m_max_drawdown_abs},
            {"peak_equity", m_peak_equity},
            {"positions", m_positions.size()},
            {"closed_trades", m_trade_ledger.size()}
        }},
        {"weights", {
            {"risk_pct", m_cfg.risk.risk_pct},
            {"max_positions", m_cfg.risk.max_positions},
            {"feature_window", m_cfg.features.window},
            {"feature_project_dim", m_cfg.features.project_dim}
        }},
        {"telemetry",       telemetry::Telemetry::instance().snapshot()}
    };
    cm.save(meta, m_step.load());
    log_action("checkpoint_saved", {{"step", m_step.load()}});
}

void TradingEngine::generate_reports(bool final_report)
{
    const auto step_now = m_step.load();
    const auto unrealized = std::accumulate(
        m_positions.begin(), m_positions.end(), 0.0,
        [](double acc, const Position& p) { return acc + p.unrealised_pnl; });
    const auto total = m_total_pnl + unrealized;

    size_t wins = 0;
    size_t losses = 0;
    double rr_sum = 0.0;
    for (const auto& t : m_trade_ledger) {
        if (t.pnl >= 0) ++wins; else ++losses;
        rr_sum += t.rr_ratio;
    }
    const auto closed_count = m_trade_ledger.size();
    const double win_rate = closed_count ? (100.0 * static_cast<double>(wins) / static_cast<double>(closed_count)) : 0.0;
    const double avg_rr = closed_count ? rr_sum / static_cast<double>(closed_count) : 0.0;

    nlohmann::json per_symbol = nlohmann::json::object();
    for (const auto& t : m_trade_ledger) {
        auto& node = per_symbol[t.symbol];
        node["realized_pnl"] = node.value("realized_pnl", 0.0) + t.pnl;
        node["closed_trades"] = node.value("closed_trades", 0) + 1;
        node["wins"] = node.value("wins", 0) + (t.pnl >= 0 ? 1 : 0);
    }
    for (const auto& p : m_positions) {
        auto& node = per_symbol[p.symbol];
        node["unrealized_pnl"] = node.value("unrealized_pnl", 0.0) + p.unrealised_pnl;
        node["open_positions"] = node.value("open_positions", 0) + 1;
    }

    nlohmann::json ledger = nlohmann::json::array();
    for (const auto& t : m_trade_ledger) {
        ledger.push_back({
            {"id", t.id},
            {"correlation_id", t.correlation_id},
            {"symbol", t.symbol},
            {"side", t.is_long ? "long" : "short"},
            {"entry_price", t.entry_price},
            {"exit_price", t.exit_price},
            {"lot_size", t.lot_size},
            {"pnl", t.pnl},
            {"rr_ratio", t.rr_ratio},
            {"fees", t.fees},
            {"slippage", t.slippage},
            {"open_step", t.open_step},
            {"close_step", t.close_step},
            {"close_reason", t.close_reason}
        });
    }

    nlohmann::json report = {
        {"mode", mode_to_string(m_cfg.runtime_mode)},
        {"step", step_now},
        {"realized_pnl", m_total_pnl},
        {"unrealized_pnl", unrealized},
        {"total_pnl", total},
        {"max_drawdown_abs", m_max_drawdown_abs},
        {"open_positions", m_positions.size()},
        {"closed_trades", closed_count},
        {"win_rate", win_rate},
        {"avg_rr", avg_rr},
        {"per_symbol", per_symbol},
        {"trade_ledger", ledger}
    };

    const std::string stem = final_report ? "final" : ("step_" + std::to_string(step_now));
    std::filesystem::create_directories(m_cfg.report_dir);

    std::ofstream jf(m_cfg.report_dir + "/pnl_" + stem + ".json", std::ios::trunc);
    jf << report.dump(2) << "\n";
    jf.close();

    std::ofstream md(m_cfg.report_dir + "/pnl_" + stem + ".md", std::ios::trunc);
    md << "# PnL Report (" << stem << ")\n\n";
    md << "- Mode: `" << mode_to_string(m_cfg.runtime_mode) << "`\n";
    md << "- Step: `" << step_now << "`\n";
    md << "- Realized PnL: `" << std::fixed << std::setprecision(2) << m_total_pnl << "`\n";
    md << "- Unrealized PnL: `" << unrealized << "`\n";
    md << "- Total PnL: `" << total << "`\n";
    md << "- Win rate: `" << win_rate << "%`\n";
    md << "- Avg RR: `" << avg_rr << "`\n";
    md << "- Max drawdown (abs): `" << m_max_drawdown_abs << "`\n";
    md << "- Open positions: `" << m_positions.size() << "`\n";
    md << "- Closed trades: `" << closed_count << "`\n\n";
    md << "## Per-symbol\n\n";
    for (auto it = per_symbol.begin(); it != per_symbol.end(); ++it) {
        md << "- `" << it.key() << "`: "
           << "realized=" << it.value().value("realized_pnl", 0.0)
           << ", unrealized=" << it.value().value("unrealized_pnl", 0.0)
           << ", closed=" << it.value().value("closed_trades", 0)
           << ", open=" << it.value().value("open_positions", 0)
           << "\n";
    }
    md.close();

    log_action(final_report ? "report_final" : "report_periodic", {
        {"path_json", m_cfg.report_dir + "/pnl_" + stem + ".json"},
        {"path_md", m_cfg.report_dir + "/pnl_" + stem + ".md"}
    });
}

data::TimeSeries& TradingEngine::series(const std::string& sym, data::Timeframe tf)
{
    auto key = sym + ":" + std::to_string(static_cast<int>(tf));
    auto it  = m_series.find(key);
    if (it == m_series.end()) {
        m_series.emplace(key, data::TimeSeries(500));
        return m_series.at(key);
    }
    return it->second;
}

std::string TradingEngine::map_symbol_to_exchange(const std::string& symbol) const
{
    auto it = m_symbol_to_exchange.find(symbol);
    if (it != m_symbol_to_exchange.end()) return it->second;
    auto custom = m_cfg.exchange.symbol_map.find(symbol);
    if (custom != m_cfg.exchange.symbol_map.end()) return custom->second;
    if (symbol.size() > 3 && symbol.substr(symbol.size() - 3) == "USD") {
        return symbol.substr(0, symbol.size() - 3) + "USDT";
    }
    return symbol;
}

std::string TradingEngine::map_symbol_to_internal(const std::string& symbol) const
{
    auto it = m_symbol_to_internal.find(symbol);
    if (it != m_symbol_to_internal.end()) return it->second;
    return symbol;
}

std::string TradingEngine::timeframe_to_bitget_granularity(data::Timeframe tf)
{
    switch (tf) {
        case data::Timeframe::M1: return "1m";
        case data::Timeframe::M5: return "5m";
        case data::Timeframe::M15: return "15m";
        case data::Timeframe::M30: return "30m";
        case data::Timeframe::H1: return "1H";
        case data::Timeframe::H4: return "4H";
        case data::Timeframe::D1: return "1D";
        default: return "1m";
    }
}

std::string TradingEngine::timeframe_to_channel(data::Timeframe tf)
{
    switch (tf) {
        case data::Timeframe::M1: return "candle1m";
        case data::Timeframe::M5: return "candle5m";
        case data::Timeframe::M15: return "candle15m";
        case data::Timeframe::M30: return "candle30m";
        case data::Timeframe::H1: return "candle1H";
        case data::Timeframe::H4: return "candle4H";
        case data::Timeframe::D1: return "candle1D";
        default: return "candle1m";
    }
}

data::Timeframe TradingEngine::channel_to_timeframe(const std::string& channel)
{
    if (channel == "candle1m") return data::Timeframe::M1;
    if (channel == "candle5m") return data::Timeframe::M5;
    if (channel == "candle15m") return data::Timeframe::M15;
    if (channel == "candle30m") return data::Timeframe::M30;
    if (channel == "candle1H") return data::Timeframe::H1;
    if (channel == "candle4H") return data::Timeframe::H4;
    if (channel == "candle1D") return data::Timeframe::D1;
    return data::Timeframe::M1;
}

std::string TradingEngine::timeframe_to_string(data::Timeframe tf)
{
    switch (tf) {
        case data::Timeframe::M1: return "M1";
        case data::Timeframe::M5: return "M5";
        case data::Timeframe::M15: return "M15";
        case data::Timeframe::M30: return "M30";
        case data::Timeframe::H1: return "H1";
        case data::Timeframe::H4: return "H4";
        case data::Timeframe::D1: return "D1";
        default: return "M1";
    }
}

void TradingEngine::shutdown()
{
    generate_reports(true);
    save_checkpoint();
    telemetry::Telemetry::instance().flush();
    dh::logging::flush_all();
#ifdef USE_BITGET
    if (m_exchange) m_exchange->stop();
#endif
    log_action("shutdown", {{"total_pnl", m_total_pnl}, {"open_positions", m_positions.size()}});
    if (m_action_file.is_open()) m_action_file.close();
    m_log->info("Engine shutdown complete.");
}

#ifdef USE_BITGET
void TradingEngine::warmup_market_data()
{
    if (!m_exchange) return;
    for (const auto& internal_sym : m_cfg.symbols) {
        const auto exchange_sym = map_symbol_to_exchange(internal_sym);
        for (const auto tf : m_cfg.timeframes) {
            const auto granularity = timeframe_to_bitget_granularity(tf);
            try {
                auto candles = m_exchange->rest().get_candles(exchange_sym, granularity, m_cfg.warmup_bars);
                std::sort(candles.begin(), candles.end(),
                          [](const auto& a, const auto& b) { return a.timestamp < b.timestamp; });
                auto& ts = series(internal_sym, tf);
                for (const auto& c : candles) {
                    ts.push(c.to_bar(tf));
                }
                log_action("market_warmup", {
                    {"symbol", internal_sym},
                    {"exchange_symbol", exchange_sym},
                    {"timeframe", timeframe_to_string(tf)},
                    {"bars", candles.size()}
                });
            } catch (const std::exception& e) {
                log_action("market_warmup_failed", {
                    {"symbol", internal_sym},
                    {"timeframe", timeframe_to_string(tf)},
                    {"error", e.what()}
                });
            }
        }
    }
}

void TradingEngine::init_live_exchange()
{
    auto& ec = m_cfg.exchange;
    exchange::bitget::BitgetAuth auth(ec.api_key, ec.secret_key, ec.passphrase);
    m_exchange = std::make_unique<exchange::bitget::BitgetClient>(
        std::move(auth), ec.product_type, ec.sandbox);

    for (const auto& s : m_cfg.symbols) {
        auto mapped = map_symbol_to_exchange(s);
        m_symbol_to_exchange[s] = mapped;
        m_symbol_to_internal[mapped] = s;
    }

    log_action("exchange_startup", {
        {"sandbox", ec.sandbox},
        {"product_type", ec.product_type},
        {"mode", mode_to_string(m_cfg.runtime_mode)}
    });

    try {
        auto remote_positions = m_exchange->rest().get_positions();
        for (auto& rp : remote_positions) {
            Position pos;
            pos.id = next_correlation_id("hydrated");
            pos.correlation_id = next_correlation_id("hydrate");
            pos.symbol = map_symbol_to_internal(rp.symbol);
            pos.is_long = rp.hold_side == "long";
            pos.entry_price = rp.avg_open_price;
            pos.lot_size = rp.total;
            pos.stop_loss = 0;
            pos.take_profit = 0;
            pos.open_bar = 0;
            pos.unrealised_pnl = rp.unrealised_pnl;
            m_positions.push_back(pos);
        }
        log_action("account_sync", {{"positions", remote_positions.size()}});
    } catch (const std::exception& e) {
        log_action("account_sync_failed", {{"error", e.what()}});
    }

    warmup_market_data();

    const auto ws_tf = m_cfg.timeframes.empty() ? data::Timeframe::M1 : m_cfg.timeframes.front();
    const auto ws_channel = timeframe_to_channel(ws_tf);
    std::vector<std::string> exchange_symbols;
    exchange_symbols.reserve(m_cfg.symbols.size());
    for (const auto& sym : m_cfg.symbols) exchange_symbols.push_back(map_symbol_to_exchange(sym));

    m_exchange->start_market_data(
        exchange_symbols,
        [this](const exchange::bitget::BitgetTicker& ticker) {
            const auto internal = map_symbol_to_internal(ticker.symbol);
            on_bar(internal, data::Timeframe::M1, ticker.to_bar());
        },
        [this](const exchange::bitget::BitgetCandle& candle,
               const std::string& exchange_symbol,
               const std::string& channel) {
            const auto internal = map_symbol_to_internal(exchange_symbol);
            const auto tf = channel_to_timeframe(channel);
            on_bar(internal, tf, candle.to_bar(tf));
        },
        ws_channel
    );

    m_exchange->start_private_feed(
        [this](const exchange::bitget::BitgetOrder& order) {
            log_action("order_update", {
                {"order_id", order.order_id},
                {"symbol", map_symbol_to_internal(order.symbol)},
                {"filled", order.filled},
                {"state", static_cast<int>(order.state)}
            });
        }
    );

    log_action("market_subscribe", {
        {"symbols", exchange_symbols},
        {"channel", ws_channel}
    });
}

void TradingEngine::execute_order(const std::string& symbol,
                                  bool is_long,
                                  double lot_size)
{
    if (!m_exchange) return;
    exchange::bitget::PlaceOrderRequest req;
    req.symbol       = map_symbol_to_exchange(symbol);
    req.product_type = m_cfg.exchange.product_type;
    req.side         = is_long ? "buy" : "sell";
    req.trade_side   = "open";
    req.order_type   = "market";
    req.size         = lot_size;
    req.client_oid   = next_correlation_id("ord");
    try {
        log_action("order_submit", {
            {"symbol", symbol},
            {"exchange_symbol", req.symbol},
            {"side", req.side},
            {"size", lot_size},
            {"client_oid", req.client_oid}
        });
        auto order = m_exchange->rest().place_order(req);
        log_action("order_submitted", {
            {"symbol", symbol},
            {"order_id", order.order_id},
            {"client_oid", req.client_oid}
        });
        telemetry::scalar("live/order_lot_size", lot_size, m_step.load());
    } catch (const std::exception& e) {
        log_action("order_submit_failed", {{"symbol", symbol}, {"error", e.what()}});
        m_log->error("Failed to place order for {}: {}", symbol, e.what());
    }
}

void TradingEngine::sync_account_equity()
{
    if (!m_exchange) return;
    try {
        auto account = m_exchange->rest().get_account();
        if (account.equity > 0) {
            m_risk.set_equity(account.equity);
            telemetry::scalar("account/equity", account.equity, m_step.load());
        }
    } catch (const std::exception& e) {
        log_action("account_sync_failed", {{"error", e.what()}});
    }
}
#endif

} // namespace dh::engine
