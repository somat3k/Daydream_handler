#include "cli.hpp"
#include "../storage/checkpoint_manager.hpp"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <stdexcept>
#include <cstring>
#include <thread>

namespace dh::cli {
namespace {

std::string trim(std::string s)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), s.end());
    return s;
}

std::vector<std::string> split_csv(const std::string& in)
{
    std::vector<std::string> out;
    std::stringstream ss(in);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        tok = trim(tok);
        if (!tok.empty()) out.push_back(tok);
    }
    return out;
}

std::string upper(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return s;
}

data::Timeframe parse_tf(const std::string& tf_raw)
{
    const auto tf = upper(tf_raw);
    if (tf == "M1") return data::Timeframe::M1;
    if (tf == "M5") return data::Timeframe::M5;
    if (tf == "M15") return data::Timeframe::M15;
    if (tf == "M30") return data::Timeframe::M30;
    if (tf == "H1") return data::Timeframe::H1;
    if (tf == "H4") return data::Timeframe::H4;
    if (tf == "D1") return data::Timeframe::D1;
    if (tf == "W1") return data::Timeframe::W1;
    if (tf == "MN") return data::Timeframe::MN;
    throw std::invalid_argument("Unsupported timeframe: " + tf_raw);
}

std::vector<data::Timeframe> parse_timeframes(const std::string& tf_csv)
{
    std::vector<data::Timeframe> out;
    std::set<data::Timeframe> seen;
    for (const auto& tf : split_csv(tf_csv)) {
        auto p = parse_tf(tf);
        if (seen.insert(p).second) out.push_back(p);
    }
    if (out.empty()) out.push_back(data::Timeframe::H1);
    return out;
}

std::vector<std::string> parse_symbols(const Args& args)
{
    auto symbols = split_csv(args.symbols);
    if (symbols.empty() && !args.symbol.empty()) symbols.push_back(args.symbol);
    if (symbols.empty()) symbols.push_back("XAUUSD");
    return symbols;
}

std::string env_or_default(const char* key, std::string current)
{
    if (!current.empty()) return current;
    if (const char* value = std::getenv(key)) return value;
    return current;
}

engine::RuntimeMode parse_mode(const Args& args)
{
    std::string raw = args.mode;
    if (raw.empty()) raw = args.command;
    const auto mode = upper(raw);
    if (mode == "REPLAY") return engine::RuntimeMode::Replay;
    if (mode == "PAPER")  return engine::RuntimeMode::Paper;
    if (mode == "LIVE")   return engine::RuntimeMode::Live;
    throw std::invalid_argument("Unsupported mode: " + raw);
}

int circling_offset(int epoch, int range)
{
    if (range <= 0 || epoch <= 0) return 0;
    const int span = 2 * range;
    const int idx = (epoch - 1) % span;
    const int magnitude = (idx / 2) + 1;
    return (idx % 2 == 0) ? magnitude : -magnitude;
}

double tuned_risk_for_epoch(const Args& args, int epoch)
{
    const int offset = circling_offset(epoch, args.tune_range);
    const double tuned = args.risk_pct + (static_cast<double>(offset) * args.tune_risk_step);
    return std::max(0.0001, tuned);
}

void write_replay_artifacts(const Args& args,
                            int epoch,
                            const engine::EngineConfig& cfg,
                            const engine::TradingEngine& eng,
                            bool best_epoch)
{
    storage::CheckpointManager cm(args.ckpt_dir);
    nlohmann::json snapshot = {
        {"epoch", epoch + 1},
        {"step", eng.step()},
        {"total_pnl", eng.total_pnl()},
        {"open_positions", eng.positions().size()},
        {"params", {
            {"risk_pct", cfg.risk.risk_pct},
            {"max_positions", cfg.risk.max_positions},
            {"checkpoint_freq", cfg.checkpoint_freq},
            {"report_freq", cfg.report_freq}
        }},
        {"weights", {
            {"pnl_weight", 1.0},
            {"drawdown_weight", 0.0},
            {"position_penalty_weight", 0.0}
        }},
        {"best_epoch", best_epoch}
    };
    cm.save(snapshot, eng.step());

    const std::string tuning_dir = args.log_dir + "/replay_tuning";
    std::filesystem::create_directories(tuning_dir);

    std::ostringstream epoch_stem;
    epoch_stem << "epoch_" << std::setw(3) << std::setfill('0') << (epoch + 1);

    std::ofstream snapshot_file(tuning_dir + "/" + epoch_stem.str() + "_snapshot.json", std::ios::trunc);
    snapshot_file << snapshot.dump(2) << "\n";
    snapshot_file.close();

    nlohmann::json weights = snapshot["weights"];
    weights["epoch"] = epoch + 1;
    weights["risk_pct"] = cfg.risk.risk_pct;
    std::ofstream weights_file(tuning_dir + "/" + epoch_stem.str() + "_weights.json", std::ios::trunc);
    weights_file << weights.dump(2) << "\n";
    weights_file.close();
}

} // namespace

Args parse(int argc, char** argv)
{
    Args args;
    if (argc < 2) return args;
    args.command = argv[1];

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc)
                throw std::invalid_argument("Missing value for " + a);
            return argv[++i];
        };
        if      (a == "--data-dir"   || a == "-d") args.data_dir   = next();
        else if (a == "--model-dir"  || a == "-m") args.model_dir  = next();
        else if (a == "--ckpt-dir"   || a == "-k") args.ckpt_dir   = next();
        else if (a == "--log-dir"    || a == "-l") args.log_dir    = next();
        else if (a == "--symbol"     || a == "-s") args.symbol     = next();
        else if (a == "--symbols")                 args.symbols    = next();
        else if (a == "--timeframe"  || a == "-t") args.timeframe  = next();
        else if (a == "--timeframes")              args.timeframes = next();
        else if (a == "--equity"     || a == "-e") args.equity     = std::stod(next());
        else if (a == "--risk"       || a == "-r") args.risk_pct   = std::stod(next());
        else if (a == "--epochs"     || a == "-n") args.epochs     = std::stoi(next());
        else if (a == "--mode")                    args.mode       = next();
        else if (a == "--runtime-seconds")         args.runtime_seconds = std::stoi(next());
        else if (a == "--sandbox")                 args.sandbox    = true;
        else if (a == "--no-sandbox")              args.sandbox    = false;
        else if (a == "--kill-switch")             args.kill_switch = true;
        else if (a == "--product-type")            args.product_type = next();
        else if (a == "--api-key")                 args.api_key = next();
        else if (a == "--secret-key")              args.secret_key = next();
        else if (a == "--passphrase")              args.passphrase = next();
        else if (a == "--leverage")                args.leverage = std::stod(next());
        else if (a == "--max-daily-loss")          args.max_daily_loss = std::stod(next());
        else if (a == "--max-positions")           args.max_positions = std::stoi(next());
        else if (a == "--tune-params")             args.tune_params = true;
        else if (a == "--tune-range")              args.tune_range = std::stoi(next());
        else if (a == "--tune-risk-step")          args.tune_risk_step = std::stod(next());
        else if (a == "--live")                    args.live_mode  = true;
        else if (a == "--verbose"    || a == "-v") args.verbose    = true;
    }
    if (!args.symbol.empty() && args.symbols == "XAUUSD") args.symbols = args.symbol;
    if (!args.timeframe.empty() && args.timeframes == "H1") args.timeframes = args.timeframe;
    if (args.mode.empty()) {
        if (args.command == "paper" || args.command == "live" || args.live_mode) {
            args.mode = args.command == "live" ? "live" : "paper";
        } else if (args.command == "replay") {
            args.mode = "replay";
        }
    }
    args.api_key = env_or_default("BITGET_API_KEY", args.api_key);
    args.secret_key = env_or_default("BITGET_SECRET_KEY", args.secret_key);
    args.passphrase = env_or_default("BITGET_PASSPHRASE", args.passphrase);
    if (args.tune_range < 0) args.tune_range = 0;
    args.tune_risk_step = std::abs(args.tune_risk_step);
    return args;
}

void print_usage()
{
    std::cout <<
R"(
Daydream Handler v1.0 – Institutional Algo-Trading Engine
==========================================================
Usage:
  daydream_handler <command> [options]

Commands:
  replay      Run LFO jitter replay over historical data
  paper       Connect to Bitget paper feed/order path
  live        Connect to Bitget live feed/order path
  info        Print loaded models and instrument specs
  checkpoint  List available checkpoints
  help        Show this help

Options:
  -d, --data-dir    <path>    Historical data directory          [data]
  -m, --model-dir   <path>    ML model directory                 [models]
  -k, --ckpt-dir    <path>    Checkpoint directory               [checkpoints]
  -l, --log-dir     <path>    Log output directory               [logs]
  -s, --symbol      <sym>     Instrument symbol                  [XAUUSD]
      --symbols     <csv>     Symbol set (comma-separated)
  -t, --timeframe   <tf>      Timeframe: M1 M5 M15 H1 H4 D1     [H1]
      --timeframes  <csv>     Timeframe set (comma-separated)
  -e, --equity      <float>   Account equity                     [10000]
  -r, --risk        <float>   Risk fraction per trade            [0.01]
  -n, --epochs      <int>     Replay epochs                      [1]
     --tune-params            Enable replay range-circling param search
     --tune-range  <int>     Range for circling offsets (+/-N)   [2]
     --tune-risk-step <float> Risk pct step per circling offset   [0.001]
     --mode        <mode>    replay|paper|live
      --runtime-seconds <s>   Paper/live runtime duration        [600]
      --sandbox/--no-sandbox  Bitget paper toggle                [sandbox]
      --product-type <type>   USDT-FUTURES|COIN-FUTURES|SPOT
      --api-key/--secret-key/--passphrase  Bitget credentials
      --leverage    <float>   Leverage                           [1]
      --max-daily-loss <amt>  Max loss guard before kill-switch
      --max-positions <int>   Hard max open positions            [3]
      --kill-switch           Disable all real order submissions
      --live                  Enable live trading mode
  -v, --verbose               Verbose console output

Examples:
  daydream_handler replay -d data/XAUUSD -s XAUUSD -t H1
  daydream_handler paper --symbols BTCUSD,ETHUSD --timeframes H1 --runtime-seconds 900
  daydream_handler info
  daydream_handler checkpoint
)" << std::endl;
}

int run(const Args& args)
{
    // ── Logging ───────────────────────────────────────────────────────────────
    auto log_level = args.verbose
                     ? spdlog::level::debug
                     : spdlog::level::info;
    dh::logging::init(args.log_dir, "daydream", log_level,
                      spdlog::level::trace);
    auto log = dh::logging::get("cli");

    // ── Telemetry ─────────────────────────────────────────────────────────────
    telemetry::Telemetry::instance().init(args.log_dir + "/telemetry");
    telemetry::hparam("command",    args.command);
    telemetry::hparam("mode",       args.mode);
    telemetry::hparam("symbol",     args.symbol);
    telemetry::hparam("symbols",    args.symbols);
    telemetry::hparam("timeframe",  args.timeframe);
    telemetry::hparam("timeframes", args.timeframes);
    telemetry::hparam("equity",     args.equity);
    telemetry::hparam("risk_pct",   args.risk_pct);
    telemetry::hparam("live_mode",  args.live_mode);
    telemetry::hparam("sandbox",    args.sandbox);
    telemetry::hparam("runtime_seconds", args.runtime_seconds);

    if (args.command == "help") {
        print_usage();
        return 0;
    }

    if (args.command == "info") {
        log->info("Daydream Handler – Model Registry");
        auto keys = ml::ModelRegistry::instance().keys();
        for (auto& k : keys) {
            auto m = ml::ModelRegistry::instance().get(k);
            log->info("  Model: {} v{} loaded={}", k, m->version(),
                      m->is_loaded());
        }
        return 0;
    }

    if (args.command == "checkpoint") {
        storage::CheckpointManager cm(args.ckpt_dir);
        auto files = cm.list_checkpoints();
        if (files.empty()) {
            log->info("No checkpoints found in {}", args.ckpt_dir);
        } else {
            for (auto& f : files) log->info("  {}", f);
        }
        return 0;
    }

    auto mode = parse_mode(args);

    if (mode == engine::RuntimeMode::Replay) {
        auto make_replay_cfg = [&](double risk_pct) {
            engine::EngineConfig cfg;
            cfg.model_dir      = args.model_dir;
            cfg.checkpoint_dir = args.ckpt_dir;
            cfg.data_dir       = args.data_dir;
            cfg.runtime_mode   = mode;
            cfg.risk.account_equity = args.equity;
            cfg.risk.risk_pct       = risk_pct;
            cfg.risk.max_positions  = static_cast<double>(args.max_positions);
            cfg.symbols = parse_symbols(args);
            cfg.timeframes = parse_timeframes(args.timeframes);
            return cfg;
        };

        if (args.tune_params) {
            double best_pnl = -std::numeric_limits<double>::infinity();
            double best_risk = args.risk_pct;
            int best_epoch = 0;
            nlohmann::json trials = nlohmann::json::array();

            for (int epoch = 0; epoch < args.epochs; ++epoch) {
                const double tuned_risk = tuned_risk_for_epoch(args, epoch);
                auto cfg = make_replay_cfg(tuned_risk);

                log->info("=== Epoch {}/{} risk_pct={:.6f} (range-circling) ===",
                          epoch + 1, args.epochs, tuned_risk);

                engine::TradingEngine eng(cfg);
                eng.init();
                eng.run_replay();
                eng.shutdown();

                const double pnl = eng.total_pnl();
                const bool is_best = pnl > best_pnl;
                if (is_best) {
                    best_pnl = pnl;
                    best_risk = tuned_risk;
                    best_epoch = epoch + 1;
                }

                write_replay_artifacts(args, epoch, cfg, eng, is_best);

                trials.push_back({
                    {"epoch", epoch + 1},
                    {"risk_pct", tuned_risk},
                    {"pnl", pnl},
                    {"best", is_best}
                });
                telemetry::scalar("epoch/pnl", pnl, static_cast<int64_t>(epoch));
            }

            const std::string tuning_dir = args.log_dir + "/replay_tuning";
            std::filesystem::create_directories(tuning_dir);
            std::ofstream best_file(tuning_dir + "/best_params.json", std::ios::trunc);
            best_file << nlohmann::json{
                {"mode", "replay_tuning"},
                {"epochs", args.epochs},
                {"range", args.tune_range},
                {"risk_step", args.tune_risk_step},
                {"best_epoch", best_epoch},
                {"best_pnl", best_pnl},
                {"best_params", {
                    {"risk_pct", best_risk},
                    {"max_positions", args.max_positions}
                }},
                {"trials", trials}
            }.dump(2) << "\n";
            best_file.close();

            log->info("Replay tuning complete. Best epoch={} risk_pct={:.6f} P&L={:.2f}",
                      best_epoch, best_risk, best_pnl);
            return 0;
        }

        auto cfg = make_replay_cfg(args.risk_pct);
        engine::TradingEngine eng(cfg);
        eng.init();

        for (int epoch = 0; epoch < args.epochs; ++epoch) {
            log->info("=== Epoch {}/{} ===", epoch + 1, args.epochs);
            eng.run_replay();
            write_replay_artifacts(args, epoch, cfg, eng, false);
            telemetry::scalar("epoch/pnl", eng.total_pnl(),
                              static_cast<int64_t>(epoch));
        }

        eng.shutdown();
        log->info("Final P&L: {:.2f}", eng.total_pnl());
        return 0;
    }

    if (mode == engine::RuntimeMode::Paper || mode == engine::RuntimeMode::Live) {
        engine::EngineConfig cfg;
        cfg.model_dir      = args.model_dir;
        cfg.checkpoint_dir = args.ckpt_dir;
        cfg.data_dir       = args.data_dir;
        cfg.runtime_mode   = mode;
        cfg.risk.account_equity = args.equity;
        cfg.risk.risk_pct       = args.risk_pct;
        cfg.risk.max_positions  = static_cast<double>(args.max_positions);
        cfg.max_daily_loss_abs  = args.max_daily_loss;
        cfg.kill_switch         = args.kill_switch;
        cfg.symbols             = parse_symbols(args);
        cfg.timeframes          = parse_timeframes(args.timeframes);
        cfg.exchange.sandbox    = args.sandbox;
        cfg.exchange.product_type = args.product_type;
        cfg.exchange.api_key    = args.api_key;
        cfg.exchange.secret_key = args.secret_key;
        cfg.exchange.passphrase = args.passphrase;
        cfg.exchange.leverage   = args.leverage;

        if (mode == engine::RuntimeMode::Live) {
            if (cfg.exchange.sandbox) {
                log->warn("Live mode with sandbox=true detected; forcing sandbox=false");
                cfg.exchange.sandbox = false;
            }
        }

        engine::TradingEngine eng(cfg);
        eng.init();
        log->info("{} mode running for {}s", mode == engine::RuntimeMode::Paper ? "Paper" : "Live",
                  args.runtime_seconds);
        std::this_thread::sleep_for(std::chrono::seconds(args.runtime_seconds));
        eng.shutdown();
        return 0;
    }

    log->error("Unknown command: {}. Run 'help' for usage.", args.command);
    return 2;
}

} // namespace dh::cli
