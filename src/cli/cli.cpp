#include "cli.hpp"
#include "../storage/checkpoint_manager.hpp"
#include <stdexcept>
#include <cstring>

namespace dh::cli {

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
        else if (a == "--timeframe"  || a == "-t") args.timeframe  = next();
        else if (a == "--equity"     || a == "-e") args.equity     = std::stod(next());
        else if (a == "--risk"       || a == "-r") args.risk_pct   = std::stod(next());
        else if (a == "--epochs"     || a == "-n") args.epochs     = std::stoi(next());
        else if (a == "--live")                    args.live_mode  = true;
        else if (a == "--verbose"    || a == "-v") args.verbose    = true;
    }
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
  live        Connect to live broker feed (requires broker plugin)
  info        Print loaded models and instrument specs
  checkpoint  List available checkpoints
  help        Show this help

Options:
  -d, --data-dir    <path>    Historical data directory          [data]
  -m, --model-dir   <path>    ML model directory                 [models]
  -k, --ckpt-dir    <path>    Checkpoint directory               [checkpoints]
  -l, --log-dir     <path>    Log output directory               [logs]
  -s, --symbol      <sym>     Instrument symbol                  [XAUUSD]
  -t, --timeframe   <tf>      Timeframe: M1 M5 M15 H1 H4 D1     [H1]
  -e, --equity      <float>   Account equity                     [10000]
  -r, --risk        <float>   Risk fraction per trade            [0.01]
  -n, --epochs      <int>     Replay epochs                      [1]
      --live                  Enable live trading mode
  -v, --verbose               Verbose console output

Examples:
  daydream_handler replay -d data/XAUUSD -s XAUUSD -t H1
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
    telemetry::hparam("symbol",     args.symbol);
    telemetry::hparam("timeframe",  args.timeframe);
    telemetry::hparam("equity",     args.equity);
    telemetry::hparam("risk_pct",   args.risk_pct);
    telemetry::hparam("live_mode",  args.live_mode);

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

    if (args.command == "replay") {
        engine::EngineConfig cfg;
        cfg.model_dir      = args.model_dir;
        cfg.checkpoint_dir = args.ckpt_dir;
        cfg.data_dir       = args.data_dir;
        cfg.live_mode      = false;
        cfg.risk.account_equity = args.equity;
        cfg.risk.risk_pct       = args.risk_pct;

        engine::TradingEngine eng(cfg);
        eng.init();

        for (int epoch = 0; epoch < args.epochs; ++epoch) {
            log->info("=== Epoch {}/{} ===", epoch + 1, args.epochs);
            eng.run_replay();
            telemetry::scalar("epoch/pnl", eng.total_pnl(),
                              static_cast<int64_t>(epoch));
        }

        eng.shutdown();
        log->info("Final P&L: {:.2f}", eng.total_pnl());
        return 0;
    }

    if (args.command == "live") {
        log->warn("Live mode requires broker plugin – not implemented in base build");
        return 1;
    }

    log->error("Unknown command: {}. Run 'help' for usage.", args.command);
    return 2;
}

} // namespace dh::cli
