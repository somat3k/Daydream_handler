#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Daydream Handler – CLI
// Console command-line interface for controlling the trading engine.
// ─────────────────────────────────────────────────────────────────────────────
#include "../core/engine/trading_engine.hpp"
#include "../logging/logger.hpp"
#include <string>
#include <vector>
#include <iostream>
#include <sstream>

namespace dh::cli {

struct Args {
    std::string command     = "help";
    std::string mode;
    std::string data_dir    = "data";
    std::string model_dir   = "models";
    std::string ckpt_dir    = "checkpoints";
    std::string log_dir     = "logs";
    std::string symbol      = "XAUUSD";
    std::string symbols     = "XAUUSD";
    std::string timeframe   = "H1";
    std::string timeframes  = "H1";
    bool        live_mode   = false;
    bool        verbose     = false;
    bool        sandbox     = true;
    bool        kill_switch = false;
    std::string product_type = "USDT-FUTURES";
    std::string api_key;
    std::string secret_key;
    std::string passphrase;
    double      risk_pct    = 0.01;
    double      equity      = 10000.0;
    double      leverage    = 1.0;
    double      max_daily_loss = 0.0;
    int         max_positions = 3;
    int         epochs      = 1;
    int         runtime_seconds = 600;
    bool        tune_params = false;
    int         tune_range = 2;
    double      tune_risk_step = 0.001;
};

/// Parse command-line arguments.
Args parse(int argc, char** argv);

/// Print usage to stdout.
void print_usage();

/// Execute the parsed command.
int run(const Args& args);

} // namespace dh::cli
