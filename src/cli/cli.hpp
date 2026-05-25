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
    std::string data_dir    = "data";
    std::string model_dir   = "models";
    std::string ckpt_dir    = "checkpoints";
    std::string log_dir     = "logs";
    std::string symbol      = "XAUUSD";
    std::string timeframe   = "H1";
    bool        live_mode   = false;
    bool        verbose     = false;
    double      risk_pct    = 0.01;
    double      equity      = 10000.0;
    int         epochs      = 1;
};

/// Parse command-line arguments.
Args parse(int argc, char** argv);

/// Print usage to stdout.
void print_usage();

/// Execute the parsed command.
int run(const Args& args);

} // namespace dh::cli
