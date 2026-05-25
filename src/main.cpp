// ─────────────────────────────────────────────────────────────────────────────
// Daydream Handler – main.cpp
// Entry point: parse CLI args, run trading engine.
// ─────────────────────────────────────────────────────────────────────────────
#include "cli/cli.hpp"

int main(int argc, char** argv)
{
    try {
        auto args = dh::cli::parse(argc, argv);
        return dh::cli::run(args);
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] " << e.what() << "\n";
        return 1;
    }
}
