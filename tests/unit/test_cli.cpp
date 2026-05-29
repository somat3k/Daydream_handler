#include <gtest/gtest.h>
#include "cli/cli.hpp"

TEST(Cli, ParsesPaperModeAndSafetyFlags)
{
    char arg0[] = "daydream_handler";
    char arg1[] = "paper";
    char arg2[] = "--symbols";
    char arg3[] = "BTCUSD,ETHUSD";
    char arg4[] = "--timeframes";
    char arg5[] = "H1,M15";
    char arg6[] = "--runtime-seconds";
    char arg7[] = "120";
    char arg8[] = "--kill-switch";
    char* argv[] = {arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8};

    auto args = dh::cli::parse(9, argv);
    EXPECT_EQ(args.command, "paper");
    EXPECT_EQ(args.mode, "paper");
    EXPECT_EQ(args.symbols, "BTCUSD,ETHUSD");
    EXPECT_EQ(args.timeframes, "H1,M15");
    EXPECT_EQ(args.runtime_seconds, 120);
    EXPECT_TRUE(args.kill_switch);
}

TEST(Cli, ParsesModeOverride)
{
    char arg0[] = "daydream_handler";
    char arg1[] = "replay";
    char arg2[] = "--mode";
    char arg3[] = "live";
    char* argv[] = {arg0, arg1, arg2, arg3};

    auto args = dh::cli::parse(4, argv);
    EXPECT_EQ(args.mode, "live");
}

TEST(Cli, ParsesReplayTuningOptions)
{
    char arg0[] = "daydream_handler";
    char arg1[] = "replay";
    char arg2[] = "--epochs";
    char arg3[] = "7";
    char arg4[] = "--tune-params";
    char arg5[] = "--tune-range";
    char arg6[] = "3";
    char arg7[] = "--tune-risk-step";
    char arg8[] = "0.0025";
    char* argv[] = {arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8};

    auto args = dh::cli::parse(9, argv);
    EXPECT_EQ(args.command, "replay");
    EXPECT_EQ(args.epochs, 7);
    EXPECT_TRUE(args.tune_params);
    EXPECT_EQ(args.tune_range, 3);
    EXPECT_DOUBLE_EQ(args.tune_risk_step, 0.0025);
}
