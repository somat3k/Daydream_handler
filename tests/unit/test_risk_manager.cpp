#include <gtest/gtest.h>
#include "core/risk/risk_manager.hpp"
#include "data/market_data.hpp"

using namespace dh;

TEST(RiskManager, PositionSizingLong)
{
    risk::RiskConfig cfg;
    cfg.account_equity = 10000.0;
    cfg.risk_pct       = 0.01;   // 1% = $100 at risk
    risk::RiskManager rm(cfg);

    auto instr = data::instruments::XAUUSD();
    auto spec  = rm.size_position(instr, 1950.0, 1940.0, 1975.0, true);

    EXPECT_EQ(spec.is_long, true);
    EXPECT_GT(spec.lot_size, 0.0);
    EXPECT_NEAR(spec.risk_amount, 100.0, 0.01);
    EXPECT_GT(spec.rr_ratio, 0.0);
    EXPECT_LT(spec.stop_loss, spec.entry_price);
    EXPECT_GT(spec.take_profit, spec.entry_price);
}

TEST(RiskManager, PositionSizingShort)
{
    risk::RiskConfig cfg;
    cfg.account_equity = 20000.0;
    cfg.risk_pct       = 0.02;
    risk::RiskManager rm(cfg);

    auto instr = data::instruments::EURUSD();
    auto spec  = rm.size_position(instr, 1.08500, 1.09000, 1.07500, false);

    EXPECT_EQ(spec.is_long, false);
    EXPECT_GT(spec.lot_size, 0.0);
    EXPECT_GT(spec.stop_loss, spec.entry_price);
    EXPECT_LT(spec.take_profit, spec.entry_price);
}

TEST(RiskManager, TPClampedToMaxPips)
{
    risk::RiskConfig cfg;
    cfg.max_tp_pips = 20.0;
    risk::RiskManager rm(cfg);

    auto instr = data::instruments::GBPUSD();
    // Attempt to set a TP 200 pips away – should be clamped to 20 pips
    double entry = 1.27000;
    double tp_far = entry + 0.0200;  // 200 pips
    auto spec = rm.size_position(instr, entry, entry - 0.0010, tp_far, true);

    double actual_tp_pips = (spec.take_profit - entry) / instr.pip_size;
    EXPECT_LE(actual_tp_pips, cfg.max_tp_pips + 0.001);
}

TEST(RiskManager, SpreadCheck)
{
    risk::RiskConfig cfg;
    cfg.max_spread_pips = 3.0;
    risk::RiskManager rm(cfg);
    auto instr = data::instruments::EURUSD();
    EXPECT_TRUE (rm.spread_ok(instr, 1.0));
    EXPECT_FALSE(rm.spread_ok(instr, 5.0));
}

TEST(RiskManager, DrawdownCheck)
{
    risk::RiskConfig cfg;
    cfg.account_equity = 10000.0;
    cfg.max_dd_pct     = 0.10;
    risk::RiskManager rm(cfg);
    EXPECT_TRUE (rm.drawdown_ok(9500.0));  //  5% DD – ok
    EXPECT_FALSE(rm.drawdown_ok(8900.0));  // 11% DD – reject
}
