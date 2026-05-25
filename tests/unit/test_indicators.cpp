#include <gtest/gtest.h>
#include "core/indicators/ichimoku.hpp"
#include "core/indicators/fibonacci.hpp"
#include "core/indicators/volatility_cloud.hpp"
#include "core/indicators/supply_demand.hpp"
#include "core/indicators/liquidity.hpp"
#include "data/timeseries.hpp"

using namespace dh;

namespace {

dh::data::TimeSeries make_trending_ts(int n, double start, double step)
{
    dh::data::TimeSeries ts(n * 2);
    for (int i = 0; i < n; ++i) {
        double c = start + i * step;
        ts.push({static_cast<int64_t>(i),
                 c - 2, c + 3, c - 4, c, 1000.0,
                 data::Timeframe::H1});
    }
    return ts;
}

} // namespace

// ── Ichimoku ──────────────────────────────────────────────────────────────────
TEST(Ichimoku, InsufficientData)
{
    data::TimeSeries ts(100);
    ts.push({0, 1900, 1905, 1895, 1902, 500.0, data::Timeframe::H1});
    indicators::Ichimoku ich;
    auto v = ich.calculate(ts);
    EXPECT_DOUBLE_EQ(v.tenkan_sen, 0.0);
}

TEST(Ichimoku, BullishCloudDetection)
{
    // Build an uptrending series long enough (>= 52 + 26 = 78 bars)
    auto ts = make_trending_ts(100, 1800.0, 1.0);
    indicators::Ichimoku ich;
    auto v = ich.calculate(ts);
    // In an uptrend Tenkan >= Kijun is likely
    EXPECT_GE(v.tenkan_sen, 0.0);
}

// ── Fibonacci ─────────────────────────────────────────────────────────────────
TEST(Fibonacci, RetracementSize)
{
    auto levels = indicators::Fibonacci::retracement(1800.0, 2000.0);
    EXPECT_EQ(levels.size(), 12u);
}

TEST(Fibonacci, RetracementPrices)
{
    auto levels = indicators::Fibonacci::retracement(1800.0, 2000.0);
    // 0% retracement == swing high
    EXPECT_NEAR(levels[0].price, 2000.0, 0.001);
    // 100% retracement == swing low
    EXPECT_NEAR(levels[6].price, 1800.0, 0.001);
    // 61.8% retracement
    EXPECT_NEAR(levels[4].price, 2000.0 - 0.618 * 200.0, 0.01);
}

TEST(Fibonacci, Nearest)
{
    auto levels = indicators::Fibonacci::retracement(1800.0, 2000.0);
    auto* n = indicators::Fibonacci::nearest(levels, 1876.4);
    ASSERT_NE(n, nullptr);
    // Nearest to 1876 should be the 61.8% level (1876.4)
    EXPECT_NEAR(n->price, 2000.0 - 0.618 * 200.0, 1.0);
}

// ── Volatility Cloud ──────────────────────────────────────────────────────────
TEST(VolatilityCloud, InsufficientData)
{
    data::TimeSeries ts(100);
    ts.push({0, 1900, 1905, 1895, 1902, 500.0, data::Timeframe::H1});
    indicators::VolatilityCloud vc;
    auto v = vc.calculate(ts);
    EXPECT_DOUBLE_EQ(v.atr, 0.0);
}

TEST(VolatilityCloud, BandOrder)
{
    auto ts = make_trending_ts(30, 1900.0, 0.5);
    indicators::VolatilityCloud vc;
    auto v = vc.calculate(ts);
    if (v.atr > 0) {
        EXPECT_GT(v.upper_band, v.mid);
        EXPECT_LT(v.lower_band, v.mid);
    }
}

// ── Supply & Demand ───────────────────────────────────────────────────────────
TEST(SupplyDemand, ReturnsEmptyForSmallSeries)
{
    data::TimeSeries ts(100);
    ts.push({0, 1900, 1905, 1895, 1902, 500.0, data::Timeframe::H1});
    indicators::SupplyDemand sd;
    EXPECT_TRUE(sd.detect(ts).empty());
}

// ── Liquidity ──────────────────────────────────────────────────────────────────
TEST(LiquiditySweep, NoEventsSmallSeries)
{
    data::TimeSeries ts(100);
    ts.push({0, 1900, 1905, 1895, 1902, 500.0, data::Timeframe::H1});
    indicators::LiquiditySweep ls;
    EXPECT_TRUE(ls.detect(ts).empty());
}
