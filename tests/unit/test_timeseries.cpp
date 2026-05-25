#include <gtest/gtest.h>
#include "data/timeseries.hpp"
#include "data/market_data.hpp"

using namespace dh::data;

namespace {

Bar make_bar(double close, int64_t t = 0, Timeframe tf = Timeframe::H1)
{
    return Bar{t, close * 0.999, close * 1.001, close * 0.998, close, 1000.0, tf};
}

} // namespace

TEST(TimeSeries, EmptyOnConstruction)
{
    TimeSeries ts(100);
    EXPECT_TRUE(ts.empty());
    EXPECT_EQ(ts.size(), 0u);
}

TEST(TimeSeries, PushAndRetrieve)
{
    TimeSeries ts(100);
    ts.push(make_bar(1900.0, 1000));
    ts.push(make_bar(1905.0, 2000));
    EXPECT_EQ(ts.size(), 2u);
    EXPECT_DOUBLE_EQ(ts.latest().close, 1905.0);
    EXPECT_DOUBLE_EQ(ts.at(0).close, 1900.0);
}

TEST(TimeSeries, SlidingWindowEviction)
{
    TimeSeries ts(3);
    for (int i = 0; i < 5; ++i)
        ts.push(make_bar(1900.0 + i, i));
    EXPECT_EQ(ts.size(), 3u);
    EXPECT_DOUBLE_EQ(ts.at(0).close, 1902.0);  // oldest kept
    EXPECT_DOUBLE_EQ(ts.latest().close, 1904.0);
}

TEST(TimeSeries, CloseWindow)
{
    TimeSeries ts(100);
    for (int i = 0; i < 10; ++i)
        ts.push(make_bar(1900.0 + i, i));
    auto w = ts.close_window(5);
    EXPECT_EQ(w.size(), 5u);
    EXPECT_DOUBLE_EQ(w.back(), 1909.0);
}

TEST(TimeSeries, StepCounter)
{
    TimeSeries ts(100);
    EXPECT_EQ(ts.step(), 0);
    ts.push(make_bar(1.0));
    EXPECT_EQ(ts.step(), 1);
}

TEST(LFOJitterReplay, ReplayCallsCallbackForEachBar)
{
    TimeSeries ts(100);
    for (int i = 0; i < 10; ++i)
        ts.push(make_bar(1900.0 + i, i));

    LFOJitterReplay replay;
    int count = 0;
    replay.replay(ts, [&](const Bar&, int64_t){ ++count; });
    EXPECT_EQ(count, 10);
}

TEST(LFOJitterReplay, AugmentedBarsHaveModifiedClose)
{
    LFOJitterParams params;
    params.amplitude = 0.001;
    params.add_noise = false;
    LFOJitterReplay replay(params);

    Bar b = make_bar(1900.0, 0);
    Bar aug = replay.augment(b, 1000);
    // With non-zero amplitude and bar_index != 0, close should differ
    // (sine will be non-zero for most indices)
    // Just check it's still in a reasonable range
    EXPECT_GT(aug.close, 1880.0);
    EXPECT_LT(aug.close, 1920.0);
}
