#include <gtest/gtest.h>
#include "ml/inference/feature_composer.hpp"
#include "data/timeseries.hpp"

using namespace dh;

namespace {

data::TimeSeries make_ts(int n)
{
    data::TimeSeries ts(n * 2);
    for (int i = 0; i < n; ++i) {
        double c = 1900.0 + i * 0.5;
        ts.push({static_cast<int64_t>(i),
                 c - 1, c + 2, c - 2, c, 500.0,
                 data::Timeframe::H1});
    }
    return ts;
}

} // namespace

TEST(FeatureComposer, ReLURemovesNegatives)
{
    std::vector<float> v = {-1.0f, 0.0f, 2.0f, -3.0f, 5.0f};
    auto out = ml::relu(v);
    EXPECT_EQ(out.size(), v.size());
    for (auto x : out) EXPECT_GE(x, 0.0f);
}

TEST(FeatureComposer, SoftmaxSumsToOne)
{
    std::vector<float> v = {1.0f, 2.0f, 3.0f};
    auto p = ml::softmax(v);
    float s = 0;
    for (auto x : p) s += x;
    EXPECT_NEAR(s, 1.0f, 1e-5f);
}

TEST(FeatureComposer, ConcatSize)
{
    std::vector<float> a = {1, 2, 3};
    std::vector<float> b = {4, 5};
    auto c = ml::concat(a, b);
    EXPECT_EQ(c.size(), 5u);
    EXPECT_FLOAT_EQ(c[0], 1.0f);
    EXPECT_FLOAT_EQ(c[4], 5.0f);
}

TEST(FeatureComposer, SplitAt)
{
    std::vector<float> v = {1, 2, 3, 4, 5};
    auto [left, right] = ml::split_at(v, 3);
    EXPECT_EQ(left.size(), 3u);
    EXPECT_EQ(right.size(), 2u);
    EXPECT_FLOAT_EQ(left.back(), 3.0f);
    EXPECT_FLOAT_EQ(right.front(), 4.0f);
}

TEST(FeatureComposer, GEMMDimensionMismatch)
{
    ml::Matrix A; A.rows = 2; A.cols = 3; A.data.assign(6, 1.0f);
    ml::Matrix B; B.rows = 2; B.cols = 2; B.data.assign(4, 1.0f);
    EXPECT_THROW(ml::gemm(A, B), std::invalid_argument);
}

TEST(FeatureComposer, GEMMCorrectResult)
{
    // [1 2] * [5; 6] = [1*5 + 2*6] = [17]
    ml::Matrix A; A.rows = 1; A.cols = 2; A.data = {1.0f, 2.0f};
    ml::Matrix B; B.rows = 2; B.cols = 1; B.data = {5.0f, 6.0f};
    auto C = ml::gemm(A, B);
    EXPECT_EQ(C.rows, 1u);
    EXPECT_EQ(C.cols, 1u);
    EXPECT_NEAR(C.data[0], 17.0f, 1e-5f);
}

TEST(FeatureComposer, ComposeProducesNonEmptyFeatures)
{
    auto ts = make_ts(60);
    indicators::Ichimoku ich;
    indicators::VolatilityCloud vol;
    auto ich_v = ich.calculate(ts);
    auto vol_v = vol.calculate(ts);

    ml::FeatureConfig cfg;
    cfg.window      = 20;
    cfg.project_dim = 32;
    ml::FeatureComposer composer(cfg);

    auto feat = composer.compose(ts, ich_v, vol_v, 1880.0, 1940.0);
    EXPECT_EQ(feat.size(), 32u);
    // All outputs should be non-negative (ReLU applied after projection)
    for (auto f : feat) EXPECT_GE(f, 0.0f);
}

TEST(FeatureComposer, NormaliseZScore)
{
    std::vector<double> prices = {10.0, 12.0, 14.0, 12.0, 8.0};
    auto n = ml::FeatureComposer::normalise_zscore(prices);
    EXPECT_EQ(n.size(), prices.size());
    // mean should be ~0
    float s = 0;
    for (auto v : n) s += v;
    EXPECT_NEAR(s / n.size(), 0.0f, 0.01f);
}
