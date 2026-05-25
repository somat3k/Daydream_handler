#include <gtest/gtest.h>
#include "ml/inference/inference_engine.hpp"

using namespace dh::ml;

TEST(StubModel, LoadAlwaysSucceeds)
{
    StubModel m("test_model");
    EXPECT_TRUE(m.load("any/path"));
    EXPECT_TRUE(m.is_loaded());
}

TEST(StubModel, InferEmptyInputReturnsHold)
{
    StubModel m("test_model");
    m.load("");
    ModelInput in;
    auto out = m.infer(in);
    EXPECT_EQ(out.label, "HOLD");
    EXPECT_EQ(out.predicted_class, 0);
}

TEST(StubModel, InferPositiveFeaturesReturnsBuy)
{
    StubModel m("test_model", "stub", 3);
    m.load("");
    ModelInput in;
    in.features = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
    auto out = m.infer(in);
    EXPECT_EQ(out.predicted_class, 1);  // BUY
    EXPECT_EQ(out.label, "BUY");
}

TEST(StubModel, InferNegativeFeaturesReturnsSell)
{
    StubModel m("test_model", "stub", 3);
    m.load("");
    ModelInput in;
    in.features = {-0.5f, -0.4f, -0.3f, -0.2f, -0.1f};
    auto out = m.infer(in);
    EXPECT_EQ(out.predicted_class, 2);  // SELL
    EXPECT_EQ(out.label, "SELL");
}

TEST(StubModel, ProbabilitiesSumToOne)
{
    StubModel m("test_model", "stub", 3);
    m.load("");
    ModelInput in;
    in.features = {0.1f, 0.2f, 0.3f};
    auto out = m.infer(in);
    float s = 0;
    for (auto p : out.probabilities) s += p;
    EXPECT_NEAR(s, 1.0f, 1e-5f);
}

TEST(ModelRegistry, RegisterAndGet)
{
    auto& reg = ModelRegistry::instance();
    auto stub = std::make_shared<StubModel>("reg_test");
    reg.register_model("reg_test", stub);
    auto got = reg.get("reg_test");
    EXPECT_EQ(got->name(), "reg_test");
}

TEST(ModelRegistry, MissingKeyThrows)
{
    auto& reg = ModelRegistry::instance();
    EXPECT_THROW(reg.get("__nonexistent__"), std::out_of_range);
}

TEST(ModelRegistry, FactoryLayer0)
{
    auto m = models::make_layer0();
    EXPECT_EQ(m->name(), "layer0_merged");
}

TEST(ModelRegistry, FactoryGrandUnified)
{
    auto m = models::make_grand_unified();
    EXPECT_EQ(m->name(), "grand_unified");
}
