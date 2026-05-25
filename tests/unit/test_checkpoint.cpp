#include <gtest/gtest.h>
#include "storage/checkpoint_manager.hpp"
#include <filesystem>

using namespace dh::storage;

namespace fs = std::filesystem;

class CheckpointTest : public ::testing::Test {
protected:
    void SetUp() override {
        m_dir = "/tmp/dh_test_ckpt_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count());
    }
    void TearDown() override {
        if (fs::exists(m_dir)) fs::remove_all(m_dir);
    }
    std::string m_dir;
};

TEST_F(CheckpointTest, SaveAndLoad)
{
    CheckpointManager cm(m_dir, 5);
    json state = {{"step", 42}, {"pnl", 123.45}};
    auto path = cm.save(state, 42);
    EXPECT_TRUE(fs::exists(path));

    auto loaded = cm.load(42);
    EXPECT_EQ(loaded["state"]["step"].get<int>(), 42);
    EXPECT_NEAR(loaded["state"]["pnl"].get<double>(), 123.45, 0.001);
}

TEST_F(CheckpointTest, LoadLatest)
{
    CheckpointManager cm(m_dir, 5);
    cm.save({{"step", 1}}, 1);
    cm.save({{"step", 2}}, 2);
    cm.save({{"step", 3}}, 3);

    auto latest = cm.load_latest();
    EXPECT_EQ(latest["step"].get<int64_t>(), 3);
}

TEST_F(CheckpointTest, Rotation)
{
    CheckpointManager cm(m_dir, 3);
    for (int i = 1; i <= 6; ++i)
        cm.save({{"step", i}}, i);

    auto files = cm.list_checkpoints();
    EXPECT_EQ(files.size(), 3u);
}

TEST_F(CheckpointTest, EmptyDirectoryThrows)
{
    CheckpointManager cm(m_dir, 5);
    EXPECT_THROW(cm.load_latest(), std::runtime_error);
}
