// ─────────────────────────────────────────────────────────────────────────────
// Daydream Handler – Model Loader & Backend Tests
// Covers: SafeTensorsFile parser, SafeTensorsModel, ModelLoader discovery,
//         training log CSV import, telemetry JSONL import.
// ─────────────────────────────────────────────────────────────────────────────
#include <gtest/gtest.h>
#include "ml/loader/safetensors_reader.hpp"
#include "ml/loader/model_loader.hpp"
#include "ml/inference/inference_engine.hpp"
#include "logging/logger.hpp"
#include "logging/telemetry.hpp"
#include <filesystem>
#include <fstream>
#include <cstdint>
#include <cstring>

namespace fs = std::filesystem;
using namespace dh::ml;
using namespace dh::ml::loader;
using namespace dh;

// ── Helpers ───────────────────────────────────────────────────────────────────

static void init_logging() {
    static bool done = false;
    if (!done) {
        dh::logging::init("/tmp/test_loader_logs", "test", spdlog::level::warn);
        done = true;
    }
}

// Write a minimal safetensors file with two F32 tensors:
//   "0.weight" shape=[2,4] and "0.bias" shape=[2]
static std::string write_test_safetensors(const std::string& dir)
{
    std::string path = dir + "/test_model.safetensors";

    // Build weight data: W[2×4] = identity-like, b[2] = zeros
    std::vector<float> W = {1.f, 0.f, 0.f, 0.f,
                             0.f, 1.f, 0.f, 0.f};
    std::vector<float> b = {0.f, 0.f};

    size_t W_bytes = W.size() * 4;
    size_t b_bytes = b.size() * 4;

    // Build JSON header
    nlohmann::json header = {
        {"__metadata__", {{"format", "pt"}}},
        {"0.weight", {
            {"dtype", "F32"},
            {"shape", {2, 4}},
            {"data_offsets", {0, W_bytes}}
        }},
        {"0.bias", {
            {"dtype", "F32"},
            {"shape", {2}},
            {"data_offsets", {W_bytes, W_bytes + b_bytes}}
        }}
    };
    std::string hdr_str = header.dump();
    uint64_t hdr_len = hdr_str.size();

    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<char*>(&hdr_len), 8);
    f.write(hdr_str.c_str(), static_cast<std::streamsize>(hdr_len));
    f.write(reinterpret_cast<char*>(W.data()), static_cast<std::streamsize>(W_bytes));
    f.write(reinterpret_cast<char*>(b.data()), static_cast<std::streamsize>(b_bytes));

    return path;
}

// Write a minimal training log CSV
static std::string write_test_csv(const std::string& dir)
{
    std::string path = dir + "/training_log.csv";
    std::ofstream f(path);
    f << "epoch,loss,accuracy\n"
      << "1,0.9,0.55\n"
      << "2,0.7,0.65\n"
      << "3,0.5,0.75\n";
    return path;
}

// Write a minimal telemetry JSONL
static std::string write_test_jsonl(const std::string& dir)
{
    std::string path = dir + "/telemetry.jsonl";
    std::ofstream f(path);
    f << R"({"tag":"loss","value":0.9,"step":1})" "\n"
      << R"({"tag":"acc","value":0.55,"step":1})" "\n"
      << R"({"scalars":{"loss":0.7,"acc":0.65},"step":2})" "\n";
    return path;
}

// Write a manifest JSON
static std::string write_test_manifest(const std::string& dir,
                                        const std::string& weights_rel)
{
    std::string path = dir + "/test_manifest.json";
    nlohmann::json mf = {
        {"model_key",    "test_manifest_model"},
        {"version",      "1.0.0"},
        {"description",  "unit test manifest"},
        {"weights_file", weights_rel},
        {"layers", nlohmann::json::array({
            {{"type","linear"},{"in_features",4},{"out_features",2},
             {"weight_key","0.weight"},{"bias_key","0.bias"}}
        })}
    };
    std::ofstream f(path);
    f << mf.dump(2);
    return path;
}

// ── SafeTensorsFile tests ─────────────────────────────────────────────────────

class SafeTensorsReaderTest : public ::testing::Test {
protected:
    void SetUp() override {
        init_logging();
        m_dir = fs::temp_directory_path() / "dh_st_test";
        fs::create_directories(m_dir);
        m_path = write_test_safetensors(m_dir.string());
    }
    fs::path m_dir;
    std::string m_path;
};

TEST_F(SafeTensorsReaderTest, ParseHeaderSucceeds)
{
    auto sf = SafeTensorsFile::parse_header(m_path);
    EXPECT_TRUE(sf.parsed);
    EXPECT_EQ(sf.tensors.size(), 2u);
    EXPECT_TRUE(sf.tensors.count("0.weight"));
    EXPECT_TRUE(sf.tensors.count("0.bias"));
}

TEST_F(SafeTensorsReaderTest, TensorShapes)
{
    auto sf = SafeTensorsFile::parse_header(m_path);
    ASSERT_TRUE(sf.parsed);
    auto& w = sf.tensors.at("0.weight");
    EXPECT_EQ(w.shape[0], 2);
    EXPECT_EQ(w.shape[1], 4);
    EXPECT_EQ(w.numel(), 8u);
    EXPECT_TRUE(w.is_2d());
    auto& bi = sf.tensors.at("0.bias");
    EXPECT_EQ(bi.shape[0], 2);
    EXPECT_EQ(bi.numel(), 2u);
    EXPECT_TRUE(bi.is_1d());
}

TEST_F(SafeTensorsReaderTest, ReadWeightTensor)
{
    auto sf = SafeTensorsFile::parse_header(m_path);
    ASSERT_TRUE(sf.parsed);
    auto W = sf.read_tensor_f32("0.weight");
    ASSERT_EQ(W.size(), 8u);
    EXPECT_FLOAT_EQ(W[0], 1.f);
    EXPECT_FLOAT_EQ(W[1], 0.f);
    EXPECT_FLOAT_EQ(W[4], 0.f);
    EXPECT_FLOAT_EQ(W[5], 1.f);
}

TEST_F(SafeTensorsReaderTest, ReadBiasTensor)
{
    auto sf = SafeTensorsFile::parse_header(m_path);
    ASSERT_TRUE(sf.parsed);
    auto b = sf.read_tensor_f32("0.bias");
    ASSERT_EQ(b.size(), 2u);
    EXPECT_FLOAT_EQ(b[0], 0.f);
    EXPECT_FLOAT_EQ(b[1], 0.f);
}

TEST_F(SafeTensorsReaderTest, MissingFileParseFails)
{
    auto sf = SafeTensorsFile::parse_header("/nonexistent/path.safetensors");
    EXPECT_FALSE(sf.parsed);
    EXPECT_TRUE(sf.tensors.empty());
}

TEST_F(SafeTensorsReaderTest, SummaryNotEmpty)
{
    auto sf = SafeTensorsFile::parse_header(m_path);
    EXPECT_FALSE(sf.summary().empty());
    EXPECT_NE(sf.summary().find("2 tensors"), std::string::npos);
}

// ── SafeTensorsModel tests ────────────────────────────────────────────────────

class SafeTensorsModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        init_logging();
        m_dir = fs::temp_directory_path() / "dh_stmodel_test";
        fs::create_directories(m_dir);
        m_path = write_test_safetensors(m_dir.string());
    }
    fs::path m_dir;
    std::string m_path;
};

TEST_F(SafeTensorsModelTest, LoadSucceeds)
{
    SafeTensorsModel m("test_model", "1.0.0-st", 2);
    EXPECT_TRUE(m.load(m_path));
    EXPECT_TRUE(m.is_loaded());
}

TEST_F(SafeTensorsModelTest, LoadMissingFileFallsBackToLoaded)
{
    // Missing file → stub fallback, but still marks as loaded for registry
    SafeTensorsModel m("test_model", "1.0.0-st", 2);
    bool ok = m.load("/tmp/nonexistent_model.safetensors");
    EXPECT_FALSE(ok);
    EXPECT_FALSE(m.is_loaded());
}

TEST_F(SafeTensorsModelTest, InferProducesValidOutput)
{
    SafeTensorsModel m("test_model", "1.0.0-st", 2);
    ASSERT_TRUE(m.load(m_path));

    ModelInput in;
    in.features = {1.f, 0.f, 0.f, 0.f};
    auto out = m.infer(in);

    // Output should have 2 classes, probabilities sum to ~1
    ASSERT_EQ(out.probabilities.size(), 2u);
    float s = 0;
    for (auto p : out.probabilities) s += p;
    EXPECT_NEAR(s, 1.f, 1e-4f);
    EXPECT_GE(out.predicted_class, 0);
    EXPECT_LT(out.predicted_class, 2);
    EXPECT_GT(out.confidence, 0.f);
    EXPECT_LE(out.confidence, 1.f);
}

TEST_F(SafeTensorsModelTest, HparamsContainsBackend)
{
    SafeTensorsModel m("test_model", "1.0.0-st", 3);
    m.load(m_path);
    auto hp = m.hparams();
    EXPECT_EQ(hp.at("backend").get<std::string>(), "safetensors");
    EXPECT_EQ(hp.at("model").get<std::string>(), "test_model");
}

TEST_F(SafeTensorsModelTest, InjectedLayersUsedForInference)
{
    SafeTensorsModel m("test_inj", "1.0.0-st", 2);

    // Inject a known identity-like layer
    SafeTensorsModel::LinearLayer ll;
    ll.in_features  = 4;
    ll.out_features = 2;
    ll.W = {1.f, 0.f, 0.f, 0.f,
            0.f, 1.f, 0.f, 0.f};
    ll.b = {0.f, 0.f};
    ll.relu_after = false;
    m.set_layers({ll});
    m.load(m_path); // sets m_loaded = true

    ModelInput in;
    in.features = {2.f, -1.f, 0.f, 0.f};
    auto out = m.infer(in);

    // W * x = [2, -1], softmax([2,-1]) → class 0
    EXPECT_EQ(out.predicted_class, 0);
    EXPECT_GT(out.probabilities[0], out.probabilities[1]);
}

// ── ModelLoader tests ─────────────────────────────────────────────────────────

class ModelLoaderTest : public ::testing::Test {
protected:
    void SetUp() override {
        init_logging();
        telemetry::Telemetry::instance().init("/tmp/test_loader_tel");

        m_dir = fs::temp_directory_path() / "dh_loader_test";
        fs::create_directories(m_dir / "layer0_model_l0");
        fs::create_directories(m_dir / "advanced_model" / "layer0_rebuild");

        // Place test safetensors in expected locations
        write_test_safetensors((m_dir / "layer0_model_l0").string());
        // Rename to expected name
        fs::rename(m_dir / "layer0_model_l0" / "test_model.safetensors",
                   m_dir / "layer0_model_l0" / "layer0.safetensors");

        m_csv  = write_test_csv((m_dir / "advanced_model" / "layer0_rebuild").string());
        m_jsonl = write_test_jsonl(m_dir.string());
    }
    fs::path m_dir;
    std::string m_csv;
    std::string m_jsonl;
};

TEST_F(ModelLoaderTest, DiscoverLoadsAtLeastOneModel)
{
    ModelLoader loader(m_dir.string());
    int n = loader.discover_and_load(ModelRegistry::instance());
    EXPECT_GE(n, 1);
    EXPECT_NO_THROW(ModelRegistry::instance().get("layer0_merged"));
    EXPECT_NO_THROW(ModelRegistry::instance().get("grand_unified"));
}

TEST_F(ModelLoaderTest, EmptyDirRegistersStubFallbacks)
{
    ModelLoader loader("/tmp/nonexistent_models_xyz");
    loader.discover_and_load(ModelRegistry::instance());
    EXPECT_NO_THROW(ModelRegistry::instance().get("layer0_merged"));
    EXPECT_NO_THROW(ModelRegistry::instance().get("grand_unified"));
    EXPECT_TRUE(ModelRegistry::instance().get("layer0_merged")->is_loaded());
}

TEST_F(ModelLoaderTest, ImportTrainingLogDoesNotThrow)
{
    ModelLoader loader(m_dir.string());
    EXPECT_NO_THROW(loader.import_training_log(m_csv));
}

TEST_F(ModelLoaderTest, ImportMissingLogDoesNotThrow)
{
    ModelLoader loader(m_dir.string());
    EXPECT_NO_THROW(loader.import_training_log("/tmp/no_such_log.csv"));
}

TEST_F(ModelLoaderTest, ImportTelemetryJsonlDoesNotThrow)
{
    ModelLoader loader(m_dir.string());
    EXPECT_NO_THROW(loader.import_telemetry_jsonl(m_jsonl));
}

TEST_F(ModelLoaderTest, ManifestLoadedModel)
{
    // Place safetensors + manifest in same dir
    std::string subdir = (m_dir / "manifest_test").string();
    fs::create_directories(subdir);
    write_test_safetensors(subdir);
    // Rename to something the manifest references
    fs::rename(subdir + "/test_model.safetensors", subdir + "/test_manifest_model.safetensors");
    write_test_manifest(subdir, "test_manifest_model.safetensors");

    ModelLoader loader(m_dir.string());
    loader.discover_and_load(ModelRegistry::instance());

    EXPECT_NO_THROW(ModelRegistry::instance().get("test_manifest_model"));
    auto m = ModelRegistry::instance().get("test_manifest_model");
    EXPECT_TRUE(m->is_loaded());
}

TEST_F(ModelLoaderTest, LoadedModelCanInfer)
{
    ModelLoader loader(m_dir.string());
    loader.discover_and_load(ModelRegistry::instance());

    auto m = ModelRegistry::instance().get("layer0_merged");
    EXPECT_TRUE(m->is_loaded());

    ModelInput in;
    in.features.assign(64, 0.1f);
    auto out = m->infer(in);
    EXPECT_GE(out.predicted_class, 0);
    EXPECT_LE(out.predicted_class, 4);
    float s = 0;
    for (auto p : out.probabilities) s += p;
    EXPECT_NEAR(s, 1.f, 1e-3f);
}

// ── StubModel registry tests (regression guard) ───────────────────────────────

TEST(ModelRegistryRegression, FactoryMakesCorrectType)
{
    // make_layer0 must return SafeTensorsModel (or OnnxModel with USE_ONNXRUNTIME),
    // never StubModel which would carry a version suffix of "-stub"
    auto m = models::make_layer0();
    EXPECT_EQ(m->name(), "layer0_merged");
    const bool is_stub_version = m->version().size() >= 5 &&
        m->version().substr(m->version().size() - 5) == "-stub";
    EXPECT_FALSE(is_stub_version)
        << "Expected non-stub backend, got version: " << m->version();
}

TEST(ModelRegistryRegression, FactoryGrandUnifiedCorrectName)
{
    auto m = models::make_grand_unified();
    EXPECT_EQ(m->name(), "grand_unified");
}

TEST(ModelRegistryRegression, FactoryModelMerged)
{
    auto m = models::make_model_merged();
    EXPECT_EQ(m->name(), "model_merged");
}

TEST(ModelRegistryRegression, FactoryLayer0Rebuild)
{
    auto m = models::make_layer0_rebuild();
    EXPECT_EQ(m->name(), "layer0_rebuild");
}
