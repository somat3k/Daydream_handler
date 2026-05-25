workspace(name = "daydream_handler")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# ── GoogleTest ────────────────────────────────────────────────────────────────
http_archive(
    name = "com_google_googletest",
    urls = ["https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz"],
    strip_prefix = "googletest-1.14.0",
    sha256 = "8ad598c73ad796e0d8280b082cebd82a630d73e73cd3c70057938a6501bba5d7",
)

# ── nlohmann/json ─────────────────────────────────────────────────────────────
http_archive(
    name = "nlohmann_json",
    urls = ["https://github.com/nlohmann/json/releases/download/v3.11.3/include.zip"],
    strip_prefix = "include",
    sha256 = "a22461d13119ac5c78f205d3df1db13403e58ce1bb1794edc9313677313f4a9d",
    build_file_content = """
cc_library(
    name = "json",
    hdrs = glob(["nlohmann/**/*.hpp"]),
    includes = ["."],
    visibility = ["//visibility:public"],
)
""",
)

# ── spdlog ────────────────────────────────────────────────────────────────────
http_archive(
    name = "spdlog",
    urls = ["https://github.com/gabime/spdlog/archive/refs/tags/v1.13.0.tar.gz"],
    strip_prefix = "spdlog-1.13.0",
    sha256 = "9df6915308f4886a5b213afe7440bdc73a0b0e31490c54dda95cf0e9ff0bf0d0",
    build_file_content = """
cc_library(
    name = "spdlog",
    hdrs = glob(["include/**/*.h"]),
    srcs = glob(["src/*.cpp"]),
    defines = ["SPDLOG_COMPILED_LIB"],
    includes = ["include"],
    visibility = ["//visibility:public"],
)
""",
)
