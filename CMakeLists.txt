cmake_minimum_required(VERSION 3.22)
project(DaydreamHandler
    VERSION 1.0.0
    DESCRIPTION "Institutional-grade C++ algo-trading system with ML inference"
    LANGUAGES CXX
)

# ── Language standard ────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Build type default ───────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ── Options ──────────────────────────────────────────────────────────────────
option(ENABLE_TESTS       "Build unit and integration tests"               ON)
option(ENABLE_TELEMETRY   "Enable hyper-extensive telemetry mode"          ON)
option(USE_LIBTORCH       "Link against LibTorch for ML inference"         ON)
option(USE_ONNXRUNTIME    "Link against ONNX Runtime for ML inference"     ON)
option(USE_REDIS          "Link against hiredis for Redis storage"         ON)
option(USE_BITGET         "Enable Bitget exchange connector (needs libcurl+OpenSSL)" OFF)
option(USE_HDF5           "Enable H5/Keras model loading via HighFive+HDF5" ON)
option(USE_TFLITE         "Enable Keras/TFLite model loading via TFLite C API" OFF)

# ── CMake modules ────────────────────────────────────────────────────────────
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(cmake/dependencies.cmake)
include(cmake/compiler_flags.cmake)

# ── External dependencies ─────────────────────────────────────────────────────
find_package(Threads REQUIRED)

if(USE_REDIS)
    find_package(hiredis QUIET)
    if(NOT hiredis_FOUND)
        message(WARNING "hiredis not found – Redis features disabled at link time")
        set(USE_REDIS OFF)
    endif()
endif()

if(USE_LIBTORCH)
    find_package(Torch QUIET)
    if(NOT Torch_FOUND)
        message(WARNING "LibTorch not found – ML inference will use stub backend")
        set(USE_LIBTORCH OFF)
    endif()
endif()

if(USE_ONNXRUNTIME)
    find_package(onnxruntime QUIET)
    if(NOT onnxruntime_FOUND)
        message(WARNING "ONNX Runtime not found – ML inference will use stub backend")
        set(USE_ONNXRUNTIME OFF)
    endif()
endif()

if(USE_HDF5)
    find_package(HDF5 QUIET COMPONENTS C CXX)
    if(NOT HDF5_FOUND)
        message(WARNING "HDF5 not found – H5/Keras model loading disabled")
        set(USE_HDF5 OFF)
    else()
        include(FetchContent)
        FetchContent_Declare(
            HighFive
            GIT_REPOSITORY https://github.com/BlueBrain/HighFive.git
            GIT_TAG        v2.9.0
        )
        set(HIGHFIVE_USE_BOOST OFF CACHE BOOL "")
        set(HIGHFIVE_BUILD_DOCS OFF CACHE BOOL "")
        set(HIGHFIVE_EXAMPLES OFF CACHE BOOL "")
        FetchContent_MakeAvailable(HighFive)
    endif()
endif()

if(USE_TFLITE)
    find_library(TFLITE_LIB tensorflow-lite QUIET)
    find_path(TFLITE_INCLUDE_DIR tensorflow/lite/c/c_api.h QUIET)
    if(NOT TFLITE_LIB OR NOT TFLITE_INCLUDE_DIR)
        message(WARNING "TFLite not found – Keras/TFLite model loading disabled")
        set(USE_TFLITE OFF)
    endif()
endif()


if(NOT spdlog_FOUND)
    message(STATUS "spdlog not found via find_package – using FetchContent fallback")
    include(FetchContent)
    FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG        v1.13.0
    )
    FetchContent_MakeAvailable(spdlog)
endif()

find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG        v3.11.3
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# ── Sub-directories ──────────────────────────────────────────────────────────
add_subdirectory(src)
add_subdirectory(src/exchange)

if(ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ── Install rules ────────────────────────────────────────────────────────────
install(TARGETS daydream_handler
    RUNTIME DESTINATION bin
)
