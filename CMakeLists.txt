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
option(ENABLE_TESTS       "Build unit and integration tests"       ON)
option(ENABLE_TELEMETRY   "Enable hyper-extensive telemetry mode"  ON)
option(USE_LIBTORCH       "Link against LibTorch for ML inference" ON)
option(USE_REDIS          "Link against hiredis for Redis storage" ON)

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

find_package(spdlog QUIET)
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

if(ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ── Install rules ────────────────────────────────────────────────────────────
install(TARGETS daydream_handler
    RUNTIME DESTINATION bin
)
