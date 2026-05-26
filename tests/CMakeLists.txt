cmake_minimum_required(VERSION 3.22)

# ── Unit tests ────────────────────────────────────────────────────────────────
add_executable(unit_tests
    unit/test_timeseries.cpp
    unit/test_indicators.cpp
    unit/test_feature_composer.cpp
    unit/test_risk_manager.cpp
    unit/test_checkpoint.cpp
    unit/test_inference.cpp
    unit/test_model_loader.cpp
)

target_include_directories(unit_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(unit_tests PRIVATE
    GTest::gtest_main
    trading_engine
    strategies
    indicators
    inference_engine
    model_loader
    storage
    logging
)

include(GoogleTest)
gtest_discover_tests(unit_tests)

add_custom_target(all_unit_tests DEPENDS unit_tests)
