add_library(trading_engine STATIC
    trading_engine.cpp
)
target_include_directories(trading_engine PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(trading_engine PUBLIC
    strategies
    indicators
    risk_manager
    inference_engine
    data_pipeline
    logging
    nlohmann_json::nlohmann_json
)

if(USE_BITGET)
    target_link_libraries(trading_engine PUBLIC exchange)
    target_compile_definitions(trading_engine PUBLIC USE_BITGET)
endif()
