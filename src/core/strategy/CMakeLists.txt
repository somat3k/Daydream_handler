add_library(strategies STATIC
    strategy.cpp
)
target_include_directories(strategies PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(strategies PUBLIC
    indicators
    risk_manager
    inference_engine
    logging
)
