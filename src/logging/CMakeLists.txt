add_library(logging STATIC
    logger.cpp
    telemetry.cpp
)
target_include_directories(logging PUBLIC
    ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(logging PUBLIC
    spdlog::spdlog
    nlohmann_json::nlohmann_json
    Threads::Threads
)
