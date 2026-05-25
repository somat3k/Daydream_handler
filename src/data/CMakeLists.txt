add_library(data_pipeline INTERFACE)
target_include_directories(data_pipeline INTERFACE
    ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(data_pipeline INTERFACE
    logging
    nlohmann_json::nlohmann_json
)
