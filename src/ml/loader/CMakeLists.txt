add_library(model_loader STATIC
    model_loader.cpp
)
target_include_directories(model_loader PUBLIC
    ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(model_loader PUBLIC
    inference_engine
    logging
    nlohmann_json::nlohmann_json
)
