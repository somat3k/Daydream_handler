add_library(inference_engine STATIC
    inference_engine.cpp
    feature_composer.cpp
)
target_include_directories(inference_engine PUBLIC
    ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(inference_engine PUBLIC
    logging
    nlohmann_json::nlohmann_json
)
if(USE_LIBTORCH)
    target_link_libraries(inference_engine PUBLIC ${TORCH_LIBRARIES})
    target_compile_definitions(inference_engine PUBLIC USE_LIBTORCH)
endif()
