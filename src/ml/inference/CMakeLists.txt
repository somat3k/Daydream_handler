add_library(inference_engine STATIC
    inference_engine.cpp
    feature_composer.cpp
    onnx_model.cpp
    manifest_feature_composer.cpp
    model_loader.cpp
    safetensors_model.cpp
    h5_model.cpp
    keras_model.cpp
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

if(USE_ONNXRUNTIME)
    target_link_libraries(inference_engine PUBLIC onnxruntime::onnxruntime)
    target_compile_definitions(inference_engine PUBLIC USE_ONNXRUNTIME)
endif()

if(USE_HDF5)
    find_package(HDF5 REQUIRED COMPONENTS C CXX)
    target_link_libraries(inference_engine PUBLIC ${HDF5_LIBRARIES} HighFive)
    target_include_directories(inference_engine PUBLIC ${HDF5_INCLUDE_DIRS})
    target_compile_definitions(inference_engine PUBLIC USE_HDF5)
endif()

if(USE_TFLITE)
    target_link_libraries(inference_engine PUBLIC tensorflow-lite)
    target_compile_definitions(inference_engine PUBLIC USE_TFLITE)
endif()
