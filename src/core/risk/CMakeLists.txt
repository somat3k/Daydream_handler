add_library(risk_manager INTERFACE)
target_include_directories(risk_manager INTERFACE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(risk_manager INTERFACE data_pipeline)
