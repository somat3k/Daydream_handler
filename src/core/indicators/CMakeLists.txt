add_library(indicators INTERFACE)
target_include_directories(indicators INTERFACE
    ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(indicators INTERFACE data_pipeline)
