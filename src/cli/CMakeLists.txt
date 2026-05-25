add_library(cli STATIC
    cli.cpp
)
target_include_directories(cli PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(cli PUBLIC
    trading_engine
    storage
    logging
)
