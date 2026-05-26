add_subdirectory(logging)
add_subdirectory(data)
add_subdirectory(storage)
add_subdirectory(ml/inference)
add_subdirectory(ml/loader)
add_subdirectory(core/indicators)
add_subdirectory(core/risk)
add_subdirectory(core/strategy)
add_subdirectory(core/engine)
add_subdirectory(cli)

add_executable(daydream_handler main.cpp)
target_link_libraries(daydream_handler PRIVATE
    cli
    trading_engine
    strategies
    indicators
    inference_engine
    model_loader
    storage
    logging
    Threads::Threads
)
target_include_directories(daydream_handler PRIVATE ${CMAKE_SOURCE_DIR}/src)
