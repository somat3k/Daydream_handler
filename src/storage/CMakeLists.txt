add_library(storage STATIC
    redis_client.cpp
)
target_include_directories(storage PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(storage PUBLIC
    logging
    nlohmann_json::nlohmann_json
)
if(USE_REDIS)
    target_link_libraries(storage PUBLIC hiredis::hiredis)
    target_compile_definitions(storage PUBLIC USE_REDIS)
endif()
