add_library(exchange STATIC
    bitget/bitget_auth.cpp
    bitget/bitget_rest_client.cpp
    bitget/bitget_ws_client.cpp
)

target_include_directories(exchange PUBLIC
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(exchange PUBLIC
    logging
    nlohmann_json::nlohmann_json
)

if(USE_BITGET)
    find_package(CURL REQUIRED)
    find_package(OpenSSL REQUIRED)
    target_link_libraries(exchange PUBLIC CURL::libcurl OpenSSL::SSL OpenSSL::Crypto)
    target_compile_definitions(exchange PUBLIC USE_BITGET)
endif()
