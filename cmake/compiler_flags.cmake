# ── Compiler flags ────────────────────────────────────────────────────────────
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall -Wextra -Wpedantic
        -Wno-unused-parameter
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-march=native>
        $<$<CONFIG:Release>:-DNDEBUG>
        $<$<CONFIG:Debug>:-g3>
    )
elseif(MSVC)
    add_compile_options(/W4 /WX- /permissive-)
    add_compile_options($<$<CONFIG:Release>:/O2>)
endif()
