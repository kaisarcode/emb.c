cmake_minimum_required(VERSION 3.14)
project(emb.c C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Performance optimizations
if(NOT MSVC)
    add_compile_options(-O3 -march=native -fno-finite-math-only)
    add_definitions(-D_GNU_SOURCE -DGGML_USE_CPU -DGGML_VERSION="1" -DGGML_COMMIT="custom")
else()
    add_compile_options(/O2 /arch:AVX2)
    add_definitions(-DGGML_USE_CPU -DGGML_VERSION="1" -DGGML_COMMIT="custom")
endif()

# Include directories
include_directories(
    .
    ggml/include
    ggml/src
    ggml/src/ggml-cpu
)

# GGML Sources (CPU & x86)
file(GLOB GGML_SOURCES 
    "ggml/src/*.c"
    "ggml/src/*.cpp"
    "ggml/src/ggml-cpu/*.c"
    "ggml/src/ggml-cpu/*.cpp"
    "ggml/src/ggml-cpu/arch/x86/*.c"
    "ggml/src/ggml-cpu/arch/x86/*.cpp"
)

# Main executable
add_executable(emb 
    emb.c 
    libemb.c 
    model.c
    ${GGML_SOURCES}
)

# System libraries
if(UNIX)
    target_link_libraries(emb pthread dl m)
endif()
