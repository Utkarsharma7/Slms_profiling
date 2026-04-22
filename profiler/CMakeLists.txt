cmake_minimum_required(VERSION 3.22)
project(llama_phase_profiler CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ----------------------------------------------------------------------
# llama.cpp dependency
#
# Two ways to provide llama.cpp:
#   1. Place the source at  profiler/third_party/llama.cpp  (recommended for
#      reproducible offline builds; pin a known-good commit).
#   2. Let CMake fetch it via FetchContent (requires network on the build host).
#
# We disable every optional extra: no llama-server, no examples, no tests,
# no curl -- just the core libllama + ggml that we link against.
# ----------------------------------------------------------------------

set(LLAMACPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp")

# Pinned SHA: update here when you bump llama.cpp (see profiler/README.md).
set(LLAMACPP_GIT_TAG "b3038" CACHE STRING "llama.cpp git tag/commit to use")

option(LLAMA_BUILD_TESTS    "" OFF)
option(LLAMA_BUILD_EXAMPLES "" OFF)
option(LLAMA_BUILD_SERVER   "" OFF)
option(LLAMA_BUILD_COMMON   "" OFF)
option(LLAMA_CURL           "" OFF)
option(LLAMA_LLGUIDANCE     "" OFF)

# Android-specific perf knobs (safe no-ops on non-Android)
option(GGML_LLAMAFILE       "" ON)
option(GGML_CPU_AARCH64     "" ON)
option(GGML_OPENMP          "" OFF)
option(GGML_CCACHE          "" OFF)

if(EXISTS "${LLAMACPP_DIR}/CMakeLists.txt")
    message(STATUS "Using local llama.cpp at: ${LLAMACPP_DIR}")
    add_subdirectory(${LLAMACPP_DIR} ${CMAKE_BINARY_DIR}/llama.cpp EXCLUDE_FROM_ALL)
else()
    message(STATUS "Fetching llama.cpp @ ${LLAMACPP_GIT_TAG} via FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        llama_cpp
        GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
        GIT_TAG        ${LLAMACPP_GIT_TAG}
        GIT_SHALLOW    TRUE
        SOURCE_DIR     ${LLAMACPP_DIR}
    )
    FetchContent_MakeAvailable(llama_cpp)
endif()

# ----------------------------------------------------------------------
# The profiler binary itself
# ----------------------------------------------------------------------
add_executable(llama-phase-profiler src/llama_phase_profiler.cpp)

target_link_libraries(llama-phase-profiler PRIVATE llama ggml)
target_include_directories(llama-phase-profiler PRIVATE
    ${LLAMACPP_DIR}/include
    ${LLAMACPP_DIR}/ggml/include
)

# Try to link statically where we can so the output is a single file on the
# device. On Android/NDK libc/libm/libdl are always shared, which is expected.
if(ANDROID)
    target_link_options(llama-phase-profiler PRIVATE -static-libstdc++)
    target_compile_options(llama-phase-profiler PRIVATE
        -Wall -Wextra -Wno-unused-parameter -O3
    )
endif()

install(TARGETS llama-phase-profiler RUNTIME DESTINATION bin)
