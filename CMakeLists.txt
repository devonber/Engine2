cmake_minimum_required(VERSION 3.25)

project(Engine2
    VERSION 0.1.0
    DESCRIPTION "Voxel game engine"
    LANGUAGES CXX)

# This project is meant to be the top level; keep build trees out of source.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "In-source builds are not allowed. Use a preset: cmake --preset <name>")
endif()

# ---- Options ---------------------------------------------------------------
option(ENGINE_BUILD_TESTS          "Build unit tests"                               ON)
option(ENGINE_BUILD_BENCHMARKS     "Build micro-benchmarks"                         ON)
option(ENGINE_BUILD_SANDBOX        "Build the sandbox executable"                   ON)
option(ENGINE_BUILD_RENDERER       "Build the Vulkan renderer (needs glslc)"         ON)
option(ENGINE_ENABLE_TRACY         "Enable the Tracy profiler client"               OFF)
option(ENGINE_ENABLE_AVX2          "Target AVX2 (x86-64-v3). Steam HW survey: ~95%+" ON)
option(ENGINE_ENABLE_LTO           "Link-time optimisation in optimised configs"     ON)
option(ENGINE_ENABLE_SANITIZERS    "AddressSanitizer (+UBSan on GCC/Clang)"         OFF)
option(ENGINE_WARNINGS_AS_ERRORS   "Treat warnings as errors in engine targets"     OFF)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # for clangd / clang-tidy

# Put every binary in one place so the sandbox, tests and benches sit together.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(CompilerOptions)   # defines engine::options, LTO, per-config flag tuning
include(Dependencies)      # third-party code via FetchContent (pinned commits)

add_subdirectory(engine)

if(ENGINE_BUILD_RENDERER)
    add_subdirectory(renderer)
endif()

if(ENGINE_BUILD_SANDBOX AND ENGINE_BUILD_RENDERER)
    add_subdirectory(apps/sandbox)
endif()

if(ENGINE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(ENGINE_BUILD_BENCHMARKS)
    add_subdirectory(bench)
endif()
