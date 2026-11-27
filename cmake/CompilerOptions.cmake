# Compiler / linker configuration shared by every first-party target.
#
# Engine targets link `engine::options`. Third-party code does not, so our
# strict warnings never fire on code we do not own.
#
# Build configurations:
#   Debug          - no optimisation, asserts on. For stepping through code.
#   Release        - full optimisation + LTO, no debug info. What ships.
#   RelWithDebInfo - same code generation as Release, plus PDBs/DWARF.
#                    This is the "Profile" config: profile what you ship.
include_guard(GLOBAL)

add_library(engine_options INTERFACE)
add_library(engine::options ALIAS engine_options)

target_compile_features(engine_options INTERFACE cxx_std_23)
target_compile_definitions(engine_options INTERFACE
    $<$<CONFIG:Debug>:ENGINE_ENABLE_ASSERTS=1>
    $<$<PLATFORM_ID:Windows>:NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS>)

set(_engine_is_msvc_frontend OFF)
if(MSVC)  # true for both cl.exe and clang-cl
    set(_engine_is_msvc_frontend ON)
endif()

# ---- Warnings ---------------------------------------------------------------
if(_engine_is_msvc_frontend)
    target_compile_options(engine_options INTERFACE
        /W4
        /permissive-          # standards conformance
        /Zc:__cplusplus       # report the real __cplusplus value
        /utf-8)
    if(ENGINE_WARNINGS_AS_ERRORS)
        target_compile_options(engine_options INTERFACE /WX)
    endif()
else()
    target_compile_options(engine_options INTERFACE
        -Wall -Wextra -Wpedantic
        -Wshadow -Wnon-virtual-dtor -Wold-style-cast -Wcast-align
        -Woverloaded-virtual -Wnull-dereference -Wdouble-promotion
        -Wimplicit-fallthrough
        # Fields omitted from designated initializers are zero-initialised by
        # the language, and the idiomatic way to fill Vulkan structs. GCC 13
        # still warns about them.
        -Wno-missing-field-initializers)
    if(ENGINE_WARNINGS_AS_ERRORS)
        target_compile_options(engine_options INTERFACE -Werror)
    endif()
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(engine_options INTERFACE
        /Zc:preprocessor      # conforming preprocessor (__VA_OPT__ etc.)
        /Zc:inline            # drop unreferenced COMDATs early
        /Zc:throwingNew
        /volatile:iso
        /diagnostics:caret
        $<$<NOT:$<CONFIG:Debug>>:/Oi /Gy /Gw>)  # intrinsics, function + data COMDAT folding
endif()

# ---- Target ISA ---------------------------------------------------------------
if(ENGINE_ENABLE_AVX2 AND CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64|x64")
    if(_engine_is_msvc_frontend)
        target_compile_options(engine_options INTERFACE /arch:AVX2)
    else()
        target_compile_options(engine_options INTERFACE -march=x86-64-v3)
    endif()
endif()

# ---- Per-config flag tuning (global, so dependencies benefit too) -------------
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # Release ships /Ob2 and RelWithDebInfo only /Ob1 (barely inlines). Use the
    # most aggressive inliner in both so the profiled binary matches shipping.
    foreach(cfg RELEASE RELWITHDEBINFO)
        string(REGEX REPLACE "/Ob[0-3]" "/Ob3" CMAKE_CXX_FLAGS_${cfg} "${CMAKE_CXX_FLAGS_${cfg}}")
    endforeach()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    string(REPLACE "-O2" "-O3" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
endif()

if(_engine_is_msvc_frontend)
    # With /DEBUG the MSVC linker disables dead-code removal and identical-code
    # folding and turns on incremental linking. Put them back for the profile build.
    foreach(kind EXE SHARED MODULE)
        string(REGEX REPLACE "/INCREMENTAL(:YES)?( |$)" "/INCREMENTAL:NO\\2"
            CMAKE_${kind}_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_${kind}_LINKER_FLAGS_RELWITHDEBINFO}")
        string(APPEND CMAKE_${kind}_LINKER_FLAGS_RELWITHDEBINFO " /OPT:REF /OPT:ICF")
    endforeach()
endif()

# ---- Link-time optimisation -----------------------------------------------------
if(ENGINE_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT _engine_ipo_ok OUTPUT _engine_ipo_msg LANGUAGES CXX)
    if(_engine_ipo_ok)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON)
    else()
        message(STATUS "LTO not supported by this toolchain: ${_engine_ipo_msg}")
    endif()
endif()

# ---- Sanitizers -------------------------------------------------------------------
if(ENGINE_ENABLE_SANITIZERS)
    if(_engine_is_msvc_frontend)
        target_compile_options(engine_options INTERFACE /fsanitize=address)
        # MSVC ASan cannot be combined with /RTC runtime checks or incremental linking.
        string(REGEX REPLACE "/RTC[1csu]+" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
        foreach(kind EXE SHARED MODULE)
            string(REGEX REPLACE "/INCREMENTAL(:YES)?( |$)" "/INCREMENTAL:NO\\2"
                CMAKE_${kind}_LINKER_FLAGS_DEBUG "${CMAKE_${kind}_LINKER_FLAGS_DEBUG}")
        endforeach()
    else()
        target_compile_options(engine_options INTERFACE
            -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined)
        target_link_options(engine_options INTERFACE -fsanitize=address,undefined)
    endif()
endif()
