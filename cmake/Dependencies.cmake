# Third-party dependencies, fetched at configure time and pinned to exact
# commits so every machine builds the same code.
include_guard(GLOBAL)
include(FetchContent)

set(FETCHCONTENT_QUIET OFF)

# Header-only libraries: SOURCE_SUBDIR points at a folder that does not exist,
# so FetchContent only downloads them and never runs their build scripts
# (which drag in their own tests, old cmake_minimum_required, etc.).
# We wrap them in our own INTERFACE targets instead.

if(ENGINE_BUILD_TESTS)
    FetchContent_Declare(doctest
        GIT_REPOSITORY https://github.com/doctest/doctest.git
        GIT_TAG        ae7a13539fb71f270b87eb2e874fbac80bc8dda2  # v2.4.11
        SOURCE_SUBDIR  _download_only)
    FetchContent_MakeAvailable(doctest)
    add_library(engine_doctest INTERFACE)
    add_library(thirdparty::doctest ALIAS engine_doctest)
    target_include_directories(engine_doctest SYSTEM INTERFACE ${doctest_SOURCE_DIR})
endif()

if(ENGINE_BUILD_BENCHMARKS)
    FetchContent_Declare(nanobench
        GIT_REPOSITORY https://github.com/martinus/nanobench.git
        GIT_TAG        a5a50c2b33eea2ff1fcb355cacdface43eb42b25  # v4.3.11
        SOURCE_SUBDIR  _download_only)
    FetchContent_MakeAvailable(nanobench)
    add_library(engine_nanobench INTERFACE)
    add_library(thirdparty::nanobench ALIAS engine_nanobench)
    target_include_directories(engine_nanobench SYSTEM INTERFACE ${nanobench_SOURCE_DIR}/src/include)
endif()

if(ENGINE_ENABLE_TRACY)
    # Tracy: frame/zone profiler. Run the Tracy GUI (same version, v0.11.1) and
    # it connects to the running game. See README.
    set(TRACY_ENABLE ON CACHE BOOL "" FORCE)
    set(TRACY_ON_DEMAND ON CACHE BOOL "Only collect while a profiler is connected" FORCE)
    FetchContent_Declare(tracy
        GIT_REPOSITORY https://github.com/wolfpld/tracy.git
        GIT_TAG        5d542dc09f3d9378d005092a4ad446bd405f819a  # v0.11.1
    )
    FetchContent_MakeAvailable(tracy)
endif()
