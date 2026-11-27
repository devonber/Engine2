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

if(ENGINE_BUILD_RENDERER)
    # ---- Vulkan: headers + volk + VMA. No Vulkan SDK is needed to *compile*:
    # volk loads vulkan-1.dll / libvulkan.so at runtime and fetches every
    # function pointer directly from the driver (skipping the loader's
    # dispatch trampoline on each call). The SDK is still needed for glslc
    # and the validation layers.
    FetchContent_Declare(vulkan_headers
        GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
        GIT_TAG        8864cdc896bbc2a9b6eb36b3218fc9ef57908d77  # vulkan-sdk-1.4.350.0
        SOURCE_SUBDIR  _download_only)
    FetchContent_Declare(volk
        GIT_REPOSITORY https://github.com/zeux/volk.git
        GIT_TAG        3ca312a4f38baa63d8006b6905abbeeb89c8087d  # vulkan-sdk-1.4.350.0
        SOURCE_SUBDIR  _download_only)
    FetchContent_Declare(vma
        GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
        GIT_TAG        1d8f600fd424278486eade7ed3e877c99f0846b1  # v3.3.0
        SOURCE_SUBDIR  _download_only)
    FetchContent_Declare(glm
        GIT_REPOSITORY https://github.com/g-truc/glm.git
        GIT_TAG        0af55ccecd98d4e5a8d1fad7de25ba429d60e863  # 1.0.1
        SOURCE_SUBDIR  _download_only)

    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL        OFF CACHE BOOL "" FORCE)
    # X11 only on Linux for now (Wayland needs wayland-scanner + protocols).
    set(GLFW_BUILD_WAYLAND  OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG        7b6aead9fb88b3623e3b3725ebb42670cbe4c579  # 3.4
        SYSTEM)

    FetchContent_MakeAvailable(vulkan_headers volk vma glm glfw)
    # GLFW is C and off the hot path: LTO gains nothing there, and with Clang
    # its archive would need llvm-ar to be linkable.
    set_target_properties(glfw PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE OFF
        INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO OFF)

    # Header targets. SYSTEM so our strict warnings ignore their code.
    add_library(engine_vulkan_headers INTERFACE)
    add_library(thirdparty::vulkan ALIAS engine_vulkan_headers)
    target_include_directories(engine_vulkan_headers SYSTEM INTERFACE
        ${vulkan_headers_SOURCE_DIR}/include
        ${volk_SOURCE_DIR}
        ${vma_SOURCE_DIR}/include)
    target_compile_definitions(engine_vulkan_headers INTERFACE
        VK_NO_PROTOTYPES                 # all calls go through volk
        VMA_STATIC_VULKAN_FUNCTIONS=0
        VMA_DYNAMIC_VULKAN_FUNCTIONS=1)
    target_link_libraries(engine_vulkan_headers INTERFACE ${CMAKE_DL_LIBS})

    add_library(engine_glm INTERFACE)
    add_library(thirdparty::glm ALIAS engine_glm)
    target_include_directories(engine_glm SYSTEM INTERFACE ${glm_SOURCE_DIR})
    target_compile_definitions(engine_glm INTERFACE GLM_FORCE_DEPTH_ZERO_TO_ONE GLM_FORCE_RADIANS)

    # ---- Shader compiler (from the Vulkan SDK, or the distro's glslc package)
    find_program(ENGINE_GLSLC glslc
        HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin"
        DOC "glslc from the Vulkan SDK (or shaderc)")
    if(NOT ENGINE_GLSLC)
        message(FATAL_ERROR
            "glslc not found. Install the Vulkan SDK from https://vulkan.lunarg.com/sdk/home "
            "(it sets VULKAN_SDK), then delete the CMake cache and reconfigure. "
            "Or configure with -DENGINE_BUILD_RENDERER=OFF.")
    endif()
endif()
