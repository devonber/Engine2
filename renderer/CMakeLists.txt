# ---- Third-party implementation units (volk, VMA) -------------------------------
# Kept in their own library so our strict warnings never apply to their code.
add_library(engine_vk_thirdparty STATIC src/ThirdPartyImpl.cpp)
target_link_libraries(engine_vk_thirdparty PUBLIC thirdparty::vulkan)

# ---- Shaders: GLSL -> SPIR-V, embedded as C arrays (glslc -mfmt=c) ---------------
set(_shader_out ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${_shader_out})
set(_shader_sources shaders/voxel.vert shaders/voxel.frag)
set(_shader_outputs)
foreach(src ${_shader_sources})
    get_filename_component(name ${src} NAME)
    set(out ${_shader_out}/${name}.spv.inc)
    add_custom_command(
        OUTPUT  ${out}
        COMMAND ${ENGINE_GLSLC} --target-env=vulkan1.3 -mfmt=c
                # Debug: -g keeps GLSL source for shader debugging in RenderDoc.
                "$<IF:$<CONFIG:Debug>,-g;-O0,-O>"
                -MD -MF ${out}.d -o ${out} ${CMAKE_CURRENT_SOURCE_DIR}/${src}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${src}
        DEPFILE ${out}.d
        COMMAND_EXPAND_LISTS
        COMMENT "Compiling shader ${name}"
        VERBATIM)
    list(APPEND _shader_outputs ${out})
endforeach()

# ---- Renderer ------------------------------------------------------------------------
add_library(engine_renderer STATIC
    src/Renderer.cpp
    src/Swapchain.cpp
    src/VulkanContext.cpp
    src/Window.cpp
    ${_shader_sources}
    ${_shader_outputs})
add_library(engine::renderer ALIAS engine_renderer)

target_include_directories(engine_renderer
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${_shader_out})
target_link_libraries(engine_renderer
    PUBLIC  engine::engine thirdparty::vulkan thirdparty::glm
    PRIVATE engine_vk_thirdparty glfw)

file(GLOB_RECURSE _renderer_headers CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)
target_sources(engine_renderer PRIVATE ${_renderer_headers})
source_group("Shaders" FILES ${_shader_sources})
