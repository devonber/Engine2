add_executable(sandbox main.cpp)
target_link_libraries(sandbox PRIVATE engine::renderer)

# Development builds read assets straight from the source tree, so edited
# textures show up on the next run without a rebuild. Override with --assets.
target_compile_definitions(sandbox PRIVATE ENGINE_ASSET_DIR="${CMAKE_SOURCE_DIR}/assets")

# Make `sandbox` the default debug target in Visual Studio's generator.
set_property(DIRECTORY ${CMAKE_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT sandbox)
