add_executable(sandbox main.cpp)
target_link_libraries(sandbox PRIVATE engine::engine)

# Make `sandbox` the default debug target in Visual Studio's generator.
set_property(DIRECTORY ${CMAKE_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT sandbox)
