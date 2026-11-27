add_library(engine STATIC
    src/core/ThreadPool.cpp
    src/voxel/ChunkStreamer.cpp
    src/voxel/Generate.cpp
    src/voxel/Mesher.cpp
    src/voxel/World.cpp)
add_library(engine::engine ALIAS engine)

target_include_directories(engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
find_package(Threads REQUIRED)
target_link_libraries(engine PUBLIC engine::options Threads::Threads)

if(ENGINE_ENABLE_TRACY)
    target_link_libraries(engine PUBLIC Tracy::TracyClient)
endif()

# Show headers in the Visual Studio solution view.
file(GLOB_RECURSE _engine_headers CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)
target_sources(engine PRIVATE ${_engine_headers})
