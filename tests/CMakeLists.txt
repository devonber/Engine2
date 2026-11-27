add_executable(engine_tests
    main.cpp
    test_chunk.cpp
    test_mesher.cpp
    test_thread_pool.cpp
    test_world.cpp)
target_link_libraries(engine_tests PRIVATE engine::engine thirdparty::doctest)

if(ENGINE_BUILD_RENDERER)
    # CPU-side renderer code only (no GPU or window needed).
    target_sources(engine_tests PRIVATE test_block_textures.cpp)
    target_link_libraries(engine_tests PRIVATE engine::renderer)
endif()

add_test(NAME engine_tests COMMAND engine_tests)
