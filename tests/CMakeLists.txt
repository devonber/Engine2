add_executable(engine_tests
    main.cpp
    test_chunk.cpp
    test_mesher.cpp)
target_link_libraries(engine_tests PRIVATE engine::engine thirdparty::doctest)

add_test(NAME engine_tests COMMAND engine_tests)
