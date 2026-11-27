add_executable(engine_bench bench_main.cpp bench_mesher.cpp)
target_link_libraries(engine_bench PRIVATE engine::engine thirdparty::nanobench)
