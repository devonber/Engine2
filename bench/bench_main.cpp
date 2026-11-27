// nanobench implementation lives in exactly one translation unit.
#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

#include <cstring>

void runMesherBenchmarks();
void runWorldBenchmarks();

// Usage: engine_bench [mesh|world]   (default: all)
int main(int argc, char** argv)
{
    const char* only = argc > 1 ? argv[1] : "";
    if (*only == 0 || std::strcmp(only, "mesh") == 0) {
        runMesherBenchmarks();
    }
    if (*only == 0 || std::strcmp(only, "world") == 0) {
        runWorldBenchmarks();
    }
    return 0;
}
