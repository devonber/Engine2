// nanobench implementation lives in exactly one translation unit.
#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

void runMesherBenchmarks();

int main()
{
    runMesherBenchmarks();
    return 0;
}
