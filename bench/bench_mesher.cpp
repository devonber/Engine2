#include "engine/voxel/Generate.hpp"
#include "engine/voxel/Mesher.hpp"

#include <nanobench.h>

#include <chrono>
#include <memory>
#include <vector>

using namespace engine;
using namespace engine::voxel;

void runMesherBenchmarks()
{
    auto chunk = std::make_unique<Chunk>();
    std::vector<PackedQuad> quads;
    quads.reserve(kChunkVolume * 3); // never reallocate inside the timed loop

    const auto bench = [&](const char* title) {
        ankerl::nanobench::Bench b;
        b.title(title).unit("chunk").warmup(50).minEpochTime(std::chrono::milliseconds(20)).relative(true);
        b.run("naive", [&] {
            meshChunkNaive(*chunk, quads);
            ankerl::nanobench::doNotOptimizeAway(quads.data());
        });
        b.run("binary", [&] {
            meshChunkBinary(*chunk, quads);
            ankerl::nanobench::doNotOptimizeAway(quads.data());
        });
    };

    fillTerrain(*chunk, 0, 0);
    bench("mesh: terrain");

    fillRandom(*chunk, 0.5, 42);
    bench("mesh: random 50% (worst case)");

    chunk->fill(1);
    bench("mesh: solid");
}
