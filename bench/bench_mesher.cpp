#include "engine/voxel/Generate.hpp"
#include "engine/voxel/Mesher.hpp"

#include <nanobench.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <memory>
#include <vector>

using namespace engine;
using namespace engine::voxel;

void runMesherBenchmarks()
{
    auto chunk = std::make_unique<Chunk>();
    std::vector<PackedQuad> quads;
    quads.reserve(kChunkVolume * 3); // never reallocate inside the timed loop

    // Terrain neighbours on all four sides, solid ground below: a chunk in
    // the middle of the sandbox world.
    std::array<std::unique_ptr<Chunk>, 4> side;
    for (auto& c : side) {
        c = std::make_unique<Chunk>();
    }
    fillTerrain(*side[0], 1, 0);
    fillTerrain(*side[1], -1, 0);
    fillTerrain(*side[2], 0, 1);
    fillTerrain(*side[3], 0, -1);
    ChunkNeighbours terrainNeighbours;
    terrainNeighbours.chunks[static_cast<usize>(Face::PosX)] = side[0].get();
    terrainNeighbours.chunks[static_cast<usize>(Face::NegX)] = side[1].get();
    terrainNeighbours.chunks[static_cast<usize>(Face::PosZ)] = side[2].get();
    terrainNeighbours.chunks[static_cast<usize>(Face::NegZ)] = side[3].get();

    const auto bench = [&](const char* title, const ChunkNeighbours& neighbours) {
        BoundaryMasks masks = buildBoundaryMasks(neighbours);
        ankerl::nanobench::Bench b;
        b.title(title).unit("chunk").warmup(50).minEpochTime(std::chrono::milliseconds(20)).relative(true);
        b.run("naive", [&] {
            meshChunkNaive(*chunk, neighbours, quads);
            ankerl::nanobench::doNotOptimizeAway(quads.data());
        });
        const usize naiveQuads = quads.size();
        b.run("binary", [&] {
            meshChunkBinary(*chunk, masks, quads);
            ankerl::nanobench::doNotOptimizeAway(quads.data());
        });
        b.run("greedy", [&] {
            meshChunkGreedy(*chunk, masks, quads);
            ankerl::nanobench::doNotOptimizeAway(quads.data());
        });
        std::printf("quads: naive/binary %zu, greedy %zu (%.1fx fewer)\n", naiveQuads, quads.size(),
                    quads.empty() ? 0.0 : static_cast<double>(naiveQuads) / static_cast<double>(quads.size()));
        b.run("boundary masks", [&] {
            masks = buildBoundaryMasks(neighbours);
            ankerl::nanobench::doNotOptimizeAway(masks.rows.data());
        });
    };

    fillTerrain(*chunk, 0, 0);
    bench("mesh: terrain, with neighbours", terrainNeighbours);

    fillRandom(*chunk, 0.5, 42);
    bench("mesh: random 50% (worst case)", {});

    chunk->fill(1);
    bench("mesh: solid", {});
}
