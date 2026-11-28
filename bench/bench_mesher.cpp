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

    // A surface chunk with all 26 terrain neighbours, as in the streamed world:
    // the layer that contains the column's middle surface height.
    constexpr i32 cx = 0, cz = 0;
    const i32 cy = [] {
        TerrainColumn column;
        terrainColumn(cx, cz, column);
        return (column.minHeight + column.maxHeight) / 2 / static_cast<i32>(kChunkSize);
    }();
    std::vector<std::unique_ptr<Chunk>> owned;
    std::vector<OuterLayers> outer(kNeighbourhoodSize);
    ChunkNeighbourhood terrainNeighbours;
    std::array<const OuterLayers*, kNeighbourhoodSize> terrainOuter{};
    for (i32 dz = -1; dz <= 1; ++dz) {
        for (i32 dy = -1; dy <= 1; ++dy) {
            for (i32 dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0 && dz == 0) {
                    continue;
                }
                owned.push_back(std::make_unique<Chunk>());
                fillTerrain(*owned.back(), cx + dx, cy + dy, cz + dz);
                const usize i = neighbourIndex(dx, dy, dz);
                terrainNeighbours.chunks[i] = owned.back().get();
                outer[i] = computeOuterLayers(*owned.back());
                terrainOuter[i] = &outer[i];
            }
        }
    }

    const auto bench = [&](const char* title, const ChunkNeighbourhood& neighbours) {
        const NeighbourShell shell = buildNeighbourShell(neighbours);
        ankerl::nanobench::Bench b;
        b.title(title).unit("chunk").warmup(50).minEpochTime(std::chrono::milliseconds(20)).relative(true);
        b.run("naive", [&] {
            meshChunkNaive(*chunk, neighbours, quads);
            ankerl::nanobench::doNotOptimizeAway(quads.data());
        });
        const usize naiveQuads = quads.size();
        b.run("binary", [&] {
            meshChunkBinary(*chunk, shell, quads);
            ankerl::nanobench::doNotOptimizeAway(quads.data());
        });
        b.run("greedy, no AO", [&] {
            meshChunkGreedy(*chunk, shell, quads, false);
            ankerl::nanobench::doNotOptimizeAway(quads.data());
        });
        const usize noAoQuads = quads.size();
        b.run("greedy", [&] {
            meshChunkGreedy(*chunk, shell, quads);
            ankerl::nanobench::doNotOptimizeAway(quads.data());
        });
        std::printf(
            "quads: naive/binary %zu, greedy %zu (%.1fx fewer), greedy without AO %zu\n", naiveQuads, quads.size(),
            quads.empty() ? 0.0 : static_cast<double>(naiveQuads) / static_cast<double>(quads.size()), noAoQuads);
    };

    fillTerrain(*chunk, cx, cy, cz);
    bench("mesh: terrain, with neighbours", terrainNeighbours);
    {
        ankerl::nanobench::Bench b;
        b.title("neighbour shell").unit("chunk").warmup(50).minEpochTime(std::chrono::milliseconds(20));
        b.run("from 26 outer layers",
              [&] { ankerl::nanobench::doNotOptimizeAway(shellFromOuterLayers(terrainOuter).corners); });
    }

    fillRandom(*chunk, 0.5, 42);
    bench("mesh: random 50% (worst case)", {});

    chunk->fill(1);
    bench("mesh: solid", {});
}
