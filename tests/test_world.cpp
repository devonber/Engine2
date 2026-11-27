#include "engine/core/ThreadPool.hpp"
#include "engine/voxel/Blocks.hpp"
#include "engine/voxel/Generate.hpp"
#include "engine/voxel/World.hpp"

#include <doctest/doctest.h>

#include <algorithm>
#include <memory>

using namespace engine;
using namespace engine::voxel;

TEST_CASE("World building gives identical results on any number of threads")
{
    const WorldBuildOptions options{.radius = 3};
    const WorldMesh serial = buildTerrainWorld(options, nullptr);
    ThreadPool pool(3);
    const WorldMesh parallel = buildTerrainWorld(options, &pool);

    REQUIRE(serial.chunks.size() == 36);
    REQUIRE(parallel.chunks.size() == serial.chunks.size());
    for (usize i = 0; i < serial.chunks.size(); ++i) {
        CHECK(parallel.chunks[i].chunkX == serial.chunks[i].chunkX);
        CHECK(parallel.chunks[i].chunkZ == serial.chunks[i].chunkZ);
        CHECK(parallel.chunks[i].firstQuad == serial.chunks[i].firstQuad);
        CHECK(parallel.chunks[i].quadCount == serial.chunks[i].quadCount);
    }
    CHECK(parallel.quads == serial.quads);
}

TEST_CASE("World chunk ranges tile the quad buffer")
{
    const WorldMesh world = buildTerrainWorld({.radius = 2, .greedy = false}, nullptr);
    u32 expected = 0;
    for (const ChunkMeshRange& c : world.chunks) {
        CHECK(c.firstQuad == expected);
        expected += c.quadCount;
    }
    CHECK(expected == world.quads.size());
}

TEST_CASE("Generators write every voxel (chunks may be allocated uninitialised)")
{
    // buildTerrainWorld constructs chunks with Chunk::Uninitialized and relies
    // on the generator to overwrite all of them. Poison and check.
    constexpr BlockId kPoison = 0xDEAD;
    auto chunk = std::make_unique<Chunk>();
    for (const i32 c : {-20, -3, 0, 5, 17}) {
        CAPTURE(c);
        chunk->fill(kPoison);
        fillTerrain(*chunk, c, -c * 3);
        CHECK(std::ranges::count(chunk->blocks(), kPoison) == 0);
    }
    chunk->fill(kPoison);
    fillRandom(*chunk, 0.5, 1);
    CHECK(std::ranges::count(chunk->blocks(), kPoison) == 0);
}

TEST_CASE("Terrain has the expected layering")
{
    auto chunk = std::make_unique<Chunk>();
    fillTerrain(*chunk, 0, 0);
    usize grass = 0, sand = 0;
    for (u32 z = 0; z < kChunkSize; ++z) {
        for (u32 x = 0; x < kChunkSize; ++x) {
            // Find the surface: the topmost solid block.
            u32 top = 0;
            for (u32 y = 0; y < kChunkSize; ++y) {
                if (chunk->get(x, y, z) != blocks::Air) {
                    top = y;
                }
            }
            const BlockId surface = chunk->get(x, top, z);
            CHECK((surface == blocks::Grass || surface == blocks::Sand));
            grass += surface == blocks::Grass;
            sand += surface == blocks::Sand;
            CHECK(chunk->get(x, 0, z) != blocks::Air); // ground goes down to y = 0
            for (u32 y = top + 1; y < kChunkSize; ++y) {
                REQUIRE(chunk->get(x, y, z) == blocks::Air); // nothing floating
            }
        }
    }
    CHECK(grass + sand == kChunkArea);
}
