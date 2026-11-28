#include "engine/voxel/Blocks.hpp"
#include "engine/voxel/Generate.hpp"

#include <doctest/doctest.h>

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

using namespace engine;
using namespace engine::voxel;

TEST_CASE("Generators write every voxel (chunks may be allocated uninitialised)")
{
    // The streamer fills pooled chunks constructed with Chunk::Uninitialized
    // and relies on the generator to overwrite all of them. Poison and check.
    constexpr BlockId kPoison = 0xDEAD;
    auto chunk = std::make_unique<Chunk>();
    for (const i32 c : {-20, -3, 0, 5, 17}) {
        for (i32 cy = 0; cy < kWorldHeightChunks; ++cy) {
            CAPTURE(c);
            CAPTURE(cy);
            chunk->fill(kPoison);
            fillTerrain(*chunk, c, cy, -c * 3);
            CHECK(std::ranges::count(chunk->blocks(), kPoison) == 0);
        }
    }
    chunk->fill(kPoison);
    fillRandom(*chunk, 0.5, 1);
    CHECK(std::ranges::count(chunk->blocks(), kPoison) == 0);
}

TEST_CASE("Terrain columns have the expected layering")
{
    // Stack all layers of a few columns and check each block column:
    // solid from y = 0 up to the surface, air above, surface grass or sand.
    auto chunk = std::make_unique<Chunk>();
    for (const auto& [cx, cz] : {std::pair{0, 0}, std::pair{7, -3}, std::pair{-12, 20}}) {
        TerrainColumn column;
        terrainColumn(cx, cz, column);
        std::vector<BlockId> stack(static_cast<usize>(kWorldHeight) * kChunkArea);
        for (i32 cy = 0; cy < kWorldHeightChunks; ++cy) {
            fillTerrainChunk(*chunk, column, cy);
            for (u32 y = 0; y < kChunkSize; ++y) {
                for (u32 i = 0; i < kChunkArea; ++i) {
                    stack[(static_cast<usize>(cy) * kChunkSize + y) * kChunkArea + i] =
                        chunk->blocks()[y * kChunkArea + i];
                }
            }
        }
        for (u32 i = 0; i < kChunkArea; ++i) {
            const i32 h = column.height[i];
            REQUIRE(h >= 1);
            REQUIRE(h < kWorldHeight);
            const auto at = [&](i32 y) { return stack[static_cast<usize>(y) * kChunkArea + i]; };
            for (i32 y = 0; y < h; ++y) {
                REQUIRE(at(y) != blocks::Air);
            }
            for (i32 y = h; y < kWorldHeight; ++y) {
                REQUIRE(at(y) == blocks::Air);
            }
            CHECK((at(h - 1) == blocks::Grass || at(h - 1) == blocks::Sand));
        }
    }
}

TEST_CASE("Uniform chunk classification matches the filled chunk")
{
    auto chunk = std::make_unique<Chunk>();
    int air = 0, stone = 0, mixed = 0;
    for (i32 cx = -40; cx <= 40; cx += 8) {
        for (i32 cz = -40; cz <= 40; cz += 8) {
            TerrainColumn column;
            terrainColumn(cx, cz, column);
            for (i32 cy = 0; cy < kWorldHeightChunks; ++cy) {
                fillTerrainChunk(*chunk, column, cy);
                const auto blocks = chunk->blocks();
                switch (classifyTerrainChunk(column, cy)) {
                case ChunkContent::Air:
                    ++air;
                    REQUIRE(std::ranges::all_of(blocks, [](BlockId b) { return b == blocks::Air; }));
                    break;
                case ChunkContent::Stone:
                    ++stone;
                    REQUIRE(std::ranges::all_of(blocks, [](BlockId b) { return b == blocks::Stone; }));
                    break;
                case ChunkContent::Mixed:
                    ++mixed;
                    break;
                }
            }
        }
    }
    // The terrain spans several layers, so all three kinds occur.
    CHECK(air > 0);
    CHECK(stone > 0);
    CHECK(mixed > 0);
}
