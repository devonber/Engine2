#include "engine/voxel/Blocks.hpp"
#include "engine/voxel/Generate.hpp"

#include <doctest/doctest.h>

#include <algorithm>
#include <cstdlib>
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

TEST_CASE("Coarse levels of detail: cells are solid where the surface covers at least half of them")
{
    auto chunk = std::make_unique<Chunk>();
    for (u32 level = 1; level <= 4; ++level) {
        CAPTURE(level);
        const i32 scale = lodScale(level);
        const i32 layers = worldHeightChunks(level);
        for (const auto& [cx, cz] : {std::pair{0, 0}, std::pair{3, -2}, std::pair{-5, 7}}) {
            TerrainColumn column;
            terrainColumn(cx, cz, column, level);
            for (u32 z = 0; z < kChunkSize; z += 5) {
                for (u32 x = 0; x < kChunkSize; x += 3) {
                    // The full-resolution height at the cell's centre block.
                    const i32 bx = (cx * static_cast<i32>(kChunkSize) + static_cast<i32>(x)) * scale + scale / 2;
                    const i32 bz = (cz * static_cast<i32>(kChunkSize) + static_cast<i32>(z)) * scale + scale / 2;
                    const auto floorDiv = [](i32 a, i32 b) { return a >= 0 ? a / b : -((-a + b - 1) / b); };
                    TerrainColumn fine;
                    terrainColumn(floorDiv(bx, 32), floorDiv(bz, 32), fine);
                    const i32 blocks = fine.height[static_cast<usize>((bz - floorDiv(bz, 32) * 32) * 32 +
                                                                      (bx - floorDiv(bx, 32) * 32))];
                    const i32 cells = column.height[z * kChunkSize + x];
                    CAPTURE(blocks);
                    CAPTURE(cells);
                    // Rounded to the nearest cell (at least one: the world has a floor).
                    CHECK(cells == std::max(1, (blocks + scale / 2) >> level));
                    CHECK((cells == 1 || std::abs(cells * scale - blocks) <= scale / 2));
                    CHECK(cells < layers * static_cast<i32>(kChunkSize)); // fits the level's layers
                }
            }
            // Filled chunks are solid up to the cell height, with a grass or sand top.
            for (i32 cy = 0; cy < kWorldHeightChunks; ++cy) {
                fillTerrainChunk(*chunk, column, cy);
                if (cy >= layers) {
                    CHECK(classifyTerrainChunk(column, cy) == ChunkContent::Air);
                }
                for (u32 i = 0; i < kChunkArea; i += 7) {
                    const i32 h = column.height[i] - cy * static_cast<i32>(kChunkSize);
                    for (i32 y = 0; y < static_cast<i32>(kChunkSize); ++y) {
                        const BlockId b = chunk->blocks()[static_cast<usize>(y) * kChunkArea + i];
                        REQUIRE((b != blocks::Air) == (y < h));
                        if (y == h - 1) {
                            CHECK((b == blocks::Grass || b == blocks::Sand));
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("Uniform chunk classification matches the filled chunk")
{
    auto chunk = std::make_unique<Chunk>();
    int air = 0, stone = 0, mixed = 0;
    for (u32 level : {0u, 1u, 2u})
        for (i32 cx = -40; cx <= 40; cx += 8) {
            for (i32 cz = -40; cz <= 40; cz += 8) {
                TerrainColumn column;
                terrainColumn(cx, cz, column, level);
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
