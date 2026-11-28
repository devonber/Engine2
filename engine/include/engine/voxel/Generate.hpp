#pragma once

#include "engine/voxel/Chunk.hpp"

#include <array>

namespace engine::voxel {

// Simple procedural terrain for tests, benchmarks and the sandbox.

// The world is columns of this many chunks stacked vertically (y = 0 at the
// bottom of chunk layer 0). Below the world counts as solid, above it as air.
inline constexpr i32 kWorldHeightChunks = 4; // 128 blocks
inline constexpr i32 kWorldHeight = kWorldHeightChunks * static_cast<i32>(kChunkSize);

// Level of detail: at level L every voxel ("cell") is 2^L blocks wide and
// 2^min(L, kMaxVerticalLodLevel) blocks tall, so a level-L chunk covers
// 32 * 2^L blocks horizontally with the same 32^3 cells. Cells stop growing
// vertically at 4 blocks: the world is only 128 blocks tall, and taller cells
// would flatten distant hills into a few huge steps (at level 7, one).
// Level 0 is the full-resolution world.
inline constexpr u32 kMaxLodLevels = 8;
inline constexpr u32 kMaxVerticalLodLevel = 2;
[[nodiscard]] constexpr u32 lodVerticalLevel(u32 level) noexcept
{
    return level < kMaxVerticalLodLevel ? level : kMaxVerticalLodLevel;
}
[[nodiscard]] constexpr i32 lodScale(u32 level) noexcept
{
    return 1 << level;
} // horizontal
[[nodiscard]] constexpr i32 lodVerticalScale(u32 level) noexcept
{
    return 1 << lodVerticalLevel(level);
} // vertical
// Chunk layers per column at a level (at least one; the top may be part air).
[[nodiscard]] constexpr i32 worldHeightChunks(u32 level) noexcept
{
    return (kWorldHeightChunks >> lodVerticalLevel(level)) > 0 ? (kWorldHeightChunks >> lodVerticalLevel(level)) : 1;
}

// Heightmap of one 32x32 column of chunks at a level: everything a chunk
// layer of the column needs, computed once and shared by all of them.
// Heights and depths are in cells (blocks at level 0).
struct TerrainColumn {
    std::array<i16, kChunkArea> height;    // top solid cell + 1, [x + z * 32]
    std::array<i16, kChunkArea> sandDepth; // surface cells that are sand (lowland), else 0
    i32 minHeight = 0;
    i32 maxHeight = 0;
    i32 dirtDepth = 4; // cells of dirt (incl. the surface cell) above stone
    u32 level = 0;
};

// Large rolling mountains and valleys with small hills on top; sand in low
// ground, grass over dirt over stone elsewhere. At level L > 0 each cell
// takes the surface height averaged over its footprint (each term of the
// heightmap is box-filtered exactly, so hills smaller than a cell fade out
// instead of aliasing into regular stripes), and a cell is solid when that
// surface covers at least half of it.
void terrainColumn(i32 chunkX, i32 chunkZ, TerrainColumn& out, u32 level = 0);

// Many chunks are a single block type; they need no storage and little or
// no meshing. Decided from the heightmap alone, before filling anything.
enum class ChunkContent : u8 { Mixed, Air, Stone };
[[nodiscard]] ChunkContent classifyTerrainChunk(const TerrainColumn& column, i32 chunkY) noexcept;

// Fills chunk layer `chunkY` of a column (at the column's level). Writes
// every voxel, so `chunk` may be constructed with Chunk::Uninitialized.
void fillTerrainChunk(Chunk& chunk, const TerrainColumn& column, i32 chunkY);

// Convenience: terrainColumn + fillTerrainChunk.
void fillTerrain(Chunk& chunk, i32 chunkX, i32 chunkY, i32 chunkZ, u32 level = 0);

// Each voxel is solid with probability `density`. Writes every voxel.
// density = 0.5 is the worst case for face count.
void fillRandom(Chunk& chunk, f64 density, u64 seed);

} // namespace engine::voxel
