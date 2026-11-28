#pragma once

#include "engine/voxel/Chunk.hpp"

#include <array>

namespace engine::voxel {

// Simple procedural terrain for tests, benchmarks and the sandbox.

// The world is columns of this many chunks stacked vertically (y = 0 at the
// bottom of chunk layer 0). Below the world counts as solid, above it as air.
inline constexpr i32 kWorldHeightChunks = 4; // 128 blocks
inline constexpr i32 kWorldHeight = kWorldHeightChunks * static_cast<i32>(kChunkSize);

// Heightmap of one 32x32 column of chunks: everything a chunk layer of the
// column needs, computed once and shared by all of them.
struct TerrainColumn {
    std::array<i16, kChunkArea> height;    // world y of the top solid block + 1, [x + z * 32]
    std::array<i16, kChunkArea> sandDepth; // 3 in lowland (sand), 0 elsewhere
    i32 minHeight = 0;
    i32 maxHeight = 0;
};

// Large rolling mountains and valleys with small hills on top; sand in low
// ground, grass over dirt over stone elsewhere.
void terrainColumn(i32 chunkX, i32 chunkZ, TerrainColumn& out);

// Many chunks are a single block type; they need no storage and little or
// no meshing. Decided from the heightmap alone, before filling anything.
enum class ChunkContent : u8 { Mixed, Air, Stone };
[[nodiscard]] ChunkContent classifyTerrainChunk(const TerrainColumn& column, i32 chunkY) noexcept;

// Fills chunk layer `chunkY` of a column. Writes every voxel, so `chunk` may
// be constructed with Chunk::Uninitialized.
void fillTerrainChunk(Chunk& chunk, const TerrainColumn& column, i32 chunkY);

// Convenience: terrainColumn + fillTerrainChunk.
void fillTerrain(Chunk& chunk, i32 chunkX, i32 chunkY, i32 chunkZ);

// Each voxel is solid with probability `density`. Writes every voxel.
// density = 0.5 is the worst case for face count.
void fillRandom(Chunk& chunk, f64 density, u64 seed);

} // namespace engine::voxel
