#pragma once

#include "engine/voxel/Chunk.hpp"

#include <array>
#include <compare>
#include <vector>

namespace engine::voxel {

enum class Face : u8 { PosX = 0, NegX, PosY, NegY, PosZ, NegZ };
inline constexpr u32 kFaceCount = 6;

// One visible rectangle of block faces, packed into 8 bytes. The GPU expands
// each one into a quad in the vertex shader ("vertex pulling"), so a quad
// costs 8 bytes instead of 4 vertices x ~20 bytes.
//
//   bits  0..5   x       position of the quad's minimum voxel in the chunk
//   bits  6..11  y
//   bits 12..17  z
//   bits 18..20  face
//   bits 21..25  width-1   extent along the face's U axis (see faceAxes)
//   bits 26..30  height-1  extent along the face's V axis
//   bits 32..47  block id
//
// Unit quads (width = height = 1) leave bits 21..30 zero.
struct PackedQuad {
    u64 bits = 0;

    [[nodiscard]] static constexpr PackedQuad make(u32 x, u32 y, u32 z, Face face, BlockId block, u32 width = 1,
                                                   u32 height = 1) noexcept
    {
        return PackedQuad{static_cast<u64>(x) | (static_cast<u64>(y) << 6) | (static_cast<u64>(z) << 12) |
                          (static_cast<u64>(face) << 18) | (static_cast<u64>(width - 1) << 21) |
                          (static_cast<u64>(height - 1) << 26) | (static_cast<u64>(block) << 32)};
    }

    [[nodiscard]] constexpr u32 x() const noexcept { return static_cast<u32>(bits & 0x3F); }
    [[nodiscard]] constexpr u32 y() const noexcept { return static_cast<u32>((bits >> 6) & 0x3F); }
    [[nodiscard]] constexpr u32 z() const noexcept { return static_cast<u32>((bits >> 12) & 0x3F); }
    [[nodiscard]] constexpr Face face() const noexcept { return static_cast<Face>((bits >> 18) & 0x7); }
    [[nodiscard]] constexpr u32 width() const noexcept { return static_cast<u32>((bits >> 21) & 0x1F) + 1; }
    [[nodiscard]] constexpr u32 height() const noexcept { return static_cast<u32>((bits >> 26) & 0x1F) + 1; }
    [[nodiscard]] constexpr BlockId block() const noexcept { return static_cast<BlockId>(bits >> 32); }

    constexpr auto operator<=>(const PackedQuad&) const = default;
};

static_assert(sizeof(PackedQuad) == 8);

// The in-plane axes of each face, as axis indices (0 = x, 1 = y, 2 = z).
// U x V points out of the block. A quad's width runs along U and its height
// along V. The vertex shader (renderer/shaders/voxel.vert) uses the same table.
struct FaceAxes {
    u8 normal, u, v;
};
inline constexpr std::array<FaceAxes, kFaceCount> kFaceAxes{{
    {0, 1, 2}, // +X: U = y, V = z
    {0, 2, 1}, // -X: U = z, V = y
    {1, 2, 0}, // +Y: U = z, V = x
    {1, 0, 2}, // -Y: U = x, V = z
    {2, 0, 1}, // +Z: U = x, V = y
    {2, 1, 0}, // -Z: U = y, V = x
}};

// The six chunks around a chunk, indexed by Face (chunks[PosX] is the chunk
// at x + 1). nullptr counts as all air.
struct ChunkNeighbours {
    std::array<const Chunk*, kFaceCount> chunks{};
};

// Solidity of the voxel layer just outside each face of a chunk: a 32x32
// bitmask per face, rows[face][row] bit `col`. This is all the fast meshers
// need from the neighbours: 6 x 128 bytes instead of 6 x 64 KiB.
//   +X / -X: row = y, col = z   (neighbour voxel at x = 0 / x = 31)
//   +Y / -Y: row = z, col = x   (neighbour voxel at y = 0 / y = 31)
//   +Z / -Z: row = y, col = x   (neighbour voxel at z = 0 / z = 31)
struct BoundaryMasks {
    std::array<std::array<u32, kChunkSize>, kFaceCount> rows{};

    void setSolid(Face face) noexcept { rows[static_cast<usize>(face)].fill(~0u); }
};

[[nodiscard]] BoundaryMasks buildBoundaryMasks(const ChunkNeighbours& neighbours);

// All meshers emit faces of solid blocks whose neighbour is air, including
// across chunk borders (using the neighbours). `out` is cleared and
// refilled; reuse the same vector between calls so it allocates once.

// Reference implementation: visits every voxel and checks 6 neighbours. One
// quad per face. Easy to read; the ground truth in tests.
void meshChunkNaive(const Chunk& chunk, const ChunkNeighbours& neighbours, std::vector<PackedQuad>& out);

// Builds 32-bit occupancy columns along each axis and finds visible faces
// for 32 voxels at once with shift/and-not. One quad per face.
void meshChunkBinary(const Chunk& chunk, const BoundaryMasks& neighbours, std::vector<PackedQuad>& out);

// Same visibility as meshChunkBinary, then merges adjacent coplanar faces of
// the same block into rectangles (greedy meshing). Far fewer quads.
void meshChunkGreedy(const Chunk& chunk, const BoundaryMasks& neighbours, std::vector<PackedQuad>& out);

} // namespace engine::voxel
