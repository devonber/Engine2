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
//   bits 48..55  ambient occlusion: 2 bits per corner, corner c = cu + 2 * cv
//                ((cu, cv) = 0/1 along U/V); 3 = open, 0 = fully enclosed
//
// Unit quads (width = height = 1) leave bits 21..30 zero.
struct PackedQuad {
    static constexpr u32 kNoOcclusion = 0xFF; // all four corners open

    u64 bits = 0;

    [[nodiscard]] static constexpr PackedQuad make(u32 x, u32 y, u32 z, Face face, BlockId block, u32 width = 1,
                                                   u32 height = 1, u32 ao = kNoOcclusion) noexcept
    {
        return PackedQuad{static_cast<u64>(x) | (static_cast<u64>(y) << 6) | (static_cast<u64>(z) << 12) |
                          (static_cast<u64>(face) << 18) | (static_cast<u64>(width - 1) << 21) |
                          (static_cast<u64>(height - 1) << 26) | (static_cast<u64>(block) << 32) |
                          (static_cast<u64>(ao & 0xFFu) << 48)};
    }

    [[nodiscard]] constexpr u32 x() const noexcept { return static_cast<u32>(bits & 0x3F); }
    [[nodiscard]] constexpr u32 y() const noexcept { return static_cast<u32>((bits >> 6) & 0x3F); }
    [[nodiscard]] constexpr u32 z() const noexcept { return static_cast<u32>((bits >> 12) & 0x3F); }
    [[nodiscard]] constexpr Face face() const noexcept { return static_cast<Face>((bits >> 18) & 0x7); }
    [[nodiscard]] constexpr u32 width() const noexcept { return static_cast<u32>((bits >> 21) & 0x1F) + 1; }
    [[nodiscard]] constexpr u32 height() const noexcept { return static_cast<u32>((bits >> 26) & 0x1F) + 1; }
    [[nodiscard]] constexpr BlockId block() const noexcept { return static_cast<BlockId>((bits >> 32) & 0xFFFF); }
    [[nodiscard]] constexpr u32 aoBits() const noexcept { return static_cast<u32>((bits >> 48) & 0xFF); }
    [[nodiscard]] constexpr u32 ao(u32 corner) const noexcept { return (aoBits() >> (2 * corner)) & 3u; }

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

// The 26 chunks around a chunk (and the chunk itself in the middle), indexed
// by neighbourIndex(dx, dy, dz). nullptr counts as all air.
[[nodiscard]] constexpr usize neighbourIndex(i32 dx, i32 dy, i32 dz) noexcept
{
    return static_cast<usize>((dx + 1) + 3 * (dy + 1) + 9 * (dz + 1));
}
inline constexpr usize kNeighbourhoodSize = 27;

struct ChunkNeighbourhood {
    std::array<const Chunk*, kNeighbourhoodSize> chunks{};

    void set(i32 dx, i32 dy, i32 dz, const Chunk* chunk) noexcept { chunks[neighbourIndex(dx, dy, dz)] = chunk; }
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

// Solidity of the whole one-voxel shell around a chunk: the 6 face layers
// (which decide which faces are visible) plus the 12 edge lines and 8 corner
// voxels (which ambient occlusion at the chunk's edges needs as well).
//   edges[c][i] bit k: the voxel at position k along axis c, on the edge whose
//                      other two axes a1 = (c+1)%3, a2 = (c+2)%3 are at
//                      -1 or 32: i = (a1 at 32 ? 1 : 0) | (a2 at 32 ? 2 : 0)
//   corners bit (x at 32) | (y at 32) << 1 | (z at 32) << 2
struct NeighbourShell {
    BoundaryMasks faces;
    std::array<std::array<u32, 4>, 3> edges{};
    u32 corners = 0;

    // Everything solid (e.g. "below the world").
    void setAllSolid() noexcept;
};

// Reference: reads the neighbouring chunks themselves.
[[nodiscard]] NeighbourShell buildNeighbourShell(const ChunkNeighbourhood& neighbours);

// Solidity of a chunk's OWN six outer layers, indexed by face: rows[PosX] is
// its x = 31 layer, rows[NegX] its x = 0 layer, and so on, with the same
// (row, col) layout as BoundaryMasks. Compute it once, right after generating
// the chunk while its data is in cache (~2 us); meshing a neighbour then
// copies 128 bytes instead of reading the whole neighbour from memory.
struct OuterLayers {
    std::array<std::array<u32, kChunkSize>, kFaceCount> rows{};
};

[[nodiscard]] OuterLayers computeOuterLayers(const Chunk& chunk);

// Fast path: the shell from the 26 neighbours' outer layers, indexed by
// neighbourIndex (nullptr = air). Copies 6 x 128 bytes and gathers a few
// hundred bits; never touches neighbour chunk memory.
[[nodiscard]] NeighbourShell
shellFromOuterLayers(const std::array<const OuterLayers*, kNeighbourhoodSize>& neighbours) noexcept;

// Outer layers of a uniformly solid / empty chunk.
[[nodiscard]] const OuterLayers& solidOuterLayers() noexcept;

// All meshers emit faces of solid blocks whose neighbour is air, including
// across chunk borders (using the neighbours). `out` is cleared and
// refilled; reuse the same vector between calls so it allocates once.
//
// With `ambientOcclusion`, each face corner gets an occlusion level from the
// three voxels touching it in front of the face (two sides and the diagonal):
// 3 - count, or 0 when both sides are solid. Greedy meshing merges only faces
// whose four levels match, which keeps the shading exactly as unmerged.

// Reference implementation: visits every voxel and checks its neighbours
// directly. One quad per face. Easy to read; the ground truth in tests.
void meshChunkNaive(const Chunk& chunk, const ChunkNeighbourhood& neighbours, std::vector<PackedQuad>& out,
                    bool ambientOcclusion = true);

// Builds 32-bit occupancy columns along each axis and finds visible faces
// for 32 voxels at once with shift/and-not. One quad per face.
void meshChunkBinary(const Chunk& chunk, const NeighbourShell& shell, std::vector<PackedQuad>& out,
                     bool ambientOcclusion = true);

// Same visibility as meshChunkBinary, then merges adjacent coplanar faces of
// the same block (and the same occlusion) into rectangles. Far fewer quads.
void meshChunkGreedy(const Chunk& chunk, const NeighbourShell& shell, std::vector<PackedQuad>& out,
                     bool ambientOcclusion = true);

} // namespace engine::voxel
