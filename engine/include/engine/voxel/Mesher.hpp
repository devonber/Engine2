#pragma once

#include "engine/voxel/Chunk.hpp"

#include <compare>
#include <vector>

namespace engine::voxel {

enum class Face : u8 { PosX = 0, NegX, PosY, NegY, PosZ, NegZ };

// One visible block face, packed into 8 bytes. The GPU expands each one into
// a quad in the vertex shader ("vertex pulling"), so a face costs 8 bytes
// instead of 4 vertices x ~20 bytes.
//
//   bits  0..5   x      (6 bits: room for 0..32 inclusive, for greedy meshing later)
//   bits  6..11  y
//   bits 12..17  z
//   bits 18..20  face
//   bits 32..47  block id
struct PackedQuad {
    u64 bits = 0;

    [[nodiscard]] static constexpr PackedQuad make(u32 x, u32 y, u32 z, Face face, BlockId block) noexcept
    {
        return PackedQuad{static_cast<u64>(x) | (static_cast<u64>(y) << 6) | (static_cast<u64>(z) << 12) |
                          (static_cast<u64>(face) << 18) | (static_cast<u64>(block) << 32)};
    }

    [[nodiscard]] constexpr u32 x() const noexcept { return static_cast<u32>(bits & 0x3F); }
    [[nodiscard]] constexpr u32 y() const noexcept { return static_cast<u32>((bits >> 6) & 0x3F); }
    [[nodiscard]] constexpr u32 z() const noexcept { return static_cast<u32>((bits >> 12) & 0x3F); }
    [[nodiscard]] constexpr Face face() const noexcept { return static_cast<Face>((bits >> 18) & 0x7); }
    [[nodiscard]] constexpr BlockId block() const noexcept { return static_cast<BlockId>(bits >> 32); }

    constexpr auto operator<=>(const PackedQuad&) const = default;
};

static_assert(sizeof(PackedQuad) == 8);

// Both meshers emit one quad per solid face whose neighbour is air.
// Faces on the chunk border are always emitted (for now neighbours outside
// the chunk count as air; cross-chunk culling comes later).
// `out` is cleared and refilled. Reuse the same vector between calls so its
// memory is allocated once.

// Reference implementation: visits every voxel and checks 6 neighbours.
// Easy to read, used as ground truth in tests.
void meshChunkNaive(const Chunk& chunk, std::vector<PackedQuad>& out);

// Fast path: builds 32-bit occupancy columns along each axis, then finds
// visible faces for 32 voxels at once with shift/and-not.
void meshChunkBinary(const Chunk& chunk, std::vector<PackedQuad>& out);

} // namespace engine::voxel
