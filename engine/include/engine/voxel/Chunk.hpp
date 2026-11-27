#pragma once

#include "engine/core/Assert.hpp"
#include "engine/core/Types.hpp"

#include <array>
#include <span>
#include <type_traits>

namespace engine::voxel {

using BlockId = u16;
inline constexpr BlockId kAir = 0;

inline constexpr u32 kChunkSizeLog2 = 5;
inline constexpr u32 kChunkSize = 1u << kChunkSizeLog2; // 32
inline constexpr u32 kChunkArea = kChunkSize * kChunkSize;
inline constexpr u32 kChunkVolume = kChunkArea * kChunkSize;

// Memory layout: x varies fastest, then z, then y.
//   index = (y << 10) | (z << 5) | x
// A horizontal 32x32 slice (one y level) is contiguous, which suits
// heightmap-style terrain generation and per-layer operations.
[[nodiscard]] constexpr u32 voxelIndex(u32 x, u32 y, u32 z) noexcept
{
    return (y << (2 * kChunkSizeLog2)) | (z << kChunkSizeLog2) | x;
}

[[nodiscard]] constexpr bool inChunk(i32 x, i32 y, i32 z) noexcept
{
    constexpr i32 n = static_cast<i32>(kChunkSize);
    return x >= 0 && x < n && y >= 0 && y < n && z >= 0 && z < n;
}

// Dense 32^3 block storage: 64 KiB, flat array, no indirection.
// Allocate on the heap (std::make_unique) - it is too big for a casual stack frame.
// Palette compression for memory comes later; keep the hot path simple first.
class Chunk {
public:
    // Tag for skipping the zero fill when every voxel is about to be written
    // anyway (e.g. by a generator): saves a 64 KiB memset per chunk.
    struct Uninitialized {};

    Chunk() noexcept { m_blocks.fill(kAir); }
    explicit Chunk(Uninitialized) noexcept {}

    [[nodiscard]] BlockId get(u32 x, u32 y, u32 z) const noexcept
    {
        ENGINE_ASSERT(x < kChunkSize && y < kChunkSize && z < kChunkSize);
        return m_blocks[voxelIndex(x, y, z)];
    }

    void set(u32 x, u32 y, u32 z, BlockId id) noexcept
    {
        ENGINE_ASSERT(x < kChunkSize && y < kChunkSize && z < kChunkSize);
        m_blocks[voxelIndex(x, y, z)] = id;
    }

    void fill(BlockId id) noexcept { m_blocks.fill(id); }

    [[nodiscard]] std::span<const BlockId, kChunkVolume> blocks() const noexcept { return m_blocks; }
    [[nodiscard]] std::span<BlockId, kChunkVolume> blocks() noexcept { return m_blocks; }

private:
    std::array<BlockId, kChunkVolume> m_blocks; // not value-initialised: see the constructors
};

static_assert(sizeof(Chunk) == kChunkVolume * sizeof(BlockId));
static_assert(std::is_trivially_destructible_v<Chunk>);

} // namespace engine::voxel
