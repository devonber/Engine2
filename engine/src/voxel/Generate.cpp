#include "engine/voxel/Generate.hpp"

#include "engine/core/Profile.hpp"
#include "engine/core/Random.hpp"
#include "engine/voxel/Blocks.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace engine::voxel {

namespace {
constexpr u32 kSandLevel = 12;
} // namespace

void fillTerrain(Chunk& chunk, i32 chunkX, i32 chunkZ)
{
    ENGINE_PROFILE_FUNCTION();
    constexpr i32 n = static_cast<i32>(kChunkSize);
    const i32 baseX = chunkX * n;
    const i32 baseZ = chunkZ * n;

    // The height function is sin(x) * cos(z) + sin(x + z): its trig terms
    // depend on x, z or x + z alone, so 32 + 32 + 63 calls replace 3 per column.
    std::array<f32, kChunkSize> sinX, cosZ;
    std::array<f32, 2 * kChunkSize - 1> sinXZ;
    for (i32 i = 0; i < n; ++i) {
        sinX[static_cast<usize>(i)] = std::sin(static_cast<f32>(baseX + i) * 0.11f);
        cosZ[static_cast<usize>(i)] = std::cos(static_cast<f32>(baseZ + i) * 0.07f);
    }
    for (i32 i = 0; i < 2 * n - 1; ++i) {
        // (wx + wz) as in the per-column formula; both are exact small integers.
        sinXZ[static_cast<usize>(i)] = std::sin(static_cast<f32>(baseX + baseZ + i) * 0.23f);
    }

    // Pass 1: per column, the surface height and how deep sand goes (3 blocks
    // in lowland, beaches/valleys; 0 elsewhere). 16-bit like the block ids, so
    // pass 2 vectorises without widening or narrowing.
    std::array<i16, kChunkArea> height;
    std::array<i16, kChunkArea> sandDepth;
    i32 minHeight = n;
    i32 maxHeight = 0;
    for (u32 z = 0; z < kChunkSize; ++z) {
        for (u32 x = 0; x < kChunkSize; ++x) {
            const f32 h = 16.0f + 6.0f * sinX[x] * cosZ[z] + 3.0f * sinXZ[x + z];
            const i32 hi = std::clamp(static_cast<i32>(h), 1, n - 1);
            minHeight = std::min(minHeight, hi);
            maxHeight = std::max(maxHeight, hi);
            height[z * kChunkSize + x] = static_cast<i16>(hi);
            sandDepth[z * kChunkSize + x] = static_cast<i16>(hi <= static_cast<i32>(kSandLevel) ? 3 : 0);
        }
    }

    // Pass 2: fill one horizontal layer at a time. Layers are contiguous in
    // memory (x fastest, then z), so this is a linear write instead of a
    // column walk that jumps 2 KiB per step.
    BlockId* ENGINE_RESTRICT out = chunk.blocks().data();
    for (i32 y = 0; y < n; ++y) {
        BlockId* ENGINE_RESTRICT layer = out + static_cast<usize>(y) * kChunkArea;
        if (y >= maxHeight) {
            std::fill_n(layer, kChunkArea, kAir);
            continue;
        }
        if (y + 4 < minHeight) {
            std::fill_n(layer, kChunkArea, blocks::Stone); // deep underground: no surface layers
            continue;
        }
        // Block by depth below the surface (1 = surface block), with bitwise
        // masks instead of `cond ? a : b`: GCC 13 does not if-convert a chain
        // of selects, sees "control flow in loop" and runs it scalar (5x slower).
        const i16 y16 = static_cast<i16>(y);
        const auto blend = [](u16 id, bool cond, u16 value) {
            const u16 m = static_cast<u16>(-static_cast<i16>(cond)); // all ones if cond
            return static_cast<u16>((id & ~m) | (value & m));
        };
        for (u32 i = 0; i < kChunkArea; ++i) {
            const i16 depth = static_cast<i16>(height[i] - y16);
            u16 id = blocks::Stone;
            id = blend(id, depth <= 4, blocks::Dirt);
            id = blend(id, depth == 1, blocks::Grass);
            id = blend(id, depth <= sandDepth[i], blocks::Sand);
            id = blend(id, depth <= 0, kAir);
            layer[i] = id;
        }
    }
}

void fillRandom(Chunk& chunk, f64 density, u64 seed)
{
    ENGINE_PROFILE_FUNCTION();
    SplitMix64 rng{seed};
    for (BlockId& b : chunk.blocks()) {
        b = rng.nextUnit() < density ? blocks::Stone : blocks::Air;
    }
}

} // namespace engine::voxel
