#include "engine/voxel/Generate.hpp"

#include "engine/core/Profile.hpp"
#include "engine/core/Random.hpp"

#include <algorithm>
#include <cmath>

namespace engine::voxel {

namespace {
constexpr BlockId kStone = 1;
constexpr BlockId kDirt = 2;
constexpr BlockId kGrass = 3;
} // namespace

void fillTerrain(Chunk& chunk, i32 chunkX, i32 chunkZ)
{
    ENGINE_PROFILE_FUNCTION();
    chunk.fill(kAir);
    constexpr i32 n = static_cast<i32>(kChunkSize);

    for (u32 z = 0; z < kChunkSize; ++z) {
        for (u32 x = 0; x < kChunkSize; ++x) {
            const f32 wx = static_cast<f32>(chunkX * n + static_cast<i32>(x));
            const f32 wz = static_cast<f32>(chunkZ * n + static_cast<i32>(z));
            const f32 h =
                16.0f + 6.0f * std::sin(wx * 0.11f) * std::cos(wz * 0.07f) + 3.0f * std::sin((wx + wz) * 0.23f);
            const u32 height = static_cast<u32>(std::clamp(static_cast<i32>(h), 1, n - 1));

            for (u32 y = 0; y < height; ++y) {
                const BlockId id = (y + 1 == height) ? kGrass : (y + 4 >= height) ? kDirt : kStone;
                chunk.set(x, y, z, id);
            }
        }
    }
}

void fillRandom(Chunk& chunk, f64 density, u64 seed)
{
    ENGINE_PROFILE_FUNCTION();
    SplitMix64 rng{seed};
    for (BlockId& b : chunk.blocks()) {
        b = rng.nextUnit() < density ? kStone : kAir;
    }
}

} // namespace engine::voxel
