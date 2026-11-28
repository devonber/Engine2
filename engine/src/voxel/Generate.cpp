#include "engine/voxel/Generate.hpp"

#include "engine/core/Profile.hpp"
#include "engine/core/Random.hpp"
#include "engine/voxel/Blocks.hpp"

#include <algorithm>
#include <array>
#include <cmath>

#if defined(__AVX2__)
    #include <immintrin.h>
#endif

namespace engine::voxel {

namespace {
constexpr i32 kSandLevel = 30; // columns whose surface is at or below this are sand
} // namespace

void terrainColumn(i32 chunkX, i32 chunkZ, TerrainColumn& out, u32 level)
{
    ENGINE_PROFILE_FUNCTION();
    constexpr i32 n = static_cast<i32>(kChunkSize);
    const i32 scale = lodScale(level);
    const i32 baseX = chunkX * n * scale;
    const i32 baseZ = chunkZ * n * scale;
    const i32 centre = scale / 2; // sample cells at their centre (level 0: the block itself)

    // Every term is f(x) * g(z) or f(x + z), so per column of chunks the trig
    // is 2 x 32 + 2 x 32 + 63 calls instead of 5 per block column.
    std::array<f32, kChunkSize> sinX, cosZ, bigX, bigZ;
    std::array<f32, 2 * kChunkSize - 1> sinXZ;
    for (i32 i = 0; i < n; ++i) {
        const f32 wx = static_cast<f32>(baseX + i * scale + centre);
        const f32 wz = static_cast<f32>(baseZ + i * scale + centre);
        sinX[static_cast<usize>(i)] = std::sin(wx * 0.11f);
        cosZ[static_cast<usize>(i)] = std::cos(wz * 0.07f);
        bigX[static_cast<usize>(i)] = std::sin(wx * 0.0123f); // mountains/valleys, ~500 blocks across
        bigZ[static_cast<usize>(i)] = std::cos(wz * 0.0171f);
    }
    for (i32 i = 0; i < 2 * n - 1; ++i) {
        // wx + wz of cells x and z with x + z = i
        sinXZ[static_cast<usize>(i)] = std::sin(static_cast<f32>(baseX + baseZ + i * scale + 2 * centre) * 0.23f);
    }

    out.level = level;
    out.dirtDepth = std::max(1, 4 >> level);
    const i16 sandCells = static_cast<i16>(std::max(1, 3 >> level));
    out.minHeight = kWorldHeight;
    out.maxHeight = 0;
    for (u32 z = 0; z < kChunkSize; ++z) {
        for (u32 x = 0; x < kChunkSize; ++x) {
            const f32 h = 48.0f + 40.0f * bigX[x] * bigZ[z] + 6.0f * sinX[x] * cosZ[z] + 3.0f * sinXZ[x + z];
            const i32 blocks = std::clamp(static_cast<i32>(h), 1, kWorldHeight - 1);
            // Cells that are at least half covered are solid (level 0: blocks).
            const i32 cells = std::max(1, (blocks + centre) >> level);
            out.minHeight = std::min(out.minHeight, cells);
            out.maxHeight = std::max(out.maxHeight, cells);
            out.height[z * kChunkSize + x] = static_cast<i16>(cells);
            out.sandDepth[z * kChunkSize + x] =
                static_cast<i16>(blocks <= kSandLevel ? (level == 0 ? 3 : sandCells) : 0);
        }
    }
}

ChunkContent classifyTerrainChunk(const TerrainColumn& column, i32 chunkY) noexcept
{
    const i32 bottom = chunkY * static_cast<i32>(kChunkSize);
    const i32 top = bottom + static_cast<i32>(kChunkSize) - 1;
    if (bottom >= column.maxHeight) {
        return ChunkContent::Air; // entirely above the surface
    }
    if (top + column.dirtDepth < column.minHeight) {
        return ChunkContent::Stone; // entirely below the surface layers
    }
    return ChunkContent::Mixed;
}

void fillTerrain(Chunk& chunk, i32 chunkX, i32 chunkY, i32 chunkZ, u32 level)
{
    TerrainColumn column;
    terrainColumn(chunkX, chunkZ, column, level);
    fillTerrainChunk(chunk, column, chunkY);
}

void fillTerrainChunk(Chunk& chunk, const TerrainColumn& column, i32 chunkY)
{
    ENGINE_PROFILE_FUNCTION();
    constexpr i32 n = static_cast<i32>(kChunkSize);
    const auto& height = column.height;
    const auto& sandDepth = column.sandDepth;

    // One horizontal layer at a time. Layers are contiguous in memory (x
    // fastest, then z), so this is a linear write instead of a column walk
    // that jumps 2 KiB per step.
    BlockId* ENGINE_RESTRICT out = chunk.blocks().data();
    for (i32 ly = 0; ly < n; ++ly) {
        const i32 y = chunkY * n + ly; // cell y (world block y at level 0)
        BlockId* ENGINE_RESTRICT layer = out + static_cast<usize>(ly) * kChunkArea;
        if (y >= column.maxHeight) {
            std::fill_n(layer, kChunkArea, kAir);
            continue;
        }
        if (y + column.dirtDepth < column.minHeight) {
            std::fill_n(layer, kChunkArea, blocks::Stone); // deep underground: no surface layers
            continue;
        }
        // Block by depth below the surface (1 = surface block).
        const i16 y16 = static_cast<i16>(y);
        u32 i = 0;
#if defined(__AVX2__)
        // Explicit AVX2, 16 columns per step: MSVC does not auto-vectorise the
        // scalar loop below (GCC 13 only does with the bitwise form).
        const __m256i vy = _mm256_set1_epi16(y16);
        const __m256i v1 = _mm256_set1_epi16(1);
        const __m256i dirtLimit = _mm256_set1_epi16(static_cast<i16>(column.dirtDepth + 1));
        const __m256i stone = _mm256_set1_epi16(static_cast<i16>(blocks::Stone));
        const __m256i dirt = _mm256_set1_epi16(static_cast<i16>(blocks::Dirt));
        const __m256i grass = _mm256_set1_epi16(static_cast<i16>(blocks::Grass));
        const __m256i sand = _mm256_set1_epi16(static_cast<i16>(blocks::Sand));
        for (; i < kChunkArea; i += 16) {
            const __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&height[i]));
            const __m256i sd = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&sandDepth[i]));
            const __m256i depth = _mm256_sub_epi16(h, vy);
            __m256i id = stone;
            id = _mm256_blendv_epi8(id, dirt, _mm256_cmpgt_epi16(dirtLimit, depth)); // depth <= dirtDepth
            id = _mm256_blendv_epi8(id, grass, _mm256_cmpeq_epi16(depth, v1));       // depth == 1
            id = _mm256_blendv_epi8(sand, id, _mm256_cmpgt_epi16(depth, sd));        // depth <= sandDepth
            id = _mm256_andnot_si256(_mm256_cmpgt_epi16(v1, depth), id);             // depth <= 0: air
            static_assert(kAir == 0);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(layer + i), id);
        }
#endif
        // Scalar path (non-AVX2 builds). Bitwise masks, not `cond ? a : b`:
        // GCC 13 does not if-convert a chain of selects and leaves it scalar.
        const auto blend = [](u16 id, bool cond, u16 value) {
            const u16 m = static_cast<u16>(-static_cast<i16>(cond)); // all ones if cond
            return static_cast<u16>((id & ~m) | (value & m));
        };
        for (; i < kChunkArea; ++i) {
            const i16 depth = static_cast<i16>(height[i] - y16);
            u16 id = blocks::Stone;
            id = blend(id, depth <= column.dirtDepth, blocks::Dirt);
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
