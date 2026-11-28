#include "engine/voxel/Generate.hpp"

#include "engine/core/Profile.hpp"
#include "engine/core/Random.hpp"
#include "engine/voxel/Blocks.hpp"
#include "engine/voxel/Noise.hpp"

#include <algorithm>
#include <array>
#include <cmath>

#if defined(__AVX2__)
    #include <immintrin.h>
#endif

namespace engine::voxel {

namespace {

constexpr i32 kSandLevel = 34; // columns whose surface is at or below this are sand
constexpr i32 kRockLevel = 96; // mountain tops above this are rock (see kRockMountain)

// Seeds of the noise fields (any distinct values).
enum Seed : u32 {
    kWarpX = 0x1001,
    kWarpZ = 0x1002,
    kContinent0 = 0x2001,
    kContinent1 = 0x2002,
    kHills = 0x3000, // + octave
    kMountainMask = 0x4001,
    kRidges = 0x5000, // + octave
};

// Shape of the height function (blocks), shared by the scalar and AVX2 code:
//   height = base + continent * c + hills * h + mountains * smooth(mask) * ridges
// then compressed above kSoftCap, so peaks round off below the world's top.
constexpr f32 kBase = 42.0f;
constexpr f32 kContinentAmp = 26.0f;
constexpr f32 kMountainAmp = 64.0f;
// Hill octaves in blocks (lattice 256, 128, 64, 32, 16): visible relief at every scale.
constexpr f32 kHillAmps[5] = {14.0f, 8.0f, 4.5f, 2.5f, 1.3f};
constexpr f32 kMaskNoise = 0.9f; // mountain mask = (noise * this + c * kMaskContinent + kMaskBias) * kMaskGain
constexpr f32 kMaskContinent = 0.5f;
constexpr f32 kMaskBias = 0.1f;
constexpr f32 kMaskGain = 2.2f;
constexpr f32 kSoftCap = 104.0f;
constexpr f32 kAboveCap = 0.45f; // slope above the cap
// Bare rock: where mountains (not just high ground) make the surface, above
// kRockLevel blocks, with the hills breaking up the edge.
constexpr f32 kRockMountain = 26.0f; // blocks of mountain needed
constexpr f32 kRockSlope = 0.9f;     // blocks of rise per block: steeper is rock

constexpr u32 kHillOctaves = 5; // lattice 256, 128, 64, 32, 16 blocks
constexpr u32 kHillTop = 8;
constexpr u32 kRidgeOctaves = 3; // lattice 512, 256, 128 blocks
constexpr u32 kRidgeTop = 9;

// Each field and octave is offset by a different whole number of blocks, so
// their lattices (where gradient noise is always 0) do not line up.
constexpr i32 offsetX(u32 seed) noexcept
{
    return static_cast<i32>((seed * 7919u) & 4095u);
}
constexpr i32 offsetZ(u32 seed) noexcept
{
    return static_cast<i32>((seed * 6271u) & 4095u);
}

f32 noiseAt(i32 bx, f32 dx, i32 bz, f32 dz, u32 shift, u32 seed) noexcept
{
    return noise::gradientNoise(bx + offsetX(seed), dx, bz + offsetZ(seed), dz, shift, seed);
}

// Average of (1 - |n|)^2 over the plane: what a ridged octave tends to when
// it is filtered out.
f32 ridgeMean() noexcept
{
    static const f32 mean = [] {
        f64 sum = 0.0;
        int count = 0;
        for (i32 z = 0; z < 1024; z += 3) {
            for (i32 x = 0; x < 1024; x += 3) {
                const f32 n = noise::gradientNoise(x, 0.5f, z, 0.5f, 4, kRidges);
                sum += static_cast<f64>((1.0f - std::abs(n)) * (1.0f - std::abs(n)));
                ++count;
            }
        }
        return static_cast<f32>(sum / count);
    }();
    return mean;
}

// Terrain shape at a level of detail: the weight of every noise octave. At
// coarse levels octaves whose lattice is not much bigger than a cell are
// faded out (by the box filter's sinc response), so a cell follows the
// average surface instead of aliasing hills smaller than itself into
// patterns; ridged octaves fade to their mean rather than to zero.
struct Shape {
    f32 warp;         // domain warp, blocks
    f32 continent[2]; // lattice 2048, 1024
    f32 mask;
    f32 hills[kHillOctaves];
    f32 ridge[kRidgeOctaves];     // weight of (1 - |n|)^2
    f32 ridgeBias[kRidgeOctaves]; // weight x mean of the faded part
};

Shape shapeFor(u32 level) noexcept
{
    const auto keep = [level](u32 shift) {
        if (level == 0) {
            return 1.0f; // blocks are sampled exactly
        }
        const f32 x = 3.14159265f * static_cast<f32>(lodScale(level)) / static_cast<f32>(1 << shift);
        return x >= 3.14159265f ? 0.0f : std::sin(x) / x;
    };
    Shape s{};
    s.warp = 64.0f * keep(10);
    s.continent[0] = keep(11) / 1.4f;
    s.continent[1] = 0.4f * keep(10) / 1.4f;
    s.mask = kMaskNoise * keep(10);
    for (u32 i = 0; i < kHillOctaves; ++i) {
        s.hills[i] = kHillAmps[i] * keep(kHillTop - i);
    }
    f32 amp = 1.0f / 1.75f; // 1 + 1/2 + 1/4
    for (u32 i = 0; i < kRidgeOctaves; ++i, amp *= 0.5f) {
        const f32 k = keep(kRidgeTop - i);
        s.ridge[i] = amp * k;
        s.ridgeBias[i] = amp * (1.0f - k) * ridgeMean();
    }
    return s;
}

f32 smooth01(f32 x) noexcept
{
    return x * x * (3.0f - 2.0f * x);
}

// The terrain at one position (scalar reference; terrainRow8 does 8 at once).
TerrainSample sampleTerrain(const Shape& s, i32 bx, f32 dx, i32 bz, f32 dz) noexcept
{
    // Domain warp: bends everything below, so features are not lined up on the noise lattice.
    const f32 wx = s.warp * noiseAt(bx, dx, bz, dz, 10, kWarpX);
    const f32 wz = s.warp * noiseAt(bx, dx, bz, dz, 10, kWarpZ);
    dx += wx;
    dz += wz;
    // Continents: broad lowlands and highlands, about -1..1.
    const f32 c = s.continent[0] * noiseAt(bx, dx, bz, dz, 11, kContinent0) +
                  s.continent[1] * noiseAt(bx, dx, bz, dz, 10, kContinent1);
    // Rolling hills: fractal sum, in blocks.
    f32 h = 0.0f;
    for (u32 i = 0; i < kHillOctaves; ++i) {
        h += s.hills[i] * noiseAt(bx, dx, bz, dz, kHillTop - i, kHills + i);
    }
    // Mountains: ridged noise (sharp crests where the noise crosses zero),
    // only where the mask (more likely on high ground) allows.
    const f32 mm = std::clamp(
        (s.mask * noiseAt(bx, dx, bz, dz, 10, kMountainMask) + kMaskContinent * c + kMaskBias) * kMaskGain, 0.0f, 1.0f);
    f32 r = 0.0f;
    for (u32 i = 0; i < kRidgeOctaves; ++i) {
        const f32 n = 1.0f - std::abs(noiseAt(bx, dx, bz, dz, kRidgeTop - i, kRidges + i));
        r = r + s.ridge[i] * (n * n); // same order as the AVX2 code
        r = r + s.ridgeBias[i];
    }
    const f32 mountain = kMountainAmp * smooth01(mm) * r;
    const f32 height = kBase + kContinentAmp * c + h + mountain;
    return {height - std::max(height - kSoftCap, 0.0f) * (1.0f - kAboveCap), h, mountain};
}

#if defined(__AVX2__)
// sampleTerrain() for 8 positions (bx + i * stepX, bz + i * stepZ), same
// operations in the same order. Outputs are optional (null: not needed).
void sampleTerrain8(const Shape& s, i32 bx, i32 bz, i32 stepX, i32 stepZ, f32 d, f32* height, f32* hills,
                    f32* mountains) noexcept
{
    const __m256i lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i x = _mm256_add_epi32(_mm256_set1_epi32(bx), _mm256_mullo_epi32(lane, _mm256_set1_epi32(stepX)));
    const __m256i z = _mm256_add_epi32(_mm256_set1_epi32(bz), _mm256_mullo_epi32(lane, _mm256_set1_epi32(stepZ)));
    const auto noise8 = [&](__m256 px, __m256 pz, u32 shift, u32 seed) {
        return noise::gradientNoise8(_mm256_add_epi32(x, _mm256_set1_epi32(offsetX(seed))), px,
                                     _mm256_add_epi32(z, _mm256_set1_epi32(offsetZ(seed))), pz, shift, seed);
    };
    __m256 fx = _mm256_set1_ps(d);
    __m256 fz = _mm256_set1_ps(d);
    const __m256 wx = _mm256_mul_ps(_mm256_set1_ps(s.warp), noise8(fx, fz, 10, kWarpX));
    const __m256 wz = _mm256_mul_ps(_mm256_set1_ps(s.warp), noise8(fx, fz, 10, kWarpZ));
    fx = _mm256_add_ps(fx, wx);
    fz = _mm256_add_ps(fz, wz);
    const __m256 c = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(s.continent[0]), noise8(fx, fz, 11, kContinent0)),
                                   _mm256_mul_ps(_mm256_set1_ps(s.continent[1]), noise8(fx, fz, 10, kContinent1)));
    __m256 h = _mm256_setzero_ps();
    for (u32 i = 0; i < kHillOctaves; ++i) {
        h = _mm256_add_ps(h, _mm256_mul_ps(_mm256_set1_ps(s.hills[i]), noise8(fx, fz, kHillTop - i, kHills + i)));
    }
    const __m256 zero = _mm256_setzero_ps();
    const __m256 one = _mm256_set1_ps(1.0f);
    __m256 mm = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(s.mask), noise8(fx, fz, 10, kMountainMask)),
                              _mm256_mul_ps(_mm256_set1_ps(kMaskContinent), c));
    mm = _mm256_mul_ps(_mm256_add_ps(mm, _mm256_set1_ps(kMaskBias)), _mm256_set1_ps(kMaskGain));
    mm = _mm256_min_ps(_mm256_max_ps(mm, zero), one);
    const __m256 absMask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    __m256 r = _mm256_setzero_ps();
    for (u32 i = 0; i < kRidgeOctaves; ++i) {
        const __m256 n = _mm256_sub_ps(one, _mm256_and_ps(noise8(fx, fz, kRidgeTop - i, kRidges + i), absMask));
        r = _mm256_add_ps(_mm256_add_ps(r, _mm256_mul_ps(_mm256_set1_ps(s.ridge[i]), _mm256_mul_ps(n, n))),
                          _mm256_set1_ps(s.ridgeBias[i]));
    }
    const __m256 sm = _mm256_mul_ps(_mm256_mul_ps(mm, mm),
                                    _mm256_sub_ps(_mm256_set1_ps(3.0f), _mm256_mul_ps(_mm256_set1_ps(2.0f), mm)));
    const __m256 mountain = _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(kMountainAmp), sm), r);
    __m256 height8 = _mm256_add_ps(_mm256_set1_ps(kBase), _mm256_mul_ps(_mm256_set1_ps(kContinentAmp), c));
    height8 = _mm256_add_ps(height8, h);
    height8 = _mm256_add_ps(height8, mountain);
    const __m256 over = _mm256_max_ps(_mm256_sub_ps(height8, _mm256_set1_ps(kSoftCap)), zero);
    height8 = _mm256_sub_ps(height8, _mm256_mul_ps(over, _mm256_set1_ps(1.0f - kAboveCap)));
    _mm256_storeu_ps(height, height8);
    if (hills != nullptr) {
        _mm256_storeu_ps(hills, h);
    }
    if (mountains != nullptr) {
        _mm256_storeu_ps(mountains, mountain);
    }
}
#endif

} // namespace

TerrainSample terrainSample(i32 bx, f32 dx, i32 bz, f32 dz, u32 level) noexcept
{
    return sampleTerrain(shapeFor(level), bx, dx, bz, dz);
}

void terrainColumn(i32 chunkX, i32 chunkZ, TerrainColumn& out, u32 level)
{
    ENGINE_PROFILE_FUNCTION();
    constexpr i32 n = static_cast<i32>(kChunkSize);
    const i32 scale = lodScale(level);
    const i32 baseX = chunkX * n * scale;
    const i32 baseZ = chunkZ * n * scale;
    // Sample cells at their centre: blocks b .. b + scale - 1 (level 0: the block itself).
    const f32 centre = static_cast<f32>(scale - 1) * 0.5f;
    const u32 vlevel = lodVerticalLevel(level);
    const i32 vscale = lodVerticalScale(level);
    const Shape shape = shapeFor(level);

    // Surface heights (blocks, before rounding) on a 33 x 33 grid: the
    // column's cells plus one more row and column, for the slope (bare rock
    // on steep ground). Computing the extra ones here rather than borrowing
    // from the neighbouring columns keeps every column self-contained, and
    // both columns at a border agree, since a cell's value depends only on
    // its position.
    constexpr i32 g = n + 1;
    std::array<f32, static_cast<usize>(g * g)> heights;
    std::array<f32, kChunkArea> hills;
    std::array<f32, kChunkArea> mountains;
    const auto scalarAt = [&](i32 x, i32 z) {
        const TerrainSample t = sampleTerrain(shape, baseX + x * scale, centre, baseZ + z * scale, centre);
        heights[static_cast<usize>(z * g + x)] = t.height;
        if (x < n && z < n) {
            hills[static_cast<usize>(z * n + x)] = t.hills;
            mountains[static_cast<usize>(z * n + x)] = t.mountain;
        }
    };
#if defined(__AVX2__)
    for (i32 z = 0; z <= n; ++z) { // rows along x (row n: heights only)
        for (i32 x = 0; x < n; x += 8) {
            const bool inside = z < n;
            sampleTerrain8(shape, baseX + x * scale, baseZ + z * scale, scale, 0, centre,
                           &heights[static_cast<usize>(z * g + x)],
                           inside ? &hills[static_cast<usize>(z * n + x)] : nullptr,
                           inside ? &mountains[static_cast<usize>(z * n + x)] : nullptr);
        }
    }
    for (i32 z = 0; z < n; z += 8) { // the extra column, 8 cells along z at a time
        std::array<f32, 8> column;
        sampleTerrain8(shape, baseX + n * scale, baseZ + z * scale, 0, scale, centre, column.data(), nullptr, nullptr);
        for (i32 k = 0; k < 8; ++k) {
            heights[static_cast<usize>((z + k) * g + n)] = column[static_cast<usize>(k)];
        }
    }
#else
    for (i32 z = 0; z <= n; ++z) {
        for (i32 x = 0; x <= n; ++x) {
            if (x < n || z < n) {
                scalarAt(x, z);
            }
        }
    }
#endif
    static_cast<void>(scalarAt);

    out.level = level;
    // Layer depths in cells, rounded up so coarse levels show what level 0
    // shows: sand 3 blocks deep covers the terraces of lowland terrain; if it
    // were rounded down (1 cell at level 1), 2-cell steps would expose dirt
    // as specks all over distant sand.
    const auto cellsCovering = [vlevel, vscale](i32 blocks) { return (blocks + vscale - 1) >> vlevel; };
    out.dirtDepth = cellsCovering(4);
    const i16 soilCells = static_cast<i16>(out.dirtDepth);
    const i16 sandCells = static_cast<i16>(cellsCovering(3));
    out.minHeight = kWorldHeight;
    out.maxHeight = 0;
    const f32 perBlock = 1.0f / static_cast<f32>(scale);
    for (usize i = 0; i < kChunkArea; ++i) {
        const usize x = i % kChunkSize;
        const usize z = i / kChunkSize;
        const f32 here = heights[z * g + x];
        const i32 blocks = std::clamp(static_cast<i32>(here), 1, kWorldHeight - 1);
        // Steepness: height change per block towards +x and +z.
        const f32 slope =
            std::max(std::abs(heights[z * g + x + 1] - here), std::abs(heights[(z + 1) * g + x] - here)) * perBlock;
        // Cells that are at least half covered are solid (level 0: blocks).
        const i32 cells = std::max(1, (blocks + vscale / 2) >> vlevel);
        out.minHeight = std::min(out.minHeight, cells);
        out.maxHeight = std::max(out.maxHeight, cells);
        out.height[i] = static_cast<i16>(cells);
        const bool sand = blocks <= kSandLevel;
        // Bare rock on steep ground (cliffs, mountainsides) and on the high
        // mountain tops; the hills break up the edge.
        const bool rock =
            slope + 0.03f * hills[i] > kRockSlope ||
            (mountains[i] > kRockMountain && static_cast<f32>(blocks) + 0.6f * hills[i] > static_cast<f32>(kRockLevel));
        out.sandDepth[i] = static_cast<i16>(sand ? sandCells : 0);
        out.soilDepth[i] = static_cast<i16>(rock ? 0 : soilCells);
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
    const auto& soilDepth = column.soilDepth;

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
        const __m256i stone = _mm256_set1_epi16(static_cast<i16>(blocks::Stone));
        const __m256i dirt = _mm256_set1_epi16(static_cast<i16>(blocks::Dirt));
        const __m256i grass = _mm256_set1_epi16(static_cast<i16>(blocks::Grass));
        const __m256i sand = _mm256_set1_epi16(static_cast<i16>(blocks::Sand));
        for (; i < kChunkArea; i += 16) {
            const __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&height[i]));
            const __m256i sd = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&sandDepth[i]));
            const __m256i soil = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&soilDepth[i]));
            const __m256i depth = _mm256_sub_epi16(h, vy);
            const __m256i inSoil = _mm256_cmpgt_epi16(_mm256_add_epi16(soil, v1), depth); // depth <= soilDepth
            __m256i id = stone;
            id = _mm256_blendv_epi8(id, dirt, inSoil);
            id = _mm256_blendv_epi8(id, grass, _mm256_and_si256(inSoil, _mm256_cmpeq_epi16(depth, v1))); // top of soil
            id = _mm256_blendv_epi8(sand, id, _mm256_cmpgt_epi16(depth, sd)); // depth <= sandDepth
            id = _mm256_andnot_si256(_mm256_cmpgt_epi16(v1, depth), id);      // depth <= 0: air
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
            const bool inSoil = depth <= soilDepth[i];
            id = blend(id, inSoil, blocks::Dirt);
            id = blend(id, inSoil && depth == 1, blocks::Grass);
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
