#include "engine/core/Random.hpp"
#include "engine/voxel/Generate.hpp"
#include "engine/voxel/Noise.hpp"

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <set>
#include <vector>

using namespace engine;
using namespace engine::voxel;

TEST_CASE("Gradient noise: bounded, zero on the lattice, continuous, and the same far from the origin")
{
    SplitMix64 rng{7};
    f32 lo = 0.0f, hi = 0.0f;
    for (int i = 0; i < 20000; ++i) {
        const i32 x = static_cast<i32>(rng.next() % 2000000) - 1000000;
        const i32 z = static_cast<i32>(rng.next() % 2000000) - 1000000;
        const f32 n = noise::gradientNoise(x, 0.5f, z, 0.5f, 4, 123);
        lo = std::min(lo, n);
        hi = std::max(hi, n);
        // Lattice points (multiples of 16 here) are zeros of gradient noise.
        CHECK(noise::gradientNoise(x & ~15, 0.0f, z & ~15, 0.0f, 4, 123) == 0.0f);
        // A tenth of a block away it barely changes (lattice 16: slope < ~0.3 per block).
        CHECK(std::abs(noise::gradientNoise(x, 0.6f, z, 0.5f, 4, 123) - n) < 0.05f);
    }
    CHECK(lo < -0.6f);
    CHECK(hi > 0.6f);
    CHECK(lo >= -1.05f);
    CHECK(hi <= 1.05f);

    // Integer + offset positions keep sub-block precision anywhere: 4,000,000
    // blocks out a float position could not even tell 0.5 from 0.6 (floats
    // are 0.25 apart there), but the noise still changes smoothly.
    const i32 far = 4'000'000;
    int changed = 0;
    for (i32 x = 0; x < 256; ++x) {
        const f32 a = noise::gradientNoise(far + x, 0.5f, -far, 0.5f, 4, 9);
        const f32 b = noise::gradientNoise(far + x, 0.6f, -far, 0.5f, 4, 9);
        CHECK(std::abs(a - b) < 0.05f);
        changed += a != b ? 1 : 0;
    }
    CHECK(changed > 240);
}

#if defined(__AVX2__)
TEST_CASE("Gradient noise: the AVX2 version matches the scalar reference")
{
    SplitMix64 rng{11};
    for (int i = 0; i < 4000; ++i) {
        alignas(32) i32 bx[8];
        alignas(32) f32 dx[8];
        alignas(32) f32 out[8];
        const i32 bz = static_cast<i32>(rng.next() % 2000000) - 1000000;
        const f32 dz = static_cast<f32>(rng.next() % 1000) * 0.37f - 180.0f; // warped offsets can be large
        for (int l = 0; l < 8; ++l) {
            bx[l] = static_cast<i32>(rng.next() % 2000000) - 1000000;
            dx[l] = static_cast<f32>(rng.next() % 1000) * 0.29f - 140.0f;
        }
        const u32 shift = 1 + static_cast<u32>(rng.next() % 11);
        _mm256_store_ps(out, noise::gradientNoise8(_mm256_load_si256(reinterpret_cast<const __m256i*>(bx)),
                                                   _mm256_load_ps(dx), _mm256_set1_epi32(bz), _mm256_set1_ps(dz), shift,
                                                   77));
        for (int l = 0; l < 8; ++l) {
            CAPTURE(shift);
            CHECK(std::abs(out[l] - noise::gradientNoise(bx[l], dx[l], bz, dz, shift, 77)) < 1e-5f);
        }
    }
}
#endif

TEST_CASE("Terrain columns match per-sample evaluation at every level")
{
    for (u32 level = 0; level < kMaxLodLevels; ++level) {
        CAPTURE(level);
        const i32 scale = lodScale(level);
        const i32 vscale = lodVerticalScale(level);
        const f32 centre = static_cast<f32>(scale - 1) * 0.5f;
        for (const auto& [cx, cz] : {std::pair{0, 0}, std::pair{-7, 13}, std::pair{30000, -25000}}) {
            TerrainColumn column;
            terrainColumn(cx, cz, column, level);
            int differ = 0;
            for (i32 z = 0; z < 32; ++z) {
                for (i32 x = 0; x < 32; ++x) {
                    const TerrainSample t =
                        terrainSample((cx * 32 + x) * scale, centre, (cz * 32 + z) * scale, centre, level);
                    const i32 blocks = std::clamp(static_cast<i32>(t.height), 1, kWorldHeight - 1);
                    const i32 cells = std::max(1, (blocks + vscale / 2) / vscale);
                    const i32 got = column.height[static_cast<usize>(z * 32 + x)];
                    CHECK(std::abs(got - cells) <= 1); // rounding of a value right at a boundary
                    differ += got != cells ? 1 : 0;
                }
            }
            CHECK(differ <= 2);
        }
    }
}

TEST_CASE("Terrain has lowlands, hills and mountains, and does not repeat")
{
    int sand = 0, rock = 0, high = 0, low = 0;
    std::set<std::vector<i16>> seen;
    for (i32 cz = -40; cz < 40; cz += 4) {
        for (i32 cx = -40; cx < 40; cx += 4) {
            TerrainColumn column;
            terrainColumn(cx, cz, column);
            for (usize i = 0; i < kChunkArea; i += 17) {
                sand += column.sandDepth[i] > 0 ? 1 : 0;
                rock += column.soilDepth[i] == 0 ? 1 : 0;
                high += column.height[i] > 95 ? 1 : 0;
                low += column.height[i] < 30 ? 1 : 0;
            }
            // The previous generator repeated every few hundred blocks; no
            // two of these 400 columns (128 blocks apart) may be identical.
            CHECK(seen.emplace(column.height.begin(), column.height.end()).second);
        }
    }
    CHECK(sand > 0);
    CHECK(rock > 0);
    CHECK(high > 0);
    CHECK(low > 0);
}
