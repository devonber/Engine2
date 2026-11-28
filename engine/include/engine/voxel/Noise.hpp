#pragma once

#include "engine/core/Platform.hpp"
#include "engine/core/Types.hpp"

#include <cmath>

#if defined(__AVX2__)
    #include <immintrin.h>
#endif

// 2D gradient noise (Perlin-style) for terrain.
//
// Positions are an integer block coordinate plus a small float offset, and
// each octave's lattice spacing is a power of two (2^shift blocks), so the
// lattice cell is found with integer shifts and the fraction is computed from
// small numbers only: the noise is exactly as precise a million blocks from
// the origin as at it. The offset carries sub-block positions (cell centres)
// and domain warping.
//
// gradientNoise() is the reference; gradientNoise8() computes 8 positions at
// once with AVX2 using the same operations in the same order (no FMA), so the
// two agree to float rounding.
namespace engine::voxel::noise {

[[nodiscard]] constexpr u32 hash(i32 x, i32 z, u32 seed) noexcept
{
    u32 h = (static_cast<u32>(x) * 0x8da6b343u) ^ (static_cast<u32>(z) * 0xd8163841u) ^ seed;
    h ^= h >> 15;
    h *= 0x2c1b3c6du;
    h ^= h >> 12;
    h *= 0x297a2d39u;
    h ^= h >> 15;
    return h;
}

// 8 gradient directions (axes and diagonals), chosen by the top 3 hash bits.
inline constexpr f32 kDiag = 0.70710678f;
inline constexpr f32 kGradX[8] = {1.0f, -1.0f, 0.0f, 0.0f, kDiag, -kDiag, kDiag, -kDiag};
inline constexpr f32 kGradZ[8] = {0.0f, 0.0f, 1.0f, -1.0f, kDiag, kDiag, -kDiag, -kDiag};
// Scales the result to about [-1, 1].
inline constexpr f32 kNoiseScale = 1.4142135f;

[[nodiscard]] inline f32 fade(f32 t) noexcept
{
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

// Noise with lattice spacing 2^shift blocks (shift 1..20) at block bx + dx, bz + dz.
[[nodiscard]] inline f32 gradientNoise(i32 bx, f32 dx, i32 bz, f32 dz, u32 shift, u32 seed) noexcept
{
    const i32 mask = (1 << shift) - 1;
    const f32 inv = 1.0f / static_cast<f32>(1 << shift); // exact: a power of two
    const f32 sx = (static_cast<f32>(bx & mask) + dx) * inv;
    const f32 sz = (static_cast<f32>(bz & mask) + dz) * inv;
    const f32 flx = std::floor(sx);
    const f32 flz = std::floor(sz);
    const i32 cx = (bx >> shift) + static_cast<i32>(flx);
    const i32 cz = (bz >> shift) + static_cast<i32>(flz);
    const f32 fx = sx - flx;
    const f32 fz = sz - flz;

    const auto dot = [](u32 h, f32 x, f32 z) {
        const u32 g = h >> 29;
        return kGradX[g] * x + kGradZ[g] * z;
    };
    const f32 d00 = dot(hash(cx, cz, seed), fx, fz);
    const f32 d10 = dot(hash(cx + 1, cz, seed), fx - 1.0f, fz);
    const f32 d01 = dot(hash(cx, cz + 1, seed), fx, fz - 1.0f);
    const f32 d11 = dot(hash(cx + 1, cz + 1, seed), fx - 1.0f, fz - 1.0f);
    const f32 u = fade(fx);
    const f32 v = fade(fz);
    const f32 a = d00 + u * (d10 - d00);
    const f32 b = d01 + u * (d11 - d01);
    return (a + v * (b - a)) * kNoiseScale;
}

#if defined(__AVX2__)
namespace detail {
[[nodiscard]] ENGINE_FORCEINLINE __m256i hash8(__m256i x, __m256i z, __m256i seed) noexcept
{
    __m256i h = _mm256_xor_si256(_mm256_mullo_epi32(x, _mm256_set1_epi32(static_cast<i32>(0x8da6b343u))),
                                 _mm256_mullo_epi32(z, _mm256_set1_epi32(static_cast<i32>(0xd8163841u))));
    h = _mm256_xor_si256(h, seed);
    h = _mm256_xor_si256(h, _mm256_srli_epi32(h, 15));
    h = _mm256_mullo_epi32(h, _mm256_set1_epi32(0x2c1b3c6d));
    h = _mm256_xor_si256(h, _mm256_srli_epi32(h, 12));
    h = _mm256_mullo_epi32(h, _mm256_set1_epi32(0x297a2d39));
    return _mm256_xor_si256(h, _mm256_srli_epi32(h, 15));
}

[[nodiscard]] ENGINE_FORCEINLINE __m256 dot8(__m256i h, __m256 x, __m256 z) noexcept
{
    const __m256i g = _mm256_srli_epi32(h, 29);
    const __m256 gx = _mm256_permutevar8x32_ps(_mm256_loadu_ps(kGradX), g);
    const __m256 gz = _mm256_permutevar8x32_ps(_mm256_loadu_ps(kGradZ), g);
    return _mm256_add_ps(_mm256_mul_ps(gx, x), _mm256_mul_ps(gz, z));
}

[[nodiscard]] ENGINE_FORCEINLINE __m256 fade8(__m256 t) noexcept
{
    const __m256 inner =
        _mm256_add_ps(_mm256_mul_ps(t, _mm256_sub_ps(_mm256_mul_ps(t, _mm256_set1_ps(6.0f)), _mm256_set1_ps(15.0f))),
                      _mm256_set1_ps(10.0f));
    return _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(t, t), t), inner);
}
} // namespace detail

// gradientNoise() for 8 positions.
[[nodiscard]] ENGINE_FORCEINLINE __m256 gradientNoise8(__m256i bx, __m256 dx, __m256i bz, __m256 dz, u32 shift,
                                                       u32 seed) noexcept
{
    using namespace detail;
    const __m256i mask = _mm256_set1_epi32((1 << shift) - 1);
    const __m256 inv = _mm256_set1_ps(1.0f / static_cast<f32>(1 << shift));
    const __m128i count = _mm_cvtsi32_si128(static_cast<int>(shift));
    const __m256 sx = _mm256_mul_ps(_mm256_add_ps(_mm256_cvtepi32_ps(_mm256_and_si256(bx, mask)), dx), inv);
    const __m256 sz = _mm256_mul_ps(_mm256_add_ps(_mm256_cvtepi32_ps(_mm256_and_si256(bz, mask)), dz), inv);
    const __m256 flx = _mm256_floor_ps(sx);
    const __m256 flz = _mm256_floor_ps(sz);
    const __m256i cx = _mm256_add_epi32(_mm256_sra_epi32(bx, count), _mm256_cvttps_epi32(flx));
    const __m256i cz = _mm256_add_epi32(_mm256_sra_epi32(bz, count), _mm256_cvttps_epi32(flz));
    const __m256 fx = _mm256_sub_ps(sx, flx);
    const __m256 fz = _mm256_sub_ps(sz, flz);

    const __m256i one = _mm256_set1_epi32(1);
    const __m256 onef = _mm256_set1_ps(1.0f);
    const __m256i vseed = _mm256_set1_epi32(static_cast<i32>(seed));
    const __m256i cx1 = _mm256_add_epi32(cx, one);
    const __m256i cz1 = _mm256_add_epi32(cz, one);
    const __m256 fx1 = _mm256_sub_ps(fx, onef);
    const __m256 fz1 = _mm256_sub_ps(fz, onef);
    const __m256 d00 = dot8(hash8(cx, cz, vseed), fx, fz);
    const __m256 d10 = dot8(hash8(cx1, cz, vseed), fx1, fz);
    const __m256 d01 = dot8(hash8(cx, cz1, vseed), fx, fz1);
    const __m256 d11 = dot8(hash8(cx1, cz1, vseed), fx1, fz1);
    const __m256 u = fade8(fx);
    const __m256 v = fade8(fz);
    const __m256 a = _mm256_add_ps(d00, _mm256_mul_ps(u, _mm256_sub_ps(d10, d00)));
    const __m256 b = _mm256_add_ps(d01, _mm256_mul_ps(u, _mm256_sub_ps(d11, d01)));
    return _mm256_mul_ps(_mm256_add_ps(a, _mm256_mul_ps(v, _mm256_sub_ps(b, a))), _mm256_set1_ps(kNoiseScale));
}
#endif

} // namespace engine::voxel::noise
