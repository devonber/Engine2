#pragma once

#include "engine/core/Types.hpp"

#include <array>

namespace engine {

namespace detail {

// One round of transpose32: for every row pair (k, k + J) in each 2J-row
// block, swap the high J bits of row k with the low J bits of row k + J.
template <u32 J>
constexpr void transposeRound(std::array<u32, 32>& a) noexcept
{
    constexpr u32 mask = J == 16  ? 0x0000FFFFu
                         : J == 8 ? 0x00FF00FFu
                         : J == 4 ? 0x0F0F0F0Fu
                         : J == 2 ? 0x33333333u
                                  : 0x55555555u;
    for (u32 block = 0; block < 32; block += 2 * J) {
        for (u32 k = block; k < block + J; ++k) {
            const u32 t = ((a[k] >> J) ^ a[k + J]) & mask;
            a[k + J] ^= t;
            a[k] ^= t << J;
        }
    }
}

} // namespace detail

// Transposes a 32x32 bit matrix in place: bit c of row r moves to bit r of
// row c (bit 0 = least significant). Five rounds of block swaps (16x16, 8x8,
// ... 1x1), 80 masked xor-swaps in total. Each round has fixed trip counts
// so compilers fully unroll it, and the J >= 8 rounds vectorise (8 rows per
// AVX2 register).
constexpr void transpose32(std::array<u32, 32>& a) noexcept
{
    detail::transposeRound<16>(a);
    detail::transposeRound<8>(a);
    detail::transposeRound<4>(a);
    detail::transposeRound<2>(a);
    detail::transposeRound<1>(a);
}

} // namespace engine
