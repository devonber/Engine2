#include "engine/core/Bits.hpp"
#include "engine/core/Random.hpp"

#include <doctest/doctest.h>

using namespace engine;

TEST_CASE("transpose32 matches a bit-by-bit transpose")
{
    SplitMix64 rng{5};
    for (int trial = 0; trial < 200; ++trial) {
        std::array<u32, 32> a;
        for (u32& row : a) {
            row = static_cast<u32>(rng.next());
        }
        std::array<u32, 32> expected{};
        for (u32 r = 0; r < 32; ++r) {
            for (u32 c = 0; c < 32; ++c) {
                expected[c] |= ((a[r] >> c) & 1u) << r;
            }
        }
        transpose32(a);
        REQUIRE(a == expected);
    }
}

TEST_CASE("transpose32 is usable at compile time and is its own inverse")
{
    constexpr auto twice = [] {
        std::array<u32, 32> a{};
        for (u32 i = 0; i < 32; ++i) {
            a[i] = i * 2654435761u;
        }
        auto b = a;
        transpose32(b);
        transpose32(b);
        return a == b;
    }();
    static_assert(twice);
}
