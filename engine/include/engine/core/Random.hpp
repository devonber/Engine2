#pragma once

#include "engine/core/Types.hpp"

namespace engine {

// SplitMix64: tiny, fast, good-quality PRNG. Deterministic across compilers
// and platforms (unlike <random> distributions), which matters for world-gen.
class SplitMix64 {
public:
    constexpr explicit SplitMix64(u64 seed) noexcept : m_state(seed) {}

    constexpr u64 next() noexcept
    {
        u64 z = (m_state += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }

    // Uniform in [0, 1).
    constexpr f64 nextUnit() noexcept { return static_cast<f64>(next() >> 11) * 0x1.0p-53; }

private:
    u64 m_state;
};

} // namespace engine
