#pragma once

#include "engine/core/Platform.hpp"

#include <cstdio>

// ENGINE_ASSERT is active only in Debug (CMake defines ENGINE_ENABLE_ASSERTS=1).
// In optimised builds it compiles to nothing, so it is free to use in hot loops.
#ifndef ENGINE_ENABLE_ASSERTS
    #define ENGINE_ENABLE_ASSERTS 0
#endif

namespace engine::detail {

ENGINE_NOINLINE inline void assertFailed(const char* expr, const char* file, int line) noexcept
{
    std::fprintf(stderr, "Assertion failed: %s\n  at %s:%d\n", expr, file, line);
    std::fflush(stderr);
}

} // namespace engine::detail

#if ENGINE_ENABLE_ASSERTS
    #define ENGINE_ASSERT(cond)                                                                                        \
        do {                                                                                                           \
            if (!(cond)) [[unlikely]] {                                                                                \
                ::engine::detail::assertFailed(#cond, __FILE__, __LINE__);                                             \
                ENGINE_DEBUGBREAK();                                                                                   \
            }                                                                                                          \
        } while (false)
#else
    #define ENGINE_ASSERT(cond) static_cast<void>(0)
#endif
