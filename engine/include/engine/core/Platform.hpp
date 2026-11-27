#pragma once

// Compiler-specific helpers. Keep this tiny: prefer standard attributes.

#if defined(_MSC_VER) && !defined(__clang__)
    #define ENGINE_FORCEINLINE __forceinline
    #define ENGINE_NOINLINE __declspec(noinline)
    #define ENGINE_DEBUGBREAK() __debugbreak()
    #define ENGINE_RESTRICT __restrict
#else
    #define ENGINE_FORCEINLINE inline __attribute__((always_inline))
    #define ENGINE_NOINLINE __attribute__((noinline))
    #define ENGINE_RESTRICT __restrict__
    #if defined(_MSC_VER)
        #define ENGINE_DEBUGBREAK() __debugbreak()
    #else
        #define ENGINE_DEBUGBREAK() __builtin_trap()
    #endif
#endif
