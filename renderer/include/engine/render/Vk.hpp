#pragma once

// Single include point for Vulkan. Everything goes through volk (function
// pointers loaded straight from the driver) and VMA (GPU memory allocation).

#include <volk.h>
// volk must come first: it defines VK_NO_PROTOTYPES and pulls in vulkan.h.
#include <cstdio>
#include <cstdlib>
#include <vk_mem_alloc.h>

namespace engine::render::detail {

[[noreturn]] inline void vkFail(VkResult result, const char* expr, const char* file, int line) noexcept
{
    std::fprintf(stderr, "Vulkan call failed (VkResult %d): %s\n  at %s:%d\n", static_cast<int>(result), expr, file,
                 line);
    std::fflush(stderr);
    std::abort();
}

[[noreturn]] inline void fatal(const char* message) noexcept
{
    std::fprintf(stderr, "Fatal: %s\n", message);
    std::fflush(stderr);
    std::abort();
}

} // namespace engine::render::detail

// Vulkan errors here are programming or driver errors with no sane recovery,
// so fail loudly at the call site. Checked in every build type.
#define VK_CHECK(expr)                                                                                                 \
    do {                                                                                                               \
        const VkResult vkCheckResult_ = (expr);                                                                        \
        if (vkCheckResult_ != VK_SUCCESS) [[unlikely]] {                                                               \
            ::engine::render::detail::vkFail(vkCheckResult_, #expr, __FILE__, __LINE__);                               \
        }                                                                                                              \
    } while (false)
