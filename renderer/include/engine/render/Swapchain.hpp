#pragma once

#include "engine/core/Types.hpp"
#include "engine/render/Vk.hpp"

#include <vector>

namespace engine::render {

class VulkanContext;

enum class PresentMode {
    Vsync,   // FIFO: capped to display refresh, lowest power
    Uncapped // MAILBOX if available (no tearing), else IMMEDIATE (tearing)
};

class Swapchain {
public:
    Swapchain(VulkanContext& ctx, VkExtent2D extent, PresentMode mode);
    ~Swapchain();
    Swapchain(const Swapchain&) = delete;
    Swapchain& operator=(const Swapchain&) = delete;

    // Rebuild for a new window size. Caller must ensure the GPU is idle.
    void recreate(VkExtent2D extent);

    [[nodiscard]] VkSwapchainKHR handle() const noexcept { return m_swapchain; }
    [[nodiscard]] VkFormat format() const noexcept { return m_format; }
    [[nodiscard]] VkExtent2D extent() const noexcept { return m_extent; }
    [[nodiscard]] u32 imageCount() const noexcept { return static_cast<u32>(m_images.size()); }
    [[nodiscard]] VkImage image(u32 i) const noexcept { return m_images[i]; }
    [[nodiscard]] VkImageView view(u32 i) const noexcept { return m_views[i]; }

    // Signalled when rendering to image i finishes; presentation waits on it.
    // One per image (not per frame in flight) so a semaphore is never reused
    // while the presentation engine may still hold it.
    [[nodiscard]] VkSemaphore renderFinished(u32 i) const noexcept { return m_renderFinished[i]; }

private:
    void create(VkExtent2D extent, VkSwapchainKHR old);
    void destroyResources();

    VulkanContext& m_ctx;
    PresentMode m_mode;
    VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;
    VkFormat m_format = VK_FORMAT_UNDEFINED;
    VkExtent2D m_extent{};
    std::vector<VkImage> m_images;
    std::vector<VkImageView> m_views;
    std::vector<VkSemaphore> m_renderFinished;
};

} // namespace engine::render
