#include "engine/render/Swapchain.hpp"

#include "engine/render/VulkanContext.hpp"

#include <algorithm>

namespace engine::render {

namespace {

VkSurfaceFormatKHR chooseFormat(VkPhysicalDevice device, VkSurfaceKHR surface)
{
    u32 count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &count, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &count, formats.data());
    // sRGB swapchain: the hardware converts linear shader output to sRGB on write.
    for (const auto& f : formats) {
        if ((f.format == VK_FORMAT_B8G8R8A8_SRGB || f.format == VK_FORMAT_R8G8B8A8_SRGB) &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return f;
        }
    }
    return formats.front();
}

VkPresentModeKHR choosePresentMode(VkPhysicalDevice device, VkSurfaceKHR surface, PresentMode mode)
{
    if (mode == PresentMode::Vsync) {
        return VK_PRESENT_MODE_FIFO_KHR; // always supported
    }
    u32 count = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &count, nullptr);
    std::vector<VkPresentModeKHR> modes(count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &count, modes.data());
    for (VkPresentModeKHR wanted : {VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR}) {
        if (std::ranges::find(modes, wanted) != modes.end()) {
            return wanted;
        }
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

} // namespace

Swapchain::Swapchain(VulkanContext& ctx, VkExtent2D extent, PresentMode mode) : m_ctx(ctx), m_mode(mode)
{
    create(extent, VK_NULL_HANDLE);
}

Swapchain::~Swapchain()
{
    destroyResources();
    vkDestroySwapchainKHR(m_ctx.device(), m_swapchain, nullptr);
}

void Swapchain::recreate(VkExtent2D extent)
{
    destroyResources();
    VkSwapchainKHR old = m_swapchain;
    create(extent, old);
    vkDestroySwapchainKHR(m_ctx.device(), old, nullptr);
}

void Swapchain::create(VkExtent2D extent, VkSwapchainKHR old)
{
    VkDevice device = m_ctx.device();
    VkPhysicalDevice gpu = m_ctx.physicalDevice();
    VkSurfaceKHR surface = m_ctx.surface();

    VkSurfaceCapabilitiesKHR caps;
    VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(gpu, surface, &caps));

    const VkSurfaceFormatKHR surfaceFormat = chooseFormat(gpu, surface);
    const VkPresentModeKHR presentMode = choosePresentMode(gpu, surface, m_mode);

    if (caps.currentExtent.width != UINT32_MAX) {
        extent = caps.currentExtent; // the surface dictates the size
    } else {
        extent.width = std::clamp(extent.width, caps.minImageExtent.width, caps.maxImageExtent.width);
        extent.height = std::clamp(extent.height, caps.minImageExtent.height, caps.maxImageExtent.height);
    }

    // One more than the minimum so we never wait on the driver to release an image.
    u32 imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount != 0) {
        imageCount = std::min(imageCount, caps.maxImageCount);
    }

    const VkSwapchainCreateInfoKHR info{
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = surface,
        .minImageCount = imageCount,
        .imageFormat = surfaceFormat.format,
        .imageColorSpace = surfaceFormat.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        // TRANSFER_SRC so screenshots can copy out of the swapchain.
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .preTransform = caps.currentTransform,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = presentMode,
        .clipped = VK_TRUE,
        .oldSwapchain = old,
    };
    VK_CHECK(vkCreateSwapchainKHR(device, &info, nullptr, &m_swapchain));

    m_format = surfaceFormat.format;
    m_extent = extent;

    u32 count = 0;
    vkGetSwapchainImagesKHR(device, m_swapchain, &count, nullptr);
    m_images.resize(count);
    vkGetSwapchainImagesKHR(device, m_swapchain, &count, m_images.data());

    m_views.resize(count);
    m_renderFinished.resize(count);
    for (u32 i = 0; i < count; ++i) {
        const VkImageViewCreateInfo viewInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = m_images[i],
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = m_format,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
        };
        VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &m_views[i]));

        const VkSemaphoreCreateInfo semInfo{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VK_CHECK(vkCreateSemaphore(device, &semInfo, nullptr, &m_renderFinished[i]));
    }
}

void Swapchain::destroyResources()
{
    VkDevice device = m_ctx.device();
    for (VkImageView v : m_views) {
        vkDestroyImageView(device, v, nullptr);
    }
    for (VkSemaphore s : m_renderFinished) {
        vkDestroySemaphore(device, s, nullptr);
    }
    m_views.clear();
    m_renderFinished.clear();
    m_images.clear();
}

} // namespace engine::render
