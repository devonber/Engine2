#pragma once

#include "engine/core/Types.hpp"
#include "engine/render/Vk.hpp"

#include <utility>

namespace engine::platform {
class Window;
}

namespace engine::render {

struct ContextDesc {
    const char* appName = "Engine2";
    bool validation = false; // Khronos validation layer + debug messenger
};

// Owns the core Vulkan objects: instance, surface, device, one graphics+present
// queue and the VMA allocator. Requires Vulkan 1.3 (dynamic rendering,
// synchronization2), which every desktop GPU driver from ~2022 on provides.
class VulkanContext {
public:
    VulkanContext(const platform::Window& window, const ContextDesc& desc);
    ~VulkanContext();
    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    [[nodiscard]] VkInstance instance() const noexcept { return m_instance; }
    [[nodiscard]] VkSurfaceKHR surface() const noexcept { return m_surface; }
    [[nodiscard]] VkPhysicalDevice physicalDevice() const noexcept { return m_physicalDevice; }
    [[nodiscard]] VkDevice device() const noexcept { return m_device; }
    [[nodiscard]] VkQueue queue() const noexcept { return m_queue; }
    [[nodiscard]] u32 queueFamily() const noexcept { return m_queueFamily; }
    [[nodiscard]] VmaAllocator allocator() const noexcept { return m_allocator; }
    [[nodiscard]] const VkPhysicalDeviceProperties& properties() const noexcept { return m_properties; }
    [[nodiscard]] bool samplerAnisotropy() const noexcept { return m_samplerAnisotropy; }

    // Record and run commands synchronously. For uploads at load time only;
    // it stalls the GPU.
    template <typename Fn>
    void immediateSubmit(Fn&& record)
    {
        VkCommandBuffer cmd = beginImmediate();
        std::forward<Fn>(record)(cmd);
        endImmediate(cmd);
    }

private:
    void createInstance(const ContextDesc& desc);
    void pickPhysicalDevice();
    void createDevice();
    void createAllocator();
    [[nodiscard]] VkCommandBuffer beginImmediate();
    void endImmediate(VkCommandBuffer cmd);

    VkInstance m_instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT m_messenger = VK_NULL_HANDLE;
    VkSurfaceKHR m_surface = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties m_properties{};
    VkDevice m_device = VK_NULL_HANDLE;
    VkQueue m_queue = VK_NULL_HANDLE;
    u32 m_queueFamily = 0;
    VmaAllocator m_allocator = nullptr;
    VkCommandPool m_immediatePool = VK_NULL_HANDLE;
    bool m_samplerAnisotropy = false;
};

// GPU buffer + its VMA allocation. `mapped` is non-null for host-visible buffers.
struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    void* mapped = nullptr;
    VkDeviceSize size = 0;
};

enum class MemoryKind {
    GpuOnly,  // device-local; filled by copies
    Upload,   // host-visible, persistently mapped, written sequentially by the CPU
    Readback, // host-visible, cached, read by the CPU
};

[[nodiscard]] Buffer createBuffer(VmaAllocator allocator, VkDeviceSize size, VkBufferUsageFlags usage, MemoryKind kind);
void destroyBuffer(VmaAllocator allocator, Buffer& buffer);

// Creates a GpuOnly buffer and fills it through a temporary staging buffer.
[[nodiscard]] Buffer createBufferWithData(VulkanContext& ctx, const void* data, VkDeviceSize size,
                                          VkBufferUsageFlags usage);

} // namespace engine::render
