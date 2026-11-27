#include "engine/render/VulkanContext.hpp"

#include "engine/platform/Window.hpp"

#include <cstring>
#include <vector>

namespace engine::render {

namespace {

constexpr const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                             VkDebugUtilsMessageTypeFlagsEXT /*types*/,
                                             const VkDebugUtilsMessengerCallbackDataEXT* data, void* /*user*/)
{
    const char* level = (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0 ? "error" : "warning";
    std::fprintf(stderr, "[vulkan %s] %s\n\n", level, data->pMessage);
    return VK_FALSE;
}

bool hasLayer(const char* name)
{
    u32 count = 0;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    std::vector<VkLayerProperties> layers(count);
    vkEnumerateInstanceLayerProperties(&count, layers.data());
    for (const auto& l : layers) {
        if (std::strcmp(l.layerName, name) == 0) {
            return true;
        }
    }
    return false;
}

bool hasDeviceExtension(VkPhysicalDevice device, const char* name)
{
    u32 count = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> exts(count);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &count, exts.data());
    for (const auto& e : exts) {
        if (std::strcmp(e.extensionName, name) == 0) {
            return true;
        }
    }
    return false;
}

// Features the renderer relies on. Checked during device selection and then
// enabled on the device.
struct RequiredFeatures {
    VkPhysicalDeviceVulkan13Features v13{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceFeatures2 core{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &v13};

    RequiredFeatures() = default;
    RequiredFeatures(const RequiredFeatures&) = delete; // pNext points into *this
    RequiredFeatures& operator=(const RequiredFeatures&) = delete;

    void setRequired()
    {
        v13.dynamicRendering = VK_TRUE;
        v13.synchronization2 = VK_TRUE;
        core.features.multiDrawIndirect = VK_TRUE;         // one draw call for all chunks
        core.features.drawIndirectFirstInstance = VK_TRUE; // firstInstance selects the chunk
    }

    [[nodiscard]] bool supportedBy(VkPhysicalDevice device)
    {
        vkGetPhysicalDeviceFeatures2(device, &core);
        return v13.dynamicRendering && v13.synchronization2 && core.features.multiDrawIndirect &&
               core.features.drawIndirectFirstInstance;
    }
};

} // namespace

VulkanContext::VulkanContext(const platform::Window& window, const ContextDesc& desc)
{
    createInstance(desc);
    m_surface = window.createSurface(m_instance);
    pickPhysicalDevice();
    createDevice();
    createAllocator();

    const VkCommandPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = m_queueFamily,
    };
    VK_CHECK(vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_immediatePool));
}

VulkanContext::~VulkanContext()
{
    vkDeviceWaitIdle(m_device);
    vkDestroyCommandPool(m_device, m_immediatePool, nullptr);
    vmaDestroyAllocator(m_allocator);
    vkDestroyDevice(m_device, nullptr);
    vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
    if (m_messenger != VK_NULL_HANDLE) {
        vkDestroyDebugUtilsMessengerEXT(m_instance, m_messenger, nullptr);
    }
    vkDestroyInstance(m_instance, nullptr);
}

void VulkanContext::createInstance(const ContextDesc& desc)
{
    u32 loaderVersion = 0;
    vkEnumerateInstanceVersion(&loaderVersion);
    if (loaderVersion < VK_API_VERSION_1_3) {
        detail::fatal("Vulkan 1.3 is required. Update your GPU driver.");
    }

    std::vector<const char*> extensions;
    for (const char* ext : platform::Window::requiredInstanceExtensions()) {
        extensions.push_back(ext);
    }

    std::vector<const char*> layers;
    bool validation = desc.validation;
    if (validation && !hasLayer(kValidationLayer)) {
        std::fprintf(stderr, "Validation requested but %s is not installed (install the Vulkan SDK).\n",
                     kValidationLayer);
        validation = false;
    }
    if (validation) {
        layers.push_back(kValidationLayer);
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    const VkApplicationInfo appInfo{
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = desc.appName,
        .applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0),
        .pEngineName = "Engine2",
        .engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0),
        .apiVersion = VK_API_VERSION_1_3,
    };

    const VkDebugUtilsMessengerCreateInfoEXT messengerInfo{
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = debugCallback,
    };

    const VkInstanceCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = validation ? &messengerInfo : nullptr, // also catches errors in create/destroy
        .pApplicationInfo = &appInfo,
        .enabledLayerCount = static_cast<u32>(layers.size()),
        .ppEnabledLayerNames = layers.data(),
        .enabledExtensionCount = static_cast<u32>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data(),
    };
    VK_CHECK(vkCreateInstance(&info, nullptr, &m_instance));
    volkLoadInstanceOnly(m_instance);

    if (validation) {
        VK_CHECK(vkCreateDebugUtilsMessengerEXT(m_instance, &messengerInfo, nullptr, &m_messenger));
        std::fprintf(stderr, "Vulkan validation enabled\n");
    }
}

void VulkanContext::pickPhysicalDevice()
{
    u32 count = 0;
    vkEnumeratePhysicalDevices(m_instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(m_instance, &count, devices.data());

    int bestScore = -1;
    for (VkPhysicalDevice device : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(device, &props);

        RequiredFeatures features;
        if (props.apiVersion < VK_API_VERSION_1_3 || !features.supportedBy(device) ||
            !hasDeviceExtension(device, VK_KHR_SWAPCHAIN_EXTENSION_NAME) || props.limits.timestampPeriod == 0.0f) {
            continue;
        }

        // One queue family that does graphics and presents: true on all desktop GPUs.
        u32 familyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &familyCount, nullptr);
        std::vector<VkQueueFamilyProperties> families(familyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &familyCount, families.data());
        for (u32 i = 0; i < familyCount; ++i) {
            VkBool32 present = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, m_surface, &present);
            const bool graphics = (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
            if (!graphics || present != VK_TRUE || families[i].timestampValidBits == 0) {
                continue;
            }
            const int score = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU     ? 3
                              : props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 2
                                                                                           : 1;
            if (score > bestScore) {
                bestScore = score;
                m_physicalDevice = device;
                m_queueFamily = i;
                m_properties = props;
            }
            break;
        }
    }

    if (m_physicalDevice == VK_NULL_HANDLE) {
        detail::fatal("No GPU supports Vulkan 1.3 with dynamic rendering, multi-draw indirect and presentation.");
    }
    std::fprintf(stderr, "GPU: %s (Vulkan %u.%u.%u)\n", m_properties.deviceName,
                 VK_API_VERSION_MAJOR(m_properties.apiVersion), VK_API_VERSION_MINOR(m_properties.apiVersion),
                 VK_API_VERSION_PATCH(m_properties.apiVersion));
}

void VulkanContext::createDevice()
{
    const f32 priority = 1.0f;
    const VkDeviceQueueCreateInfo queueInfo{
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = m_queueFamily,
        .queueCount = 1,
        .pQueuePriorities = &priority,
    };

    VkPhysicalDeviceFeatures available;
    vkGetPhysicalDeviceFeatures(m_physicalDevice, &available);
    m_samplerAnisotropy = available.samplerAnisotropy == VK_TRUE; // optional; universal on desktop

    RequiredFeatures features;
    features.setRequired();
    features.core.features.samplerAnisotropy = available.samplerAnisotropy;
    const char* extensions[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    const VkDeviceCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &features.core,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queueInfo,
        .enabledExtensionCount = 1,
        .ppEnabledExtensionNames = extensions,
    };
    VK_CHECK(vkCreateDevice(m_physicalDevice, &info, nullptr, &m_device));
    // Device-level function pointers straight from the driver: no loader
    // dispatch on hot calls like vkCmdDraw*.
    volkLoadDevice(m_device);
    vkGetDeviceQueue(m_device, m_queueFamily, 0, &m_queue);
}

void VulkanContext::createAllocator()
{
    const VmaVulkanFunctions functions{
        .vkGetInstanceProcAddr = vkGetInstanceProcAddr,
        .vkGetDeviceProcAddr = vkGetDeviceProcAddr,
    };
    const VmaAllocatorCreateInfo info{
        .physicalDevice = m_physicalDevice,
        .device = m_device,
        .pVulkanFunctions = &functions,
        .instance = m_instance,
        .vulkanApiVersion = VK_API_VERSION_1_3,
    };
    VK_CHECK(vmaCreateAllocator(&info, &m_allocator));
}

VkCommandBuffer VulkanContext::beginImmediate()
{
    const VkCommandBufferAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = m_immediatePool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(m_device, &allocInfo, &cmd));
    const VkCommandBufferBeginInfo begin{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    VK_CHECK(vkBeginCommandBuffer(cmd, &begin));
    return cmd;
}

void VulkanContext::endImmediate(VkCommandBuffer cmd)
{
    VK_CHECK(vkEndCommandBuffer(cmd));
    const VkCommandBufferSubmitInfo cmdInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .commandBuffer = cmd,
    };
    const VkSubmitInfo2 submit{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &cmdInfo,
    };
    VK_CHECK(vkQueueSubmit2(m_queue, 1, &submit, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(m_queue));
    vkFreeCommandBuffers(m_device, m_immediatePool, 1, &cmd);
}

// ---- Buffers --------------------------------------------------------------------

Buffer createBuffer(VmaAllocator allocator, VkDeviceSize size, VkBufferUsageFlags usage, MemoryKind kind)
{
    const VkBufferCreateInfo bufferInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VmaAllocationCreateInfo allocInfo{.usage = VMA_MEMORY_USAGE_AUTO};
    switch (kind) {
    case MemoryKind::GpuOnly:
        allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        break;
    case MemoryKind::Upload:
        allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        break;
    case MemoryKind::Readback:
        allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        break;
    }

    Buffer buffer;
    buffer.size = size;
    VmaAllocationInfo info{};
    VK_CHECK(vmaCreateBuffer(allocator, &bufferInfo, &allocInfo, &buffer.buffer, &buffer.allocation, &info));
    buffer.mapped = info.pMappedData;
    return buffer;
}

void destroyBuffer(VmaAllocator allocator, Buffer& buffer)
{
    if (buffer.buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator, buffer.buffer, buffer.allocation);
    }
    buffer = {};
}

Buffer createBufferWithData(VulkanContext& ctx, const void* data, VkDeviceSize size, VkBufferUsageFlags usage)
{
    Buffer staging = createBuffer(ctx.allocator(), size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, MemoryKind::Upload);
    std::memcpy(staging.mapped, data, size);
    VK_CHECK(vmaFlushAllocation(ctx.allocator(), staging.allocation, 0, VK_WHOLE_SIZE));

    Buffer gpu = createBuffer(ctx.allocator(), size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, MemoryKind::GpuOnly);
    ctx.immediateSubmit([&](VkCommandBuffer cmd) {
        const VkBufferCopy region{.size = size};
        vkCmdCopyBuffer(cmd, staging.buffer, gpu.buffer, 1, &region);
    });
    destroyBuffer(ctx.allocator(), staging);
    return gpu;
}

} // namespace engine::render
