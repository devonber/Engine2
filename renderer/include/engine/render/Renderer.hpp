#pragma once

#include "engine/core/Types.hpp"
#include "engine/render/BlockTextures.hpp"
#include "engine/render/Swapchain.hpp"
#include "engine/render/VulkanContext.hpp"
#include "engine/voxel/Mesher.hpp"

#include <array>
#include <memory>
#include <span>
#include <string>

namespace engine::platform {
class Window;
}

namespace engine::render {

struct Camera;

// One chunk's slice of the world quad buffer.
struct ChunkDraw {
    i32 originX = 0, originY = 0, originZ = 0; // world position of voxel (0,0,0), in blocks
    u32 firstQuad = 0;
    u32 quadCount = 0;
};

struct RendererDesc {
    PresentMode presentMode = PresentMode::Uncapped;
    f32 fogDistance = 300.0f; // blocks
};

struct FrameStats {
    f64 gpuMs = 0.0; // GPU time of the most recently completed frame
    u32 drawnChunks = 0;
    u64 drawnQuads = 0;
};

class Renderer {
public:
    static constexpr u32 kFramesInFlight = 2;

    Renderer(VulkanContext& ctx, platform::Window& window, const RendererDesc& desc);
    ~Renderer();
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    // Replace the whole world. Quads of chunk i live at
    // quads[chunks[i].firstQuad .. +quadCount]. Stalls the GPU; load-time only.
    void uploadWorld(std::span<const voxel::PackedQuad> quads, std::span<const ChunkDraw> chunks);

    // Upload block textures and the (block, face) -> texture layer table.
    // Must be called before anything is drawn. Stalls the GPU; load-time only.
    void setBlockTextures(const BlockTextureSet& textures);

    FrameStats renderFrame(const Camera& camera);

    // Write the next rendered frame to a 24-bit .bmp file.
    void requestScreenshot(std::string path) { m_screenshotPath = std::move(path); }

private:
    struct Frame {
        VkCommandPool pool = VK_NULL_HANDLE;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkFence inFlight = VK_NULL_HANDLE;
        VkSemaphore imageAcquired = VK_NULL_HANDLE;
        Buffer indirect; // VkDrawIndexedIndirectCommand per visible chunk, written by the CPU
        bool timestampsWritten = false;
    };

    void createPipeline();
    void createDepth();
    void destroyDepth();
    void createIndexBuffer();
    void destroyWorld();
    void destroyTextures();
    void recreateSwapchain();
    [[nodiscard]] u32 cullAndWriteDraws(Frame& frame, const Camera& camera, u64& drawnQuads) const;
    void recordFrame(Frame& frame, u32 imageIndex, u32 drawCount, const Camera& camera, bool screenshot);
    void writeScreenshot();
    [[nodiscard]] f64 readGpuTime(u32 frameIndex) const;

    VulkanContext& m_ctx;
    platform::Window& m_window;
    RendererDesc m_desc;
    std::unique_ptr<Swapchain> m_swapchain;

    VkImage m_depthImage = VK_NULL_HANDLE;
    VmaAllocation m_depthAllocation = nullptr;
    VkImageView m_depthView = VK_NULL_HANDLE;

    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet m_set = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkFormat m_pipelineColorFormat = VK_FORMAT_UNDEFINED;

    Buffer m_indexBuffer; // shared quad index pattern, sized for the densest possible chunk
    Buffer m_quadBuffer;
    Buffer m_chunkBuffer;
    std::vector<ChunkDraw> m_chunks;

    VkImage m_textureImage = VK_NULL_HANDLE;
    VmaAllocation m_textureAllocation = nullptr;
    VkImageView m_textureView = VK_NULL_HANDLE;
    VkSampler m_sampler = VK_NULL_HANDLE;
    Buffer m_faceLayerBuffer;

    std::array<Frame, kFramesInFlight> m_frames{};
    u32 m_frameIndex = 0;
    VkQueryPool m_queryPool = VK_NULL_HANDLE;

    std::string m_screenshotPath;
    Buffer m_screenshotBuffer;
};

} // namespace engine::render
