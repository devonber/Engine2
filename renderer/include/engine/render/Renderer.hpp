#pragma once

#include "engine/core/Types.hpp"
#include "engine/render/BlockTextures.hpp"
#include "engine/render/Swapchain.hpp"
#include "engine/render/VulkanContext.hpp"
#include "engine/voxel/Mesher.hpp"

#include <array>
#include <glm/glm.hpp>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace engine::platform {
class Window;
}

namespace engine::render {

struct Camera;

struct RendererDesc {
    PresentMode presentMode = PresentMode::Uncapped;
    f32 fogDistance = 300.0f; // blocks

    // Chunk mesh storage (fixed at start-up).
    u32 maxChunks = 65536;                    // resident chunk meshes
    u64 quadCapacity = 8u * 1024 * 1024;      // initial quads in the shared buffer (8 bytes each: 64 MiB); grows
    u64 uploadBudgetBytes = 4u * 1024 * 1024; // mesh data copied to the GPU per frame, at most
    bool faceCulling = true;                  // skip each chunk's face directions that point away from the camera
};

struct FrameStats {
    f64 gpuMs = 0.0;     // GPU time of the most recently completed frame
    u32 drawnChunks = 0; // chunks that passed frustum/distance culling
    u32 draws = 0;       // indirect draws (up to 3 per chunk: runs of face directions)
    u64 drawnQuads = 0;
    f64 cullMs = 0.0;      // CPU time culling chunks and writing the draws
    u64 uploadedBytes = 0; // mesh data copied to the GPU this frame
};

struct MeshMemoryStats {
    u32 chunks = 0;
    u64 quads = 0;
    u64 usedBytes = 0;
    u64 capacityBytes = 0;
    u64 pendingBytes = 0; // waiting for upload budget
    u32 failed = 0;       // meshes rejected: all chunk slots in use
    u32 grows = 0;        // times the quad buffer was enlarged (each stalls the GPU once)
};

class ChunkMeshStore;

class Renderer {
public:
    static constexpr u32 kFramesInFlight = 2;

    Renderer(VulkanContext& ctx, platform::Window& window, const RendererDesc& desc);
    ~Renderer();
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    // Add or replace the mesh of the chunk identified by `key`; `origin` is the
    // world position of its voxel (0,0,0). The quads are copied. Uploaded in the background over
    // the next frames (see RendererDesc::uploadBudgetBytes). False if there is
    // no room. Call between frames.
    bool setChunkMesh(u64 key, glm::ivec3 origin, std::span<const voxel::PackedQuad> quads);
    void removeChunkMesh(u64 key);
    [[nodiscard]] MeshMemoryStats meshStats() const;

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
    void destroyTextures();
    void recreateSwapchain();
    [[nodiscard]] u32 cullAndWriteDraws(Frame& frame, const Camera& camera, FrameStats& stats) const;
    [[nodiscard]] VkDeviceSize recordFrame(Frame& frame, u32 imageIndex, u32 drawCount, const Camera& camera,
                                           bool screenshot);
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
    std::unique_ptr<ChunkMeshStore> m_meshes;

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
