#pragma once

#include "engine/core/Types.hpp"
#include "engine/render/BlockTextures.hpp"
#include "engine/render/Sky.hpp"
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
    bool gpuCulling = true; // cull on the GPU (compute + draw-indirect-count); on the CPU if off or unsupported

    // Levels of detail (see setChunkMesh). Level L >= 1 is hidden closer to
    // the camera than lodSeam * 2^(L - 1) blocks (horizontally), where the
    // finer level L - 1 is drawn instead; the finer level is drawn as far as
    // it has meshes, a little beyond the seam, so there are no gaps.
    // 0 = no seams (every chunk drawn wherever it is).
    f32 lodSeam = 0.0f;

    // Sun/moon shadows: cascaded shadow maps (4 layers of shadowMapSize^2)
    // covering terrain up to shadowDistance blocks from the camera (and no
    // further than lodSeam: only full-detail chunks cast shadows).
    bool shadows = true;
    u32 shadowMapSize = 2048;
    f32 shadowDistance = 256.0f;
};

struct FrameStats {
    f64 gpuMs = 0.0;     // GPU time of the most recently completed frame
    u32 drawnChunks = 0; // chunks that passed frustum/distance culling
    u32 draws = 0;       // indirect draws (up to 3 per chunk: runs of face directions)
    u64 drawnQuads = 0;
    f64 cullMs = 0.0;       // CPU time culling chunks and writing the draws (CPU culling only)
    f64 waitMs = 0.0;       // CPU time blocked waiting for the GPU (frame slot fence, swapchain image)
    u32 seamDraws = 0;      // of `draws`: coarse-level draws crossing a seam (need the discard pipeline)
    u32 shadowDraws = 0;    // draws into the shadow cascades (not included in `draws`)
    bool gpuCulled = false; // counts above come from the GPU, a couple of frames late
    u64 uploadedBytes = 0;  // mesh data copied to the GPU this frame
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
    // world position (in blocks) of its voxel (0,0,0). At level of detail L
    // each voxel is 2^L blocks wide and 2^min(L, 2) tall (voxel::lodScale,
    // voxel::lodVerticalScale). The quads are copied. Uploaded in the background
    // over the next frames (see RendererDesc::uploadBudgetBytes). False if there is no room. Call between frames.
    bool setChunkMesh(u64 key, glm::ivec3 origin, std::span<const voxel::PackedQuad> quads, u32 level = 0);
    void removeChunkMesh(u64 key);
    [[nodiscard]] MeshMemoryStats meshStats() const;
    [[nodiscard]] bool gpuCulling() const noexcept { return m_gpuCulling; }

    // Lighting: the direction towards the sun (see Sky.hpp: sunDirectionAt).
    // Sets the sky, fog, ambient and direct light (the moon at night).
    void setSun(glm::vec3 towardSun) noexcept { m_sky = skyLightFor(towardSun); }
    [[nodiscard]] const SkyLight& skyLight() const noexcept { return m_sky; }

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
        Buffer indirect; // CPU culling: VkDrawIndexedIndirectCommand per visible run, written by the CPU
        bool timestampsWritten = false;

        // GPU culling: draws written by the culling compute shader, the draw
        // count + statistics, and a host-readable copy of the counters.
        Buffer gpuDraws;
        Buffer counters;
        Buffer countersReadback;
        VkDescriptorSet cullSet = VK_NULL_HANDLE;
        bool countersWritten = false;
    };

    static constexpr u32 kCascades = 4;

    // This frame's shadow cascades, all camera-relative like everything else.
    struct Cascades {
        bool enabled = false;
        std::array<glm::mat4, kCascades> viewProj{};              // light clip space (depth 0..1, standard)
        std::array<std::array<glm::vec4, 5>, kCascades> planes{}; // left, right, bottom, top, far
        glm::vec4 splits{0.0f};                                   // view depth where each cascade ends
        glm::vec4 texelSize{0.0f};                                // world size of a shadow texel, per cascade
        u32 faceMask = 0; // face directions that face the light (bit per voxel::Face): the casters drawn
        f32 distance = 0.0f;
    };
    [[nodiscard]] Cascades computeCascades(const Camera& camera) const;

    void createPipeline();
    void createShadows();
    void destroyShadows();
    void recordShadows(VkCommandBuffer cmd, const Frame& frame, const Cascades& cascades, const Camera& camera,
                       const std::array<u32, kCascades>& counts) const;
    void createCulling();
    void destroyCulling();
    void createDepth();
    void destroyDepth();
    void createIndexBuffer();
    void destroyTextures();
    void recreateSwapchain();
    // Draw lists in the indirect buffers: interior draws at [0, m_maxDraws),
    // seam draws at [m_maxDraws, 2 * m_maxDraws), then the draws of shadow
    // cascade c at shadowListOffset(c), m_shadowCap each. With GPU culling
    // these counts are maximums; the real ones are in the counters buffer.
    struct DrawCounts {
        u32 interior = 0;
        u32 seam = 0;
        std::array<u32, kCascades> shadow{};
    };
    [[nodiscard]] u32 shadowListOffset(u32 cascade) const noexcept { return 2 * m_maxDraws + cascade * m_shadowCap; }
    [[nodiscard]] u32 totalDraws() const noexcept { return shadowListOffset(kCascades); }
    void recordCulling(VkCommandBuffer cmd, Frame& frame, const Camera& camera, const Cascades& cascades) const;
    [[nodiscard]] DrawCounts cullAndWriteDraws(Frame& frame, const Camera& camera, const Cascades& cascades,
                                               FrameStats& stats) const;
    [[nodiscard]] VkDeviceSize recordFrame(Frame& frame, u32 imageIndex, const DrawCounts& draws, const Camera& camera,
                                           const Cascades& cascades, bool screenshot);
    void writeLighting(u32 frameSlot, const Camera& camera, const Cascades& cascades) const;
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
    VkPipeline m_seamPipeline = VK_NULL_HANDLE;   // same, discarding fragments inside a chunk's LOD seam
    VkPipeline m_shadowPipeline = VK_NULL_HANDLE; // depth only, into one shadow cascade
    VkFormat m_pipelineColorFormat = VK_FORMAT_UNDEFINED;

    // Lighting: shadow cascades (one depth array image) and per-frame uniforms.
    SkyLight m_sky;
    VkImage m_shadowImage = VK_NULL_HANDLE;
    VmaAllocation m_shadowAllocation = nullptr;
    VkImageView m_shadowArrayView = VK_NULL_HANDLE;          // sampled
    std::array<VkImageView, kCascades> m_shadowLayerViews{}; // rendered into
    VkSampler m_shadowSampler = VK_NULL_HANDLE;              // depth comparison, bilinear (PCF)
    Buffer m_lightingBuffer;                                 // kFramesInFlight x m_lightingStride
    VkDeviceSize m_lightingStride = 0;
    u32 m_shadowCap = 0; // draws per shadow cascade list

    Buffer m_indexBuffer; // shared quad index pattern, sized for the densest possible chunk
    std::unique_ptr<ChunkMeshStore> m_meshes;

    // GPU culling
    bool m_gpuCulling = false;
    u32 m_maxDraws = 0;
    VkDescriptorSetLayout m_cullSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_cullPool = VK_NULL_HANDLE;
    VkPipelineLayout m_cullPipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_cullPipeline = VK_NULL_HANDLE;

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
