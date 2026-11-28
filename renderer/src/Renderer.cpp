#include "engine/render/Renderer.hpp"

#include "engine/core/Profile.hpp"
#include "engine/platform/Window.hpp"
#include "engine/render/Camera.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstring>
#include <fstream>
#include <vector>

#include "ChunkMeshStore.hpp"

namespace engine::render {

namespace {

// SPIR-V compiled from renderer/shaders at build time and embedded, so the
// executable has no shader files to find at runtime.
constexpr u32 kVoxelVertSpv[] =
#include "voxel.vert.spv.inc"
    ;
constexpr u32 kCullCompSpv[] =
#include "cull.comp.spv.inc"
    ;
constexpr u32 kVoxelFragSpv[] =
#include "voxel.frag.spv.inc"
    ;

constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;
// 16 bits are plenty for shadow depth (a cascade's depth range is a few
// hundred blocks) and halve the shadow pass's bandwidth.
constexpr VkFormat kShadowFormat = VK_FORMAT_D16_UNORM;

// Per-frame lighting uniforms (binding 5, std140; see voxel.frag).
struct LightingUniforms {
    glm::mat4 cascades[4];    // camera-relative position -> light clip space, per cascade
    glm::vec4 splits;         // view depth where each cascade ends
    glm::vec4 texelSize;      // world size of a shadow texel, per cascade
    glm::vec4 lightDirection; // xyz: towards the sun/moon; w: 1 if shadows are on
    glm::vec4 lightColor;     // rgb; w: shadow distance (shadows fade out towards it)
    glm::vec4 ambientSky;
    glm::vec4 ambientGround;
    glm::vec4 cameraForward;
};

// Fully fogged beyond this distance from the camera to a chunk's centre: fog
// distance plus (more than) half a chunk's diagonal (27.7 blocks at level 0).
f32 cullDistance(f32 fogDistance, u32 level) noexcept
{
    return fogDistance + 56.0f * static_cast<f32>(1u << level);
}

// Horizontal radius inside which a chunk of `level` is not drawn: there the
// finer level is. 0 for level 0 or without LOD seams.
f32 lodInnerRadius(f32 seam, u32 level) noexcept
{
    return level == 0 ? 0.0f : seam * static_cast<f32>(1u << (level - 1));
}

// Densest possible chunk is a 3D checkerboard: half the voxels solid, all 6 faces visible.
constexpr u32 kMaxQuadsPerChunk = voxel::kChunkVolume / 2 * 6;

struct PushConstants {
    glm::mat4 viewProj;       // projection * rotation only: the camera is at the origin
    glm::ivec4 cameraBlock;   // xyz: integer block containing the camera
    glm::vec4 cameraFraction; // xyz: camera position within that block; w: LOD seam radius of level 1 (0: none)
    glm::vec4 fog;
};
static_assert(sizeof(PushConstants) <= 128, "128 bytes is the guaranteed minimum push constant size");

// Push constants of the culling compute shader (shaders/cull.comp).
struct CullPushConstants {
    glm::ivec4 cameraBlock;
    glm::vec4 cameraFraction; // w: fog distance
    glm::vec4 planes[5];
    glm::uvec4 params; // x: slot count, y: face culling, z: draw capacity (per list), w: LOD seam radius (f32 bits)
};
static_assert(sizeof(CullPushConstants) == 128);

constexpr u32 kCullGroupSize = 64; // local_size_x in cull.comp
// Counters written by cull.comp: interior draws, chunks, quads, seam draws,
// then the draw count of each shadow cascade.
constexpr u32 kCounterCount = 8;
constexpr u32 kShadowCounter = 4;

VkShaderModule createShaderModule(VkDevice device, std::span<const u32> spirv)
{
    const VkShaderModuleCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spirv.size_bytes(),
        .pCode = spirv.data(),
    };
    VkShaderModule module = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(device, &info, nullptr, &module));
    return module;
}

VkImageMemoryBarrier2 imageBarrier(VkImage image, VkImageAspectFlags aspect, VkPipelineStageFlags2 srcStage,
                                   VkAccessFlags2 srcAccess, VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess,
                                   VkImageLayout oldLayout, VkImageLayout newLayout, u32 layers = 1)
{
    // Single mip: render targets and swapchain images.
    return VkImageMemoryBarrier2{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask = srcStage,
        .srcAccessMask = srcAccess,
        .dstStageMask = dstStage,
        .dstAccessMask = dstAccess,
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = {aspect, 0, 1, 0, layers},
    };
}

VkImageMemoryBarrier2 wholeImageBarrier(VkImageMemoryBarrier2 b)
{
    b.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
    b.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;
    return b;
}

void pipelineBarriers(VkCommandBuffer cmd, std::span<const VkImageMemoryBarrier2> barriers)
{
    const VkDependencyInfo dep{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = static_cast<u32>(barriers.size()),
        .pImageMemoryBarriers = barriers.data(),
    };
    vkCmdPipelineBarrier2(cmd, &dep);
}

} // namespace

Renderer::Renderer(VulkanContext& ctx, platform::Window& window, const RendererDesc& desc)
    : m_ctx(ctx), m_window(window), m_desc(desc)
{
    VkDevice device = m_ctx.device();
    const platform::Extent e = m_window.framebufferExtent();
    m_swapchain = std::make_unique<Swapchain>(m_ctx, VkExtent2D{e.width, e.height}, desc.presentMode);
    createDepth();

    // One descriptor set for the whole world:
    //   0: quads, 1: chunk origins, 3: (block, face) -> texture layer  (vertex shader)
    //   2: block texture array, 4: shadow cascades                    (fragment shader)
    //   5: lighting uniforms, a dynamic offset per frame in flight
    const VkDescriptorSetLayoutBinding bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr},
        {4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {5, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
    };
    const VkDescriptorSetLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 6,
        .pBindings = bindings,
    };
    VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &m_setLayout));

    const VkDescriptorPoolSize poolSizes[] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1},
    };
    const VkDescriptorPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 3,
        .pPoolSizes = poolSizes,
    };
    VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &m_descriptorPool));
    const VkDescriptorSetAllocateInfo setInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = m_descriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &m_setLayout,
    };
    VK_CHECK(vkAllocateDescriptorSets(device, &setInfo, &m_set));
    createShadows(); // also writes bindings 4 and 5

    const VkPushConstantRange pushRange{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                                        sizeof(PushConstants)};
    const VkPipelineLayoutCreateInfo pipelineLayoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &m_setLayout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushRange,
    };
    VK_CHECK(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &m_pipelineLayout));
    createPipeline();
    createIndexBuffer();

    m_meshes = std::make_unique<ChunkMeshStore>(m_ctx, desc.quadCapacity, desc.maxChunks, kFramesInFlight,
                                                desc.uploadBudgetBytes);
    const VkDescriptorBufferInfo meshBuffers[] = {
        {m_meshes->quadBuffer(), 0, VK_WHOLE_SIZE},
        {m_meshes->chunkTable(), 0, VK_WHOLE_SIZE},
    };
    VkWriteDescriptorSet meshWrites[2];
    for (u32 i = 0; i < 2; ++i) {
        meshWrites[i] = VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = m_set,
            .dstBinding = i,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &meshBuffers[i],
        };
    }
    vkUpdateDescriptorSets(device, 2, meshWrites, 0, nullptr);
    m_maxDraws = desc.maxChunks * 3; // up to 3 runs of face directions per chunk
    // Shadow casters are full-detail chunks near the camera: far fewer.
    m_shadowCap = std::min(m_maxDraws, 3u * 16384u);

    for (Frame& f : m_frames) {
        const VkCommandPoolCreateInfo cmdPoolInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
            .queueFamilyIndex = m_ctx.queueFamily(),
        };
        VK_CHECK(vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &f.pool));
        const VkCommandBufferAllocateInfo cmdInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = f.pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        VK_CHECK(vkAllocateCommandBuffers(device, &cmdInfo, &f.cmd));

        const VkFenceCreateInfo fenceInfo{
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT, // first wait returns immediately
        };
        VK_CHECK(vkCreateFence(device, &fenceInfo, nullptr, &f.inFlight));
        const VkSemaphoreCreateInfo semInfo{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VK_CHECK(vkCreateSemaphore(device, &semInfo, nullptr, &f.imageAcquired));
        // Worst case: every resident chunk visible.
        // Up to 3 draws per chunk (runs of face directions facing the camera).
        // All draw lists (see DrawCounts).
        f.indirect = createBuffer(m_ctx.allocator(), totalDraws() * sizeof(VkDrawIndexedIndirectCommand),
                                  VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, MemoryKind::Upload);
    }

    const VkQueryPoolCreateInfo queryInfo{
        .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .queryType = VK_QUERY_TYPE_TIMESTAMP,
        .queryCount = 2 * kFramesInFlight,
    };
    VK_CHECK(vkCreateQueryPool(device, &queryInfo, nullptr, &m_queryPool));

    m_gpuCulling = desc.gpuCulling && m_ctx.drawIndirectCount();
    if (desc.gpuCulling && !m_gpuCulling) {
        std::fprintf(stderr, "GPU culling unavailable (no drawIndirectCount): culling on the CPU\n");
    }
    if (m_gpuCulling) {
        createCulling();
    }
}

void Renderer::createCulling()
{
    VkDevice device = m_ctx.device();
    const VkDescriptorSetLayoutBinding bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // chunk table
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // draws
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // counters
    };
    const VkDescriptorSetLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = bindings,
    };
    VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &m_cullSetLayout));

    const VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 * kFramesInFlight};
    const VkDescriptorPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = kFramesInFlight,
        .poolSizeCount = 1,
        .pPoolSizes = &poolSize,
    };
    VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &m_cullPool));

    const VkPushConstantRange pushRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(CullPushConstants)};
    const VkPipelineLayoutCreateInfo pipelineLayoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &m_cullSetLayout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushRange,
    };
    VK_CHECK(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &m_cullPipelineLayout));

    VkShaderModule module = createShaderModule(device, kCullCompSpv);
    const VkComputePipelineCreateInfo pipelineInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                  .module = module,
                  .pName = "main"},
        .layout = m_cullPipelineLayout,
    };
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_cullPipeline));
    vkDestroyShaderModule(device, module, nullptr);

    for (Frame& f : m_frames) {
        f.gpuDraws =
            createBuffer(m_ctx.allocator(), totalDraws() * sizeof(VkDrawIndexedIndirectCommand), // all lists
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, MemoryKind::GpuOnly);
        f.counters = createBuffer(m_ctx.allocator(), kCounterCount * sizeof(u32),
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                  MemoryKind::GpuOnly);
        f.countersReadback = createBuffer(m_ctx.allocator(), kCounterCount * sizeof(u32),
                                          VK_BUFFER_USAGE_TRANSFER_DST_BIT, MemoryKind::Readback);

        const VkDescriptorSetAllocateInfo setInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = m_cullPool,
            .descriptorSetCount = 1,
            .pSetLayouts = &m_cullSetLayout,
        };
        VK_CHECK(vkAllocateDescriptorSets(device, &setInfo, &f.cullSet));
        const VkDescriptorBufferInfo infos[] = {
            {m_meshes->chunkTable(), 0, VK_WHOLE_SIZE},
            {f.gpuDraws.buffer, 0, VK_WHOLE_SIZE},
            {f.counters.buffer, 0, VK_WHOLE_SIZE},
        };
        VkWriteDescriptorSet writes[3];
        for (u32 i = 0; i < 3; ++i) {
            writes[i] = VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = f.cullSet,
                .dstBinding = i,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &infos[i],
            };
        }
        vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);
    }
}

void Renderer::destroyCulling()
{
    VkDevice device = m_ctx.device();
    for (Frame& f : m_frames) {
        destroyBuffer(m_ctx.allocator(), f.gpuDraws);
        destroyBuffer(m_ctx.allocator(), f.counters);
        destroyBuffer(m_ctx.allocator(), f.countersReadback);
    }
    vkDestroyPipeline(device, m_cullPipeline, nullptr);
    vkDestroyPipelineLayout(device, m_cullPipelineLayout, nullptr);
    vkDestroyDescriptorPool(device, m_cullPool, nullptr);
    vkDestroyDescriptorSetLayout(device, m_cullSetLayout, nullptr);
}

void Renderer::recordCulling(VkCommandBuffer cmd, Frame& frame, const Camera& camera, const Cascades& cascades) const
{
    ENGINE_PROFILE_FUNCTION();
    // This frame slot's culling buffers were last used two frames ago and its
    // fence has been waited on since, so nothing can overlap in practice. Say
    // so to Vulkan anyway: order all earlier indirect reads, copies and
    // compute writes before this frame's resets and writes. They all happen
    // at the very start of earlier frames, so this costs nothing.
    const VkMemoryBarrier2 reuse{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT |
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
    };
    const VkDependencyInfo reuseDep{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &reuse};
    vkCmdPipelineBarrier2(cmd, &reuseDep);

    // Reset the draw count and statistics.
    vkCmdFillBuffer(cmd, frame.counters.buffer, 0, VK_WHOLE_SIZE, 0);
    const VkMemoryBarrier2 cleared{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
        .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
    };
    const VkDependencyInfo clearDep{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &cleared};
    vkCmdPipelineBarrier2(cmd, &clearDep);

    const VkExtent2D extent = m_swapchain->extent();
    const f32 aspect = static_cast<f32>(extent.width) / static_cast<f32>(extent.height);
    const Frustum frustum = Frustum::fromViewProj(camera.projection(aspect) * camera.viewRotation());
    const u32 slots = m_meshes->slotCount();
    CullPushConstants pc{
        .cameraBlock = glm::ivec4{camera.cameraBlock(), 0},
        .cameraFraction = glm::vec4{camera.cameraFraction(), m_desc.fogDistance},
        .planes = {},
        .params = glm::uvec4{slots, m_desc.faceCulling ? 1u : 0u, m_maxDraws, std::bit_cast<u32>(m_desc.lodSeam)},
    };
    for (usize i = 0; i < 5; ++i) {
        pc.planes[i] = frustum.planes[i];
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_cullPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_cullPipelineLayout, 0, 1, &frame.cullSet, 0,
                            nullptr);
    vkCmdPushConstants(cmd, m_cullPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    if (slots > 0) {
        vkCmdDispatch(cmd, (slots + kCullGroupSize - 1) / kCullGroupSize, 1, 1);
    }
    // Shadow casters, one dispatch per cascade (cameraBlock.w = 1 + cascade):
    // full-detail chunks in the cascade's box, faces towards the light.
    if (cascades.enabled && slots > 0) {
        for (u32 c = 0; c < kCascades; ++c) {
            CullPushConstants sp{
                .cameraBlock = glm::ivec4{camera.cameraBlock(), static_cast<i32>(1 + c)},
                .cameraFraction = glm::vec4{camera.cameraFraction(), 0.0f},
                .planes = {},
                .params = glm::uvec4{slots, cascades.faceMask, m_shadowCap, shadowListOffset(c)},
            };
            for (usize i = 0; i < 5; ++i) {
                sp.planes[i] = cascades.planes[c][i];
            }
            vkCmdPushConstants(cmd, m_cullPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(sp), &sp);
            vkCmdDispatch(cmd, (slots + kCullGroupSize - 1) / kCullGroupSize, 1, 1);
        }
    }

    // Draws and count are read by the indirect draw; the statistics copied out.
    const VkMemoryBarrier2 culled{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COPY_BIT,
        .dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT,
    };
    const VkDependencyInfo culledDep{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &culled};
    vkCmdPipelineBarrier2(cmd, &culledDep);

    const VkBufferCopy region{.size = kCounterCount * sizeof(u32)};
    vkCmdCopyBuffer(cmd, frame.counters.buffer, frame.countersReadback.buffer, 1, &region);
    const VkMemoryBarrier2 toHost{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
        .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
        .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT,
    };
    const VkDependencyInfo hostDep{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &toHost};
    vkCmdPipelineBarrier2(cmd, &hostDep);
}

Renderer::~Renderer()
{
    VkDevice device = m_ctx.device();
    vkDeviceWaitIdle(device);

    for (Frame& f : m_frames) {
        destroyBuffer(m_ctx.allocator(), f.indirect);
        vkDestroySemaphore(device, f.imageAcquired, nullptr);
        vkDestroyFence(device, f.inFlight, nullptr);
        vkDestroyCommandPool(device, f.pool, nullptr);
    }
    vkDestroyQueryPool(device, m_queryPool, nullptr);
    m_meshes.reset();
    destroyTextures();
    destroyBuffer(m_ctx.allocator(), m_indexBuffer);
    destroyBuffer(m_ctx.allocator(), m_screenshotBuffer);
    if (m_gpuCulling) {
        destroyCulling();
    }
    vkDestroyPipeline(device, m_pipeline, nullptr);
    vkDestroyPipeline(device, m_seamPipeline, nullptr);
    vkDestroyPipeline(device, m_shadowPipeline, nullptr);
    destroyShadows();
    vkDestroyPipelineLayout(device, m_pipelineLayout, nullptr);
    vkDestroyDescriptorPool(device, m_descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(device, m_setLayout, nullptr);
    destroyDepth();
    m_swapchain.reset();
}

void Renderer::createPipeline()
{
    VkDevice device = m_ctx.device();
    if (m_pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, m_pipeline, nullptr);
        vkDestroyPipeline(device, m_seamPipeline, nullptr);
        vkDestroyPipeline(device, m_shadowPipeline, nullptr);
    }

    VkShaderModule vert = createShaderModule(device, kVoxelVertSpv);
    VkShaderModule frag = createShaderModule(device, kVoxelFragSpv);

    // Two variants: the seam pipeline discards fragments inside the chunk's
    // LOD seam (specialisation constant 0 in voxel.frag). Everything else
    // uses the plain one, so early depth testing stays fully effective.
    const VkSpecializationMapEntry discardEntry{.constantID = 0, .offset = 0, .size = sizeof(VkBool32)};
    const VkBool32 discardOff = VK_FALSE;
    const VkBool32 discardOn = VK_TRUE;
    const VkSpecializationInfo plainSpec{
        .mapEntryCount = 1, .pMapEntries = &discardEntry, .dataSize = sizeof(VkBool32), .pData = &discardOff};
    const VkSpecializationInfo seamSpec{
        .mapEntryCount = 1, .pMapEntries = &discardEntry, .dataSize = sizeof(VkBool32), .pData = &discardOn};
    VkPipelineShaderStageCreateInfo stages[] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = vert,
         .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = frag,
         .pName = "main",
         .pSpecializationInfo = &plainSpec},
    };
    const VkPipelineVertexInputStateCreateInfo vertexInput{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO}; // vertex pulling: no attributes
    const VkPipelineInputAssemblyStateCreateInfo inputAssembly{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };
    const VkPipelineViewportStateCreateInfo viewport{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1,
    };
    const VkPipelineRasterizationStateCreateInfo raster{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_BACK_BIT,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .lineWidth = 1.0f,
    };
    const VkPipelineMultisampleStateCreateInfo multisample{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };
    const VkPipelineDepthStencilStateCreateInfo depth{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL, // reversed Z: nearer = larger
    };
    const VkPipelineColorBlendAttachmentState blendAttachment{
        .colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };
    const VkPipelineColorBlendStateCreateInfo blend{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &blendAttachment,
    };
    const VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    const VkPipelineDynamicStateCreateInfo dynamic{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2,
        .pDynamicStates = dynamicStates,
    };

    m_pipelineColorFormat = m_swapchain->format();
    // Dynamic rendering (Vulkan 1.3): attachment formats instead of a VkRenderPass.
    const VkPipelineRenderingCreateInfo rendering{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &m_pipelineColorFormat,
        .depthAttachmentFormat = kDepthFormat,
    };

    VkGraphicsPipelineCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &rendering,
        .stageCount = 2,
        .pStages = stages,
        .pVertexInputState = &vertexInput,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewport,
        .pRasterizationState = &raster,
        .pMultisampleState = &multisample,
        .pDepthStencilState = &depth,
        .pColorBlendState = &blend,
        .pDynamicState = &dynamic,
        .layout = m_pipelineLayout,
    };
    VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &m_pipeline));
    stages[1].pSpecializationInfo = &seamSpec;
    VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &m_seamPipeline));

    // Shadow casters: the same vertex shader with a light-space viewProj,
    // depth only. Only faces towards the light are drawn (chosen by the
    // culling), so no face culling here; the bias keeps lit surfaces from
    // shadowing themselves (acne).
    const VkPipelineRasterizationStateCreateInfo shadowRaster{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .depthBiasEnable = VK_TRUE,
        .depthBiasConstantFactor = 2.0f,
        .depthBiasSlopeFactor = 2.0f,
        .lineWidth = 1.0f,
    };
    const VkPipelineDepthStencilStateCreateInfo shadowDepth{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL, // standard depth: 0 nearest the light
    };
    const VkPipelineColorBlendStateCreateInfo noColor{.sType =
                                                          VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    const VkPipelineRenderingCreateInfo shadowRendering{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .depthAttachmentFormat = kShadowFormat,
    };
    VkGraphicsPipelineCreateInfo shadowInfo = info;
    shadowInfo.pNext = &shadowRendering;
    shadowInfo.stageCount = 1; // vertex shader only
    shadowInfo.pRasterizationState = &shadowRaster;
    shadowInfo.pDepthStencilState = &shadowDepth;
    shadowInfo.pColorBlendState = &noColor;
    VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &shadowInfo, nullptr, &m_shadowPipeline));

    vkDestroyShaderModule(device, vert, nullptr);
    vkDestroyShaderModule(device, frag, nullptr);
}

void Renderer::createDepth()
{
    const VkExtent2D extent = m_swapchain->extent();
    const VkImageCreateInfo imageInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = kDepthFormat,
        .extent = {extent.width, extent.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    const VmaAllocationCreateInfo allocInfo{
        .flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT, // render targets prefer their own allocation
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
    };
    VK_CHECK(vmaCreateImage(m_ctx.allocator(), &imageInfo, &allocInfo, &m_depthImage, &m_depthAllocation, nullptr));

    const VkImageViewCreateInfo viewInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = m_depthImage,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = kDepthFormat,
        .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1},
    };
    VK_CHECK(vkCreateImageView(m_ctx.device(), &viewInfo, nullptr, &m_depthView));
}

void Renderer::destroyDepth()
{
    vkDestroyImageView(m_ctx.device(), m_depthView, nullptr);
    vmaDestroyImage(m_ctx.allocator(), m_depthImage, m_depthAllocation);
    m_depthView = VK_NULL_HANDLE;
    m_depthImage = VK_NULL_HANDLE;
    m_depthAllocation = nullptr;
}

void Renderer::createShadows()
{
    VkDevice device = m_ctx.device();
    // When shadows are off the map is never rendered, but the descriptor
    // still needs an image: a 1x1 one.
    const u32 size = m_desc.shadows ? std::max(m_desc.shadowMapSize, 64u) : 1u;
    const VkImageCreateInfo imageInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = kShadowFormat,
        .extent = {size, size, 1},
        .mipLevels = 1,
        .arrayLayers = kCascades,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage =
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    const VmaAllocationCreateInfo allocInfo{
        .flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
    };
    VK_CHECK(vmaCreateImage(m_ctx.allocator(), &imageInfo, &allocInfo, &m_shadowImage, &m_shadowAllocation, nullptr));

    VkImageViewCreateInfo viewInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = m_shadowImage,
        .viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY,
        .format = kShadowFormat,
        .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, kCascades},
    };
    VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &m_shadowArrayView));
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    for (u32 c = 0; c < kCascades; ++c) {
        viewInfo.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, c, 1};
        VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &m_shadowLayerViews[c]));
    }

    // Hardware depth comparison with bilinear filtering: each lookup is a
    // 2x2 percentage-closer filter. Outside the map: lit (border = depth 1).
    const VkSamplerCreateInfo samplerInfo{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .compareEnable = VK_TRUE,
        .compareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
        .maxLod = 0.0f,
        .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
    };
    VK_CHECK(vkCreateSampler(device, &samplerInfo, nullptr, &m_shadowSampler));

    // The image is sampled in DEPTH_READ_ONLY_OPTIMAL; frames that render it
    // transition it and back. Start it there (cleared to "lit").
    m_ctx.immediateSubmit([&](VkCommandBuffer cmd) {
        const VkImageMemoryBarrier2 toClear{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
            .dstStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT,
            .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = m_shadowImage,
            .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, kCascades},
        };
        pipelineBarriers(cmd, {&toClear, 1});
        const VkClearDepthStencilValue one{1.0f, 0};
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, kCascades};
        vkCmdClearDepthStencilImage(cmd, m_shadowImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &one, 1, &range);
        const VkImageMemoryBarrier2 toRead{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT,
            .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            .dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = m_shadowImage,
            .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, kCascades},
        };
        pipelineBarriers(cmd, {&toRead, 1});
    });

    // Lighting uniforms: one block per frame in flight, selected with a dynamic offset.
    const VkDeviceSize align = std::max<VkDeviceSize>(m_ctx.properties().limits.minUniformBufferOffsetAlignment, 16);
    m_lightingStride = (sizeof(LightingUniforms) + align - 1) / align * align;
    m_lightingBuffer = createBuffer(m_ctx.allocator(), m_lightingStride * kFramesInFlight,
                                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, MemoryKind::Upload);

    const VkDescriptorImageInfo shadowInfo{m_shadowSampler, m_shadowArrayView, VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL};
    const VkDescriptorBufferInfo lightingInfo{m_lightingBuffer.buffer, 0, sizeof(LightingUniforms)};
    const VkWriteDescriptorSet writes[] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = m_set,
         .dstBinding = 4,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .pImageInfo = &shadowInfo},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = m_set,
         .dstBinding = 5,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
         .pBufferInfo = &lightingInfo},
    };
    vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
}

void Renderer::destroyShadows()
{
    VkDevice device = m_ctx.device();
    destroyBuffer(m_ctx.allocator(), m_lightingBuffer);
    vkDestroySampler(device, m_shadowSampler, nullptr);
    for (VkImageView view : m_shadowLayerViews) {
        vkDestroyImageView(device, view, nullptr);
    }
    vkDestroyImageView(device, m_shadowArrayView, nullptr);
    vmaDestroyImage(m_ctx.allocator(), m_shadowImage, m_shadowAllocation);
}

Renderer::Cascades Renderer::computeCascades(const Camera& camera) const
{
    Cascades out;
    f32 distance = m_desc.shadowDistance;
    if (m_desc.lodSeam > 0.0f) {
        distance = std::min(distance, m_desc.lodSeam); // only full-detail chunks cast shadows
    }
    const glm::vec3 light = glm::normalize(m_sky.lightDirection);
    // No shadows without a light worth shadowing (deep twilight) or one at
    // the horizon (shadows would stretch to infinity).
    if (!m_desc.shadows || distance <= 1.0f || light.y < 0.05f || glm::dot(m_sky.lightColor, glm::vec3{1.0f}) < 0.02f) {
        return out;
    }
    out.enabled = true;
    out.distance = distance;

    const VkExtent2D extent = m_swapchain->extent();
    const f32 aspect = static_cast<f32>(extent.width) / static_cast<f32>(extent.height);
    const f32 tanY = std::tan(camera.fovY * 0.5f);
    const f32 tanX = tanY * aspect;
    const glm::vec3 forward = camera.forward();
    const glm::vec3 right = camera.right();
    const glm::vec3 up = glm::cross(right, forward);
    const f32 size = static_cast<f32>(std::max(m_desc.shadowMapSize, 64u));

    // Light view axes (world = camera-relative orientation).
    const glm::dvec3 l{light};
    const glm::dvec3 helper = std::abs(l.y) > 0.99 ? glm::dvec3{0.0, 0.0, 1.0} : glm::dvec3{0.0, 1.0, 0.0};
    const glm::dvec3 lx = glm::normalize(glm::cross(helper, l));
    const glm::dvec3 ly = glm::cross(l, lx);

    // Cascades end at distance * (1/4)^2, (2/4)^2, (3/4)^2, 1: dense near the camera.
    f32 nearDepth = camera.nearPlane;
    for (u32 c = 0; c < kCascades; ++c) {
        const f32 t = static_cast<f32>(c + 1) / static_cast<f32>(kCascades);
        const f32 farDepth = distance * t * t;
        out.splits[static_cast<glm::length_t>(c)] = farDepth;

        // Bounding sphere of this slice of the view frustum. It does not
        // change shape as the camera turns, so the cascade's size (and texel
        // size) stays constant: no shimmering from rotation.
        glm::vec3 corners[8];
        glm::vec3 centre{0.0f};
        for (u32 i = 0; i < 8; ++i) {
            const f32 d = (i & 4u) ? farDepth : nearDepth;
            const f32 sx = (i & 1u) ? 1.0f : -1.0f;
            const f32 sy = (i & 2u) ? 1.0f : -1.0f;
            corners[i] = forward * d + right * (sx * d * tanX) + up * (sy * d * tanY);
            centre += corners[i] * 0.125f;
        }
        f32 radius = 0.0f;
        for (const glm::vec3& p : corners) {
            radius = std::max(radius, glm::length(p - centre));
        }
        radius = std::ceil(radius * 8.0f) / 8.0f;
        const f64 texel = 2.0 * static_cast<f64>(radius) / static_cast<f64>(size);
        out.texelSize[static_cast<glm::length_t>(c)] = static_cast<f32>(texel);

        // Snap the centre to whole texels along the light's axes, in world
        // space (double precision): moving the camera then shifts the shadow
        // map by whole texels, so shadow edges do not crawl.
        const glm::dvec3 world = camera.position + glm::dvec3{centre};
        glm::dvec3 snapped = world;
        const f64 ax = std::floor(glm::dot(world, lx) / texel) * texel;
        const f64 ay = std::floor(glm::dot(world, ly) / texel) * texel;
        snapped += lx * (ax - glm::dot(world, lx)) + ly * (ay - glm::dot(world, ly));

        // Light looks along -l from beyond the sphere; casters up to the top
        // of the world (and a margin) in front of it are included.
        const f64 back = static_cast<f64>(radius) + 256.0;
        const glm::dvec3 eye = snapped + l * back;
        const glm::dmat4 view = glm::lookAtRH(eye, snapped, ly);
        const f64 r = static_cast<f64>(radius);
        const glm::dmat4 proj = glm::orthoRH_ZO(-r, r, -r, r, 0.0, back + r);
        // Camera-relative: p_world = p_relative + camera position.
        const glm::dmat4 relative = proj * view * glm::translate(glm::dmat4{1.0}, camera.position);
        const glm::mat4 m{relative};
        out.viewProj[c] = m;

        // Culling planes in camera-relative space: the 4 sides and the far
        // plane. No near plane: casters between the light and the box count.
        const auto row = [&](int k) { return glm::vec4{m[0][k], m[1][k], m[2][k], m[3][k]}; };
        const glm::vec4 r0 = row(0), r1 = row(1), r2 = row(2), r3 = row(3);
        out.planes[c] = {r3 + r0, r3 - r0, r3 + r1, r3 - r1, r3 - r2};
        for (glm::vec4& p : out.planes[c]) {
            p /= glm::length(glm::vec3{p});
        }
        nearDepth = farDepth;
    }
    // Faces whose outward normal points towards the light (voxel::Face order).
    out.faceMask = (light.x > 0.0f ? 1u : 0u) | (light.x < 0.0f ? 2u : 0u) | (light.y > 0.0f ? 4u : 0u) |
                   (light.y < 0.0f ? 8u : 0u) | (light.z > 0.0f ? 16u : 0u) | (light.z < 0.0f ? 32u : 0u);
    return out;
}

void Renderer::writeLighting(u32 frameSlot, const Camera& camera, const Cascades& cascades) const
{
    LightingUniforms u{};
    for (u32 c = 0; c < kCascades; ++c) {
        u.cascades[c] = cascades.viewProj[c];
    }
    u.splits = cascades.splits;
    u.texelSize = cascades.texelSize;
    u.lightDirection = glm::vec4{glm::normalize(m_sky.lightDirection), cascades.enabled ? 1.0f : 0.0f};
    u.lightColor = glm::vec4{m_sky.lightColor, cascades.distance};
    u.ambientSky = glm::vec4{m_sky.ambientSky, 0.0f};
    u.ambientGround = glm::vec4{m_sky.ambientGround, 0.0f};
    u.cameraForward = glm::vec4{camera.forward(), 0.0f};
    auto* dst = static_cast<std::byte*>(m_lightingBuffer.mapped) + frameSlot * m_lightingStride;
    std::memcpy(dst, &u, sizeof(u));
    VK_CHECK(
        vmaFlushAllocation(m_ctx.allocator(), m_lightingBuffer.allocation, frameSlot * m_lightingStride, sizeof(u)));
}

void Renderer::createIndexBuffer()
{
    // Quad q uses corners 4q..4q+3 as triangles (0,1,2) and (2,1,3). Each
    // chunk's draw starts at index 0 and offsets vertices with vertexOffset, so
    // one buffer sized for the densest chunk serves every draw.
    std::vector<u32> indices(static_cast<usize>(kMaxQuadsPerChunk) * 6);
    for (u32 q = 0; q < kMaxQuadsPerChunk; ++q) {
        const u32 v = q * 4;
        u32* dst = &indices[static_cast<usize>(q) * 6];
        dst[0] = v + 0;
        dst[1] = v + 1;
        dst[2] = v + 2;
        dst[3] = v + 2;
        dst[4] = v + 1;
        dst[5] = v + 3;
    }
    m_indexBuffer =
        createBufferWithData(m_ctx, indices.data(), indices.size() * sizeof(u32), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
}

bool Renderer::setChunkMesh(u64 key, glm::ivec3 origin, std::span<const voxel::PackedQuad> quads, u32 level)
{
    ENGINE_ASSERT(quads.size() <= kMaxQuadsPerChunk);
    ENGINE_ASSERT(level < 16);
    return m_meshes->set(key, origin, quads, level);
}

void Renderer::removeChunkMesh(u64 key)
{
    m_meshes->remove(key);
}

MeshMemoryStats Renderer::meshStats() const
{
    const ChunkMeshStore::Stats s = m_meshes->stats();
    return {s.chunks, s.quads, s.usedBytes, s.capacityBytes, s.pendingBytes, s.failed, s.grows};
}

void Renderer::destroyTextures()
{
    VkDevice device = m_ctx.device();
    vkDestroySampler(device, m_sampler, nullptr);
    vkDestroyImageView(device, m_textureView, nullptr);
    if (m_textureImage != VK_NULL_HANDLE) {
        vmaDestroyImage(m_ctx.allocator(), m_textureImage, m_textureAllocation);
    }
    destroyBuffer(m_ctx.allocator(), m_faceLayerBuffer);
    m_sampler = VK_NULL_HANDLE;
    m_textureView = VK_NULL_HANDLE;
    m_textureImage = VK_NULL_HANDLE;
    m_textureAllocation = nullptr;
}

void Renderer::setBlockTextures(const BlockTextureSet& set)
{
    ENGINE_PROFILE_FUNCTION();
    VkDevice device = m_ctx.device();
    vkDeviceWaitIdle(device);
    destroyTextures();

    const TextureArrayData& tex = set.textures;
    const VkImageCreateInfo imageInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = VK_FORMAT_R8G8B8A8_SRGB, // decoded to linear by the sampler
        .extent = {tex.size, tex.size, 1},
        .mipLevels = tex.mipLevels,
        .arrayLayers = tex.layers,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    const VmaAllocationCreateInfo allocInfo{.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    VK_CHECK(vmaCreateImage(m_ctx.allocator(), &imageInfo, &allocInfo, &m_textureImage, &m_textureAllocation, nullptr));

    Buffer staging =
        createBuffer(m_ctx.allocator(), tex.pixels.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, MemoryKind::Upload);
    std::memcpy(staging.mapped, tex.pixels.data(), tex.pixels.size());
    VK_CHECK(vmaFlushAllocation(m_ctx.allocator(), staging.allocation, 0, VK_WHOLE_SIZE));

    // One copy region per mip level; each covers every layer of that level.
    std::vector<VkBufferImageCopy> regions(tex.mipLevels);
    for (u32 level = 0; level < tex.mipLevels; ++level) {
        regions[level] = VkBufferImageCopy{
            .bufferOffset = tex.levelOffsets[level],
            .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, tex.layers},
            .imageExtent = {tex.levelSize(level), tex.levelSize(level), 1},
        };
    }
    m_ctx.immediateSubmit([&](VkCommandBuffer cmd) {
        const VkImageMemoryBarrier2 toTransfer = wholeImageBarrier(imageBarrier(
            m_textureImage, VK_IMAGE_ASPECT_COLOR_BIT, VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COPY_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL));
        pipelineBarriers(cmd, {&toTransfer, 1});
        vkCmdCopyBufferToImage(cmd, staging.buffer, m_textureImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               static_cast<u32>(regions.size()), regions.data());
        const VkImageMemoryBarrier2 toShader = wholeImageBarrier(imageBarrier(
            m_textureImage, VK_IMAGE_ASPECT_COLOR_BIT, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
        pipelineBarriers(cmd, {&toShader, 1});
    });
    destroyBuffer(m_ctx.allocator(), staging);

    const VkImageViewCreateInfo viewInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = m_textureImage,
        .viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY,
        .format = VK_FORMAT_R8G8B8A8_SRGB,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, tex.mipLevels, 0, tex.layers},
    };
    VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &m_textureView));

    // Crisp texels up close (nearest magnification), smooth and shimmer-free
    // at a distance (trilinear + anisotropic minification).
    const bool aniso = m_ctx.samplerAnisotropy();
    const VkSamplerCreateInfo samplerInfo{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_NEAREST,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
        // Clamp: tiling across merged quads is done in the shader (fract), and
        // REPEAT would bleed each texture's opposite edge into block borders.
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .anisotropyEnable = aniso ? VK_TRUE : VK_FALSE,
        .maxAnisotropy = aniso ? std::min(16.0f, m_ctx.properties().limits.maxSamplerAnisotropy) : 1.0f,
        .maxLod = VK_LOD_CLAMP_NONE,
    };
    VK_CHECK(vkCreateSampler(device, &samplerInfo, nullptr, &m_sampler));

    m_faceLayerBuffer = createBufferWithData(m_ctx, set.faceLayers.data(), set.faceLayers.size() * sizeof(u32),
                                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    const VkDescriptorImageInfo imageDescriptor{m_sampler, m_textureView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    const VkDescriptorBufferInfo faceLayerDescriptor{m_faceLayerBuffer.buffer, 0, VK_WHOLE_SIZE};
    const VkWriteDescriptorSet writes[] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = m_set,
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &imageDescriptor,
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = m_set,
            .dstBinding = 3,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &faceLayerDescriptor,
        },
    };
    vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
}

void Renderer::recreateSwapchain()
{
    m_window.waitWhileMinimised();
    vkDeviceWaitIdle(m_ctx.device());
    const platform::Extent e = m_window.framebufferExtent();
    m_swapchain->recreate({e.width, e.height});
    destroyDepth();
    createDepth();
    if (m_swapchain->format() != m_pipelineColorFormat) {
        createPipeline();
    }
}

namespace {

// Appends one draw per contiguous run of visible face directions of a chunk
// (at most 3) to `list`, up to `capacity`. Returns the quads drawn.
u64 appendRuns(const ChunkMeshStore::Slot& c, u32 slot, const bool (&visible)[voxel::kFaceCount],
               VkDrawIndexedIndirectCommand* list, u32& count, u32 capacity)
{
    u64 drawn = 0;
    for (u32 f = 0; f < voxel::kFaceCount;) {
        if (!visible[f]) {
            ++f;
            continue;
        }
        u32 end = f + 1;
        while (end < voxel::kFaceCount && visible[end]) {
            ++end;
        }
        const u32 first = c.faceStart[f];
        const u32 quads = c.faceStart[end] - first;
        if (quads > 0 && count < capacity) {
            list[count++] = VkDrawIndexedIndirectCommand{
                .indexCount = quads * 6,
                .instanceCount = 1,
                .firstIndex = 0,
                .vertexOffset = static_cast<i32>((c.firstQuad + first) * 4),
                .firstInstance = slot, // slot index: selects the chunk origin in the shader
            };
            drawn += quads;
        }
        f = end;
    }
    return drawn;
}

} // namespace

Renderer::DrawCounts Renderer::cullAndWriteDraws(Frame& frame, const Camera& camera, const Cascades& cascades,
                                                 FrameStats& stats) const
{
    ENGINE_PROFILE_FUNCTION();
    const auto start = std::chrono::steady_clock::now();
    const VkExtent2D extent = m_swapchain->extent();
    const f32 aspect = static_cast<f32>(extent.width) / static_cast<f32>(extent.height);
    // Everything here is relative to the camera, like the vertex shader.
    const Frustum frustum = Frustum::fromViewProj(camera.projection(aspect) * camera.viewRotation());
    const glm::ivec3 camBlock = camera.cameraBlock();
    const glm::vec3 camFrac = camera.cameraFraction();

    auto* cmds = static_cast<VkDrawIndexedIndirectCommand*>(frame.indirect.mapped);
    DrawCounts counts;
    const std::span<const ChunkMeshStore::Slot> slots = m_meshes->slots();
    for (u32 i = 0; i < slots.size(); ++i) {
        const ChunkMeshStore::Slot& c = slots[i];
        if (!c.drawable || c.quadCount == 0) {
            continue;
        }
        const f32 size = static_cast<f32>(voxel::kChunkSize << c.level);
        const glm::vec3 min = glm::vec3{c.origin - camBlock} - camFrac; // chunk box, camera at 0
        const glm::vec3 max = min + size;

        // Shadow casters: full-detail chunks in each cascade's box, faces
        // towards the light (the same tests as cull.comp).
        if (cascades.enabled && c.level == 0) {
            bool lit[voxel::kFaceCount];
            for (u32 f = 0; f < voxel::kFaceCount; ++f) {
                lit[f] = ((cascades.faceMask >> f) & 1u) != 0;
            }
            for (u32 k = 0; k < kCascades; ++k) {
                if (Frustum{cascades.planes[k]}.intersects(min, max)) {
                    appendRuns(c, i, lit, cmds + shadowListOffset(k), counts.shadow[k], m_shadowCap);
                }
            }
        }

        const glm::vec3 d = (min + max) * 0.5f;
        const f32 maxDist = cullDistance(m_desc.fogDistance, c.level);
        if (glm::dot(d, d) > maxDist * maxDist || !frustum.intersects(min, max)) {
            continue;
        }
        // Levels of detail: skip chunks entirely inside their seam (the finer
        // level covers that); chunks crossing it go to the discard pipeline.
        bool seam = false;
        if (const f32 inner = lodInnerRadius(m_desc.lodSeam, c.level); inner > 0.0f) {
            const glm::vec2 nearest{std::max({min.x, -max.x, 0.0f}), std::max({min.z, -max.z, 0.0f})};
            const glm::vec2 farthest{std::max(-min.x, max.x), std::max(-min.z, max.z)};
            if (glm::dot(farthest, farthest) <= inner * inner) {
                continue;
            }
            seam = glm::dot(nearest, nearest) < inner * inner;
        }

        // Face-direction culling: a face can only be seen from in front of
        // its plane. +X faces lie on planes from min.x + 1 to max.x, so none
        // is visible unless the camera (at 0) is beyond min.x; likewise for
        // the others. Each chunk's quads are grouped by direction
        // (PosX, NegX, PosY, NegY, PosZ, NegZ), so visible directions form at
        // most 3 contiguous runs: one draw each.
        const bool all = !m_desc.faceCulling;
        const bool visible[voxel::kFaceCount] = {
            all || min.x < 0.0f, all || max.x > 0.0f, // +X, -X
            all || min.y < 0.0f, all || max.y > 0.0f, // +Y, -Y
            all || min.z < 0.0f, all || max.z > 0.0f, // +Z, -Z
        };
        u32& count = seam ? counts.seam : counts.interior;
        const u32 drawsBefore = count;
        stats.drawnQuads += appendRuns(c, i, visible, cmds + (seam ? m_maxDraws : 0), count, m_maxDraws);
        if (count > drawsBefore) {
            ++stats.drawnChunks; // counted like the GPU path: only chunks that produced a draw
        }
    }
    constexpr VkDeviceSize kCmd = sizeof(VkDrawIndexedIndirectCommand);
    const auto flush = [&](u32 first, u32 count) {
        if (count > 0) {
            VK_CHECK(vmaFlushAllocation(m_ctx.allocator(), frame.indirect.allocation, first * kCmd, count * kCmd));
        }
    };
    flush(0, counts.interior);
    flush(m_maxDraws, counts.seam);
    for (u32 k = 0; k < kCascades; ++k) {
        flush(shadowListOffset(k), counts.shadow[k]);
        stats.shadowDraws += counts.shadow[k];
    }
    stats.draws = counts.interior + counts.seam;
    stats.seamDraws = counts.seam;
    stats.cullMs = std::chrono::duration<f64, std::milli>(std::chrono::steady_clock::now() - start).count();
    return counts;
}

f64 Renderer::readGpuTime(u32 frameIndex) const
{
    u64 ts[2] = {};
    const VkResult r = vkGetQueryPoolResults(m_ctx.device(), m_queryPool, frameIndex * 2, 2, sizeof(ts), ts,
                                             sizeof(u64), VK_QUERY_RESULT_64_BIT);
    if (r != VK_SUCCESS) {
        return 0.0;
    }
    const f64 ticks = static_cast<f64>(ts[1] - ts[0]);
    return ticks * static_cast<f64>(m_ctx.properties().limits.timestampPeriod) * 1e-6;
}

FrameStats Renderer::renderFrame(const Camera& camera)
{
    ENGINE_PROFILE_FUNCTION();
    VkDevice device = m_ctx.device();
    Frame& frame = m_frames[m_frameIndex];
    FrameStats stats;

    {
        ENGINE_PROFILE_SCOPE("waitForFrameFence");
        const auto waitStart = std::chrono::steady_clock::now();
        VK_CHECK(vkWaitForFences(device, 1, &frame.inFlight, VK_TRUE, UINT64_MAX));
        stats.waitMs = std::chrono::duration<f64, std::milli>(std::chrono::steady_clock::now() - waitStart).count();
    }
    if (frame.timestampsWritten) {
        stats.gpuMs = readGpuTime(m_frameIndex);
    }
    m_meshes->beginFrame(m_frameIndex); // the GPU is done with what this frame slot last used
    if (m_meshes->consumeBufferChanged()) {
        // The quad buffer grew (in setChunkMesh, which waited for the GPU, so
        // no submitted work still uses this descriptor).
        const VkDescriptorBufferInfo quadInfo{m_meshes->quadBuffer(), 0, VK_WHOLE_SIZE};
        const VkWriteDescriptorSet write{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = m_set,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &quadInfo,
        };
        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }

    if (m_window.consumeResized()) {
        recreateSwapchain();
    }

    u32 imageIndex = 0;
    const auto acquireStart = std::chrono::steady_clock::now();
    VkResult acquire = vkAcquireNextImageKHR(device, m_swapchain->handle(), UINT64_MAX, frame.imageAcquired,
                                             VK_NULL_HANDLE, &imageIndex);
    // Acquiring can block too (until the presentation engine frees an image).
    stats.waitMs += std::chrono::duration<f64, std::milli>(std::chrono::steady_clock::now() - acquireStart).count();
    if (acquire == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapchain();
        return stats; // the semaphore was not signalled; try again next frame
    }
    if (acquire != VK_SUCCESS && acquire != VK_SUBOPTIMAL_KHR) {
        VK_CHECK(acquire);
    }
    VK_CHECK(vkResetFences(device, 1, &frame.inFlight));

    const Cascades cascades = computeCascades(camera);
    writeLighting(m_frameIndex, camera, cascades); // this slot's previous frame is done with it

    DrawCounts drawCounts;
    if (m_gpuCulling) {
        // Statistics of this slot's previous frame (the GPU is done with it).
        if (frame.countersWritten) {
            VK_CHECK(vmaInvalidateAllocation(m_ctx.allocator(), frame.countersReadback.allocation, 0, VK_WHOLE_SIZE));
            const auto* counters = static_cast<const u32*>(frame.countersReadback.mapped);
            stats.seamDraws = std::min(counters[3], m_maxDraws);
            stats.draws = std::min(counters[0], m_maxDraws) + stats.seamDraws;
            stats.drawnChunks = counters[1];
            stats.drawnQuads = counters[2];
            for (u32 c = 0; c < kCascades; ++c) {
                stats.shadowDraws += std::min(counters[kShadowCounter + c], m_shadowCap);
            }
        }
        stats.gpuCulled = true;
        // The real counts come from the GPU; these are the maximums.
        const bool any = m_meshes->slotCount() > 0;
        drawCounts.interior = any ? m_maxDraws : 0;
        drawCounts.seam = any && m_desc.lodSeam > 0.0f ? m_maxDraws : 0;
        drawCounts.shadow.fill(any && cascades.enabled ? m_shadowCap : 0);
    } else {
        drawCounts = cullAndWriteDraws(frame, camera, cascades, stats);
    }

    const bool screenshot = !m_screenshotPath.empty();
    stats.uploadedBytes = recordFrame(frame, imageIndex, drawCounts, camera, cascades, screenshot);

    const VkSemaphoreSubmitInfo waitInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .semaphore = frame.imageAcquired,
        .stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
    };
    VkSemaphore renderFinished = m_swapchain->renderFinished(imageIndex);
    const VkSemaphoreSubmitInfo signalInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .semaphore = renderFinished,
        .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
    };
    const VkCommandBufferSubmitInfo cmdInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .commandBuffer = frame.cmd,
    };
    const VkSubmitInfo2 submit{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .waitSemaphoreInfoCount = 1,
        .pWaitSemaphoreInfos = &waitInfo,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &cmdInfo,
        .signalSemaphoreInfoCount = 1,
        .pSignalSemaphoreInfos = &signalInfo,
    };
    VK_CHECK(vkQueueSubmit2(m_ctx.queue(), 1, &submit, frame.inFlight));
    frame.timestampsWritten = true;
    frame.countersWritten = m_gpuCulling;

    if (screenshot) {
        VK_CHECK(vkWaitForFences(device, 1, &frame.inFlight, VK_TRUE, UINT64_MAX));
        writeScreenshot();
    }

    VkSwapchainKHR swapchain = m_swapchain->handle();
    const VkPresentInfoKHR present{
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &renderFinished,
        .swapchainCount = 1,
        .pSwapchains = &swapchain,
        .pImageIndices = &imageIndex,
    };
    const VkResult presentResult = vkQueuePresentKHR(m_ctx.queue(), &present);
    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR) {
        recreateSwapchain();
    } else if (presentResult != VK_SUCCESS) {
        VK_CHECK(presentResult);
    }

    m_frameIndex = (m_frameIndex + 1) % kFramesInFlight;
    m_meshes->endFrame(m_frameIndex);
    return stats;
}

void Renderer::recordShadows(VkCommandBuffer cmd, const Frame& frame, const Cascades& cascades, const Camera& camera,
                             const std::array<u32, kCascades>& counts) const
{
    ENGINE_PROFILE_FUNCTION();
    // The previous frame sampled the cascades; their contents are replaced.
    const VkImageMemoryBarrier2 toRender =
        imageBarrier(m_shadowImage, VK_IMAGE_ASPECT_DEPTH_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, 0,
                     VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                     VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                     VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, kCascades);
    pipelineBarriers(cmd, {&toRender, 1});

    const u32 size = std::max(m_desc.shadowMapSize, 64u);
    const VkViewport viewport{0.0f, 0.0f, static_cast<f32>(size), static_cast<f32>(size), 0.0f, 1.0f};
    const VkRect2D scissor{{0, 0}, {size, size}};
    constexpr u32 kStride = sizeof(VkDrawIndexedIndirectCommand);
    const u32 lightingOffset = static_cast<u32>(m_frameIndex * m_lightingStride);
    for (u32 c = 0; c < kCascades; ++c) {
        const VkRenderingAttachmentInfo depthAttachment{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = m_shadowLayerViews[c],
            .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue = {.depthStencil = {1.0f, 0}}, // standard depth: 1 is furthest from the light
        };
        const VkRenderingInfo renderingInfo{
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .renderArea = scissor,
            .layerCount = 1,
            .pDepthAttachment = &depthAttachment,
        };
        vkCmdBeginRendering(cmd, &renderingInfo);
        if (counts[c] > 0) {
            vkCmdSetViewport(cmd, 0, 1, &viewport);
            vkCmdSetScissor(cmd, 0, 1, &scissor);
            const PushConstants pc{
                .viewProj = cascades.viewProj[c],
                .cameraBlock = glm::ivec4{camera.cameraBlock(), 0},
                .cameraFraction = glm::vec4{camera.cameraFraction(), 0.0f},
                .fog = glm::vec4{0.0f},
            };
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_shadowPipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_set, 1,
                                    &lightingOffset);
            vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                               sizeof(pc), &pc);
            vkCmdBindIndexBuffer(cmd, m_indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
            const VkDeviceSize offset = VkDeviceSize{shadowListOffset(c)} * kStride;
            if (m_gpuCulling) {
                vkCmdDrawIndexedIndirectCount(cmd, frame.gpuDraws.buffer, offset, frame.counters.buffer,
                                              (kShadowCounter + c) * sizeof(u32), counts[c], kStride);
            } else {
                vkCmdDrawIndexedIndirect(cmd, frame.indirect.buffer, offset, counts[c], kStride);
            }
        }
        vkCmdEndRendering(cmd);
    }

    const VkImageMemoryBarrier2 toSample =
        imageBarrier(m_shadowImage, VK_IMAGE_ASPECT_DEPTH_BIT, VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                     VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                     VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                     VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL, kCascades);
    pipelineBarriers(cmd, {&toSample, 1});
}

VkDeviceSize Renderer::recordFrame(Frame& frame, u32 imageIndex, const DrawCounts& draws, const Camera& camera,
                                   const Cascades& cascades, bool screenshot)
{
    ENGINE_PROFILE_FUNCTION();
    VkCommandBuffer cmd = frame.cmd;
    VK_CHECK(vkResetCommandPool(m_ctx.device(), frame.pool, 0));
    const VkCommandBufferBeginInfo begin{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    VK_CHECK(vkBeginCommandBuffer(cmd, &begin));

    const u32 query = m_frameIndex * 2;
    vkCmdResetQueryPool(cmd, m_queryPool, query, 2);
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, m_queryPool, query);

    // New chunk meshes, copied in before anything draws (outside the render pass).
    const VkDeviceSize uploaded = m_meshes->recordUploads(cmd, m_frameIndex);
    if (m_gpuCulling) {
        recordCulling(cmd, frame, camera, cascades); // also outside the render pass
    }
    if (cascades.enabled && m_textureImage != VK_NULL_HANDLE) {
        recordShadows(cmd, frame, cascades, camera, draws.shadow);
    }

    VkImage colorImage = m_swapchain->image(imageIndex);
    const VkExtent2D extent = m_swapchain->extent();

    // Old contents are discarded (UNDEFINED); both images are cleared on load.
    // The depth barrier also orders against the previous frame's depth writes,
    // since one depth image is shared by all frames in flight.
    const VkImageMemoryBarrier2 toRender[] = {
        imageBarrier(colorImage, VK_IMAGE_ASPECT_COLOR_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                     VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL),
        imageBarrier(m_depthImage, VK_IMAGE_ASPECT_DEPTH_BIT, VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                     VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                     VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                     VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL),
    };
    pipelineBarriers(cmd, toRender);

    const VkRenderingAttachmentInfo colorAttachment{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = m_swapchain->view(imageIndex),
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue = {.color = {{m_sky.skyColor.r, m_sky.skyColor.g, m_sky.skyColor.b, 1.0f}}},
    };
    const VkRenderingAttachmentInfo depthAttachment{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = m_depthView,
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE, // depth is not needed after the pass
        .clearValue = {.depthStencil = {0.0f, 0}},   // reversed Z: 0 is infinitely far
    };
    const VkRenderingInfo renderingInfo{
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = {{0, 0}, extent},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorAttachment,
        .pDepthAttachment = &depthAttachment,
    };
    vkCmdBeginRendering(cmd, &renderingInfo);

    // Every descriptor must be written before drawing (world + textures).
    if (draws.interior + draws.seam > 0 && m_textureImage != VK_NULL_HANDLE) {
        const VkViewport viewport{0.0f, 0.0f, static_cast<f32>(extent.width), static_cast<f32>(extent.height),
                                  0.0f, 1.0f};
        const VkRect2D scissor{{0, 0}, extent};
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        const f32 aspect = static_cast<f32>(extent.width) / static_cast<f32>(extent.height);
        const PushConstants pc{
            .viewProj = camera.projection(aspect) * camera.viewRotation(),
            .cameraBlock = glm::ivec4{camera.cameraBlock(), 0},
            .cameraFraction = glm::vec4{camera.cameraFraction(), m_desc.lodSeam},
            .fog = glm::vec4{m_sky.skyColor, m_desc.fogDistance},
        };
        const u32 lightingOffset = static_cast<u32>(m_frameIndex * m_lightingStride);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_set, 1,
                                &lightingOffset);
        vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(pc), &pc);
        vkCmdBindIndexBuffer(cmd, m_indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
        // The whole visible world in (at most) two calls: chunks clear of any
        // LOD seam, then the coarse chunks crossing one, which discard the
        // fragments that the finer level covers.
        constexpr u32 kStride = sizeof(VkDrawIndexedIndirectCommand);
        const VkDeviceSize seamOffset = VkDeviceSize{m_maxDraws} * kStride;
        const std::pair<VkPipeline, u32> lists[] = {{m_pipeline, draws.interior}, {m_seamPipeline, draws.seam}};
        for (u32 list = 0; list < 2; ++list) {
            const auto [pipeline, count] = lists[list];
            if (count == 0) {
                continue;
            }
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
            if (m_gpuCulling) {
                // As many draws as the culling shader produced (at most count);
                // counters[0] counts interior draws, counters[3] seam draws.
                vkCmdDrawIndexedIndirectCount(cmd, frame.gpuDraws.buffer, list * seamOffset, frame.counters.buffer,
                                              list * 3 * sizeof(u32), count, kStride);
            } else {
                vkCmdDrawIndexedIndirect(cmd, frame.indirect.buffer, list * seamOffset, count, kStride);
            }
        }
    }
    vkCmdEndRendering(cmd);

    // Transition to PRESENT_SRC. The destination stage must be a real stage
    // (not NONE) so the transition is ordered before the semaphore signal
    // that presentation waits on.
    if (screenshot) {
        const VkDeviceSize size = static_cast<VkDeviceSize>(extent.width) * extent.height * 4;
        if (m_screenshotBuffer.size < size) {
            destroyBuffer(m_ctx.allocator(), m_screenshotBuffer);
            m_screenshotBuffer =
                createBuffer(m_ctx.allocator(), size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, MemoryKind::Readback);
        }
        const VkImageMemoryBarrier2 toCopy = imageBarrier(
            colorImage, VK_IMAGE_ASPECT_COLOR_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        pipelineBarriers(cmd, {&toCopy, 1});
        const VkBufferImageCopy region{
            .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .imageExtent = {extent.width, extent.height, 1},
        };
        vkCmdCopyImageToBuffer(cmd, colorImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, m_screenshotBuffer.buffer, 1,
                               &region);
        const VkImageMemoryBarrier2 toPresent = imageBarrier(
            colorImage, VK_IMAGE_ASPECT_COLOR_BIT, VK_PIPELINE_STAGE_2_COPY_BIT, 0, VK_PIPELINE_STAGE_2_COPY_BIT, 0,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
        pipelineBarriers(cmd, {&toPresent, 1});
    } else {
        const VkImageMemoryBarrier2 toPresent =
            imageBarrier(colorImage, VK_IMAGE_ASPECT_COLOR_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
        pipelineBarriers(cmd, {&toPresent, 1});
    }

    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, m_queryPool, query + 1);
    VK_CHECK(vkEndCommandBuffer(cmd));
    return uploaded;
}

void Renderer::writeScreenshot()
{
    const VkExtent2D extent = m_swapchain->extent();
    VK_CHECK(vmaInvalidateAllocation(m_ctx.allocator(), m_screenshotBuffer.allocation, 0, VK_WHOLE_SIZE));
    const auto* pixels = static_cast<const u8*>(m_screenshotBuffer.mapped);
    const bool bgra =
        m_swapchain->format() == VK_FORMAT_B8G8R8A8_SRGB || m_swapchain->format() == VK_FORMAT_B8G8R8A8_UNORM;

    // 24-bit BMP: rows bottom-up, BGR, each row padded to 4 bytes.
    const u32 w = extent.width;
    const u32 h = extent.height;
    const u32 rowBytes = (w * 3 + 3) & ~3u;
    const u32 dataSize = rowBytes * h;
    u8 header[54] = {'B', 'M'};
    const auto put32 = [&](usize at, u32 v) {
        for (usize i = 0; i < 4; ++i) {
            header[at + i] = static_cast<u8>(v >> (8 * i));
        }
    };
    put32(2, 54 + dataSize); // file size
    put32(10, 54);           // pixel data offset
    put32(14, 40);           // BITMAPINFOHEADER size
    put32(18, w);
    put32(22, h);
    header[26] = 1;  // planes
    header[28] = 24; // bits per pixel
    put32(34, dataSize);

    std::vector<u8> data(dataSize, 0);
    for (u32 y = 0; y < h; ++y) {
        const u8* src = pixels + static_cast<usize>(h - 1 - y) * w * 4;
        u8* dst = data.data() + static_cast<usize>(y) * rowBytes;
        for (u32 x = 0; x < w; ++x) {
            const u8* p = src + static_cast<usize>(x) * 4;
            dst[x * 3 + 0] = bgra ? p[0] : p[2];
            dst[x * 3 + 1] = p[1];
            dst[x * 3 + 2] = bgra ? p[2] : p[0];
        }
    }

    std::ofstream file(m_screenshotPath, std::ios::binary);
    file.write(reinterpret_cast<const char*>(header), sizeof(header));
    file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    std::fprintf(stderr, "Screenshot written to %s\n", m_screenshotPath.c_str());
    m_screenshotPath.clear();
}

} // namespace engine::render
