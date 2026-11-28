#include "engine/render/Renderer.hpp"

#include "engine/core/Profile.hpp"
#include "engine/platform/Window.hpp"
#include "engine/render/Camera.hpp"

#include <algorithm>
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
constexpr u32 kVoxelFragSpv[] =
#include "voxel.frag.spv.inc"
    ;

constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;

// Densest possible chunk is a 3D checkerboard: half the voxels solid, all 6 faces visible.
constexpr u32 kMaxQuadsPerChunk = voxel::kChunkVolume / 2 * 6;

// Sky colour, linear. Also the fog colour so distant terrain fades into it.
constexpr f32 kSky[3] = {0.36f, 0.58f, 0.92f};

struct PushConstants {
    glm::mat4 viewProj;       // projection * rotation only: the camera is at the origin
    glm::ivec4 cameraBlock;   // xyz: integer block containing the camera
    glm::vec4 cameraFraction; // xyz: camera position within that block
    glm::vec4 fog;
};
static_assert(sizeof(PushConstants) <= 128, "128 bytes is the guaranteed minimum push constant size");

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
                                   VkImageLayout oldLayout, VkImageLayout newLayout)
{
    // Single mip, single layer: render targets and swapchain images.
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
        .subresourceRange = {aspect, 0, 1, 0, 1},
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
    //   2: block texture array                                        (fragment shader)
    const VkDescriptorSetLayoutBinding bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr},
    };
    const VkDescriptorSetLayoutCreateInfo layoutInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 4,
        .pBindings = bindings,
    };
    VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &m_setLayout));

    const VkDescriptorPoolSize poolSizes[] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
    };
    const VkDescriptorPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 2,
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
        {m_meshes->originBuffer(), 0, VK_WHOLE_SIZE},
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
        f.indirect = createBuffer(m_ctx.allocator(), desc.maxChunks * 3 * sizeof(VkDrawIndexedIndirectCommand),
                                  VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, MemoryKind::Upload);
    }

    const VkQueryPoolCreateInfo queryInfo{
        .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .queryType = VK_QUERY_TYPE_TIMESTAMP,
        .queryCount = 2 * kFramesInFlight,
    };
    VK_CHECK(vkCreateQueryPool(device, &queryInfo, nullptr, &m_queryPool));
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
    vkDestroyPipeline(device, m_pipeline, nullptr);
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
    }

    VkShaderModule vert = createShaderModule(device, kVoxelVertSpv);
    VkShaderModule frag = createShaderModule(device, kVoxelFragSpv);

    const VkPipelineShaderStageCreateInfo stages[] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = vert,
         .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = frag,
         .pName = "main"},
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

    const VkGraphicsPipelineCreateInfo info{
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

bool Renderer::setChunkMesh(u64 key, glm::ivec3 origin, std::span<const voxel::PackedQuad> quads)
{
    ENGINE_ASSERT(quads.size() <= kMaxQuadsPerChunk);
    return m_meshes->set(key, origin, quads);
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

u32 Renderer::cullAndWriteDraws(Frame& frame, const Camera& camera, FrameStats& stats) const
{
    ENGINE_PROFILE_FUNCTION();
    const auto start = std::chrono::steady_clock::now();
    const VkExtent2D extent = m_swapchain->extent();
    const f32 aspect = static_cast<f32>(extent.width) / static_cast<f32>(extent.height);
    // Everything here is relative to the camera, like the vertex shader.
    const Frustum frustum = Frustum::fromViewProj(camera.projection(aspect) * camera.viewRotation());
    const f32 maxDistSq = (m_desc.fogDistance + 56.0f) * (m_desc.fogDistance + 56.0f); // fully fogged beyond
    const glm::ivec3 camBlock = camera.cameraBlock();
    const glm::vec3 camFrac = camera.cameraFraction();

    auto* cmds = static_cast<VkDrawIndexedIndirectCommand*>(frame.indirect.mapped);
    u32 count = 0;
    constexpr f32 kSize = static_cast<f32>(voxel::kChunkSize);
    const std::span<const ChunkMeshStore::Slot> slots = m_meshes->slots();
    for (u32 i = 0; i < slots.size(); ++i) {
        const ChunkMeshStore::Slot& c = slots[i];
        if (!c.drawable || c.quadCount == 0) {
            continue;
        }
        const glm::vec3 min = glm::vec3{c.origin - camBlock} - camFrac; // chunk box, camera at 0
        const glm::vec3 max = min + kSize;
        const glm::vec3 d = (min + max) * 0.5f;
        if (glm::dot(d, d) > maxDistSq || !frustum.intersects(min, max)) {
            continue;
        }
        ++stats.drawnChunks;

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
            if (quads > 0) {
                cmds[count++] = VkDrawIndexedIndirectCommand{
                    .indexCount = quads * 6,
                    .instanceCount = 1,
                    .firstIndex = 0,
                    .vertexOffset = static_cast<i32>((c.firstQuad + first) * 4),
                    .firstInstance = i, // slot index: selects the chunk origin in the shader
                };
                stats.drawnQuads += quads;
            }
            f = end;
        }
    }
    if (count > 0) {
        VK_CHECK(vmaFlushAllocation(m_ctx.allocator(), frame.indirect.allocation, 0,
                                    count * sizeof(VkDrawIndexedIndirectCommand)));
    }
    stats.draws = count;
    stats.cullMs = std::chrono::duration<f64, std::milli>(std::chrono::steady_clock::now() - start).count();
    return count;
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
        VK_CHECK(vkWaitForFences(device, 1, &frame.inFlight, VK_TRUE, UINT64_MAX));
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
    VkResult acquire = vkAcquireNextImageKHR(device, m_swapchain->handle(), UINT64_MAX, frame.imageAcquired,
                                             VK_NULL_HANDLE, &imageIndex);
    if (acquire == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapchain();
        return stats; // the semaphore was not signalled; try again next frame
    }
    if (acquire != VK_SUCCESS && acquire != VK_SUBOPTIMAL_KHR) {
        VK_CHECK(acquire);
    }
    VK_CHECK(vkResetFences(device, 1, &frame.inFlight));

    const u32 drawCount = cullAndWriteDraws(frame, camera, stats);

    const bool screenshot = !m_screenshotPath.empty();
    stats.uploadedBytes = recordFrame(frame, imageIndex, drawCount, camera, screenshot);

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

VkDeviceSize Renderer::recordFrame(Frame& frame, u32 imageIndex, u32 drawCount, const Camera& camera, bool screenshot)
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
        .clearValue = {.color = {{kSky[0], kSky[1], kSky[2], 1.0f}}},
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
    if (drawCount > 0 && m_textureImage != VK_NULL_HANDLE) {
        const VkViewport viewport{0.0f, 0.0f, static_cast<f32>(extent.width), static_cast<f32>(extent.height),
                                  0.0f, 1.0f};
        const VkRect2D scissor{{0, 0}, extent};
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        const f32 aspect = static_cast<f32>(extent.width) / static_cast<f32>(extent.height);
        const PushConstants pc{
            .viewProj = camera.projection(aspect) * camera.viewRotation(),
            .cameraBlock = glm::ivec4{camera.cameraBlock(), 0},
            .cameraFraction = glm::vec4{camera.cameraFraction(), 0.0f},
            .fog = glm::vec4{kSky[0], kSky[1], kSky[2], m_desc.fogDistance},
        };
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_set, 0, nullptr);
        vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(pc), &pc);
        vkCmdBindIndexBuffer(cmd, m_indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
        // The whole visible world in one call.
        vkCmdDrawIndexedIndirect(cmd, frame.indirect.buffer, 0, drawCount, sizeof(VkDrawIndexedIndirectCommand));
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
