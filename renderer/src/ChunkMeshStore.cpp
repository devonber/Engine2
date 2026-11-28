#include "ChunkMeshStore.hpp"

#include "engine/core/Profile.hpp"

#include <algorithm>
#include <cstring>

namespace engine::render {

namespace {
constexpr VkDeviceSize kQuadBytes = sizeof(voxel::PackedQuad);
constexpr VkDeviceSize kOriginBytes = sizeof(glm::ivec4);
// The densest possible chunk (3D checkerboard) must fit in one frame's upload.
constexpr VkDeviceSize kMaxChunkMeshBytes = voxel::kChunkVolume / 2 * 6 * kQuadBytes;
} // namespace

ChunkMeshStore::ChunkMeshStore(VulkanContext& ctx, u64 quadCapacity, u32 maxChunks, u32 framesInFlight,
                               VkDeviceSize uploadBudgetPerFrame)
    : m_ctx(ctx), m_framesInFlight(framesInFlight),
      m_uploadBudget(std::max(uploadBudgetPerFrame, kMaxChunkMeshBytes + kOriginBytes))
{
    VmaAllocator allocator = ctx.allocator();
    m_quads = createBuffer(allocator, quadCapacity * kQuadBytes,
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, MemoryKind::GpuOnly);
    m_origins =
        createBuffer(allocator, static_cast<VkDeviceSize>(maxChunks) * kOriginBytes,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, MemoryKind::GpuOnly);

    const VmaVirtualBlockCreateInfo blockInfo{.size = quadCapacity * kQuadBytes};
    VK_CHECK(vmaCreateVirtualBlock(&blockInfo, &m_quadBlock));

    m_staging.resize(framesInFlight);
    for (Buffer& b : m_staging) {
        b = createBuffer(allocator, m_uploadBudget, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, MemoryKind::Upload);
    }
    m_slots.resize(maxChunks);
    m_deferred.resize(framesInFlight);
}

ChunkMeshStore::~ChunkMeshStore()
{
    // The renderer waits for the GPU before destroying this.
    vmaClearVirtualBlock(m_quadBlock);
    vmaDestroyVirtualBlock(m_quadBlock);
    VmaAllocator allocator = m_ctx.allocator();
    for (Buffer& b : m_staging) {
        destroyBuffer(allocator, b);
    }
    destroyBuffer(allocator, m_quads);
    destroyBuffer(allocator, m_origins);
}

bool ChunkMeshStore::set(u64 key, glm::ivec3 origin, std::span<const voxel::PackedQuad> quads)
{
    if (const auto it = m_slotOf.find(key); it != m_slotOf.end()) {
        releaseSlot(it->second); // the old mesh may still be drawn by in-flight frames
        m_slotOf.erase(it);
    }

    u32 index = 0;
    if (!m_freeSlots.empty()) {
        index = m_freeSlots.back();
        m_freeSlots.pop_back();
    } else if (m_highWater < m_slots.size()) {
        index = m_highWater++;
    } else {
        ++m_failed;
        return false;
    }

    Slot& slot = m_slots[index];
    slot.allocation = VK_NULL_HANDLE;
    slot.firstQuad = 0;
    if (!quads.empty()) {
        const VmaVirtualAllocationCreateInfo allocInfo{.size = quads.size() * kQuadBytes, .alignment = kQuadBytes};
        VkDeviceSize offset = 0;
        if (vmaVirtualAllocate(m_quadBlock, &allocInfo, &slot.allocation, &offset) != VK_SUCCESS) {
            m_freeSlots.push_back(index); // never seen by the GPU: reusable at once
            ++m_failed;
            return false;
        }
        slot.firstQuad = static_cast<u32>(offset / kQuadBytes);
    }
    slot.origin = origin;
    slot.quadCount = static_cast<u32>(quads.size());
    slot.used = true;
    slot.drawable = false;
    ++slot.generation;

    m_slotOf.emplace(key, index);
    ++m_chunks;
    m_quadCount += slot.quadCount;
    if (!quads.empty()) {
        m_pendingBytes += quads.size() * kQuadBytes;
        m_pending.push_back({index, slot.generation, std::vector<voxel::PackedQuad>(quads.begin(), quads.end())});
    }
    return true;
}

void ChunkMeshStore::remove(u64 key)
{
    if (const auto it = m_slotOf.find(key); it != m_slotOf.end()) {
        releaseSlot(it->second);
        m_slotOf.erase(it);
    }
}

void ChunkMeshStore::releaseSlot(u32 index)
{
    Slot& slot = m_slots[index];
    --m_chunks;
    m_quadCount -= slot.quadCount;
    slot.used = false;
    slot.drawable = false;
    ++slot.generation; // a pending upload for this slot is now stale

    // Frames already recorded may still read this range and slot. The frame
    // slot that comes round again just before the next frame (k + N - 1) is
    // waited on only after frame k - 1, the last one that can use it, is done.
    const u32 when = (m_nextFrameSlot + m_framesInFlight - 1) % m_framesInFlight;
    m_deferred[when].push_back({index, slot.allocation});
    slot.allocation = VK_NULL_HANDLE;
}

void ChunkMeshStore::beginFrame(u32 frameSlot)
{
    for (const Deferred& d : m_deferred[frameSlot]) {
        if (d.allocation != VK_NULL_HANDLE) {
            vmaVirtualFree(m_quadBlock, d.allocation);
        }
        m_freeSlots.push_back(d.slot);
    }
    m_deferred[frameSlot].clear();
}

VkDeviceSize ChunkMeshStore::recordUploads(VkCommandBuffer cmd, u32 frameSlot)
{
    ENGINE_PROFILE_FUNCTION();
    Buffer& staging = m_staging[frameSlot];
    auto* dst = static_cast<u8*>(staging.mapped);
    VkDeviceSize offset = 0;
    std::vector<VkBufferCopy> quadCopies;
    std::vector<VkBufferCopy> originCopies;

    while (!m_pending.empty()) {
        Pending& p = m_pending.front();
        const VkDeviceSize bytes = p.quads.size() * kQuadBytes;
        Slot& slot = m_slots[p.slot];
        if (!slot.used || slot.generation != p.generation) {
            m_pendingBytes -= bytes; // removed or replaced before it was uploaded
            m_pending.pop_front();
            continue;
        }
        if (offset + bytes + kOriginBytes > m_uploadBudget) {
            break; // the rest goes next frame
        }
        std::memcpy(dst + offset, p.quads.data(), bytes);
        quadCopies.push_back({offset, slot.firstQuad * kQuadBytes, bytes});
        offset += bytes;

        offset = (offset + kOriginBytes - 1) & ~(kOriginBytes - 1);
        const glm::ivec4 origin{slot.origin, 0};
        std::memcpy(dst + offset, &origin, kOriginBytes);
        originCopies.push_back({offset, p.slot * kOriginBytes, kOriginBytes});
        offset += kOriginBytes;

        slot.drawable = true; // copied before this frame's draws (barrier below)
        m_pendingBytes -= bytes;
        m_pending.pop_front();
    }

    if (quadCopies.empty()) {
        return 0;
    }
    VK_CHECK(vmaFlushAllocation(m_ctx.allocator(), staging.allocation, 0, offset));

    // The ranges written here are not read by any frame still in flight (new
    // allocations, or freed after their last reader finished). But draws bind
    // the whole buffer, so in Vulkan's terms the previous frame's vertex shader
    // reads all of it: order the copies after it (execution-only, no cache
    // flushes). Only costs anything on frames that upload.
    const VkMemoryBarrier2 afterReads{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
    };
    const VkDependencyInfo beforeCopy{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &afterReads,
    };
    vkCmdPipelineBarrier2(cmd, &beforeCopy);

    vkCmdCopyBuffer(cmd, staging.buffer, m_quads.buffer, static_cast<u32>(quadCopies.size()), quadCopies.data());
    vkCmdCopyBuffer(cmd, staging.buffer, m_origins.buffer, static_cast<u32>(originCopies.size()), originCopies.data());

    const VkMemoryBarrier2 barrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
        .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
    };
    const VkDependencyInfo dep{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &barrier,
    };
    vkCmdPipelineBarrier2(cmd, &dep);
    return offset;
}

ChunkMeshStore::Stats ChunkMeshStore::stats() const
{
    VmaStatistics vs{};
    vmaGetVirtualBlockStatistics(m_quadBlock, &vs);
    return {m_chunks, m_quadCount, vs.allocationBytes, vs.blockBytes, m_pendingBytes, m_failed};
}

} // namespace engine::render
