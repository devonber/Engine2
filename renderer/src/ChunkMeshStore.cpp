#include "ChunkMeshStore.hpp"

#include "engine/core/Profile.hpp"

#include <algorithm>
#include <cstring>

namespace engine::render {

namespace {
constexpr VkDeviceSize kQuadBytes = sizeof(voxel::PackedQuad);
constexpr VkDeviceSize kRecordBytes = sizeof(GpuChunk);
// The densest possible chunk (3D checkerboard) must fit in one frame's upload.
constexpr VkDeviceSize kMaxChunkMeshBytes = voxel::kChunkVolume / 2 * 6 * kQuadBytes;
} // namespace

ChunkMeshStore::ChunkMeshStore(VulkanContext& ctx, u64 quadCapacity, u32 maxChunks, u32 framesInFlight,
                               VkDeviceSize uploadBudgetPerFrame)
    : m_ctx(ctx), m_framesInFlight(framesInFlight),
      m_uploadBudget(std::max(uploadBudgetPerFrame, kMaxChunkMeshBytes + kRecordBytes))
{
    VmaAllocator allocator = ctx.allocator();
    m_quads = createBuffer(allocator, quadCapacity * kQuadBytes,
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT, // source when growing
                           MemoryKind::GpuOnly);
    // Zero-filled at start, so every record reads "not drawable".
    std::vector<GpuChunk> empty(maxChunks, GpuChunk{});
    m_chunkTable = createBufferWithData(ctx, empty.data(), empty.size() * sizeof(GpuChunk),
                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    const VmaVirtualBlockCreateInfo blockInfo{.size = quadCapacity * kQuadBytes};
    VK_CHECK(vmaCreateVirtualBlock(&blockInfo, &m_quadBlock));

    // Staging: the mesh-data budget, plus room for every slot's GPU record so
    // records (tiny, but possibly many after a mass unload) always fit.
    m_staging.resize(framesInFlight);
    for (Buffer& b : m_staging) {
        b = createBuffer(allocator, m_uploadBudget + 16 + static_cast<VkDeviceSize>(maxChunks) * kRecordBytes,
                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT, MemoryKind::Upload);
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
    destroyBuffer(allocator, m_chunkTable);
}

bool ChunkMeshStore::set(u64 key, glm::ivec3 origin, std::span<const voxel::PackedQuad> quads, u32 level)
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
            // Full (or too fragmented): double the buffer and try again.
            grow(std::max<u64>(2 * (m_quads.size / kQuadBytes), m_quadCount + quads.size()));
            VK_CHECK(vmaVirtualAllocate(m_quadBlock, &allocInfo, &slot.allocation, &offset));
        }
        slot.firstQuad = static_cast<u32>(offset / kQuadBytes);
    }

    // Group the quads by face direction (counting sort; the meshers already
    // emit them in this order, so usually this is a straight copy).
    std::array<u32, voxel::kFaceCount + 1> start{};
    for (const voxel::PackedQuad& q : quads) {
        ++start[static_cast<usize>(q.face()) + 1];
    }
    for (usize f = 0; f < voxel::kFaceCount; ++f) {
        start[f + 1] += start[f];
    }
    std::vector<voxel::PackedQuad> sorted(quads.size());
    std::array<u32, voxel::kFaceCount + 1> next = start;
    for (const voxel::PackedQuad& q : quads) {
        sorted[next[static_cast<usize>(q.face())]++] = q;
    }

    slot.origin = origin;
    slot.level = level;
    slot.quadCount = static_cast<u32>(quads.size());
    slot.faceStart = start;
    slot.used = true;
    slot.drawable = false;
    ++slot.generation;

    m_slotOf.emplace(key, index);
    ++m_chunks;
    m_quadCount += slot.quadCount;
    if (!quads.empty()) {
        m_pendingBytes += quads.size() * kQuadBytes;
        m_pending.push_back({index, slot.generation, std::move(sorted)});
    }
    return true;
}

void ChunkMeshStore::grow(u64 minQuads)
{
    ENGINE_PROFILE_FUNCTION();
    VkDevice device = m_ctx.device();
    vkDeviceWaitIdle(device); // nothing in flight reads the old buffer after this

    // Everything deferred can go now that the GPU is idle.
    for (u32 f = 0; f < m_framesInFlight; ++f) {
        beginFrame(f);
    }

    const u64 newQuads = std::max<u64>(minQuads, 2 * (m_quads.size / kQuadBytes));
    Buffer bigger = createBuffer(m_ctx.allocator(), newQuads * kQuadBytes,
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                 MemoryKind::GpuOnly);
    VmaVirtualBlock block = VK_NULL_HANDLE;
    const VmaVirtualBlockCreateInfo blockInfo{.size = newQuads * kQuadBytes};
    VK_CHECK(vmaCreateVirtualBlock(&blockInfo, &block));

    // Re-pack every live range into the new buffer (compacts it as a bonus).
    std::vector<VkBufferCopy> copies;
    for (u32 i = 0; i < m_highWater; ++i) {
        Slot& slot = m_slots[i];
        if (!slot.used || slot.allocation == VK_NULL_HANDLE) {
            continue;
        }
        const VkDeviceSize bytes = static_cast<VkDeviceSize>(slot.quadCount) * kQuadBytes;
        const VmaVirtualAllocationCreateInfo allocInfo{.size = bytes, .alignment = kQuadBytes};
        VkDeviceSize offset = 0;
        VK_CHECK(vmaVirtualAllocate(block, &allocInfo, &slot.allocation, &offset));
        if (slot.drawable) { // not-yet-uploaded meshes are copied in later, to the new place
            copies.push_back({static_cast<VkDeviceSize>(slot.firstQuad) * kQuadBytes, offset, bytes});
        }
        slot.firstQuad = static_cast<u32>(offset / kQuadBytes);
    }
    if (!copies.empty()) {
        m_ctx.immediateSubmit([&](VkCommandBuffer cmd) {
            vkCmdCopyBuffer(cmd, m_quads.buffer, bigger.buffer, static_cast<u32>(copies.size()), copies.data());
        });
    }

    // Every chunk's first quad moved: rewrite the whole GPU chunk table now
    // (the GPU is idle), so no frame ever draws with stale offsets.
    {
        std::vector<GpuChunk> table(m_slots.size(), GpuChunk{});
        for (u32 i = 0; i < m_highWater; ++i) {
            table[i] = gpuRecord(m_slots[i]);
            m_slots[i].gpuDirty = false;
        }
        m_dirty.clear();
        Buffer staging = createBuffer(m_ctx.allocator(), table.size() * sizeof(GpuChunk),
                                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT, MemoryKind::Upload);
        std::memcpy(staging.mapped, table.data(), table.size() * sizeof(GpuChunk));
        VK_CHECK(vmaFlushAllocation(m_ctx.allocator(), staging.allocation, 0, VK_WHOLE_SIZE));
        m_ctx.immediateSubmit([&](VkCommandBuffer cmd) {
            const VkBufferCopy region{.size = table.size() * sizeof(GpuChunk)};
            vkCmdCopyBuffer(cmd, staging.buffer, m_chunkTable.buffer, 1, &region);
        });
        destroyBuffer(m_ctx.allocator(), staging);
    }

    vmaClearVirtualBlock(m_quadBlock);
    vmaDestroyVirtualBlock(m_quadBlock);
    destroyBuffer(m_ctx.allocator(), m_quads);
    m_quads = bigger;
    m_quadBlock = block;
    m_bufferChanged = true;
    ++m_grows;
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
    if (slot.drawable && slot.level == 0) {
        ++m_fullDetailVersion;
    }
    --m_chunks;
    m_quadCount -= slot.quadCount;
    slot.used = false;
    slot.drawable = false;
    ++slot.generation; // a pending upload for this slot is now stale
    markDirty(index);  // the GPU stops drawing it from the next frame on

    // Frames already recorded may still read this range and slot. The frame
    // slot that comes round again just before the next frame (k + N - 1) is
    // waited on only after frame k - 1, the last one that can use it, is done.
    const u32 when = (m_nextFrameSlot + m_framesInFlight - 1) % m_framesInFlight;
    m_deferred[when].push_back({index, slot.allocation});
    slot.allocation = VK_NULL_HANDLE;
}

void ChunkMeshStore::markDirty(u32 index)
{
    if (!m_slots[index].gpuDirty) {
        m_slots[index].gpuDirty = true;
        m_dirty.push_back(index);
    }
}

GpuChunk ChunkMeshStore::gpuRecord(const Slot& slot) const noexcept
{
    const auto& f = slot.faceStart;
    const u32 drawable = slot.used && slot.drawable && slot.quadCount > 0 ? 1u : 0u;
    return GpuChunk{
        .originFirst = glm::ivec4{slot.origin, static_cast<i32>(slot.firstQuad)},
        .faces0 = glm::uvec4{f[0], f[1], f[2], f[3]},
        .faces1 = glm::uvec4{f[4], f[5], f[6], drawable | (slot.level << 1)},
    };
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
    std::vector<VkBufferCopy> recordCopies;

    // Mesh data up to the budget; GPU records after it in their own reserved
    // space, so a chunk's record is always written in the frame its quads are.
    while (!m_pending.empty()) {
        Pending& p = m_pending.front();
        const VkDeviceSize bytes = p.quads.size() * kQuadBytes;
        Slot& slot = m_slots[p.slot];
        if (!slot.used || slot.generation != p.generation) {
            m_pendingBytes -= bytes; // removed or replaced before it was uploaded
            m_pending.pop_front();
            continue;
        }
        if (offset + bytes > m_uploadBudget) {
            break; // the rest goes next frame
        }
        std::memcpy(dst + offset, p.quads.data(), bytes);
        quadCopies.push_back({offset, slot.firstQuad * kQuadBytes, bytes});
        offset += bytes;

        slot.drawable = true; // copied before this frame's culling and draws (barrier below)
        if (slot.level == 0) {
            ++m_fullDetailVersion;
        }
        markDirty(p.slot);
        m_pendingBytes -= bytes;
        m_pending.pop_front();
    }

    // GPU chunk records: newly drawable and removed slots.
    offset = (offset + 15) & ~VkDeviceSize{15};
    for (const u32 index : m_dirty) {
        Slot& slot = m_slots[index];
        slot.gpuDirty = false;
        const GpuChunk record = gpuRecord(slot);
        std::memcpy(dst + offset, &record, kRecordBytes);
        recordCopies.push_back({offset, index * kRecordBytes, kRecordBytes});
        offset += kRecordBytes;
    }
    m_dirty.clear();

    if (quadCopies.empty() && recordCopies.empty()) {
        return 0;
    }
    VK_CHECK(vmaFlushAllocation(m_ctx.allocator(), staging.allocation, 0, offset));

    // The ranges written here are not read by any frame still in flight (new
    // allocations, or freed after their last reader finished). But culling
    // and draws bind whole buffers, so in Vulkan's terms the previous frame's
    // compute and vertex shaders read all of them: order the copies after
    // those (execution-only, no cache flushes). Only on frames that upload.
    const VkMemoryBarrier2 afterReads{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
    };
    const VkDependencyInfo beforeCopy{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &afterReads,
    };
    vkCmdPipelineBarrier2(cmd, &beforeCopy);

    if (!quadCopies.empty()) {
        vkCmdCopyBuffer(cmd, staging.buffer, m_quads.buffer, static_cast<u32>(quadCopies.size()), quadCopies.data());
    }
    if (!recordCopies.empty()) {
        vkCmdCopyBuffer(cmd, staging.buffer, m_chunkTable.buffer, static_cast<u32>(recordCopies.size()),
                        recordCopies.data());
    }

    const VkMemoryBarrier2 barrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
        .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
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
    return {m_chunks, m_quadCount, vs.allocationBytes, vs.blockBytes, m_pendingBytes, m_failed, m_grows};
}

} // namespace engine::render
