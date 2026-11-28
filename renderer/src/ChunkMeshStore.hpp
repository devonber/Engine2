#pragma once

#include "engine/core/Types.hpp"
#include "engine/render/VulkanContext.hpp"
#include "engine/voxel/Mesher.hpp"

#include <array>
#include <deque>
#include <glm/glm.hpp>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace engine::render {

// GPU storage for chunk meshes that come and go while the game runs.
//
// - One big quad buffer, divided up with a VMA virtual allocator; each chunk
//   owns a range. One chunk table on the GPU, indexed by chunk slot
//   (firstInstance), with what the culling compute shader and the vertex
//   shader need to know about each chunk (GpuChunk).
// - New meshes are queued and copied in through a per-frame staging buffer at
//   the start of a frame's command buffer, so a chunk becomes drawable in the
//   frame its upload is recorded. A per-frame byte budget spreads bursts of
//   new chunks over several frames instead of causing a hitch.
// - A removed or replaced mesh's range and slot are freed only once every
//   frame that could still read them has finished on the GPU.
// One chunk's record in the GPU chunk table (std430: 48 bytes).
struct GpuChunk {
    glm::ivec4 originFirst; // xyz: world position (blocks) of voxel (0,0,0); w: first quad in the quad buffer
    glm::uvec4 faces0;      // faceStart[0..3]
    glm::uvec4 faces1;      // xyz: faceStart[4..6] (= quad count); w: bit 0 drawable, bits 1+: level of detail
};
static_assert(sizeof(GpuChunk) == 48);

class ChunkMeshStore {
public:
    struct Slot {
        glm::ivec3 origin{0};
        u32 firstQuad = 0;
        u32 quadCount = 0;
        u32 generation = 0; // bumped on every set(); cancels stale pending uploads
        u32 level = 0;      // level of detail: voxels are 2^level blocks
        // Quads are stored grouped by face direction (voxel::Face order):
        // direction f is quads [faceStart[f], faceStart[f + 1]) of this mesh,
        // so draws can skip directions that face away from the camera.
        std::array<u32, voxel::kFaceCount + 1> faceStart{};
        bool used = false;
        bool drawable = false; // upload recorded
        bool gpuDirty = false; // GPU chunk record needs rewriting
        VmaVirtualAllocation allocation = VK_NULL_HANDLE;
    };

    struct Stats {
        u32 chunks = 0;
        u64 quads = 0;
        u64 usedBytes = 0;
        u64 capacityBytes = 0;
        u64 pendingBytes = 0; // queued, not yet uploaded
        u32 failed = 0;       // meshes rejected: no free chunk slot
        u32 grows = 0;        // times the quad buffer was enlarged
    };

    ChunkMeshStore(VulkanContext& ctx, u64 quadCapacity, u32 maxChunks, u32 framesInFlight,
                   VkDeviceSize uploadBudgetPerFrame);
    ~ChunkMeshStore();
    ChunkMeshStore(const ChunkMeshStore&) = delete;
    ChunkMeshStore& operator=(const ChunkMeshStore&) = delete;

    // Add or replace a chunk's mesh. Grows the quad buffer when it is full
    // (stalls the GPU once). False only if every chunk slot is in use.
    // Call between frames.
    bool set(u64 key, glm::ivec3 origin, std::span<const voxel::PackedQuad> quads, u32 level = 0);
    void remove(u64 key);

    // Frame protocol, called by the renderer:
    //   beginFrame(slot)  after waiting for the frame slot's fence: releases
    //                     memory whose last reader has now finished
    //   recordUploads()   before rendering, in the frame's command buffer
    //   endFrame(next)    with the slot of the next frame to be recorded
    void beginFrame(u32 frameSlot);
    VkDeviceSize recordUploads(VkCommandBuffer cmd, u32 frameSlot);
    void endFrame(u32 nextFrameSlot) noexcept { m_nextFrameSlot = nextFrameSlot; }

    [[nodiscard]] std::span<const Slot> slots() const noexcept { return {m_slots.data(), m_highWater}; }
    [[nodiscard]] u32 maxChunks() const noexcept { return static_cast<u32>(m_slots.size()); }
    [[nodiscard]] VkBuffer quadBuffer() const noexcept { return m_quads.buffer; }
    [[nodiscard]] VkBuffer chunkTable() const noexcept { return m_chunkTable.buffer; }
    // Slots [0, slotCount()) may be in use: the culling dispatch size.
    [[nodiscard]] u32 slotCount() const noexcept { return m_highWater; }
    [[nodiscard]] Stats stats() const;
    // Changes whenever a full-detail (level 0) chunk becomes drawable or is
    // removed: what shadow casters are. Cached shadow maps compare it.
    [[nodiscard]] u64 fullDetailVersion() const noexcept { return m_fullDetailVersion; }

    // True once after the quad buffer was replaced by a bigger one: the
    // renderer must point its descriptor at quadBuffer() again.
    [[nodiscard]] bool consumeBufferChanged() noexcept { return std::exchange(m_bufferChanged, false); }

private:
    struct Pending {
        u32 slot;
        u32 generation;
        std::vector<voxel::PackedQuad> quads;
    };
    struct Deferred {
        u32 slot;
        VmaVirtualAllocation allocation;
    };

    void releaseSlot(u32 slot); // defer freeing until no in-flight frame uses it
    void markDirty(u32 slot);
    [[nodiscard]] GpuChunk gpuRecord(const Slot& slot) const noexcept;
    // Replaces the quad buffer with one of at least `minQuads` capacity and
    // re-packs every chunk's range into it. Waits for the GPU.
    void grow(u64 minQuads);

    VulkanContext& m_ctx;
    u32 m_framesInFlight;
    VkDeviceSize m_uploadBudget;

    Buffer m_quads;
    Buffer m_chunkTable;
    VmaVirtualBlock m_quadBlock = VK_NULL_HANDLE;
    std::vector<Buffer> m_staging; // one per frame slot

    std::vector<Slot> m_slots;
    u32 m_highWater = 0; // slots [0, m_highWater) may be in use
    std::vector<u32> m_freeSlots;
    std::unordered_map<u64, u32> m_slotOf;

    std::deque<Pending> m_pending;
    std::vector<u32> m_dirty; // slots whose GPU record changes
    u64 m_pendingBytes = 0;
    std::vector<std::vector<Deferred>> m_deferred; // per frame slot
    u32 m_nextFrameSlot = 0;

    u32 m_chunks = 0;
    u64 m_quadCount = 0;
    u32 m_failed = 0;
    u32 m_grows = 0;
    bool m_bufferChanged = false;
    u64 m_fullDetailVersion = 0;
};

} // namespace engine::render
