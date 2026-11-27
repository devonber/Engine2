#pragma once

#include "engine/core/Types.hpp"
#include "engine/render/VulkanContext.hpp"
#include "engine/voxel/Mesher.hpp"

#include <deque>
#include <glm/glm.hpp>
#include <span>
#include <unordered_map>
#include <vector>

namespace engine::render {

// GPU storage for chunk meshes that come and go while the game runs.
//
// - One big quad buffer, divided up with a VMA virtual allocator; each chunk
//   owns a range. One origins buffer indexed by chunk slot (firstInstance).
// - New meshes are queued and copied in through a per-frame staging buffer at
//   the start of a frame's command buffer, so a chunk becomes drawable in the
//   frame its upload is recorded. A per-frame byte budget spreads bursts of
//   new chunks over several frames instead of causing a hitch.
// - A removed or replaced mesh's range and slot are freed only once every
//   frame that could still read them has finished on the GPU.
class ChunkMeshStore {
public:
    struct Slot {
        glm::ivec3 origin{0};
        u32 firstQuad = 0;
        u32 quadCount = 0;
        u32 generation = 0; // bumped on every set(); cancels stale pending uploads
        bool used = false;
        bool drawable = false; // upload recorded
        VmaVirtualAllocation allocation = VK_NULL_HANDLE;
    };

    struct Stats {
        u32 chunks = 0;
        u64 quads = 0;
        u64 usedBytes = 0;
        u64 capacityBytes = 0;
        u64 pendingBytes = 0; // queued, not yet uploaded
        u32 failed = 0;       // meshes rejected because the quad buffer was full
    };

    ChunkMeshStore(VulkanContext& ctx, u64 quadCapacity, u32 maxChunks, u32 framesInFlight,
                   VkDeviceSize uploadBudgetPerFrame);
    ~ChunkMeshStore();
    ChunkMeshStore(const ChunkMeshStore&) = delete;
    ChunkMeshStore& operator=(const ChunkMeshStore&) = delete;

    // Add or replace a chunk's mesh. False if there is no room (the chunk
    // is then not drawn). Call between frames.
    bool set(u64 key, glm::ivec3 origin, std::vector<voxel::PackedQuad>&& quads);
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
    [[nodiscard]] VkBuffer originBuffer() const noexcept { return m_origins.buffer; }
    [[nodiscard]] Stats stats() const;

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

    VulkanContext& m_ctx;
    u32 m_framesInFlight;
    VkDeviceSize m_uploadBudget;

    Buffer m_quads;
    Buffer m_origins;
    VmaVirtualBlock m_quadBlock = VK_NULL_HANDLE;
    std::vector<Buffer> m_staging; // one per frame slot

    std::vector<Slot> m_slots;
    u32 m_highWater = 0; // slots [0, m_highWater) may be in use
    std::vector<u32> m_freeSlots;
    std::unordered_map<u64, u32> m_slotOf;

    std::deque<Pending> m_pending;
    u64 m_pendingBytes = 0;
    std::vector<std::vector<Deferred>> m_deferred; // per frame slot
    u32 m_nextFrameSlot = 0;

    u32 m_chunks = 0;
    u64 m_quadCount = 0;
    u32 m_failed = 0;
};

} // namespace engine::render
