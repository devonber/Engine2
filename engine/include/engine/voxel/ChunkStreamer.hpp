#pragma once

#include "engine/voxel/Mesher.hpp"

#include <compare>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace engine {
class ThreadPool;
}

namespace engine::voxel {

// Horizontal chunk position (the world is one chunk tall for now).
struct ChunkCoord {
    i32 x = 0;
    i32 z = 0;

    constexpr auto operator<=>(const ChunkCoord&) const = default;

    // Unique 64-bit key, e.g. for hash maps and the renderer.
    [[nodiscard]] constexpr u64 key() const noexcept
    {
        return (static_cast<u64>(static_cast<u32>(x)) << 32) | static_cast<u32>(z);
    }
};

// Chunk containing a world-space position (floor division, so -0.5 is chunk -1).
[[nodiscard]] ChunkCoord chunkAt(f32 worldX, f32 worldZ) noexcept;

// Recycles chunk storage. Chunks are 64 KiB; fresh allocations page-fault on
// first touch, which is slow (and serialised per process on Windows), so
// freed chunks are kept and reused. Main thread only.
class ChunkPool {
public:
    ChunkPool() = default;
    ~ChunkPool();
    ChunkPool(const ChunkPool&) = delete;
    ChunkPool& operator=(const ChunkPool&) = delete;

    // Storage for one chunk, NOT cleared (constructed with Chunk::Uninitialized).
    [[nodiscard]] Chunk* acquire();
    void release(Chunk* chunk) noexcept;

    [[nodiscard]] usize allocatedChunks() const noexcept { return m_slabs.size() * kChunksPerSlab; }
    [[nodiscard]] usize freeChunks() const noexcept { return m_free.size(); }

private:
    static constexpr usize kChunksPerSlab = 32; // 2 MiB per slab

    std::vector<Chunk*> m_slabs;
    std::vector<Chunk*> m_free;
};

struct StreamingOptions {
    i32 viewRadius = 12;     // chunks within this distance (in chunks) get meshed
    bool greedy = true;      // greedy meshing (else one quad per block face)
    u32 maxJobsInFlight = 0; // 0 = 32 per pool thread (enough to last a frame)
};

// A change for the renderer. `removed` = drop the chunk's mesh.
struct ChunkMeshEvent {
    ChunkCoord coord;
    std::vector<PackedQuad> quads;
    bool removed = false;
};

// Keeps the chunks around a moving centre generated and meshed, using
// background jobs on a ThreadPool, and unloads chunks that fall behind.
//
//   generate:  within viewRadius + 1 (so every meshed chunk has neighbours)
//   mesh:      within viewRadius, once all 4 horizontal neighbours exist
//   unmesh:    beyond viewRadius + 1      (hysteresis: no flicker on borders)
//   unload:    beyond viewRadius + 2
//
// Work is scheduled nearest-first with a cap on jobs in flight, so the area
// around the player fills in first and the main thread stays responsive.
// All methods are main-thread only.
class ChunkStreamer {
public:
    ChunkStreamer(ThreadPool& pool, const StreamingOptions& options);
    ~ChunkStreamer(); // waits for its jobs
    ChunkStreamer(const ChunkStreamer&) = delete;
    ChunkStreamer& operator=(const ChunkStreamer&) = delete;

    // Once per frame: collect finished jobs, unload far chunks, schedule work.
    void update(ChunkCoord center);

    // Mesh changes since the last call, in completion order.
    [[nodiscard]] std::vector<ChunkMeshEvent> takeMeshEvents();

    struct Stats {
        u32 loadedChunks = 0; // generated chunk data in memory
        u32 meshedChunks = 0; // meshes handed to the renderer
        u32 jobsInFlight = 0;
        usize poolChunks = 0; // chunk storage allocated (in use + free)
    };
    [[nodiscard]] Stats stats() const;

    // True when everything around the last centre is generated and meshed.
    [[nodiscard]] bool settled() const noexcept { return m_settled; }

private:
    enum class State : u8 { Generating, Ready };

    struct Entry {
        Chunk* chunk = nullptr;
        State state = State::Generating;
        bool meshing = false; // mesh job in flight
        bool meshed = false;  // renderer has (or is about to get) a mesh
        OuterLayers outer;    // solidity of the chunk's own outer layers (valid once Ready)
    };

    struct Completion {
        ChunkCoord coord;
        bool isMesh = false;
        std::vector<PackedQuad> quads; // mesh jobs
        OuterLayers outer;             // generate jobs
    };

    void collectCompletions();
    void unloadFar();
    void schedule();
    void submitGenerate(ChunkCoord coord);
    void submitMesh(ChunkCoord coord, Entry& entry);
    void complete(Completion&& completion);
    [[nodiscard]] Entry* find(ChunkCoord coord);
    [[nodiscard]] i64 distanceSq(ChunkCoord coord) const noexcept;

    ThreadPool& m_pool;
    StreamingOptions m_options;
    u32 m_maxInFlight = 0;
    ChunkPool m_chunkPool;
    std::unordered_map<u64, Entry> m_entries;
    std::vector<std::pair<i32, i32>> m_offsets; // (dx, dz) within viewRadius + 1, nearest first
    ChunkCoord m_center;
    bool m_hasCenter = false;
    bool m_settled = false;
    u32 m_inFlight = 0; // jobs submitted and not yet collected (main thread)
    u32 m_meshed = 0;

    std::vector<ChunkMeshEvent> m_events;

    // Written by jobs, drained by the main thread. A finishing job signals
    // m_jobsDone while still holding the mutex, so once the destructor sees
    // m_running == 0 no job touches this object again.
    void finish(Completion&& completion);
    std::mutex m_completionMutex;
    std::condition_variable m_jobsDone;
    std::vector<Completion> m_completions;
    u32 m_running = 0; // jobs that have not finished executing (guarded by m_completionMutex)
};

} // namespace engine::voxel
