#include "engine/voxel/World.hpp"

#include "engine/core/Profile.hpp"
#include "engine/core/ThreadPool.hpp"
#include "engine/voxel/Generate.hpp"

#include <chrono>
#include <memory>
#include <new>

namespace engine::voxel {

namespace {

template <typename Fn>
void forEach(ThreadPool* pool, u32 count, Fn&& fn)
{
    if (pool != nullptr) {
        pool->parallelFor(count, fn);
    } else {
        for (u32 i = 0; i < count; ++i) {
            fn(i);
        }
    }
}

// Uninitialised, 64-byte aligned array of chunks. Chunk is trivially
// destructible, so freeing the memory is all the cleanup needed.
class ChunkArray {
public:
    explicit ChunkArray(u32 count)
        : m_data(static_cast<Chunk*>(::operator new(sizeof(Chunk) * count, std::align_val_t{64})))
    {
    }
    ~ChunkArray() { ::operator delete(m_data, std::align_val_t{64}); }
    ChunkArray(const ChunkArray&) = delete;
    ChunkArray& operator=(const ChunkArray&) = delete;

    Chunk& operator[](usize i) const noexcept { return m_data[i]; }

private:
    Chunk* m_data;
};

f64 msSince(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<f64, std::milli>(std::chrono::steady_clock::now() - start).count();
}

} // namespace

WorldMesh buildTerrainWorld(const WorldBuildOptions& options, ThreadPool* pool)
{
    ENGINE_PROFILE_FUNCTION();
    const i32 radius = options.radius;
    const i32 side = 2 * radius;
    const u32 count = static_cast<u32>(side * side);
    WorldMesh world;

    // Chunk i sits at (i % side - radius, i / side - radius).
    const auto coords = [&](u32 i) {
        return std::pair{static_cast<i32>(i % static_cast<u32>(side)) - radius,
                         static_cast<i32>(i / static_cast<u32>(side)) - radius};
    };

    // Phase 1: generate every chunk (independent, so trivially parallel).
    auto start = std::chrono::steady_clock::now();
    // Raw storage, constructed inside the parallel loop: no serial 64 KiB zero
    // fill per chunk (fillTerrain writes every voxel), and each thread takes
    // the first-touch page faults for its own chunks instead of one thread for all.
    const ChunkArray chunks(count);
    forEach(pool, count, [&](u32 i) {
        const auto [cx, cz] = coords(i);
        Chunk* chunk = std::construct_at(&chunks[i], Chunk::Uninitialized{});
        fillTerrain(*chunk, cx, cz);
    });
    world.generateMs = msSince(start);

    // Phase 2: mesh every chunk against its (now read-only) neighbours. Each
    // chunk writes its own vector, so the result does not depend on timing.
    start = std::chrono::steady_clock::now();
    const auto at = [&](i32 cx, i32 cz) -> const Chunk* {
        if (cx < -radius || cx >= radius || cz < -radius || cz >= radius) {
            return nullptr; // outside the world: air
        }
        return &chunks[static_cast<usize>((cz + radius) * side + (cx + radius))];
    };
    std::vector<std::vector<PackedQuad>> perChunk(count);
    forEach(pool, count, [&](u32 i) {
        const auto [cx, cz] = coords(i);
        BoundaryMasks masks;
        if (options.borderCulling) {
            ChunkNeighbours nb;
            nb.chunks[static_cast<usize>(Face::PosX)] = at(cx + 1, cz);
            nb.chunks[static_cast<usize>(Face::NegX)] = at(cx - 1, cz);
            nb.chunks[static_cast<usize>(Face::PosZ)] = at(cx, cz + 1);
            nb.chunks[static_cast<usize>(Face::NegZ)] = at(cx, cz - 1);
            masks = buildBoundaryMasks(nb);
            masks.setSolid(Face::NegY); // nothing is ever seen below the world
        }
        // Mesh into a per-thread scratch buffer whose capacity survives between
        // chunks, then copy out with a single exact-size allocation, instead of
        // growing every chunk's vector from empty.
        thread_local std::vector<PackedQuad> scratch;
        if (options.greedy) {
            meshChunkGreedy(chunks[i], masks, scratch);
        } else {
            meshChunkBinary(chunks[i], masks, scratch);
        }
        perChunk[i].assign(scratch.begin(), scratch.end());
    });

    // Concatenate in chunk order.
    usize total = 0;
    for (const auto& q : perChunk) {
        total += q.size();
    }
    world.quads.reserve(total);
    world.chunks.reserve(count);
    for (u32 i = 0; i < count; ++i) {
        const auto [cx, cz] = coords(i);
        world.chunks.push_back({cx, cz, static_cast<u32>(world.quads.size()), static_cast<u32>(perChunk[i].size())});
        world.quads.insert(world.quads.end(), perChunk[i].begin(), perChunk[i].end());
    }
    world.meshMs = msSince(start);
    return world;
}

} // namespace engine::voxel
