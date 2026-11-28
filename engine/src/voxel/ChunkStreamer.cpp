#include "engine/voxel/ChunkStreamer.hpp"

#include "engine/core/Profile.hpp"
#include "engine/core/ThreadPool.hpp"
#include "engine/voxel/Generate.hpp"

#include <algorithm>
#include <cmath>
#include <new>
#include <utility>

namespace engine::voxel {

namespace {

constexpr std::align_val_t kChunkAlign{64};

// Horizontal neighbours.
constexpr std::pair<Face, std::pair<i32, i32>> kSideNeighbours[] = {
    {Face::PosX, {+1, 0}},
    {Face::NegX, {-1, 0}},
    {Face::PosZ, {0, +1}},
    {Face::NegZ, {0, -1}},
};

} // namespace

ChunkCoord chunkAt(f32 worldX, f32 worldZ) noexcept
{
    constexpr f32 size = static_cast<f32>(kChunkSize);
    return {static_cast<i32>(std::floor(worldX / size)), static_cast<i32>(std::floor(worldZ / size))};
}

// ---- ChunkPool ---------------------------------------------------------------------

ChunkPool::~ChunkPool()
{
    for (Chunk* slab : m_slabs) {
        ::operator delete(slab, kChunkAlign);
    }
}

Chunk* ChunkPool::acquire()
{
    if (m_free.empty()) {
        auto* slab = static_cast<Chunk*>(::operator new(sizeof(Chunk) * kChunksPerSlab, kChunkAlign));
        m_slabs.push_back(slab);
        for (usize i = kChunksPerSlab; i-- > 0;) {
            m_free.push_back(slab + i);
        }
    }
    Chunk* chunk = m_free.back();
    m_free.pop_back();
    return std::construct_at(chunk, Chunk::Uninitialized{}); // trivially destructible: nothing to destroy later
}

void ChunkPool::release(Chunk* chunk) noexcept
{
    m_free.push_back(chunk);
}

// ---- ChunkStreamer -------------------------------------------------------------------

ChunkStreamer::ChunkStreamer(ThreadPool& pool, const StreamingOptions& options) : m_pool(pool), m_options(options)
{
    m_options.viewRadius = std::max(m_options.viewRadius, 1);
    // The queue is only refilled once per frame, so it must hold enough work
    // to keep every thread busy for a whole frame (a job is ~10-50 us).
    // Queued jobs are nearest-first; stale ones after a teleport cost little.
    m_maxInFlight = m_options.maxJobsInFlight != 0 ? m_options.maxJobsInFlight : 32 * pool.concurrency();

    // Every offset within the generation radius, nearest first.
    const i32 r = m_options.viewRadius + 1;
    for (i32 dz = -r; dz <= r; ++dz) {
        for (i32 dx = -r; dx <= r; ++dx) {
            if (dx * dx + dz * dz <= r * r) {
                m_offsets.emplace_back(dx, dz);
            }
        }
    }
    std::ranges::stable_sort(m_offsets, {}, [](const auto& o) { return o.first * o.first + o.second * o.second; });
}

ChunkStreamer::~ChunkStreamer()
{
    std::unique_lock lock(m_completionMutex);
    m_jobsDone.wait(lock, [this] { return m_running == 0; });
}

ChunkStreamer::Entry* ChunkStreamer::find(ChunkCoord coord)
{
    const auto it = m_entries.find(coord.key());
    return it != m_entries.end() ? &it->second : nullptr;
}

i64 ChunkStreamer::distanceSq(ChunkCoord coord) const noexcept
{
    const i64 dx = coord.x - m_center.x;
    const i64 dz = coord.z - m_center.z;
    return dx * dx + dz * dz;
}

void ChunkStreamer::update(ChunkCoord center)
{
    ENGINE_PROFILE_FUNCTION();
    const bool moved = !m_hasCenter || center != m_center;
    m_center = center;
    m_hasCenter = true;

    const usize before = m_inFlight;
    collectCompletions();
    const bool completed = m_inFlight != before;

    if (moved || completed) {
        m_settled = false;
        unloadFar(); // finished jobs may unblock unloading of their chunks
    }
    if (!m_settled) {
        schedule();
    }
}

std::vector<ChunkMeshEvent> ChunkStreamer::takeMeshEvents()
{
    return std::exchange(m_events, {});
}

ChunkStreamer::Stats ChunkStreamer::stats() const
{
    return {static_cast<u32>(m_entries.size()), m_meshed, m_inFlight, m_chunkPool.allocatedChunks()};
}

void ChunkStreamer::collectCompletions()
{
    std::vector<Completion> done;
    {
        std::lock_guard lock(m_completionMutex);
        done.swap(m_completions);
    }
    for (Completion& c : done) {
        --m_inFlight;
        complete(std::move(c));
    }
}

void ChunkStreamer::complete(Completion&& c)
{
    Entry* entry = find(c.coord);
    ENGINE_ASSERT(entry != nullptr); // in-flight chunks are never unloaded
    if (!c.isMesh) {
        entry->state = State::Ready;
        entry->outer = c.outer;
        return;
    }

    entry->meshing = false;

    // The player may have moved on while this was meshing.
    const i64 keep = m_options.viewRadius + 1;
    if (distanceSq(c.coord) <= keep * keep) {
        entry->meshed = true;
        ++m_meshed;
        m_events.push_back({c.coord, std::move(c.quads), false});
    }
}

void ChunkStreamer::unloadFar()
{
    ENGINE_PROFILE_FUNCTION();
    const i64 unmesh = m_options.viewRadius + 1;
    const i64 unload = m_options.viewRadius + 2;
    for (auto it = m_entries.begin(); it != m_entries.end();) {
        const ChunkCoord coord{static_cast<i32>(it->first >> 32), static_cast<i32>(static_cast<u32>(it->first))};
        Entry& e = it->second;
        const i64 d2 = distanceSq(coord);
        if (e.meshed && d2 > unmesh * unmesh) {
            e.meshed = false;
            --m_meshed;
            m_events.push_back({coord, {}, true});
        }
        // Mesh jobs read only their own chunk (neighbour data travels as
        // OuterLayers), so only a chunk being generated or meshed must stay.
        if (d2 > unload * unload && e.state == State::Ready && !e.meshing) {
            m_chunkPool.release(e.chunk);
            it = m_entries.erase(it);
        } else {
            ++it;
        }
    }
}

void ChunkStreamer::schedule()
{
    ENGINE_PROFILE_FUNCTION();
    const i64 meshRadiusSq = static_cast<i64>(m_options.viewRadius) * m_options.viewRadius;
    bool workLeft = false;

    for (const auto& [dx, dz] : m_offsets) {
        const ChunkCoord coord{m_center.x + dx, m_center.z + dz};
        Entry* e = find(coord);
        if (e == nullptr) {
            workLeft = true;
            if (m_inFlight >= m_maxInFlight) {
                break;
            }
            submitGenerate(coord);
            continue;
        }
        if (e->state == State::Generating || e->meshing) {
            workLeft = true;
            continue;
        }
        if (e->meshed || static_cast<i64>(dx) * dx + static_cast<i64>(dz) * dz > meshRadiusSq) {
            continue; // done, or in the generate-only outer ring
        }
        bool neighboursReady = true;
        for (const auto& [face, d] : kSideNeighbours) {
            const Entry* n = find({coord.x + d.first, coord.z + d.second});
            neighboursReady &= n != nullptr && n->state == State::Ready;
        }
        workLeft = true;
        if (!neighboursReady) {
            continue; // they are nearer the edge: generated soon
        }
        if (m_inFlight >= m_maxInFlight) {
            break;
        }
        submitMesh(coord, *e);
    }
    m_settled = !workLeft && m_inFlight == 0;
}

void ChunkStreamer::finish(Completion&& completion)
{
    std::lock_guard lock(m_completionMutex);
    m_completions.push_back(std::move(completion));
    --m_running;
    m_jobsDone.notify_all(); // under the lock: see the header
}

void ChunkStreamer::submitGenerate(ChunkCoord coord)
{
    Chunk* chunk = m_chunkPool.acquire();
    m_entries.emplace(coord.key(), Entry{.chunk = chunk});
    ++m_inFlight;
    {
        std::lock_guard lock(m_completionMutex);
        ++m_running;
    }
    m_pool.submit([this, coord, chunk] {
        ENGINE_PROFILE_SCOPE("generate chunk");
        fillTerrain(*chunk, coord.x, coord.z); // writes every voxel
        // While the chunk is still in this core's cache: record its outer
        // layers, which is all its neighbours' meshing needs from it.
        finish({.coord = coord, .isMesh = false, .quads = {}, .outer = computeOuterLayers(*chunk)});
    });
}

void ChunkStreamer::submitMesh(ChunkCoord coord, Entry& entry)
{
    entry.meshing = true; // keeps this chunk loaded while the job reads it

    // Boundary masks from the neighbours' outer layers, built here so the job
    // never touches neighbour chunk memory (cold in cache, and possibly
    // unloaded while the job runs).
    std::array<const OuterLayers*, kFaceCount> nb{};
    for (const auto& [face, d] : kSideNeighbours) {
        nb[static_cast<usize>(face)] = &find({coord.x + d.first, coord.z + d.second})->outer;
    }
    BoundaryMasks masks = boundaryFromNeighbours(nb);
    masks.setSolid(Face::NegY); // nothing is ever seen below the world

    ++m_inFlight;
    {
        std::lock_guard lock(m_completionMutex);
        ++m_running;
    }
    m_pool.submit([this, coord, chunk = entry.chunk, masks, greedy = m_options.greedy] {
        ENGINE_PROFILE_SCOPE("mesh chunk");
        // Mesh into a per-thread scratch buffer (capacity reused), then copy
        // out with one exact-size allocation.
        thread_local std::vector<PackedQuad> scratch;
        if (greedy) {
            meshChunkGreedy(*chunk, masks, scratch);
        } else {
            meshChunkBinary(*chunk, masks, scratch);
        }
        finish({.coord = coord, .isMesh = true, .quads = std::vector<PackedQuad>(scratch.begin(), scratch.end())});
    });
}

} // namespace engine::voxel
