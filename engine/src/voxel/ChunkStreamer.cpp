#include "engine/voxel/ChunkStreamer.hpp"

#include "engine/core/Profile.hpp"
#include "engine/core/ThreadPool.hpp"
#include "engine/voxel/Blocks.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <new>
#include <utility>

namespace engine::voxel {

namespace {

constexpr std::align_val_t kChunkAlign{64};

// Shared read-only data for uniform stone chunks, so meshing one needs no
// storage of its own.
const Chunk& stoneChunk()
{
    static const Chunk chunk = [] {
        Chunk c{Chunk::Uninitialized{}};
        c.fill(blocks::Stone);
        return c;
    }();
    return chunk;
}

OuterLayers uniformOuterLayers(ChunkContent content)
{
    OuterLayers out;
    if (content == ChunkContent::Stone) {
        for (auto& face : out.rows) {
            face.fill(~0u);
        }
    }
    return out; // air: all zero
}

bool allSolid(const BoundaryMasks& masks) noexcept
{
    for (const auto& face : masks.rows) {
        for (const u32 row : face) {
            if (row != ~0u) {
                return false;
            }
        }
    }
    return true;
}

} // namespace

ColumnCoord columnAt(f64 worldX, f64 worldZ) noexcept
{
    constexpr f64 size = static_cast<f64>(kChunkSize);
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
    // Generate within viewRadius + 2: a meshed chunk (within viewRadius)
    // needs all 8 surrounding columns, and the diagonal ones are up to
    // sqrt(2) further out.
    const i32 r = m_options.viewRadius + 2;
    for (i32 dz = -r; dz <= r; ++dz) {
        for (i32 dx = -r; dx <= r; ++dx) {
            if (dx * dx + dz * dz <= r * r) {
                m_offsets.emplace_back(dx, dz);
            }
        }
    }
    std::ranges::stable_sort(m_offsets, {}, [](const auto& o) { return o.first * o.first + o.second * o.second; });
    static_cast<void>(stoneChunk()); // initialise now rather than inside the first mesh job
    m_completions.reserve(m_maxInFlight);
    m_drained.reserve(m_maxInFlight);
}

ChunkStreamer::~ChunkStreamer()
{
    std::unique_lock lock(m_completionMutex);
    m_jobsDone.wait(lock, [this] { return m_running == 0; });
}

ChunkStreamer::Column* ChunkStreamer::find(ColumnCoord coord)
{
    const auto it = m_columns.find(coord.key());
    return it != m_columns.end() ? &it->second : nullptr;
}

i64 ChunkStreamer::distanceSq(ColumnCoord coord) const noexcept
{
    const i64 dx = coord.x - m_center.x;
    const i64 dz = coord.z - m_center.z;
    return dx * dx + dz * dz;
}

void ChunkStreamer::update(ColumnCoord center)
{
    ENGINE_PROFILE_FUNCTION();
    const bool moved = !m_hasCenter || center != m_center;
    m_center = center;
    m_hasCenter = true;
    if (moved) {
        m_scanStart = 0; // "finished" is relative to the centre
    }

    const usize before = m_inFlight;
    collectCompletions();
    const bool completed = m_inFlight != before;

    if (moved || completed) {
        m_settled = false;
        unloadFar(); // finished jobs may unblock unloading of their columns
    }
    if (!m_settled) {
        schedule();
    }

    // Everything scheduled above goes to the pool in one batch: one lock and
    // one wake-up for all workers instead of one per job.
    if (!m_batch.empty()) {
        {
            std::lock_guard lock(m_completionMutex);
            m_running += static_cast<u32>(m_batch.size());
        }
        m_pool.submit(std::move(m_batch)); // leaves m_batch empty, keeping its capacity
    }
}

void ChunkStreamer::recycle(std::vector<PackedQuad>&& quads)
{
    // Keep enough for every job in flight; drop the rest.
    if (quads.capacity() != 0 && m_spareMeshes.size() < m_maxInFlight) {
        quads.clear();
        m_spareMeshes.push_back(std::move(quads));
    }
}

std::vector<ChunkMeshEvent> ChunkStreamer::takeMeshEvents()
{
    return std::exchange(m_events, {});
}

ChunkStreamer::Stats ChunkStreamer::stats() const
{
    return {static_cast<u32>(m_columns.size()), m_stored, m_meshed, m_inFlight, m_chunkPool.allocatedChunks()};
}

void ChunkStreamer::collectCompletions()
{
    {
        // Swap with the main thread's (empty, but allocated) vector, so both
        // keep their capacity and jobs never reallocate while holding the lock.
        std::lock_guard lock(m_completionMutex);
        m_drained.swap(m_completions);
    }
    for (Completion& c : m_drained) {
        --m_inFlight;
        complete(std::move(c));
    }
    m_drained.clear();
}

void ChunkStreamer::complete(Completion&& c)
{
    Column* column = find(c.column);
    ENGINE_ASSERT(column != nullptr); // columns with jobs in flight are never unloaded

    if (c.layer < 0) {
        column->state = State::Ready;
        const GenerateOutput& out = *column->generated;
        for (usize i = 0; i < kLayers; ++i) {
            ChunkState& cs = column->chunks[i];
            cs.content = out.contents[i];
            cs.outer = out.outer[i];
            if (cs.content != ChunkContent::Mixed) {
                m_chunkPool.release(cs.data); // uniform: no storage needed
                cs.data = nullptr;
            } else {
                ++m_stored;
            }
        }
        column->generated.reset();
        return;
    }

    ChunkState& cs = column->chunks[static_cast<usize>(c.layer)];
    cs.meshing = false;
    // The player may have moved on while this was meshing.
    const i64 keep = m_options.viewRadius + 1;
    if (distanceSq(c.column) > keep * keep) {
        column->allMeshed = false;
        return;
    }
    cs.meshed = true;
    if (!c.quads.empty()) {
        cs.hasMesh = true;
        ++m_meshed;
        m_events.push_back({{c.column.x, c.layer, c.column.z}, std::move(c.quads), false});
    } else {
        recycle(std::move(c.quads));
    }
}

void ChunkStreamer::unloadFar()
{
    ENGINE_PROFILE_FUNCTION();
    const i64 unmesh = m_options.viewRadius + 1;
    const i64 unload = m_options.viewRadius + 3;
    for (auto it = m_columns.begin(); it != m_columns.end();) {
        Column& col = it->second;
        const i64 d2 = distanceSq(col.coord);
        if (d2 <= unmesh * unmesh) {
            ++it;
            continue;
        }
        bool meshing = false;
        for (usize i = 0; i < kLayers; ++i) {
            ChunkState& cs = col.chunks[i];
            if (cs.hasMesh) {
                cs.hasMesh = false;
                --m_meshed;
                m_events.push_back({{col.coord.x, static_cast<i32>(i), col.coord.z}, {}, true});
            }
            cs.meshed = false;
            meshing |= cs.meshing;
        }
        col.allMeshed = false;
        if (d2 > unload * unload && col.state == State::Ready && !meshing) {
            for (ChunkState& cs : col.chunks) {
                if (cs.data != nullptr) {
                    m_chunkPool.release(cs.data);
                    --m_stored;
                }
            }
            it = m_columns.erase(it);
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
    bool prefix = true; // still in the run of finished columns at the start of the scan

    for (usize i = m_scanStart; i < m_offsets.size(); ++i) {
        const auto [dx, dz] = m_offsets[i];
        const ColumnCoord coord{m_center.x + dx, m_center.z + dz};
        Column* col = find(coord);
        // Finished: generated, and meshed unless it is in the generate-only outer ring.
        const bool finished = col != nullptr && col->state == State::Ready &&
                              (col->allMeshed || static_cast<i64>(dx) * dx + static_cast<i64>(dz) * dz > meshRadiusSq);
        if (prefix && finished) {
            m_scanStart = i + 1;
            continue;
        }
        prefix = false;
        if (col == nullptr) {
            workLeft = true;
            if (m_inFlight >= m_maxInFlight) {
                break;
            }
            submitGenerate(coord);
            continue;
        }
        if (col->state == State::Generating) {
            workLeft = true;
            continue;
        }
        if (col->allMeshed || static_cast<i64>(dx) * dx + static_cast<i64>(dz) * dz > meshRadiusSq) {
            continue; // done, or in the generate-only outer ring
        }
        bool neighboursReady = true; // all 8 surrounding columns (ambient occlusion needs the diagonals)
        for (i32 nz = -1; nz <= 1; ++nz) {
            for (i32 nx = -1; nx <= 1; ++nx) {
                const Column* n = find({coord.x + nx, coord.z + nz});
                neighboursReady &= n != nullptr && n->state == State::Ready;
            }
        }
        if (!neighboursReady) {
            workLeft = true; // they are nearer the edge: generated soon
            continue;
        }
        const bool withinBudget = scheduleMeshes(*col);
        workLeft |= !col->allMeshed;
        if (!withinBudget) {
            break;
        }
    }
    m_settled = !workLeft && m_inFlight == 0;
}

void ChunkStreamer::trackSubmit()
{
    ++m_inFlight; // m_running is raised for the whole batch at the end of update()
}

void ChunkStreamer::finish(Completion&& completion)
{
    std::lock_guard lock(m_completionMutex);
    m_completions.push_back(std::move(completion));
    if (--m_running == 0) {
        m_jobsDone.notify_all(); // under the lock: see the header. Only the destructor waits for this.
    }
}

void ChunkStreamer::submitGenerate(ColumnCoord coord)
{
    // Storage for every layer up front (the pool is main-thread only); the
    // layers that turn out uniform are returned on completion.
    std::array<Chunk*, kLayers> storage;
    Column col{.coord = coord};
    for (usize i = 0; i < kLayers; ++i) {
        storage[i] = m_chunkPool.acquire();
        col.chunks[i].data = storage[i];
    }
    col.generated = std::make_unique<GenerateOutput>();
    GenerateOutput* out = col.generated.get(); // stable: owned by the column until completion
    m_columns.emplace(coord.key(), std::move(col));
    trackSubmit();

    m_batch.emplace_back([this, coord, storage, out] {
        ENGINE_PROFILE_SCOPE("generate column");
        TerrainColumn terrain;
        terrainColumn(coord.x, coord.z, terrain); // one heightmap for every layer
        for (usize i = 0; i < kLayers; ++i) {
            const i32 layer = static_cast<i32>(i);
            out->contents[i] = classifyTerrainChunk(terrain, layer);
            if (out->contents[i] == ChunkContent::Mixed) {
                fillTerrainChunk(*storage[i], terrain, layer); // writes every voxel
                // While the chunk is in this core's cache: record its outer
                // layers, which is all its neighbours' meshing needs from it.
                out->outer[i] = computeOuterLayers(*storage[i]);
            } else {
                out->outer[i] = uniformOuterLayers(out->contents[i]);
            }
        }
        finish({.column = coord, .layer = -1, .quads = {}});
    });
}

bool ChunkStreamer::scheduleMeshes(Column& col)
{
    // The 3 x 3 columns around and including this one, [dx + 1 + 3 * (dz + 1)].
    std::array<const Column*, 9> around{};
    for (i32 dz = -1; dz <= 1; ++dz) {
        for (i32 dx = -1; dx <= 1; ++dx) {
            around[static_cast<usize>((dx + 1) + 3 * (dz + 1))] = find({col.coord.x + dx, col.coord.z + dz});
        }
    }

    bool budgetLeft = true;
    for (usize i = 0; i < kLayers; ++i) {
        ChunkState& cs = col.chunks[i];
        if (cs.meshed || cs.meshing) {
            continue;
        }
        if (cs.content == ChunkContent::Air) {
            cs.meshed = true; // nothing to draw
            continue;
        }

        // The one-voxel shell around the chunk from its 26 neighbours' outer
        // layers, built here so the job never touches neighbour chunk memory.
        // Below the world is solid, above it air.
        std::array<const OuterLayers*, kNeighbourhoodSize> nb{};
        for (i32 dz = -1; dz <= 1; ++dz) {
            for (i32 dy = -1; dy <= 1; ++dy) {
                for (i32 dx = -1; dx <= 1; ++dx) {
                    const i32 layer = static_cast<i32>(i) + dy;
                    const OuterLayers* layers = nullptr;
                    if (layer < 0) {
                        layers = &solidOuterLayers();
                    } else if (layer < static_cast<i32>(kLayers)) {
                        layers = &around[static_cast<usize>((dx + 1) + 3 * (dz + 1))]
                                      ->chunks[static_cast<usize>(layer)]
                                      .outer;
                    }
                    nb[neighbourIndex(dx, dy, dz)] = layers;
                }
            }
        }
        const NeighbourShell shell = shellFromOuterLayers(nb);
        if (cs.content == ChunkContent::Stone && allSolid(shell.faces)) {
            cs.meshed = true; // buried: no visible faces
            continue;
        }

        if (m_inFlight >= m_maxInFlight) {
            budgetLeft = false;
            break;
        }
        cs.meshing = true; // keeps the column loaded while the job reads the chunk
        trackSubmit();
        const Chunk* chunk = cs.data != nullptr ? cs.data : &stoneChunk();
        // The job meshes straight into a recycled vector (see recycle()), so
        // once warmed up it does no heap allocation: allocating on many
        // threads at once contends inside the allocator.
        std::vector<PackedQuad> out;
        if (!m_spareMeshes.empty()) {
            out = std::move(m_spareMeshes.back());
            m_spareMeshes.pop_back();
        }
        m_batch.emplace_back([this, coord = col.coord, layer = static_cast<i32>(i), chunk, shell,
                              greedy = m_options.greedy, ao = m_options.ambientOcclusion,
                              out = std::move(out)]() mutable {
            ENGINE_PROFILE_SCOPE("mesh chunk");
            if (greedy) {
                meshChunkGreedy(*chunk, shell, out, ao);
            } else {
                meshChunkBinary(*chunk, shell, out, ao);
            }
            finish({.column = coord, .layer = layer, .quads = std::move(out)});
        });
    }

    col.allMeshed = std::ranges::all_of(col.chunks, [](const ChunkState& cs) { return cs.meshed; });
    return budgetLeft;
}

} // namespace engine::voxel
