#include "engine/core/Random.hpp"
#include "engine/core/ThreadPool.hpp"
#include "engine/voxel/ChunkStreamer.hpp"
#include "engine/voxel/Generate.hpp"

#include <doctest/doctest.h>

#include <chrono>
#include <map>
#include <memory>
#include <set>
#include <thread>

using namespace engine;
using namespace engine::voxel;

namespace {

// Applies mesh events like a renderer would, checking they are consistent.
struct MeshSet {
    std::map<ChunkCoord, std::vector<PackedQuad>> meshes;

    void apply(std::vector<ChunkMeshEvent>&& events)
    {
        for (ChunkMeshEvent& e : events) {
            if (e.removed) {
                REQUIRE(meshes.erase(e.coord) == 1); // only remove what exists
            } else {
                REQUIRE(!e.quads.empty());           // empty meshes are not sent
                REQUIRE(meshes.count(e.coord) == 0); // never meshed twice without removal
                meshes[e.coord] = std::move(e.quads);
            }
        }
    }
};

// Updates until settled (or fails after a timeout).
void settle(ChunkStreamer& streamer, ColumnCoord center, MeshSet& set)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    do {
        streamer.update(center);
        set.apply(streamer.takeMeshEvents());
        REQUIRE(std::chrono::steady_clock::now() < deadline);
        std::this_thread::yield();
    } while (!streamer.settled());
}

// What the mesh of `coord` must be: meshed directly against its 26
// generated neighbours (the slow reference path, reading neighbour chunks).
// Above the world is air; below it, solid.
std::vector<PackedQuad> referenceMesh(ChunkCoord c, u32 level)
{
    static const auto solid = [] {
        auto chunk = std::make_unique<Chunk>();
        chunk->fill(1);
        return chunk;
    }();
    auto self = std::make_unique<Chunk>();
    fillTerrain(*self, c.x, c.y, c.z, level);
    std::vector<std::unique_ptr<Chunk>> owned;
    ChunkNeighbourhood nb;
    for (i32 dz = -1; dz <= 1; ++dz) {
        for (i32 dy = -1; dy <= 1; ++dy) {
            for (i32 dx = -1; dx <= 1; ++dx) {
                const i32 y = c.y + dy;
                if ((dx == 0 && dy == 0 && dz == 0) || y >= kWorldHeightChunks) {
                    continue; // the chunk itself / above the world: air
                }
                if (y < 0) {
                    nb.set(dx, dy, dz, solid.get());
                    continue;
                }
                owned.push_back(std::make_unique<Chunk>());
                fillTerrain(*owned.back(), c.x + dx, y, c.z + dz, level);
                nb.set(dx, dy, dz, owned.back().get());
            }
        }
    }
    std::vector<PackedQuad> quads;
    meshChunkGreedy(*self, buildNeighbourShell(nb), quads);
    return quads;
}

// Every non-empty chunk mesh within `radius` columns of `center`.
std::map<ChunkCoord, std::vector<PackedQuad>> expectedMeshes(ColumnCoord center, i32 radius, u32 level)
{
    std::map<ChunkCoord, std::vector<PackedQuad>> out;
    for (i32 dz = -radius; dz <= radius; ++dz) {
        for (i32 dx = -radius; dx <= radius; ++dx) {
            if (dx * dx + dz * dz > radius * radius) {
                continue;
            }
            for (i32 y = 0; y < kWorldHeightChunks; ++y) {
                const ChunkCoord c{center.x + dx, y, center.z + dz};
                auto quads = referenceMesh(c, level);
                if (!quads.empty()) {
                    out[c] = std::move(quads);
                }
            }
        }
    }
    return out;
}

void checkMatches(const MeshSet& set, ColumnCoord center, i32 radius, u32 level = 0)
{
    // Exactly the meshes within `radius`, plus possibly some within
    // radius + 1 (unmeshing has hysteresis), all identical to direct meshing.
    const auto expected = expectedMeshes(center, radius, level);
    usize extra = 0;
    for (const auto& [coord, quads] : set.meshes) {
        if (expected.count(coord) != 0) {
            continue;
        }
        CAPTURE(coord.x);
        CAPTURE(coord.y);
        CAPTURE(coord.z);
        const i64 dx = coord.x - center.x;
        const i64 dz = coord.z - center.z;
        REQUIRE(dx * dx + dz * dz <= static_cast<i64>(radius + 1) * (radius + 1));
        REQUIRE(quads == referenceMesh(coord, level));
        ++extra;
    }
    REQUIRE(set.meshes.size() == expected.size() + extra);
    for (const auto& [coord, quads] : expected) {
        CAPTURE(coord.x);
        CAPTURE(coord.y);
        CAPTURE(coord.z);
        const auto it = set.meshes.find(coord);
        REQUIRE(it != set.meshes.end());
        REQUIRE(it->second == quads);
    }
}

} // namespace

TEST_CASE("Coordinates: columnAt floors, chunk keys are unique")
{
    CHECK(columnAt(0.0, 0.0) == ColumnCoord{0, 0});
    CHECK(columnAt(31.9, 32.0) == ColumnCoord{0, 1});
    CHECK(columnAt(-0.1, -32.0) == ColumnCoord{-1, -1});
    CHECK(columnAt(-32.1, 5.0) == ColumnCoord{-2, 0});
    // Far from the origin (double precision): 1,000,000 / 32 = 31250 exactly.
    CHECK(columnAt(1'000'000.0 - 0.001, 1'000'000.0) == ColumnCoord{31249, 31250});
    // Coarser levels: columns 32 << level blocks wide.
    CHECK(columnAt(63.9, -0.1, 1) == ColumnCoord{0, -1});
    CHECK(columnAt(64.0, 255.9, 3) == ColumnCoord{0, 0});
    CHECK(columnAt(-256.1, 256.0, 3) == ColumnCoord{-2, 1});
    std::set<u64> keys;
    for (u32 level : {0u, 1u, 3u}) {
        for (i32 x : {-2, -1, 0, 1, 2}) {
            for (i32 y : {-1, 0, 1, 3}) {
                for (i32 z : {-2, -1, 0, 1, 2}) {
                    REQUIRE(keys.insert(ChunkCoord{x, y, z}.key(level)).second);
                }
            }
        }
    }
}

TEST_CASE("Streaming meshes exactly the visible chunks in view, matching direct meshing")
{
    ThreadPool pool(3);
    ChunkStreamer streamer(pool, {.viewRadius = 3});
    MeshSet set;
    settle(streamer, {0, 0}, set);
    checkMatches(set, {0, 0}, 3);
    CHECK(streamer.stats().jobsInFlight == 0);
    // Uniform chunks (all air or all stone) take no storage.
    CHECK(streamer.stats().storedChunks < streamer.stats().loadedColumns * kWorldHeightChunks);
}

TEST_CASE("Coarse levels of detail stream like level 0, with their own generator and coordinates")
{
    ThreadPool pool(3);
    for (u32 level : {1u, 2u, 4u}) {
        CAPTURE(level);
        ChunkStreamer streamer(pool, {.viewRadius = 2, .level = level});
        MeshSet set;
        settle(streamer, {-3, 5}, set);
        checkMatches(set, {-3, 5}, 2, level);
        // Coarse levels are shorter in cells: no meshes above their top layer.
        for (const auto& [coord, quads] : set.meshes) {
            CHECK(coord.y < worldHeightChunks(level));
        }
    }
}

TEST_CASE("Moving away unloads old columns and reuses their memory")
{
    ThreadPool pool(3);
    ChunkStreamer streamer(pool, {.viewRadius = 4});
    MeshSet set;
    settle(streamer, {0, 0}, set);
    const usize poolAfterFirst = streamer.stats().poolChunks;

    settle(streamer, {100, -40}, set);
    checkMatches(set, {100, -40}, 4);
    CHECK(streamer.stats().poolChunks <= 2 * poolAfterFirst); // recycled, not grown

    settle(streamer, {0, 0}, set);
    checkMatches(set, {0, 0}, 4);
}

TEST_CASE("Without keepChunkData, meshed chunks release their voxels and meshes stay correct")
{
    ThreadPool pool(3);
    ChunkStreamer streamer(pool, {.viewRadius = 3, .level = 1, .keepChunkData = false});
    MeshSet set;
    settle(streamer, {2, -1}, set);
    checkMatches(set, {2, -1}, 3, 1);
    // Only the generate-only outer ring (not meshed) still holds voxels.
    ChunkStreamer keeping(pool, {.viewRadius = 3, .level = 1});
    MeshSet other;
    settle(keeping, {2, -1}, other);
    CHECK(streamer.stats().storedChunks < keeping.stats().storedChunks);
    // Moving back and forth re-meshes released columns by regenerating them.
    for (const ColumnCoord c : {ColumnCoord{12, -1}, ColumnCoord{2, -1}, ColumnCoord{3, 0}, ColumnCoord{2, -1}}) {
        settle(streamer, c, set);
        checkMatches(set, c, 3, 1);
    }
}

TEST_CASE("Erratic movement never leaves stale or duplicate meshes")
{
    for (const bool keep : {true, false}) {
        CAPTURE(keep);
        ThreadPool pool(3);
        ChunkStreamer streamer(pool, {.viewRadius = 3, .maxJobsInFlight = 5, .keepChunkData = keep});
        MeshSet set;
        SplitMix64 rng{42};
        ColumnCoord center{0, 0};
        for (int step = 0; step < 300; ++step) {
            // Mostly small steps (crossing borders), sometimes teleports.
            if (rng.next() % 20 == 0) {
                center = {static_cast<i32>(rng.next() % 200) - 100, static_cast<i32>(rng.next() % 200) - 100};
            } else {
                center.x += static_cast<i32>(rng.next() % 3) - 1;
                center.z += static_cast<i32>(rng.next() % 3) - 1;
            }
            streamer.update(center);
            set.apply(streamer.takeMeshEvents());
        }
        settle(streamer, center, set);
        checkMatches(set, center, 3);
    }
}

TEST_CASE("Destroying a streamer with jobs in flight is safe")
{
    ThreadPool pool(3);
    for (int i = 0; i < 20; ++i) {
        ChunkStreamer streamer(pool, {.viewRadius = 6});
        streamer.update({i, -i}); // schedules jobs, then goes out of scope at once
    }
}
