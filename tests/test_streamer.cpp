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

// What the mesh of `coord` must be: meshed directly against its six
// generated neighbours (the slow reference path, reading neighbour chunks).
std::vector<PackedQuad> referenceMesh(ChunkCoord c)
{
    auto self = std::make_unique<Chunk>();
    fillTerrain(*self, c.x, c.y, c.z);
    std::vector<std::unique_ptr<Chunk>> owned;
    ChunkNeighbours nb;
    const auto add = [&](Face face, i32 x, i32 y, i32 z) {
        if (y < 0 || y >= kWorldHeightChunks) {
            return; // above the world: air (below is handled with setSolid)
        }
        owned.push_back(std::make_unique<Chunk>());
        fillTerrain(*owned.back(), x, y, z);
        nb.chunks[static_cast<usize>(face)] = owned.back().get();
    };
    add(Face::PosX, c.x + 1, c.y, c.z);
    add(Face::NegX, c.x - 1, c.y, c.z);
    add(Face::PosZ, c.x, c.y, c.z + 1);
    add(Face::NegZ, c.x, c.y, c.z - 1);
    add(Face::PosY, c.x, c.y + 1, c.z);
    add(Face::NegY, c.x, c.y - 1, c.z);
    BoundaryMasks masks = buildBoundaryMasks(nb);
    if (c.y == 0) {
        masks.setSolid(Face::NegY);
    }
    std::vector<PackedQuad> quads;
    meshChunkGreedy(*self, masks, quads);
    return quads;
}

// Every non-empty chunk mesh within `radius` columns of `center`.
std::map<ChunkCoord, std::vector<PackedQuad>> expectedMeshes(ColumnCoord center, i32 radius)
{
    std::map<ChunkCoord, std::vector<PackedQuad>> out;
    for (i32 dz = -radius; dz <= radius; ++dz) {
        for (i32 dx = -radius; dx <= radius; ++dx) {
            if (dx * dx + dz * dz > radius * radius) {
                continue;
            }
            for (i32 y = 0; y < kWorldHeightChunks; ++y) {
                const ChunkCoord c{center.x + dx, y, center.z + dz};
                auto quads = referenceMesh(c);
                if (!quads.empty()) {
                    out[c] = std::move(quads);
                }
            }
        }
    }
    return out;
}

void checkMatches(const MeshSet& set, ColumnCoord center, i32 radius)
{
    const auto expected = expectedMeshes(center, radius);
    REQUIRE(set.meshes.size() == expected.size());
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
    CHECK(columnAt(0.0f, 0.0f) == ColumnCoord{0, 0});
    CHECK(columnAt(31.9f, 32.0f) == ColumnCoord{0, 1});
    CHECK(columnAt(-0.1f, -32.0f) == ColumnCoord{-1, -1});
    CHECK(columnAt(-32.1f, 5.0f) == ColumnCoord{-2, 0});
    std::set<u64> keys;
    for (i32 x : {-2, -1, 0, 1, 2}) {
        for (i32 y : {-1, 0, 1, 3}) {
            for (i32 z : {-2, -1, 0, 1, 2}) {
                REQUIRE(keys.insert(ChunkCoord{x, y, z}.key()).second);
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

TEST_CASE("Erratic movement never leaves stale or duplicate meshes")
{
    ThreadPool pool(3);
    ChunkStreamer streamer(pool, {.viewRadius = 3, .maxJobsInFlight = 5});
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

TEST_CASE("Destroying a streamer with jobs in flight is safe")
{
    ThreadPool pool(3);
    for (int i = 0; i < 20; ++i) {
        ChunkStreamer streamer(pool, {.viewRadius = 6});
        streamer.update({i, -i}); // schedules jobs, then goes out of scope at once
    }
}
