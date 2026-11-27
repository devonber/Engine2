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
                REQUIRE(meshes.count(e.coord) == 0); // never meshed twice without removal
                meshes[e.coord] = std::move(e.quads);
            }
        }
    }
};

// Updates until settled (or fails after a timeout).
void settle(ChunkStreamer& streamer, ChunkCoord center, MeshSet& set)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    do {
        streamer.update(center);
        set.apply(streamer.takeMeshEvents());
        REQUIRE(std::chrono::steady_clock::now() < deadline);
        std::this_thread::yield();
    } while (!streamer.settled());
}

std::set<ChunkCoord> expectedMeshes(ChunkCoord center, i32 radius)
{
    std::set<ChunkCoord> out;
    for (i32 dz = -radius; dz <= radius; ++dz) {
        for (i32 dx = -radius; dx <= radius; ++dx) {
            if (dx * dx + dz * dz <= radius * radius) {
                out.insert({center.x + dx, center.z + dz});
            }
        }
    }
    return out;
}

std::set<ChunkCoord> keys(const MeshSet& set)
{
    std::set<ChunkCoord> out;
    for (const auto& [coord, quads] : set.meshes) {
        out.insert(coord);
    }
    return out;
}

// What the mesh of `coord` must be: meshed directly against generated neighbours.
std::vector<PackedQuad> referenceMesh(ChunkCoord coord)
{
    auto self = std::make_unique<Chunk>();
    fillTerrain(*self, coord.x, coord.z);
    std::unique_ptr<Chunk> n[4];
    const i32 d[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    const Face faces[4] = {Face::PosX, Face::NegX, Face::PosZ, Face::NegZ};
    ChunkNeighbours nb;
    for (int i = 0; i < 4; ++i) {
        n[i] = std::make_unique<Chunk>();
        fillTerrain(*n[i], coord.x + d[i][0], coord.z + d[i][1]);
        nb.chunks[static_cast<usize>(faces[i])] = n[i].get();
    }
    BoundaryMasks masks = buildBoundaryMasks(nb);
    masks.setSolid(Face::NegY);
    std::vector<PackedQuad> quads;
    meshChunkGreedy(*self, masks, quads);
    return quads;
}

} // namespace

TEST_CASE("chunkAt floors toward negative infinity")
{
    CHECK(chunkAt(0.0f, 0.0f) == ChunkCoord{0, 0});
    CHECK(chunkAt(31.9f, 32.0f) == ChunkCoord{0, 1});
    CHECK(chunkAt(-0.1f, -32.0f) == ChunkCoord{-1, -1});
    CHECK(chunkAt(-32.1f, 5.0f) == ChunkCoord{-2, 0});
    CHECK(ChunkCoord{-1, 2}.key() != ChunkCoord{2, -1}.key());
}

TEST_CASE("Streaming meshes exactly the chunks in view, matching direct meshing")
{
    ThreadPool pool(3);
    ChunkStreamer streamer(pool, {.viewRadius = 3});
    MeshSet set;
    settle(streamer, {0, 0}, set);

    CHECK(keys(set) == expectedMeshes({0, 0}, 3));
    CHECK(streamer.stats().jobsInFlight == 0);
    for (const auto& [coord, quads] : set.meshes) {
        CAPTURE(coord.x);
        CAPTURE(coord.z);
        REQUIRE(quads == referenceMesh(coord));
    }
}

TEST_CASE("Moving away unloads old chunks and reuses their memory")
{
    ThreadPool pool(3);
    ChunkStreamer streamer(pool, {.viewRadius = 4});
    MeshSet set;
    settle(streamer, {0, 0}, set);
    const usize poolAfterFirst = streamer.stats().poolChunks;

    settle(streamer, {100, -40}, set);
    CHECK(keys(set) == expectedMeshes({100, -40}, 4));
    // Everything near the origin is gone, and storage was recycled, not grown
    // (only in-flight overlap may need a little more).
    CHECK(streamer.stats().loadedChunks <= expectedMeshes({0, 0}, 6).size());
    CHECK(streamer.stats().poolChunks <= 2 * poolAfterFirst);

    settle(streamer, {0, 0}, set);
    CHECK(keys(set) == expectedMeshes({0, 0}, 4));
}

TEST_CASE("Erratic movement never leaves stale or duplicate meshes")
{
    ThreadPool pool(3);
    ChunkStreamer streamer(pool, {.viewRadius = 3, .maxJobsInFlight = 5});
    MeshSet set;
    SplitMix64 rng{42};
    ChunkCoord center{0, 0};
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
    CHECK(keys(set) == expectedMeshes(center, 3));
    for (const auto& [coord, quads] : set.meshes) {
        REQUIRE(quads == referenceMesh(coord));
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
