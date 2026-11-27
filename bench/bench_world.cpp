#include "engine/core/ThreadPool.hpp"
#include "engine/voxel/ChunkStreamer.hpp"
#include "engine/voxel/Generate.hpp"
#include "engine/voxel/World.hpp"

#include <nanobench.h>

#include <chrono>
#include <cstdio>
#include <memory>

using namespace engine;
using namespace engine::voxel;

void runWorldBenchmarks()
{
    {
        auto chunk = std::make_unique<Chunk>();
        ankerl::nanobench::Bench b;
        b.title("generate").unit("chunk").warmup(20).minEpochTime(std::chrono::milliseconds(20));
        i32 x = 0;
        b.run("fillTerrain", [&] {
            fillTerrain(*chunk, x++ & 15, 3); // vary the chunk so nothing is cached
            ankerl::nanobench::doNotOptimizeAway(chunk->blocks().data());
        });
    }

    // Whole world build (radius 8 = 256 chunks), serial vs thread pool.
    ThreadPool pool; // all hardware threads
    const WorldBuildOptions options{.radius = 8};
    ankerl::nanobench::Bench b;
    b.title("build world (256 chunks)").unit("world").warmup(1).epochs(5).epochIterations(1).relative(true);
    b.run("1 thread", [&] { ankerl::nanobench::doNotOptimizeAway(buildTerrainWorld(options, nullptr).quads.size()); });
    char name[32];
    std::snprintf(name, sizeof(name), "%u threads", pool.concurrency());
    b.run(name, [&] { ankerl::nanobench::doNotOptimizeAway(buildTerrainWorld(options, &pool).quads.size()); });

    // Streaming, without a renderer: how fast the area around a fresh centre
    // is generated and meshed when the main loop spins freely.
    ankerl::nanobench::Bench s;
    s.title("stream in radius 12 (529 generated, 441 meshed)").unit("area").warmup(1).epochs(5).epochIterations(1);
    i32 x = 0;
    s.run(name, [&] {
        ChunkStreamer streamer(pool, {.viewRadius = 12});
        x += 1000; // fresh area every time
        do {
            streamer.update({x, 0});
            ankerl::nanobench::doNotOptimizeAway(streamer.takeMeshEvents().size());
        } while (!streamer.settled());
    });
}
