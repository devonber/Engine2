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

    // Streaming, without a renderer: the area around a new centre, generated
    // and meshed with a free-running main loop. Steady state: one streamer,
    // moved far away each time, so chunk memory is reused (as when flying).
    // The first area page-faults its fresh memory and is reported separately.
    ChunkStreamer streamer(pool, {.viewRadius = 12});
    i32 x = 0;
    const auto streamArea = [&] {
        x += 1000; // a completely new area every time
        do {
            streamer.update({x, 0});
            ankerl::nanobench::doNotOptimizeAway(streamer.takeMeshEvents().size());
        } while (!streamer.settled());
    };
    const auto t0 = std::chrono::steady_clock::now();
    streamArea();
    std::printf("stream in radius 12, first area (fresh memory, page faults): %.2f ms\n",
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    ankerl::nanobench::Bench s;
    s.title("stream in radius 12 (529 generated, 441 meshed), memory reused")
        .unit("area")
        .warmup(1)
        .epochs(7)
        .epochIterations(1);
    s.run(name, streamArea);
}
