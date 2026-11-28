#include "engine/core/ThreadPool.hpp"
#include "engine/voxel/ChunkStreamer.hpp"
#include "engine/voxel/Generate.hpp"

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
            fillTerrain(*chunk, x++ & 15, 1, 3); // vary the chunk so nothing is cached
            ankerl::nanobench::doNotOptimizeAway(chunk->blocks().data());
        });
    }

    // Streaming, without a renderer: the area around a new centre, generated
    // and meshed with a free-running main loop. Steady state: one streamer,
    // moved far away each time, so chunk memory is reused (as when flying).
    // The first area page-faults its fresh memory and is reported separately.
    ankerl::nanobench::Bench s;
    s.title("stream radius 12: 529 columns x 4 chunks, memory reused")
        .unit("area")
        .warmup(1)
        .epochs(7)
        .epochIterations(1)
        .relative(true);

    const auto streamBench = [&](ThreadPool& pool, const char* label) {
        ChunkStreamer streamer(pool, {.viewRadius = 12});
        i32 x = 0;
        usize meshes = 0;
        const auto streamArea = [&] {
            x += 1000; // a completely new area every time
            meshes = 0;
            do {
                streamer.update({x, 0});
                meshes += streamer.takeMeshEvents().size();
            } while (!streamer.settled());
        };
        const auto t0 = std::chrono::steady_clock::now();
        streamArea();
        std::printf("%s: first area (fresh memory, page faults) %.2f ms, %zu chunk meshes\n", label,
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(), meshes);
        s.run(label, streamArea);
    };

    // One worker does all the jobs (the main thread only schedules).
    ThreadPool single(1);
    streamBench(single, "1 worker");
    ThreadPool pool; // all hardware threads
    char name[32];
    std::snprintf(name, sizeof(name), "%u workers", pool.workerCount());
    streamBench(pool, name);
}
