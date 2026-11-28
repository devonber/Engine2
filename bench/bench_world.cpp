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
        u64 updates = 0, areas = 0;
        f64 updateMs = 0.0, wallMs = 0.0;
        const auto streamArea = [&] {
            x += 1000; // a completely new area every time
            meshes = 0;
            const auto start = std::chrono::steady_clock::now();
            do {
                const auto u0 = std::chrono::steady_clock::now();
                streamer.update({x, 0});
                meshes += streamer.takeMeshEvents().size();
                updateMs += std::chrono::duration<f64, std::milli>(std::chrono::steady_clock::now() - u0).count();
                ++updates;
            } while (!streamer.settled());
            wallMs += std::chrono::duration<f64, std::milli>(std::chrono::steady_clock::now() - start).count();
            ++areas;
        };
        const auto t0 = std::chrono::steady_clock::now();
        streamArea();
        std::printf("%s: first area (fresh memory, page faults) %.2f ms, %zu chunk meshes\n", label,
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(), meshes);
        updates = 0;
        areas = 0;
        updateMs = wallMs = 0.0;
        pool.resetJobStats();
        s.run(label, streamArea);
        const ThreadPool::JobStats js = pool.jobStats();
        // Workers busy for a small fraction of the wall time = starved (by the
        // main thread or the queue), not short of work.
        std::printf("%s: per area %.0f update() calls, %.0f jobs; workers busy %.0f%% of the time\n", label,
                    static_cast<f64>(updates) / static_cast<f64>(areas),
                    static_cast<f64>(js.jobs) / static_cast<f64>(areas),
                    100.0 * js.busyMs / (wallMs * static_cast<f64>(pool.workerCount())));
    };

    // One worker does all the jobs (the main thread only schedules).
    ThreadPool single(1);
    streamBench(single, "1 worker");
    ThreadPool pool; // all hardware threads
    char name[32];
    std::snprintf(name, sizeof(name), "%u workers", pool.workerCount());
    streamBench(pool, name);
}
