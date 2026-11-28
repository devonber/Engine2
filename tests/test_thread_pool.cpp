#include "engine/core/ThreadPool.hpp"

#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <numeric>
#include <thread>
#include <vector>

using namespace engine;

TEST_CASE("parallelFor visits every index exactly once")
{
    ThreadPool pool(3);
    CHECK(pool.workerCount() == 3);
    CHECK(pool.concurrency() == 4);

    for (const u32 grain : {1u, 3u, 64u, 1000u}) {
        CAPTURE(grain);
        std::vector<std::atomic<u32>> hits(1000);
        pool.parallelFor(static_cast<u32>(hits.size()), [&](u32 i) { hits[i].fetch_add(1); }, grain);
        for (const auto& h : hits) {
            REQUIRE(h.load() == 1);
        }
    }
}

TEST_CASE("parallelFor handles empty and tiny ranges")
{
    ThreadPool pool(2);
    int calls = 0;
    pool.parallelFor(0, [&](u32) { ++calls; });
    CHECK(calls == 0);
    std::atomic<int> one{0};
    pool.parallelFor(1, [&](u32) { one.fetch_add(1); });
    CHECK(one.load() == 1);
}

TEST_CASE("Back-to-back parallelFor calls do not mix tasks")
{
    // Many short tasks in a row exercise the hand-over between tasks, where
    // a late worker could otherwise run a stale task.
    ThreadPool pool(3);
    for (u32 round = 0; round < 2000; ++round) {
        std::vector<u32> out(17, 0);
        pool.parallelFor(static_cast<u32>(out.size()), [&](u32 i) { out[i] = i + round; });
        for (u32 i = 0; i < out.size(); ++i) {
            REQUIRE(out[i] == i + round);
        }
    }
}

TEST_CASE("A single-worker pool still covers the whole range")
{
    ThreadPool single(1);
    std::vector<u32> v(100);
    single.parallelFor(100, [&](u32 i) { v[i] = i * 2; });
    CHECK(std::accumulate(v.begin(), v.end(), 0u) == 9900u);
}

namespace {
// Waits (with a generous timeout) for a counter to reach a value.
bool waitFor(const std::atomic<u32>& counter, u32 value)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (counter.load() < value) {
        if (std::chrono::steady_clock::now() > deadline) {
            return false;
        }
        std::this_thread::yield();
    }
    return true;
}
} // namespace

TEST_CASE("Submitted jobs all run, on worker threads")
{
    ThreadPool pool(3);
    std::atomic<u32> done{0};
    std::atomic<u32> onCaller{0};
    const auto caller = std::this_thread::get_id();
    for (u32 i = 0; i < 500; ++i) {
        pool.submit([&] {
            onCaller.fetch_add(std::this_thread::get_id() == caller ? 1 : 0);
            done.fetch_add(1);
        });
    }
    REQUIRE(waitFor(done, 500));
    CHECK(onCaller.load() == 0);
    CHECK(pool.queuedJobs() == 0);
}

TEST_CASE("parallelFor completes while background jobs are running")
{
    ThreadPool pool(3);
    std::atomic<bool> release{false};
    std::atomic<u32> started{0};
    std::atomic<u32> finished{0};
    // Occupy every worker with a job that waits for us.
    for (u32 i = 0; i < 3; ++i) {
        pool.submit([&] {
            started.fetch_add(1);
            while (!release.load()) {
                std::this_thread::yield();
            }
            finished.fetch_add(1);
        });
    }
    REQUIRE(waitFor(started, 3));

    // The caller must be able to finish the whole range alone.
    std::vector<u32> out(100, 0);
    pool.parallelFor(100, [&](u32 i) { out[i] = i + 1; });
    for (u32 i = 0; i < 100; ++i) {
        REQUIRE(out[i] == i + 1);
    }

    release.store(true);
    REQUIRE(waitFor(finished, 3));
}

TEST_CASE("Mixing jobs and parallelFor under load")
{
    ThreadPool pool(3);
    std::atomic<u32> jobs{0};
    for (u32 round = 0; round < 200; ++round) {
        pool.submit([&] { jobs.fetch_add(1); });
        std::atomic<u32> sum{0};
        pool.parallelFor(64, [&](u32 i) { sum.fetch_add(i); });
        REQUIRE(sum.load() == 64 * 63 / 2);
    }
    REQUIRE(waitFor(jobs, 200));
}

TEST_CASE("Batch submit runs every job, in small and large batches")
{
    ThreadPool pool(3);
    std::atomic<u32> done{0};
    u32 expected = 0;
    for (const u32 size : {1u, 2u, 3u, 50u, 1000u}) {
        std::vector<ThreadPool::Job> batch;
        for (u32 i = 0; i < size; ++i) {
            batch.emplace_back([&] { done.fetch_add(1); });
        }
        pool.submit(std::move(batch));
        CHECK(batch.empty());
        expected += size;
    }
    REQUIRE(waitFor(done, expected));

    // Sparse submits with idle gaps: workers go through spin -> sleep -> wake.
    for (u32 i = 0; i < 20; ++i) {
        pool.submit([&] { done.fetch_add(1); });
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    REQUIRE(waitFor(done, expected + 20));
}
