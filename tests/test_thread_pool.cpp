#include "engine/core/ThreadPool.hpp"

#include <doctest/doctest.h>

#include <atomic>
#include <numeric>
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
