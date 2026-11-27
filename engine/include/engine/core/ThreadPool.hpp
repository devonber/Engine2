#pragma once

#include "engine/core/Types.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

namespace engine {

// Fixed set of worker threads for data-parallel batches ("do this for every
// chunk"). parallelFor blocks until the whole range is done, and the calling
// thread works too, so there is no idle wait on the caller's core.
//
// Work is handed out in small ranges through one atomic counter: threads that
// finish early simply take more, which balances uneven items (e.g. empty vs
// complex chunks) without a scheduler.
//
// One parallelFor runs at a time; concurrent callers queue up. Asynchronous
// background jobs (chunk streaming) will need a task queue on top of this.
class ThreadPool {
public:
    // workers = 0: one per hardware thread, minus the caller's.
    explicit ThreadPool(u32 workers = 0);
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    [[nodiscard]] u32 workerCount() const noexcept { return static_cast<u32>(m_threads.size()); }
    // Threads that run a parallelFor: the workers plus the caller.
    [[nodiscard]] u32 concurrency() const noexcept { return workerCount() + 1; }

    // Calls fn(i) for every i in [0, count), in parallel, in no particular
    // order. `grain` indices are claimed at a time; raise it for tiny items.
    // fn must not throw.
    template <typename Fn>
    void parallelFor(u32 count, Fn&& fn, u32 grain = 1)
    {
        static_assert(std::is_invocable_v<Fn&, u32>);
        if (count == 0) {
            return;
        }
        auto invoke = [](void* ctx, u32 begin, u32 end) {
            Fn& f = *static_cast<std::remove_reference_t<Fn>*>(ctx);
            for (u32 i = begin; i < end; ++i) {
                f(i);
            }
        };
        run(Task{invoke, const_cast<void*>(static_cast<const void*>(&fn)), count, std::max(grain, 1u)});
    }

private:
    struct Task {
        void (*invoke)(void* ctx, u32 begin, u32 end) = nullptr;
        void* ctx = nullptr;
        u32 count = 0;
        u32 grain = 1;
    };

    void run(const Task& task);
    void workerLoop();
    void process(const Task& task);

    std::vector<std::thread> m_threads;

    std::mutex m_submit; // serialises parallelFor calls
    std::mutex m_mutex;  // guards everything below except the atomics
    std::condition_variable m_wake;
    std::condition_variable m_idle;
    Task m_task;
    u64 m_generation = 0; // bumped for every task; workers run each generation once
    u32 m_busy = 0;       // workers still inside the current task
    bool m_stop = false;

    std::atomic<u32> m_next{0}; // next index to hand out
};

} // namespace engine
