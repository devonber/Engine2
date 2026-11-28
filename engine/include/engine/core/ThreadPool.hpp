#pragma once

#include "engine/core/Types.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
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
// One parallelFor runs at a time; concurrent callers queue up.
//
// Background jobs (submit) run on workers whenever they are not helping with
// a parallelFor, in FIFO order. The caller of a parallelFor always works on
// it, so long-running jobs never block it.
class ThreadPool {
public:
    using Job = std::move_only_function<void()>;

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

    // Queue a job to run on a worker thread. The job must not throw, and
    // whatever it references must outlive it: jobs still queued when the pool
    // is destroyed are dropped, so owners wait for their own jobs first.
    // With no workers the job runs immediately on the caller.
    void submit(Job job);

    // Queue many jobs at once: one lock, and all needed workers woken
    // together instead of one wake-up per job. Prefer this for bursts of
    // small jobs.
    void submit(std::vector<Job>&& jobs);

    // Jobs queued but not yet started.
    [[nodiscard]] u32 queuedJobs() const;

    // Background-job totals since construction or the last reset, for
    // measuring worker utilisation (busy time / (wall time * workers)).
    struct JobStats {
        u64 jobs = 0;
        f64 busyMs = 0.0;
    };
    [[nodiscard]] JobStats jobStats() const noexcept;
    void resetJobStats() noexcept;

private:
    struct Task {
        void (*invoke)(void* ctx, u32 begin, u32 end) = nullptr;
        void* ctx = nullptr;
        u32 count = 0;
        u32 grain = 1;
    };

    void run(const Task& task);
    void workerLoop();
    // Busy-waits briefly for new work before a worker goes to sleep: waking a
    // sleeping thread through the OS costs about as much as a small job.
    void spinForWork(u64 seenGeneration) const noexcept;
    void process(const Task& task);

    std::vector<std::thread> m_threads;

    std::mutex m_submit;        // serialises parallelFor calls
    mutable std::mutex m_mutex; // guards everything below except the atomics
    std::condition_variable m_wake;
    std::condition_variable m_idle;
    Task m_task;
    u64 m_generation = 0; // bumped for every task; workers run each generation once
    u32 m_busy = 0;       // workers still inside the current task
    bool m_stop = false;
    std::deque<Job> m_jobs;

    std::atomic<u32> m_next{0}; // next index to hand out
    // Lock-free mirrors (written under m_mutex) that spinning workers poll.
    std::atomic<u32> m_queuedHint{0};
    std::atomic<u64> m_generationHint{0};
    std::atomic<bool> m_stopHint{false};

    std::atomic<u64> m_jobsRun{0};
    std::atomic<u64> m_jobBusyNs{0};
};

} // namespace engine
