#include "engine/core/ThreadPool.hpp"

#include "engine/core/Profile.hpp"

namespace engine {

ThreadPool::ThreadPool(u32 workers)
{
    if (workers == 0) {
        const u32 hw = std::thread::hardware_concurrency();
        workers = hw > 1 ? hw - 1 : 0;
    }
    m_threads.reserve(workers);
    for (u32 i = 0; i < workers; ++i) {
        m_threads.emplace_back([this] { workerLoop(); });
    }
}

ThreadPool::~ThreadPool()
{
    {
        std::lock_guard lock(m_mutex);
        m_stop = true;
    }
    m_wake.notify_all();
    for (std::thread& t : m_threads) {
        t.join();
    }
}

void ThreadPool::run(const Task& task)
{
    std::lock_guard submitLock(m_submit);
    {
        std::unique_lock lock(m_mutex);
        // A worker that woke late for the previous task may still hold a copy
        // of it. It will find no indices left, but resetting m_next under it
        // would let it run the old function on new indices, so wait it out.
        m_idle.wait(lock, [this] { return m_busy == 0; });
        m_task = task;
        m_next.store(0, std::memory_order_relaxed);
        ++m_generation;
    }
    m_wake.notify_all();

    process(task); // the caller works too

    // All indices are claimed once process() returns, but workers may still be
    // finishing theirs. Wait for them before the task (and fn) goes out of scope.
    std::unique_lock lock(m_mutex);
    m_idle.wait(lock, [this] { return m_busy == 0; });
}

void ThreadPool::process(const Task& task)
{
    for (;;) {
        const u32 begin = m_next.fetch_add(task.grain, std::memory_order_relaxed);
        if (begin >= task.count) {
            return;
        }
        task.invoke(task.ctx, begin, std::min(begin + task.grain, task.count));
    }
}

void ThreadPool::submit(Job job)
{
    if (m_threads.empty()) {
        job();
        return;
    }
    {
        std::lock_guard lock(m_mutex);
        m_jobs.push_back(std::move(job));
    }
    m_wake.notify_one();
}

u32 ThreadPool::queuedJobs() const
{
    std::lock_guard lock(m_mutex);
    return static_cast<u32>(m_jobs.size());
}

void ThreadPool::workerLoop()
{
    u64 seen = 0;
    for (;;) {
        Task task;
        Job job;
        {
            std::unique_lock lock(m_mutex);
            m_wake.wait(lock, [&] { return m_stop || m_generation != seen || !m_jobs.empty(); });
            if (m_stop) {
                return;
            }
            if (m_generation != seen) {
                // A parallelFor takes priority over queued jobs.
                seen = m_generation;
                task = m_task;
                ++m_busy; // run() cannot return (or start another task) until we are done
            } else {
                job = std::move(m_jobs.front());
                m_jobs.pop_front();
            }
        }

        if (job) {
            ENGINE_PROFILE_SCOPE("ThreadPool job");
            job();
            continue;
        }

        {
            ENGINE_PROFILE_SCOPE("ThreadPool task");
            process(task);
        }

        bool last = false;
        {
            std::lock_guard lock(m_mutex);
            last = --m_busy == 0;
        }
        if (last) {
            m_idle.notify_one();
        }
    }
}

} // namespace engine
