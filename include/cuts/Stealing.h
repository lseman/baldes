#pragma once
#include "RNG.h"
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

class WorkStealingPool {
private:
    struct WorkQueue {
        std::deque<std::function<void()>> tasks;
        std::mutex                        mutex;
        std::condition_variable           cv;
    };

    std::vector<std::unique_ptr<WorkQueue>> queues;
    std::vector<std::thread>                threads;
    std::atomic<bool>                       running{true};
    std::atomic<size_t>                     active_tasks{0};
    static thread_local size_t              thread_id;
    std::condition_variable                 idle_cv;
    std::mutex                              idle_mutex;
    std::mutex                              resize_mutex;

public:
    explicit WorkStealingPool(size_t num_threads = std::thread::hardware_concurrency()) { init_threads(num_threads); }

    ~WorkStealingPool() { shutdown(); }

    template <typename F>
    void submit(F &&f) {
        size_t queue_idx = thread_id < queues.size() ? thread_id : std::random_device{}() % queues.size();

        auto &queue = *queues[queue_idx];
        {
            std::lock_guard<std::mutex> lock(queue.mutex);
            queue.tasks.emplace_back([f = std::forward<F>(f), this]() mutable {
                try {
                    std::invoke(f);
                } catch (...) {
                    // Log or handle exception
                }
                active_tasks.fetch_sub(1, std::memory_order_release);
                idle_cv.notify_one();
            });
        }
        active_tasks.fetch_add(1, std::memory_order_acquire);
        queue.cv.notify_one();
    }

    void wait_idle() {
        std::unique_lock<std::mutex> lock(idle_mutex);
        idle_cv.wait(lock, [this] { return active_tasks.load(std::memory_order_acquire) == 0; });
    }

    void resize(size_t new_size) {
        std::lock_guard<std::mutex> lock(resize_mutex);
        shutdown();
        init_threads(new_size);
    }

    size_t size() const noexcept { return queues.size(); }

    size_t queue_size(size_t queue_idx) const {
        if (queue_idx >= queues.size()) return 0;
        std::lock_guard<std::mutex> lock(queues[queue_idx]->mutex);
        return queues[queue_idx]->tasks.size();
    }

private:
    void init_threads(size_t num_threads) {
        queues.resize(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            queues[i] = std::make_unique<WorkQueue>();
            assert(queues[i] != nullptr && "Failed to initialize WorkQueue");
        }

        threads.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            threads.emplace_back([this, i] {
                thread_id = i;
                run();
            });
        }
    }

    void shutdown() {
        wait_idle();
        running.store(false, std::memory_order_release);

        for (auto &queue : queues) {
            if (queue) queue->cv.notify_all();
        }

        for (auto &thread : threads) {
            if (thread.joinable()) { thread.join(); }
        }

        threads.clear();
        queues.clear();
    }

    void run() {
        Xoroshiro128Plus rng; // You can use any seed you prefer

        while (running.load(std::memory_order_acquire)) {
            std::function<void()> task;
            if (try_pop_task_from_local_queue(task) || try_steal_task(task, rng)) {
                task();
            } else {
                std::unique_lock<std::mutex> lock(queues[thread_id]->mutex);
                queues[thread_id]->cv.wait_for(lock, std::chrono::milliseconds(1), [this] {
                    return !queues[thread_id]->tasks.empty() || !running.load(std::memory_order_acquire);
                });
            }
        }
    }

    bool try_pop_task_from_local_queue(std::function<void()> &task) {
        auto                       &queue = *queues[thread_id];
        std::lock_guard<std::mutex> lock(queue.mutex);
        if (!queue.tasks.empty()) {
            task = std::move(queue.tasks.front());
            queue.tasks.pop_front();
            return true;
        }
        return false;
    }

    bool try_steal_task(std::function<void()> &task, Xoroshiro128Plus &rng) {
        std::uniform_int_distribution<size_t> dist(0, queues.size() - 1);
        size_t                                start_idx = dist(rng);

        for (size_t i = 0; i < queues.size(); ++i) {
            size_t idx = (start_idx + i) % queues.size();
            if (idx == thread_id) continue;

            auto                       &queue = *queues[idx];
            std::lock_guard<std::mutex> lock(queue.mutex);
            if (!queue.tasks.empty()) {
                task = std::move(queue.tasks.back());
                queue.tasks.pop_back();
                return true;
            }
        }
        return false;
    }
};