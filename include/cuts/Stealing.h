#pragma once
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

#include "RNG.h"

class WorkStealingPool {
   private:
    struct WorkQueue {
        std::deque<std::function<void()>> tasks;
        std::mutex mutex;
        std::condition_variable cv;
        std::atomic<bool> active{true};
        std::chrono::steady_clock::time_point last_activity;
    };

    std::vector<std::unique_ptr<WorkQueue>> queues;
    std::vector<std::thread> threads;
    std::atomic<bool> running{true};
    std::atomic<size_t> active_tasks{0};
    std::atomic<size_t> active_threads{0};
    static thread_local size_t thread_id;
    std::condition_variable idle_cv;
    std::mutex idle_mutex;

    static constexpr auto IDLE_TIMEOUT = std::chrono::seconds(5);
    static constexpr size_t MIN_ACTIVE_THREADS = 1;

   public:
    explicit WorkStealingPool(
        size_t num_threads = std::thread::hardware_concurrency()) {
        if (num_threads == 0) num_threads = 1;
        init_threads(num_threads);
    }

    ~WorkStealingPool() { shutdown(); }

    template <typename F>
    bool submit(F&& f, size_t max_queue_size = 1000) {
        size_t queue_idx = thread_id;
        if (queue_idx >= queues.size()) {
            std::random_device rd;
            queue_idx = rd() % queues.size();
        }

        auto& queue = *queues[queue_idx];
        std::lock_guard<std::mutex> lock(queue.mutex);

        if (queue.tasks.size() >= max_queue_size) {
            return false;
        }

        queue.tasks.emplace_back([f = std::forward<F>(f), this]() mutable {
            try {
                std::invoke(f);
            } catch (...) {
                // Log exception
            }
            if (active_tasks.fetch_sub(1, std::memory_order_release) == 1) {
                idle_cv.notify_all();
            }
        });

        active_tasks.fetch_add(1, std::memory_order_acquire);
        queue.last_activity = std::chrono::steady_clock::now();
        queue.cv.notify_one();

        // Wake an idle thread if needed
        if (!queue.active.load(std::memory_order_acquire)) {
            queue.active.store(true, std::memory_order_release);
        }

        return true;
    }

    bool wait_idle(
        std::chrono::milliseconds timeout = std::chrono::milliseconds::max()) {
        std::unique_lock<std::mutex> lock(idle_mutex);
        return idle_cv.wait_for(lock, timeout, [this] {
            return active_tasks.load(std::memory_order_acquire) == 0;
        });
    }

    size_t size() const noexcept { return queues.size(); }

    size_t active_thread_count() const noexcept {
        return active_threads.load(std::memory_order_acquire);
    }

    size_t queue_size(size_t queue_idx) const noexcept {
        if (queue_idx >= queues.size()) return 0;
        std::lock_guard<std::mutex> lock(queues[queue_idx]->mutex);
        return queues[queue_idx]->tasks.size();
    }

   private:
    void init_threads(size_t num_threads) {
        running.store(true, std::memory_order_release);
        active_tasks.store(0, std::memory_order_release);
        active_threads.store(0, std::memory_order_release);

        queues.clear();
        queues.reserve(num_threads);

        for (size_t i = 0; i < num_threads; ++i) {
            queues.push_back(std::make_unique<WorkQueue>());
        }

        threads.clear();
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

        for (auto& queue : queues) {
            queue->active.store(true, std::memory_order_release);
            queue->cv.notify_all();
        }

        for (auto& thread : threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }

        threads.clear();
        queues.clear();
    }

    void run() {
        Xoroshiro128Plus rng;
        active_threads.fetch_add(1, std::memory_order_release);

        while (running.load(std::memory_order_acquire)) {
            std::function<void()> task;
            const size_t my_id = thread_id;

            if (my_id >= queues.size()) {
                break;
            }

            auto& my_queue = *queues[my_id];

            if (try_pop_task_from_local_queue(task, my_queue) ||
                try_steal_task(task, my_id, rng)) {
                my_queue.last_activity = std::chrono::steady_clock::now();
                task();
                continue;
            }

            std::unique_lock<std::mutex> lock(my_queue.mutex);

            if (should_thread_sleep(my_id)) {
                my_queue.active.store(false, std::memory_order_release);
                active_threads.fetch_sub(1, std::memory_order_release);

                my_queue.cv.wait(lock, [this, &my_queue] {
                    return my_queue.active.load(std::memory_order_acquire) ||
                           !running.load(std::memory_order_acquire);
                });

                if (my_queue.active.load(std::memory_order_acquire)) {
                    active_threads.fetch_add(1, std::memory_order_release);
                }
            } else {
                my_queue.cv.wait_for(lock, std::chrono::milliseconds(1));
            }
        }

        active_threads.fetch_sub(1, std::memory_order_release);
    }

    bool should_thread_sleep(size_t thread_id) const {
        if (active_threads.load(std::memory_order_acquire) <=
            MIN_ACTIVE_THREADS) {
            return false;
        }

        if (thread_id >= queues.size()) {
            return true;
        }

        auto& queue = *queues[thread_id];
        auto now = std::chrono::steady_clock::now();
        return now - queue.last_activity > IDLE_TIMEOUT;
    }

    bool try_pop_task_from_local_queue(std::function<void()>& task,
                                       WorkQueue& queue) {
        std::lock_guard<std::mutex> lock(queue.mutex);
        if (!queue.tasks.empty()) {
            task = std::move(queue.tasks.front());
            queue.tasks.pop_front();
            return true;
        }
        return false;
    }

    bool try_steal_task(std::function<void()>& task, size_t my_id,
                        Xoroshiro128Plus& rng) {
        if (queues.empty()) return false;

        std::uniform_int_distribution<size_t> dist(0, queues.size() - 1);
        size_t start_idx = dist(rng);

        for (size_t i = 0; i < queues.size(); ++i) {
            size_t idx = (start_idx + i) % queues.size();
            if (idx == my_id) continue;

            auto& queue = *queues[idx];
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
