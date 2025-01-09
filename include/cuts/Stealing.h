#pragma once
#include "RNG.h"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <deque>
#include <expected>
#include <format>
#include <functional>
#include <generator>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <sched.h>
#include <span>
#include <thread>
#include <vector>

enum class PoolError { TaskFailed, QueueFull, ThreadCreationFailed };

class WorkStealingPool {
private:
    static constexpr size_t QUEUE_SIZE_LIMIT = 10000;

    struct alignas(64) WorkQueue {
        std::deque<std::move_only_function<void()>> tasks;
        mutable std::mutex                          mutex;
        std::condition_variable                     cv;
        std::atomic<size_t>                         steal_attempts{0};
        std::atomic<size_t>                         successful_steals{0};
        std::atomic<size_t>                         local_processed{0};
        std::atomic<size_t>                         task_count{0}; // Track the number of tasks

        bool try_push(std::move_only_function<void()> &&task) {
            std::lock_guard lock(mutex);
            if (tasks.size() >= QUEUE_SIZE_LIMIT) return false;
            tasks.push_back(std::move(task));
            task_count.fetch_add(1, std::memory_order_relaxed);
            cv.notify_one();
            return true;
        }

        bool try_pop(std::move_only_function<void()> &task) {
            std::lock_guard lock(mutex);
            if (tasks.empty()) return false;
            task = std::move(tasks.front());
            tasks.pop_front();
            task_count.fetch_sub(1, std::memory_order_relaxed);
            local_processed++;
            return true;
        }

        bool try_steal(std::move_only_function<void()> &task) {
            std::lock_guard lock(mutex);
            if (tasks.empty()) return false;
            task = std::move(tasks.back());
            tasks.pop_back();
            task_count.fetch_sub(1, std::memory_order_relaxed);
            return true;
        }
    };

    std::vector<std::unique_ptr<WorkQueue>> queues;
    std::vector<std::jthread>               threads;
    std::atomic<bool>                       running{true};
    std::atomic<size_t>                     active_tasks{0};
    std::atomic<size_t>                     total_tasks_completed{0};
    static thread_local size_t              thread_id;

public:
    explicit WorkStealingPool(size_t num_threads = std::thread::hardware_concurrency()) { init_threads(num_threads); }

    ~WorkStealingPool() { shutdown(); }

    template <typename F>
    bool submit(F &&f) {
        if (!running.load(std::memory_order_acquire)) return false;

        const size_t queue_idx = thread_id < queues.size() ? thread_id : std::random_device{}() % queues.size();

        auto task_wrapper = [f = std::forward<F>(f), this]() mutable {
            std::invoke(f);
            total_tasks_completed.fetch_add(1, std::memory_order_relaxed);
            active_tasks.fetch_sub(1, std::memory_order_release);
        };

        auto &queue = *queues[queue_idx];
        if (!queue.try_push(std::move(task_wrapper))) {
            metrics.queue_overflow_count.fetch_add(1, std::memory_order_relaxed);
            return false;
        }

        active_tasks.fetch_add(1, std::memory_order_acquire);
        return true;
    }

    struct PoolMetrics {
        size_t tasks_processed;
        size_t steal_attempts;
        size_t successful_steals;
        size_t queue_overflow_count;
        double steal_success_rate;
    };

    struct alignas(64) Metrics {
        std::atomic<size_t>   tasks_processed{0};
        std::atomic<size_t>   steal_attempts{0};
        std::atomic<size_t>   successful_steals{0};
        std::atomic<size_t>   queue_overflow_count{0};
        std::atomic<uint64_t> total_task_latency_ns{0};
        std::atomic<size_t>   task_count_for_latency{0};
    } metrics;

    void wait_idle() {
        while (active_tasks.load(std::memory_order_acquire) > 0) { std::this_thread::yield(); }
    }

    void resize(size_t new_size) {
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
        std::ranges::generate(queues, [] { return std::make_unique<WorkQueue>(); });

        threads.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            threads.emplace_back([this, i] {
                thread_id = i;
                pin_thread_to_core(i);
                run();
            });
        }
    }

    void run() {
        Xoroshiro128Plus rng;

        while (running.load(std::memory_order_acquire)) {
            std::move_only_function<void()> task;
            if (try_pop_task_from_local_queue(task) || try_steal_task(task, rng)) {
                task();
            } else {
                auto            &queue = *queues[thread_id];
                std::unique_lock lock(queue.mutex);
                queue.cv.wait_for(lock, std::chrono::milliseconds(1), [this, &queue] {
                    return !queue.tasks.empty() || !running.load(std::memory_order_acquire);
                });
            }
        }
    }

    void shutdown() {
        running.store(false, std::memory_order_release);

        for (auto &queue : queues) {
            if (queue) {
                std::lock_guard<std::mutex> lock(queue->mutex);
                queue->cv.notify_all();
            }
        }

        for (auto &thread : threads) {
            if (thread.joinable()) { thread.join(); }
        }

        threads.clear();
        queues.clear();
    }

    bool try_pop_task_from_local_queue(std::move_only_function<void()> &task) {
        auto                       &queue = *queues[thread_id];
        std::lock_guard<std::mutex> lock(queue.mutex);
        if (!queue.tasks.empty()) {
            task = std::move(queue.tasks.front());
            queue.tasks.pop_front();
            return true;
        }
        return false;
    }

    bool try_steal_task(std::move_only_function<void()> &task, Xoroshiro128Plus &rng) {
        // Create a vector of queue indices
        std::vector<size_t> indices(queues.size());
        std::iota(indices.begin(), indices.end(), 0);

        // Sort indices by task count (descending order)
        std::ranges::sort(indices, [this](size_t a, size_t b) {
            return queues[a]->task_count.load(std::memory_order_relaxed) >
                   queues[b]->task_count.load(std::memory_order_relaxed);
        });

        // Limit the number of steal attempts
        constexpr size_t MAX_STEAL_ATTEMPTS = 3;
        size_t           attempts           = 0;

        for (size_t idx : indices) {
            if (idx == thread_id) continue; // Skip the local queue

            auto &queue = *queues[idx];
            metrics.steal_attempts.fetch_add(1, std::memory_order_relaxed);
            queue.steal_attempts.fetch_add(1, std::memory_order_relaxed);

            // Try to lock the queue without blocking
            std::unique_lock<std::mutex> lock(queue.mutex, std::try_to_lock);
            if (!lock.owns_lock()) {
                continue; // Skip if the queue is already locked
            }

            if (!queue.tasks.empty()) {
                task = std::move(queue.tasks.back());
                queue.tasks.pop_back();
                queue.task_count.fetch_sub(1, std::memory_order_relaxed);
                metrics.successful_steals.fetch_add(1, std::memory_order_relaxed);
                queue.successful_steals.fetch_add(1, std::memory_order_relaxed);
                return true; // Successfully stole a task
            }

            if (++attempts >= MAX_STEAL_ATTEMPTS) {
                break; // Stop after a limited number of attempts
            }
        }
        return false; // No task was stolen
    }
    void pin_thread_to_core(size_t core_id) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core_id, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    }
};