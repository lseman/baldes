#pragma once
#include <algorithm>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <deque>
#include <format>   // For std::format
#include <functional>
#include <generator> // For std::generator (C++23)
#include <mutex>
#include <numeric>
#include <random>
#include <sched.h> // For thread affinity
#include <span>    // For std::span
#include <thread>
#include <vector>

class WorkStealingPool {
private:
    struct WorkQueue {
        std::deque<std::move_only_function<void()>> tasks; // Move-only tasks
        std::mutex                                  mutex;
        std::condition_variable                     cv;
    };

    std::vector<std::unique_ptr<WorkQueue>> queues;
    std::vector<std::jthread>               threads; // Automatically joins on destruction
    std::atomic<bool>                       running{true};
    std::atomic<size_t>                     active_tasks{0};
    static thread_local size_t              thread_id;

public:
    explicit WorkStealingPool(size_t num_threads = std::thread::hardware_concurrency()) { init_threads(num_threads); }

    ~WorkStealingPool() { shutdown(); }

    // Submit a single task
    template <typename F>
    void submit(F &&f) {
        size_t queue_idx = thread_id < queues.size() ? thread_id : std::random_device{}() % queues.size();
        auto  &queue     = *queues[queue_idx];
        {
            std::lock_guard<std::mutex> lock(queue.mutex);
            queue.tasks.emplace_back([f = std::forward<F>(f), this]() mutable {
                try {
                    std::invoke(f);
                } catch (const std::exception &e) {
                    std::cerr << std::format("Task failed: {}\n", e.what());
                } catch (...) {
                    std::cerr << "Task failed with unknown exception\n";
                }
                active_tasks.fetch_sub(1, std::memory_order_release);
            });
        }
        active_tasks.fetch_add(1, std::memory_order_acquire);
        queue.cv.notify_one();
    }

    // Submit a batch of tasks
    template <typename F>
    void submit_batch(std::span<F> tasks) {
        size_t queue_idx = thread_id < queues.size() ? thread_id : std::random_device{}() % queues.size();
        auto  &queue     = *queues[queue_idx];
        {
            std::lock_guard<std::mutex> lock(queue.mutex);
            for (const auto &f : tasks) {
                queue.tasks.emplace_back([f, this]() mutable {
                    try {
                        std::invoke(f);
                    } catch (const std::exception &e) {
                        std::cerr << std::format("Task failed: {}\n", e.what());
                    } catch (...) {
                        std::cerr << "Task failed with unknown exception\n";
                    }
                    active_tasks.fetch_sub(1, std::memory_order_release);
                });
            }
        }
        active_tasks.fetch_add(tasks.size(), std::memory_order_acquire);
        queue.cv.notify_all();
    }

    // Wait until all tasks are completed
    void wait_idle() {
        while (active_tasks.load(std::memory_order_acquire) > 0) {
            std::this_thread::yield();
        }
    }

    // Resize the thread pool
    void resize(size_t new_size) {
        shutdown();
        init_threads(new_size);
    }

    // Get the number of threads in the pool
    size_t size() const noexcept { return queues.size(); }

    // Get the number of tasks in a specific queue
    size_t queue_size(size_t queue_idx) const {
        if (queue_idx >= queues.size()) return 0;
        std::lock_guard<std::mutex> lock(queues[queue_idx]->mutex);
        return queues[queue_idx]->tasks.size();
    }

private:
    // Initialize threads and queues
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
                pin_thread_to_core(i); // Pin thread to core for better performance
                run();
            });
        }
    }

    // Shutdown the thread pool
    void shutdown() {
        running.store(false, std::memory_order_release);

        // Notify all queues to wake up threads
        for (auto &queue : queues) {
            if (queue) {
                std::lock_guard<std::mutex> lock(queue->mutex);
                queue->cv.notify_all();
            }
        }

        // Wait for all threads to finish
        for (auto &thread : threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }

        threads.clear();
        queues.clear();
    }

    // Main worker thread loop
    void run() {
        std::mt19937 rng(std::random_device{}());

        while (running.load(std::memory_order_acquire)) {
            std::move_only_function<void()> task;
            if (try_pop_task_from_local_queue(task) || try_steal_task(task, rng)) {
                task();
            } else {
                auto &queue = *queues[thread_id];
                std::unique_lock<std::mutex> lock(queue.mutex);
                queue.cv.wait_for(lock, std::chrono::milliseconds(1), [this, &queue] {
                    return !queue.tasks.empty() || !running.load(std::memory_order_acquire);
                });
            }
        }
    }

    // Try to pop a task from the local queue
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

    // Try to steal a task from another queue
    bool try_steal_task(std::move_only_function<void()> &task, std::mt19937 &rng) {
        std::vector<size_t> indices(queues.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        for (size_t idx : indices) {
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

    // Pin thread to a specific CPU core
    void pin_thread_to_core(size_t core_id) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core_id, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    }
};