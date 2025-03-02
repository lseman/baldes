#pragma once
#include <algorithm>
#include <condition_variable>
#include <coroutine>
#include <exec/task.hpp>
#include <iostream>
#include <mutex>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

template <typename T, typename TaskType>
concept HasPerturbation =
    requires(T mother, const std::vector<TaskType>& tasks) {
        { mother.perturbation(tasks) } -> std::same_as<std::vector<TaskType>>;
    };

template <typename TaskType, typename MotherClass>
class TaskQueue {
   public:
    static constexpr size_t MAX_PROCESSED_TASKS =
        1000000;  // Adjust based on your needs

    TaskQueue(int threshold, exec::static_thread_pool::scheduler sched,
              MotherClass& mother)
        : task_threshold(threshold),
          scheduler(sched),
          mother_class(mother),
          shutdown(false) {
        if (threshold <= 0) {
            throw std::invalid_argument("Task threshold must be positive");
        }
        // Optionally, pre-reserve some capacity for the task queue if you have
        // a rough idea. Start the worker thread (using jthread if available).
#ifdef __APPLE__
        worker_thread = std::thread([this] { run_worker(); });
#else
        worker_thread = std::jthread([this] { run_worker(); });
#endif
    }

    ~TaskQueue() {
        stop_worker();
#ifdef __APPLE__
        if (worker_thread.joinable()) {
            worker_thread.join();
        }
#endif
    }

    // Add a new task to the queue with perfect forwarding.
    template <typename T>
    void submit_task(T&& task) {
        {
            std::scoped_lock lock(queue_mutex);
            task_queue.emplace(std::forward<T>(task));
        }
        queue_condition.notify_one();
    }

    // Retrieve processed tasks (moves out the tasks to avoid unnecessary
    // copies).
    std::vector<TaskType> get_processed_tasks() {
        std::scoped_lock lock(processed_mutex);
        return std::move(processed_tasks);
    }

   private:
    // Data members for task queue.
    std::queue<TaskType> task_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_condition;
    bool shutdown;
    const int task_threshold;
    exec::static_thread_pool::scheduler scheduler;
    MotherClass& mother_class;

    std::vector<TaskType> processed_tasks;
    std::mutex processed_mutex;

#ifdef __APPLE__
    std::thread worker_thread;
#else
    std::jthread worker_thread;
#endif

    // Coroutine: Wait until tasks are available or a shutdown is requested.
    exec::task<void> wait_for_tasks() {
        std::unique_lock lock(queue_mutex);
        queue_condition.wait(lock, [this] {
            return shutdown ||
                   (task_queue.size() >= static_cast<size_t>(task_threshold)) ||
                   !task_queue.empty();
        });
        co_return;
    }

    // Coroutine: Process a batch of tasks.
    exec::task<void> process_tasks() {
        // If shutdown is true and no tasks remain, return.
        if (shutdown && task_queue.empty()) {
            co_return;
        }

        // Use a local vector for batch processing.
        std::vector<TaskType> tasks_to_process;
        {
            std::scoped_lock lock(queue_mutex);
            const size_t batch_size = std::min(
                static_cast<size_t>(task_threshold), task_queue.size());
            tasks_to_process.reserve(batch_size);
            for (size_t i = 0; i < batch_size; ++i) {
                tasks_to_process.push_back(std::move(task_queue.front()));
                task_queue.pop();
            }
        }

        if (tasks_to_process.empty()) {
            co_return;
        }

        // Process tasks, applying perturbation if available.
        std::vector<TaskType> result;
        try {
            if constexpr (HasPerturbation<MotherClass, TaskType>) {
                // If perturbation() can work on an rvalue vector, move
                // tasks_to_process.
                result = mother_class.perturbation(std::move(tasks_to_process));
            } else {
                result = std::move(tasks_to_process);
            }
        } catch (const std::exception& e) {
            std::cerr << "Error processing tasks: " << e.what() << std::endl;
            co_return;
        }

        // Store processed results if any.
        if (!result.empty()) {
            std::scoped_lock lock(processed_mutex);
            // Check against maximum allowed capacity.
            if (processed_tasks.size() >
                processed_tasks.max_size() - result.size()) {
                throw std::runtime_error("Exceeding maximum vector size");
            }
            if (processed_tasks.size() + result.size() > MAX_PROCESSED_TASKS) {
                throw std::runtime_error(
                    "Exceeding maximum allowed processed tasks");
            }
            processed_tasks.reserve(processed_tasks.size() + result.size());
            processed_tasks.insert(processed_tasks.end(),
                                   std::make_move_iterator(result.begin()),
                                   std::make_move_iterator(result.end()));
        }
        co_return;
    }

    // Coroutine: Main task worker loop.
    exec::task<void> task_worker() {
        try {
            while (!shutdown || !task_queue.empty()) {
                co_await wait_for_tasks();
                co_await process_tasks();
            }
        } catch (const std::exception& e) {
            std::cerr << "Error in task worker: " << e.what() << std::endl;
        }
        co_return;
    }

    // Worker thread entry point.
    void run_worker() {
        while (!shutdown || !task_queue.empty()) {
            try {
                auto work = task_worker();
                stdexec::sync_wait(std::move(work));
            } catch (const std::exception& e) {
                std::cerr << "Worker encountered an error: " << e.what()
                          << std::endl;
                if (shutdown) break;
            }
        }
    }

    // Signal the worker thread to stop and wake it up.
    void stop_worker() {
        {
            std::lock_guard lock(queue_mutex);
            shutdown = true;
        }
        queue_condition.notify_all();
    }
};
