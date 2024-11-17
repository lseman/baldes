
#pragma once
#include <condition_variable>
#include <coroutine>
#include <exec/task.hpp>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

template <typename T, typename TaskType>
concept HasPerturbation = requires(T mother, const std::vector<TaskType> &tasks) {
    { mother.perturbation(tasks) } -> std::same_as<std::vector<TaskType>>;
};

template <typename TaskType, typename MotherClass>
class TaskQueue {
public:
    TaskQueue(int threshold, exec::static_thread_pool::scheduler sched, MotherClass &mother)
        : task_threshold(threshold), scheduler(sched), mother_class(mother), shutdown(false) {
// Start the worker thread
#ifdef __APPLE__
        worker_thread = std::thread([this] { run_worker(); });
#else
        // use jthread
        worker_thread = std::jthread([this] { run_worker(); });
#endif
    }

    ~TaskQueue() {
        stop_worker();
#ifdef __APPLE__
        if (worker_thread.joinable()) worker_thread.join();
#endif
    }

    // Add a new task to the queue with perfect forwarding
    template <typename T>
    void submit_task(T &&task) {
        {
            std::scoped_lock lock(queue_mutex);
            task_queue.emplace(std::forward<T>(task));
        }
        queue_condition.notify_one();
    }

    // Retrieve processed tasks
    std::vector<TaskType> get_processed_tasks() {
        std::scoped_lock lock(processed_mutex);
        return std::move(processed_tasks);
    }

private:
    // Task queue and synchronization primitives
    std::queue<TaskType>                task_queue;
    std::mutex                          queue_mutex;
    std::condition_variable             queue_condition;
    bool                                shutdown;
    const int                           task_threshold;
    exec::static_thread_pool::scheduler scheduler;
    MotherClass                        &mother_class;

    std::vector<TaskType> processed_tasks;
    std::mutex            processed_mutex;

#ifdef __APPLE__
    std::thread worker_thread;
#else
    std::jthread worker_thread;
#endif

    // Wait for enough tasks to be available
    exec::task<void> wait_for_tasks() {
        std::unique_lock<std::mutex> lock(queue_mutex);
        queue_condition.wait(lock, [this] { return task_queue.size() >= task_threshold || shutdown; });
        co_return;
    }

    // Process a batch of tasks
    exec::task<void> process_tasks() {
        std::vector<TaskType> tasks_to_process;
        {
            std::scoped_lock lock(queue_mutex);
            for (int i = 0; i < task_threshold && !task_queue.empty(); ++i) {
                tasks_to_process.push_back(std::move(task_queue.front()));
                task_queue.pop();
            }
        }
        auto result = tasks_to_process;
        // Process tasks using the mother class's perturbation method
        if constexpr (HasPerturbation<MotherClass, TaskType>) {
            result = mother_class.perturbation(tasks_to_process);

            // Store the results in processed_tasks
            {
                std::scoped_lock lock(processed_mutex);
                processed_tasks.insert(processed_tasks.end(), std::make_move_iterator(result.begin()),
                                       std::make_move_iterator(result.end()));
            }
        }

        // Store the results in processed_tasks
        {
            std::scoped_lock lock(processed_mutex);
            processed_tasks.insert(processed_tasks.end(), std::make_move_iterator(result.begin()),
                                   std::make_move_iterator(result.end()));
        }
        co_return;
    }

    // Continuous task processing loop using coroutines
    exec::task<void> task_worker() {
        while (!shutdown) {
            co_await wait_for_tasks();
            co_await process_tasks();
        }
        co_return;
    }

    void run_worker() {
        while (!shutdown) {
            auto work = task_worker();
            stdexec::sync_wait(std::move(work));
        }
    }

    void stop_worker() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            shutdown = true;
        }
        queue_condition.notify_all();
    }
};
