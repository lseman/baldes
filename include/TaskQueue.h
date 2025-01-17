#pragma once
#include <condition_variable>
#include <coroutine>
#include <exec/task.hpp>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <stdexcept>

template <typename T, typename TaskType>
concept HasPerturbation = requires(T mother, const std::vector<TaskType>& tasks) {
    { mother.perturbation(tasks) } -> std::same_as<std::vector<TaskType>>;
};

template <typename TaskType, typename MotherClass>
class TaskQueue {
public:
    static constexpr size_t MAX_PROCESSED_TASKS = 1000000; // Adjust based on your needs

    TaskQueue(int threshold, exec::static_thread_pool::scheduler sched, MotherClass& mother)
        : task_threshold(threshold), scheduler(sched), mother_class(mother), shutdown(false) {
        if (threshold <= 0) {
            throw std::invalid_argument("Task threshold must be positive");
        }
        
        // Start the worker thread
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

    // Add a new task to the queue with perfect forwarding
    template <typename T>
    void submit_task(T&& task) {
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

    // Wait for enough tasks to be available
    exec::task<void> wait_for_tasks() {
        std::unique_lock<std::mutex> lock(queue_mutex);
        queue_condition.wait(lock, [this] { 
            return task_queue.size() >= task_threshold || shutdown || !task_queue.empty(); 
        });
        co_return;
    }

    // Process a batch of tasks
    exec::task<void> process_tasks() {
        // Early return if shutdown
        if (shutdown && task_queue.empty()) {
            co_return;
        }

        std::vector<TaskType> tasks_to_process;
        tasks_to_process.reserve(task_threshold); // Pre-allocate space

        // Get tasks from queue
        {
            std::scoped_lock lock(queue_mutex);
            const size_t batch_size = std::min(
                static_cast<size_t>(task_threshold), 
                task_queue.size()
            );
            
            for (size_t i = 0; i < batch_size; ++i) {
                tasks_to_process.push_back(std::move(task_queue.front()));
                task_queue.pop();
            }
        }

        if (tasks_to_process.empty()) {
            co_return;
        }

        // Process tasks
        std::vector<TaskType> result;
        try {
            if constexpr (HasPerturbation<MotherClass, TaskType>) {
                result = mother_class.perturbation(std::move(tasks_to_process));
            } else {
                result = std::move(tasks_to_process);
            }
        } catch (const std::exception& e) {
            // Log or handle the error appropriately
            co_return;
        }

        // Store results
        if (!result.empty()) {
            std::scoped_lock lock(processed_mutex);
            
            // Check capacity
            if (processed_tasks.size() + result.size() > processed_tasks.max_size()) {
                throw std::runtime_error("Would exceed maximum vector size");
            }

            // Check against MAX_PROCESSED_TASKS
            if (processed_tasks.size() + result.size() > MAX_PROCESSED_TASKS) {
                // Optional: could clear processed_tasks here if needed
                // processed_tasks.clear();
                throw std::runtime_error("Would exceed maximum allowed processed tasks");
            }

            try {
                processed_tasks.reserve(processed_tasks.size() + result.size());
                processed_tasks.insert(
                    processed_tasks.end(),
                    std::make_move_iterator(result.begin()),
                    std::make_move_iterator(result.end())
                );
            } catch (const std::exception& e) {
                // Handle or rethrow as needed
                throw;
            }
        }

        co_return;
    }

    // Continuous task processing loop using coroutines
    exec::task<void> task_worker() {
        try {
            while (!shutdown || !task_queue.empty()) {
                co_await wait_for_tasks();
                co_await process_tasks();
            }
        } catch (const std::exception& e) {
            // Log or handle error appropriately
        }
        co_return;
    }

    void run_worker() {
        while (!shutdown || !task_queue.empty()) {
            try {
                auto work = task_worker();
                stdexec::sync_wait(std::move(work));
            } catch (const std::exception& e) {
                // Log or handle error appropriately
                if (shutdown) break;
            }
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