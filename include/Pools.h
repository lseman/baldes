/**
 * @file Pools.h
 * @brief Defines classes to manage pools of paths and labels.
 *
 * This file defines the SchrodingerPool and LabelPool classes, which manage
 * pools of paths and labels, respectively. The SchrodingerPool class manages a
 * collection of paths with a limited lifespan, computes reduced costs, and
 * filters paths based on their reduced costs. The LabelPool class manages a
 * pool of Label objects, providing methods to acquire and release labels from
 * the pool, as well as resetting the pool to its initial state.
 *
 */

#pragma once

#include <deque>
#include <exec/static_thread_pool.hpp>
#include <exec/task.hpp>
#include <stdexec/execution.hpp>

#include "Common.h"
#include "Label.h"
#include "Path.h"
#include "TaskQueue.h"
#include "VRPNode.h"
/**
 * @class SchrodingerPool
 * @brief Manages a pool of paths with a limited lifespan, computes reduced
 * costs, and filters paths based on their reduced costs.
 *
 * This class is designed to manage a collection of paths, each associated with
 * an iteration when it was added. Paths have a limited lifespan, defined by
 * `max_live_time`, after which they are removed from the pool. The class also
 * provides functionality to compute reduced costs for paths and filter paths
 * based on their reduced costs.
 *
 */
class SchrodingerPool {
   private:
    std::deque<std::tuple<int, Path>>
        paths;  // Stores tuples of (iteration added, Path)
    int current_iteration = 0;
    int max_live_time;          // Max iterations a Path can stay active
    std::vector<double> duals;  // Dual variables for each path
    std::vector<VRPNode> *nodes = nullptr;

   public:
    std::vector<std::vector<double>>
        distance_matrix;  // Distance matrix for the graph

    exec::static_thread_pool pool = exec::static_thread_pool(5);
    exec::static_thread_pool::scheduler sched = pool.get_scheduler();

    // SchrodingerPool(int live_time) : max_live_time(live_time) {}
    SchrodingerPool(int live_time)
        : max_live_time(live_time),
          pool(5),
          task_queue(5, pool.get_scheduler(), *this) {}

    ~SchrodingerPool() {}

    TaskQueue<Path, SchrodingerPool> task_queue;

    void setNodes(std::vector<VRPNode> *nodes) { this->nodes = nodes; }

    int getcij(int i, int j) { return distance_matrix[i][j]; }

    void remove_old_paths() {
        // Remove old paths that have lived beyond their allowed time
        while (!paths.empty() && std::get<0>(paths.front()) + max_live_time <=
                                     current_iteration) {
            paths.pop_front();  // Remove the oldest path
        }
    }

    void add_path(const Path &path) {
        // Add new path with the current iteration
        paths.push_back({current_iteration, path});
    }

    // Submit a task to add new paths
    void add_paths(const std::vector<Path> &new_paths) {
        for (const Path &path : new_paths) {
            task_queue.submit_task(path);
        }
    }

    void computeRC() {
        for (auto &path : paths) {
            int iteration_added =
                std::get<0>(path);  // Get the iteration when the path was added

            // Stop processing if the path is older than current_iteration +
            // max_life
            if (iteration_added + max_live_time < current_iteration) {
                break;
            }

            Path &p = std::get<1>(path);
            p.red_cost = p.cost;

            if (p.size() > 3) {
                for (int i = 1; i < p.size() - 1; i++) {
                    auto &node =
                        (*nodes)[p[i]];  // Dereference nodes and access element
                    p.red_cost -= node.cost;
                }
            }
        }
    }

    std::vector<Path> perturbation(const std::vector<Path> &paths_to_process) {
        remove_old_paths();
        for (const Path &path : paths_to_process) {
            add_path(path);
        }
        computeRC();
        auto result = _get_paths_with_negative_red_cost();
        return result;
    }

    std::vector<Path> _get_paths_with_negative_red_cost() {
        std::vector<Path> result;

        // Remove the paths that are older than current_iteration + max_life or
        // have a negative red_cost
        auto it = paths.begin();
        while (it != paths.end()) {
            int iteration_added =
                std::get<0>(*it);  // Get the iteration when the path was added
            const Path &p = std::get<1>(*it);

            // If the path is older than max_live_time, or has a negative
            // red_cost
            if (iteration_added + max_live_time < current_iteration ||
                p.red_cost < 0) {
                if (p.red_cost < 0) {
                    result.push_back(
                        p);  // Add paths with negative red_cost to the result
                }
                it = paths.erase(it);  // Remove from paths and move iterator to
                                       // the next element
            } else {
                ++it;  // Only move the iterator if not erasing
            }
        }

        // Sort the result based on red_cost
        pdqsort(result.begin(), result.end(), [](const Path &a, const Path &b) {
            return a.red_cost < b.red_cost;
        });

        return result;
    }

    std::vector<Path> get_paths() {
        auto tasks = task_queue.get_processed_tasks();
        std::vector<Path> result;
        result.insert(result.end(), tasks.begin(), tasks.end());
        return result;
    }

    void iterate() { current_iteration++; }
};

/**
 * @class LabelPool
 * @brief A class that manages a pool of Label objects.
 *
 * The LabelPool class is responsible for managing a pool of Label objects. It
 * provides methods to acquire and release labels from the pool, as well as
 * resetting the pool to its initial state.
 *
 * The pool size is determined during construction and can be optionally limited
 * to a maximum size. Labels can be acquired from the pool using the `acquire()`
 * method, and released back to the pool using the `release()` method. If the
 * pool is full, a new label will be allocated.
 *
 */
class LabelPool {
   private:
    static constexpr size_t BLOCK_SIZE = 128;  // Adjust based on profiling

    // Each MemoryBlock holds a fixed number of Label objects.
    struct MemoryBlock {
        std::unique_ptr<Label[]> data;
        size_t used = 0;  // Number of labels already allocated from this block

        MemoryBlock() : data(std::make_unique<Label[]>(BLOCK_SIZE)), used(0) {}
    };

    size_t pool_size;
    size_t max_pool_size;
    std::vector<MemoryBlock> memory_blocks;

    // free_list holds pointers to labels that are available for reuse.
    std::vector<Label *> free_list;

    // in_use_labels holds pointers to labels that are currently acquired.
    std::vector<Label *> in_use_labels;

    // Allocate 'count' new labels and add them to the free list.
    void allocate_labels(size_t count) {
        size_t blocks_needed = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
        // Reserve capacity for the new memory blocks and free list expansion.
        memory_blocks.reserve(memory_blocks.size() + blocks_needed);
        free_list.reserve(free_list.size() + count);

        for (size_t b = 0; b < blocks_needed; ++b) {
            memory_blocks.emplace_back();
            auto &block = memory_blocks.back();
            // Calculate how many labels to allocate in this block.
            size_t alloc_in_block =
                std::min(count - b * BLOCK_SIZE, BLOCK_SIZE);
            for (size_t i = 0; i < alloc_in_block; ++i) {
                free_list.push_back(&block.data[i]);
            }
            block.used = alloc_in_block;
        }
    }

   public:
    explicit LabelPool(size_t initial_pool_size, size_t max_pool_size = 5000000)
        : pool_size(initial_pool_size), max_pool_size(max_pool_size) {
        memory_blocks.reserve(initial_pool_size / BLOCK_SIZE + 1);
        free_list.reserve(initial_pool_size);
        in_use_labels.reserve(initial_pool_size);
        allocate_labels(pool_size);
    }

    // Acquire a Label from the pool.
    Label *acquire() {
        if (!free_list.empty()) {
            Label *label = free_list.back();
            free_list.pop_back();
            label->reset();
            in_use_labels.push_back(label);
            return label;
        }

        // If no free labels, allocate a new block if needed.
        if (memory_blocks.empty() || memory_blocks.back().used >= BLOCK_SIZE) {
            memory_blocks.emplace_back();
            auto &block = memory_blocks.back();
            Label *label = &block.data[0];
            block.used = 1;
            in_use_labels.push_back(label);
            return label;
        }

        // Otherwise, use an available label from the current block.
        auto &block = memory_blocks.back();
        Label *label = &block.data[block.used];
        block.used++;
        in_use_labels.push_back(label);
        return label;
    }

    // Reset the pool by moving all in-use labels back to the free list.
    void reset() {
        free_list.insert(free_list.end(), in_use_labels.begin(),
                         in_use_labels.end());
        in_use_labels.clear();
    }

    // Cleanup all allocated memory.
    void cleanup() {
        free_list.clear();
        in_use_labels.clear();
        memory_blocks.clear();
    }

    ~LabelPool() { cleanup(); }
};
