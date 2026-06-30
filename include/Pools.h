/**
 * @file Pools.h
 * @brief Defines classes to manage pools of paths and labels.
 *
 */

#pragma once

#include <algorithm>
#include <deque>
#include <exec/static_thread_pool.hpp>
#include <exec/task.hpp>
#include <memory>
#include <new>
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
    std::deque<std::tuple<int, Path>> paths; // Stores tuples of (iteration added, Path)
    int                               current_iteration = 0;
    int                               max_live_time; // Max iterations a Path can stay active
    std::vector<double>               duals;         // Dual variables for each path
    std::vector<VRPNode>             *nodes = nullptr;

public:
    std::vector<std::vector<double>> distance_matrix; // Distance matrix for the graph

    exec::static_thread_pool            pool  = exec::static_thread_pool(5);
    exec::static_thread_pool::scheduler sched = pool.get_scheduler();

    // SchrodingerPool(int live_time) : max_live_time(live_time) {}
    SchrodingerPool(int live_time) : max_live_time(live_time), pool(5), task_queue(5, pool.get_scheduler(), *this) {}

    ~SchrodingerPool() {}

    TaskQueue<Path, SchrodingerPool> task_queue;

    void setNodes(std::vector<VRPNode> *nodes) { this->nodes = nodes; }

    int getcij(int i, int j) { return distance_matrix[i][j]; }

    void remove_old_paths() {
        // Remove old paths that have lived beyond their allowed time
        while (!paths.empty() && std::get<0>(paths.front()) + max_live_time <= current_iteration) {
            paths.pop_front(); // Remove the oldest path
        }
    }

    void add_path(const Path &path) {
        // Add new path with the current iteration
        paths.push_back({current_iteration, path});
    }

    // Submit a task to add new paths
    void add_paths(const std::vector<Path> &new_paths) {
        for (const Path &path : new_paths) { task_queue.submit_task(path); }
    }

    void computeRC() {
        for (auto &path : paths) {
            int iteration_added = std::get<0>(path); // Get the iteration when the path was added

            // Stop processing if the path is older than current_iteration +
            // max_life
            if (iteration_added + max_live_time < current_iteration) { break; }

            Path &p    = std::get<1>(path);
            p.red_cost = p.cost;

            if (p.size() > 3) {
                for (int i = 1; i < p.size() - 1; i++) {
                    auto &node = (*nodes)[p[i]]; // Dereference nodes and access element
                    p.red_cost -= node.cost;
                }
            }
        }
    }

    std::vector<Path> perturbation(const std::vector<Path> &paths_to_process) {
        remove_old_paths();
        for (const Path &path : paths_to_process) { add_path(path); }
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
            int         iteration_added = std::get<0>(*it); // Get the iteration when the path was added
            const Path &p               = std::get<1>(*it);

            // If the path is older than max_live_time, or has a negative
            // red_cost
            if (iteration_added + max_live_time < current_iteration || p.red_cost < 0) {
                if (p.red_cost < 0) {
                    result.push_back(p); // Add paths with negative red_cost to the result
                }
                it = paths.erase(it); // Remove from paths and move iterator to
                                      // the next element
            } else {
                ++it; // Only move the iterator if not erasing
            }
        }

        // Sort the result based on red_cost
        pdqsort(result.begin(), result.end(), [](const Path &a, const Path &b) { return a.red_cost < b.red_cost; });

        return result;
    }

    std::vector<Path> get_paths() {
        auto              tasks = task_queue.get_processed_tasks();
        std::vector<Path> result;
        result.insert(result.end(), tasks.begin(), tasks.end());
        return result;
    }

    void iterate() { current_iteration++; }
};
/**
 * @class LabelPool
 * @brief A highly optimized pool manager for Label objects.
 *
 * The LabelPool class manages a pool of Label objects with optimized allocation
 * and reuse strategies. It uses block-based allocation and fast pointer management
 * for minimal overhead.
 */
class LabelPool {
private:
    using Index                                  = uint32_t;
    static constexpr size_t construct_batch_size = 4096;

    size_t max_pool_size{};
    Label *arena{nullptr};
    size_t constructed_count{};
    size_t next_unused{};

    std::vector<Index> free_indices;
    size_t             free_list_size{};

    [[nodiscard]] inline Index pointer_to_index(const Label *label) const noexcept {
        return static_cast<Index>(label - arena);
    }

    [[nodiscard]] inline Label *label_at(Index index) noexcept { return arena + index; }

    [[nodiscard]] inline const Label *label_at(Index index) const noexcept { return arena + index; }

    inline void construct_until(size_t count) {
        while (constructed_count < count) {
            std::construct_at(arena + constructed_count);
            ++constructed_count;
        }
    }

    inline void construct_for_next_unused(size_t required_count) {
        if (required_count <= constructed_count) return;
        const size_t target_count =
            std::min(max_pool_size, std::max(required_count, constructed_count + construct_batch_size));
        construct_until(target_count);
    }

public:
    explicit LabelPool(size_t initial_pool_size, size_t max_pool_size = 5000000)
        : max_pool_size(std::max(initial_pool_size, max_pool_size)) {

        arena = static_cast<Label *>(
            ::operator new[](this->max_pool_size * sizeof(Label), std::align_val_t{alignof(Label)}));
        free_indices.reserve(initial_pool_size);
        construct_until(initial_pool_size);
        next_unused = initial_pool_size;
        for (Index i = 0; i < static_cast<Index>(initial_pool_size); ++i) { free_indices.push_back(i); }
        free_list_size = initial_pool_size;
    }

    // Fast acquire with minimal branching
    [[nodiscard]] inline Label *acquire() noexcept {
        if (__builtin_expect(free_list_size > 0, 1)) {
            Label *label = label_at(free_indices[--free_list_size]);
            label->reset();
            return label;
        }

        if (__builtin_expect(next_unused >= max_pool_size, 0)) { std::abort(); }

        const size_t index = next_unused++;
        construct_for_next_unused(next_unused);
        Label *label = arena + index;
        label->reset();
        return label;
    }

    // Batch acquire for better performance when multiple labels needed
    inline void acquire_batch(Label **out_labels, size_t count) noexcept {
        size_t acquired = 0;

        // First, acquire from free list
        if (free_list_size > 0) {
            const size_t from_free = std::min(count, free_list_size);

            for (size_t i = 0; i < from_free; ++i) {
                Label *label = label_at(free_indices[--free_list_size]);
                label->reset();
                out_labels[i] = label;
            }

            acquired = from_free;
        }

        // Then allocate remaining from blocks
        while (acquired < count) {
            if (__builtin_expect(next_unused >= max_pool_size, 0)) { std::abort(); }
            const size_t available   = max_pool_size - next_unused;
            const size_t to_allocate = std::min(count - acquired, available);
            const size_t base_index  = next_unused;
            next_unused += to_allocate;
            construct_for_next_unused(next_unused);

            Label *base_ptr = arena + base_index;
            for (size_t i = 0; i < to_allocate; ++i) {
                (base_ptr + i)->reset();
                out_labels[acquired + i] = base_ptr + i;
            }
            acquired += to_allocate;
        }
    }

    // Fast release without tracking in-use labels
    inline void release(Label *label) noexcept {
        if (!label) return;
        if (__builtin_expect(free_list_size >= free_indices.size(), 0)) { free_indices.resize(free_list_size + 1); }
        free_indices[free_list_size++] = pointer_to_index(label);
    }

    // Batch release for better performance
    inline void release_batch(Label **labels, size_t count) noexcept {
        // Ensure capacity
        if (__builtin_expect(free_indices.size() < free_list_size + count, 0)) {
            free_indices.resize(free_list_size + count);
        }

        for (size_t i = 0; i < count; ++i) { free_indices[free_list_size + i] = pointer_to_index(labels[i]); }
        free_list_size += count;
    }

    // Reset the pool - optimized version
    inline void reset() noexcept {
        if (free_indices.size() < next_unused) { free_indices.resize(next_unused); }
        for (Index i = 0; i < static_cast<Index>(next_unused); ++i) { free_indices[i] = i; }
        free_list_size = next_unused;
    }

    // Fast reset without rebuilding free list (for scenarios where we'll allocate fresh)
    inline void fast_reset() noexcept {
        free_list_size = 0;
        next_unused    = 0;
    }

    // Get statistics
    [[nodiscard]] inline size_t get_free_count() const noexcept { return free_list_size; }

    [[nodiscard]] inline size_t get_total_capacity() const noexcept { return max_pool_size; }

    [[nodiscard]] inline size_t get_memory_usage() const noexcept { return max_pool_size * sizeof(Label); }

    // Cleanup all allocated memory
    void cleanup() noexcept {
        if (arena) {
            for (size_t i = 0; i < constructed_count; ++i) { std::destroy_at(arena + i); }
            ::operator delete[](arena, std::align_val_t{alignof(Label)});
        }
        arena             = nullptr;
        constructed_count = 0;
        next_unused       = 0;
        free_indices.clear();
        free_list_size = 0;
    }

    // Shrink to fit - reduce memory usage by removing unused blocks
    void shrink_to_fit() {
        free_indices.resize(free_list_size);
        free_indices.shrink_to_fit();
    }

    ~LabelPool() { cleanup(); }
};
