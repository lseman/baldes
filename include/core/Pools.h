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

#include "math/Common.h"
#include "model/Label.h"
#include "model/Path.h"
#include "core/TaskQueue.h"
#include "model/VRPNode.h"
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
    static constexpr size_t labels_per_block = 64;
    static constexpr size_t block_alignment  = 64;

    struct BucketArena {
        std::vector<Label *> blocks;
        size_t               next = 0;
    };

    size_t                   max_pool_size{};
    size_t                   labels_in_use{};
    size_t                   allocated_labels{};
    std::vector<BucketArena> bucket_arenas;
    std::vector<Label *>     recycled_labels;

    BucketArena &arena_for(int bucket_id) {
        const size_t index = bucket_id < 0 ? 0 : static_cast<size_t>(bucket_id) + 1;
        if (index >= bucket_arenas.size()) bucket_arenas.resize(index + 1);
        return bucket_arenas[index];
    }

    Label *allocate_block(BucketArena &arena) {
        if (allocated_labels + labels_per_block > max_pool_size) std::abort();
        auto *block = static_cast<Label *>(
            ::operator new[](labels_per_block * sizeof(Label), std::align_val_t{block_alignment}));
        for (size_t i = 0; i < labels_per_block; ++i) std::construct_at(block + i);
        arena.blocks.push_back(block);
        allocated_labels += labels_per_block;
        return block;
    }

public:
    explicit LabelPool(size_t initial_pool_size, size_t max_pool_size = 5000000)
        : max_pool_size(std::max(initial_pool_size, max_pool_size)), bucket_arenas(1) {}

    // Destination-aware allocation keeps labels of a bucket in contiguous
    // cache-line-aligned blocks. bucket_id < 0 uses the fallback arena.
    [[nodiscard]] inline Label *acquire(int bucket_id = -1) noexcept {
        if (!recycled_labels.empty()) {
            Label *label = recycled_labels.back();
            recycled_labels.pop_back();
            label->reset();
            ++labels_in_use;
            return label;
        }

        if (__builtin_expect(labels_in_use >= max_pool_size, 0)) std::abort();
        BucketArena &arena = arena_for(bucket_id);
        const size_t block_index = arena.next / labels_per_block;
        const size_t slot        = arena.next % labels_per_block;
        if (block_index == arena.blocks.size()) allocate_block(arena);
        Label *label = arena.blocks[block_index] + slot;
        ++arena.next;
        ++labels_in_use;
        label->reset();
        return label;
    }

    inline void acquire_batch(Label **out_labels, size_t count, int bucket_id = -1) noexcept {
        for (size_t i = 0; i < count; ++i) out_labels[i] = acquire(bucket_id);
    }

    inline void release(Label *label) noexcept {
        if (!label) return;
        recycled_labels.push_back(label);
        --labels_in_use;
    }

    inline void release_batch(Label **labels, size_t count) noexcept {
        recycled_labels.reserve(recycled_labels.size() + count);
        for (size_t i = 0; i < count; ++i) release(labels[i]);
    }

    inline void reset() noexcept { fast_reset(); }

    inline void fast_reset() noexcept {
        for (auto &arena : bucket_arenas) arena.next = 0;
        recycled_labels.clear();
        labels_in_use = 0;
    }

    [[nodiscard]] inline size_t get_free_count() const noexcept { return allocated_labels - labels_in_use; }

    [[nodiscard]] inline size_t get_total_capacity() const noexcept { return max_pool_size; }

    [[nodiscard]] inline size_t get_memory_usage() const noexcept { return allocated_labels * sizeof(Label); }

    // Cleanup all allocated memory
    void cleanup() noexcept {
        for (auto &arena : bucket_arenas) {
            for (Label *block : arena.blocks) {
                for (size_t i = 0; i < labels_per_block; ++i) std::destroy_at(block + i);
                ::operator delete[](block, std::align_val_t{block_alignment});
            }
        }
        bucket_arenas.clear();
        recycled_labels.clear();
        allocated_labels = 0;
        labels_in_use = 0;
    }

    // Shrink to fit - reduce memory usage by removing unused blocks
    void shrink_to_fit() {
        recycled_labels.shrink_to_fit();
        for (auto &arena : bucket_arenas) arena.blocks.shrink_to_fit();
    }

    ~LabelPool() { cleanup(); }
};
