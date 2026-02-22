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
    static constexpr size_t BLOCK_SIZE      = 256; // Increased for better amortization
    static constexpr size_t CACHE_LINE_SIZE = 64;

    // Align MemoryBlock to cache line for better performance
    struct alignas(CACHE_LINE_SIZE) MemoryBlock {
        std::unique_ptr<Label[]> data;
        size_t                   used; // Number of labels allocated from this block

        MemoryBlock() : data(std::make_unique<Label[]>(BLOCK_SIZE)), used(0) {}

        // Disable copy, enable move
        MemoryBlock(const MemoryBlock &)                = delete;
        MemoryBlock &operator=(const MemoryBlock &)     = delete;
        MemoryBlock(MemoryBlock &&) noexcept            = default;
        MemoryBlock &operator=(MemoryBlock &&) noexcept = default;
    };

    size_t pool_size;
    size_t max_pool_size;

    std::vector<MemoryBlock> memory_blocks;

    // free_list holds pointers to labels that are available for reuse
    // Use raw array with index-based management for better cache locality
    std::vector<Label *> free_list;
    size_t               free_list_size; // Track size separately to avoid .size() calls

    // Track in-use labels only when needed (removed for performance)
    // Most use cases don't need this tracking

    // Current block index for fast allocation
    size_t current_block_idx;

    // Allocate 'count' new labels and add them to the free list
    void allocate_labels(size_t count) {
        const size_t blocks_needed = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;

        // Reserve capacity once to avoid multiple reallocations
        memory_blocks.reserve(memory_blocks.size() + blocks_needed);
        free_list.reserve(free_list.size() + count);

        for (size_t b = 0; b < blocks_needed; ++b) {
            memory_blocks.emplace_back();
            auto &block = memory_blocks.back();

            // Calculate labels to allocate in this block
            const size_t remaining_count = count - b * BLOCK_SIZE;
            const size_t alloc_in_block  = std::min(remaining_count, BLOCK_SIZE);

            // Batch add pointers to free list
            Label *base_ptr = block.data.get();
            for (size_t i = 0; i < alloc_in_block; ++i) { free_list.push_back(base_ptr + i); }

            block.used = alloc_in_block;
            free_list_size += alloc_in_block;
        }

        current_block_idx = memory_blocks.size() - 1;
    }

public:
    explicit LabelPool(size_t initial_pool_size, size_t max_pool_size = 5000000)
        : pool_size(initial_pool_size), max_pool_size(max_pool_size), free_list_size(0), current_block_idx(0) {

        // Pre-allocate with some headroom to reduce reallocations
        const size_t reserve_size = initial_pool_size + (initial_pool_size >> 2); // +25%
        memory_blocks.reserve((reserve_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
        free_list.reserve(reserve_size);

        allocate_labels(pool_size);
    }

    // Fast acquire with minimal branching
    [[nodiscard]] inline Label *acquire() noexcept {
        Label *label;

        // Fast path: reuse from free list
        if (__builtin_expect(free_list_size > 0, 1)) {
            --free_list_size;
            label = free_list[free_list_size];
            label->reset();
            return label;
        }

        // Slow path: allocate from current block or create new block
        auto &current_block = memory_blocks[current_block_idx];

        if (__builtin_expect(current_block.used < BLOCK_SIZE, 1)) {
            // Use next available label from current block
            label = &current_block.data[current_block.used];
            ++current_block.used;
            label->reset();
            return label;
        }

        // Need a new block
        memory_blocks.emplace_back();
        current_block_idx = memory_blocks.size() - 1;
        auto &new_block   = memory_blocks[current_block_idx];
        label             = &new_block.data[0];
        new_block.used    = 1;
        label->reset();
        return label;
    }

    // Batch acquire for better performance when multiple labels needed
    inline void acquire_batch(Label **out_labels, size_t count) noexcept {
        size_t acquired = 0;

        // First, acquire from free list
        if (free_list_size > 0) {
            const size_t from_free = std::min(count, free_list_size);

            // Copy pointers in batch
            for (size_t i = 0; i < from_free; ++i) {
                --free_list_size;
                Label *label = free_list[free_list_size];
                label->reset();
                out_labels[i] = label;
            }

            acquired = from_free;
        }

        // Then allocate remaining from blocks
        while (acquired < count) {
            auto &current_block = memory_blocks[current_block_idx];

            if (current_block.used < BLOCK_SIZE) {
                const size_t available   = BLOCK_SIZE - current_block.used;
                const size_t to_allocate = std::min(count - acquired, available);

                // Batch allocate from current block
                Label *base_ptr = &current_block.data[current_block.used];
                for (size_t i = 0; i < to_allocate; ++i) {
                    (base_ptr + i)->reset();
                    out_labels[acquired + i] = base_ptr + i;
                }

                current_block.used += to_allocate;
                acquired += to_allocate;
            } else {
                // Need new block
                memory_blocks.emplace_back();
                current_block_idx = memory_blocks.size() - 1;
            }
        }
    }

    // Fast release without tracking in-use labels
    inline void release(Label *label) noexcept {
        if (__builtin_expect(free_list_size >= free_list.size(), 0)) {
            free_list.push_back(label);
            ++free_list_size;
            return;
        }
        free_list[free_list_size] = label;
        ++free_list_size;
    }

    // Batch release for better performance
    inline void release_batch(Label **labels, size_t count) noexcept {
        // Ensure capacity
        if (__builtin_expect(free_list.size() < free_list_size + count, 0)) {
            free_list.resize(free_list_size + count);
        }

        // Batch copy pointers
        std::memcpy(&free_list[free_list_size], labels, count * sizeof(Label *));
        free_list_size += count;
    }

    // Reset the pool - optimized version
    inline void reset() noexcept {
        size_t total_used = 0;
        for (const auto &block : memory_blocks) { total_used += block.used; }

        if (free_list.size() < total_used) { free_list.resize(total_used); }

        free_list_size = 0;
        for (auto &block : memory_blocks) {
            Label *base_ptr = block.data.get();
            for (size_t i = 0; i < block.used; ++i) { free_list[free_list_size++] = base_ptr + i; }
        }

        current_block_idx = memory_blocks.empty() ? 0 : memory_blocks.size() - 1;
    }

    // Fast reset without rebuilding free list (for scenarios where we'll allocate fresh)
    inline void fast_reset() noexcept {
        free_list_size    = 0;
        current_block_idx = 0;

        // Reset block usage counters
        for (auto &block : memory_blocks) { block.used = 0; }
    }

    // Get statistics
    [[nodiscard]] inline size_t get_free_count() const noexcept { return free_list_size; }

    [[nodiscard]] inline size_t get_total_capacity() const noexcept {
        size_t capacity = 0;
        for (const auto &block : memory_blocks) { capacity += block.used; }
        return capacity;
    }

    [[nodiscard]] inline size_t get_memory_usage() const noexcept {
        return memory_blocks.size() * BLOCK_SIZE * sizeof(Label);
    }

    // Cleanup all allocated memory
    void cleanup() noexcept {
        free_list.clear();
        free_list_size = 0;
        memory_blocks.clear();
        current_block_idx = 0;
    }

    // Shrink to fit - reduce memory usage by removing unused blocks
    void shrink_to_fit() {
        // Remove empty trailing blocks
        while (!memory_blocks.empty() && memory_blocks.back().used == 0) { memory_blocks.pop_back(); }

        // Shrink free list
        free_list.resize(free_list_size);
        free_list.shrink_to_fit();

        memory_blocks.shrink_to_fit();

        if (!memory_blocks.empty()) { current_block_idx = memory_blocks.size() - 1; }
    }

    ~LabelPool() { cleanup(); }
};
