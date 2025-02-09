#pragma once
#include "Common.h"

class UnionFind {
   private:
    // Pack data together for better cache locality
    struct alignas(32) NodeData {
        int parent;
        int subset_index;
        int16_t rank;  // Using int16_t since rank rarely grows large
        int16_t pad;   // Maintain alignment
    };

    alignas(32) std::vector<NodeData> nodes;

    // Optimization: Cache the last accessed root for repeated operations
    mutable struct {
        int element{-1};
        int root{-1};
    } cache;

   public:
    UnionFind() = default;

    explicit UnionFind(size_t size) {
        nodes.resize(size);

        // Direct initialization for better performance
        for (size_t i = 0; i < size; ++i) {
            auto& node = nodes[i];
            node.parent = i;
            node.subset_index = -1;
            node.rank = 0;
        }
    }

    explicit UnionFind(const std::vector<std::vector<int>>& sccs) {
        // Find max element with single pass over contiguous memory
        int max_elem = -1;
        for (const auto& scc : sccs) {
            if (!scc.empty()) {
                const int* const data = scc.data();
                const size_t size = scc.size();
                for (size_t i = 0; i < size; ++i) {
                    max_elem = std::max(max_elem, data[i]);
                }
            }
        }

        if (max_elem < 0) return;

        const size_t size = max_elem + 1;
        nodes.resize(size);

        // Initialize with direct memory access
        for (size_t i = 0; i < size; ++i) {
            auto& node = nodes[i];
            node.parent = i;
            node.subset_index = -1;
            node.rank = 0;
        }

        // Process SCCs with optimized memory access
        int subset_counter = 0;
        for (const auto& scc : sccs) {
            if (scc.empty()) continue;

            const int first_elem = scc[0];
            nodes[first_elem].subset_index = subset_counter;

            // Unite remaining elements with first_elem
            const int* const data = scc.data();
            const size_t scc_size = scc.size();
            for (size_t i = 1; i < scc_size; ++i) {
                unite_with_path_compression(first_elem, data[i]);
            }
            subset_counter++;
        }
    }

    // Fast find without path compression for read-only operations
    [[nodiscard]] inline int find(int x) const noexcept {
        // Check cache first
        if (x == cache.element) {
            return cache.root;
        }

        int root = x;
        while (nodes[root].parent != root) {
            root = nodes[root].parent;
        }

        // Update cache
        cache.element = x;
        cache.root = root;

        return root;
    }

    // Fast getSubset that combines find and subset lookup
    [[nodiscard]] inline int getSubset(int x) const noexcept {
        // Use the cached root if available
        if (x == cache.element) {
            return nodes[cache.root].subset_index;
        }

        int root = x;
        while (nodes[root].parent != root) {
            root = nodes[root].parent;
        }

        // Update cache
        cache.element = x;
        cache.root = root;

        return nodes[root].subset_index;
    }

    // Path compression version for unite operations
    inline int find_and_compress(int x) noexcept {
        int root = x;

        // First pass: find root
        while (nodes[root].parent != root) {
            root = nodes[root].parent;
        }

        // Second pass: path compression with direct memory access
        while (x != root) {
            auto& node = nodes[x];
            int next = node.parent;
            node.parent = root;
            x = next;
        }

        // Update cache
        cache.element = x;
        cache.root = root;

        return root;
    }

    // Optimized unite with immediate path compression
    inline void unite_with_path_compression(int x, int y) noexcept {
        int root_x = find_and_compress(x);
        int root_y = find_and_compress(y);

        if (root_x == root_y) return;

        auto& node_x = nodes[root_x];
        auto& node_y = nodes[root_y];

        // Union by rank with direct struct access
        if (node_x.rank < node_y.rank) {
            std::swap(root_x, root_y);
            std::swap(node_x, node_y);
        }

        // Attach smaller rank tree under root of high rank tree
        node_y.parent = root_x;
        node_y.subset_index = node_x.subset_index;

        // If ranks are same, increment rank of root_x
        if (node_x.rank == node_y.rank) {
            node_x.rank++;
        }

        // Invalidate cache
        cache.element = -1;
    }

    [[nodiscard]] inline bool compareSubsets(int x, int y) const noexcept {
        // Use cached values if possible
        const int root_x = find(x);
        const int root_y = find(y);
        return nodes[root_x].subset_index < nodes[root_y].subset_index;
    }
};
