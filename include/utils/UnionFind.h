#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "Common.h"

class UnionFind {
   private:
    // NodeData: Packed structure for better cache locality.
    // Stores the parent pointer, the subset index, and a rank (using int16_t).
    struct alignas(32) NodeData {
        int parent;        // Parent pointer for union-find.
        int subset_index;  // Additional subset index (e.g., from SCCs).
        int16_t rank;      // Rank used for union by rank.
        int16_t pad;       // Padding to maintain 32-byte alignment.
    };

    // The vector of nodes.
    alignas(32) std::vector<NodeData> nodes;

    // A simple mutable cache for the last queried element and its root.
    mutable struct {
        int element{-1};  // Last queried element.
        int root{-1};     // Its corresponding root.
    } cache;

   public:
    UnionFind() = default;

    // Constructs a UnionFind with a given size.
    explicit UnionFind(size_t size) {
        nodes.resize(size);
        for (size_t i = 0; i < size; ++i) {
            auto &node = nodes[i];
            node.parent = static_cast<int>(i);
            node.subset_index = -1;
            node.rank = 0;
            // 'pad' remains uninitialized, which is acceptable.
        }
    }

    // Constructs a UnionFind structure from strongly connected components
    // (SCCs). Each SCC is processed and all elements within the SCC are united.
    explicit UnionFind(const std::vector<std::vector<int>> &sccs) {
        int max_elem = -1;
        // Determine the maximum element value in one pass.
        for (const auto &scc : sccs) {
            if (!scc.empty()) {
                for (int elem : scc) {
                    max_elem = std::max(max_elem, elem);
                }
            }
        }
        if (max_elem < 0) return;

        const size_t size = static_cast<size_t>(max_elem) + 1;
        nodes.resize(size);
        for (size_t i = 0; i < size; ++i) {
            auto &node = nodes[i];
            node.parent = static_cast<int>(i);
            node.subset_index = -1;
            node.rank = 0;
        }

        // Process each SCC, uniting all elements within the SCC.
        int subset_counter = 0;
        for (const auto &scc : sccs) {
            if (scc.empty()) continue;
            const int first_elem = scc[0];
            nodes[first_elem].subset_index = subset_counter;
            for (size_t i = 1; i < scc.size(); ++i) {
                unite_with_path_compression(first_elem, scc[i]);
            }
            subset_counter++;
        }
    }

    // Standard find with a simple cache. (No path compression.)
    [[nodiscard]] inline int find(int x) const noexcept {
        if (x == cache.element) {
            return cache.root;
        }
        int root = x;
        while (nodes[root].parent != root) {
            root = nodes[root].parent;
        }
        // Update cache.
        cache.element = x;
        cache.root = root;
        return root;
    }

    // Get the subset index for element x, combining find and lookup.
    [[nodiscard]] inline int getSubset(int x) const noexcept {
        if (x == cache.element) {
            return nodes[cache.root].subset_index;
        }
        int root = x;
        while (nodes[root].parent != root) {
            root = nodes[root].parent;
        }
        // Update cache.
        cache.element = x;
        cache.root = root;
        return nodes[root].subset_index;
    }

    // Find with full path compression.
    inline int find_and_compress(int x) noexcept {
        int root = x;
        // First pass: find the root.
        while (nodes[root].parent != root) {
            root = nodes[root].parent;
        }
        // Second pass: path compression.
        int current = x;
        while (current != root) {
            int next = nodes[current].parent;
            nodes[current].parent = root;
            current = next;
        }
        // Update cache.
        cache.element = x;
        cache.root = root;
        return root;
    }

    // Union two elements using union by rank and immediate path compression.
    inline void unite_with_path_compression(int x, int y) noexcept {
        int root_x = find_and_compress(x);
        int root_y = find_and_compress(y);
        if (root_x == root_y) return;

        auto &node_x = nodes[root_x];
        auto &node_y = nodes[root_y];

        // Make the tree with higher rank the new root.
        if (node_x.rank < node_y.rank) {
            std::swap(root_x, root_y);
        }
        nodes[root_y].parent = root_x;
        // Propagate subset index from the new root.
        nodes[root_y].subset_index = node_x.subset_index;
        if (node_x.rank == node_y.rank) {
            node_x.rank++;
        }
        // Invalidate the cache.
        cache.element = -1;
    }

    // Compare the subset indices of x and y.
    // Returns true if x's subset index is less than y's.
    [[nodiscard]] inline bool compareSubsets(int x, int y) const noexcept {
        int root_x = find(x);
        int root_y = find(y);
        return nodes[root_x].subset_index < nodes[root_y].subset_index;
    }
};
