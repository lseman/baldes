/**
 * @file UnionFind.h
 * @brief This file contains the definition of the UnionFind class.
 *
 * This file contains the definition of the UnionFind class, which implements
 * the Union-Find data structure for finding connected components in a graph.
 * The class provides methods for finding the root of a set, uniting two sets,
 * and getting the subset index of an element.
 *
 */

#pragma once
#include "Common.h"

class UnionFind {
   public:
    UnionFind() = default;

    explicit UnionFind(size_t size) {
        parent.resize(size);
        subset_index.resize(size, -1);
        rank.resize(size, 0);

        // Initialize arrays - use memset for better performance with large
        // sizes
        std::iota(parent.begin(), parent.end(), 0);
    }

    explicit UnionFind(const std::vector<std::vector<int>>& sccs) {
        // Find max element with single pass
        int max_elem = -1;
        for (const auto& scc : sccs) {
            for (int elem : scc) {
                max_elem = std::max(max_elem, elem);
            }
        }

        if (max_elem < 0) return;

        const size_t size = max_elem + 1;
        parent.resize(size);
        subset_index.resize(size, -1);
        rank.resize(size, 0);

        // Initialize parent array
        std::iota(parent.begin(), parent.end(), 0);

        // Process SCCs
        int subset_counter = 0;
        for (const auto& scc : sccs) {
            if (scc.empty()) continue;

            const int first_elem = scc[0];
            subset_index[first_elem] = subset_counter;

            // Unite remaining elements
            const size_t scc_size = scc.size();
            for (size_t i = 1; i < scc_size; ++i) {
                unite_with_path_compression(first_elem, scc[i]);
            }
            subset_counter++;
        }
    }

    // Fast find without path compression for read-only operations
    [[nodiscard]] inline int find(int x) const noexcept {
        while (parent[x] != x) {
            x = parent[x];
        }
        return x;
    }

    // Fast getSubset that combines find and subset lookup
    [[nodiscard]] inline int getSubset(int x) const noexcept {
        // Find root without path compression
        while (parent[x] != x) {
            x = parent[x];
        }
        return subset_index[x];
    }

    // Path compression version for unite operations
    inline int find_and_compress(int x) noexcept {
        int root = x;

        // First pass: find root
        while (parent[root] != root) {
            root = parent[root];
        }

        // Second pass: path compression
        while (x != root) {
            int next = parent[x];
            parent[x] = root;
            x = next;
        }

        return root;
    }

    // Optimized unite with immediate path compression
    inline void unite_with_path_compression(int x, int y) noexcept {
        int root_x = find_and_compress(x);
        int root_y = find_and_compress(y);

        if (root_x == root_y) return;

        // Union by rank
        if (rank[root_x] < rank[root_y]) {
            std::swap(root_x, root_y);
        }

        // Attach smaller rank tree under root of high rank tree
        parent[root_y] = root_x;
        subset_index[root_y] = subset_index[root_x];

        // If ranks are same, increment rank of root_x
        rank[root_x] += (rank[root_x] == rank[root_y]);
    }

    [[nodiscard]] inline bool compareSubsets(int x, int y) const noexcept {
        // Find roots without path compression, combined in one loop
        int root_x = x;
        int root_y = y;

        // Load both paths simultaneously for better cache usage
        while (true) {
            if (parent[root_x] != root_x) {
                root_x = parent[root_x];
            }
            if (parent[root_y] != root_y) {
                root_y = parent[root_y];
            }
            if (parent[root_x] == root_x && parent[root_y] == root_y) {
                break;
            }
        }
        return subset_index[root_x] < subset_index[root_y];
    }

   private:
    // Aligned memory for better cache performance
    alignas(64) std::vector<int> parent;
    alignas(64) std::vector<int> subset_index;  // Renamed for clarity
    alignas(64) std::vector<int> rank;
};
