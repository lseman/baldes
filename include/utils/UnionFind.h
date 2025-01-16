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

    // Constructor with pre-computed size
    explicit UnionFind(size_t size) {
        parent.resize(size);
        rank.resize(size, 0);
        subsetIndex.resize(size, -1);

        // Initialize parent array - each element is its own parent initially
        for (size_t i = 0; i < size; ++i) {
            parent[i] = i;
        }
    }

    // Constructor for SCCs with optimized initialization
    explicit UnionFind(const std::vector<std::vector<int>>& sccs) {
        // Find max element with single pass
        int max_elem = -1;
        size_t total_elements = 0;
        for (const auto& scc : sccs) {
            total_elements += scc.size();
            for (int elem : scc) {
                max_elem = std::max(max_elem, elem);
            }
        }

        // Early return for empty input
        if (max_elem < 0) return;

        // Pre-allocate with exact sizes
        const size_t size = max_elem + 1;
        parent.resize(size);
        rank.resize(size, 0);
        subsetIndex.resize(size, -1);

        // Initialize parent array - each element is its own parent initially
        for (size_t i = 0; i < size; ++i) {
            parent[i] = i;
        }

        // Reserve space for path compression
        if (total_elements > 0) {
            path_compression_stack.reserve(
                static_cast<size_t>(std::log2(total_elements)) + 1);
        }

        // Process SCCs
        int subset_counter = 0;
        for (const auto& scc : sccs) {
            if (scc.empty()) continue;

            // Set subset index for first element
            const int first_elem = scc[0];
            subsetIndex[first_elem] = subset_counter;

            // Unite all other elements with the first
            for (size_t i = 1; i < scc.size(); ++i) {
                unite(first_elem, scc[i]);
            }

            subset_counter++;
        }
    }

    // Find with path compression using stack instead of recursion
    inline int find(int x) const noexcept {
        // Early return if x is its own parent
        if (parent[x] == x) return x;

        // Find root
        int root = x;
        while (parent[root] != root) {
            root = parent[root];
        }

        return root;
    }

    // Non-const find for operations that need to modify the structure
    inline int find_and_compress(int x) noexcept {
        // Early return if x is its own parent
        if (parent[x] == x) return x;

        // Find root
        int root = x;
        path_compression_stack.clear();

        while (parent[root] != root) {
            path_compression_stack.push_back(root);
            root = parent[root];
        }

        // Path compression
        for (int node : path_compression_stack) {
            parent[node] = root;
        }

        return root;
    }

    // Unite with rank and subset index update
    void unite(int x, int y) noexcept {
        int rootX = find_and_compress(x);
        int rootY = find_and_compress(y);

        if (rootX == rootY) return;

        // Union by rank
        if (rank[rootX] < rank[rootY]) {
            std::swap(rootX, rootY);
        }

        // Attach smaller rank tree under root of high rank tree
        parent[rootY] = rootX;
        subsetIndex[rootY] = subsetIndex[rootX];

        // If ranks are same, increment rank of rootX
        if (rank[rootX] == rank[rootY]) {
            ++rank[rootX];
        }
    }

    // Get subset index
    inline int getSubset(int x) const noexcept { return subsetIndex[find(x)]; }

   private:
    mutable std::vector<int> parent;
    std::vector<int> rank;
    std::vector<int> subsetIndex;
    std::vector<int>
        path_compression_stack;  // Reusable stack for path compression
};
