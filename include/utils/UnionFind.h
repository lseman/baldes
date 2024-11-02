/**
 * @file UnionFind.h
 * @brief This file contains the definition of the UnionFind class.
 *
 * This file contains the definition of the UnionFind class, which implements the Union-Find data structure
 * for finding connected components in a graph. The class provides methods for finding the root of a set,
 * uniting two sets, and getting the subset index of an element.
 *
 */
#pragma once

#include "Common.h"

class UnionFind {
public:
    // default constructor
    UnionFind() = default;
    // Constructor initializes based on the elements in the SCCs
    UnionFind(const std::vector<std::vector<int>> &sccs) {
        // Find the maximum element in SCCs to size the parent, rank, and subsetIndex vectors
        int max_elem = 0;
        for (const auto &scc : sccs) {
            for (int elem : scc) { max_elem = std::max(max_elem, elem); }
        }

        // Initialize parent, rank, and subsetIndex vectors based on the max element
        parent.resize(max_elem + 1);
        rank.resize(max_elem + 1, 0);
        subsetIndex.resize(max_elem + 1, -1); // Initialize with -1 (not assigned yet)
        int subset_counter = 0;

        for (const auto &scc : sccs) {
            if (!scc.empty()) {
                // Assign subset number to all elements in the current SCC
                for (size_t i = 0; i < scc.size(); ++i) {
                    int elem          = scc[i];
                    parent[elem]      = elem;           // Initially, each element is its own parent
                    subsetIndex[elem] = subset_counter; // Assign subset number
                }
                // Unite all elements within this SCC
                for (size_t i = 1; i < scc.size(); ++i) {
                    unite(scc[0], scc[i]); // Unite the first element with others
                }
                subset_counter++; // Move to the next subset
            }
        }
    }

    // Find the root of the set containing x with path compression
    inline int find(int x) {
        int root = x;
        // Find the root
        while (root != parent[root]) { root = parent[root]; }
        // Path compression
        while (x != root) {
            int next  = parent[x];
            parent[x] = root;
            x         = next;
        }
        return root;
    }

    // Union two sets by rank, and update the subset index
    void unite(int x, int y) {
        int rootX = find(x);
        int rootY = find(y);
        if (rootX != rootY) {
            // Union by rank
            if (rank[rootX] > rank[rootY]) {
                parent[rootY] = rootX;
                // Update subsetIndex for all elements in the rootY tree
                subsetIndex[rootY] = subsetIndex[rootX];
            } else if (rank[rootX] < rank[rootY]) {
                parent[rootX] = rootY;
                // Update subsetIndex for all elements in the rootX tree
                subsetIndex[rootX] = subsetIndex[rootY];
            } else {
                parent[rootY] = rootX;
                ++rank[rootX];
                // Update subsetIndex for rootY
                subsetIndex[rootY] = subsetIndex[rootX];
            }
        }
    }

    // Function to get the subset index of an element
    int getSubset(int x) {
        int root = find(x);       // Find the root of the set
        return subsetIndex[root]; // Return the subset index
    }

private:
    std::vector<int> parent;
    std::vector<int> rank;
    std::vector<int> subsetIndex; // Store the subset number of each element
};
