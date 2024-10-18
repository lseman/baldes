#include <algorithm>
#include <iostream>
#include <vector>

class UnionFind {
public:
    // default empty constructor
    UnionFind() {}
    // Constructor that initializes based on the elements in the SCCs
    UnionFind(const std::vector<std::vector<int>> &sccs) {
        // Find the maximum element in SCCs to size the parent and rank vectors
        int max_elem = 0;
        for (const auto &scc : sccs) {
            for (int elem : scc) {
                if (elem > max_elem) { max_elem = elem; }
            }
        }

        // Initialize parent and rank vectors based on the max element
        parent.resize(max_elem + 1);
        rank.resize(max_elem + 1, 0);
        for (int i = 0; i <= max_elem; ++i) {
            parent[i] = i; // Initially, each element is its own parent
        }

        // Unite elements within each SCC
        for (const auto &scc : sccs) {
            if (!scc.empty()) {
                for (size_t i = 1; i < scc.size(); ++i) {
                    unite(scc[0], scc[i]); // Unite the first element with others
                }
            }
        }
    }

    // Find the root of the set containing x with path compression
    size_t find(size_t x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]); // Path compression
        }
        return parent[x];
    }

    // Union two sets by rank
    void unite(size_t x, size_t y) {
        size_t rootX = find(x);
        size_t rootY = find(y);
        if (rootX != rootY) {
            // Union by rank
            if (rank[rootX] > rank[rootY]) {
                parent[rootY] = rootX;
            } else if (rank[rootX] < rank[rootY]) {
                parent[rootX] = rootY;
            } else {
                parent[rootY] = rootX;
                ++rank[rootX];
            }
        }
    }

private:
    std::vector<size_t> parent;
    std::vector<size_t> rank;
};
