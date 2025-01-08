#pragma once
#include <vector>
#include <cstdint>

class UnionFind {
public:
    UnionFind() = default;

    UnionFind(const std::vector<std::vector<int>>& sccs) {
        // Pre-calculate size needed
        int max_elem = 0;
        size_t total_elements = 0;
        for (const auto& scc : sccs) {
            total_elements += scc.size();
            for (int elem : scc) {
                max_elem = (elem > max_elem) ? elem : max_elem;
            }
        }

        // Pre-allocate all vectors at once
        const size_t size = max_elem + 1;
        parent.resize(size);
        rank.resize(size);
        subsetIndex.resize(size, -1);

        // Initialize in batches for better cache usage
        int subset_counter = 0;
        for (const auto& scc : sccs) {
            if (!scc.empty()) {
                const int first_elem = scc[0];
                const int current_subset = subset_counter++;
                
                // Initialize first element
                parent[first_elem] = first_elem;
                subsetIndex[first_elem] = current_subset;

                // Batch process remaining elements
                const size_t scc_size = scc.size();
                #pragma GCC ivdep
                for (size_t i = 1; i < scc_size; ++i) {
                    const int elem = scc[i];
                    parent[elem] = first_elem;  // Point directly to root
                    subsetIndex[elem] = current_subset;
                }
            }
        }
    }

    // Fast path for root finding
    __attribute__((always_inline)) 
    inline int find(int x) const noexcept {
        int root = x;
        
        // Quick check if already root
        if (__builtin_expect(parent[x] == x, 1)) {
            return x;
        }

        // Find root (without path halving since const)
        while (root != parent[root]) {
            root = parent[root];
        }
        
        return root;
    }

    // Non-const version for path compression
    __attribute__((always_inline)) 
    inline int find_and_compress(int x) noexcept {
        int root = x;
        
        // Quick check if already root
        if (__builtin_expect(parent[x] == x, 1)) {
            return x;
        }

        // Find root with path halving
        while (root != parent[root]) {
            root = parent[root] = parent[parent[root]];  // Path halving
        }
        
        return root;
    }

    // Optimized union operation
    __attribute__((always_inline))
    inline void unite(int x, int y) noexcept {
        int rootX = find_and_compress(x);
        int rootY = find_and_compress(y);

        if (rootX != rootY) {
            // Union by rank with direct subset update
            if (rank[rootX] < rank[rootY]) {
                parent[rootX] = rootY;
                subsetIndex[rootX] = subsetIndex[rootY];
            } else {
                parent[rootY] = rootX;
                subsetIndex[rootY] = subsetIndex[rootX];
                if (rank[rootX] == rank[rootY]) {
                    ++rank[rootX];
                }
            }
        }
    }

    // Fast subset lookup
    __attribute__((always_inline))
    inline int getSubset(int x) const noexcept {
        return subsetIndex[find(x)];
    }

private:
    alignas(64) std::vector<int> parent;     // Aligned for cache line
    alignas(64) std::vector<int> rank;       // Aligned for cache line
    alignas(64) std::vector<int> subsetIndex;// Aligned for cache line
};