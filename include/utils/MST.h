/**
 * @file
 * @brief This file contains the implementation of the Minimum Spanning Tree
 * (MST) algorithm for clustering.
 *
 * This file contains the implementation of the Minimum Spanning Tree (MST)
 * algorithm for clustering. The MST algorithm is used to find the minimum
 * spanning tree of a graph and then cluster the nodes based on the edges of the
 * MST. The file includes the implementation of the MST class, which computes
 * the MST using Prim's algorithm and performs clustering based on edge weights.
 *
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <queue>
#include <tuple>
#include <vector>

#include "Common.h"
#include "VRPNode.h"

class MST {
   public:
    using Arc = std::tuple<double, int, int>;  // weight, from_node, to_node

    MST(const std::vector<VRPNode> &nodes,
        std::function<double(int, int)> get_cost)
        : nodes_(nodes), getcij(get_cost) {}

    // Function to compute the MST using Prim's algorithm
    std::vector<Arc> compute_mst() {
        std::priority_queue<Arc, std::vector<Arc>, std::greater<Arc>> min_heap;
        std::vector<bool> in_mst(nodes_.size(), false);
        std::vector<Arc> mst;
        mst.reserve(nodes_.size() - 1);

        int start_node = 0;
        in_mst[start_node] = true;
        add_adjacent_arcs(start_node, min_heap);

        while (!min_heap.empty() && mst.size() < nodes_.size() - 1) {
            auto [weight, from, to] = min_heap.top();
            min_heap.pop();

            if (in_mst[to]) continue;

            mst.emplace_back(weight, from, to);
            in_mst[to] = true;
            add_adjacent_arcs(to, min_heap);
        }

        return mst;
    }

    // Function to perform clustering by removing edges with costs exceeding
    // threshold Θ
    std::vector<std::vector<int>> cluster(double theta) {
        auto mst = compute_mst();

        // Calculate average and standard deviation of edge weights in the MST
        double sum = 0.0;
        for (const auto &[weight, from, to] : mst) {
            sum += weight;
        }
        double avg = sum / mst.size();

        double sum_sq = 0.0;
        for (const auto &[weight, from, to] : mst) {
            sum_sq += std::pow(weight - avg, 2);
        }
        double std_dev = std::sqrt(sum_sq / mst.size());

        // Threshold Θ = avg(T) + θ * std(T)
        double threshold = avg + theta * std_dev;

        // Remove edges with weights greater than the threshold to form clusters
        std::vector<Arc> filtered_edges;
        filtered_edges.reserve(mst.size());
        for (const auto &[weight, from, to] : mst) {
            if (weight <= threshold) {
                filtered_edges.emplace_back(weight, from, to);
            }
        }

        // Find the connected components (clusters) in the filtered MST
        return find_clusters(filtered_edges);
    }

   private:
    const std::vector<VRPNode> &nodes_;
    std::function<double(int, int)>
        getcij;  // Store the reference to the getcij function

    // Helper function to add adjacent arcs of a node to the priority queue
    void add_adjacent_arcs(int node_id,
                           std::priority_queue<Arc, std::vector<Arc>,
                                               std::greater<Arc>> &min_heap) {
        const auto &node = nodes_[node_id];

        for (const auto &next_node : nodes_) {
            if (node.id == next_node.id) continue;

            auto travel_cost = getcij(node.id, next_node.id);
            min_heap.emplace(travel_cost, node.id, next_node.id);
        }
    }

    // Helper function to find connected components (clusters) from the filtered
    // edges
    std::vector<std::vector<int>> find_clusters(const std::vector<Arc> &edges) {
        UnionFind uf(
            nodes_
                .size());  // Use a union-find data structure to track clusters
        for (const auto &[weight, from, to] : edges) {
            uf.union_sets(from, to);
        }

        // Gather the clusters based on the union-find structure
        std::vector<std::vector<int>> clusters(nodes_.size());
        for (int i = 0; i < nodes_.size(); ++i) {
            clusters[uf.find(i)].push_back(i);
        }

        // Remove empty clusters
        clusters.erase(std::remove_if(clusters.begin(), clusters.end(),
                                      [](const std::vector<int> &cluster) {
                                          return cluster.empty();
                                      }),
                       clusters.end());

        return clusters;
    }

    // Union-Find class for finding connected components
    class UnionFind {
       public:
        UnionFind(int n) : parent(n), rank(n, 0) {
            for (int i = 0; i < n; ++i) parent[i] = i;
        }

        int find(int x) {
            if (parent[x] != x) parent[x] = find(parent[x]);
            return parent[x];
        }

        void union_sets(int x, int y) {
            int root_x = find(x);
            int root_y = find(y);
            if (root_x != root_y) {
                if (rank[root_x] > rank[root_y])
                    parent[root_y] = root_x;
                else if (rank[root_x] < rank[root_y])
                    parent[root_x] = root_y;
                else {
                    parent[root_y] = root_x;
                    rank[root_x]++;
                }
            }
        }

       private:
        std::vector<int> parent;
        std::vector<int> rank;
    };
};
