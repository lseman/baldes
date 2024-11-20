#pragma once
#include <algorithm>
#include <iostream>
#include <limits>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class MUSSP {
private:
    struct Edge;

public:
    struct Node {
        int                 id;
        double              distance;
        std::vector<Edge *> out_edges;
        Node               *branch_node; // For branch optimization

        Node(int id_) : id(id_), distance(std::numeric_limits<double>::infinity()), branch_node(nullptr) {}
    };

private:
    struct Edge {
        Node  *from;
        Node  *to;
        double cost;
        int    capacity;
        int    flow;
        Edge  *reverse;

        Edge(Node *from_, Node *to_, double cost_, int capacity_)
            : from(from_), to(to_), cost(cost_), capacity(capacity_), flow(0), reverse(nullptr) {}

        int residual_capacity() const { return capacity - flow; }
    };

    std::vector<Node *> nodes;
    std::vector<Edge *> edges;
    Node               *source;
    Node               *sink;

    std::vector<Edge *> findShortestPath() {
        // Reset distances
        for (Node *node : nodes) { node->distance = std::numeric_limits<double>::infinity(); }
        source->distance = 0;

        std::priority_queue<std::pair<double, Node *>, std::vector<std::pair<double, Node *>>, std::greater<>> pq;
        pq.push({0, source});

        std::unordered_map<Node *, Edge *> prev;

        while (!pq.empty()) {
            auto [dist, node] = pq.top();
            pq.pop();

            if (dist > node->distance) continue;

            for (Edge *edge : node->out_edges) {
                if (edge->residual_capacity() <= 0) continue;

                Node  *next     = edge->to;
                double new_dist = node->distance + edge->cost;

                if (new_dist < next->distance) {
                    next->distance = new_dist;
                    prev[next]     = edge;
                    pq.push({new_dist, next});
                }
            }
        }

        // Reconstruct path
        std::vector<Edge *> path;
        Node               *curr = sink;
        while (curr != source && prev.count(curr) > 0) {
            Edge *edge = prev[curr];
            path.push_back(edge);
            curr = edge->from;
        }
        std::reverse(path.begin(), path.end());
        return path;
    }

    void clipPermanentEdges() {
        std::vector<Edge *> to_remove;

        for (Edge *edge : edges) {
            if (edge->flow > 0) {
                if (edge->from == source || edge->to == sink) {
                    to_remove.push_back(edge);
                    if (edge->reverse) to_remove.push_back(edge->reverse);
                }
            }
        }

        if (!to_remove.empty()) { std::cout << "Clipping " << to_remove.size() / 2 << " permanent edges\n"; }

        for (Edge *edge : to_remove) {
            auto &out_edges = edge->from->out_edges;
            out_edges.erase(std::remove(out_edges.begin(), out_edges.end(), edge), out_edges.end());
        }

        edges.erase(std::remove_if(
                        edges.begin(), edges.end(),
                        [&](Edge *e) { return std::find(to_remove.begin(), to_remove.end(), e) != to_remove.end(); }),
                    edges.end());
    }

    void updateBranchNodes(const std::vector<Edge *> &path) {
        for (Node *node : nodes) { node->branch_node = nullptr; }

        if (!path.empty()) {
            Node *branch = path[0]->to;
            for (Edge *edge : path) { edge->to->branch_node = branch; }
        }
    }

    std::vector<Node *> identifyNodesToUpdate(const std::vector<Edge *> &path) {
        std::vector<Node *> to_update;
        if (path.empty()) return to_update;

        Node *branch = path[0]->to;
        for (Node *node : nodes) {
            if (node->branch_node == branch) { to_update.push_back(node); }
        }
        return to_update;
    }

public:
    MUSSP() {}

    void setSource(Node *source_) { source = source_; }
    void setSink(Node *sink_) { sink = sink_; }

    ~MUSSP() {
        for (Node *node : nodes) delete node;
        for (Edge *edge : edges) delete edge;
    }

    Node *addNode() {
        Node *node = new Node(nodes.size());
        nodes.push_back(node);
        return node;
    }

    void addEdge(Node *from, Node *to, double cost, int capacity) {
        Edge *forward = new Edge(from, to, cost, capacity);
        Edge *reverse = new Edge(to, from, -cost, 0);

        forward->reverse = reverse;
        reverse->reverse = forward;

        from->out_edges.push_back(forward);
        to->out_edges.push_back(reverse);

        edges.push_back(forward);
        edges.push_back(reverse);
    }

    Node *getSource() { return source; }
    Node *getSink() { return sink; }

    // Add helper to check if paths are independent
    bool arePathsIndependent(const std::vector<Edge *> &path1, const std::vector<Edge *> &path2) {
        std::unordered_set<Node *> nodes1;
        for (Edge *edge : path1) {
            nodes1.insert(edge->from);
            nodes1.insert(edge->to);
        }

        for (Edge *edge : path2) {
            if (nodes1.count(edge->from) > 0 || nodes1.count(edge->to) > 0) { return false; }
        }
        return true;
    }

    // Add function to find multiple independent paths
    std::vector<std::vector<Edge *>> findMultiplePaths() {
        std::vector<std::vector<Edge *>> paths;

        // Find first path
        auto path = findShortestPath();
        if (path.empty()) return paths;

        // Check if it's a valid negative cost path
        double path_cost = 0;
        for (Edge *edge : path) { path_cost += edge->cost; }
        if (path_cost >= 0) return paths;

        // Add first path
        paths.push_back(path);

        // Try to find more independent paths
        std::unordered_set<Node *> used_nodes;
        for (Edge *edge : path) {
            used_nodes.insert(edge->from);
            used_nodes.insert(edge->to);
        }

        // Keep finding paths until we can't find any more independent ones
        while (true) {
            bool found_independent = false;

            // Find another path
            path = findShortestPath();
            if (path.empty()) break;

            // Check path cost
            path_cost = 0;
            for (Edge *edge : path) { path_cost += edge->cost; }
            if (path_cost >= 0) break;

            // Check if this path is independent from all existing paths
            bool is_independent = true;
            for (const auto &existing_path : paths) {
                if (!arePathsIndependent(path, existing_path)) {
                    is_independent = false;
                    break;
                }
            }

            if (is_independent) {
                paths.push_back(path);
                // Mark nodes as used
                for (Edge *edge : path) {
                    used_nodes.insert(edge->from);
                    used_nodes.insert(edge->to);
                }
                found_independent = true;
            }

            if (!found_independent) break;
        }

        return paths;
    }

    void solve() {
        // Initial branch node computation
        std::vector<Edge *> initial_path = findShortestPath();
        updateBranchNodes(initial_path);

        int iteration = 1;
        while (true) {
            std::cout << "\nIteration " << iteration++ << ":\n";

            // Print current graph state
            std::cout << "Current graph state:" << std::endl;
            for (Edge *edge : edges) {
                if (edge->capacity > 0) {
                    std::cout << edge->from->id << " -> " << edge->to->id << " (cost=" << edge->cost
                              << ", flow=" << edge->flow << "/" << edge->capacity
                              << ", residual=" << edge->residual_capacity() << ")" << std::endl;
                }
            }

            // Find multiple independent paths
            auto paths = findMultiplePaths();
            if (paths.empty()) {
                std::cout << "No negative cost paths found. Done!" << std::endl;
                break;
            }

            std::cout << "Found " << paths.size() << " independent paths" << std::endl;

            // Process all paths
            for (const auto &path : paths) {
                // Calculate path properties
                double path_cost    = 0;
                int    min_residual = std::numeric_limits<int>::max();

                std::cout << "\nProcessing path: ";
                for (Edge *edge : path) {
                    std::cout << edge->from->id << " -> " << edge->to->id << " ";
                    path_cost += edge->cost;
                    min_residual = std::min(min_residual, edge->residual_capacity());
                }

                std::cout << "\nPath cost: " << path_cost;
                std::cout << "\nMin residual capacity: " << min_residual << std::endl;

                // Augment flow
                std::cout << "Augmenting flow by " << min_residual << std::endl;
                for (Edge *edge : path) {
                    edge->flow += min_residual;
                    edge->reverse->flow -= min_residual;
                }

                clipPermanentEdges();
            }

            // Update branch nodes using first path
            if (!paths.empty()) {
                auto nodes_to_update = identifyNodesToUpdate(paths[0]);
                std::cout << "Updating distances for " << nodes_to_update.size() << " nodes\n";
                updateBranchNodes(paths[0]);
            }
        }

        // Print final flow state
        std::cout << "\nFinal flow state:" << std::endl;
        for (Edge *edge : edges) {
            if (edge->capacity > 0) {
                std::cout << edge->from->id << " -> " << edge->to->id << " (cost=" << edge->cost
                          << ", flow=" << edge->flow << "/" << edge->capacity << ")" << std::endl;
            }
        }
    }
};