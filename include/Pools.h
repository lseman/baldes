/**
 * @file Pools.h
 * @brief Defines classes to manage pools of paths and labels.
 *
 * This file defines the SchrodingerPool and LabelPool classes, which manage pools of paths and labels, respectively.
 * The SchrodingerPool class manages a collection of paths with a limited lifespan, computes reduced costs, and filters
 * paths based on their reduced costs. The LabelPool class manages a pool of Label objects, providing methods to acquire
 * and release labels from the pool, as well as resetting the pool to its initial state.
 *
 */

#pragma once

#include "Definitions.h"
#include "Label.h"
#include "Path.h"
#include "VRPNode.h"

#include <deque>
#include <tuple>
#include <utility>
#include <vector>

#include "ankerl/unordered_dense.h"

/**
 * @class SchrodingerPool
 * @brief Manages a pool of paths with a limited lifespan, computes reduced costs, and filters paths based on their
 * reduced costs.
 *
 * This class is designed to manage a collection of paths, each associated with an iteration when it was added.
 * Paths have a limited lifespan, defined by `max_live_time`, after which they are removed from the pool.
 * The class also provides functionality to compute reduced costs for paths and filter paths based on their reduced
 * costs.
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

    SchrodingerPool(int live_time) : max_live_time(live_time) {}

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

    void add_paths(const std::vector<Path> &new_paths) {
        remove_old_paths();
        for (const Path &path : new_paths) { add_path(path); }
        computeRC();
    }

    void computeRC() {
        for (auto &path : paths) {
            int iteration_added = std::get<0>(path); // Get the iteration when the path was added

            // Stop processing if the path is older than current_iteration + max_life
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

    std::vector<Path> get_paths_with_negative_red_cost() {
        std::vector<Path> result;

        // Remove the paths that are older than current_iteration + max_life or have a negative red_cost
        auto it = paths.begin();
        while (it != paths.end()) {
            int         iteration_added = std::get<0>(*it); // Get the iteration when the path was added
            const Path &p               = std::get<1>(*it);

            // If the path is older than max_live_time, or has a negative red_cost
            if (iteration_added + max_live_time < current_iteration || p.red_cost < 0) {
                if (p.red_cost < 0) {
                    result.push_back(p); // Add paths with negative red_cost to the result
                }
                it = paths.erase(it); // Remove from paths and move iterator to the next element
            } else {
                ++it; // Only move the iterator if not erasing
            }
        }

        // Sort the result based on red_cost
        std::sort(result.begin(), result.end(), [](const Path &a, const Path &b) { return a.red_cost < b.red_cost; });

        return result;
    }

    void iterate() { current_iteration++; }
};

/**
 * @class LabelPool
 * @brief A class that manages a pool of Label objects.
 *
 * The LabelPool class is responsible for managing a pool of Label objects. It provides methods to acquire and
 * release labels from the pool, as well as resetting the pool to its initial state.
 *
 * The pool size is determined during construction and can be optionally limited to a maximum size. Labels can
 * be acquired from the pool using the `acquire()` method, and released back to the pool using the `release()`
 * method. If the pool is full, a new label will be allocated.
 *
 */
#include <deque>

class LabelPool {
public:
    explicit LabelPool(size_t initial_pool_size, size_t max_pool_size = 5000000)
        : pool_size(initial_pool_size), max_pool_size(max_pool_size)
    {
        allocate_labels(pool_size);
    }

    Label* acquire() {
        if (!available_labels.empty()) {
            Label* new_label = available_labels.back();
            available_labels.pop_back();
            in_use_labels.push_back(new_label);
            new_label->reset();
            return new_label;
        }

        // Allocate a new label and add it to in_use_labels
        auto* new_label = new Label();
        in_use_labels.push_back(new_label);
        return new_label;
    }

    void reset() {
        available_labels.reserve(available_labels.size() + in_use_labels.size());

        // Move in-use labels back to the available pool
        available_labels.insert(available_labels.end(), std::make_move_iterator(in_use_labels.begin()),
                                std::make_move_iterator(in_use_labels.end()));
        in_use_labels.clear();
    }

    ~LabelPool() { cleanup(); }

    void cleanup() {
        // Properly delete each Label to free memory in available and in-use pools
        for (Label* label : available_labels) {
            delete label;
        }
        available_labels.clear();

        for (Label* label : in_use_labels) {
            delete label;
        }
        in_use_labels.clear();

        // Delete any additional states if needed
        for (Label* label : deleted_states) {
            delete label;
        }
        deleted_states.clear();
    }

private:
    void allocate_labels(size_t count) {
        available_labels.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            available_labels.push_back(new Label());
        }
    }

    size_t pool_size;
    size_t max_pool_size;

    std::vector<Label*> available_labels; // Labels ready to be acquired
    std::vector<Label*> in_use_labels;    // Labels currently in use
    std::vector<Label*> deleted_states;   // Labels available for recycling
};


/**
 * @struct PSTEPDuals
 * @brief A structure to manage dual values for arcs and nodes in a network.
 *
 * This structure provides methods to set, get, and clear dual values for arcs and nodes.
 * Dual values are stored in unordered maps for efficient access.
 *
 */
struct PSTEPDuals {
    using Arc = std::pair<int, int>; // Represents an arc as a pair (from, to)

    ankerl::unordered_dense::map<Arc, double> arcDuals;          // Stores dual values for arcs
    ankerl::unordered_dense::map<int, double> three_two_Duals;   // Stores dual values for nodes
    ankerl::unordered_dense::map<int, double> three_three_Duals; // Stores dual values for nodes

    // Set dual values for arcs
    void setArcDualValues(const std::vector<std::pair<Arc, double>> &values) {
        for (const auto &[arc, value] : values) { arcDuals[arc] = value; }
    }

    // Set dual values for nodes
    void setThreeTwoDualValues(const std::vector<std::pair<int, double>> &values) {
        for (const auto &[node, value] : values) { three_two_Duals[node] = value; }
    }

    void setThreeThreeDualValues(const std::vector<std::pair<int, double>> &values) {
        for (const auto &[node, value] : values) { three_three_Duals[node] = value; }
    }

    // Clear all dual values (arcs and nodes)
    void clearDualValues() {
        arcDuals.clear();
        three_two_Duals.clear();
        three_three_Duals.clear();
    }

    // Get dual value for arcs (from, to)
    double getArcDualValue(int from, int to) const {
        Arc  arc   = {from, to};
        auto arcIt = arcDuals.find(arc);
        if (arcIt != arcDuals.end()) { return arcIt->second; }
        return 0.0; // Default value if arc is not found
    }

    // Get dual value from three_two_Duals for nodes
    double getThreeTwoDualValue(int node) const {
        auto it = three_two_Duals.find(node);
        if (it != three_two_Duals.end()) { return it->second; }
        return 0.0; // Default value if node is not found
    }

    // Get dual value from three_three_Duals for nodes
    double getThreeThreeDualValue(int node) const {
        auto it = three_three_Duals.find(node);
        if (it != three_three_Duals.end()) { return it->second; }
        return 0.0; // Default value if node is not found
    }
};
