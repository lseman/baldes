#pragma once

#include "Common.h"
#include "Label.h"
#include "Path.h"
#include "VRPNode.h"


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

    void printDuals() {
        fmt::print("Arc duals:\n");
        for (const auto &[arc, value] : arcDuals) {
            fmt::print("({}, {}): {}\n", arc.first, arc.second, value);
        }

        fmt::print("Three two duals:\n");
        for (const auto &[node, value] : three_two_Duals) {
            fmt::print("{}: {}\n", node, value);
        }

        fmt::print("Three three duals:\n");
        for (const auto &[node, value] : three_three_Duals) {
            fmt::print("{}: {}\n", node, value);
        }
    }
};
