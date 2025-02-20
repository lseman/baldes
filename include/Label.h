/**
 * @file Label.h
 * @brief This file contains the definition of the Label struct and
 * LabelComparator class.
 *
 * The Label struct represents a label used in a solver, containing various
 * properties and methods related to the label. The LabelComparator class
 * provides a comparator for comparing two Label objects based on their cost.
 *
 */

#pragma once
#include "Common.h"

/**
 * @struct Label
 * @brief Represents a label used in a solver.
 *
 * This struct contains various properties and methods related to a label used
 * in a solver. It stores information such as the set of Fj, id, is_extended
 * flag, vertex, cost, real_cost, SRC_cost, resources, predecessor, is_dominated
 * flag, nodes_covered, nodes_ordered, node_id, cut_storage, parent, children,
 * status, visited, and SRCmap.
 *
 * The struct provides constructors to initialize the label with or without a
 * node_id. It also provides methods to set the covered nodes, add a node to the
 * covered nodes, check if a node is already covered, and initialize the label
 * with new values.
 *
 * The struct overloads the equality and greater than operators for comparison.
 */
struct Label {
    // int                   id;
    bool is_extended = false;
    int vertex;
    double cost = 0.0;
    double real_cost = 0.0;
    std::array<double, R_SIZE> resources = {};
    std::vector<uint16_t> nodes_covered = {};  // Add nodes_covered to Label
    int node_id = -1;                          // Add node_id to Label
    Label *parent = nullptr;
    bool fresh = true;
    bool is_dominated = false;
    SRC_MODE_BLOCK(std::vector<uint16_t> SRCmap;)

    // uint64_t             visited_bitmap; // Bitmap for visited nodes
    std::array<uint64_t, num_words> visited_bitmap = {0};
#ifdef UNREACHABLE_DOMINANCE
    std::array<uint64_t, num_words> unreachable_bitmap = {0};
#endif
    std::vector<Label *> children;

    // std::vector<Label *> children;
    // Constructor with node_id
    Label(int v, double c, const std::vector<double> &res, int pred,
          int node_id)
        : vertex(v), cost(c), resources({res[0]}), node_id(node_id) {}

    // Constructor without node_id
    Label(int v, double c, const std::vector<double> &res, int pred)
        : vertex(v), cost(c), resources({res[0]}), node_id(-1) {}

    // Default constructor
    Label() : vertex(-1), cost(0), resources({0.0}), node_id(-1) {}

    void set_extended(bool extended) { is_extended = extended; }
    void set_dominated(bool dominated) { is_dominated = dominated; }

    auto &getRoute() const { return nodes_covered; }

    void addRoute(const std::vector<int> &route) {
        nodes_covered.insert(nodes_covered.end(), route.begin(), route.end());
    }

    void addRoute(const std::vector<uint16_t> &route) {
        nodes_covered.insert(nodes_covered.end(), route.begin(), route.end());
    }
    /**
     * @brief Checks if a node has been visited.
     *
     * This function determines whether a node, identified by its node_id, has
     * been visited. It uses a bitmask (visited_bitmap) where each bit
     * represents the visit status of a node.
     *
     */
    bool visits(int node_id) const {
        return visited_bitmap[node_id / 64] & (1ULL << (node_id % 64));
    }

    /**
     * @brief Resets the state of the object to its initial values.
     *
     */
    inline void reset() {
        // Reset basic properties
        vertex = -1;
        cost = 0.0;
        node_id = -1;
        real_cost = 0.0;
        parent = nullptr;
        is_extended = false;
        is_dominated = false;
        fresh = true;
        // Reset resources container (assuming operator= clears properly)
        resources = {};

        // Clear and preallocate nodes_covered to avoid reallocations later.
        nodes_covered.clear();
        nodes_covered.reserve(N_SIZE / 3);

        // Clear children container.
        children.clear();

        // Zero out the bitmaps efficiently.
        std::memset(visited_bitmap.data(), 0,
                    visited_bitmap.size() * sizeof(uint64_t));
#ifdef UNREACHABLE_DOMINANCE
        std::memset(unreachable_bitmap.data(), 0,
                    unreachable_bitmap.size() * sizeof(uint64_t));
#endif

        // If using source mode mapping, clear it.
        SRC_MODE_BLOCK(SRCmap.clear();)
    }

    void addNode(int node) { nodes_covered.push_back(node); }

    /**
     * @brief Initializes the object with the given parameters.
     *
     */
    inline void initialize(int vertex, double cost,
                           const std::vector<double> &resources, int node_id) {
        this->vertex = vertex;
        this->cost = cost;

        // Assuming `resources` is a vector or array-like structure with the
        // same size as the input
        std::copy(resources.begin(), resources.end(), this->resources.begin());

        this->node_id = node_id;
    }

    bool operator>(const Label &other) const { return cost > other.cost; }

    bool operator<(const Label &other) const { return cost < other.cost; }
};

/**
 * @class LabelComparator
 * @brief Comparator class for comparing two Label objects based on their cost.
 *
 * This class provides an overloaded operator() that allows for comparison
 * between two Label pointers. The comparison is based on the cost attribute
 * of the Label objects, with the comparison being in descending order.
 */
class LabelComparator {
   public:
    bool operator()(Label *a, Label *b) { return a->cost > b->cost; }
};
