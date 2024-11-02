/**
 * @file Path.h
 * @brief This file contains the definition of the Path struct.
 *
 * This file contains the definition of the Path struct, which represents a path with a route and its associated cost.
 * The Path struct encapsulates a route represented as a vector of integers and a cost associated with the route.
 * It provides various utility methods to interact with the route, such as checking for the presence of elements,
 * counting occurrences, and managing arcs between route points.
 *
 */

#pragma once
#include "Arc.h"
#include "Common.h"

#include "RNG.h"

/**
 * @struct Path
 * @brief Represents a path with a route and its associated cost.
 *
 * The Path struct encapsulates a route represented as a vector of integers and a cost associated with the
 * route. It provides various utility methods to interact with the route, such as checking for the presence of
 * elements, counting occurrences, and managing arcs between route points.
 *
 */
struct Path {
    std::vector<int> route;
    double           cost;
    double           red_cost = std::numeric_limits<double>::max();
    double           frac_x   = 0.0;
    bool             operator==(const Path &other) const { return route == other.route && cost == other.cost; }

    // default constructor
    Path() : route({}), cost(0.0) {}
    Path(const std::vector<int> &route, double cost) : route(route), cost(cost) {
        precomputeArcs(); // Precompute the arcs once the route is initialized

        // auto future = std::async(std::launch::async, &Path::precomputeArcs, this); // Ignore the future
    }

    // define begin and end methods linking to route
    auto begin() { return route.begin(); }
    auto end() { return route.end(); }
    // define begin and end const
    auto begin() const { return route.begin(); }
    auto end() const { return route.end(); }
    // define size
    auto size() { return route.size(); }
    // make the [] operator available
    int operator[](int i) const { return route[i]; }

    /**
     * @brief Checks if the given integer is present in the route.
     *
     * This function searches for the specified integer within the route
     * and returns true if the integer is found, otherwise false.
     *
     */
    bool contains(int i) { return std::find(route.begin(), route.end(), i) != route.end(); }

    /**
     * @brief Counts the occurrences of a given integer in the route.
     *
     * This function iterates through the 'route' container and counts how many times
     * the specified integer 'i' appears in it.
     *
     */
    int countOccurrences(int i) { return std::count(route.begin(), route.end(), i); }

    /**
     * @brief Counts the number of times an arc (i, j) appears in the route.
     *
     * This function iterates through the route and counts how many times the arc
     * from node i to node j appears consecutively.
     *
     */
    int timesArc(int i, int j) const {
        int       times = 0;
        const int size  = route.size();
        for (int n = 1; n < size; ++n) {
            if ((route[n - 1] == i && route[n] == j)) { times++; }
        }

        return times;
    }

    ankerl::unordered_dense::map<std::pair<int, int>, int> arcMap; // Maps arcs to their counts

    /**
     * @brief Adds an arc between two nodes and increments its count in the arc map.
     *
     * This function creates a pair representing an arc between nodes `i` and `j`,
     * and increments the count of this arc in the `arcMap`. If the arc does not
     * already exist in the map, it is added with an initial count of 1.
     *
     */
    void addArc(int i, int j) {
        std::pair<int, int> arc = std::make_pair(i, j);
        // check if the arc exists in the map
        if (arcMap.find(arc) != arcMap.end()) {
            arcMap[arc]++; // Increment the count if the arc exists
        } else {
            arcMap[arc] = 1; // Add the arc with a count of 1 if it does not exist
        }
    }

    /**
     * @brief Precomputes arcs for the given route.
     *
     * This function iterates through the route and adds arcs between consecutive nodes.
     * It assumes that the route is a valid sequence of nodes and that the addArc function
     * is defined to handle the addition of arcs between nodes.
     */
    void precomputeArcs() {
        for (int n = 0; n < route.size() - 1; ++n) { addArc(route[n], route[n + 1]); }
    }
};

inline int random_seed() {
    // Create an instance of Xoroshiro128Plus with a seed
    Xoroshiro128Plus rng; // You can set your seed here

    // Generate a random number and fit it into the desired range [0, 1000000]
    return rng() % 1000001; // Use modulo to constrain the value
}

struct PathHash {
    std::size_t operator()(const Path &p) const {
        // Initialize the seed for the xxhash function
        const uint64_t seed = random_seed();

        // Use XXH3_64bits for hashing instead of XXH64
        XXH3_state_t *state = XXH3_createState();
        XXH3_64bits_reset_withSeed(state, seed);

        // Hash the entire route vector
        for (const auto &node : p.route) { XXH3_64bits_update(state, &node, sizeof(node)); }

        // Hash the cost (assuming it's a double, which is 8 bytes)
        XXH3_64bits_update(state, &p.cost, sizeof(p.cost));

        // Finalize the hash
        std::size_t hash_val = XXH3_64bits_digest(state);

        // Clean up the state
        XXH3_freeState(state);
        return hash_val;
    }
};
