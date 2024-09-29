/**
* @file Path.h
* @brief This file contains the definition of the Path struct.
*/
#pragma once
#include "Common.h"

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

    // default constructor
    Path() : route({}), cost(0.0) {}
    Path(const std::vector<int> &route, double cost) : route(route), cost(cost) {
        [[maybe_unused]] auto future = std::async(std::launch::async, &Path::precomputeArcs, this); // Ignore the future
    }

    // define begin and end methods linking to route
    auto begin() { return route.begin(); }
    auto end() { return route.end(); }
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

    std::unordered_map<std::pair<int, int>, int, pair_hash> arcMap; // Maps arcs to their counts

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
        arcMap[arc]++; // Increment the count of the arc
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

    /**
     * @brief Retrieves the count of arcs between two nodes.
     *
     * This function takes two integers representing nodes and returns the count
     * of arcs between them. If the arc pair (i, j) exists in the arcMap, the
     * function returns the associated count. Otherwise, it returns 0.
     *
     */
    auto getArcCount(int i, int j) const {
        // Construct the arc pair
        std::pair<int, int> arc = std::make_pair(i, j);
        return (arcMap.find(arc) != arcMap.end()) ? arcMap.at(arc) : 0;
    }

    /**
     * @brief Retrieves the count of a specified arc.
     *
     * This function takes an arc represented by an RCCarc object and constructs
     * a pair from its 'from' and 'to' members. It then checks if this pair exists
     * in the arcMap. If the pair is found, the function returns the count associated
     * with the arc. If the pair is not found, it returns 0.
     *
     */
};
