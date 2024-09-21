/**
 * @file Knapsack.h
 * @brief Header file for the Knapsack class, which implements the solution to the knapsack problem.
 *
 * This file defines the `Knapsack` class, which provides methods to solve the knapsack problem using a dynamic
 * programming approach. The knapsack problem is a well-known combinatorial optimization problem where the goal
 * is to maximize the total value of items that can be placed in a knapsack with a given weight capacity.
 *
 * The main components of this file include:
 * - The `Item` structure: Holds the value and weight of individual items that can be placed in the knapsack.
 * - Methods for adding items, setting the capacity of the knapsack, solving the problem using dynamic programming,
 *   and clearing the item list for reuse.
 *
 * The dynamic programming solution is implemented using a single-row DP array to save space. Additionally, items
 * can be optionally sorted by value-to-weight ratio for heuristic or greedy approaches.
 */

#pragma once

#include <algorithm>
#include <vector>

/**
 * @class Knapsack
 * @brief A class to represent and solve the knapsack problem.
 *
 * The Knapsack class provides methods to set the capacity of the knapsack,
 * add items with specific values and weights, solve the knapsack problem
 * using a dynamic programming approach, and clear the items for reuse.
 *
 * @note The knapsack problem is a combinatorial optimization problem where
 *       the goal is to maximize the total value of items that can be placed
 *       in a knapsack with a given capacity.
 */
class Knapsack {
private:
    struct Item {
        double value;  // The value of the item (e.g., the dual value)
        int    weight; // The weight of the item (e.g., demand)
    };

    std::vector<Item> items;
    int               capacity;

public:
    // Constructor to initialize the knapsack with a given capacity
    Knapsack() : capacity(0) {}

    // Set the capacity of the knapsack
    void setCapacity(int cap) { capacity = cap; }

    // Add an item to the knapsack
    void addItem(double value, int weight) { items.push_back({value, weight}); }

    // Solve the knapsack problem and return the upper bound
    double solve() {
        int n = items.size();

        // Sort items by value-to-weight ratio (optional, useful for heuristic or greedy approaches)
        std::sort(items.begin(), items.end(),
                  [](const Item &a, const Item &b) { return (a.value / a.weight) > (b.value / b.weight); });

        // Use a single-row DP array to save space
        std::vector<double> dp(capacity + 1, 0.0);

        for (const auto &item : items) {
            // Process from the back to avoid overwriting previous results
            if (item.weight <= capacity) {
                for (int w = capacity; w >= item.weight; --w) {
                    dp[w] = std::max(dp[w], dp[w - item.weight] + item.value);
                }
            }
        }

        // The maximum value that can be achieved with the given capacity
        return dp[capacity];
    }

    // Clear the items for reuse
    void clear() { items.clear(); }
};
