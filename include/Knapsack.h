#pragma once

#include <algorithm>
#include <map>    // For sparse DP optimization
#include <thread> // For parallelization (optional)
#include <vector>

class Knapsack {
private:
    struct Item {
        double value;  // The value of the item
        int    weight; // The weight of the item
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

    // Greedy heuristic to compute an upper bound (fractional knapsack)
    double greedyUpperBound() {
        // Use partial sort to only sort as much as needed
        std::partial_sort(items.begin(), items.begin() + std::min<size_t>(items.size(), 10), items.end(),
                          [](const Item &a, const Item &b) { return (a.value / a.weight) > (b.value / b.weight); });

        double totalValue    = 0.0;
        int    currentWeight = 0;

        for (const auto &item : items) {
            if (currentWeight + item.weight <= capacity) {
                // Take the whole item
                totalValue += item.value;
                currentWeight += item.weight;
            } else {
                // Take fractional part of the item
                double remainingCapacity = capacity - currentWeight;
                totalValue += (item.value * (remainingCapacity / item.weight));
                break; // Since we can't take any more items, we stop here
            }
        }
        return totalValue;
    }

    // Optimized dynamic programming solution with early termination
    double solve() {
        int n = items.size();

        // Use the greedy heuristic to quickly estimate an upper bound
        double greedySolution = greedyUpperBound();

        // Set a threshold for greedy solution (e.g., 90% of the optimal)
        const double greedyThreshold = 0.9 * greedySolution;

        // If the greedy solution is very close to optimal, return it
        if (greedySolution >= greedyThreshold) { return greedySolution; }

        // Otherwise, solve the problem exactly using dynamic programming
        std::vector<double> dp(capacity + 1, 0.0);

        // For large capacity problems, use parallelization
        const int numThreads = std::thread::hardware_concurrency();
        if (numThreads > 1) {
            // Parallelizing the DP update loop
            std::vector<std::thread> threads;
            auto                     updateDpRange = [&](int start, int end) {
                for (const auto &item : items) {
                    if (item.weight <= capacity) {
                        // Process from the back to avoid overwriting previous results
                        for (int w = end; w >= start; --w) {
                            if (w >= item.weight) { dp[w] = std::max(dp[w], dp[w - item.weight] + item.value); }
                        }
                    }
                }
            };

            int chunkSize = capacity / numThreads;
            for (int i = 0; i < numThreads; ++i) {
                int start = i * chunkSize;
                int end   = (i == numThreads - 1) ? capacity : (start + chunkSize);
                threads.push_back(std::thread(updateDpRange, start, end));
            }

            for (auto &t : threads) { t.join(); }
        } else {
            // Single-threaded DP update
            for (const auto &item : items) {
                if (item.weight <= capacity) {
                    // Process from the back to avoid overwriting previous results
                    for (int w = capacity; w >= item.weight; --w) {
                        dp[w] = std::max(dp[w], dp[w - item.weight] + item.value);
                    }
                }
            }
        }

        // The maximum value that can be achieved with the given capacity
        return dp[capacity];
    }

    // Clear the items for reuse
    void clear() { items.clear(); }
};
