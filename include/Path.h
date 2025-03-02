/**
 * @file Path.h
 * @brief Memory-optimized implementation of the Path struct.
 */

#pragma once
#include <ankerl/unordered_dense.h>  // For ankerl::unordered_dense::map

#include <algorithm>
#include <cstdint>
#include <future>  // For std::async and std::future
#include <iostream>
#include <limits>
#include <vector>

#include "Arc.h"
#include "Common.h"
#include "RNG.h"
#include "Serializer.h"

struct Path {
    // Use a more memory-efficient container for small integers
    std::vector<uint16_t> route;  // Assuming route nodes are < 65536
    float cost;                   // Using float for cost
    float red_cost = std::numeric_limits<float>::max();
    float frac_x = 0.0f;

    // Equality operator
    bool operator==(const Path &other) const {
        return route == other.route && cost == other.cost;
    }

    // Default constructor
    Path() : route{}, cost(0.0f) {}

    REFLECT(route, cost, red_cost)

    // Constructor with vector<int> and double cost
    Path(const std::vector<int> &r, double c) : cost(static_cast<float>(c)) {
        route.resize(r.size());
        std::transform(r.begin(), r.end(), route.begin(),
                       [](int val) { return static_cast<uint16_t>(val); });
        precomputeArcsAsync();  // Run precomputeArcs in the background
    }

    // Constructor with vector<uint16_t> and float cost
    Path(const std::vector<uint16_t> &r, float c) : route(r), cost(c) {
        precomputeArcsAsync();
    }

    // Constructor with vector<uint16_t> and double cost
    Path(const std::vector<uint16_t> &r, double c)
        : route(r), cost(static_cast<float>(c)) {
        precomputeArcsAsync();
    }

    // Iterator methods
    auto begin() { return route.begin(); }
    auto end() { return route.end(); }
    auto begin() const { return route.begin(); }
    auto end() const { return route.end(); }
    auto size() const { return route.size(); }
    uint16_t operator[](size_t i) const { return route[i]; }

    // Check if the route contains a specific node
    bool contains(uint16_t i) const {
        return std::find(route.begin(), route.end(), i) != route.end();
    }

    // Count occurrences of a specific node in the route
    int countOccurrences(uint16_t i) const {
        return static_cast<int>(std::count(route.begin(), route.end(), i));
    }

    // Count occurrences of a specific arc (i -> j) in the route
    int timesArc(uint16_t i, uint16_t j) const {
        int times = 0;
        const size_t sz = route.size();
        for (size_t n = 1; n < sz; ++n) {
            if (route[n - 1] == i && route[n] == j) {
                ++times;
            }
        }
        return times;
    }

    // Use a more compact representation for arc counts
    struct ArcKey {
        uint16_t from;
        uint16_t to;

        bool operator==(const ArcKey &other) const {
            return from == other.from && to == other.to;
        }
    };

    struct ArcKeyHash {
        size_t operator()(const ArcKey &key) const {
            // Simple hash: combine from and to into a 32-bit value
            return (static_cast<size_t>(key.from) << 16) | key.to;
        }
    };

    // More compact map to count arcs.
    ankerl::unordered_dense::map<ArcKey, uint16_t, ArcKeyHash> arcMap;

    // Add an arc (i -> j) to the arcMap.
    void addArc(uint16_t i, uint16_t j) {
        ArcKey arc{i, j};
        auto it = arcMap.find(arc);
        if (it != arcMap.end()) {
            it->second++;
        } else {
            arcMap.emplace(arc, 1);
        }
    }

    // Precompute arcs (iterates over the route and counts arcs)
    void precomputeArcs() {
        // Pre-allocate expected size for better performance.
        arcMap.reserve(route.size() > 0 ? route.size() - 1 : 0);
        for (size_t n = 0; n + 1 < route.size(); ++n) {
            addArc(route[n], route[n + 1]);
        }
    }

    // Run precomputeArcs asynchronously.
    std::future<void> precomputeArcsAsync() {
        return std::async(std::launch::async, [this]() { precomputeArcs(); });
    }

    std::vector<int> getIntVector() {
        std::vector<int> intRoute(route.begin(), route.end());
        return intRoute;
    }
};

struct PathHash {
    std::size_t operator()(const Path &p) const {
        // Use one-shot XXH3 function directly on the contiguous vector memory
        return XXH3_64bits(p.route.data(), p.route.size() * sizeof(p.route[0]));
    }
};
