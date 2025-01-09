/**
 * @file Path.h
 * @brief Memory-optimized implementation of the Path struct.
 */

#pragma once
#include "Arc.h"
#include "Common.h"
#include "RNG.h"

struct Path {
    // Use a more memory-efficient container for small integers
    std::vector<uint16_t> route; // Changed from vector<int> assuming route nodes are < 65536
    float                 cost;  // Changed from double unless high precision is needed
    float                 red_cost = std::numeric_limits<float>::max();
    float                 frac_x   = 0.0f;

    bool operator==(const Path &other) const { return route == other.route && cost == other.cost; }

    Path() : route({}), cost(0.0f) {}

    Path(const std::vector<int> &r, double c) : cost(static_cast<float>(c)) {
        route.reserve(r.size());
        for (const int val : r) { route.push_back(static_cast<uint16_t>(val)); }
        precomputeArcs();
    }

    // Constructor that accepts vector<uint16_t>
    Path(const std::vector<uint16_t> &r, float c) : route(r), cost(c) { precomputeArcs(); }
    Path(const std::vector<uint16_t> &r, double c) : route(r), cost(c) { precomputeArcs(); }

    // Keep iterator methods but make them return uint16_t
    auto     begin() { return route.begin(); }
    auto     end() { return route.end(); }
    auto     begin() const { return route.begin(); }
    auto     end() const { return route.end(); }
    auto     size() const { return route.size(); }
    uint16_t operator[](size_t i) const { return route[i]; }

    bool contains(uint16_t i) const { return std::find(route.begin(), route.end(), i) != route.end(); }

    int countOccurrences(uint16_t i) const { return std::count(route.begin(), route.end(), i); }

    int timesArc(uint16_t i, uint16_t j) const {
        int          times = 0;
        const size_t sz    = route.size();
        for (size_t n = 1; n < sz; ++n) {
            if (route[n - 1] == i && route[n] == j) { times++; }
        }
        return times;
    }

    // Use a more compact representation for arc counts
    struct ArcKey {
        uint16_t from;
        uint16_t to;

        bool operator==(const ArcKey &other) const { return from == other.from && to == other.to; }
    };

    struct ArcKeyHash {
        size_t operator()(const ArcKey &key) const { return (size_t)key.from << 16 | key.to; }
    };

    // Replace std::pair with our more compact ArcKey
    ankerl::unordered_dense::map<ArcKey, uint16_t, ArcKeyHash> arcMap;

    void addArc(uint16_t i, uint16_t j) {
        ArcKey arc{i, j};
        auto   it = arcMap.find(arc);
        if (it != arcMap.end()) {
            it->second++;
        } else {
            arcMap.emplace(arc, 1);
        }
    }

    void precomputeArcs() {
        arcMap.reserve(route.size() - 1); // Pre-allocate expected size
        for (size_t n = 0; n < route.size() - 1; ++n) { addArc(route[n], route[n + 1]); }
    }
};

// Simplified hash function that doesn't allocate memory
struct PathHash {
    std::size_t operator()(const Path &p) const {
        size_t hash = 0;
        for (const auto &node : p.route) { hash = hash * 31 + node; }
        // Combine with cost using XOR and bit rotation
        uint32_t cost_bits;
        memcpy(&cost_bits, &p.cost, sizeof(cost_bits));
        hash ^= ((size_t)cost_bits << 32) | ((size_t)cost_bits >> 32);
        return hash;
    }
};
