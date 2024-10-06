#pragma once

#include <functional>
#include <string>
#include <utility>

#include <functional>
#include <string>

struct VRPCandidate {
    int    routeIndex; // Route index for the VRP candidate
    int    sourceNode; // Source node in the VRP path
    double boundValue; // Bound value for the candidate (floor or ceil of fractional value)

    enum class BoundType { Upper, Lower }; // Type of bound (upper or lower)
    BoundType boundType;                   // Upper or lower bound type

    std::string name; // Candidate name (optional for debugging or output)
    size_t      hash; // Hash value for the candidate

    // Constructor
    VRPCandidate(int routeIndex, int sourceNode, BoundType boundType, double boundValue)
        : routeIndex(routeIndex), sourceNode(sourceNode), boundType(boundType), boundValue(boundValue) {
        name = "x[" + std::to_string(routeIndex) + "," + std::to_string(sourceNode) + "]";
        hash = hash_value();
    }

    // Hash function
    size_t hash_value() const {
        size_t seed = 0;
        seed ^= std::hash<int>{}(routeIndex) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int>{}(sourceNode) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int>{}(static_cast<int>(boundType)) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }

    // Equality operator (needed for hash maps/sets)
    bool operator==(const VRPCandidate &other) const {
        return routeIndex == other.routeIndex && sourceNode == other.sourceNode && boundType == other.boundType &&
               boundValue == other.boundValue;
    }

    // Optional: Custom hasher for unordered maps/sets
    struct Hasher {
        size_t operator()(const VRPCandidate &candidate) const { return candidate.hash_value(); }
    };
};
