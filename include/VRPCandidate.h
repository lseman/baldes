#pragma once

#include "Definitions.h"

#include <functional>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <variant>

class VRPCandidate {
public:
    // Enum to represent the type of the candidate

    // Fields common to all candidates
    int                sourceNode;    // Source node in the candidate
    int                targetNode;    // Target node in the candidate
    double             boundValue;    // Bound value for the candidate
    BranchingDirection boundType;     // Upper or lower bound type
    CandidateType      candidateType; // Type of the candidate (Vehicle, Node, or Edge)

    // Variant to hold different types of data based on candidate type
    std::optional<std::variant<int, std::pair<int, int>>> payload = std::nullopt;

    // Disable copy constructor and copy assignment
    VRPCandidate(const VRPCandidate &)            = delete;
    VRPCandidate &operator=(const VRPCandidate &) = delete;

    // Allow move constructor and move assignment
    VRPCandidate(VRPCandidate &&)            = default;
    VRPCandidate &operator=(VRPCandidate &&) = default;

    // Constructor with correct variant initialization
    template <typename T>
    VRPCandidate(int sourceNode, int targetNode, BranchingDirection boundType, double boundValue,
                 CandidateType candidateType, T &&payloadData)
        : sourceNode(sourceNode), targetNode(targetNode), boundType(boundType), boundValue(boundValue),
          candidateType(candidateType), payload(std::forward<T>(payloadData)) {}

    // Compute hash value (for use in unordered containers)
    size_t computeHash() const {
        size_t seed = 0;
        seed ^= std::hash<int>{}(sourceNode) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int>{}(static_cast<int>(boundType)) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int>{}(static_cast<int>(candidateType)) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        if (payload) {
            std::visit(
                [&](auto &&arg) {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<T, int>) {
                        seed ^= std::hash<int>{}(arg) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                    } else if constexpr (std::is_same_v<T, std::pair<int, int>>) {
                        seed ^= std::hash<int>{}(arg.first) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                        seed ^= std::hash<int>{}(arg.second) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                    }
                },
                *payload);
        }
        return seed;
    }

    // Equality operator (for use in unordered containers)
    bool operator==(const VRPCandidate &other) const {
        return sourceNode == other.sourceNode && boundType == other.boundType && boundValue == other.boundValue &&
               candidateType == other.candidateType && payload == other.payload;
    }

    // Print function to display the candidate information
    void print() const {
        std::cout << "Candidate type: ";
        switch (candidateType) {
        case CandidateType::Vehicle: std::cout << "VehicleCandidate, "; break;
        case CandidateType::Node: std::cout << "NodeCandidate, "; break;
        case CandidateType::Edge: std::cout << "EdgeCandidate, "; break;
        }
        std::cout << "Source node: " << sourceNode << ", Bound value: " << boundValue << ", Payload: ";
        if (payload) {
            std::visit(
                [](auto &&arg) {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<T, int>) {
                        std::cout << arg;
                    } else if constexpr (std::is_same_v<T, std::pair<int, int>>) {
                        std::cout << "(" << arg.first << ", " << arg.second << ")";
                    }
                },
                *payload);
        } else {
            std::cout << "None";
        }
        std::cout << std::endl;
    }
};
