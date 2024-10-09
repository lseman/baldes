/*
 * @file Arc.h
 * @brief This file contains the definition of the Arc struct and related structures.
 *
 * This file contains the definition of the Arc struct, which represents an arc between two nodes in a graph.
 * The Arc struct contains information about the source node, target node, resource increments, cost increment,
 * and other properties of the arc. It also includes related structures such as BucketArc and JumpArc.
 *
 */
#pragma once
#include "Common.h"
#include <functional>

#include "xxhash.h"

struct Arc {
    int                 from;
    int                 to;
    std::vector<double> resource_increment;
    double              cost_increment;
    bool                fixed    = false;
    double              priority = 0;

    Arc(int from, int to, const std::vector<double> &res_inc, double cost_inc);

    Arc(int from, int to, const std::vector<double> &res_inc, double cost_inc, bool fixed);

    Arc(int from, int to, const std::vector<double> &res_inc, double cost_inc, double priority);
};

/**
 * @struct BucketArc
 * @brief Represents an arc between two buckets.
 *
 * This struct contains information about an arc between two buckets in a solver.
 * It stores the index of the source bucket, the index of the target bucket,
 * the resource increments associated with the arc, the cost increment,
 * and a flag indicating whether the arc is fixed.
 *
 */
struct BucketArc {
    int                 from_bucket;
    int                 to_bucket;
    std::vector<double> resource_increment;
    double              cost_increment;
    bool                fixed = false;

    bool operator==(const BucketArc &other) const {
        return from_bucket == other.from_bucket && to_bucket == other.to_bucket;
    }

    BucketArc(int from, int to, const std::vector<double> &res_inc, double cost_inc);

    BucketArc(int from, int to, const std::vector<double> &res_inc, double cost_inc, bool fixed);

    // Overload < operator for map comparison
    bool operator<(const BucketArc &other) const {
        if (from_bucket != other.from_bucket) return from_bucket < other.from_bucket;
        if (to_bucket != other.to_bucket) return to_bucket < other.to_bucket;
        if (cost_increment != other.cost_increment) return cost_increment < other.cost_increment;
        if (resource_increment != other.resource_increment) return resource_increment < other.resource_increment;
        return fixed < other.fixed;
    }
};

/**
 * @struct JumpArc
 * @brief Represents a jump arc between two buckets.
 *
 * This struct contains information about a jump arc, including the base bucket, jump bucket,
 * resource increment, and cost increment.
 *
 */
struct JumpArc {
    int                 base_bucket;
    int                 jump_bucket;
    std::vector<double> resource_increment;
    double              cost_increment;

    JumpArc(int base, int jump, const std::vector<double> &res_inc, double cost_inc);
};

using ArcVariant = std::variant<Arc, BucketArc>;

struct RawArc {
    int from;
    int to;

    // Constructor with member initializer list
    RawArc(int from, int to) : from(from), to(to) {}

    // Equality operator for comparisons
    bool operator==(const RawArc &other) const { return std::tie(from, to) == std::tie(other.from, other.to); }
};

// Hash function for RawArc
struct RawArcHash {
    std::size_t operator()(const RawArc &arc) const {
        // Take the address of 'arc.from' and 'arc.to' for hashing
        XXH64_hash_t hash_value = XXH3_64bits(&arc.from, sizeof(arc.from));                  // Hash the 'from' field
        hash_value              = XXH3_64bits_withSeed(&arc.to, sizeof(arc.to), hash_value); // Combine with 'to' field

        return static_cast<std::size_t>(hash_value); // Return the final hash value
    }
};
