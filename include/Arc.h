/*
 * @file Arc.h
 * @brief This file contains the definition of the Arc struct and related
 * structures.
 *
 * This file contains the definition of the Arc struct, which represents an arc
 * between two nodes in a graph. The Arc struct contains information about the
 * source node, target node, resource increments, cost increment, and other
 * properties of the arc. It also includes related structures such as BucketArc
 * and JumpArc.
 *
 */
#pragma once
#include <functional>

#include "Common.h"
#include "xxhash.h"

struct Arc {
    int from;
    int to;
    std::vector<double> resource_increment;
    double cost_increment;
    bool fixed;
    double priority;

    Arc()
        : from(-1), to(-1), cost_increment(0.0), fixed(false), priority(0.0) {}

    Arc(int from, int to, const std::vector<double> &res_inc, double cost_inc);

    Arc(int from, int to, const std::vector<double> &res_inc, double cost_inc,
        bool fixed);

    Arc(int from, int to, const std::vector<double> &res_inc, double cost_inc,
        double priority);

    // Equality operator
    bool operator==(const Arc &other) const {
        return from == other.from && to == other.to &&
               resource_increment == other.resource_increment &&
               cost_increment == other.cost_increment && fixed == other.fixed &&
               priority == other.priority;
    }
};

// Custom hash function for Arc
struct arc_hash {
    std::size_t operator()(const Arc &arc) const {
        XXH64_state_t *const state = XXH64_createState();
        XXH64_reset(state, 0);  // Use 0 as seed, or choose another seed

        // Add all members to the hash state
        XXH64_update(state, &arc.from, sizeof(arc.from));
        XXH64_update(state, &arc.to, sizeof(arc.to));
        XXH64_update(state, &arc.cost_increment, sizeof(arc.cost_increment));
        XXH64_update(state, &arc.fixed, sizeof(arc.fixed));
        XXH64_update(state, &arc.priority, sizeof(arc.priority));

        // Add the resource_increment vector
        if (!arc.resource_increment.empty()) {
            XXH64_update(state, arc.resource_increment.data(),
                         arc.resource_increment.size() * sizeof(double));
        }

        const std::size_t hash = XXH64_digest(state);
        XXH64_freeState(state);
        return hash;
    }
};

/**
 * @struct BucketArc
 * @brief Represents an arc between two buckets.
 *
 * This struct contains information about an arc between two buckets in a
 * solver. It stores the index of the source bucket, the index of the target
 * bucket, the resource increments associated with the arc, the cost increment,
 * and a flag indicating whether the arc is fixed.
 *
 */
struct BucketArc {
    int from_bucket;
    int to_bucket;
    std::vector<double> resource_increment;
    double cost_increment;
    bool jump = false;

    bool operator==(const BucketArc &other) const {
        return from_bucket == other.from_bucket && to_bucket == other.to_bucket;
    }

    BucketArc(int from, int to, const std::vector<double> &res_inc,
              double cost_inc);

    BucketArc(int from, int to, const std::vector<double> &res_inc,
              double cost_inc, bool fixed);

    // Overload < operator for map comparison
    bool operator<(const BucketArc &other) const {
        if (from_bucket != other.from_bucket)
            return from_bucket < other.from_bucket;
        if (to_bucket != other.to_bucket) return to_bucket < other.to_bucket;
        if (cost_increment != other.cost_increment)
            return cost_increment < other.cost_increment;
        if (resource_increment != other.resource_increment)
            return resource_increment < other.resource_increment;
        return jump < other.jump;
    }

    ~BucketArc() {
        resource_increment
            .clear();  // Explicitly clear to release vector memory if pooled
    }
};

/**
 * @struct JumpArc
 * @brief Represents a jump arc between two buckets.
 *
 * This struct contains information about a jump arc, including the base bucket,
 * jump bucket, resource increment, and cost increment.
 *
 */
struct JumpArc {
    int base_bucket;
    int jump_bucket;
    std::vector<double> resource_increment;
    double cost_increment;
    int to_job = -1;

    JumpArc(int base, int jump, const std::vector<double> &res_inc,
            double cost_inc);
    JumpArc(int base, int jump, const std::vector<double> &res_inc,
            double cost_inc, int to_job);
};

using ArcVariant = std::variant<Arc, BucketArc>;

struct RawArc {
    int from;
    int to;

    // Constructor with member initializer list
    RawArc(int from, int to) : from(from), to(to) {}

    // Equality operator for comparisons
    bool operator==(const RawArc &other) const {
        return std::tie(from, to) == std::tie(other.from, other.to);
    }
};

// Hash function for RawArc
struct RawArcHash {
    std::size_t operator()(const RawArc &arc) const {
        // Take the address of 'arc.from' and 'arc.to' for hashing
        XXH64_hash_t hash_value =
            XXH3_64bits(&arc.from, sizeof(arc.from));  // Hash the 'from' field
        hash_value = XXH3_64bits_withSeed(
            &arc.to, sizeof(arc.to), hash_value);  // Combine with 'to' field

        return static_cast<std::size_t>(
            hash_value);  // Return the final hash value
    }
};

class ArcList {
   public:
    // Add a predefined arc to the list
    void add_arc(const Arc &arc) { predefined_arcs.emplace_back(arc); }

    // Method to add arcs using int -> int format, with optional resource
    // increment and cost
    void add_connections(const std::vector<std::pair<int, int>> &connections,
                         const std::vector<double> &default_resource_increment =
                             {1.0},  // Default increment
                         double default_cost_increment = 0.0,
                         double default_priority = 1.0) {
        for (const auto &conn : connections) {
            int from = conn.first;
            int to = conn.second;

            // Create an Arc with the provided default resource increment and
            // cost increment
            Arc arc(from, to, default_resource_increment,
                    default_cost_increment, default_priority);
            add_arc(arc);
        }
    }

    // Check if an arc exists between two nodes
    bool has_arc(int from, int to) const {
        for (const auto &arc : predefined_arcs) {
            if (arc.from == from && arc.to == to) {
                return true;
            }
        }
        return false;
    }

    // Retrieve the arc if it exists, otherwise return nullptr
    const Arc *get_arc(int from, int to) const {
        for (const auto &arc : predefined_arcs) {
            if (arc.from == from && arc.to == to) {
                return &arc;
            }
        }
        return nullptr;
    }

    // define get_arcs
    const std::vector<Arc> &get_arcs() const { return predefined_arcs; }

   private:
    std::vector<Arc> predefined_arcs;  // List of predefined arcs
};

class RawArcList {
   public:
    // Add a predefined arc to the list
    void add_arc(const RawArc &arc) { predefined_arcs.emplace_back(arc); }

    // Check if an arc exists between two nodes
    bool has_arc(int from, int to) const {
        for (const auto &arc : predefined_arcs) {
            if (arc.from == from && arc.to == to) {
                return true;
            }
        }
        return false;
    }

    // define get_arcs
    const std::vector<RawArc> &get_arcs() const { return predefined_arcs; }

   private:
    std::vector<RawArc> predefined_arcs;  // List of predefined arcs
};
