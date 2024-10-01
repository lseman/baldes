#pragma once
#include "Common.h"

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

// Structure to represent a jump arc
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
