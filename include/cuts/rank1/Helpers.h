/*
 * @file cuts/rank1/Helpers.h
 * @brief Declares CutHelper interfaces and types used by the BALDES solver.
 *
 * This file declares the CutHelper interfaces and helper functions used by the BALDES solver.
 *
 */

#pragma once

// Move config outside as namespace constants
#include "cuts/model/Cut.h"
#include <bit>
namespace LocalSearchConfig {
constexpr double MIN_WEIGHT             = 0.01;
constexpr int    SEGMENT_SIZE           = 20;
constexpr int    MAX_DIVERSE_SOLUTIONS  = 5;
constexpr double DIVERSITY_THRESHOLD    = 0.3;
constexpr double QUALITY_WEIGHT         = 0.7;
constexpr double DIVERSITY_WEIGHT       = 0.3;
constexpr double BASE_ACCEPTANCE_RATE   = 0.3;
constexpr double MIN_ACCEPTANCE_RATE    = 0.1;
constexpr double MAX_ACCEPTANCE_RATE    = 0.5;
constexpr int    MAX_REMOVE_COUNT       = 2;
constexpr int    MAX_ADD_COUNT          = 2;
constexpr double IMPROVEMENT_BONUS      = 1.5;
constexpr double MAX_DETERIORATION      = 0.1;
constexpr double OPERATOR_LEARNING_RATE = 0.1;
constexpr double INITIAL_TEMPERATURE    = 100.0;
constexpr double COOLING_RATE           = 0.95;
constexpr double REHEATING_FACTOR       = 1.5;
constexpr int    REHEAT_INTERVAL        = 50;
} // namespace LocalSearchConfig

// Inline helper function: Given a base vector and a denominator, generate all
// unique runtime permutations (using std::next_permutation) and return them as
// a vector of Permutations.
inline std::vector<SRCPermutation> generateRuntimePermutations(const std::vector<int> &base, int den) {
    std::vector<SRCPermutation> perms;
    std::vector<int>            temp = base;
    std::sort(temp.begin(), temp.end());
    do { perms.emplace_back(temp, den); } while (std::next_permutation(temp.begin(), temp.end()));
    return perms;
}

// Inline genetic generator: For a given candidate size, apply several heuristic
// plans, generate permutations for each plan, and return all as a vector of
// Permutations. This function mimics the structure of the reference generator
// with plans 0 through 6.
inline std::vector<SRCPermutation> generateGeneticPermutations(int candidateSize) {
    std::vector<SRCPermutation>                   allPerms;
    std::vector<std::pair<std::vector<int>, int>> plans;

    // Basic rank-3 and rank-5 patterns rely on small denominators.
    plans.emplace_back(std::vector<int>(candidateSize, 1), 2);

    if (candidateSize >= 4) {
        plans.emplace_back(std::vector<int>(candidateSize, 1), 3);
        plans.emplace_back(std::vector<int>(candidateSize, 1), 4);
    }

    if (candidateSize >= 5) { plans.emplace_back(std::vector<int>(candidateSize, 1), 5); }

    if (candidateSize >= 4) {
        auto base = std::vector<int>(candidateSize, 1);
        base[0]   = candidateSize - 2;
        plans.emplace_back(base, candidateSize - 1);
        plans.emplace_back(base, candidateSize);
        plans.emplace_back(base, candidateSize + 1);
    }

    if (candidateSize >= 5) {
        auto base = std::vector<int>(candidateSize, 1);
        base[0]   = candidateSize - 3;
        base[1]   = 2;
        plans.emplace_back(base, candidateSize - 1);
        plans.emplace_back(base, candidateSize);
    }

    if (candidateSize >= 5) {
        auto base = std::vector<int>(candidateSize, 1);
        base[0]   = candidateSize - 2;
        base[1]   = candidateSize - 2;
        if (candidateSize >= 3) base[2] = 2;
        plans.emplace_back(base, candidateSize - 2);
        plans.emplace_back(base, candidateSize - 1);
    }

    if (candidateSize >= 5) {
        auto base = std::vector<int>(candidateSize, 1);
        base[0]   = candidateSize - 2;
        base[1]   = 2;
        base[2]   = 2;
        plans.emplace_back(base, candidateSize);
    }

    if (candidateSize >= 4) {
        auto base = std::vector<int>(candidateSize, 1);
        base[0]   = candidateSize - 2;
        base[1]   = candidateSize - 2;
        base[2]   = 2;
        if (candidateSize >= 4) base[3] = 2;
        plans.emplace_back(base, candidateSize - 1);
        plans.emplace_back(base, candidateSize);
    }

    // Deduplicate plans by base/den pair.
    std::vector<std::pair<std::vector<int>, int>> unique_plans;
    for (auto &plan : plans) {
        bool found = false;
        for (const auto &existing : unique_plans) {
            if (existing.second == plan.second && existing.first == plan.first) {
                found = true;
                break;
            }
        }
        if (found) continue;
        unique_plans.emplace_back(plan);
        auto perms = generateRuntimePermutations(plan.first, plan.second);
        allPerms.insert(allPerms.end(), perms.begin(), perms.end());
    }

    return allPerms;
}

inline std::vector<SRCPermutation> generateExactPermutations(int candidateSize) {
    std::vector<std::pair<std::vector<int>, int>> plans;
    auto                                          add_plan = [&](std::vector<int> base, int den) {
        if (den < 2) return;
        plans.emplace_back(std::move(base), den);
    };

    if (candidateSize % 2 == 1) { add_plan(std::vector<int>(candidateSize, 1), 2); }

    if (candidateSize >= 5 && (candidateSize - 2) % 3 == 0) { add_plan(std::vector<int>(candidateSize, 1), 3); }

    if (candidateSize >= 4) {
        auto plan = std::vector<int>(candidateSize, 1);
        plan[0]   = candidateSize - 2;
        add_plan(plan, candidateSize - 1);
    }

    if (candidateSize >= 5) {
        auto plan2 = std::vector<int>(candidateSize, 1);
        plan2[0]   = candidateSize - 3;
        plan2[1]   = candidateSize - 3;
        plan2[2]   = 2;
        add_plan(plan2, candidateSize - 2);

        auto plan3 = std::vector<int>(candidateSize, 1);
        plan3[0]   = candidateSize - 2;
        plan3[1]   = candidateSize - 2;
        plan3[2]   = 2;
        plan3[3]   = 2;
        add_plan(plan3, candidateSize - 1);

        auto plan4 = std::vector<int>(candidateSize, 1);
        plan4[0]   = candidateSize - 3;
        plan4[1]   = 2;
        add_plan(plan4, candidateSize - 1);

        auto plan6 = std::vector<int>(candidateSize, 1);
        plan6[0]   = candidateSize - 2;
        plan6[1]   = 2;
        plan6[2]   = 2;
        add_plan(plan6, candidateSize);
    }

    if (candidateSize >= 4) {
        auto plan5 = std::vector<int>(candidateSize, 1);
        plan5[0]   = candidateSize - 2;
        add_plan(plan5, candidateSize - 1);
    }

    std::sort(plans.begin(), plans.end(), [](const auto &a, const auto &b) {
        if (a.second != b.second) return a.second < b.second;
        return a.first < b.first;
    });
    plans.erase(std::unique(plans.begin(), plans.end()), plans.end());

    std::vector<SRCPermutation> allPerms;
    for (auto &plan : plans) {
        auto perms = generateRuntimePermutations(plan.first, plan.second);
        allPerms.insert(allPerms.end(), perms.begin(), perms.end());
    }

    return allPerms;
}

struct IntVectorHasher {
    size_t operator()(const std::vector<int> &values) const noexcept {
        if (values.empty()) return 0;
        return static_cast<size_t>(XXH64(values.data(), values.size() * sizeof(int), 0));
    }
};

struct NodeScore {
    int    node       = 0;
    int    other_node = 0;
    double cost_score = 0.0;

    NodeScore() = default;

    NodeScore(int i, int j, double c) : node(i), other_node(j), cost_score(c) {}

    bool operator<(const NodeScore &other) const { return cost_score < other.cost_score; }
};

struct CandidateSet {
    ankerl::unordered_dense::set<int> nodes;
    double                            violation;
    SRCPermutation                    perm;
    ankerl::unordered_dense::set<int> neighbor;
    double                            rhs = 0.0;

    // CandidateSet(const std::vector<int> &n, double v, const Permutation &p,
    //              const std::vector<int> &neigh, double r = 0.0)
    //     : nodes(n), violation(v), perm(p), neighbor(neigh), rhs(r) {}

    CandidateSet(const ankerl::unordered_dense::set<int> &n, double v, const SRCPermutation &p,
                 const ankerl::unordered_dense::set<int> &neigh, double r = 0.0)
        : nodes(n), violation(v), perm(p), neighbor(neigh), rhs(r) {}

    CandidateSet(const std::vector<int> &n, double v, const SRCPermutation &p,
                 const ankerl::unordered_dense::set<int> &neigh, double r = 0.0)
        : nodes(n.begin(), n.end()), violation(v), perm(p), neighbor(neigh), rhs(r) {}

    CandidateSet(const std::vector<int> &n, double v, const SRCPermutation &p, const std::vector<int> &neigh,
                 double r = 0.0)
        : nodes(n.begin(), n.end()), violation(v), perm(p), neighbor(neigh.begin(), neigh.end()), rhs(r) {}

    // Equality operator for comparison
    bool operator==(const CandidateSet &other) const {
        return nodes == other.nodes && neighbor == other.neighbor && perm.den == other.perm.den &&
               perm.num == other.perm.num;
    }

    // Less than operator for ordered containers.
    bool operator<(const CandidateSet &other) const {
        if (nodes == other.nodes && perm.num == other.perm.num && perm.den == other.perm.den) {
            // If they're the same, keep the one with higher violation
            // by making it "less than" so it wins
            return violation > other.violation;
        }

        // For different elements, establish consistent ordering
        if (nodes != other.nodes) return nodes.size() < other.nodes.size();
        if (perm.num != other.perm.num) return perm.num < other.perm.num;
        return perm.den < other.perm.den;
    }
};

struct CandidateSetEqual {
    bool operator()(const CandidateSet &lhs, const CandidateSet &rhs) const noexcept { return lhs == rhs; }
};

struct CandidateSetHasher {
    using is_avalanching = void;

    static uint64_t mix(uint64_t value) noexcept {
        value += 0x9e3779b97f4a7c15ULL;
        value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31U);
    }

    static uint64_t hash_unordered_set(const ankerl::unordered_dense::set<int> &values, uint64_t seed) noexcept {
        // Commutative accumulators make the result independent of hash-table
        // iteration order without sorting or allocating temporary vectors.
        uint64_t sum  = mix(seed ^ values.size());
        uint64_t xors = 0;
        for (int value : values) {
            const uint64_t element_hash = mix(static_cast<uint32_t>(value) ^ seed);
            sum += element_hash;
            xors ^= std::rotl(element_hash, static_cast<int>(element_hash & 63U));
        }
        return mix(sum ^ xors);
    }

    uint64_t operator()(const CandidateSet &candidate) const noexcept {
        uint64_t hash = hash_unordered_set(candidate.nodes, 0x243f6a8885a308d3ULL);
        hash ^= std::rotl(hash_unordered_set(candidate.neighbor, 0x13198a2e03707344ULL), 17);
        hash ^= mix(static_cast<uint32_t>(candidate.perm.den));
        if (!candidate.perm.num.empty()) {
            hash ^= XXH3_64bits(candidate.perm.num.data(), candidate.perm.num.size() * sizeof(int));
        }
        return mix(hash);
    }

    uint64_t mixed_hash(const CandidateSet &candidate) const noexcept { return operator()(candidate); }
};

using CandidateSetCollection = ankerl::unordered_dense::set<CandidateSet, CandidateSetHasher, CandidateSetEqual>;

namespace std {
template <>
struct hash<CandidateSet> {
    size_t operator()(const CandidateSet &candidate) const noexcept { return CandidateSetHasher{}(candidate); }
};
} // namespace std
