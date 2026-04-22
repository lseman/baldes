#pragma once

// Move config outside as namespace constants
#include "Cut.h"
namespace LocalSearchConfig {
constexpr double MIN_WEIGHT = 0.01;
constexpr int SEGMENT_SIZE = 20;
constexpr int MAX_DIVERSE_SOLUTIONS = 5;
constexpr double DIVERSITY_THRESHOLD = 0.3;
constexpr double QUALITY_WEIGHT = 0.7;
constexpr double DIVERSITY_WEIGHT = 0.3;
constexpr double BASE_ACCEPTANCE_RATE = 0.3;
constexpr double MIN_ACCEPTANCE_RATE = 0.1;
constexpr double MAX_ACCEPTANCE_RATE = 0.5;
constexpr int MAX_REMOVE_COUNT = 2;
constexpr int MAX_ADD_COUNT = 2;
constexpr double IMPROVEMENT_BONUS = 1.5;
constexpr double MAX_DETERIORATION = 0.1;
constexpr double OPERATOR_LEARNING_RATE = 0.1;
constexpr double INITIAL_TEMPERATURE = 100.0;
constexpr double COOLING_RATE = 0.95;
constexpr double REHEATING_FACTOR = 1.5;
constexpr int REHEAT_INTERVAL = 50;
}  // namespace LocalSearchConfig

// Inline helper function: Given a base vector and a denominator, generate all
// unique runtime permutations (using std::next_permutation) and return them as
// a vector of Permutations.
inline std::vector<SRCPermutation> generateRuntimePermutations(
    const std::vector<int> &base, int den) {
    std::vector<SRCPermutation> perms;
    std::vector<int> temp = base;
    std::sort(temp.begin(), temp.end());
    do {
        perms.emplace_back(temp, den);
    } while (std::next_permutation(temp.begin(), temp.end()));
    return perms;
}

// Inline genetic generator: For a given candidate size, apply several heuristic
// plans, generate permutations for each plan, and return all as a vector of
// Permutations. This function mimics the structure of the reference generator
// with plans 0 through 6.
inline std::vector<SRCPermutation> generateGeneticPermutations(
    int candidateSize) {
    std::vector<SRCPermutation> allPerms;
    std::vector<std::pair<std::vector<int>, int>> plans;

    // Basic rank-3 and rank-5 patterns rely on small denominators.
    plans.emplace_back(std::vector<int>(candidateSize, 1), 2);

    if (candidateSize >= 4) {
        plans.emplace_back(std::vector<int>(candidateSize, 1), 3);
        plans.emplace_back(std::vector<int>(candidateSize, 1), 4);
    }

    if (candidateSize >= 5) {
        plans.emplace_back(std::vector<int>(candidateSize, 1), 5);
    }

    if (candidateSize >= 4) {
        auto base = std::vector<int>(candidateSize, 1);
        base[0] = candidateSize - 2;
        plans.emplace_back(base, candidateSize - 1);
        plans.emplace_back(base, candidateSize);
        plans.emplace_back(base, candidateSize + 1);
    }

    if (candidateSize >= 5) {
        auto base = std::vector<int>(candidateSize, 1);
        base[0] = candidateSize - 3;
        base[1] = 2;
        plans.emplace_back(base, candidateSize - 1);
        plans.emplace_back(base, candidateSize);
    }

    if (candidateSize >= 5) {
        auto base = std::vector<int>(candidateSize, 1);
        base[0] = candidateSize - 2;
        base[1] = candidateSize - 2;
        if (candidateSize >= 3) base[2] = 2;
        plans.emplace_back(base, candidateSize - 2);
        plans.emplace_back(base, candidateSize - 1);
    }

    if (candidateSize >= 5) {
        auto base = std::vector<int>(candidateSize, 1);
        base[0] = candidateSize - 2;
        base[1] = 2;
        base[2] = 2;
        plans.emplace_back(base, candidateSize);
    }

    if (candidateSize >= 4) {
        auto base = std::vector<int>(candidateSize, 1);
        base[0] = candidateSize - 2;
        base[1] = candidateSize - 2;
        base[2] = 2;
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

namespace std {
template <>
struct hash<std::vector<int>> {
    size_t operator()(const std::vector<int> &v) const {
        // Handle empty vectors.
        if (v.empty()) {
            return 0;
        }
        // Compute the hash using XXH64.
        // v.data() returns a pointer to contiguous storage.
        // The length is v.size() * sizeof(int) bytes.
        return static_cast<size_t>(XXH64(v.data(), v.size() * sizeof(int), 0));
    }
};
}  // namespace std

struct NodeScore {
    int node = 0;
    int other_node = 0;
    double cost_score = 0.0;

    NodeScore() = default;

    NodeScore(int i, int j, double c) : node(i), other_node(j), cost_score(c) {}

    bool operator<(const NodeScore &other) const {
        return cost_score < other.cost_score;
    }
};

struct CandidateSet {
    ankerl::unordered_dense::set<int> nodes;
    double violation;
    SRCPermutation perm;
    ankerl::unordered_dense::set<int> neighbor;
    double rhs = 0.0;

    // CandidateSet(const std::vector<int> &n, double v, const Permutation &p,
    //              const std::vector<int> &neigh, double r = 0.0)
    //     : nodes(n), violation(v), perm(p), neighbor(neigh), rhs(r) {}

    CandidateSet(const ankerl::unordered_dense::set<int> &n, double v,
                 const SRCPermutation &p,
                 const ankerl::unordered_dense::set<int> &neigh, double r = 0.0)
        : nodes(n), violation(v), perm(p), neighbor(neigh), rhs(r) {}

    // Equality operator for comparison
    bool operator==(const CandidateSet &other) const {
        return nodes == other.nodes && perm.den == other.perm.den &&
               perm.num == other.perm.num;
    }

    // Less than operator for std::set
    bool operator<(const CandidateSet &other) const {
        auto a = *this;
        auto b = other;
        if (a.nodes == b.nodes && a.perm.num == b.perm.num &&
            a.perm.den == b.perm.den) {
            // If they're the same, keep the one with higher violation
            // by making it "less than" so it wins
            return a.violation > b.violation;
        }

        // For different elements, establish consistent ordering
        if (a.nodes != b.nodes) return a.nodes.size() < b.nodes.size();
        if (a.perm.num != b.perm.num) return a.perm.num < b.perm.num;
        return a.perm.den < b.perm.den;
    }
};

struct CandidateSetCompare {
    bool operator()(const CandidateSet &a, const CandidateSet &b) const {
        // First check if they represent the same core elements
        if (a.nodes == b.nodes && a.perm.num == b.perm.num &&
            a.perm.den == b.perm.den) {
            // If they're the same, keep the one with higher violation
            // by making it "less than" so it wins
            return a.violation > b.violation;
        }

        // For different elements, establish consistent ordering
        if (a.nodes != b.nodes) return a.nodes.size() < b.nodes.size();
        if (a.perm.num != b.perm.num) return a.perm.num < b.perm.num;
        return a.perm.den < b.perm.den;
    }
};

struct CandidateSetHasher {
    using is_transparent = void;
    uint64_t operator()(const CandidateSet &cs) const {
        XXH3_state_t *state = XXH3_createState();
        assert(state != nullptr);
        XXH3_64bits_reset(state);
        // Convert unordered nodes to a sorted vector.
        std::vector<int> sorted_nodes(cs.nodes.begin(), cs.nodes.end());
        std::sort(sorted_nodes.begin(), sorted_nodes.end());
        XXH3_64bits_update(state, sorted_nodes.data(),
                           sorted_nodes.size() * sizeof(int));
        // Hash the permutation numerator (assumed to be a vector<int>).
        XXH3_64bits_update(state, cs.perm.num.data(),
                           cs.perm.num.size() * sizeof(int));
        // Hash the permutation denominator.
        XXH3_64bits_update(state, &cs.perm.den, sizeof(int));
        uint64_t hash = XXH3_64bits_digest(state);
        XXH3_freeState(state);
        return hash;
    }

    uint64_t mixed_hash(const CandidateSet &cs) const { return operator()(cs); }
};

namespace std {
template <>
struct hash<CandidateSet> {
    size_t operator()(const CandidateSet &cs) const {
        // Create a state for the hash.
        XXH3_state_t *state = XXH3_createState();
        assert(state != nullptr);
        XXH3_64bits_reset(state);
        // Convert unordered nodes to a sorted vector.
        std::vector<int> sorted_nodes(cs.nodes.begin(), cs.nodes.end());
        std::sort(sorted_nodes.begin(), sorted_nodes.end());
        XXH3_64bits_update(state, sorted_nodes.data(),
                           sorted_nodes.size() * sizeof(int));
        // Hash the permutation numerator (assumed to be a vector<int>).
        XXH3_64bits_update(state, cs.perm.num.data(),
                           cs.perm.num.size() * sizeof(int));
        // Hash the permutation denominator.
        XXH3_64bits_update(state, &cs.perm.den, sizeof(int));
        uint64_t hash_val = XXH3_64bits_digest(state);
        XXH3_freeState(state);
        return hash_val;
    }
};
}  // namespace std
