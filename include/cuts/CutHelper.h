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

    // Plan 0: For candidate sizes that are odd.
    if (candidateSize % 2 != 0) {
        std::vector<int> base(candidateSize, 1);
        int den = 2;  // Example: denominator 2
        auto perms = generateRuntimePermutations(base, den);
        allPerms.insert(allPerms.end(), perms.begin(), perms.end());
    }

    // Plan 1: When (candidateSize - 2) is divisible by 3 and candidateSize
    // >= 5.
    if (candidateSize >= 5 && ((candidateSize - 2) % 3 == 0)) {
        std::vector<int> base(candidateSize, 1);
        int den = 3;  // Example: denominator 3
        auto perms = generateRuntimePermutations(base, den);
        allPerms.insert(allPerms.end(), perms.begin(), perms.end());
    }

    // Plan 2: For candidateSize >= 5.
    // Modify the base: set the first two entries to (candidateSize - 3) and the
    // third to 2.
    if (candidateSize >= 5) {
        std::vector<int> base(candidateSize, 1);
        base[0] = candidateSize - 3;
        base[1] = candidateSize - 3;
        if (candidateSize >= 3) {
            base[2] = 2;
        }
        int den = candidateSize - 2;  // Example denominator
        auto perms = generateRuntimePermutations(base, den);
        allPerms.insert(allPerms.end(), perms.begin(), perms.end());
    }

    // Plan 3: For candidateSize >= 5.
    // Modify the base: set the first two entries to (candidateSize - 2) and the
    // next two to 2.
    if (candidateSize >= 5 && candidateSize >= 4) {
        std::vector<int> base(candidateSize, 1);
        base[0] = candidateSize - 2;
        base[1] = candidateSize - 2;
        base[2] = 2;
        base[3] = 2;
        int den = candidateSize - 1;  // Example denominator
        auto perms = generateRuntimePermutations(base, den);
        allPerms.insert(allPerms.end(), perms.begin(), perms.end());
    }

    // Plan 4: For candidateSize >= 5.
    // Modify the base: set the first element to (candidateSize - 3) and the
    // second to 2.
    if (candidateSize >= 5) {
        std::vector<int> base(candidateSize, 1);
        base[0] = candidateSize - 3;
        if (candidateSize >= 2) base[1] = 2;
        int den = candidateSize - 1;
        auto perms = generateRuntimePermutations(base, den);
        allPerms.insert(allPerms.end(), perms.begin(), perms.end());
    }

    // Plan 5: For candidateSize >= 4.
    // Modify the base: set the first element to (candidateSize - 2).
    if (candidateSize >= 4) {
        std::vector<int> base(candidateSize, 1);
        base[0] = candidateSize - 2;
        int den = candidateSize - 1;
        auto perms = generateRuntimePermutations(base, den);
        allPerms.insert(allPerms.end(), perms.begin(), perms.end());
    }

    // Plan 6: For candidateSize >= 5.
    // Modify the base: set the first element to (candidateSize - 2) and the
    // next two to 2.
    if (candidateSize >= 5 && candidateSize >= 3) {
        std::vector<int> base(candidateSize, 1);
        base[0] = candidateSize - 2;
        base[1] = 2;
        base[2] = 2;
        int den = candidateSize;  // Example denominator
        auto perms = generateRuntimePermutations(base, den);
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
