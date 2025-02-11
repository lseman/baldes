#pragma once

// Move config outside as namespace constants
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
constexpr int MAX_REMOVE_COUNT = 3;
constexpr double IMPROVEMENT_BONUS = 1.5;
constexpr double MAX_DETERIORATION = 0.1;
constexpr double OPERATOR_LEARNING_RATE = 0.1;
constexpr double INITIAL_TEMPERATURE = 100.0;
constexpr double COOLING_RATE = 0.95;
constexpr double REHEATING_FACTOR = 1.5;
constexpr int REHEAT_INTERVAL = 50;
}  // namespace LocalSearchConfig

// Permutation structures and helper functions
struct Permutation {
    std::vector<int> num;
    int den;

    Permutation(const std::vector<int> &n, int d) : num(n), den(d) {}
};
inline std::vector<Permutation> getPermutationsForSize5() {
    static const std::vector<std::pair<std::vector<int>, int>> base_perms = {
        {{2, 2, 1, 1, 1}, 4},
        {{3, 1, 1, 1, 1}, 4},
        {{3, 2, 2, 1, 1}, 5},
        {{2, 2, 1, 1, 1}, 3},
        {{3, 3, 2, 2, 1}, 4}};

    // Pre-calculate total size needed (can be computed at compile-time)
    constexpr size_t total_perms =
        10 + 5 + 30 + 10 + 30;  // Based on unique permutations possible
    std::vector<Permutation> all_perms;
    all_perms.reserve(total_perms);

    for (const auto &[nums, den] : base_perms) {
        std::vector<int> p = nums;
        do {
            all_perms.emplace_back(p, den);
        } while (std::next_permutation(p.begin(), p.end()));
    }
    return all_perms;
}

inline std::vector<Permutation> getPermutationsForSize4() {
    static const std::vector<int> base = {2, 1, 1, 1};
    std::vector<Permutation> perms;
    perms.reserve(4);  // We know exactly how many permutations we'll get

    std::vector<int> p = base;
    do {
        perms.emplace_back(p, 3);
    } while (std::next_permutation(p.begin(), p.end()));
    return perms;
}

// Custom hash function for vector<int>
namespace std {
template <>
struct hash<vector<int>> {
    size_t operator()(const vector<int> &v) const {
        size_t seed = v.size();
        for (const int &i : v) {
            seed ^= hash<int>{}(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
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
        return cost_score > other.cost_score;
    }
};

struct CandidateSet {
    std::vector<int> nodes;
    double violation;
    Permutation perm;
    std::vector<int> neighbor;
    double rhs = 0.0;

    CandidateSet(const std::vector<int> &n, double v, const Permutation &p,
                 const std::vector<int> &neigh, double r = 0.0)
        : nodes(n), violation(v), perm(p), neighbor(neigh), rhs(r) {}

    // Equality operator for comparison
    bool operator==(const CandidateSet &other) const {
        return nodes == other.nodes && perm.den == other.perm.den &&
               perm.num == other.perm.num;
    }

    // Less than operator for std::set
    bool operator<(const CandidateSet &other) const {
        return nodes < other.nodes;  // Compare only nodes for ordering
    }

    std::unordered_map<int, double>
        neighbor_scores;  // Track scores for each neighbor
    std::vector<std::pair<int, int>>
        tabu_moves;  // Store (position, node) pairs

    double best_violation_seen = 0.0;  // Track best violation for aspiration

    // Method to update neighbor scores based on node_scores
    void updateNeighborScores(const std::vector<std::set<int>> &node_scores) {
        neighbor_scores.clear();
        for (int node : nodes) {
            for (int potential_neighbor : node_scores[node]) {
                if (std::find(nodes.begin(), nodes.end(), potential_neighbor) ==
                    nodes.end()) {
                    neighbor_scores[potential_neighbor]++;
                }
            }
        }
    }

    // Method to get promising neighbors sorted by score
    std::vector<int> getPromisingNeighbors(int k = 10) const {
        std::vector<std::pair<int, double>> scored_neighbors(
            neighbor_scores.begin(), neighbor_scores.end());

        pdqsort(
            scored_neighbors.begin(), scored_neighbors.end(),
            [](const auto &a, const auto &b) { return a.second > b.second; });

        std::vector<int> result;
        for (int i = 0; i < std::min(k, (int)scored_neighbors.size()); ++i) {
            result.push_back(scored_neighbors[i].first);
        }
        return result;
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
        if (a.nodes != b.nodes) return a.nodes < b.nodes;
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
        XXH3_64bits_update(state, cs.nodes.data(),
                           cs.nodes.size() * sizeof(int));
        XXH3_64bits_update(state, cs.perm.num.data(),
                           cs.perm.num.size() * sizeof(int));
        XXH3_64bits_update(state, &cs.perm.den, sizeof(int));
        uint64_t hash = XXH3_64bits_digest(state);
        XXH3_freeState(state);
        return hash;
    }
    uint64_t mixed_hash(const CandidateSet &cs) const { return operator()(cs); }
};
