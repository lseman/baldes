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

// Permutation structure: stores a vector of numerators and a denominator.
struct Permutation {
    std::vector<int> num;
    int den;

    Permutation(const std::vector<int> &n, int d) : num(n), den(d) {}
    Permutation() = default;
};

// Helper function: convert an std::array to an std::vector using only the first
// 'size' elements.
template <size_t N>
constexpr std::vector<int> to_vector(const std::array<int, N> &arr,
                                     size_t size) {
    std::vector<int> result;
    result.reserve(size);
    for (size_t i = 0; i < size; ++i) {
        result.push_back(arr[i]);
    }
    return result;
}

// Helper function: generate all permutations of the first N elements of the
// 'base' array, with the given denominator, and return a pair of the
// permutation array and the count.
template <size_t N>
constexpr auto generate_permutations(const std::array<int, N> &base, int den) {
    // We assume 120 is sufficient (i.e. 5! for N==5)
    std::array<Permutation, 120> result{};
    size_t count = 0;

    // Create a working copy; if N is less than 5, pad the remaining entries
    // with 0.
    std::array<int, 5> current{};
    std::copy(base.begin(), base.end(), current.begin());
    if constexpr (N < 5) {
        std::fill(current.begin() + N, current.end(), 0);
    }

    // Generate permutations over the first N elements.
    do {
        std::vector<int> vec = to_vector(current, N);
        result[count++] = Permutation(vec, den);
    } while (std::next_permutation(current.begin(), current.begin() + N));

    return std::make_pair(result, count);
}

constexpr auto getPermutationsForSize3() {
    constexpr std::array<int, 3> base{{1, 1, 1}};
    auto [perms, count] = generate_permutations(base, 2);

    std::vector<Permutation> result;
    result.reserve(1);
    for (size_t i = 0; i < count; ++i) {
        result.push_back(perms[i]);
    }
    return result;
}
// Generate all permutations for a candidate set of size 5 using a set of base
// permutations.
constexpr auto getPermutationsForSize5() {
    // Each pair holds a base array (of 5 elements) and a denominator.
    constexpr std::array<std::pair<std::array<int, 5>, int>, 5> base_perms{
        {{{{2, 2, 1, 1, 1}}, 4},
         {{{3, 1, 1, 1, 1}}, 4},
         {{{3, 2, 2, 1, 1}}, 5},
         {{{2, 2, 2, 1, 1}}, 3},
         {{{3, 3, 2, 2, 1}}, 4}}};

    constexpr size_t total_perms = 10 + 5 + 30 + 10 + 30;
    std::vector<Permutation> all_perms;
    all_perms.reserve(total_perms);

    // For each base, generate its permutations and append them to all_perms.
    for (const auto &[nums, den] : base_perms) {
        auto [perms, count] = generate_permutations(nums, den);
        for (size_t i = 0; i < count; ++i) {
            all_perms.push_back(perms[i]);
        }
    }

    return all_perms;
}

// Generate all permutations for a candidate set of size 4.
constexpr auto getPermutationsForSize4() {
    constexpr std::array<int, 4> base{{2, 1, 1, 1}};
    auto [perms, count] = generate_permutations(base, 3);

    std::vector<Permutation> result;
    result.reserve(4);
    for (size_t i = 0; i < count; ++i) {
        result.push_back(perms[i]);
    }
    return result;
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
        auto a = *this;
        auto b = other;
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
