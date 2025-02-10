#pragma once
#include <algorithm>
#include <numeric>
#include <set>

#include "Definitions.h"
#include "Path.h"
#include "RNG.h"
#include "SRC.h"
#include "VRPNode.h"

// adding duals
#include "Arc.h"
#include "Dual.h"

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
        // First check if they represent the same set and permutation
        bool same_candidate = (a.nodes == b.nodes && a.perm.den == b.perm.den &&
                               a.perm.num == b.perm.num);

        if (same_candidate) {
            // If they're the same candidate, keep the one with higher violation
            return a.violation > b.violation;
        }

        // If they're different candidates, treat them as different
        // We need some consistent ordering for the set to work
        if (a.nodes != b.nodes) return a.nodes > b.nodes;
        if (a.perm.num != b.perm.num) return a.perm.num > b.perm.num;
        return a.perm.den > b.perm.den;
    }
};

struct CandidateSetHasher {
    using is_transparent = void;

    uint64_t operator()(const CandidateSet &cs) const {
        XXH3_state_t *state = XXH3_createState();
        assert(state != nullptr);
        XXH3_64bits_reset(state);

        // Hash nodes vector as one block since it's contiguous
        XXH3_64bits_update(state, cs.nodes.data(),
                           cs.nodes.size() * sizeof(int));

        // Hash num vector as one block
        XXH3_64bits_update(state, cs.perm.num.data(),
                           cs.perm.num.size() * sizeof(int));

        // Hash den and violation
        XXH3_64bits_update(state, &cs.perm.den, sizeof(int));

        uint64_t hash = XXH3_64bits_digest(state);
        XXH3_freeState(state);
        return hash;
    }

    // mixed_hash required by ankerl::unordered_dense
    uint64_t mixed_hash(const CandidateSet &cs) const { return operator()(cs); }
};

class HighRankCuts {
   private:
    const int MIN_RANK = 4;
    const int MAX_RANK = 5;
    static constexpr int MAX_CANDIDATES_PER_NODE = 25;

    // Add vertex-route mapping
    std::vector<std::vector<int>> vertex_route_map;
    std::map<std::vector<int>, double> candidate_cache;

    void initializeVertexRouteMap() {
        vertex_route_map.assign(N_SIZE, std::vector<int>(allPaths.size(), 0));

        // Populate the map with route appearances for each vertex
        for (size_t r = 0; r < allPaths.size(); ++r) {
            for (const auto &vertex : allPaths[r].route) {
                if (vertex > 0 && vertex < N_SIZE - 1) {
                    ++vertex_route_map[vertex][r];
                }
            }
        }
    }

    std::vector<std::set<int>> computeNodeScores(const SparseMatrix &A,
                                                 const std::vector<double> &x) {
        std::vector<std::set<int>> scores;
        scores.resize(N_SIZE);
        auto active_cuts = cutStorage->getActiveCuts();

        for (int i = 1; i < N_SIZE - 1; ++i) {
            std::vector<NodeScore> node_scores;
            node_scores.reserve(N_SIZE);

            for (int j = 1; j < N_SIZE - 1; ++j) {
                if (i != j) {
                    auto cost_score = distances[i][j] -
                                      (nodes[i].cost + nodes[j].cost) / 2 -
                                      arc_duals.getDual(i, j);
                    for (const auto &cut : active_cuts) {
                        if (cut.type == CutType::ThreeRow &&
                            cut.isSRCset(i, j)) {
                            cost_score -= cut.dual_value;
                        }
                    }
                    node_scores.emplace_back(i, j, cost_score);
                }
            }

            std::partial_sort(node_scores.begin(),
                              node_scores.begin() + MAX_CANDIDATES_PER_NODE,
                              node_scores.end(),
                              [](const auto &a, const auto &b) {
                                  return a.cost_score < b.cost_score;
                              });

            std::set<int> &i_scores = scores[i];
            for (int j = 0; j < MAX_CANDIDATES_PER_NODE; ++j) {
                // scores.insert(node_scores[j].node);
                i_scores.insert(node_scores[j].other_node);
            }
        }

        return scores;
    }

    const int MAX_COMBINATIONS = 25;
    const int MAX_WORKING_SET_SIZE = 12;

    exec::static_thread_pool pool =
        exec::static_thread_pool(std::thread::hardware_concurrency());
    exec::static_thread_pool::scheduler sched = pool.get_scheduler();

    ankerl::unordered_dense::set<CandidateSet, CandidateSetHasher,
                                 CandidateSetCompare>
    generateCandidates(const std::vector<std::set<int>> &scores,
                       const std::vector<std::vector<int>> &row_indices_map,
                       const SparseMatrix &A, const std::vector<double> &x) {
        // Thread-local data to avoid contention
        struct ThreadLocalData {
            ankerl::unordered_dense::set<CandidateSet, CandidateSetHasher,
                                         CandidateSetCompare>
                candidates;

            ankerl::unordered_dense::map<std::vector<int>, double>
                candidate_cache;
        };

        // Vector to hold thread-local data
        std::vector<ThreadLocalData> threadData(
            std::thread::hardware_concurrency());

        // Mutex for final merging
        std::mutex finalMergeMutex;
        ankerl::unordered_dense::set<CandidateSet, CandidateSetHasher,
                                     CandidateSetCompare>
            finalCandidates;
        const int JOBS = std::thread::hardware_concurrency();

        // Parallelize the outer loop over i
        auto bulk_sender =
            stdexec::bulk(stdexec::just(), N_SIZE - 2, [&](std::size_t i) {
                if (i == 0 || i == N_SIZE - 1) return;

                // Get thread-local data
                thread_local size_t threadIndex = []() {
                    static std::atomic<size_t> counter{0};
                    return counter.fetch_add(1);
                }();
                auto &localData = threadData[threadIndex % JOBS];

                const auto &heuristic_memory = scores[i];
                ankerl::unordered_dense::set<int> heuristic_memory_lookup(
                    heuristic_memory.begin(), heuristic_memory.end());

                // Build working set from routes containing vertex i
                std::vector<int> working_set;
                for (size_t r = 0; r < allPaths.size(); ++r) {
                    if (vertex_route_map[i][r] <= 0) continue;

                    for (const auto &v : allPaths[r].route) {
                        if (v > 0 && v < N_SIZE - 1 &&
                            heuristic_memory_lookup.count(v)) {
                            working_set.push_back(v);
                        }
                    }
                }
                std::vector<int> heuristic_memory_vector(
                    heuristic_memory.begin(), heuristic_memory.end());

                // Remove duplicates and sort
                pdqsort(working_set.begin(), working_set.end());
                working_set.erase(
                    std::unique(working_set.begin(), working_set.end()),
                    working_set.end());

                // Limit the size of the working set based on a more intelligent
                // heuristic
                if (working_set.size() > MAX_WORKING_SET_SIZE) {
                    std::nth_element(
                        working_set.begin(),
                        working_set.begin() + MAX_WORKING_SET_SIZE,
                        working_set.end(), [&](int a, int b) {
                            return scores[i].count(a) >
                                   scores[i].count(b);  // Prioritize vertices
                                                        // with higher scores
                        });
                    working_set.resize(MAX_WORKING_SET_SIZE);
                }

                // Generate candidate sets
                for (size_t r = 0; r < allPaths.size(); ++r) {
                    if (vertex_route_map[i][r] > 0) continue;

                    std::vector<int> candidate_set = {static_cast<int>(i)};
                    for (const auto &v : working_set) {
                        candidate_set.push_back(v);
                    }

                    if (candidate_set.size() >= MIN_RANK &&
                        candidate_set.size() <= MAX_RANK) {
                        pdqsort(candidate_set.begin(), candidate_set.end());

                        auto cache_it =
                            localData.candidate_cache.find(candidate_set);
                        if (cache_it != localData.candidate_cache.end()) {
                            if (cache_it->second > 1e-3) {
                                auto [violation, perm, rhs] =
                                    computeViolationWithBestPerm(
                                        candidate_set, heuristic_memory_vector,
                                        row_indices_map, A, x);
                                localData.candidates.emplace(
                                    candidate_set, violation, perm,
                                    heuristic_memory_vector, rhs);
                            }
                        } else {
                            auto [violation, perm, rhs] =
                                computeViolationWithBestPerm(
                                    candidate_set, heuristic_memory_vector,
                                    row_indices_map, A, x);
                            localData.candidate_cache[candidate_set] =
                                violation;

                            if (violation > 1e-3) {
                                localData.candidates.emplace(
                                    candidate_set, violation, perm,
                                    heuristic_memory_vector, rhs);
                            }
                        }
                    }
                }

                // Generate additional candidate sets by combining vertices
                if (working_set.size() > 1) {
                    size_t combination_count = 0;

                    for (size_t k = 2;
                         k <= MAX_RANK && combination_count < MAX_COMBINATIONS;
                         ++k) {
                        std::vector<bool> selector(working_set.size(), false);
                        std::fill(selector.begin(), selector.begin() + k, true);

                        do {
                            std::vector<int> candidate_set = {
                                static_cast<int>(i)};
                            for (size_t j = 0; j < working_set.size(); ++j) {
                                if (selector[j]) {
                                    candidate_set.push_back(working_set[j]);
                                }
                            }

                            if (candidate_set.size() >= MIN_RANK &&
                                candidate_set.size() <= MAX_RANK) {
                                pdqsort(candidate_set.begin(),
                                        candidate_set.end());

                                auto cache_it = localData.candidate_cache.find(
                                    candidate_set);
                                if (cache_it !=
                                    localData.candidate_cache.end()) {
                                    if (cache_it->second > 1e-3) {
                                        auto [violation, perm, rhs] =
                                            computeViolationWithBestPerm(
                                                candidate_set,
                                                heuristic_memory_vector,
                                                row_indices_map, A, x);
                                        localData.candidates.emplace(
                                            candidate_set, violation, perm,
                                            heuristic_memory_vector, rhs);
                                        combination_count++;
                                    }
                                } else {
                                    auto [violation, perm, rhs] =
                                        computeViolationWithBestPerm(
                                            candidate_set,
                                            heuristic_memory_vector,
                                            row_indices_map, A, x);
                                    localData.candidate_cache[candidate_set] =
                                        violation;

                                    if (violation > 1e-3) {
                                        localData.candidates.emplace(
                                            candidate_set, violation, perm,
                                            heuristic_memory_vector, rhs);
                                        combination_count++;
                                    }
                                }
                            }
                        } while (std::prev_permutation(selector.begin(),
                                                       selector.end()) &&
                                 combination_count < MAX_COMBINATIONS);
                    }
                }
            });

        // Execute the bulk sender on the thread pool
        auto work = stdexec::starts_on(sched, std::move(bulk_sender));
        stdexec::sync_wait(std::move(work));

        // Merge thread-local results into the final set
        for (auto &data : threadData) {
            std::lock_guard<std::mutex> lock(finalMergeMutex);
            finalCandidates.insert(data.candidates.begin(),
                                   data.candidates.end());
        }

        return finalCandidates;
    }
    std::vector<Permutation> getPermutations(const int RANK) const {
        if (RANK == 4) {
            return getPermutationsForSize4();
        } else if (RANK == 5) {
            return getPermutationsForSize5();
        }
        return {};
    }

    std::tuple<double, Permutation, double> computeViolationWithBestPerm(
        const std::vector<int> &nodes, const std::vector<int> &memory,
        const std::vector<std::vector<int>> &row_indices_map,
        const SparseMatrix &A, const std::vector<double> &x) {
        static constexpr double EPSILON = 1e-3;
        const int RANK = static_cast<int>(nodes.size());

        // Pre-allocate result variables
        double best_violation = 0.0;
        double best_rhs = 0.0;
        const Permutation *best_perm = nullptr;

        // Initialize bit arrays for efficient set operations
        std::array<uint64_t, num_words> candidate_bits = {};
        std::array<uint64_t, num_words> augmented_memory_bits = {};

        // Create node ordering lookup
        std::vector<int> node_order(N_SIZE,
                                    -1);  // Initialize with invalid index

        // Populate bit arrays and ordering
        for (int i = 0; i < RANK; ++i) {
            const int node = nodes[i];
            const int word_idx = node / 64;
            const uint64_t bit_mask = bit_mask_lookup[node % 64];

            candidate_bits[word_idx] |= bit_mask;
            augmented_memory_bits[word_idx] |= bit_mask;
            node_order[node] = i;
        }

        // Add memory nodes to augmented memory bits
        for (const int node : memory) {
            augmented_memory_bits[node / 64] |= (1ULL << (node % 64));
        }

        // Get permutations only once
        const auto &permutations = getPermutations(RANK);

        // Pre-allocate vectors for coefficient computation
        std::vector<double> cut_coefficients;
        cut_coefficients.reserve(allPaths.size());

        // Process each permutation
        for (const auto &perm : permutations) {
            // Calculate RHS value
            double rhs = 0.0;
            for (int i = 0; i < RANK; ++i) {
                rhs += static_cast<double>(perm.num[i]) / perm.den;
            }
            rhs = std::floor(rhs);

            // Create SRC permutation
            const SRCPermutation src_perm{perm.num, perm.den};

            // Reset and compute cut coefficients
            cut_coefficients.clear();
            double lhs = 0.0;

            // Process paths efficiently
            for (size_t path_idx = 0; path_idx < allPaths.size(); ++path_idx) {
                const double x_val = x[path_idx];

                // Skip paths with zero coefficient
                if (std::abs(x_val) < std::numeric_limits<double>::epsilon()) {
                    cut_coefficients.push_back(0.0);
                    continue;
                }

                // Compute coefficient
                const double coeff = computeLimitedMemoryCoefficient(
                    candidate_bits, augmented_memory_bits, src_perm,
                    allPaths[path_idx].route, node_order);

                cut_coefficients.push_back(coeff);
                lhs += coeff * x_val;
            }

            // Check violation
            const double violation = lhs - (rhs + EPSILON);
            if (violation > best_violation) {
                best_violation = lhs;
                best_perm = &perm;
                best_rhs = rhs;
            }
        }

        // Return results, using default permutation if none found
        return {best_violation, best_perm ? *best_perm : Permutation({}, 0),
                best_rhs};
    }

    Xoroshiro128Plus rng;

    void addCutToCutStorage(const CandidateSet &candidate,
                            std::vector<int> &order) {
        std::array<uint64_t, num_words> C = {};
        std::array<uint64_t, num_words> AM = {};

        auto ordering = 0;
        for (auto node : candidate.nodes) {
            C[node / 64] |= bit_mask_lookup[node % 64];
            AM[node / 64] |= bit_mask_lookup[node % 64];
        }

        for (auto node : candidate.neighbor) {
            AM[node / 64] |= bit_mask_lookup[node % 64];
        }

        SRCPermutation p;
        p.num = candidate.perm.num;
        p.den = candidate.perm.den;

        std::vector<double> cut_coefficients(allPaths.size());
        auto z = 0;
        for (auto &path : allPaths) {
            cut_coefficients[z++] =
                computeLimitedMemoryCoefficient(C, AM, p, path.route, order);
        }

        Cut cut(C, AM, cut_coefficients, p);
        cut.baseSetOrder = order;
        auto cut_rank = candidate.nodes.size();
        if (cut_rank == 4) {
            cut.type = CutType::FourRow;
        } else if (cut_rank == 5) {
            cut.type = CutType::FiveRow;
        }
        cutStorage->addCut(cut);
    }

   public:
    HighRankCuts() {}

    CutStorage *cutStorage = nullptr;

    std::vector<Path> allPaths;

    ArcDuals arc_duals;

    std::vector<VRPNode> nodes;
    std::vector<std::vector<double>> distances;

    void separate(const SparseMatrix &A, const std::vector<double> &x) {
        initializeVertexRouteMap();
        auto node_scores = computeNodeScores(A, x);

        std::vector<std::vector<int>> row_indices_map(N_SIZE);
        for (int idx = 0; idx < A.values.size(); ++idx) {
            int row = A.rows[idx];
            if (row > N_SIZE - 2) continue;
            row_indices_map[row + 1].push_back(idx);
        }

        auto candidates =
            generateCandidates(node_scores, row_indices_map, A, x);

        // print_cut("Generated {} candidates\n", candidates.size());

        const int JOBS = std::thread::hardware_concurrency();

        std::mutex candidates_mutex;
        ankerl::unordered_dense::set<CandidateSet, CandidateSetHasher,
                                     CandidateSetCompare>
            improved_candidates;

        auto bulk_sender = stdexec::bulk(
            stdexec::just(), candidates.size(),
            [this, &row_indices_map, &A, &x, &candidates_mutex, &node_scores,
             &improved_candidates, &candidates](std::size_t idx) {
                LocalSearch local_search(this);

                auto it = candidates.begin();
                std::advance(it, idx);
                auto rhs = it->rhs;
                auto violation = it->violation;

                auto improved_list =
                    local_search.solve(*it, row_indices_map, A, x, node_scores);

                {
                    bool improved_found = false;
                    for (const auto &improved : improved_list) {
                        if (improved.violation > violation) {
                            improved_found = true;
                            std::lock_guard<std::mutex> lock(candidates_mutex);
                            improved_candidates.insert(improved);
                        }
                    }
                    if (!improved_found) {
                        std::lock_guard<std::mutex> lock(candidates_mutex);
                        improved_candidates.insert(*it);
                    }
                }
            });

        auto work = stdexec::starts_on(sched, bulk_sender);
        stdexec::sync_wait(std::move(work));

        // print improved_candidates size
        // print_cut("Improved candidates: {}\n", improved_candidates.size());

        auto cuts_called = 0;
        auto initial_cut_size = cutStorage->size();
        // Use the custom comparator in a set

        // Add all candidates - the set will automatically keep the one with
        // highest violation when duplicates are found
        std::vector<CandidateSet> unique_candidates(improved_candidates.begin(),
                                                    improved_candidates.end());
        pdqsort(unique_candidates.begin(), unique_candidates.end(),
                [](const CandidateSet &a, const CandidateSet &b) {
                    return a.violation > b.violation;
                });

        // Now process the unique candidates (limited to max_cuts)
        const int max_cuts = 10;
        int cuts_added = 0;
        for (const auto &candidate : unique_candidates) {
            if (cuts_added >= max_cuts) break;

            std::vector<int> order(N_SIZE);
            int ordering = 0;
            for (int node : candidate.nodes) {
                order[node] = ordering++;
            }
            addCutToCutStorage(candidate, order);
            cuts_added++;
        }
        auto final_cut_size = cutStorage->size();
        print_cut(
            "Candidates: {} | Improved Candidates: {} | Added {} SRC 4-5 "
            "cuts\n",
            candidates.size(), improved_candidates.size(),
            final_cut_size - initial_cut_size);
    }

    static constexpr std::array<uint64_t, 64> bit_mask_lookup = []() {
        std::array<uint64_t, 64> masks{};
        for (size_t i = 0; i < 64; ++i) {
            masks[i] = 1ULL << i;
        }
        return masks;
    }();

    double computeLimitedMemoryCoefficient(
        const std::array<uint64_t, num_words> &C,
        const std::array<uint64_t, num_words> &AM, const SRCPermutation &p,
        const std::vector<uint16_t> &P, std::vector<int> &order) noexcept {
        double alpha = 0.0;
        int S = 0;
        auto den = p.den;

        for (size_t j = 1; j < P.size() - 1; ++j) {
            int vj = P[j];

            uint64_t am_mask = bit_mask_lookup[vj % 64];
            uint64_t am_index = vj >> 6;

            if (!(AM[am_index] & am_mask)) {
                S = 0;
            } else if (C[am_index] & am_mask) {
                int pos = order[vj];

                S += p.num[pos];
                if (S >= den) {
                    S -= den;
                    alpha += 1;
                }
            }
        }

        return alpha;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Adaptive Large Neighborhood Search
    ////////////////////////////////////////////////////////////////////////////
    class LocalSearch {
       private:
        enum class OperatorType : uint8_t {
            SWAP_NODES,
            REMOVE_ADD_NODE,
            UPDATE_NEIGHBORS
        };

        struct alignas(64) OperatorStats {
            double score = 1.0;
            int32_t usage = 0;
            int32_t success = 0;
            double avg_improvement = 0.0;
        };

        struct alignas(32) SegmentStats {
            double avg_violation = 0.0;
            double best_violation = 0.0;
            int32_t improvements = 0;
            int32_t accepted_moves = 0;
        };

        std::array<OperatorStats, 3> operators;
        std::deque<SegmentStats> history;
        SegmentStats current_segment{};
        Xoroshiro128Plus rng;
        int iterations_since_improvement = 0;
        int segment_iterations = 0;
        double temperature;
        HighRankCuts *parent;

        double computeSimilarity(const std::vector<int> &set1,
                                 const std::vector<int> &set2) const {
            std::vector<int> intersection;
            intersection.reserve(std::min(set1.size(), set2.size()));

            std::set_intersection(set1.begin(), set1.end(), set2.begin(),
                                  set2.end(), std::back_inserter(intersection));

            return 2.0 * intersection.size() / (set1.size() + set2.size());
        }

        OperatorType selectOperator() {
            struct WeightedOperator {
                double cumulative_weight;
                OperatorType type;
            };

            std::array<WeightedOperator, 3> weighted_ops;
            double cumsum = 0.0;

            for (size_t i = 0; i < operators.size(); ++i) {
                cumsum += operators[i].score;
                weighted_ops[i] = {cumsum, static_cast<OperatorType>(i)};
            }

            std::uniform_real_distribution<double> dist(0.0, cumsum);
            const double r = dist(rng);

            return std::lower_bound(weighted_ops.begin(), weighted_ops.end(), r,
                                    [](const auto &op, double val) {
                                        return op.cumulative_weight < val;
                                    })
                ->type;
        }

        void applySwapNodes(CandidateSet &current) {
            if (current.nodes.size() >= 2) {
                std::uniform_int_distribution<> node_dist(
                    0, current.nodes.size() - 1);
                const int pos1 = node_dist(rng);
                int pos2;
                do {
                    pos2 = node_dist(rng);
                } while (pos2 == pos1);
                std::swap(current.nodes[pos1], current.nodes[pos2]);
            }
        }

        void applyRemoveAddNode(CandidateSet &current,
                                const std::vector<std::set<int>> &node_scores) {
            if (current.nodes.size() > 1 && !current.neighbor.empty()) {
                std::uniform_int_distribution<> node_dist(
                    0, current.nodes.size() - 1);
                const int remove_pos = node_dist(rng);
                const int removed_node = current.nodes[remove_pos];

                current.nodes.erase(current.nodes.begin() + remove_pos);
                // current.neighbor.insert(removed_node);

                std::vector<std::pair<int, double>> scored_neighbors;
                scored_neighbors.reserve(current.neighbor.size());

                for (const auto &n : current.neighbor) {
                    double score = std::count_if(
                        current.nodes.begin(), current.nodes.end(),
                        [&node_scores, n](int node) {
                            return node_scores[node].count(n);
                        });
                    scored_neighbors.emplace_back(n, score);
                }

                if (!scored_neighbors.empty()) {
                    pdqsort(scored_neighbors.begin(), scored_neighbors.end(),
                            [](const auto &a, const auto &b) {
                                return a.second > b.second;
                            });

                    double temp_factor = std::max(
                        0.1,
                        temperature / LocalSearchConfig::INITIAL_TEMPERATURE);
                    std::exponential_distribution<> exp_dist(
                        1.0 /
                        (temp_factor * (1.0 + iterations_since_improvement)));

                    const int rank =
                        std::min<int>(scored_neighbors.size() - 1,
                                      static_cast<int>(exp_dist(rng)));

                    const int new_node = scored_neighbors[rank].first;
                    current.nodes.push_back(new_node);
                    if (std::find(current.neighbor.begin(),
                                  current.neighbor.end(),
                                  new_node) == current.neighbor.end()) {
                        current.neighbor.push_back(new_node);
                    }
                }
            }
        }

        void applyUpdateNeighbors(
            CandidateSet &current,
            const std::vector<std::set<int>> &node_scores) {
            const int remove_count =
                std::min(LocalSearchConfig::MAX_REMOVE_COUNT,
                         static_cast<int>(current.neighbor.size()));

            std::vector<int> neighbor_vec(current.neighbor.begin(),
                                          current.neighbor.end());
            std::shuffle(neighbor_vec.begin(), neighbor_vec.end(), rng);

            for (int i = 0; i < remove_count; ++i) {
                // current.neighbor.erase(neighbor_vec[i]);
                current.neighbor.erase(std::find(current.neighbor.begin(),
                                                 current.neighbor.end(),
                                                 neighbor_vec[i]));
            }

            std::vector<int> potential_neighbors;
            for (const int &node : current.nodes) {
                for (const int &neighbor : node_scores[node]) {
                    if (!std::binary_search(current.nodes.begin(),
                                            current.nodes.end(), neighbor) &&
                        // current.neighbor.count(neighbor) == 0) {
                        std::find(current.neighbor.begin(),
                                  current.neighbor.end(),
                                  neighbor) == current.neighbor.end()) {
                        // potential_neighbors.insert(neighbor);
                        potential_neighbors.push_back(neighbor);
                    }
                }
            }

            std::vector<int> new_neighbors(potential_neighbors.begin(),
                                           potential_neighbors.end());
            std::shuffle(new_neighbors.begin(), new_neighbors.end(), rng);

            const int add_count =
                std::min(LocalSearchConfig::MAX_REMOVE_COUNT,
                         static_cast<int>(new_neighbors.size()));

            for (int i = 0; i < add_count; ++i) {
                // current.neighbor.insert(new_neighbors[i]);
                current.neighbor.push_back(new_neighbors[i]);
            }
        }

        bool acceptMove(double delta, const CandidateSet &current) {
            if (delta > 0) return true;

            std::uniform_real_distribution<double> dist(0.0, 1.0);
            const double acceptance_prob = std::exp(delta / temperature);

            if (history.size() >= 2) {
                const double recent_improvement_rate =
                    static_cast<double>(history.back().improvements) /
                    LocalSearchConfig::SEGMENT_SIZE;

                const double acceptance_rate =
                    std::clamp(0.4 * (1.0 - recent_improvement_rate),
                               LocalSearchConfig::MIN_ACCEPTANCE_RATE,
                               LocalSearchConfig::MAX_ACCEPTANCE_RATE);

                const double avg_recent_violation =
                    std::accumulate(history.begin(), history.end(), 0.0,
                                    [](double sum, const auto &seg) {
                                        return sum + seg.avg_violation;
                                    }) /
                    history.size();

                if (current.violation < avg_recent_violation) {
                    return (dist(rng) <
                            acceptance_rate *
                                LocalSearchConfig::IMPROVEMENT_BONUS) &&
                           (delta >
                            -std::abs(current.violation *
                                      LocalSearchConfig::MAX_DETERIORATION));
                }

                return (dist(rng) < acceptance_rate) &&
                       (delta >
                        -std::abs(current.violation *
                                  LocalSearchConfig::MAX_DETERIORATION));
            }

            return dist(rng) < acceptance_prob;
        }

        void updateStatistics(OperatorType op, double delta,
                              CandidateSet &current, double new_violation) {
            auto &op_stats = operators[static_cast<size_t>(op)];
            op_stats.usage++;

            if (delta > 0) {
                op_stats.success++;
                op_stats.avg_improvement =
                    0.9 * op_stats.avg_improvement + 0.1 * delta;
            }

            op_stats.score = std::max(
                LocalSearchConfig::MIN_WEIGHT,
                op_stats.score *
                        (1 - LocalSearchConfig::OPERATOR_LEARNING_RATE) +
                    LocalSearchConfig::OPERATOR_LEARNING_RATE *
                        (1.0 + std::max(0.0, delta)));

            current_segment.accepted_moves++;
            current_segment.avg_violation =
                (current_segment.avg_violation * segment_iterations +
                 new_violation) /
                (segment_iterations + 1);
        }

        void updateTemperature() {
            temperature *= LocalSearchConfig::COOLING_RATE;
            if (iterations_since_improvement >=
                LocalSearchConfig::REHEAT_INTERVAL) {
                temperature *= LocalSearchConfig::REHEATING_FACTOR;
                iterations_since_improvement = 0;
            }
        }

        void strategicRestart(CandidateSet &current, const CandidateSet &best,
                              std::vector<CandidateSet> &diverse_solutions) {
            if (iterations_since_improvement >
                    LocalSearchConfig::SEGMENT_SIZE &&
                !diverse_solutions.empty()) {
                std::vector<std::pair<double, size_t>> restart_candidates;
                restart_candidates.reserve(diverse_solutions.size());

                for (size_t i = 0; i < diverse_solutions.size(); ++i) {
                    const auto &sol = diverse_solutions[i];
                    const double quality_score = sol.violation / best.violation;
                    double diversity_score = 0.0;

                    for (const auto &other : diverse_solutions) {
                        if (&other != &sol) {
                            const double structural_div =
                                1.0 - computeSimilarity(sol.nodes, other.nodes);
                            const double violation_div =
                                std::abs(other.violation - sol.violation);
                            diversity_score +=
                                0.5 * (structural_div + violation_div);
                        }
                    }

                    restart_candidates.emplace_back(
                        quality_score * LocalSearchConfig::QUALITY_WEIGHT +
                            diversity_score *
                                LocalSearchConfig::DIVERSITY_WEIGHT,
                        i);
                }

                pdqsort(restart_candidates.begin(), restart_candidates.end());
                current = diverse_solutions[restart_candidates.back().second];
                iterations_since_improvement = 0;
                temperature = LocalSearchConfig::INITIAL_TEMPERATURE;
            }
        }

       public:
        LocalSearch(HighRankCuts *p)
            : rng(std::random_device{}()),
              temperature(LocalSearchConfig::INITIAL_TEMPERATURE),
              parent(p) {}

        std::vector<CandidateSet> solve(
            const CandidateSet &initial,
            const std::vector<std::vector<int>> &row_indices_map,
            const SparseMatrix &A, const std::vector<double> &x,
            const std::vector<std::set<int>> &node_scores,
            int max_iterations = 200) {
            CandidateSet best = initial;
            CandidateSet current = initial;
            std::vector<CandidateSet> diverse_solutions;
            diverse_solutions.reserve(LocalSearchConfig::MAX_DIVERSE_SOLUTIONS +
                                      1);

            for (int iter = 0; iter < max_iterations; ++iter) {
                const auto backup = current;
                const auto selected_op = selectOperator();

                switch (selected_op) {
                    case OperatorType::SWAP_NODES:
                        applySwapNodes(current);
                        break;
                    case OperatorType::REMOVE_ADD_NODE:
                        applyRemoveAddNode(current, node_scores);
                        break;
                    case OperatorType::UPDATE_NEIGHBORS:
                        applyUpdateNeighbors(current, node_scores);
                        break;
                }

                const auto [new_violation, new_perm, rhs] =
                    parent->computeViolationWithBestPerm(
                        current.nodes, current.neighbor, row_indices_map, A, x);

                const double delta = new_violation - backup.violation;

                if (acceptMove(delta, current)) {
                    current.violation = new_violation;
                    current.perm = std::move(new_perm);
                    current.rhs = std::move(rhs);

                    updateStatistics(selected_op, delta, current,
                                     new_violation);

                    if (new_violation > best.violation) {
                        best = current;
                        current_segment.improvements++;
                        iterations_since_improvement = 0;
                        current_segment.best_violation = std::max(
                            current_segment.best_violation, new_violation);
                    } else {
                        iterations_since_improvement++;
                    }

                    bool is_diverse = true;
                    for (const auto &sol : diverse_solutions) {
                        const double similarity =
                            computeSimilarity(current.nodes, sol.nodes);
                        if (similarity > 0.7 ||
                            std::abs(sol.violation - new_violation) <
                                LocalSearchConfig::DIVERSITY_THRESHOLD) {
                            is_diverse = false;
                            break;
                        }
                    }

                    if (is_diverse) {
                        diverse_solutions.push_back(current);
                        if (diverse_solutions.size() >
                            LocalSearchConfig::MAX_DIVERSE_SOLUTIONS) {
                            diverse_solutions.erase(std::min_element(
                                diverse_solutions.begin(),
                                diverse_solutions.end(),
                                [](const auto &a, const auto &b) {
                                    return a.violation < b.violation;
                                }));
                        }
                    }
                } else {
                    current = backup;
                    operators[static_cast<size_t>(selected_op)].score =
                        std::max(
                            LocalSearchConfig::MIN_WEIGHT,
                            operators[static_cast<size_t>(selected_op)].score *
                                (1 -
                                 LocalSearchConfig::OPERATOR_LEARNING_RATE));
                }

                // Update segment statistics
                segment_iterations++;
                if (segment_iterations == LocalSearchConfig::SEGMENT_SIZE) {
                    history.push_back(current_segment);
                    if (history.size() > 3) {
                        history.pop_front();
                    }
                    current_segment = SegmentStats{};
                    segment_iterations = 0;
                }

                updateTemperature();
                strategicRestart(current, best, diverse_solutions);
            }

            // Final solution processing
            diverse_solutions.push_back(std::move(best));

            pdqsort(diverse_solutions.begin(), diverse_solutions.end(),
                    [](const auto &a, const auto &b) {
                        return a.violation > b.violation;
                    });

            return diverse_solutions;
        }
    };
};
