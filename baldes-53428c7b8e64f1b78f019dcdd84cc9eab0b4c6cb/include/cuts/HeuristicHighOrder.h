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
    std::set<int> neighbor;
    double rhs = 0.0;

    CandidateSet(const std::vector<int> &n, double v, const Permutation &p,
                 const std::set<int> &neigh, double r = 0.0)
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

    // Method to check if a move is tabu
    bool isTabu(int position, int node) const {
        return std::find(tabu_moves.begin(), tabu_moves.end(),
                         std::make_pair(position, node)) != tabu_moves.end();
    }

    // Method to add a move to tabu list
    void addTabuMove(int position, int node, int tabu_tenure) {
        tabu_moves.emplace_back(position, node);
        if (tabu_moves.size() > tabu_tenure) {
            tabu_moves.erase(tabu_moves.begin());
        }
    }
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
    const int MAX_WORKING_SET_SIZE = 20;

    exec::static_thread_pool pool =
        exec::static_thread_pool(std::thread::hardware_concurrency());
    exec::static_thread_pool::scheduler sched = pool.get_scheduler();

    std::set<CandidateSet> generateCandidates(
        const std::vector<std::set<int>> &scores,
        const std::vector<std::vector<int>> &row_indices_map,
        const SparseMatrix &A, const std::vector<double> &x) {
        // Thread-local data to avoid contention
        struct ThreadLocalData {
            std::set<CandidateSet> candidates;
            ankerl::unordered_dense::map<std::vector<int>, double>
                candidate_cache;
        };

        // Vector to hold thread-local data
        std::vector<ThreadLocalData> threadData(
            std::thread::hardware_concurrency());

        // Mutex for final merging
        std::mutex finalMergeMutex;
        std::set<CandidateSet> finalCandidates;

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
                                        candidate_set, heuristic_memory,
                                        row_indices_map, A, x);
                                localData.candidates.emplace(
                                    candidate_set, violation, perm,
                                    heuristic_memory, rhs);
                            }
                        } else {
                            auto [violation, perm, rhs] =
                                computeViolationWithBestPerm(
                                    candidate_set, heuristic_memory,
                                    row_indices_map, A, x);
                            localData.candidate_cache[candidate_set] =
                                violation;

                            if (violation > 1e-3) {
                                localData.candidates.emplace(
                                    candidate_set, violation, perm,
                                    heuristic_memory, rhs);
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
                                                candidate_set, heuristic_memory,
                                                row_indices_map, A, x);
                                        localData.candidates.emplace(
                                            candidate_set, violation, perm,
                                            heuristic_memory, rhs);
                                        combination_count++;
                                    }
                                } else {
                                    auto [violation, perm, rhs] =
                                        computeViolationWithBestPerm(
                                            candidate_set, heuristic_memory,
                                            row_indices_map, A, x);
                                    localData.candidate_cache[candidate_set] =
                                        violation;

                                    if (violation > 1e-3) {
                                        localData.candidates.emplace(
                                            candidate_set, violation, perm,
                                            heuristic_memory, rhs);
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
        const std::vector<int> &nodes, const std::set<int> &memory,
        const std::vector<std::vector<int>> &row_indices_map,
        const SparseMatrix &A, const std::vector<double> &x) {
        const int RANK = nodes.size();
        double best_violation = 0.0;
        Permutation *best_perm = nullptr;
        auto permutations = getPermutations(RANK);
        double best_rhs = 0.0;

        std::array<uint64_t, num_words> C = {};
        std::array<uint64_t, num_words> AM = {};

        std::vector<int> order(N_SIZE, 0);
        auto ordering = 0;
        for (auto node : nodes) {
            C[node / 64] |= (1ULL << (node % 64));
            AM[node / 64] |= (1ULL << (node % 64));
            order[node] = ordering++;
        }

        for (auto node : memory) {
            AM[node / 64] |= (1ULL << (node % 64));
        }

        for (auto &perm : permutations) {
            double rhs = 0;
            for (size_t i = 0; i < RANK; ++i) {
                rhs += static_cast<double>(perm.num[i]) / perm.den;
            }
            rhs = std::floor(rhs);
            SRCPermutation p;
            p.num = perm.num;
            p.den = perm.den;

            std::vector<double> cut_coefficients(allPaths.size());
            auto z = 0;
            auto lhs = 0.0;
            for (auto &path : allPaths) {
                if (x[z] == 0.0) {
                    z++;
                    continue;
                }
                cut_coefficients[z] = computeLimitedMemoryCoefficient(
                    C, AM, p, path.route, order);
                lhs += cut_coefficients[z] * x[z];
                z++;
            }

            auto violation = lhs - (rhs + 1e-3);

            if (violation > best_violation) {
                best_violation = lhs;
                best_perm = &perm;
                best_rhs = rhs;
            }
        }

        return {best_violation, best_perm ? *best_perm : Permutation({}, 0),
                best_rhs};
    }

    Xoroshiro128Plus rng;

    std::vector<CandidateSet> localSearch(
        const CandidateSet &initial,
        const std::vector<std::vector<int>> &row_indices_map,
        const SparseMatrix &A, const std::vector<double> &x,
        const std::vector<std::set<int>> &node_scores,
        int max_iterations = 100) {
        CandidateSet best = initial;
        CandidateSet current = initial;

        // Adaptive parameters
        const double MIN_WEIGHT = 0.01;
        const int SEGMENT_SIZE =
            20;  // Number of iterations per segment for adaptation
        double acceptance_threshold =
            0.0;  // Dynamic threshold for accepting worse solutions

        // Track solution history for adaptation
        struct SegmentStats {
            double avg_violation = 0.0;
            double best_violation = 0.0;
            int improvements = 0;
            int accepted_moves = 0;
        };
        std::deque<SegmentStats> history;
        SegmentStats current_segment;

        // Track diverse solutions
        std::vector<CandidateSet> diverse_solutions;
        const int MAX_DIVERSE_SOLUTIONS = 5;
        const double DIVERSITY_THRESHOLD = 0.3;

        // ALNS operator scores and weights
        enum OperatorType { SWAP_NODES, REMOVE_ADD_NODE, UPDATE_NEIGHBORS };
        std::vector<double> operator_scores = {1.0, 1.0, 1.0};
        std::vector<int> operator_usage = {0, 0, 0};
        std::vector<int> operator_success = {0, 0, 0};

        // Main optimization loop
        int iterations_since_improvement = 0;
        int segment_iterations = 0;

        for (int iter = 0; iter < max_iterations; ++iter) {
            auto backup = current;
            bool improved = false;

            // Select operator with normalized weights
            std::vector<double> weights = operator_scores;
            double total_weight = 0.0;
            for (size_t i = 0; i < weights.size(); ++i) {
                if (operator_usage[i] > 0) {
                    weights[i] *= (static_cast<double>(operator_success[i]) /
                                   operator_usage[i]);
                }
                weights[i] = std::max(weights[i], MIN_WEIGHT);
                total_weight += weights[i];
            }
            for (auto &w : weights) w /= total_weight;

            // Select operator using weights
            double r = static_cast<double>(rng()) / rng.max();
            int selected_op = 0;
            double cumsum = weights[0];
            while (selected_op < weights.size() - 1 && r > cumsum) {
                selected_op++;
                cumsum += weights[selected_op];
            }
            operator_usage[selected_op]++;

            // Apply selected operator
            switch (selected_op) {
                case SWAP_NODES: {
                    if (current.nodes.size() >= 2) {
                        int pos1 = rng() % current.nodes.size();
                        int pos2;
                        do {
                            pos2 = rng() % current.nodes.size();
                        } while (pos2 == pos1);
                        std::swap(current.nodes[pos1], current.nodes[pos2]);
                    }
                    break;
                }

                case REMOVE_ADD_NODE: {
                    if (current.nodes.size() > MIN_RANK &&
                        !current.neighbor.empty()) {
                        // Remove random node
                        int remove_pos = rng() % current.nodes.size();
                        int removed_node = current.nodes[remove_pos];
                        current.nodes.erase(current.nodes.begin() + remove_pos);
                        current.neighbor.insert(removed_node);

                        // Add promising neighbor based on scores
                        std::vector<std::pair<int, double>> scored_neighbors;
                        for (int n : current.neighbor) {
                            double score = 0;
                            for (int node : current.nodes) {
                                if (node_scores[node].count(n)) score++;
                            }
                            scored_neighbors.emplace_back(n, score);
                        }

                        if (!scored_neighbors.empty()) {
                            pdqsort(scored_neighbors.begin(),
                                    scored_neighbors.end(),
                                    [](const auto &a, const auto &b) {
                                        return a.second > b.second;
                                    });

                            // Selection based on rank with some randomization
                            int rank = std::min<int>(
                                scored_neighbors.size() - 1,
                                std::exponential_distribution<>(
                                    1.0 /
                                    (1.0 + iterations_since_improvement))(rng));

                            int new_node = scored_neighbors[rank].first;
                            current.nodes.push_back(new_node);
                            current.neighbor.erase(new_node);
                        }
                    }
                    break;
                }

                case UPDATE_NEIGHBORS: {
                    int remove_count =
                        1 + rng() % std::min(3, (int)current.neighbor.size());
                    for (int i = 0; i < remove_count; ++i) {
                        if (current.neighbor.empty()) break;
                        auto it = current.neighbor.begin();
                        std::advance(it, rng() % current.neighbor.size());
                        current.neighbor.erase(it);
                    }

                    std::set<int> potential_neighbors;
                    for (int node : current.nodes) {
                        for (int neighbor : node_scores[node]) {
                            if (std::find(current.nodes.begin(),
                                          current.nodes.end(),
                                          neighbor) == current.nodes.end() &&
                                current.neighbor.count(neighbor) == 0) {
                                potential_neighbors.insert(neighbor);
                            }
                        }
                    }

                    int add_count = 1 + rng() % 3;
                    while (add_count-- > 0 && !potential_neighbors.empty()) {
                        auto it = potential_neighbors.begin();
                        std::advance(it, rng() % potential_neighbors.size());
                        current.neighbor.insert(*it);
                        potential_neighbors.erase(it);
                    }
                    break;
                }
            }

            // Evaluate new solution
            auto [new_violation, new_perm, rhs] = computeViolationWithBestPerm(
                current.nodes, current.neighbor, row_indices_map, A, x);

            double delta = new_violation - backup.violation;

            // Adaptive acceptance criteria
            bool accept = false;
            if (delta > 0) {
                accept = true;  // Always accept improvements
            } else if (history.size() >= 2) {
                // Use historical performance to determine acceptance
                double recent_improvement_rate =
                    static_cast<double>(history.back().improvements) /
                    SEGMENT_SIZE;

                // Accept worse moves more often if we're not finding
                // improvements
                double acceptance_rate = std::min(
                    0.5, std::max(0.1, 0.4 * (1.0 - recent_improvement_rate)));

                // Scale threshold based on solution quality trend
                double avg_recent_violation = 0.0;
                for (const auto &seg : history) {
                    avg_recent_violation += seg.avg_violation;
                }
                avg_recent_violation /= history.size();

                // More lenient acceptance if we're below average quality
                if (current.violation < avg_recent_violation) {
                    acceptance_rate *= 1.5;
                }

                accept =
                    (static_cast<double>(rng()) / rng.max() <
                     acceptance_rate) &&
                    (delta >
                     -std::abs(current.violation *
                               0.1));  // Don't accept very large deteriorations
            } else {
                // Early iterations - use more aggressive acceptance
                accept = (static_cast<double>(rng()) / rng.max() < 0.3);
            }

            if (accept) {
                current.violation = new_violation;
                current.perm = new_perm;
                current.rhs = rhs;

                // Update operator score and segment statistics
                operator_success[selected_op]++;
                operator_scores[selected_op] =
                    std::max(MIN_WEIGHT, operator_scores[selected_op] * 0.9 +
                                             0.1 * std::max(0.0, delta));

                current_segment.accepted_moves++;
                current_segment.avg_violation =
                    (current_segment.avg_violation * segment_iterations +
                     new_violation) /
                    (segment_iterations + 1);

                // Update best solution if improved
                if (new_violation > best.violation) {
                    best = current;
                    current_segment.improvements++;
                    iterations_since_improvement = 0;

                    // Update best violation for segment
                    current_segment.best_violation =
                        std::max(current_segment.best_violation, new_violation);
                } else {
                    iterations_since_improvement++;
                }

                // Maintain diverse solutions
                bool is_diverse = true;
                for (const auto &sol : diverse_solutions) {
                    if (std::abs(sol.violation - new_violation) <
                        DIVERSITY_THRESHOLD) {
                        is_diverse = false;
                        break;
                    }
                }

                if (is_diverse) {
                    diverse_solutions.push_back(current);
                    if (diverse_solutions.size() > MAX_DIVERSE_SOLUTIONS) {
                        diverse_solutions.erase(std::min_element(
                            diverse_solutions.begin(), diverse_solutions.end(),
                            [](const auto &a, const auto &b) {
                                return a.violation < b.violation;
                            }));
                    }
                }
            } else {
                current = backup;
                operator_scores[selected_op] =
                    std::max(MIN_WEIGHT, operator_scores[selected_op] * 0.9);
            }

            // Update segment statistics
            segment_iterations++;
            if (segment_iterations == SEGMENT_SIZE) {
                history.push_back(current_segment);
                if (history.size() > 3) {  // Keep last 3 segments
                    history.pop_front();
                }
                current_segment = SegmentStats();
                segment_iterations = 0;
            }

            // Adaptive restart strategy
            if (iterations_since_improvement > SEGMENT_SIZE &&
                !diverse_solutions.empty()) {
                // Select restart solution based on both quality and diversity
                std::vector<std::pair<double, int>> restart_candidates;
                for (size_t i = 0; i < diverse_solutions.size(); ++i) {
                    double quality_score =
                        diverse_solutions[i].violation / best.violation;
                    double diversity_score = 0.0;
                    for (const auto &other : diverse_solutions) {
                        if (&other != &diverse_solutions[i]) {
                            diversity_score +=
                                std::abs(other.violation -
                                         diverse_solutions[i].violation);
                        }
                    }
                    restart_candidates.emplace_back(
                        quality_score * 0.7 + diversity_score * 0.3, i);
                }

                pdqsort(restart_candidates.begin(), restart_candidates.end());
                current = diverse_solutions[restart_candidates.back().second];
                iterations_since_improvement = 0;
            }
        }

        // Return best diverse solutions
        diverse_solutions.push_back(best);
        pdqsort(diverse_solutions.begin(), diverse_solutions.end(),
                [](const auto &a, const auto &b) {
                    return a.violation > b.violation;
                });

        return diverse_solutions;
    }

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

        print_cut("Generated {} candidates\n", candidates.size());

        const int JOBS = std::thread::hardware_concurrency();

        std::mutex candidates_mutex;
        std::vector<CandidateSet> improved_candidates;

        auto bulk_sender = stdexec::bulk(
            stdexec::just(), candidates.size(),
            [this, &row_indices_map, &A, &x, &candidates_mutex, &node_scores,
             &improved_candidates, &candidates](std::size_t idx) {
                auto it = candidates.begin();
                std::advance(it, idx);
                auto rhs = it->rhs;
                auto violation = it->violation;
                auto improved_list =
                    localSearch(*it, row_indices_map, A, x, node_scores);
                // print improved.violation
                //
                {
                    bool improved_found = false;
                    for (auto improved : improved_list) {
                        if (improved.violation > violation) {
                            std::lock_guard<std::mutex> lock(candidates_mutex);
                            improved_candidates.push_back(improved);
                            improved_found = true;
                        }
                    }
                    if (!improved_found) {
                        std::lock_guard<std::mutex> lock(candidates_mutex);
                        improved_candidates.push_back(*it);
                    }
                }
            });

        auto work = stdexec::starts_on(sched, bulk_sender);
        stdexec::sync_wait(std::move(work));

        // print improved_candidates size
        print_cut("Improved candidates: {}\n", improved_candidates.size());

        pdqsort(improved_candidates.begin(), improved_candidates.end(),
                [](const auto &a, const auto &b) {
                    return a.violation > b.violation;
                });

        auto initial_cut_size = cutStorage->size();
        const int max_cuts = 10;
        for (int i = 0;
             i <
             std::min(max_cuts, static_cast<int>(improved_candidates.size()));
             ++i) {
            std::vector<int> order(N_SIZE);
            int ordering = 0;
            for (int node : improved_candidates[i].nodes) {
                order[node] = ordering++;
            }
            addCutToCutStorage(improved_candidates[i], order);
        }
        auto final_cut_size = cutStorage->size();
        print_cut("Added {} SRC 4-5 cuts\n", final_cut_size - initial_cut_size);
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
};
