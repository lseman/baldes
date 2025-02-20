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

// include CutHelper
#include "CutHelper.h"
#include "CutIntelligence.h"

// namespace std {
// template <>
// struct hash<std::pair<int, int>> {
//     size_t operator()(const std::pair<int, int> &p) const {
//         // Combine the hash values of both integers
//         size_t h1 = std::hash<int>{}(p.first);
//         size_t h2 = std::hash<int>{}(p.second);
//         return h1 ^ (h2 << 1);
//     }
// };
// }  // namespace std

class HighRankCuts {
   private:
    static constexpr int MIN_RANK = 3;
    static constexpr int MAX_RANK = 5;
    static constexpr int MAX_COMBINATIONS = 10;
    static constexpr int MAX_WORKING_SET_SIZE = 12;

    exec::static_thread_pool pool =
        exec::static_thread_pool(std::thread::hardware_concurrency());
    exec::static_thread_pool::scheduler sched = pool.get_scheduler();

    // Add vertex-route mapping
    std::vector<std::vector<int>> vertex_route_map;
    std::map<std::vector<int>, double> candidate_cache;

    void initializeVertexRouteMap() {
        // Initialize the vertex_route_map with N_SIZE rows and allPaths.size()
        // columns, all set to 0.
        vertex_route_map.assign(N_SIZE, std::vector<int>(allPaths.size(), 0));

        // Populate the map: for each path, count the appearance of each vertex.
        for (size_t r = 0; r < allPaths.size(); ++r) {
            for (const auto &vertex : allPaths[r].route) {
                // Only consider vertices that are not the depot (assumed at
                // indices 0 and N_SIZE-1).
                if (vertex > 0 && vertex < N_SIZE - 1) {
                    ++vertex_route_map[vertex][r];
                }
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    // Node scoring
    ////////////////////////////////////////////////////////////////////////////

    AdaptiveNodeScorer ml_scorer;
    std::vector<std::vector<int>> computeNodeScores(
        const SparseMatrix &A, const std::vector<double> &x) {
        auto scores = ml_scorer.computeNodeScores(A, x, distances, nodes,
                                                  arc_duals, *cutStorage);

        return scores;
    }
    // std::vector<std::vector<int>> computeNodeScores(
    //     const SparseMatrix &A, const std::vector<double> &x) {
    //     return ml_scorer->computeNodeScores(A, x, distances, nodes,
    //     arc_duals,
    //                                         *cutStorage);
    // }
    ankerl::unordered_dense::set<CandidateSet, CandidateSetHasher,
                                 CandidateSetCompare>
    generateCandidates(const std::vector<std::vector<int>> &scores,
                       const SparseMatrix &A, const std::vector<double> &x) {
        // Thread-local storage to avoid contention.
        struct ThreadLocalData {
            ankerl::unordered_dense::set<CandidateSet, CandidateSetHasher,
                                         CandidateSetCompare>
                candidates;
            ankerl::unordered_dense::map<std::vector<int>, double>
                candidate_cache;
            ThreadLocalData() {
                // Pre-reserve to reduce rehashing.
                candidates.reserve(1000);
                candidate_cache.reserve(1000);
            }
        };

        // Get the number of hardware threads.
        const int JOBS = std::thread::hardware_concurrency();
        // Create one ThreadLocalData per hardware thread.
        std::vector<ThreadLocalData> threadData(JOBS);

        // Final candidate set to be returned.
        ankerl::unordered_dense::set<CandidateSet, CandidateSetHasher,
                                     CandidateSetCompare>
            finalCandidates;

        // Bulk sender: parallelize over vertices (exclude first and last).
        auto bulk_sender =
            stdexec::bulk(stdexec::just(), (N_SIZE - 2), [&](std::size_t i) {
                // Skip if vertex is first or last.
                if (i == 0 || i == N_SIZE - 1) return;

                // Obtain a thread-local index using a thread_local atomic
                // counter.
                thread_local size_t threadIndex = []() -> size_t {
                    static std::atomic<size_t> counter{0};
                    return counter.fetch_add(1);
                }();
                auto &localData = threadData[threadIndex % JOBS];

                // Build a lookup (set) for the heuristic memory of vertex i.
                const auto &heuristic_memory = scores[i];
                ankerl::unordered_dense::set<int> heuristic_memory_lookup(
                    heuristic_memory.begin(), heuristic_memory.end());

                // Build the working set from routes that contain vertex i.
                std::vector<int> working_set;
                working_set.reserve(
                    std::min(static_cast<size_t>(MAX_WORKING_SET_SIZE),
                             allPaths.size()));
                for (size_t r = 0; r < allPaths.size(); ++r) {
                    // vertex_route_map[i][r] indicates if route r contains
                    // vertex i.
                    if (vertex_route_map[i][r] <= 0) continue;
                    for (const auto &v : allPaths[r].route) {
                        // Consider only valid vertices that are in heuristic
                        // memory.
                        if (v > 0 && v < N_SIZE - 1 &&
                            heuristic_memory_lookup.count(v)) {
                            working_set.push_back(v);
                        }
                    }
                }
                // Create a vector version of heuristic_memory for later use.
                std::vector<int> heuristic_memory_vector(
                    heuristic_memory.begin(), heuristic_memory.end());

                // Remove duplicates and sort working_set.
                pdqsort(working_set.begin(), working_set.end());
                working_set.erase(
                    std::unique(working_set.begin(), working_set.end()),
                    working_set.end());

                // Limit working_set size using a heuristic based on total
                // score.
                if (working_set.size() > MAX_WORKING_SET_SIZE) {
                    std::nth_element(
                        working_set.begin(),
                        working_set.begin() + MAX_WORKING_SET_SIZE,
                        working_set.end(),
                        [&scores](const int &a, const int &b) {
                            int sum_a = std::accumulate(scores[a].begin(),
                                                        scores[a].end(), 0);
                            int sum_b = std::accumulate(scores[b].begin(),
                                                        scores[b].end(), 0);
                            return sum_a < sum_b;
                        });
                    working_set.resize(MAX_WORKING_SET_SIZE);
                }

                // --- Generate candidate sets based on working_set ---
                // For each route that does NOT contain vertex i.
                for (size_t r = 0; r < allPaths.size(); ++r) {
                    if (vertex_route_map[i][r] > 0)
                        continue;  // Only process routes that do NOT contain
                                   // vertex i.
                    // Start candidate_set with vertex i.
                    std::vector<int> candidate_set = {static_cast<int>(i)};
                    candidate_set.reserve(working_set.size() + 1);
                    // Append working_set vertices.
                    for (const auto &v : working_set) {
                        candidate_set.push_back(v);
                    }
                    // Only consider candidate sets with valid size.
                    if (candidate_set.size() >= MIN_RANK &&
                        candidate_set.size() <= MAX_RANK) {
                        pdqsort(candidate_set.begin(), candidate_set.end());
                        // Check candidate cache.
                        auto cache_it =
                            localData.candidate_cache.find(candidate_set);
                        if (cache_it != localData.candidate_cache.end()) {
                            if (cache_it->second > 1e-3) {
                                auto [violation, perm, rhs] =
                                    computeViolationWithBestPerm(
                                        candidate_set, heuristic_memory_vector,
                                        A, x);
                                localData.candidates.emplace(
                                    candidate_set, violation, perm,
                                    heuristic_memory_vector, rhs);
                            }
                        } else {
                            auto [violation, perm, rhs] =
                                computeViolationWithBestPerm(
                                    candidate_set, heuristic_memory_vector, A,
                                    x);
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

                // --- Generate additional candidate sets by combining vertices
                // from working_set ---
                if (working_set.size() > 1) {
                    size_t combination_count = 0;
                    for (size_t k = 2;
                         k <= MAX_RANK && combination_count < MAX_COMBINATIONS;
                         ++k) {
                        if (k > working_set.size()) break;
                        // Start with a bitmask with the k lowest bits set.
                        uint32_t mask = (1u << k) - 1;
                        while (mask < (1u << working_set.size()) &&
                               combination_count < MAX_COMBINATIONS) {
                            std::vector<int> candidate_set = {
                                static_cast<int>(i)};
                            candidate_set.reserve(k + 1);
                            // Extract candidate vertices based on the current
                            // bitmask.
                            for (size_t bit = 0; bit < working_set.size();
                                 ++bit) {
                                if (mask & (1u << bit)) {
                                    candidate_set.push_back(working_set[bit]);
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
                                                heuristic_memory_vector, A, x);
                                        localData.candidates.emplace(
                                            candidate_set, violation, perm,
                                            heuristic_memory_vector, rhs);
                                        combination_count++;
                                    }
                                } else {
                                    auto [violation, perm, rhs] =
                                        computeViolationWithBestPerm(
                                            candidate_set,
                                            heuristic_memory_vector, A, x);
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
                            // Generate the next combination using Gosper's
                            // hack.
                            uint32_t rightmost_one = mask & -mask;
                            uint32_t next_higher_one = mask + rightmost_one;
                            uint32_t rightmost_ones = mask ^ next_higher_one;
                            uint32_t shifted_ones =
                                (rightmost_ones >> 2) / rightmost_one;
                            mask = next_higher_one | shifted_ones;
                        }
                    }
                }
            });

        // Submit the bulk sender on the scheduler and wait for all tasks to
        // complete.
        auto work = stdexec::starts_on(sched, std::move(bulk_sender));
        stdexec::sync_wait(std::move(work));

        // Merge thread-local candidate sets into the final candidate set.
        for (auto &data : threadData) {
            finalCandidates.insert(data.candidates.begin(),
                                   data.candidates.end());
        }
        return finalCandidates;
    }

    std::vector<Permutation> getPermutations(const int RANK) const {
        if (RANK == 3) {
            return getPermutationsForSize3();
        } else if (RANK == 4) {
            return getPermutationsForSize4();
        } else if (RANK == 5) {
            return getPermutationsForSize5();
        }
        return {};
    }

   public:
    std::tuple<double, Permutation, double> computeViolationWithBestPerm(
        const std::vector<int> &nodes, const std::vector<int> &memory,
        const SparseMatrix &A, const std::vector<double> &x) {
        static constexpr double EPSILON = 1e-3;
        const int RANK = static_cast<int>(nodes.size());

        // Initialize result variables.
        double best_violation = 0.0;
        double best_rhs = 0.0;
        const Permutation *best_perm = nullptr;

        // Initialize bit arrays for efficient set operations.
        std::array<uint64_t, num_words>
            candidate_bits{};  // all bits initially 0
        std::array<uint64_t, num_words>
            augmented_memory_bits{};  // all bits initially 0

        // Create a node order lookup vector; initialize with -1 (invalid
        // index).
        std::vector<int> node_order(N_SIZE, -1);

        // Populate candidate_bits and node_order based on candidate nodes.
        for (int i = 0; i < RANK; ++i) {
            const int node = nodes[i];
            const int word_idx = node / 64;
            const uint64_t bit_mask = bit_mask_lookup[node % 64];

            candidate_bits[word_idx] |= bit_mask;
            augmented_memory_bits[word_idx] |= bit_mask;
            node_order[node] = i;
        }

        // Add memory nodes to augmented_memory_bits.
        for (const int node : memory) {
            int word_idx = node / 64;
            augmented_memory_bits[word_idx] |= (1ULL << (node % 64));
        }

        // Get all permutations for the given rank (compute once).
        const auto &permutations = getPermutations(RANK);

        // Pre-allocate a vector for storing cut coefficients.
        std::vector<double> cut_coefficients;
        cut_coefficients.reserve(allPaths.size());

        // Evaluate each permutation.
        for (const auto &perm : permutations) {
            // Compute the RHS value from the permutation.
            double rhs = 0.0;
            for (int i = 0; i < RANK; ++i) {
                rhs += static_cast<double>(perm.num[i]) / perm.den;
            }
            rhs = std::floor(rhs);

            // Create an SRCPermutation from the current permutation.
            const SRCPermutation src_perm{perm.num, perm.den};

            // Reset coefficient vector and accumulator for LHS.
            cut_coefficients.clear();
            double lhs = 0.0;

            // Process each path from the global allPaths container.
            for (size_t path_idx = 0; path_idx < allPaths.size(); ++path_idx) {
                const double x_val = x[path_idx];

                // If the path's x value is effectively zero, skip computation.
                if (std::abs(x_val) < std::numeric_limits<double>::epsilon()) {
                    cut_coefficients.push_back(0.0);
                    continue;
                }

                // Compute the coefficient using the candidate bits, augmented
                // memory, current permutation, route from the path, and the
                // node ordering.
                const double coeff = computeLimitedMemoryCoefficient(
                    candidate_bits, augmented_memory_bits, src_perm,
                    allPaths[path_idx].route, node_order);
                cut_coefficients.push_back(coeff);
                lhs += coeff * x_val;
            }

            // Compute the violation: the difference between lhs and (rhs +
            // EPSILON).
            const double violation = lhs - (rhs + EPSILON);
            if (violation > best_violation) {
                best_violation = violation;
                best_perm = &perm;
                best_rhs = rhs;
            }
        }

        // Return the best violation, corresponding permutation (or default if
        // none), and best rhs.
        return {best_violation, best_perm ? *best_perm : Permutation({}, 0),
                best_rhs};
    }

    Xoroshiro128Plus rng;

    void addCutToCutStorage(const CandidateSet &candidate,
                            std::vector<int> &order) {
        // Initialize bit arrays for the candidate (C) and augmented memory
        // (AM).
        std::array<uint64_t, num_words> C = {};
        std::array<uint64_t, num_words> AM = {};

        // Set bits for candidate nodes.
        for (auto node : candidate.nodes) {
            C[node / 64] |= bit_mask_lookup[node % 64];
            AM[node / 64] |= bit_mask_lookup[node % 64];
        }
        // Set bits for candidate neighbors into the augmented memory.
        for (auto node : candidate.neighbor) {
            AM[node / 64] |= bit_mask_lookup[node % 64];
        }

        // Prepare the permutation from the candidate.
        SRCPermutation p;
        p.num = candidate.perm.num;
        p.den = candidate.perm.den;

        // Pre-allocate the cut coefficients vector (one value per path in
        // allPaths).
        std::vector<double> cut_coefficients(allPaths.size());
        for (size_t i = 0; i < allPaths.size(); ++i) {
            cut_coefficients[i] = computeLimitedMemoryCoefficient(
                C, AM, p, allPaths[i].route, order);
        }

        // Construct the Cut object using the candidate data.
        Cut cut(C, AM, cut_coefficients, p);
        cut.baseSetOrder = order;

        // Set the cut type based on the number of candidate nodes.
        const auto cut_rank = candidate.nodes.size();
        if (cut_rank == 4) {
            cut.type = CutType::FourRow;
        } else if (cut_rank == 5) {
            cut.type = CutType::FiveRow;
        }

        // Add the cut to the global cut storage.
        cutStorage->addCut(cut);
    }

    // std::unique_ptr<AdaptiveNodeScorer> ml_scorer;

   public:
    // HighRankCuts() { ml_scorer = std::make_unique<AdaptiveNodeScorer>();
    // }
    HighRankCuts() = default;

    CutStorage *cutStorage = nullptr;

    std::vector<Path> allPaths;

    ArcDuals arc_duals;

    std::vector<VRPNode> nodes;
    std::vector<std::vector<double>> distances;
    ankerl::unordered_dense::map<int, std::vector<int>> row_indices_map;

    void separate(const SparseMatrix &A, const std::vector<double> &x) {
        // Initialize route mapping and compute node scores.
        initializeVertexRouteMap();
        auto node_scores = computeNodeScores(A, x);

        // Generate candidate sets using the computed node scores.
        auto candidates = generateCandidates(node_scores, A, x);

        // Prepare for parallel local search.
        const int JOBS = std::thread::hardware_concurrency();
        std::mutex candidates_mutex;
        ankerl::unordered_dense::set<CandidateSet, CandidateSetHasher,
                                     CandidateSetCompare>
            improved_candidates;

        // Parallelize the local search on each candidate.
        auto bulk_sender = stdexec::bulk(
            stdexec::just(), candidates.size(),
            [this, &A, &x, &candidates_mutex, &node_scores,
             &improved_candidates, &candidates](std::size_t idx) {
                LocalSearch local_search(this);

                // Retrieve candidate at position idx.
                auto it = candidates.begin();
                std::advance(it, idx);
                const auto rhs = it->rhs;
                const auto violation = it->violation;

                // Perform local search to potentially improve the candidate.
                auto improved_list = local_search.solve(*it, A, x, node_scores);

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
            });

        // Execute the bulk sender on the scheduler and wait for completion.
        auto work = stdexec::starts_on(sched, std::move(bulk_sender));
        stdexec::sync_wait(std::move(work));

        // Provide feedback using the improved candidates.
        // ml_scorer.provideFeedback(improved_candidates);

        // Process and sort unique candidates based on violation.
        int initial_cut_size = cutStorage->size();
        std::vector<CandidateSet> unique_candidates(improved_candidates.begin(),
                                                    improved_candidates.end());
        pdqsort(unique_candidates.begin(), unique_candidates.end(),
                [](const CandidateSet &a, const CandidateSet &b) {
                    return a.violation > b.violation;
                });

        // Limit the number of cuts to add (max_cuts), and try up to max_trials.
        const int max_cuts = 10;
        int cuts_orig_size = cutStorage->size();
        int cuts_added = 0;
        int max_trials = 50;
        for (const auto &candidate : unique_candidates) {
            if (cuts_added >= max_cuts || max_trials <= 0) break;

            // Build the vertex ordering for the candidate.
            std::vector<int> order(N_SIZE);
            int ordering = 0;
            for (int node : candidate.nodes) {
                order[node] = ordering++;
            }
            addCutToCutStorage(candidate, order);
            cuts_added = cutStorage->size() - cuts_orig_size;
            --max_trials;
        }
        int final_cut_size = cutStorage->size();

        // Print summary of the candidate processing.
        print_cut(
            "Candidates: {} | Improved Candidates: {} | Added {} SRC 3-4-5 "
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
        int cumulative_sum = 0;
        const int denominator = p.den;

        // Loop over the positions in P, ignoring the first and last elements.
        for (size_t j = 1; j < P.size() - 1; ++j) {
            int vertex = P[j];

            // Determine the bit mask and word index for the vertex.
            const uint64_t mask = bit_mask_lookup[vertex % 64];
            const size_t word_index = vertex >> 6;

            // If the augmented memory does not contain the vertex, reset
            // cumulative sum.
            if (!(AM[word_index] & mask)) {
                cumulative_sum = 0;
            }
            // Otherwise, if the candidate set contains the vertex,
            // add the corresponding numerator from the permutation.
            else if (C[word_index] & mask) {
                int pos = order[vertex];
                cumulative_sum += p.num[pos];
                // If cumulative sum exceeds or equals the denominator, update
                // alpha.
                if (cumulative_sum >= denominator) {
                    cumulative_sum -= denominator;
                    alpha += 1;
                }
            }
        }

        return alpha;
    }
};
