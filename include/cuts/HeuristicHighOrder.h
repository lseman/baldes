#pragma once
#include <algorithm>
#include <limits>
#include <map>

#include "Definitions.h"
#include "Dual.h"
#include "Path.h"
#include "RNG.h"
#include "VRPNode.h"
#include "utils/NumericUtils.h"

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
    static constexpr int MAX_COMBINATIONS     = 5;
    static constexpr int MAX_WORKING_SET_SIZE = 12;

    exec::static_thread_pool            pool  = exec::static_thread_pool(std::thread::hardware_concurrency());
    exec::static_thread_pool::scheduler sched = pool.get_scheduler();

    // Add vertex-route mapping
    std::map<std::vector<int>, double> candidate_cache;
    int                                last_path_idx = 0;
    std::vector<std::vector<int>>      rank1_sep_heur_mem4_vertex;
    void                               initializeVertexRouteMap() {
        if (last_path_idx == 0) {
            row_indices_map.clear();
            row_indices_map.reserve(N_SIZE);
        }

        // Initialize the vertex_route_map with N_SIZE rows and allPaths.size()
        // columns, all set to 0.
        if (last_path_idx == 0) {
            vertex_route_map.assign(N_SIZE, std::vector<int>(allPaths.size(), 0));
        } else {
            // assign 0 from last_path_idx to allPaths.size()
            for (size_t i = 0; i < N_SIZE; ++i) {
                vertex_route_map[i].resize(allPaths.size());
                for (size_t j = last_path_idx; j < allPaths.size(); ++j) { vertex_route_map[i][j] = 0; }
            }
        }

        // Populate the map: for each path, count the appearance of each vertex.
        std::vector<unsigned char> seen_in_path(N_SIZE, 0);
        std::vector<int>           touched_vertices;
        for (size_t r = last_path_idx; r < allPaths.size(); ++r) {
            for (const auto &vertex : allPaths[r].route) {
                // Only consider vertices that are not the depot (assumed at
                // indices 0 and N_SIZE-1).
                if (vertex > 0 && vertex < N_SIZE - 1) {
                    ++vertex_route_map[vertex][r];
                    if (!seen_in_path[vertex]) {
                        seen_in_path[vertex] = 1;
                        touched_vertices.push_back(vertex);
                        row_indices_map[vertex].push_back(static_cast<int>(r));
                    }
                }
            }
            for (int vertex : touched_vertices) { seen_in_path[vertex] = 0; }
            touched_vertices.clear();
        }
        last_path_idx = allPaths.size();
    }

    void initializeRank1Memory(const std::vector<std::vector<NodeScore>> &scores) {
        rank1_sep_heur_mem4_vertex.clear();
        rank1_sep_heur_mem4_vertex.resize(N_SIZE);
        for (int i = 1; i < N_SIZE - 1; ++i) {
            auto scored = scores[i];
            std::sort(scored.begin(), scored.end(),
                      [](const NodeScore &a, const NodeScore &b) { return a.cost_score < b.cost_score; });
            const int limit = std::min(MAX_WORKING_SET_SIZE, static_cast<int>(scored.size()));
            for (int j = 0; j < limit; ++j) { rank1_sep_heur_mem4_vertex[i].push_back(scored[j].node); }
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    // Node scoring
    ////////////////////////////////////////////////////////////////////////////

    AdaptiveNodeScorer                  ml_scorer;
    std::vector<std::vector<NodeScore>> computeNodeScores(const SparseMatrix &A, const std::vector<double> &x) {
        auto scores = ml_scorer.computeNodeScores(A, x, distances, nodes, arc_duals, *cutStorage);

        return scores;
    }
    ankerl::unordered_dense::set<CandidateSet, CandidateSetHasher, CandidateSetCompare>
    generateCandidates(const std::vector<std::vector<NodeScore>> &scores, const SparseMatrix &A,
                       const std::vector<double> &x) {
        // Thread-local data for each thread.
        struct ThreadLocalData {
            ankerl::unordered_dense::set<CandidateSet, CandidateSetHasher, CandidateSetCompare> candidates;
            ankerl::unordered_dense::map<CandidateSet, double>                                  candidate_cache;
            ThreadLocalData() {
                candidates.reserve(1000);
                candidate_cache.reserve(1000);
            }
        };

        const size_t                 numThreads = std::thread::hardware_concurrency();
        std::vector<ThreadLocalData> threadData;
        for (size_t t = 0; t < numThreads; ++t) { threadData.emplace_back(); }

        ankerl::unordered_dense::set<CandidateSet, CandidateSetHasher, CandidateSetCompare> finalCandidates;
        finalCandidates.reserve(numThreads * 1000);

        // Use a lambda that accepts an explicit thread index.
        auto process_vertex = [&](std::size_t i, size_t threadID) {
            // Skip first and last vertices.
            if (i == 0 || i == N_SIZE - 1) return;
            auto &localData = threadData[threadID % numThreads];

            // Build a lookup for heuristic memory from scores[i].
            const auto                       &heuristic_memory_scores = scores[i];
            ankerl::unordered_dense::set<int> heuristic_memory_lookup;
            heuristic_memory_lookup.reserve(heuristic_memory_scores.size());
            std::vector<double> node_score_lookup(N_SIZE, std::numeric_limits<double>::infinity());
            for (const auto &score : heuristic_memory_scores) {
                heuristic_memory_lookup.insert(score.node);
                node_score_lookup[score.node] = score.cost_score;
            }

            // Build the working set from routes that contain vertex i.
            ankerl::unordered_dense::set<int> working_set;
            working_set.reserve(std::min<size_t>(MAX_WORKING_SET_SIZE, allPaths.size()));
            if (auto row_it = row_indices_map.find(static_cast<int>(i)); row_it != row_indices_map.end()) {
                for (int r : row_it->second) {
                    for (const auto &v : allPaths[r].route) {
                        if (v > 0 && v < N_SIZE - 1 && heuristic_memory_lookup.contains(v)) working_set.insert(v);
                    }
                }
            }

            // Add additional memory-based candidates for vertices with sparse
            // route support.
            if (working_set.size() < static_cast<size_t>(MIN_RANK - 1) &&
                i < static_cast<size_t>(rank1_sep_heur_mem4_vertex.size())) {
                for (int v : rank1_sep_heur_mem4_vertex[i]) {
                    if (static_cast<int>(working_set.size()) >= MAX_WORKING_SET_SIZE) break;
                    if (v != static_cast<int>(i)) { working_set.insert(v); }
                }
            }

            // Limit working_set size using a heuristic if necessary.
            if (working_set.size() > MAX_WORKING_SET_SIZE) {
                std::vector<int> working_set_vec(working_set.begin(), working_set.end());
                std::nth_element(working_set_vec.begin(), working_set_vec.begin() + MAX_WORKING_SET_SIZE,
                                 working_set_vec.end(),
                                 [&](int a, int b) { return node_score_lookup[a] < node_score_lookup[b]; });
                working_set_vec.resize(MAX_WORKING_SET_SIZE);
                working_set = ankerl::unordered_dense::set<int>(working_set_vec.begin(), working_set_vec.end());
            }

            // Helper lambda to process a candidate set.
            auto process_candidate = [&](const ankerl::unordered_dense::set<int> &candidate_nodes) {
                if (candidate_nodes.size() < MIN_RANK || candidate_nodes.size() > MAX_RANK) return false;
                CandidateSet temp_candidate(candidate_nodes,
                                            0.0,                   // dummy violation
                                            SRCPermutation({}, 0), // dummy permutation
                                            heuristic_memory_lookup, 0.0);
                // Check cache to avoid duplicate work.
                auto           cache_it  = localData.candidate_cache.find(temp_candidate);
                double         violation = 0.0;
                SRCPermutation perm({}, 0);
                double         rhs = 0.0;
                if (cache_it != localData.candidate_cache.end()) {
                    violation = cache_it->second;
                } else {
                    std::tie(violation, perm, rhs) =
                        computeViolationWithBestPerm(candidate_nodes, heuristic_memory_lookup, A, x);
                    localData.candidate_cache.emplace(
                        CandidateSet(candidate_nodes, violation, perm, heuristic_memory_lookup, rhs), violation);
                }
                if (violation > 1e-3) {
                    CandidateSet candidate_set(candidate_nodes, violation, perm, heuristic_memory_lookup, rhs);
                    localData.candidates.emplace(std::move(candidate_set));
                    return true;
                }
                return false;
            };

            // Build the working set vector sorted by heuristic score.
            std::vector<int> working_set_vec(working_set.begin(), working_set.end());
            std::sort(working_set_vec.begin(), working_set_vec.end(),
                      [&](int a, int b) { return node_score_lookup[a] < node_score_lookup[b]; });

            // --- 1. Generate promising root candidates from the best
            // scored working set neighbors.
            const int max_root_size = std::min(static_cast<int>(working_set_vec.size()), MAX_RANK - 1);
            for (int root_size = MIN_RANK - 1; root_size <= max_root_size; ++root_size) {
                ankerl::unordered_dense::set<int> candidate_nodes;
                candidate_nodes.insert(static_cast<int>(i));
                for (int idx = 0; idx < root_size; ++idx) { candidate_nodes.insert(working_set_vec[idx]); }
                process_candidate(candidate_nodes);
            }

            // --- 2. Generate additional candidate sets by combining vertices
            // ---
            if (working_set.size() > 1) {
                size_t combination_count = 0;
                // Make sure the working set is sorted by score.
                // Generate combinations of size k.
                for (size_t k = 2; k <= MAX_RANK && combination_count < MAX_COMBINATIONS; ++k) {
                    if (k > working_set_vec.size()) break;
                    // Initialize combination bitmask: k ones.
                    uint32_t mask = (1u << k) - 1;
                    while (mask < (1u << working_set_vec.size()) && combination_count < MAX_COMBINATIONS) {
                        ankerl::unordered_dense::set<int> candidate_nodes;
                        candidate_nodes.insert(static_cast<int>(i));
                        // Insert vertices corresponding to bits set in mask.
                        for (size_t bit = 0; bit < working_set_vec.size(); ++bit) {
                            if (mask & (1u << bit)) candidate_nodes.insert(working_set_vec[bit]);
                        }
                        if (process_candidate(candidate_nodes)) ++combination_count;

                        // Gosper’s hack to get next combination.
                        uint32_t c = mask & -mask;
                        uint32_t r = mask + c;
                        if (c == 0) break; // safeguard against infinite loop
                        mask = (((r ^ mask) >> 2) / c) | r;
                    }
                }
            }
        };

        // Parallelize over vertices. Here we assign explicit thread indices.
        std::vector<std::thread> threads;
        std::atomic_size_t       vertexIndex{1}; // starting from 1; skipping index 0
        for (size_t t = 0; t < numThreads; ++t) {
            threads.emplace_back([&, t]() {
                while (true) {
                    size_t i = vertexIndex.fetch_add(1);
                    if (i >= N_SIZE - 1) break;
                    process_vertex(i, t);
                }
            });
        }
        for (auto &th : threads) th.join();

        // Merge thread-local candidate sets into finalCandidates.
        for (const auto &data : threadData) { finalCandidates.insert(data.candidates.begin(), data.candidates.end()); }
        return finalCandidates;
    }

    ankerl::unordered_dense::map<int, std::vector<SRCPermutation>> permutations_cache;

    const std::vector<SRCPermutation> &getPermutations(const int RANK) {
        // check if permutations are already computed
        if (permutations_cache.find(RANK) == permutations_cache.end()) {
            std::throw_with_nested(
                std::runtime_error("Permutations for rank " + std::to_string(RANK) + " not found in cache"));
        }
        return permutations_cache[RANK];
        // return perms;
        // if (RANK == 3) {
        //     return getPermutationsForSize3();
        // } else if (RANK == 4) {
        //     return getPermutationsForSize4();
        // } else if (RANK == 5) {
        //     return getPermutationsForSize5();
        // }
        // return {};
    }

public:
    std::vector<std::vector<int>> vertex_route_map;

    std::tuple<double, SRCPermutation, double> computeViolationWithBestPerm(ankerl::unordered_dense::set<int> nodes,
                                                                            ankerl::unordered_dense::set<int> memory,
                                                                            const SparseMatrix               &A,
                                                                            const std::vector<double>        &x) {
        static constexpr double EPSILON = 1e-6;
        // Convert nodes set to a sorted vector to allow indexed access.
        // std::vector<int> node_vec(nodes.begin(), nodes.end());
        // std::sort(node_vec.begin(), node_vec.end());
        const int RANK = static_cast<int>(nodes.size());

        // Initialize result variables.
        double                best_violation = 0.0;
        double                best_rhs       = 0.0;
        const SRCPermutation *best_perm      = nullptr;

        // Initialize bit arrays for efficient set operations.
        std::array<uint64_t, num_words> candidate_bits{};        // all bits initially 0
        std::array<uint64_t, num_words> augmented_memory_bits{}; // all bits initially 0

        // Create a node order lookup vector; initialize with -1 (invalid
        // index).
        std::vector<int> node_order(N_SIZE, -1);

        // Populate candidate_bits and node_order based on candidate nodes.
        int i = 0;
        for (auto node : nodes) {
            const int      word_idx = node / 64;
            const uint64_t bit_mask = bit_mask_lookup[node % 64];

            candidate_bits[word_idx] |= bit_mask;
            augmented_memory_bits[word_idx] |= bit_mask;
            node_order[node] = i++;
        }

        // Add memory nodes to augmented_memory_bits.
        for (const int node : memory) {
            int word_idx = node / 64;
            augmented_memory_bits[word_idx] |= (1ULL << (node % 64));
        }

        // Get all permutations for the given rank (compute once).
        const auto &permutations = getPermutations(RANK);

        // Evaluate each permutation.
        for (const auto &perm : permutations) {
            // Compute the RHS value from the permutation.
            double rhs = perm.getRHS();
            // Create an SRCPermutation from the current permutation.
            const SRCPermutation src_perm{perm.num, perm.den};

            // Reset coefficient vector and accumulator for LHS.
            // cut_coefficients.clear();
            double lhs = 0.0;

            // Process each path from the global allPaths container.
            for (int path_idx : nonzero_paths) {
                const double x_val = x[path_idx];

                // Compute the coefficient using candidate_bits,
                // augmented_memory_bits, current permutation, the route from
                // the path, and the node ordering.
                const int coeff = computeLimitedMemoryCoefficient(candidate_bits, augmented_memory_bits, src_perm,
                                                                  allPaths[path_idx].route, node_order);
                // cut_coefficients.push_back(coeff);
                lhs += coeff * x_val;
            }

            // Compute the violation: the difference between lhs and (rhs +
            // EPSILON).
            const double violation = lhs - (rhs + EPSILON);
            if (violation > best_violation) {
                best_violation = violation;
                best_perm      = &perm;
                best_rhs       = rhs;
            }
        }

        // Return the best violation, corresponding permutation (or default if
        // none), and best rhs.
        return {best_violation, best_perm ? *best_perm : SRCPermutation({}, 0), best_rhs};
    }

    Xoroshiro128Plus rng;

    void addCutToCutStorage(const CandidateSet &candidate, std::vector<int> &order) {
        // Initialize bit arrays for the candidate (C) and augmented memory
        // (AM).
        std::array<uint64_t, num_words> C  = {};
        std::array<uint64_t, num_words> AM = {};

        // Set bits for candidate nodes.
        for (auto node : candidate.nodes) {
            C[node / 64] |= bit_mask_lookup[node % 64];
            AM[node / 64] |= bit_mask_lookup[node % 64];
        }
        // Set bits for candidate neighbors into the augmented memory.
        for (auto node : candidate.neighbor) { AM[node / 64] |= bit_mask_lookup[node % 64]; }

        // Prepare the permutation from the candidate.
        SRCPermutation p;
        p.num = candidate.perm.num;
        p.den = candidate.perm.den;

        // Store only the nonzero coefficients. This keeps high-order cuts
        // lighter in memory while still allowing us to build the initial
        // master constraint exactly.
        std::vector<int>    coefficient_indices;
        std::vector<double> coefficient_values;
        coefficient_indices.reserve(allPaths.size() / 8 + 1);
        coefficient_values.reserve(allPaths.size() / 8 + 1);
        for (size_t i = 0; i < allPaths.size(); ++i) {
            const double coeff = computeLimitedMemoryCoefficient(C, AM, p, allPaths[i].route, order);
            if (!numericutils::isZero(coeff)) {
                coefficient_indices.push_back(static_cast<int>(i));
                coefficient_values.push_back(coeff);
            }
        }

        // Construct the Cut object using the candidate data.
        Cut cut(C, AM, {}, p);
        cut.baseSetOrder        = order;
        cut.coefficient_indices = std::move(coefficient_indices);
        cut.coefficient_values  = std::move(coefficient_values);

        // Set the cut type based on the number of candidate nodes.
        const auto cut_rank = candidate.nodes.size();
        if (cut_rank == 4) {
            cut.type = CutType::FourRow;
            // return;
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
    HighRankCuts() {
        for (int i = MIN_RANK; i <= MAX_RANK; ++i) {
            fmt::print("Generating permutations for rank {}\n", i);
            permutations_cache[i] = generateGeneticPermutations(i);
        }
    }

    CutStorage *cutStorage = nullptr;

    std::vector<Path> allPaths;

    ArcDuals arc_duals;

    std::vector<VRPNode>                                nodes;
    std::vector<std::vector<double>>                    distances;
    ankerl::unordered_dense::map<int, std::vector<int>> row_indices_map;
    std::vector<int>                                    nonzero_paths;

    std::vector<double> solution;

    void separate(const SparseMatrix &A, const std::vector<double> &x) {
        // Set the solution.
        solution = x;
        initializeVertexRouteMap();
        nonzero_paths.clear();
        nonzero_paths.reserve(x.size());
        for (size_t path_idx = 0; path_idx < x.size(); ++path_idx) {
            if (!numericutils::isZero(x[path_idx])) { nonzero_paths.push_back(static_cast<int>(path_idx)); }
        }

        // Compute node scores.
        auto node_scores = computeNodeScores(A, x);
        initializeRank1Memory(node_scores);

        // Generate candidate sets.
        // (Assuming generateCandidates returns a container type that is not
        // random access.)
        auto                      candidates_set = generateCandidates(node_scores, A, x);
        std::vector<CandidateSet> candidates(candidates_set.begin(), candidates_set.end());

        // Prepare for parallel local search.
        const size_t num_candidates = candidates.size();
        std::mutex   candidates_mutex;
        ankerl::unordered_dense::set<CandidateSet, CandidateSetHasher, CandidateSetCompare> improved_candidates;

        // Parallelize the local search using stdexec::bulk.
        auto bulk_sender = stdexec::bulk(
            stdexec::just(), num_candidates,
            [this, &A, &x, &node_scores, &improved_candidates, &candidates, &candidates_mutex](std::size_t idx) {
                // Each thread creates its own LocalSearch object.
                LocalSearch local_search(this);

                // Retrieve candidate via index.
                const CandidateSet &current_candidate = candidates[idx];
                const auto          current_violation = current_candidate.violation;

                // Perform local search to get potential improvements.
                auto improved_list = local_search.solve(current_candidate, A, x, node_scores);

                // Collect improved candidates locally to reduce locking
                // frequency.
                std::vector<CandidateSet> local_improved;
                for (const auto &candidate : improved_list) {
                    if (candidate.violation > current_violation) { local_improved.push_back(candidate); }
                }
                if (local_improved.empty()) { local_improved.push_back(current_candidate); }

                // Insert all local improved candidates into the global set with
                // one lock.
                {
                    std::lock_guard<std::mutex> lock(candidates_mutex);
                    for (const auto &candidate : local_improved) { improved_candidates.insert(candidate); }
                }
            });

        // Execute the bulk sender on the scheduler and wait for completion.
        auto work = stdexec::starts_on(sched, std::move(bulk_sender));
        stdexec::sync_wait(std::move(work));

        // Provide feedback using the improved candidates.
        ml_scorer.provideFeedback(improved_candidates);

        // Process and sort unique candidates based on violation.
        int                       initial_cut_size = cutStorage->size();
        std::vector<CandidateSet> unique_candidates(improved_candidates.begin(), improved_candidates.end());
        pdqsort(unique_candidates.begin(), unique_candidates.end(),
                [](const CandidateSet &a, const CandidateSet &b) { return a.violation > b.violation; });

        // Add cuts from top candidates. Limit the total added to max_cuts (and
        // try up to max_trials).
        const int max_cuts   = 10;
        int       cuts_added = 0;
        int       max_trials = 50;
        for (const auto &candidate : unique_candidates) {
            if (cuts_added >= max_cuts || max_trials <= 0) break;

            // Build the vertex ordering for the candidate.
            std::vector<int> order(N_SIZE);
            int              ordering = 0;
            for (int node : candidate.nodes) { order[node] = ordering++; }
            addCutToCutStorage(candidate, order);
            ++cuts_added;
            // Optionally decrement max_trials if further limiting is desired.
            --max_trials;
        }
        int final_cut_size = cutStorage->size();

        // Print summary of the candidate processing.
        print_cut("Candidates: {} | Improved Candidates: {} | Added {} SRC 3-4-5 "
                  "cuts\n",
                  candidates.size(), improved_candidates.size(), final_cut_size - initial_cut_size);
    }

    static constexpr std::array<uint64_t, 64> bit_mask_lookup = []() {
        std::array<uint64_t, 64> masks{};
        for (size_t i = 0; i < 64; ++i) { masks[i] = 1ULL << i; }
        return masks;
    }();

    int computeLimitedMemoryCoefficient(const std::array<uint64_t, num_words> &C,
                                        const std::array<uint64_t, num_words> &AM, const SRCPermutation &p,
                                        const std::vector<uint16_t> &P, std::vector<int> &order) noexcept {
        int       alpha          = 0.0;
        int       cumulative_sum = 0;
        const int denominator    = p.den;

        // Loop over the positions in P, ignoring the first and last elements.
        for (size_t j = 1; j < P.size() - 1; ++j) {
            const int vertex = P[j];

            // Determine the bit mask and word index for the vertex.
            const size_t   word_index = vertex >> 6;
            const uint64_t bit_mask   = 1ULL << (vertex & 63);

            // If the augmented memory does not contain the vertex, reset
            // cumulative sum.
            if (!(AM[word_index] & bit_mask)) {
                cumulative_sum = 0;
            }
            // Otherwise, if the candidate set contains the vertex,
            // add the corresponding numerator from the permutation.
            else if (C[word_index] & bit_mask) {
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
