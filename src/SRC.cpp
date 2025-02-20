/**
 * @file SRC.cpp
 * @brief Implementation of functions and classes for solving VRP problems using
 * Limited Memory Rank-1 Cuts.
 *
 * This file contains the implementation of various functions and classes used
 * for solving Vehicle Routing Problems (VRP) using Limited Memory Rank-1 Cuts.
 * The main functionalities include computing unique cut keys, adding cuts to
 * storage, separating solution vectors into cuts, and generating cut
 * coefficients.
 *
 * The implementation leverages parallel processing using thread pools and
 * schedulers to efficiently handle large datasets and complex computations.
 *
 */

#include "cuts/SRC.h"

#include "Cut.h"
#include "Definitions.h"
#include "HeuristicHighOrder.h"
#include "bnb/Node.h"
#ifdef IPM
#include "ipm/IPSolver.h"
#endif

#ifdef HIGHS
#include <Highs.h>
#endif
using Cuts = std::vector<Cut>;

#include "CutIntelligence.h"

/**
 * @brief Computes a unique cut key based on the provided base set and
 * multipliers.
 *
 * This function generates a hash value that uniquely identifies the combination
 * of the base set and multipliers. The base set is an array of uint64_t values,
 * and the multipliers are a vector of double values. The order of elements in
 * both the base set and multipliers is preserved during hashing.
 *
 */
std::size_t compute_cut_key(const std::array<uint64_t, num_words> &baseSet,
                            const std::vector<int> &perm_num,
                            const int perm_den) {
    XXH3_state_t *state = XXH3_createState();
    assert(state != nullptr);
    XXH3_64bits_reset(state);

    // Hash baseSet (array of uint64_t)
    XXH3_64bits_update(state, baseSet.data(),
                       baseSet.size() * sizeof(uint64_t));
    // Hash perm_num
    XXH3_64bits_update(state, perm_num.data(), perm_num.size() * sizeof(int));
    // Hash perm_den
    XXH3_64bits_update(state, &perm_den, sizeof(int));

    std::size_t cut_key = XXH3_64bits_digest(state);
    XXH3_freeState(state);
    return cut_key;
}
/**
 * @brief Adds a cut to the CutStorage.
 *
 * This function adds a given cut to the CutStorage. It first computes a unique
 * key for the cut based on its base set and multipliers. If a cut with the same
 * key already exists in the storage, it updates the existing cut. Otherwise, it
 * adds the new cut to the storage and updates the necessary mappings.
 *
 */
void CutStorage::addCut(Cut &cut) {
    // Compute a unique key for the cut based on its base set and probability
    // data.
    cut.key = compute_cut_key(cut.baseSet, cut.p.num, cut.p.den);

    // Check if the cut already exists in our map.
    if (auto it = cutMaster_to_cut_map.find(cut.key);
        it != cutMaster_to_cut_map.end()) {
        // The cut already exists; update its id and merge neighbor information.
        cut.id = it->second;
        const auto &old_neighbors = cuts[cut.id].neighbors;
        for (size_t i = 0; i < num_words; ++i) {
            cut.neighbors[i] |= old_neighbors[i];
        }
        // Preserve 'added' flag if the existing cut has been marked as added.
        if (cuts[cut.id].added) {
            cut.added = true;
            cut.updated = true;
        }
        cuts[cut.id] = cut;
    } else {
        // The cut is new; assign a new id and store it.
        cut.id = cuts.size();
        cuts.push_back(cut);
        cutMaster_to_cut_map[cut.key] = cut.id;
    }

    // Safely update indexCuts with the cut's id.
    indexCuts[cut.key].push_back(cut.id);
}

LimitedMemoryRank1Cuts::LimitedMemoryRank1Cuts(std::vector<VRPNode> &nodes)
    : nodes(nodes) {}

/**
 * @brief Separates the given solution vector into cuts using Limited Memory
 * Rank-1 Cuts.
 *
 * This function generates cuts for the given solution vector `x` based on the
 * sparse model `A`. It uses a parallel approach to evaluate combinations of
 * indices and identify violations that exceed the specified threshold.
 *
 */
void LimitedMemoryRank1Cuts::separate(const SparseMatrix &A,
                                      const std::vector<double> &x) {
    // Temporary storage for computed cuts.
    std::vector<std::pair<double, Cut>> tmp_cuts;

    // Set up parallelization parameters.
    const int JOBS = std::thread::hardware_concurrency();
    auto nC = N_SIZE;
    const int chunk_size = tasks.size() / JOBS;
#ifdef NSYNC
    nsync::nsync_mu cuts_mutex = NSYNC_MU_INIT;
#else
    std::mutex cuts_mutex;
#endif

    // Initialize SRCPermutation 'p' and compute a right-hand-side value.
    SRCPermutation p;
    p.num = {1, 1, 1};
    p.den = 2;
    double rhs = 0.0;
    for (size_t i = 0; i < 3; ++i) {
        rhs += static_cast<double>(p.num[i]) / p.den;
    }
    rhs = std::floor(rhs);

    // === Parallel Processing of Tasks in Chunks ===
    auto bulk_sender = stdexec::bulk(
        stdexec::just(), (tasks.size() + chunk_size - 1) / chunk_size,
        [this, &A, &x, nC, &cuts_mutex, chunk_size, &p, &rhs,
         &tmp_cuts](std::size_t chunk_idx) {
            size_t start_idx = chunk_idx * chunk_size;
            size_t end_idx = std::min(start_idx + chunk_size, tasks.size());

            // Process each task in this chunk.
            for (size_t task_idx = start_idx; task_idx < end_idx; ++task_idx) {
                // Unpack task parameters (i, j, k).
                const auto &[i, j, k] = tasks[task_idx];

                // Initialize a vector to count occurrences for each column.
                std::vector<int> expanded(A.num_cols, 0);
                std::vector<int> buffer_int(A.num_cols);
                int buffer_int_n = 0;
                double lhs = 0.0;

                // Accumulate counts from rows corresponding to i, j, and k.
                for (int col : row_indices_map[i]) expanded[col] += 1;
                for (int col : row_indices_map[j]) expanded[col] += 1;
                for (int col : row_indices_map[k]) expanded[col] += 1;

                // Compute lhs value based on columns with at least 2 counts.
                for (int idx = 0; idx < A.num_cols; ++idx) {
                    if (expanded[idx] >= 2) {
                        lhs += std::floor(expanded[idx] * 0.5) * x[idx];
                        buffer_int[buffer_int_n++] = idx;
                    }
                }

                // Only proceed if lhs exceeds threshold.
                if (lhs > 1.0 + 1e-3) {
                    // Initialize bitmask arrays for cut representation.
                    std::array<uint64_t, num_words> C = {};
                    std::array<uint64_t, num_words> AM = {};
                    std::vector<int> order(N_SIZE, 0);
                    int ordering = 0;

                    // Build the base cut indices from the task parameters.
                    std::vector<int> C_index = {i, j, k};
                    for (int node : C_index) {
                        C[node / 64] |= (1ULL << (node % 64));
                        AM[node / 64] |= (1ULL << (node % 64));
                        order[node] = ordering++;
                    }
                    // Get the remaining nodes from buffer_int.
                    std::vector<int> remainingNodes(
                        buffer_int.begin(), buffer_int.begin() + buffer_int_n);

                    // Initialize cut coefficients for all paths.
                    std::vector<double> cut_coefficients(allPaths.size(), 0.0);

                    // Build a set from the base cut indices.
                    ankerl::unordered_dense::set<int> C_set(C_index.begin(),
                                                            C_index.end());

                    // Update AM based on the positions of nodes in each
                    // consumer path.
                    for (int node : remainingNodes) {
                        auto &consumers = allPaths[node];  // Access consumers.
                        int first = -1, second = -1;
                        // Identify first and second occurrences of elements
                        // from C_set.
                        for (size_t pos = 1; pos < consumers.size() - 1;
                             ++pos) {
                            if (C_set.contains(consumers[pos])) {
                                if (first == -1) {
                                    first = pos;
                                } else {
                                    second = pos;
                                    for (int pos_inner = first + 1;
                                         pos_inner < second; ++pos_inner) {
                                        AM[consumers[pos_inner] / 64] |=
                                            (1ULL
                                             << (consumers[pos_inner] % 64));
                                    }
                                    break;
                                }
                            }
                        }
                    }

                    // Compute the cut coefficients and accumulate lhs.
                    double computed_lhs = 0.0;
                    for (size_t z = 0; z < allPaths.size(); ++z) {
                        auto &clients = allPaths[z].route;
                        cut_coefficients[z] = computeLimitedMemoryCoefficient(
                            C, AM, p, clients, order);
                        computed_lhs += cut_coefficients[z] * x[z];
                    }
                    double violation = computed_lhs - (rhs + 1e-3);

                    // If violation is positive, record the cut.
                    if (violation > 0.0) {
                        Cut cut(C, AM, cut_coefficients, p);
                        cut.baseSetOrder = order;
#ifdef NSYNC
                        nsync::nsync_mu_lock(&cuts_mutex);
#else
                        std::lock_guard<std::mutex> lock(cuts_mutex);
#endif
                        tmp_cuts.emplace_back(computed_lhs, cut);
#ifdef NSYNC
                        nsync::nsync_mu_unlock(&cuts_mutex);
#endif
                    }
                }
            }
        });

    // Submit the bulk work and wait for all tasks to complete.
    auto work = stdexec::starts_on(sched, bulk_sender);
    stdexec::sync_wait(std::move(work));

    // === Post-Processing: Sort and Add Cuts ===
    pdqsort(tmp_cuts.begin(), tmp_cuts.end(),
            [](const auto &a, const auto &b) { return a.first > b.first; });

    // Determine limits for adding cuts.
    int max_cuts = std::min(static_cast<int>(tmp_cuts.size()), 2);
    int max_trials = 15;
    int cuts_added = 0;
    int cuts_orig_size = cutStorage.size();

    // Add cuts from tmp_cuts, up to limits.
    for (int i = 0; i < static_cast<int>(tmp_cuts.size()); ++i) {
        if (cuts_added >= max_cuts || max_trials <= 0) break;
        auto &cut = tmp_cuts[i].second;
        cutStorage.addCut(cut);
        cuts_added++;  // = cutStorage.size() - cuts_orig_size;
    }
}

std::pair<bool, bool> LimitedMemoryRank1Cuts::runSeparation(
    BNBNode *node, std::vector<baldesCtrPtr> &SRCconstraints) {
    // Pointer to our cut storage
    auto *cuts = &cutStorage;

    // Extract the sparse model data from the node
    ModelData matrix = node->extractModelDataSparse();

    // Record the number of cuts before separation
    const size_t cuts_before = cuts->size();

    // Obtain the solution from the node (macro or function)
    std::vector<double> solution;
    GET_SOL(node);

    // === Build a Map of Row Indices from allPaths ===
    // Maps each valid node (row) to all path indices where it appears.
    // size_t path_idx = 0;
    // for (auto path_idx = last_path_idx; path_idx < allPaths.size();
    //      ++path_idx) {
    //     auto &path = allPaths[path_idx];
    //     for (int node : path.route) {
    //         // Only consider nodes in the range [1, N_SIZE - 2].
    //         if (node < 1 || node > N_SIZE - 2) continue;
    //         row_indices_map[node].push_back(static_cast<int>(path_idx));
    //     }
    // }
    // last_path_idx = allPaths.size();

    // Perform separation routines for Rank-1 cuts.
    // separateR1C1(matrix.A_sparse, solution);
    // separate(matrix.A_sparse, solution);

    // Record the cut count after the first separation phase.
    const size_t initial_cut_count = cuts->size();
    int rank3_cuts_size = static_cast<int>(cuts->size() - initial_cut_count);

    ////////////////////////////////////////////////////
    // High-Rank Cuts Separation
    ////////////////////////////////////////////////////
    high_rank_cuts.cutStorage = &cutStorage;
    high_rank_cuts.allPaths = allPaths;
    high_rank_cuts.nodes = nodes;
    high_rank_cuts.arc_duals = arc_duals;
    high_rank_cuts.separate(matrix.A_sparse, solution);

    const size_t cuts_after_separation = cuts->size();

    ////////////////////////////////////////////////////
    // Optionally, clean non-violated cuts (currently disabled)
    ////////////////////////////////////////////////////
    // bool cleared = cutCleaner(node, SRCconstraints);
    bool cleared = false;

    // Calculate how many cuts were removed (if any)
    const size_t n_cuts_removed = cuts_after_separation - cuts->size();

    // Determine if any cuts have changed compared to before the separation.
    bool cuts_changed = (cuts_before != cuts->size());
    return std::make_pair(cuts_changed, cleared);
}

bool LimitedMemoryRank1Cuts::cutCleaner(
    BNBNode *node, std::vector<baldesCtrPtr> &SRCconstraints) {
    // Pointer to the cut storage.
    auto *cuts = &cutStorage;
    std::vector<double> solution;
    bool cleaned = false;

    // if (cuts->busy) return false;

    GET_SOL(node);  // Retrieves the solution into 'solution'.

    // Traverse SRCconstraints in reverse.
    for (auto it = SRCconstraints.rbegin(); it != SRCconstraints.rend();) {
        auto constr = *it;
        int current_index = constr->index();
        double slack = node->getSlack(current_index, solution);

        // If the slack is positive (non-violated), remove the constraint.
        if (std::abs(slack) > 0) {
            cleaned = true;
            node->remove(constr);

            // Convert the reverse iterator to a normal iterator.
            // it.base() points to the element *after* the one we want to erase.
            auto normal_it = it.base();
            --normal_it;  // Now 'normal_it' points to the element to be erased.
            // Compute the index of this element.
            int index = std::distance(SRCconstraints.begin(), normal_it);
            cuts->removeCut(cuts->getID(index));

            // Erase the element and update the reverse iterator.
            it = std::make_reverse_iterator(SRCconstraints.erase(normal_it));
        } else {
            ++it;
        }
    }
    return cleaned;
}

/*
 * @brief Computes the limited memory coefficient for a given set of nodes.
 *
 */
double LimitedMemoryRank1Cuts::computeLimitedMemoryCoefficient(
    const std::array<uint64_t, num_words> &C,
    const std::array<uint64_t, num_words> &AM, const SRCPermutation &p,
    const std::vector<uint16_t> &P, std::vector<int> &order) noexcept {
    double alpha = 0.0;
    int S = 0;
    auto den = p.den;

    for (size_t j = 1; j < P.size() - 1; ++j) {
        int vj = P[j];

        // Precompute bitshift values for reuse
        uint64_t am_mask = (1ULL << (vj & 63));
        uint64_t am_index = vj >> 6;

        // Check if vj is in AM using precomputed values
        if (!(AM[am_index] & am_mask)) {
            S = 0;  // Reset S if vj is not in AM
        } else if (C[am_index] & am_mask) {
            // Get the position of vj in C by counting the set bits up to vj
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
std::vector<CandidateSet> LocalSearch::solve(
    const CandidateSet &initial, const SparseMatrix &A,
    const std::vector<double> &x,
    const std::vector<std::vector<int>> &node_scores, int max_iterations) {
    // Initialize best and current candidate sets.
    CandidateSet best = initial;
    CandidateSet current = initial;
    std::vector<CandidateSet> diverse_solutions;
    diverse_solutions.reserve(LocalSearchConfig::MAX_DIVERSE_SOLUTIONS + 1);

    // Lambda to determine if a candidate solution is "diverse" enough.
    auto isSolutionDiverse = [&](const CandidateSet &candidate) -> bool {
        for (const auto &sol : diverse_solutions) {
            const double similarity =
                computeSimilarity(candidate.nodes, sol.nodes);
            if (similarity > 0.7 ||
                std::abs(sol.violation - candidate.violation) <
                    LocalSearchConfig::DIVERSITY_THRESHOLD) {
                return false;
            }
        }
        return true;
    };

    // Lambda to update the list of diverse solutions.
    auto updateDiverseSolutions = [&](const CandidateSet &candidate) {
        if (isSolutionDiverse(candidate)) {
            diverse_solutions.push_back(candidate);
            if (diverse_solutions.size() >
                LocalSearchConfig::MAX_DIVERSE_SOLUTIONS) {
                auto min_it = std::min_element(
                    diverse_solutions.begin(), diverse_solutions.end(),
                    [](const CandidateSet &a, const CandidateSet &b) {
                        return a.violation < b.violation;
                    });
                diverse_solutions.erase(min_it);
            }
        }
    };

    // Lambda to update segment statistics.
    auto updateSegmentStatistics = [&]() {
        segment_iterations++;
        if (segment_iterations == LocalSearchConfig::SEGMENT_SIZE) {
            history.push_back(current_segment);
            if (history.size() > 3) {
                history.pop_front();
            }
            current_segment = SegmentStats{};
            segment_iterations = 0;
        }
    };

    // Main iterative search loop.
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Backup the current solution.
        CandidateSet backup = current;
        const OperatorType selected_op = selectOperator();

        // Apply the selected operator.
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

        // Evaluate the new candidate using parent's violation computation.
        auto [new_violation, new_perm, rhs] =
            parent->computeViolationWithBestPerm(current.nodes,
                                                 current.neighbor, A, x);

        const double delta = new_violation - backup.violation;

        // Accept or reject the move.
        if (acceptMove(delta, current)) {
            // Update the candidate's statistics.
            current.violation = new_violation;
            current.perm = std::move(new_perm);
            current.rhs = std::move(rhs);

            updateStatistics(selected_op, delta, current, new_violation);

            // If improvement, update best and reset counter.
            if (new_violation > best.violation) {
                best = current;
                current_segment.improvements++;
                iterations_since_improvement = 0;
                current_segment.best_violation =
                    std::max(current_segment.best_violation, new_violation);
            } else {
                iterations_since_improvement++;
            }

            updateDiverseSolutions(current);
        } else {
            // Revert move and adjust operator score.
            current = backup;
            operators[static_cast<size_t>(selected_op)].score =
                std::max(LocalSearchConfig::MIN_WEIGHT,
                         operators[static_cast<size_t>(selected_op)].score *
                             (1 - LocalSearchConfig::OPERATOR_LEARNING_RATE));
        }

        updateSegmentStatistics();
        updateTemperature();
        strategicRestart(current, best, diverse_solutions);
    }

    // Final processing: add the best solution and sort the diverse set.
    diverse_solutions.push_back(std::move(best));
    pdqsort(diverse_solutions.begin(), diverse_solutions.end(),
            [](const CandidateSet &a, const CandidateSet &b) {
                return a.violation > b.violation;
            });

    return diverse_solutions;
}
