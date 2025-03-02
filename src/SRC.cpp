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
    // Determine parallel parameters.
    const int JOBS = std::thread::hardware_concurrency();
    auto nC = N_SIZE;
    const int chunk_size = (tasks.size() + JOBS - 1) / JOBS;
    const size_t num_chunks = (tasks.size() + chunk_size - 1) / chunk_size;

    // Reserve per-chunk storage to avoid locking.
    std::vector<std::vector<std::pair<double, Cut>>> chunk_cuts(num_chunks);
    for (auto &vec : chunk_cuts) {
        vec.reserve(20);  // Reserve an estimate; adjust as needed.
    }

    // Initialize SRCPermutation 'p' and compute a right-hand-side value.
    SRCPermutation p;
    p.num = {1, 1, 1};
    p.den = 2;
    double rhs = p.getRHS();

    // === Parallel Processing of Tasks in Chunks ===
    auto bulk_sender = stdexec::bulk(
        stdexec::just(), num_chunks,
        [this, &A, &x, nC, chunk_size, &p, rhs,
         &chunk_cuts](std::size_t chunk_idx) {
            // Each chunk gets its own temporary vector.
            std::vector<std::pair<double, Cut>> local_cuts;
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
                // Compute lhs value based on columns with at least 2 counts.
                for (int idx = 0; idx < A.num_cols; ++idx) {
                    expanded[idx] = std::min(1, vertex_route_map[i][idx]) +
                                    std::min(1, vertex_route_map[j][idx]) +
                                    std::min(1, vertex_route_map[k][idx]);
                    if (expanded[idx] >= 2) {
                        lhs += std::floor(expanded[idx] * 0.5) * x[idx];
                        buffer_int[buffer_int_n++] = idx;
                    }
                }

                // Only proceed if lhs exceeds threshold.
                if (lhs > rhs + 1e-3) {
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
                    for (auto z : remainingNodes) {
                        auto &clients = allPaths[z].route;
                        // Instead of calling a potentially expensive function,
                        // here we simply assign a coefficient of 1.0 (you can
                        // plug in your compute function).
                        cut_coefficients[z] = computeLimitedMemoryCoefficient(
                            C, AM, p, clients, order);
                        computed_lhs += cut_coefficients[z] * x[z];
                    }

                    // If violation is positive, record the cut.
                    if (numericutils::gte(computed_lhs, rhs)) {
                        Cut cut(C, AM, cut_coefficients, p);
                        cut.baseSetOrder = order;
                        local_cuts.emplace_back(computed_lhs, cut);
                    }
                }
            }
            // Save the local cuts into the global vector for this chunk.
            chunk_cuts[chunk_idx] = std::move(local_cuts);
        });

    // Submit the bulk work and wait for all tasks to complete.
    auto work = stdexec::starts_on(sched, bulk_sender);
    stdexec::sync_wait(std::move(work));

    // === Merge results from all chunks ===
    std::vector<std::pair<double, Cut>> tmp_cuts;
    for (const auto &vec : chunk_cuts) {
        tmp_cuts.insert(tmp_cuts.end(), vec.begin(), vec.end());
    }

    // === Post-Processing: Sort and Add Cuts ===
    pdqsort(tmp_cuts.begin(), tmp_cuts.end(),
            [](const auto &a, const auto &b) { return a.first > b.first; });

    // Determine limits for adding cuts.
    int max_cuts = std::min(static_cast<int>(tmp_cuts.size()), 3);
    int max_trials = 15;
    int cuts_added = 0;
    int cuts_orig_size = cutStorage.size();

    // Add cuts from tmp_cuts, up to limits.
    for (int i = 0; i < static_cast<int>(tmp_cuts.size()); ++i) {
        if (cuts_added >= max_cuts || max_trials <= 0) break;
        auto &cut = tmp_cuts[i].second;
        cutStorage.addCut(cut);
        ++cuts_added;
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

    fmt::print("Starting separation phase...\n");
    initializeVertexRouteMap();

    // Perform separation routines for Rank-1 cuts.
    fmt::print("Separating Rank-1 cuts...\n");
    separateR1C1(matrix.A_sparse, solution);
    fmt::print("Separation Rank-3 cuts...\n");
    separate(matrix.A_sparse, solution);

    // Record the cut count after the first separation phase.
    const size_t initial_cut_count = cuts->size();
    int rank3_cuts_size = static_cast<int>(cuts->size() - initial_cut_count);

    ////////////////////////////////////////////////////
    // High-Rank Cuts Separation
    ////////////////////////////////////////////////////
    high_rank_cuts.cutStorage = &cutStorage;
    high_rank_cuts.vertex_route_map = vertex_route_map;
    high_rank_cuts.allPaths = allPaths;
    high_rank_cuts.nodes = nodes;
    high_rank_cuts.arc_duals = arc_duals;
    fmt::print("Separating High-Rank cuts...\n");
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
        if (numericutils::gt(slack, 0.0)) {
            cleaned = true;
            node->remove(constr);

            // Convert the reverse iterator to a normal iterator.
            // it.base() points to the element *after* the one we want to
            // erase.
            auto normal_it = it.base();
            --normal_it;  // Now 'normal_it' points to the element to be
                          // erased.
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
        const int vj = P[j];

        // Precompute bitshift values for reuse
        const size_t word = vj >> 6;
        const uint64_t bit_mask = 1ULL << (vj & 63);

        // Check if vj is in AM using precomputed values
        if (!(AM[word] & bit_mask)) {
            S = 0;  // Reset S if vj is not in AM
        } else if (C[word] & bit_mask) {
            // Get the position of vj in C by counting the set bits up to vj
            const int pos = order[vj];
            S += p.num[pos];
            if (S >= den) {
                S -= den;
                alpha += 1.0;
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

    // Lambda: Check if a candidate solution is "diverse" enough.
    auto isSolutionDiverse = [&](const CandidateSet &candidate) -> bool {
        // Convert candidate nodes into a sorted vector.
        std::vector<int> cand_nodes(candidate.nodes.begin(),
                                    candidate.nodes.end());
        std::sort(cand_nodes.begin(), cand_nodes.end());
        for (const auto &sol : diverse_solutions) {
            std::vector<int> sol_nodes(sol.nodes.begin(), sol.nodes.end());
            std::sort(sol_nodes.begin(), sol_nodes.end());
            double similarity = computeSimilarity(cand_nodes, sol_nodes);
            if (similarity > 0.7 ||
                std::abs(sol.violation - candidate.violation) <
                    LocalSearchConfig::DIVERSITY_THRESHOLD) {
                return false;
            }
        }
        return true;
    };

    // Lambda: Update the list of diverse solutions.
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

    // Lambda: Update segment statistics after each segment.
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
        OperatorType selected_op = selectOperator();

        // Apply the selected operator.
        switch (selected_op) {
            case OperatorType::REMOVE_NODE:
                applyRemoveNode(current);
                break;
            case OperatorType::ADD_NODE:
                applyAddNode(current, node_scores);
                break;
            case OperatorType::REMOVE_NEIGHBORS:
                applyRemoveNeighbors(current,
                                     LocalSearchConfig::MAX_REMOVE_COUNT);
                break;
            case OperatorType::ADD_NEIGHBORS:
                applyAddNeighbors(current, node_scores,
                                  LocalSearchConfig::MAX_ADD_COUNT);
                break;
            default:
                // Optionally fallback or combine neighbor updates.
                applyUpdateNeighbors(current, node_scores);
                break;
        }

        // Evaluate the new candidate using parent's violation computation.
        auto [new_violation, new_perm, rhs] =
            parent->computeViolationWithBestPerm(current.nodes,
                                                 current.neighbor, A, x);
        double delta = new_violation - backup.violation;

        // Accept or reject the move.
        if (acceptMove(delta, current)) {
            // Accept move: update candidate solution.
            current.violation = new_violation;
            current.perm = std::move(new_perm);
            current.rhs = std::move(rhs);
            updateStatistics(selected_op, delta, current, new_violation);
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
