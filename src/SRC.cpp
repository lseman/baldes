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
template <typename T>
std::size_t compute_cut_key(const std::array<uint64_t, num_words> &baseSet,
                            const std::vector<T> &multipliers) {
    XXH3_state_t *state = XXH3_createState();  // Initialize the XXH3 state
    assert(state != nullptr);  // Ensure state creation succeeded
    XXH3_64bits_reset(state);  // Reset the hashing state

    // Hash the baseSet (uint64_t values) while preserving order
    for (std::size_t i = 0; i < baseSet.size(); ++i) {
        XXH3_64bits_update(state, &baseSet[i], sizeof(uint64_t));
    }

    // Hash the multipliers (numeric values) while preserving order
    for (std::size_t i = 0; i < multipliers.size(); ++i) {
        XXH3_64bits_update(state, &multipliers[i], sizeof(T));
    }

    // Finalize the hash
    std::size_t cut_key = XXH3_64bits_digest(state);

    XXH3_freeState(state);  // Free the state

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
    auto baseSet = cut.baseSet;
    auto p_num = cut.p.num;
    auto p_den = cut.p.den;

    // Concatenate p_num and p_den into a single vector
    std::vector<int> p_cat;
    p_cat.reserve(p_num.size() + 1);  // Pre-allocate memory
    p_cat.insert(p_cat.end(), p_num.begin(), p_num.end());
    p_cat.push_back(p_den);

    // Compute the unique cut key
    std::size_t cut_key = compute_cut_key(baseSet, p_cat);

    // Check if the cut already exists
    auto it = cutMaster_to_cut_map.find(cut_key);
    if (it != cutMaster_to_cut_map.end()) {
        // Update the existing cut
        if (cuts[it->second].added) {
            cut.added = true;
            cut.updated = true;
        }

        cut.id = it->second;
        cut.key = cut_key;
        cuts[cut.id] = cut;
    } else {
        // Add the new cut
        cut.id = cuts.size();
        cut.key = cut_key;
        cuts.push_back(cut);
        cutMaster_to_cut_map[cut_key] = cut.id;
    }

    // Update the indexCuts map
    indexCuts[cut_key].push_back(cuts.size() - 1);
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
    // Create a map for non-zero entries by rows
    std::vector<std::vector<int>> row_indices_map(N_SIZE);
    // print num_rows
    // fmt::print("Num rows: {}\n", A.num_rows);
    for (int idx = 0; idx < A.values.size(); ++idx) {
        int row = A.rows[idx];
        // fmt::print("Row: {}\n", row);
        if (row > N_SIZE - 2) {
            continue;
        }
        row_indices_map[row + 1].push_back(idx);
    }
    std::vector<std::pair<double, Cut>> tmp_cuts;

    const int JOBS = std::thread::hardware_concurrency();

    auto nC = N_SIZE;
    std::vector<std::tuple<int, int, int>> tasks;
    tasks.reserve((nC * (nC - 1) * (nC - 2)) /
                  6);  // Preallocate task space to avoid reallocations

    // Create tasks for each combination of (i, j, k)
    for (int i = 1; i < N_SIZE - 1; ++i) {
        for (int j = i + 1; j < N_SIZE - 1; ++j) {
            for (int k = j + 1; k < N_SIZE - 1; ++k) {
                tasks.emplace_back(i, j, k);
            }
        }
    }

    // Define chunk size to reduce parallelization overhead
    const int chunk_size = (N_SIZE - 1) / JOBS;
#ifdef NSYNC
    nsync::nsync_mu cuts_mutex = NSYNC_MU_INIT;
#else
    std::mutex cuts_mutex;  // Protect access to shared resources
#endif

    // Parallel execution in chunks
    auto bulk_sender = stdexec::bulk(
        stdexec::just(), (tasks.size() + chunk_size - 1) / chunk_size,
        [this, &row_indices_map, &A, &x, nC, &cuts_mutex, &tasks, chunk_size,
         &tmp_cuts](std::size_t chunk_idx) {
            size_t start_idx = chunk_idx * chunk_size;
            size_t end_idx = std::min(start_idx + chunk_size, tasks.size());

            // Process a chunk of tasks
            for (size_t task_idx = start_idx; task_idx < end_idx; ++task_idx) {
                const auto &[i, j, k] = tasks[task_idx];
                std::vector<int> expanded(A.num_cols, 0);
                std::vector<int> buffer_int(A.num_cols);
                int buffer_int_n = 0;
                double lhs = 0.0;

                // Combine the three updates into one loop for efficiency
                for (int idx : row_indices_map[i]) expanded[A.cols[idx]] += 1;
                for (int idx : row_indices_map[j]) expanded[A.cols[idx]] += 1;
                for (int idx : row_indices_map[k]) expanded[A.cols[idx]] += 1;

                // Accumulate LHS cut values
                for (int idx = 0; idx < A.num_cols; ++idx) {
                    if (expanded[idx] >= 2) {
                        lhs += std::floor(expanded[idx] * 0.5) * x[idx];
                        buffer_int[buffer_int_n++] = idx;
                    }
                }

                // If lhs violation found, insert the cut
                if (lhs > 1.0 + 1e-3) {
                    std::array<uint64_t, num_words> C = {};
                    std::array<uint64_t, num_words> AM = {};
                    std::vector<int> order(N_SIZE, 0);
                    int ordering = 0;

                    std::vector<int> C_index = {i, j, k};
                    for (auto node : C_index) {
                        C[node / 64] |= (1ULL << (node % 64));
                        AM[node / 64] |= (1ULL << (node % 64));
                        order[node] = ordering++;
                    }
                    std::vector<int> remainingNodes;
                    remainingNodes.assign(buffer_int.begin(),
                                          buffer_int.begin() + buffer_int_n);
                    SRCPermutation p;
                    p.num = {1, 1, 1};
                    p.den = 2;
                    double rhs = 0.0;
                    for (size_t i = 0; i < 3; ++i) {
                        rhs += static_cast<double>(p.num[i]) / p.den;
                    }
                    rhs = std::floor(rhs);

                    std::vector<double> cut_coefficients(allPaths.size(), 0.0);

                    ankerl::unordered_dense::set<int> C_set(C_index.begin(),
                                                            C_index.end());
                    for (auto node : remainingNodes) {
                        auto &consumers =
                            allPaths[node];  // Reference to the consumers for
                        // in-place modification

                        int first = -1, second = -1;

                        // Find the first and second appearances of any
                        // element in C_set within consumers
                        for (size_t i = 1; i < consumers.size() - 1; ++i) {
                            if (C_set.find(consumers[i]) != C_set.end()) {
                                if (first == -1) {
                                    first = i;
                                } else {
                                    second = i;
                                    break;  // We found both first and second,
                                    // so we can exit the loop
                                }
                            }
                        }

                        // If we found both the first and second indices,
                        // mark nodes in AM
                        if (first != -1 && second != -1) {
                            for (int i = first + 1; i < second; ++i) {
                                AM[consumers[i] / 64] |=
                                    (1ULL
                                     << (consumers[i] %
                                         64));  // Set the bit for the consumer
                            }
                            // break;
                        }
                    }

                    // for (int idx = 0; idx < N_SIZE; ++idx) {
                    //     AM[idx >> 6] |= (1ULL << (idx & 63));
                    // }
                    auto z = 0;
                    auto lhs = 0.0;
                    for (auto path : allPaths) {
                        auto &clients = path.route;  // Reference to clients for
                                                     // in-place modification
                        cut_coefficients[z++] = computeLimitedMemoryCoefficient(
                            C, AM, p, clients, order);
                        lhs += cut_coefficients[z - 1] * x[z - 1];
                    }
                    auto violation = lhs - (rhs + 1e-3);

                    if (violation > 0.0) {
                        Cut cut(C, AM, cut_coefficients, p);
                        cut.baseSetOrder = order;

// print lhs
#ifdef NSYNC
                        nsync::nsync_mu_lock(&cuts_mutex);
#else
                        std::lock_guard<std::mutex> lock(cuts_mutex);
#endif
                        tmp_cuts.emplace_back(lhs, cut);
#ifdef NSYNC
                        nsync::nsync_mu_unlock(&cuts_mutex);
#endif
                    }
                }
            }
        });

    // Submit work to the thread pool
    auto work = stdexec::starts_on(sched, bulk_sender);
    stdexec::sync_wait(std::move(work));

    pdqsort(tmp_cuts.begin(), tmp_cuts.end(),
            [](const auto &a, const auto &b) { return a.first > b.first; });

    auto max_cuts = 3;
    for (int i = 0; i < std::min(max_cuts, static_cast<int>(tmp_cuts.size()));
         ++i) {
        auto &cut = tmp_cuts[i].second;
        cutStorage.addCut(cut);
    }
    return;
}

std::pair<bool, bool> LimitedMemoryRank1Cuts::runSeparation(
    BNBNode *node, std::vector<baldesCtrPtr> &SRCconstraints) {
    auto cuts = &cutStorage;
    ModelData matrix = node->extractModelDataSparse();
    auto cuts_before = cuts->size();
    std::vector<double> solution;

    // print size of SRCconstraints
    // RUN_OPTIMIZATION(node, 1e-8)
    GET_SOL(node);
    //(node)->optimize();
    // solution = (node)->extractSolution();
    // solution = (node)->ipSolver->getPrimals();
    //
    // int z = 0;
    // for (auto &path : allPaths) {
    //     path.frac_x = solution[z];
    //     z++;
    // }

    // separateR1C1(matrix.A_sparse, solution);
    // if (cuts_before != cuts->size()) {
    //     return {true, false};
    // }
    separate(matrix.A_sparse, solution);
    if (cuts_before != cuts->size()) {
        return {true, false};
    }
    auto initial_cut_count = cuts->size();
    int rank3_cuts_size = cuts->size() - initial_cut_count;
    high_rank_cuts.cutStorage = &cutStorage;
    high_rank_cuts.allPaths = allPaths;
    high_rank_cuts.distances = generator->cost_mat4_vertex;
    high_rank_cuts.nodes = nodes;
    high_rank_cuts.arc_duals = arc_duals;

    high_rank_cuts.separate(matrix.A_sparse, solution);
    auto cuts_after_separation = cuts->size();

    ////////////////////////////////////////////////////
    // Handle non-violated cuts in a single pass
    ////////////////////////////////////////////////////
    // bool cleared = cutCleaner(node, SRCconstraints);
    bool cleared = false;
    auto n_cuts_removed = cuts_after_separation - cuts->size();

    // Simplify the final check
    bool cuts_changed = (cuts_before != cuts->size());
    return std::make_pair(cuts_changed, cleared);
}

bool LimitedMemoryRank1Cuts::cutCleaner(
    BNBNode *node, std::vector<baldesCtrPtr> &SRCconstraints) {
    auto cuts = &cutStorage;
    std::vector<double> solution;

    bool cleaned = false;
    GET_SOL(node);

    // Use reverse iterators to traverse the container in reverse order
    for (auto it = SRCconstraints.rbegin(); it != SRCconstraints.rend();) {
        auto constr = *it;
        int current_index = constr->index();
        double slack = node->getSlack(current_index, solution);

        // If the slack is positive, it means the constraint is not
        // violated
        if (std::abs(slack) > 1e-3) {
            cleaned = true;
            node->remove(constr);
            cuts->removeCut(cuts->getID(
                std::distance(SRCconstraints.begin(), it.base()) - 1));

            // Remove from SRCconstraints using reverse iterator
            it = decltype(it){SRCconstraints.erase(std::next(it).base())};
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
