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
    // Pre-allocate row indices map with reserve
    std::vector<std::vector<int>> row_indices_map(N_SIZE);
    for (auto &row : row_indices_map) {
        row.reserve(32);  // Typical size, adjust based on your data
    }

    // Build row indices map
    for (int idx = 0; idx < A.values.size(); ++idx) {
        const int row = A.rows[idx];
        if (row <= N_SIZE - 2) {
            row_indices_map[row + 1].push_back(idx);
        }
    }

    // Pre-allocate thread-local storage to avoid allocations in parallel
    // section
    const int JOBS = std::thread::hardware_concurrency();
    std::vector<std::vector<std::pair<double, Cut>>> thread_local_cuts(JOBS);
    for (auto &cuts : thread_local_cuts) {
        cuts.reserve(20);  // Adjust based on expected number of cuts per thread
    }

    exec::static_thread_pool pool(JOBS);
    auto sched = pool.get_scheduler();

    // Create tasks more efficiently
    std::vector<std::tuple<int, int, int>> tasks;
    const int n = N_SIZE - 1;
    tasks.reserve((n * (n - 1) * (n - 2)) / 6);

    for (int i = 1; i < N_SIZE - 1; ++i) {
        for (int j = i + 1; j < N_SIZE - 1; ++j) {
            for (int k = j + 1; k < N_SIZE - 1; ++k) {
                tasks.emplace_back(i, j, k);
            }
        }
    }

    // Thread-local buffers to avoid reallocations
    struct ThreadLocalBuffers {
        std::vector<int> expanded;
        std::vector<int> buffer_int;
        std::vector<int> remainingNodes;
        std::vector<double> cut_coefficients;
        std::vector<int> order;
        ankerl::unordered_dense::set<int> C_set;

        ThreadLocalBuffers(int num_cols)
            : expanded(num_cols),
              buffer_int(num_cols),
              remainingNodes(num_cols),
              cut_coefficients(),
              order(N_SIZE),
              C_set(3) {}  // Reserve space for C_set
    };

    const int chunk_size =
        std::max(1000, static_cast<int>((tasks.size() + JOBS - 1) / JOBS));

    auto bulk_sender = stdexec::bulk(
        stdexec::just(), (tasks.size() + chunk_size - 1) / chunk_size,
        [this, &row_indices_map, &A, &x, &thread_local_cuts, &tasks,
         chunk_size](std::size_t chunk_idx) {
            // Thread-local buffers
            ThreadLocalBuffers buffers(A.num_cols);
            buffers.cut_coefficients.resize(
                this->allPaths.size());  // Resize here

            auto &local_cuts =
                thread_local_cuts[chunk_idx % thread_local_cuts.size()];

            size_t start_idx = chunk_idx * chunk_size;
            size_t end_idx = std::min(start_idx + chunk_size, tasks.size());

            for (size_t task_idx = start_idx; task_idx < end_idx; ++task_idx) {
                const auto &[i, j, k] = tasks[task_idx];

                // Clear expanded array efficiently
                if (buffers.expanded.size() > 0) {
                    std::memset(buffers.expanded.data(), 0,
                                buffers.expanded.size() * sizeof(int));
                }

                int buffer_int_n = 0;
                double lhs = 0.0;

                // Process rows more efficiently
                for (int idx : row_indices_map[i])
                    buffers.expanded[A.cols[idx]]++;
                for (int idx : row_indices_map[j])
                    buffers.expanded[A.cols[idx]]++;
                for (int idx : row_indices_map[k])
                    buffers.expanded[A.cols[idx]]++;

// Vectorization hint for modern compilers
#pragma GCC ivdep
                for (int idx = 0; idx < A.num_cols; ++idx) {
                    if (buffers.expanded[idx] >= 2) {
                        lhs += std::floor(buffers.expanded[idx] * 0.5) * x[idx];
                        buffers.buffer_int[buffer_int_n++] = idx;
                    }
                }

                if (lhs > 1.0 + 1e-3) {
                    std::array<uint64_t, num_words> C = {};
                    std::array<uint64_t, num_words> AM = {};
                    std::fill(buffers.order.begin(), buffers.order.end(), 0);

                    // Set bits more efficiently
                    const std::array<int, 3> C_index = {i, j, k};
                    for (int idx = 0; idx < 3; ++idx) {
                        const int node = C_index[idx];
                        C[node >> 6] |= (1ULL << (node & 63));
                        AM[node >> 6] |= (1ULL << (node & 63));
                        buffers.order[node] = idx;
                    }

                    buffers.remainingNodes.assign(
                        buffers.buffer_int.begin(),
                        buffers.buffer_int.begin() + buffer_int_n);

                    SRCPermutation p;
                    p.num = {1, 1, 1};
                    p.den = 2;

                    // Reuse C_set
                    buffers.C_set.clear();
                    buffers.C_set.insert(C_index.begin(), C_index.end());

                    // Process remaining nodes
                    for (int node : buffers.remainingNodes) {
                        const auto &consumers = allPaths[node];
                        const size_t n = consumers.size();

                        // Skip paths that are too short
                        if (n < 3) continue;

                        // Direct array access instead of hash lookups
                        const int c1 = C_index[0], c2 = C_index[1],
                                  c3 = C_index[2];

                        size_t first = n, second = n;

// Vectorization hint
#pragma GCC ivdep
                        for (size_t idx = 1; idx < n - 1; ++idx) {
                            const int curr = consumers[idx];
                            // Check against all three C_set elements at once
                            const bool is_in_c =
                                (curr == c1 || curr == c2 || curr == c3);

                            // Branchless updates
                            const bool update_first = is_in_c && (first == n);
                            const bool update_second =
                                is_in_c && (first != n) && (idx > first);

                            first = update_first ? idx : first;
                            second = update_second ? idx : second;
                        }

                        // If we found two positions
                        if (second != n) {
// Process elements between first and second
#pragma GCC ivdep
                            for (size_t idx = first + 1; idx < second;
                                 idx += 4) {
                                // Process up to 4 elements at once
                                const size_t remain = second - idx;
                                if (remain >= 4) {
                                    const uint16_t n1 = consumers[idx];
                                    const uint16_t n2 = consumers[idx + 1];
                                    const uint16_t n3 = consumers[idx + 2];
                                    const uint16_t n4 = consumers[idx + 3];

                                    AM[n1 >> 6] |= (1ULL << (n1 & 63));
                                    AM[n2 >> 6] |= (1ULL << (n2 & 63));
                                    AM[n3 >> 6] |= (1ULL << (n3 & 63));
                                    AM[n4 >> 6] |= (1ULL << (n4 & 63));
                                } else {
                                    // Handle remaining elements
                                    for (size_t j = 0; j < remain; ++j) {
                                        const uint16_t n = consumers[idx + j];
                                        AM[n >> 6] |= (1ULL << (n & 63));
                                    }
                                    break;
                                }
                            }
                            break;
                        }
                    }

                    // Compute coefficients
                    size_t z = 0;
                    for (const auto &path : allPaths) {
                        buffers.cut_coefficients[z++] =
                            computeLimitedMemoryCoefficient(
                                C, AM, p, path.route, buffers.order);
                    }

                    Cut cut(C, AM, buffers.cut_coefficients, p);
                    cut.baseSetOrder = buffers.order;
                    local_cuts.emplace_back(lhs, cut);
                }
            }
        });

    // Execute work
    stdexec::sync_wait(stdexec::on(sched, std::move(bulk_sender)));

    // Merge and sort cuts from all threads
    std::vector<std::pair<double, Cut>> final_cuts;
    for (const auto &thread_cuts : thread_local_cuts) {
        final_cuts.insert(final_cuts.end(), thread_cuts.begin(),
                          thread_cuts.end());
    }

    pdqsort(final_cuts.begin(), final_cuts.end(),
            [](const auto &a, const auto &b) { return a.first > b.first; });

    // Add top cuts
    const int max_cuts = 2;
    for (int i = 0; i < std::min(max_cuts, static_cast<int>(final_cuts.size()));
         ++i) {
        cutStorage.addCut(final_cuts[i].second);
    }
}

void LimitedMemoryRank1Cuts::separateBG(
    BNBNode *node, std::vector<baldesCtrPtr> &SRCconstraints) {
    // cutCleaner(node, SRCconstraints);

    if (generator->checkBackgroundTask()) {
        return;
    }

    std::vector<double> solution;
    GET_SOL(node);

    size_t i = 0;
    std::for_each(allPaths.begin(), allPaths.end(),
                  [&](auto &path) { path.frac_x = solution[i++]; });

    auto cuts = &cutStorage;
    auto cuts_after_separation = cuts->size();
    generator->setArcDuals(arc_duals);
    generator->setNodes(nodes);
    generator->max_heuristic_sep_mem4_row_rank1 = 8;
    generator->initialize(allPaths);
    generator->generateSepHeurMem4Vertex();
    generator->generateCutsInBackground();
}

bool LimitedMemoryRank1Cuts::getCutsBG() {
    if (!generator->readyGeneratedCuts()) {
        cuts_harvested = false;
        return false;
    }

    std::vector<double> solution;

    cuts_harvested = true;

    auto cuts = &cutStorage;
    auto cuts_before = cuts->size();

    auto cutsBG = generator->returnBGcuts();

    n_high_order = cutsBG.size();
    if (n_high_order == 0) {
        return false;
    }
    // print_info("Processing {} background cuts\n", cutsBG.size());
    processCuts(cutsBG);

    auto cuts_after_separation = cuts->size();
    auto cuts_changed = cuts_after_separation - cuts_before;

    // cuts->updateActiveCuts();
    if (cuts_changed > 0) {
        return true;
    }
    return false;
}

std::pair<bool, bool> LimitedMemoryRank1Cuts::runSeparation(
    BNBNode *node, std::vector<baldesCtrPtr> &SRCconstraints) {
    auto cuts = &cutStorage;
    ModelData matrix = node->extractModelDataSparse();
    auto cuts_before = cuts->size();
    std::vector<double> solution;
    GET_SOL(node);
    separateR1C1(matrix.A_sparse, solution);
    separate(matrix.A_sparse, solution);
    auto cuts_after_separation = cuts->size();

    // if (cuts_after_separation != cuts_before) {
    //     bool cleared = cutCleaner(node, SRCconstraints);
    //     return std::make_pair(true, cleared);
    // }

    if (cuts_harvested) {
        cuts_before -= n_high_order;
        cuts_after_separation -= n_high_order;
    }

    if (!cuts_harvested) {
        bool readGeneration = generator->readyGeneratedCuts();
        if (readGeneration) {
            auto cutsBG = generator->returnBGcuts();
            // print_info("Processing {} background cuts\n", cutsBG.size());
            cuts_harvested = true;
            n_high_order = cutsBG.size();
            if (n_high_order > 0) {
                processCuts(cutsBG);
            }
        } else {
            separateBG(node, SRCconstraints);
            cuts_harvested = false;
        }
    }

    ////////////////////////////////////////////////////
    // Handle non-violated cuts in a single pass
    ////////////////////////////////////////////////////
    int cuts_after = cuts_after_separation + n_high_order;
    bool cleared = cutCleaner(node, SRCconstraints);
    int n_cuts_removed = cuts_after - cuts->size();

    // Simplify the final check
    bool cuts_changed = (cuts_before != cuts->size() + n_cuts_removed);
    if (!cuts_harvested) {
        cuts_changed = true;
    }
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
        if (slack > 0) {
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
 * @brief Computes the limited memory coefficient for a given set of
 * nodes.
 *
 */
double LimitedMemoryRank1Cuts::computeLimitedMemoryCoefficient(
    const std::array<uint64_t, num_words> &C,
    const std::array<uint64_t, num_words> &AM, const SRCPermutation &p,
    const std::vector<uint16_t> &P, const std::vector<int> &order) noexcept {
    double alpha = 0.0;
    int S = 0;
    const int den = p.den;
    const size_t P_size = P.size();

// Vectorization hint
#pragma GCC ivdep
    for (size_t j = 1; j < P_size - 1; ++j) {
        const int vj = P[j];
        // Combine bit operations - shift once
        const uint64_t word_idx = vj >> 6;
        const uint64_t bit_mask = 1ULL << (vj & 63);

        // Branchless updates using arithmetic
        const bool in_am = (AM[word_idx] & bit_mask) != 0;
        const bool in_c = (C[word_idx] & bit_mask) != 0;

        // Reset S if not in AM
        S *= in_am;

        // Update S and alpha if in both AM and C
        if (in_am && in_c) {
            S += p.num[order[vj]];
            // Branchless alpha update
            alpha += S >= den;
            S = S >= den ? S - den : S;
        }
    }

    return alpha;
}
