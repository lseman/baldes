/**
 * @file SRC.cpp
 * @brief Implementation of functions and classes for solving VRP problems using Limited Memory Rank-1 Cuts.
 *
 * This file contains the implementation of various functions and classes used for solving Vehicle Routing Problems
 * (VRP) using Limited Memory Rank-1 Cuts. The main functionalities include computing unique cut keys, adding cuts to
 * storage, separating solution vectors into cuts, and generating cut coefficients.
 *
 * The implementation leverages parallel processing using thread pools and schedulers to efficiently handle large
 * datasets and complex computations.
 *
 */

#include "cuts/SRC.h"

#include "Cut.h"
#include "Definitions.h"

#include "bnb/Node.h"

#ifdef IPM
#include "ipm/IPSolver.h"
#endif

#ifdef HIGHS
#include <Highs.h>
#endif
using Cuts = std::vector<Cut>;

/**
 * @brief Computes a unique cut key based on the provided base set and multipliers.
 *
 * This function generates a hash value that uniquely identifies the combination
 * of the base set and multipliers. The base set is an array of uint64_t values,
 * and the multipliers are a vector of double values. The order of elements in
 * both the base set and multipliers is preserved during hashing.
 *
 */
std::size_t compute_cut_key(const std::array<uint64_t, num_words> &baseSet, const std::vector<int> &multipliers) {
    XXH3_state_t *state = XXH3_createState(); // Initialize the XXH3 state
    XXH3_64bits_reset(state);                 // Reset the hashing state

    // Hash the baseSet (uint64_t values) while preserving order
    for (std::size_t i = 0; i < baseSet.size(); ++i) {
        XXH3_64bits_update(state, &baseSet[i], sizeof(uint64_t)); // Hash each element in the baseSet
    }

    // Hash the multipliers (double values) while preserving order
    for (std::size_t i = 0; i < multipliers.size(); ++i) {
        XXH3_64bits_update(state, &multipliers[i], sizeof(double)); // Hash each multiplier
    }

    // Finalize the hash
    std::size_t cut_key = XXH3_64bits_digest(state);

    XXH3_freeState(state); // Free the state

    return cut_key;
}

std::size_t compute_cut_key(const std::array<uint64_t, num_words> &baseSet, const std::vector<double> &multipliers) {
    XXH3_state_t *state = XXH3_createState(); // Initialize the XXH3 state
    XXH3_64bits_reset(state);                 // Reset the hashing state

    // Hash the baseSet (uint64_t values) while preserving order
    for (std::size_t i = 0; i < baseSet.size(); ++i) {
        XXH3_64bits_update(state, &baseSet[i], sizeof(uint64_t)); // Hash each element in the baseSet
    }

    // Hash the multipliers (double values) while preserving order
    for (std::size_t i = 0; i < multipliers.size(); ++i) {
        XXH3_64bits_update(state, &multipliers[i], sizeof(double)); // Hash each multiplier
    }

    // Finalize the hash
    std::size_t cut_key = XXH3_64bits_digest(state);

    XXH3_freeState(state); // Free the state

    return cut_key;
}

/**
 * @brief Adds a cut to the CutStorage.
 *
 * This function adds a given cut to the CutStorage. It first computes a unique key for the cut
 * based on its base set and multipliers. If a cut with the same key already exists in the storage,
 * it updates the existing cut. Otherwise, it adds the new cut to the storage and updates the
 * necessary mappings.
 *
 */
void CutStorage::addCut(Cut &cut) {

    auto baseSet   = cut.baseSet;
    auto neighbors = cut.neighbors;
    auto p_num     = cut.p.num;
    auto p_dem     = cut.p.den;

    // concatenate p_num and p_dem in a single vector
    auto p_cat = p_num;
    p_cat.push_back(p_dem);

    std::size_t cut_key = compute_cut_key(baseSet, p_cat);

    auto it = cutMaster_to_cut_map.find(cut_key);
    if (it != cutMaster_to_cut_map.end()) {

        if (cuts[it->second].added) {
            cut.added   = true;
            cut.updated = true;
        }

        cut.id       = it->second;
        cut.key      = cut_key;
        cuts[cut.id] = cut;
    } else {
        //   If the cut does not exist, add it to the cuts vector and update the map
        cut.id  = cuts.size();
        cut.key = cut_key;
        cuts.push_back(cut);
        cutMaster_to_cut_map[cut_key] = cut.id;
    }
    indexCuts[cut_key].push_back(cuts.size() - 1);
}

LimitedMemoryRank1Cuts::LimitedMemoryRank1Cuts(std::vector<VRPNode> &nodes) : nodes(nodes) {}

/**
 * @brief Separates the given solution vector into cuts using Limited Memory Rank-1 Cuts.
 *
 * This function generates cuts for the given solution vector `x` based on the sparse model `A`.
 * It uses a parallel approach to evaluate combinations of indices and identify violations
 * that exceed the specified threshold.
 *
 */
void LimitedMemoryRank1Cuts::separate(const SparseMatrix &A, const std::vector<double> &x) {
    // Create a map for non-zero entries by rows
    std::vector<std::vector<int>> row_indices_map(N_SIZE);
    // print num_rows
    // fmt::print("Num rows: {}\n", A.num_rows);
    for (int idx = 0; idx < A.values.size(); ++idx) {
        int row = A.rows[idx];
        // fmt::print("Row: {}\n", row);
        if (row > N_SIZE - 2) { continue; }
        row_indices_map[row + 1].push_back(idx);
    }
    std::vector<std::pair<double, Cut>> tmp_cuts;

    const int                JOBS = std::thread::hardware_concurrency();
    exec::static_thread_pool pool(JOBS);
    auto                     sched = pool.get_scheduler();

    auto                                   nC = N_SIZE;
    std::vector<std::tuple<int, int, int>> tasks;
    tasks.reserve((nC * (nC - 1) * (nC - 2)) / 6); // Preallocate task space to avoid reallocations

    // Create tasks for each combination of (i, j, k)
    for (int i = 1; i < N_SIZE - 1; ++i) {
        for (int j = i + 1; j < N_SIZE - 1; ++j) {
            for (int k = j + 1; k < N_SIZE - 1; ++k) { tasks.emplace_back(i, j, k); }
        }
    }

    // Define chunk size to reduce parallelization overhead
    const int chunk_size = (N_SIZE - 1) / JOBS;
#ifdef NSYNC
    nsync::nsync_mu cuts_mutex = NSYNC_MU_INIT;
#else
    std::mutex cuts_mutex; // Protect access to shared resources
#endif

    // Parallel execution in chunks
    auto bulk_sender = stdexec::bulk(
        stdexec::just(), (tasks.size() + chunk_size - 1) / chunk_size,
        [this, &row_indices_map, &A, &x, nC, &cuts_mutex, &tasks, chunk_size, &tmp_cuts](std::size_t chunk_idx) {
            size_t start_idx = chunk_idx * chunk_size;
            size_t end_idx   = std::min(start_idx + chunk_size, tasks.size());

            // Process a chunk of tasks
            for (size_t task_idx = start_idx; task_idx < end_idx; ++task_idx) {
                const auto &[i, j, k] = tasks[task_idx];
                std::vector<int> expanded(A.num_cols, 0);
                std::vector<int> buffer_int(A.num_cols);
                int              buffer_int_n = 0;
                double           lhs          = 0.0;

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
                    std::array<uint64_t, num_words> C  = {};
                    std::array<uint64_t, num_words> AM = {};
                    std::vector<int>                order(N_SIZE, 0);
                    int                             ordering = 0;

                    std::vector<int> C_index = {i, j, k};
                    for (auto node : C_index) {
                        C[node / 64] |= (1ULL << (node % 64));
                        AM[node / 64] |= (1ULL << (node % 64));
                        order[node] = ordering++;
                    }
                    std::vector<int> remainingNodes;
                    remainingNodes.assign(buffer_int.begin(), buffer_int.begin() + buffer_int_n);
                    SRCPermutation p;
                    p.num = {1, 1, 1};
                    p.den = 2;
                    std::vector<double> cut_coefficients(allPaths.size(), 0.0);

                    ankerl::unordered_dense::set<int> C_set(C_index.begin(), C_index.end());
                    for (auto node : remainingNodes) {
                        auto &consumers = allPaths[node]; // Reference to the consumers for in-place modification

                        int first = -1, second = -1;

                        // Find the first and second appearances of any element in C_set within consumers
                        for (size_t i = 1; i < consumers.size() - 1; ++i) {
                            if (C_set.find(consumers[i]) != C_set.end()) {
                                if (first == -1) {
                                    first = i;
                                } else {
                                    second = i;
                                    break; // We found both first and second, so we can exit the loop
                                }
                            }
                        }

                        // If we found both the first and second indices, mark nodes in AM
                        if (first != -1 && second != -1) {
                            for (int i = first + 1; i < second; ++i) {
                                AM[consumers[i] / 64] |= (1ULL << (consumers[i] % 64)); // Set the bit for the consumer
                            }
                            break;
                        }
                    }
                    auto z = 0;
                    for (auto path : allPaths) {
                        auto &clients         = path.route; // Reference to clients for in-place modification
                        cut_coefficients[z++] = computeLimitedMemoryCoefficient(C, AM, p, clients, order);
                    }

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
        });

    // Submit work to the thread pool
    auto work = stdexec::starts_on(sched, bulk_sender);
    stdexec::sync_wait(std::move(work));

    // Generate cut coefficients
    // get the top 3 cuts
    // auto cuts    = VRPTW_SRC();

    pdqsort(tmp_cuts.begin(), tmp_cuts.end(), [](const auto &a, const auto &b) { return a.first > b.first; });

    auto max_cuts = 2;
    for (int i = 0; i < std::min(max_cuts, static_cast<int>(tmp_cuts.size())); ++i) {
        auto &cut = tmp_cuts[i].second;
        cutStorage.addCut(cut);
    }
    return;
}

std::pair<bool, bool> LimitedMemoryRank1Cuts::runSeparation(BNBNode *node, std::vector<baldesCtrPtr > &SRCconstraints) {
    auto                cuts = &cutStorage;
    ModelData           matrix;
    auto                cuts_before = cuts->size();
    std::vector<double> solution;

    // print size of SRCconstraints
    // RUN_OPTIMIZATION(node, 1e-8)
    (node)->optimize();
    solution = (node)->extractSolution();
    size_t i = 0;
    std::for_each(allPaths.begin(), allPaths.end(), [&](auto &path) { path.frac_x = solution[i++]; });

    //separateR1C1(matrix.A_sparse, solution);
    separate(matrix.A_sparse, solution);

    // TupleBasedSeparator r1c4(allPaths);
    // r1c4.separate4R1Cs();
    auto cuts_after_separation = cuts->size();

    // print solution size and allPaths size
    auto processCuts = [&]() {
        // Initialize paths with solution values
        auto multipliers = generator->map_rank1_multiplier;
        auto cortes      = generator->getCuts();
        auto cut_ctr     = 0;

        // Process each cut
        for (auto &cut : cortes) {
            auto &cut_info = cut.info_r1c;

            // Skip if arc_mem is empty
            if (cut.arc_mem.empty()) { continue; }

            // Pre-allocate cut_indices to avoid repeated resizing
            std::vector<int> cut_indices;
            cut_indices.reserve(cut_info.first.size());
            for (auto &node : cut_info.first) { cut_indices.push_back(node); }

            // Use std::array with pre-initialization for bit arrays
            std::array<uint64_t, num_words> C  = {};
            std::array<uint64_t, num_words> AM = {};

            // Set bits in C and AM for cut_indices
            for (auto &node : cut_indices) {
                C[node / 64] |= (1ULL << (node % 64));
                AM[node / 64] |= (1ULL << (node % 64));
            }
            // Set bits in AM for arcs in arc_mem
            for (auto &arc : cut.arc_mem) { AM[arc / 64] |= (1ULL << (arc % 64)); }

            // Retrieve multiplier information
            auto          &mult = multipliers[cut_info.first.size()][cut_info.second];
            SRCPermutation p;
            p.num = std::get<0>(mult);
            p.den = std::get<1>(mult);

            // Initialize order without an extra resize, only if N_SIZE is constant
            std::vector<int> order(N_SIZE, 0);
            int              ordering = 0;
            for (auto node : cut_indices) { order[node] = ordering++; }

            // Pre-allocate coeffs to allPaths.size()
            std::vector<double> coeffs(allPaths.size(), 0.0);
            bool                has_coeff = false;

            // Compute coefficients for each path
            for (size_t i = 0; i < allPaths.size(); ++i) {
                auto &clients = allPaths[i].route;
                auto  coeff   = computeLimitedMemoryCoefficient(C, AM, p, clients, order);

                // Check if coefficient is above threshold
                if (coeff > 1e-3) { has_coeff = true; }
                coeffs[i] = coeff;
            }

            // Skip adding cut if no coefficients met threshold
            if (!has_coeff) { continue; }

            // Create and add new cut
            Cut corte(C, AM, coeffs, p);
            corte.baseSetOrder = order;
            cuts->addCut(corte);
            cut_ctr++;
        }
        return cut_ctr;
    };

    // if (cuts_before == cuts_after_separation) {
    generator->setNodes(nodes);
    generator->generateSepHeurMem4Vertex();
    generator->initialize(allPaths);
    //generator->fillMemory();
    generator->getHighDimCuts();
    // generator->setMemFactor(0.15);
    // generator->constructMemoryVertexBased();

    // auto cut_number = processCuts();
    for (double memFactor = 0.15; memFactor <= 0.55; memFactor += 0.20) {
        generator->setMemFactor(memFactor);
        generator->constructMemoryVertexBased();
        auto cut_number = processCuts();
        if (cut_number != 0) {
            break; // Exit the loop if cuts are successfully processed
        }
    }

    // }
    ////////////////////////////////////////////////////
    // Handle non-violated cuts in a single pass
    ////////////////////////////////////////////////////
    bool cleared        = false;
    auto n_cuts_removed = 0;
    // Iterate over the constraints in reverse order to remove non-violated cuts
    // sort SRCconstraints by index
    pdqsort(SRCconstraints.begin(), SRCconstraints.end(),
            [](const baldesCtrPtr a, const baldesCtrPtr b) { return a->index() < b->index(); });

    for (int i = SRCconstraints.size() - 1; i >= 0; --i) {
        auto constr = SRCconstraints[i];

        // Get the slack value of the constraint
        double slack = node->getSlack(constr->index(), solution);
        // double slack = slacks[constr->index()];

        // If the slack is positive, it means the constraint is not violated
        if (slack > 1e-3) {
            cleared = true;

            // Remove the constraint from the model and cut storage
            node->remove(constr);
            cuts->removeCut(cuts->getID(i));
            n_cuts_removed++;

            // Remove from SRCconstraints
            SRCconstraints.erase(SRCconstraints.begin() + i);
        }
    }

    if (cuts_before == cuts->size() + n_cuts_removed) { return std::make_pair(false, cleared); }

    return std::make_pair(true, cleared);
}

/*
 * @brief Computes the limited memory coefficient for a given set of nodes.
 *
 */
double LimitedMemoryRank1Cuts::computeLimitedMemoryCoefficient(const std::array<uint64_t, num_words> &C,
                                                               const std::array<uint64_t, num_words> &AM,
                                                               const SRCPermutation &p, const std::vector<int> &P,
                                                               std::vector<int> &order) {
    double alpha = 0.0;
    int    S     = 0;
    auto   den   = p.den;

    for (size_t j = 1; j < P.size() - 1; ++j) {
        int vj = P[j];

        // Precompute bitshift values for reuse
        uint64_t am_mask  = (1ULL << (vj & 63));
        uint64_t am_index = vj >> 6;

        // Check if vj is in AM using precomputed values
        if (!(AM[am_index] & am_mask)) {
            S = 0; // Reset S if vj is not in AM
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
