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
#include <stdexec/execution.hpp>

#include <cstddef>
#include <functional>
#include <mutex>
#include <vector>

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
std::vector<std::vector<double>> LimitedMemoryRank1Cuts::separate(const SparseMatrix &A, const std::vector<double> &x) {
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

    auto cuts    = VRPTW_SRC();
    cuts.S_n_max = VRPTW_SRC_max_S_n;
    cuts.S.resize(cuts.S_n_max + 1);
    cuts.S_n       = 0;
    cuts.S_C_P_max = 50 * cuts.S_n_max;
    cuts.S_C_P.resize(cuts.S_C_P_max);

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
        [this, &row_indices_map, &A, &x, nC, &cuts, &cuts_mutex, &tasks, chunk_size](std::size_t chunk_idx) {
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
// print lhs
#ifdef NSYNC
                    nsync::nsync_mu_lock(&cuts_mutex);
#else
                    std::lock_guard<std::mutex> lock(cuts_mutex);
#endif
                    insertSet(cuts, i, j, k, buffer_int, buffer_int_n, lhs);
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
    std::vector<std::vector<double>> coefficients;
    if (cuts.S_n > 0) { generateCutCoefficients(cuts, coefficients, A.num_cols, A, x); }
    cuts.S.resize(cuts.S_n + 1);
    cuts.S[cuts.S_n] = cuts.S_C_P.size();

    return coefficients;
}

/**
 * @brief Inserts a set of indices and associated data into the VRPTW_SRC cuts structure.
 *
 * This function inserts a set of indices (i, j, k) and additional buffer data into the
 * VRPTW_SRC cuts structure. It ensures that the underlying storage has enough capacity
 * to accommodate the new data and resizes it if necessary. The function also updates
 * the best sets and the S_C_P vector with the new data.
 *
 */
void LimitedMemoryRank1Cuts::insertSet(VRPTW_SRC &cuts, int i, int j, int k, const std::vector<int> &buffer_int,
                                       int buffer_int_n, double LHS_cut) {
    // Estimate size to avoid frequent reallocations
    size_t required_size = cuts.S_C_P.size() + 3 + buffer_int_n;
    if (required_size > cuts.S_C_P.capacity()) {
        size_t new_size =
            static_cast<size_t>(std::ceil(required_size * 2.0)); // Double the size to avoid frequent reallocations
        cuts.S_C_P.reserve(new_size);
        cuts.S_C_P_max = new_size;
    }

    // Insert best set with emplace_back to avoid extra copies
    cuts.best_sets.emplace_back(LHS_cut, cuts.S_n);

    // Resize cuts.S only if needed and in larger chunks
    if (cuts.S.size() <= cuts.S_n) {
        cuts.S.resize(cuts.S_n + 10); // Resize in larger chunks to reduce frequent resizing
    }

    // Store the current index of cuts.S_C_P for this set
    cuts.S[cuts.S_n] = cuts.S_C_P.size();

    // Directly insert the values into S_C_P
    cuts.S_C_P.push_back(i);
    cuts.S_C_P.push_back(j);
    cuts.S_C_P.push_back(k);

    // Use bulk insert for buffer_int to reduce loop overhead
    cuts.S_C_P.insert(cuts.S_C_P.end(), buffer_int.begin(), buffer_int.begin() + buffer_int_n);

    // Increment the set counter
    cuts.S_n++;
}

/**
 * @brief Generates cut coefficients for the given VRPTW_SRC cuts.
 *
 * This function generates cut coefficients for the given VRPTW_SRC cuts using a limited memory rank-1 approach.
 * It processes the cuts in parallel using a thread pool and scheduler, ensuring thread-safe access to shared resources.
 *
 */
void LimitedMemoryRank1Cuts::generateCutCoefficients(VRPTW_SRC &cuts, std::vector<std::vector<double>> &coefficients,
                                                     int numNodes, const SparseMatrix &A,
                                                     const std::vector<double> &x) {
    double primal_violation   = 0.0;
    int    max_number_of_cuts = 10;

    if (cuts.S_n > 0) {
        int m_max = std::min(cuts.S_n, max_number_of_cuts);

        // Prepare thread pool and scheduler
        exec::static_thread_pool pool(std::thread::hardware_concurrency());
        auto                     sched = pool.get_scheduler();

        std::mutex cuts_mutex; // Mutex for cutStorage to ensure thread-safe access

        auto input_sender = stdexec::just();

        // Sort best_sets
        pdqsort(cuts.best_sets.begin(), cuts.best_sets.end(), std::greater<>());

        // Define the bulk operation to process each cut
        auto bulk_sender = stdexec::bulk(
            input_sender, m_max, [this, &cuts, &coefficients, &x, &numNodes, &cuts_mutex](std::size_t ii) {
                if (cuts.best_sets.empty()) return;

                int aux_int = cuts.best_sets[ii].second;

                int start = cuts.S[aux_int];
                int end   = cuts.S[aux_int + 1];
                if (end == 0) return;

                // Set up data for each thread
                std::array<uint64_t, num_words> C  = {}; // Reset C for each cut
                std::array<uint64_t, num_words> AM = {};
                std::vector<int>                order(N_SIZE, 0);
                std::vector<double>             coefficients_aux(numNodes, 0.0);
                std::vector<int>                remainingNodes;
                remainingNodes.reserve(numNodes);

                // Build the C set and remaining nodes
                std::vector<int> C_index;
                C_index.reserve(3);
                for (int j = start; j < start + 3; ++j) {
                    int node = cuts.S_C_P[j];
                    C[node / 64] |= (1ULL << (node % 64)); // Set the bit for node in C
                    C_index.push_back(node);
                }
                remainingNodes.assign(cuts.S_C_P.begin() + start + 3, cuts.S_C_P.begin() + end);

#ifndef SRC
                for (int node = 0; node < N_SIZE; ++node) {
                    AM[node / 64] |= (1ULL << (node % 64)); // Set the bit for node in AM
                }
#else

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
                    }
                }
#endif
                // Set order and coefficients_aux
                int ordering = 0;
                for (auto node : C_index) {
                    AM[node / 64] |= (1ULL << (node % 64)); // Set the bit for node in AM
                    order[node] = ordering;
                    ordering++;
                }

                SRCPermutation p;
                p.num = {1, 1, 1};
                p.den = 2;

                // Iterate over remaining nodes and calculate the coefficients_aux
                for (auto node : remainingNodes) {
                    auto &clients          = allPaths[node].route; // Reference to the clients for in-place modification
                    coefficients_aux[node] = computeLimitedMemoryCoefficient(C, AM, p, clients, order);
                }

                // Create and store the cut
                Cut cut(C, AM, coefficients_aux);
                cut.baseSetOrder = order;

                // Thread-safe addition of the cut
                {
                    std::lock_guard<std::mutex> lock(cuts_mutex);
                    cutStorage.addCut(cut);
                }
            });

        auto work = stdexec::starts_on(sched, bulk_sender);
        stdexec::sync_wait(std::move(work));
    }
}

std::pair<bool, bool> LimitedMemoryRank1Cuts::runSeparation(BNBNode *node, std::vector<Constraint *> &SRCconstraints) {
    auto                cuts = &cutStorage;
    ModelData           matrix;
    auto                cuts_before = cuts->size();
    std::vector<double> solution;

    // print size of SRCconstraints
    RUN_OPTIMIZATION(node, 1e-8)

    // print size of solution
    separate(matrix.A_sparse, solution);
    auto   cuts_after_separation = cuts->size();
    size_t i                     = 0;
    std::for_each(allPaths.begin(), allPaths.end(), [&](auto &path) { path.frac_x = solution[i++]; });

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
        // Generator operations
        generator->initialize(allPaths);
        generator->generateR1C1();
        generator->setMemFactor(0.15);
        generator->fillMemory();
        generator->getHighDimCuts();
        generator->constructMemoryVertexBased();

        auto cut_number = processCuts();
        if (cut_number == 0) {
            for (double memFactor = 0.25; memFactor <= 0.45; memFactor += 0.10) {
                generator->setMemFactor(memFactor);
                generator->constructMemoryVertexBased();
                auto cut_number = processCuts();
                if (cut_number != 0) {
                    break; // Exit the loop if cuts are successfully processed
                }
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
            [](const Constraint *a, const Constraint *b) { return a->index() < b->index(); });

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
