/**
 * @file SRC.cpp
 * @brief Implementation of functions and classes for solving VRP problems using Limited Memory Rank-1 Cuts.
 *
 * This file contains the implementation of various functions and classes used for solving Vehicle Routing Problems
 * (VRP) using Limited Memory Rank-1 Cuts. The main functionalities include computing unique cut keys, adding cuts to
 * storage, separating solution vectors into cuts, and generating cut coefficients.
 *
 * The file includes the following main components:
 * - `hash_double`: A function to hash double values with high precision.
 * - `hash_combine`: A function to combine hash values robustly.
 * - `compute_cut_key`: A function to compute a unique cut key based on base sets and multipliers.
 * - `CutStorage::addCut`: A method to add a cut to the CutStorage.
 * - `LimitedMemoryRank1Cuts::separate`: A method to separate solution vectors into cuts using Limited Memory Rank-1
 * Cuts.
 * - `LimitedMemoryRank1Cuts::insertSet`: A method to insert a set of indices and associated data into the VRPTW_SRC
 * cuts structure.
 * - `findRoutesVisitingNodes`: A function to find routes that visit specified nodes in a sparse model.
 * - `LimitedMemoryRank1Cuts::generateCutCoefficients`: A method to generate cut coefficients for VRPTW_SRC cuts.
 *
 * The implementation leverages parallel processing using thread pools and schedulers to efficiently handle large
 * datasets and complex computations.
 *
 * @note The code assumes the existence of certain external structures and constants such as `num_words`, `N_SIZE`,
 * `VRPTW_SRC`, `VRPTW_SRC_max_S_n`, `SparseModel`, `VRPNode`, `CutType`, `exec::static_thread_pool`, `stdexec::bulk`,
 * and `stdexec::sync_wait`.
 */

#include "cuts/SRC.h"

#include "Definitions.h"
#include <stdexec/execution.hpp>

#include <bitset>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <mutex>
#include <vector>

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
std::size_t compute_cut_key(const std::array<uint64_t, num_words> &baseSet, const std::vector<double> &multipliers) {
    std::size_t cut_key = 0;

    // Hash the baseSet (uint64_t values) while preserving order
    for (std::size_t i = 0; i < baseSet.size(); ++i) {
        hash_combine(cut_key, baseSet[i]); // Combine both the value and the index
    }

    // Hash the multipliers (double values) while preserving order
    for (std::size_t i = 0; i < multipliers.size(); ++i) {
        std::size_t multiplier_hash = hash_double(multipliers[i], i); // Hash with precision and index
        hash_combine(cut_key, multiplier_hash);
    }

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

    auto baseSet     = cut.baseSet;
    auto neighbors   = cut.neighbors;
    auto multipliers = cut.multipliers;

    std::size_t cut_key = compute_cut_key(baseSet, multipliers);

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
    std::vector<std::vector<int>> row_indices_map(A.num_rows + 1);
    for (int idx = 0; idx < A.elements.size(); ++idx) {
        int row = A.elements[idx].row;
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

    std::mutex cuts_mutex; // Mutex to protect shared resource

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
    const int chunk_size = 10; // Adjust chunk size based on performance experiments

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
                for (int idx : row_indices_map[i]) expanded[A.elements[idx].col] += 1;
                for (int idx : row_indices_map[j]) expanded[A.elements[idx].col] += 1;
                for (int idx : row_indices_map[k]) expanded[A.elements[idx].col] += 1;

                // Accumulate LHS cut values
                for (int idx = 0; idx < A.num_cols; ++idx) {
                    if (expanded[idx] >= 2) {
                        lhs += std::floor(expanded[idx] * 0.5) * x[idx];
                        buffer_int[buffer_int_n++] = idx;
                    }
                }

                // If lhs violation found, insert the cut
                if (lhs > 1.0 + 1e-3) {
                    std::lock_guard<std::mutex> lock(cuts_mutex);
                    insertSet(cuts, i, j, k, buffer_int, buffer_int_n, lhs);
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
    int    max_number_of_cuts = 3;

    if (cuts.S_n > 0) {
        int m_max = std::min(cuts.S_n, max_number_of_cuts);

        // Prepare thread pool and scheduler
        exec::static_thread_pool pool(std::thread::hardware_concurrency());
        auto                     sched = pool.get_scheduler();

        std::mutex cuts_mutex; // Mutex for cutStorage to ensure thread-safe access

        auto input_sender = stdexec::just();

        // Sort best_sets
        std::sort(cuts.best_sets.begin(), cuts.best_sets.end(), std::greater<>());

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

                std::unordered_set<int> C_set(C_index.begin(), C_index.end());
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
                for (int j = start; j < start + 3; ++j) {
                    int node = cuts.S_C_P[j];
                    AM[node / 64] |= (1ULL << (node % 64)); // Set the bit for node in AM
                    order[node] = ordering++;
                }

                auto p = {0.5, 0.5, 0.5}; // Example

                // Iterate over remaining nodes and calculate the coefficients_aux
                for (auto node : remainingNodes) {
#ifdef SRC3
                    for (auto c : C_index) { coefficients_aux[node] += allPaths[node].countOccurrences(c) * 0.5; }
                    coefficients_aux[node] = std::floor(coefficients_aux[node]);
#endif
#ifdef SRC
                    auto &clients          = allPaths[node].route; // Reference to the clients for in-place modification
                    coefficients_aux[node] = computeLimitedMemoryCoefficient(C, AM, p, clients, order);
#endif
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
