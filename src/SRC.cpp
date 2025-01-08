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
void LimitedMemoryRank1Cuts::separate(const SparseMatrix& A, const std::vector<double>& x) {
    // Create a map for non-zero entries by rows with pre-allocation
    std::vector<std::vector<int>> row_indices_map(N_SIZE);
    for (auto& row : row_indices_map) {
        row.reserve(A.num_cols / N_SIZE); // Estimate average row size
    }
    
    // Populate row indices map
    for (int idx = 0; idx < A.values.size(); ++idx) {
        int row = A.rows[idx];
        if (row > N_SIZE - 2) { continue; }
        row_indices_map[row + 1].push_back(idx);
    }

    // Pre-allocate vector for storing cuts
    std::vector<std::pair<double, Cut>> tmp_cuts;
    tmp_cuts.reserve(N_SIZE * 2); // Reserve space for estimated number of cuts
    
    const int JOBS = std::thread::hardware_concurrency();
    exec::static_thread_pool pool(JOBS);
    auto sched = pool.get_scheduler();

    // Pre-calculate tasks with reserve
    std::vector<std::tuple<int, int, int>> tasks;
    tasks.reserve((N_SIZE * (N_SIZE - 1) * (N_SIZE - 2)) / 6);
    
    // Generate tasks efficiently
    for (int i = 1; i < N_SIZE - 1; ++i) {
        for (int j = i + 1; j < N_SIZE - 1; ++j) {
            for (int k = j + 1; k < N_SIZE - 1; ++k) {
                tasks.emplace_back(i, j, k);
            }
        }
    }

    const int chunk_size = std::max(1, static_cast<int>((tasks.size() + JOBS - 1) / JOBS));
    std::mutex cuts_mutex;

    auto bulk_sender = stdexec::bulk(
        stdexec::just(), 
        (tasks.size() + chunk_size - 1) / chunk_size,
        [this, &row_indices_map, &A, &x, &cuts_mutex, &tasks, chunk_size, &tmp_cuts]
        (std::size_t chunk_idx) {
            const size_t start_idx = chunk_idx * chunk_size;
            const size_t end_idx = std::min(start_idx + chunk_size, tasks.size());
            
            // Reuse vectors to avoid repeated allocations
            std::vector<int> expanded(A.num_cols, 0);
            std::vector<int> buffer_int(A.num_cols);
            std::vector<int> order(N_SIZE);
            
            // Local storage for cuts to reduce mutex contention
            std::vector<std::pair<double, Cut>> local_cuts;
            local_cuts.reserve(chunk_size / 10); // Estimate number of violations
            
            for (size_t task_idx = start_idx; task_idx < end_idx; ++task_idx) {
                const auto& [i, j, k] = tasks[task_idx];
                std::fill(expanded.begin(), expanded.end(), 0);
                int buffer_int_n = 0;
                double lhs = 0.0;

                // Combine updates for better cache utilization
                for (int idx : row_indices_map[i]) expanded[A.cols[idx]]++;
                for (int idx : row_indices_map[j]) expanded[A.cols[idx]]++;
                for (int idx : row_indices_map[k]) expanded[A.cols[idx]]++;

                // Calculate LHS efficiently
                for (int idx = 0; idx < A.num_cols; ++idx) {
                    if (expanded[idx] >= 2) {
                        lhs += std::floor(expanded[idx] * 0.5) * x[idx];
                        buffer_int[buffer_int_n++] = idx;
                    }
                }

                if (lhs > 1.0 + 1e-3) {
                    std::array<uint64_t, num_words> C{};
                    std::array<uint64_t, num_words> AM{};
                    std::fill(order.begin(), order.end(), 0);
                    
                    const std::array<int, 3> C_index{i, j, k};
                    int ordering = 0;
                    
                    for (const auto node : C_index) {
                        C[node / 64] |= (1ULL << (node % 64));
                        AM[node / 64] |= (1ULL << (node % 64));
                        order[node] = ordering++;
                    }

                    std::vector<int> remainingNodes(buffer_int.begin(), 
                                                  buffer_int.begin() + buffer_int_n);
                    
                    SRCPermutation p;
                    p.num = {1, 1, 1};
                    p.den = 2;
                    
                    std::vector<double> cut_coefficients(allPaths.size(), 0.0);
                    
                    // Use unordered_set for faster lookups
                    ankerl::unordered_dense::set<int> C_set(C_index.begin(), C_index.end());

                    for (const auto node : remainingNodes) {
                        const auto& consumers = allPaths[node];
                        int first = -1, second = -1;

                        // Find first two occurrences efficiently
                        for (size_t i = 1; i < consumers.size() - 1; ++i) {
                            if (C_set.contains(consumers[i])) {
                                if (first == -1) {
                                    first = i;
                                } else {
                                    second = i;
                                    break;
                                }
                            }
                        }

                        if (first != -1 && second != -1) {
                            for (int i = first + 1; i < second; ++i) {
                                AM[consumers[i] / 64] |= (1ULL << (consumers[i] % 64));
                            }
                            break;
                        }
                    }

                    size_t z = 0;
                    for (const auto& path : allPaths) {
                        cut_coefficients[z++] = computeLimitedMemoryCoefficient(
                            C, AM, p, path.route, order);
                    }

                    Cut cut(C, AM, cut_coefficients, p);
                    cut.baseSetOrder = order;
                    local_cuts.emplace_back(lhs, std::move(cut));
                }
            }
            
            // Batch update shared cuts to reduce lock contention
            if (!local_cuts.empty()) {
                std::lock_guard<std::mutex> lock(cuts_mutex);
                tmp_cuts.insert(
                    tmp_cuts.end(),
                    std::make_move_iterator(local_cuts.begin()),
                    std::make_move_iterator(local_cuts.end())
                );
            }
        });

    // Submit work to thread pool
    auto work = stdexec::starts_on(sched, bulk_sender);
    stdexec::sync_wait(std::move(work));

    // Sort cuts by violation
    pdqsort(tmp_cuts.begin(), tmp_cuts.end(), 
            [](const auto& a, const auto& b) { return a.first > b.first; });

    // Add top cuts to storage
    const int max_cuts = std::min(2, static_cast<int>(tmp_cuts.size()));
    for (int i = 0; i < max_cuts; ++i) {
        Cut& cut = tmp_cuts[i].second;
        cutStorage.addCut(cut);
    }
}

std::pair<bool, bool> LimitedMemoryRank1Cuts::runSeparation(BNBNode *node, std::vector<baldesCtrPtr> &SRCconstraints) {
    auto cuts = &cutStorage;
    ModelData matrix;
    auto cuts_before = cuts->size();
    std::vector<double> solution;

    GET_SOL(node);

    size_t i = 0;
    std::for_each(allPaths.begin(), allPaths.end(), [&](auto &path) { path.frac_x = solution[i++]; });

    separateR1C1(matrix.A_sparse, solution);
    separate(matrix.A_sparse, solution);
    auto cuts_after_separation = cuts->size();
    auto multipliers = generator->map_rank1_multiplier;

    auto processCuts = [&]() {
        auto cortes = generator->getCuts();
        auto cut_ctr = 0;

        for (auto &cut : cortes) {
            auto &cut_info = cut.info_r1c;

            if (cut.arc_mem.empty()) { continue; }

            std::vector<int> cut_indices;
            cut_indices.reserve(cut_info.first.size());
            for (auto &node : cut_info.first) { cut_indices.push_back(node); }

            std::array<uint64_t, num_words> C = {};
            std::array<uint64_t, num_words> AM = {};

            // Optimize bit operations
            for (auto &node : cut_indices) {
                const uint32_t word_idx = node >> 6;  // Divide by 64
                const uint64_t bit_mask = 1ULL << (node & 63);  // Modulo 64
                C[word_idx] |= bit_mask;
                AM[word_idx] |= bit_mask;
            }

            for (auto &arc : cut.arc_mem) {
                AM[arc >> 6] |= (1ULL << (arc & 63));
            }

            auto &mult = multipliers[cut_info.first.size()][cut_info.second];
            SRCPermutation p;
            p.num = std::get<0>(mult);
            p.den = std::get<1>(mult);

            std::vector<int> order(N_SIZE, 0);
            int ordering = 0;
            for (auto node : cut_indices) { order[node] = ordering++; }

            std::vector<double> coeffs(allPaths.size(), 0.0);

#if defined(__cpp_lib_parallel_algorithm)
            std::atomic<bool> has_coeff{false};
            std::transform(std::execution::par_unseq, allPaths.begin(), allPaths.end(), coeffs.begin(),
                           [&](const auto &path) {
                               auto coeff = computeLimitedMemoryCoefficient(C, AM, p, path.route, order);
                               if (coeff > 1e-3) has_coeff.store(true, std::memory_order_relaxed);
                               return coeff;
                           });
            if (!has_coeff.load(std::memory_order_relaxed)) { continue; }
#else
            bool has_coeff = false;
            std::transform(allPaths.begin(), allPaths.end(), coeffs.begin(), [&](const auto &path) {
                auto coeff = computeLimitedMemoryCoefficient(C, AM, p, path.route, order);
                if (coeff > 1e-3) has_coeff = true;
                return coeff;
            });
            if (!has_coeff) { continue; }
#endif

            Cut corte(C, AM, coeffs, p);
            corte.baseSetOrder = order;
            cuts->addCut(corte);
            cut_ctr++;
        }
        return cut_ctr;
    };

    if (cuts_before == cuts_after_separation) {
        generator->setNodes(nodes);
        generator->generateSepHeurMem4Vertex();
        generator->initialize(allPaths);
        generator->getHighDimCuts();

        for (double memFactor = 0.15; memFactor <= 0.65; memFactor += 0.10) {
            generator->setMemFactor(memFactor);
            generator->constructMemoryVertexBased();
            auto cut_number = processCuts();
            if (cut_number != 0) {
                break;
            }
        }
    }

    if (cuts_before == cuts_after_separation) {
        generator->max_heuristic_sep_mem4_row_rank1 = 12;
        generator->generateSepHeurMem4Vertex();
        generator->initialize(allPaths);
        generator->getHighDimCuts();
        generator->constructMemoryVertexBased();
        auto cut_number = processCuts();
    }

    generator->clearMemory();

    bool cleared = false;
    auto n_cuts_removed = 0;

    pdqsort(SRCconstraints.begin(), SRCconstraints.end(),
            [](const baldesCtrPtr a, const baldesCtrPtr b) { return a->index() < b->index(); });

    auto it = SRCconstraints.end();
    while (it != SRCconstraints.begin()) {
        --it;
        auto constr = *it;
        int current_index = constr->index();
        double slack = node->getSlack(current_index, solution);

        if (slack > 1e-3) {
            cleared = true;
            node->remove(constr);
            cuts->removeCut(cuts->getID(std::distance(SRCconstraints.begin(), it)));
            n_cuts_removed++;
            it = SRCconstraints.erase(it);
        }
    }

    if (cuts_before == cuts->size() + n_cuts_removed) { return std::make_pair(false, cleared); }
    return std::make_pair(true, cleared);
}

/*
 * @brief Computes the limited memory coefficient for a given set of nodes.
 *
 */
double LimitedMemoryRank1Cuts::computeLimitedMemoryCoefficient(
    const std::array<uint64_t, num_words>& C,
    const std::array<uint64_t, num_words>& AM,
    const SRCPermutation& p,
    const std::vector<uint16_t>& P,
    std::vector<int>& order) {
    
    double alpha = 0.0;
    int S = 0;
    const int den = p.den;  // Cache denominator
    const auto& num = p.num;  // Cache numerator reference
    
    // Main loop with minimal bounds checking
    for (size_t j = 1; j < P.size() - 1; ++j) {
        const uint16_t vj = P[j];
        const uint32_t word_idx = vj >> 6;      // Divide by 64
        const uint64_t bit_mask = 1ULL << (vj & 63);  // Modulo 64
        
        // Check if vj is in AM using cached values
        if (!(AM[word_idx] & bit_mask)) {
            S = 0;  // Reset S if vj is not in AM
        } else if (C[word_idx] & bit_mask) {
            S += num[order[vj]];
            if (S >= den) {
                S -= den;
                alpha += 1;
            }
        }
    }
    
    return alpha;
}