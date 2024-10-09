/**
 * @file SRC.h
 * @brief Definitions for handling cuts and optimizations for the Vehicle Routing Problem with Time Windows (VRPTW).
 *
 * This header file contains the structure and class definitions required for the limited memory rank-1 cuts,
 * including the handling of cuts for optimization algorithms used in the Vehicle Routing Problem with Time Windows
 * (VRPTW).
 *
 *
 * It also includes utility functions to compute coefficients, generate cuts, and work with sparse models.
 * The file facilitates the optimization process by allowing computation of limited memory coefficients and
 * the generation of cuts via heuristics. The file makes use of Gurobi for handling constraints in the solver.
 *
 * @note Several methods are optimized for parallel execution using thread pools.
 */

#pragma once

#include "Definitions.h"
#include "SparseMatrix.h"

#include "Cut.h"

#include "Pools.h"

#include "ankerl/unordered_dense.h"
#include "bucket/BucketGraph.h"

#include <queue>
#include <random>

#include <unordered_set>

#include "xxhash.h"

#define VRPTW_SRC_max_S_n 10000

/**
 * @struct VRPTW_SRC
 * @brief A structure to represent the Vehicle Routing Problem with Time Windows (VRPTW) source data.
 *
 * This structure holds various vectors and integers that are used in the context of solving the VRPTW.
 */
struct VRPTW_SRC {
    std::vector<int>                    S;
    std::vector<int>                    S_C_P;
    int                                 S_C_P_max;
    int                                 S_n;
    std::vector<std::pair<double, int>> best_sets;
    int                                 S_n_max = 10000;
};

struct VectorStringHash;
struct UnorderedSetStringHash;
struct PairHash;

struct SparseMatrix;

/**
 * @class LimitedMemoryRank1Cuts
 * @brief A class for handling limited memory rank-1 cuts in optimization problems.
 *
 * This class provides methods for separating cuts by enumeration, generating cut coefficients,
 * and computing limited memory coefficients. It also includes various utility functions for
 * managing and printing base sets, inserting sets, and performing heuristics.
 *
 */
class LimitedMemoryRank1Cuts {
public:
    LimitedMemoryRank1Cuts(std::vector<VRPNode> &nodes);

    // default constructor
    LimitedMemoryRank1Cuts() = default;

    CutStorage cutStorage = CutStorage();

    void              printBaseSets();
    std::vector<Path> allPaths;

    std::vector<std::vector<int>>    labels;
    int                              labels_counter = 0;
    std::vector<std::vector<double>> separate(const SparseMatrix &A, const std::vector<double> &x);
    void insertSet(VRPTW_SRC &cuts, int i, int j, int k, const std::vector<int> &buffer_int, int buffer_int_n,
                   double LHS_cut);

    void generateCutCoefficients(VRPTW_SRC &cuts, std::vector<std::vector<double>> &coefficients, int numNodes,
                                 const SparseMatrix &A, const std::vector<double> &x);

    template <CutType T>
    void the45Heuristic(const SparseMatrix &A, const std::vector<double> &x);

    /**
     * @brief Computes the limited memory coefficient based on the given parameters.
     *
     * This function calculates the coefficient by iterating through the elements of the vector P,
     * checking their presence in the bitwise arrays C and AM, and updating the coefficient based on
     * the values in the vector p and the order vector.
     *
     * @param C A constant reference to an array of uint64_t representing the bitwise array C.
     * @param AM A constant reference to an array of uint64_t representing the bitwise array AM.
     * @param p A constant reference to a vector of doubles representing the values associated with each position.
     * @param P A constant reference to a vector of integers representing the positions to be checked.
     * @param order A reference to a vector of integers representing the order of positions in C.
     * @return A double representing the computed limited memory coefficient.
     */
    double computeLimitedMemoryCoefficient(const std::array<uint64_t, num_words> &C,
                                           const std::array<uint64_t, num_words> &AM, const std::vector<double> &p,
                                           const std::vector<int> &P, std::vector<int> &order) {
        double alpha = 0;
        double S     = 0;

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

                S += p[pos];

                if (S >= 1) {
                    S -= 1;
                    alpha += 1;
                }
            }
        }

        return alpha;
    }

    /**
     * @brief Selects the indices of the highest coefficients from a given vector.
     *
     * This function takes a vector of doubles and an integer specifying the maximum number of nodes to select.
     * It filters out elements with coefficients less than or equal to 1e-2, sorts the remaining elements in
     * descending order based on their coefficients, and returns the indices of the top elements up to the
     * specified maximum number of nodes.
     *
     */
    inline std::vector<int> selectHighestCoefficients(const std::vector<double> &x, int maxNodes) {
        std::vector<std::pair<int, double>> nodeCoefficients;
        for (int i = 0; i < x.size(); ++i) {
            if (x[i] > 1e-2) { nodeCoefficients.push_back({i, x[i]}); }
        }

        // Sort nodes by coefficient in descending order
        std::sort(nodeCoefficients.begin(), nodeCoefficients.end(),
                  [](const auto &a, const auto &b) { return a.second > b.second; });

        std::vector<int> selectedNodes;
        for (int i = 0; i < std::min(maxNodes, (int)nodeCoefficients.size()); ++i) {
            selectedNodes.push_back(nodeCoefficients[i].first);
        }

        return selectedNodes;
    }

    std::vector<int> the45selectedNodes;
    void             prepare45Heuristic(const SparseMatrix &A, const std::vector<double> &x) {
        int max_important_nodes = 5;
        the45selectedNodes      = selectHighestCoefficients(x, max_important_nodes);
    }

private:
    static std::mutex cache_mutex;

    static ankerl::unordered_dense::map<int, std::pair<std::vector<int>, std::vector<int>>> column_cache;

    std::vector<VRPNode>                                nodes;
    std::vector<std::vector<int>>                       baseSets;
    ankerl::unordered_dense::map<int, std::vector<int>> neighborSets;
    CutType                                             cutType;

    // Function to create a unique key from an unordered set of strings
    std::string createKey(const ankerl::unordered_dense::set<std::string> &set);

    int alpha(const std::vector<int> &C, const std::vector<int> &M, const std::vector<double> &p,
              const std::vector<int> &r);
};

/**
 * @brief Generates all unique permutations for a predefined set of vectors of size 5.
 *
 * This function creates permutations for a predefined set of vectors, each containing
 * five double values. The permutations are generated in lexicographical order.
 *
 */
inline std::vector<std::vector<double>> getPermutationsForSize5() {
    std::vector<std::vector<double>> permutations;
    std::vector<std::vector<double>> p_all = {{0.5, 0.5, 0.25, 0.25, 0.25},
                                              {0.75, 0.25, 0.25, 0.25, 0.25},
                                              {0.6, 0.4, 0.4, 0.2, 0.2},
                                              {0.6667, 0.6667, 0.3333, 0.3333, 0.3333},
                                              {0.75, 0.75, 0.5, 0.5, 0.25}};

    for (auto &p : p_all) {
        std::sort(p.begin(), p.end()); // Ensure we start with the lowest lexicographical order
        do { permutations.push_back(p); } while (std::next_permutation(p.begin(), p.end()));
    }
    return permutations;
}

/**
 * @brief Generates all unique permutations of a fixed-size vector.
 *
 * This function generates all unique permutations of a vector of size 4
 * containing the elements {2.0/3.0, 1.0/3.0, 1.0/3.0, 1.0/3.0}. The permutations
 * are generated in lexicographical order.
 *
 */
inline std::vector<std::vector<double>> getPermutationsForSize4() {
    std::vector<std::vector<double>> permutations;
    std::vector<double>              p = {2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};
    std::sort(p.begin(), p.end()); // Ensure we start with the lowest lexicographical order
    do { permutations.push_back(p); } while (std::next_permutation(p.begin(), p.end()));
    return permutations;
}

/**
 * @brief Generates all combinations of a given size from a set of elements.
 *
 * This function computes all possible combinations of `k` elements from the
 * input vector `elements` and stores them in the `result` vector.
 *
 */
/*
template <typename T>
inline void combinations(const std::vector<T> &elements, int k, std::vector<std::vector<T>> &result) {
    std::vector<bool> v(elements.size());
    std::fill(v.begin(), v.begin() + k, true);
    do {
        std::vector<T> combination;
        for (size_t i = 0; i < elements.size(); ++i) {
            if (v[i]) { combination.push_back(elements[i]); }
        }
        result.push_back(combination);
    } while (std::prev_permutation(v.begin(), v.end()));
}
*/
template <typename T>
inline void combinations(const std::vector<T> &elements, int k, std::vector<std::vector<T>> &result) {
    std::vector<int> indices(k);
    std::iota(indices.begin(), indices.end(), 0);

    while (true) {
        // Generate the current combination based on the selected indices
        std::vector<T> combination;
        for (int idx : indices) { combination.push_back(elements[idx]); }
        result.push_back(combination);

        // Find the rightmost index that can be incremented
        int i = k - 1;
        while (i >= 0 && indices[i] == elements.size() - k + i) { --i; }

        // If no such index exists, all combinations are generated
        if (i < 0) break;

        // Increment this index
        ++indices[i];

        // Update the following indices
        for (int j = i + 1; j < k; ++j) { indices[j] = indices[j - 1] + 1; }
    }
}

/**
 * @brief Finds the visiting nodes in a sparse model based on selected nodes.
 *
 * This function processes a sparse model to identify and return the visiting nodes
 * for the given selected nodes. It filters the columns of the sparse model and
 * returns only those that are relevant to the selected nodes.
 *
 */
inline std::vector<std::vector<int>> findVisitingNodes(const SparseMatrix &A, const std::vector<int> &selectedNodes) {
    std::vector<std::vector<int>>     consumers;
    ankerl::unordered_dense::set<int> selectedNodeSet(selectedNodes.begin(), selectedNodes.end());

    // Reserve space for consumers to prevent multiple allocations
    consumers.reserve(selectedNodes.size());

    // Preprocess the selected columns
    std::vector<std::vector<int>> col_to_rows(A.num_cols);
    for (const auto &elem : A.elements) {
        // print A.elements' value
        // fmt::print("A.elements.value: {}\n", elem.value);

        if (elem.value == -1) { col_to_rows[elem.col].push_back(elem.row); }
    }

    // Filter only the selected nodes
    for (int col : selectedNodes) {
        if (!col_to_rows[col].empty()) { consumers.push_back(std::move(col_to_rows[col])); }
    }

    return consumers;
}

// Hash function for a vector of integers
inline uint64_t hashVector(const std::vector<int> &vec) {
    uint64_t hash = 0; // Initialize seed for XXHash

    for (const int &elem : vec) {
        // Combine each element's hash using XXHash and XOR it with the running hash value
        hash ^= XXH3_64bits_withSeed(&elem, sizeof(elem), hash);
    }

    return hash;
}

// Hash function for a vector of doubles
inline uint64_t hashVector(const std::vector<double> &vec) {
    uint64_t hash = 0; // Initialize seed for XXHash

    for (const double &elem : vec) {
        // Convert the double to uint64_t for hashing
        std::uint64_t bit_rep = std::bit_cast<std::uint64_t>(elem);
        // Combine each element's hash using XXHash and XOR it with the running hash value
        hash ^= XXH3_64bits_withSeed(&bit_rep, sizeof(bit_rep), hash);
    }

    return hash;
}
using ViolatedCut = std::pair<double, Cut>;

// Custom comparator to compare only the first element (the violation)
struct CompareCuts {
    bool operator()(const ViolatedCut &a, const ViolatedCut &b) const {
        return a.first < b.first; // Min-heap: smallest violation comes first
    }
};

/**
 * @brief Implements the 45 Heuristic for generating limited memory rank-1 cuts.
 *
 * This function generates limited memory rank-1 cuts using a heuristic approach.
 * It processes sets of customers and permutations to identify violations and generate cuts.
 * The function is parallelized to improve efficiency.
 *
 */
template <CutType T>
void LimitedMemoryRank1Cuts::the45Heuristic(const SparseMatrix &A, const std::vector<double> &x) {
    int    max_number_of_cuts  = 5; // Max number of cuts to generate
    double violation_threshold = 1e-3;
    int    max_generated_cuts  = 15;

    auto &selectedNodes = the45selectedNodes;
    // Ensure selectedNodes is valid

    std::vector<std::vector<double>> permutations;
    if constexpr (T == CutType::FourRow) {
        permutations = getPermutationsForSize4();
    } else if constexpr (T == CutType::FiveRow) {
        permutations = getPermutationsForSize5();
    }

    // Shuffle permutations and limit to 3 for efficiency
    std::random_device rd;
    std::mt19937       g(rd());
    if (permutations.size() > 4) {
        std::shuffle(permutations.begin(), permutations.end(), g);
        permutations.resize(4);
    }

    ankerl::unordered_dense::set<uint64_t> processedSetsCache;
    ankerl::unordered_dense::set<uint64_t> processedPermutationsCache;
    std::mutex                             cuts_mutex;    // Protect access to shared resources
    std::atomic<int>                       cuts_count(0); // Thread-safe counter for cuts

    // Create tasks for each selected node to parallelize
    std::vector<int> tasks(selectedNodes.size());
    std::iota(tasks.begin(), tasks.end(), 0); // Filling tasks with indices 0 to selectedNodes.size()

    exec::static_thread_pool pool(std::thread::hardware_concurrency());
    auto                     sched = pool.get_scheduler();

    auto input_sender = stdexec::just();

    using CutPriorityQueue = std::priority_queue<ViolatedCut, std::vector<ViolatedCut>, CompareCuts>;

    CutPriorityQueue cutQueue;

    // Define the bulk operation to process sets of customers
    auto bulk_sender = stdexec::bulk(
        input_sender, tasks.size(),
        [this, &permutations, &processedSetsCache, &processedPermutationsCache, &cuts_mutex, &cuts_count, &x,
         &selectedNodes, &cutQueue, &max_number_of_cuts, &max_generated_cuts,
         violation_threshold](std::size_t task_idx) {
            std::vector<double> coefficients_aux(x.size(), 0.0);

            auto &consumer = allPaths[selectedNodes[task_idx]].route;

            if constexpr (T == CutType::FourRow) {
                if (consumer.size() < 4) { return; }
            } else if constexpr (T == CutType::FiveRow) {
                if (consumer.size() < 5) { return; }
            }

            std::vector<std::vector<int>> setsOf45;
            if constexpr (T == CutType::FourRow) {
                combinations(consumer, 4, setsOf45);
            } else if constexpr (T == CutType::FiveRow) {
                combinations(consumer, 5, setsOf45);
            }

            // limit the number of sets to process
            if (setsOf45.size() > 2000) { setsOf45.resize(2000); }

            std::vector<Cut> threadCuts;

            for (const auto &set45 : setsOf45) {

                uint64_t setHash = hashVector(set45);
                {
                    std::lock_guard<std::mutex> cache_lock(cuts_mutex);
                    if (processedSetsCache.find(setHash) != processedSetsCache.end()) {
                        continue; // Skip already processed sets
                    }
                    processedSetsCache.insert(setHash);
                }
                std::array<uint64_t, num_words> AM      = {};
                std::array<uint64_t, num_words> baseSet = {};
                std::vector<int>                order(N_SIZE, 0);

                for (const auto &p : permutations) {
                    uint64_t pHash = hashVector(p);
                    // concatenate processed set and permutation
                    uint64_t setPermutationHash = setHash ^ pHash;

                    {
                        std::lock_guard<std::mutex> cache_lock(cuts_mutex);
                        if (processedPermutationsCache.find(setPermutationHash) != processedPermutationsCache.end()) {
                            continue; // Skip already processed permutations
                        }
                        processedPermutationsCache.insert(setPermutationHash);
                    }

                    AM.fill(0);
                    baseSet.fill(0);
                    std::fill(order.begin(), order.end(), 0);

                    for (auto c : set45) { baseSet[c >> 6] |= (1ULL << (c & 63)); }
                    int ordering = 0;
                    for (auto node : set45) {
                        AM[node >> 6] |= (1ULL << (node & 63)); // Set the bit for node in AM
                        order[node] = ordering++;
                    }

                    std::fill(coefficients_aux.begin(), coefficients_aux.end(), 0.0);
                    int    rhs             = std::floor(std::accumulate(p.begin(), p.end(), 0.0));
                    double alpha           = 0;
                    bool   violation_found = false;

                    for (auto j = 0; j < selectedNodes.size(); ++j) {
                        if (selectedNodes[j] == selectedNodes[task_idx]) {}
                        auto &consumer_inner = allPaths[selectedNodes[j]].route;

                        int max_limit = (T == CutType::FourRow) ? 3 : 4;

                        int match_count = 0;
                        for (auto &node : set45) {
                            if (std::ranges::find(consumer_inner, node) != consumer_inner.end()) {
                                if (++match_count == max_limit) { break; }
                            }
                        }

                        if (match_count < max_limit) continue;

                        for (auto c : consumer_inner) { AM[c >> 6] |= (1ULL << (c & 63)); }

                        double alpha_inner = computeLimitedMemoryCoefficient(baseSet, AM, p, consumer_inner, order);
                        alpha += alpha_inner;

                        coefficients_aux[selectedNodes[j]] = alpha_inner;

                        if (alpha > rhs + violation_threshold) { violation_found = true; }

                        if (violation_found) {
                            for (int i = 1; i < N_SIZE - 2; ++i) {
                                // Skip nodes that are part of baseSet (i.e., cannot be removed from AM)
                                if (!(baseSet[i >> 6] & (1ULL << (i & 63)))) {

                                    // Check if the node is currently in AM
                                    if (AM[i >> 6] & (1ULL << (i & 63))) {

                                        // Temporarily remove node i from AM
                                        uint64_t tempAM = AM[i >> 6];
                                        AM[i >> 6] &= ~(1ULL << (i & 63));

                                        // Use a local variable for the reduced alpha to avoid data races
                                        double reduced_alpha = 0;
                                        for (auto j = 0; j < selectedNodes.size(); ++j) {
                                            auto &consumer_inner = allPaths[selectedNodes[j]].route;
                                            reduced_alpha +=
                                                computeLimitedMemoryCoefficient(baseSet, AM, p, consumer_inner, order);
                                        }

                                        // If the violation no longer holds, restore the node in AM
                                        if (reduced_alpha <= rhs + violation_threshold) {
                                            AM[i >> 6] = tempAM; // Restore node i in AM
                                        } else {
                                            // The violation still holds, update alpha to reduced_alpha
                                            alpha = reduced_alpha;
                                        }
                                    }
                                }
                            }

                            // compute coefficients_aux for all the other nodes
                            for (auto k = 0; k < allPaths.size(); k++) {
                                if (k == selectedNodes[j]) continue;
                                auto &consumer_inner = allPaths[k];

                                int max_limit = 0;
                                if constexpr (T == CutType::FourRow) {
                                    max_limit = 3;
                                } else if constexpr (T == CutType::FiveRow) {
                                    max_limit = 4;
                                }
                                int match_count = 0;
                                for (auto &node : set45) {
                                    if (std::ranges::find(consumer_inner, node) != consumer_inner.end()) {
                                        if (++match_count == max_limit) { break; }
                                    }
                                }

                                if (match_count < max_limit) continue;

                                std::vector<int> thePath(consumer_inner.begin(), consumer_inner.end());

                                double alpha_inner  = computeLimitedMemoryCoefficient(baseSet, AM, p, thePath, order);
                                coefficients_aux[k] = alpha_inner;
                            }

                            Cut cut(baseSet, AM, coefficients_aux, p);
                            cut.baseSetOrder = order;
                            cut.rhs          = rhs;

                            ViolatedCut vCut{alpha, cut}; // Pair: first is the violation, second is the cut

                            std::lock_guard<std::mutex> cut_lock(cuts_mutex);
                            if (cutQueue.size() < max_number_of_cuts) {
                                cutQueue.push(vCut);
                            } else if (cutQueue.top().first < vCut.first) { // Compare violations
                                cutQueue.pop();                             // Remove the least violated cut
                                cutQueue.push(vCut);
                            }
                            if (cuts_count.load() > max_generated_cuts) { break; }
                        }
                    }
                }
            }
        });

    auto work = stdexec::starts_on(sched, bulk_sender);
    stdexec::sync_wait(std::move(work));

    while (!cutQueue.empty()) {
        auto topCut = cutQueue.top();
        cutStorage.addCut(topCut.second);
        cutQueue.pop();
    }
}
