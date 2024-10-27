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
 */

#pragma once

#include "Definitions.h"
#include "SparseMatrix.h"

#include "Cut.h"
#include "Pools.h"

#include "ankerl/unordered_dense.h"
#include "bucket/BucketGraph.h"

#include "../third_party/concurrentqueue.h"
#include <bitset> // If N_SIZE is large, switch back to unordered_dense_set or std::unordered_set
#include <queue>
#include <random>

#include "miphandler/MIPHandler.h"

#include <unordered_set>

#include "SRCHelper.h"
// #include "xxhash.h"

#include "RNG.h"
// include nsync
#ifdef NSYNC
extern "C" {
#include "nsync_mu.h"
}
#endif

#define VRPTW_SRC_max_S_n 10000

struct CachedCut {
    Cut    cut;
    double violation;
};

class BNBNode;

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

#include "RNG.h"

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
    std::vector<CachedCut> cutCache4;
    std::vector<CachedCut> cutCache5;

    Xoroshiro128Plus      rp; // Seed it (you can change the seed)
    HighDimCutsGenerator *generator = new HighDimCutsGenerator(N_SIZE, 5, 1e-6);

    void setDistanceMatrix(const std::vector<std::vector<double>> &distances) {
        generator->cost_mat4_vertex = distances;
        generator->generateSepHeurMem4Vertex();
    }
    LimitedMemoryRank1Cuts(std::vector<VRPNode> &nodes);

    void setDuals(const std::vector<double> &duals) {
        // print nodes.size
        for (size_t i = 1; i < N_SIZE - 1; ++i) { nodes[i].setDuals(duals[i - 1]); }
    }

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
     */
    double computeLimitedMemoryCoefficient(const std::array<uint64_t, num_words> &C,
                                           const std::array<uint64_t, num_words> &AM, const SRCPermutation &p,
                                           const std::vector<int> &P, std::vector<int> &order);

    /**
     * @brief Selects the indices of the highest coefficients from a given vector.
     *
     * This function takes a vector of doubles and an integer specifying the maximum number of nodes to select.
     * It filters out elements with coefficients less than or equal to 1e-2, sorts the remaining elements in
     * descending order based on their coefficients, and returns the indices of the top elements up to the
     * specified maximum number of nodes.
     *
     */
    std::vector<int> selectHighestCoefficients(const std::vector<double> &x, int maxNodes);

    std::vector<int> the45selectedNodes;
    void             prepare45Heuristic(const SparseMatrix &A, const std::vector<double> &x) {
        int max_important_nodes = 20;
        the45selectedNodes      = selectHighestCoefficients(x, max_important_nodes);
    }

    std::pair<bool, bool> runSeparation(BNBNode *node, std::vector<Constraint *> &SRCconstraints);

    using ViolatedCut = std::pair<double, Cut>;
    // Custom comparator to compare only the first element (the violation)
    struct CompareCuts {
        bool operator()(const ViolatedCut &a, const ViolatedCut &b) const {
            return a.first > b.first; // Min-heap: smallest violation comes first
        }
    };
    using CutPriorityQueue = std::priority_queue<ViolatedCut, std::vector<ViolatedCut>, CompareCuts>;

private:
    static std::mutex cache_mutex;

    static ankerl::unordered_dense::map<int, std::pair<std::vector<int>, std::vector<int>>> column_cache;

    std::vector<VRPNode>                                nodes;
    std::vector<std::vector<int>>                       baseSets;
    ankerl::unordered_dense::map<int, std::vector<int>> neighborSets;
    CutType                                             cutType;
};

/**
 * @brief Generates all unique permutations for a predefined set of vectors of size 5.
 *
 * This function creates permutations for a predefined set of vectors, each containing
 * five double values. The permutations are generated in lexicographical order.
 *
 */
inline std::vector<SRCPermutation> getPermutationsForSize5() {
    std::vector<SRCPermutation>   permutations;
    std::vector<std::vector<int>> p_num = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {3, 1, 1, 1, 1}, {3, 2, 2, 1, 1},
                                           {2, 2, 1, 1, 1}, {3, 3, 2, 2, 1}, {2, 2, 2, 1, 1}};
    std::vector<int>              p_dem = {2, 3, 4, 5, 4, 4, 3};
    /*
    std::vector<std::vector<double>> p_frac = {
        {1.0 / 2.0, 1.0 / 2.0, 1.0 / 2.0, 1.0 / 2.0, 1.0 / 2.0}, // {1, 1, 1, 1, 1} / 2
        {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0}, // {1, 1, 1, 1, 1} / 3
        {3.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0}, // {3, 1, 1, 1, 1} / 4
        {3.0 / 5.0, 2.0 / 5.0, 2.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0}, // {3, 2, 2, 1, 1} / 5
        {2.0 / 4.0, 2.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0}, // {2, 2, 1, 1, 1} / 4
        {3.0 / 4.0, 3.0 / 4.0, 2.0 / 4.0, 2.0 / 4.0, 1.0 / 4.0}, // {3, 3, 2, 2, 1} / 4
        {2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0}  // {2, 2, 2, 1, 1} / 3
    };
    */

    // random get 10 p_num vectors, and create 10 SRCPermutation vectors with shufle p_num[i] and the same p_dem
    // Shuffle indices to randomly access p_num and p_dem
    int              permutation_count = 10;
    Xoroshiro128Plus rng; // Seed it (you can change the seed)

    for (int i = 0; i < permutation_count; ++i) { // Get 10 random p_num vectors
        SRCPermutation perm;

        // generate random number from 0 to p_frac.size()
        auto random_index = rng() % p_num.size();

        // Use shuffled index to randomly select p_num and p_dem vectors
        // int random_index = indices[i];
        perm.num = p_num[random_index];
        perm.den = p_dem[random_index];

        // Optionally shuffle perm.num again if you want to shuffle within the vector
        std::shuffle(perm.num.begin(), perm.num.end(), rng);

        permutations.push_back(perm);
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
inline std::vector<SRCPermutation> getPermutationsForSize4() {
    std::vector<SRCPermutation>   permutations;
    std::vector<std::vector<int>> p_num = {{2, 1, 1, 1}, {1, 2, 1, 1}, {1, 1, 2, 1}, {1, 1, 1, 2}};
    int                           p_den = 3;
    // std::vector<double> p_frac = {2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};

    Xoroshiro128Plus rng; // Seed it (you can change the seed)

    // random permute p_num and create 4 SRCPermutation vectors
    for (int i = 0; i < 4; ++i) {
        SRCPermutation perm;
        perm.num = p_num[i];
        perm.den = p_den;
        // std::shuffle(perm.frac.begin(), perm.frac.end(), rng);
        //  perm.den = p_den;
        permutations.push_back(perm);
    }

    return permutations;
}

/**
 * @brief Generates all combinations of a given size from a set of elements.
 *
 * This function computes all possible combinations of `k` elements from the
 * input vector `elements` and stores them in the `result` vector.
 *
 */
template <typename T>
inline void combinations(const std::vector<T> &elements, int k, int max_combinations,
                         std::vector<std::vector<T>> &result) {
    std::vector<int> indices(k);
    std::iota(indices.begin(), indices.end(), 0);

    // Random number generator
    Xoroshiro128Plus                 rng; // Seed it (you can change the seed)
    std::uniform_real_distribution<> dis(0.0, 1.0);

    int total_combinations = 0; // Track total combinations generated

    while (true) {
        // Generate the current combination based on the selected indices
        std::vector<T> combination;
        for (int idx : indices) { combination.push_back(elements[idx]); }

        // Randomly select this combination
        ++total_combinations;
        if (result.size() < max_combinations || dis(rng) < static_cast<double>(max_combinations) / total_combinations) {
            if (result.size() < max_combinations) {
                result.push_back(combination);
            } else {
                // Replace a random element in the result if the result is already full
                std::uniform_int_distribution<> idx_dis(0, max_combinations - 1);
                result[idx_dis(rng)] = combination;
            }
        }

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
    int    max_number_of_cuts  = 2; // Max number of cuts to generate
    double violation_threshold = 1e-3;
    int    max_generated_cuts  = 30;

    auto &selectedNodes = the45selectedNodes;
    // Ensure selectedNodes is valid

    std::vector<SRCPermutation> permutations;
    if constexpr (T == CutType::FourRow) {
        permutations = getPermutationsForSize4();
    } else if constexpr (T == CutType::FiveRow) {
        permutations = getPermutationsForSize5();
    }

    // Shuffle permutations and limit to 3 for efficiency
    // int seed = std::chrono::system_clock::now().time_since_epoch().count(); // Use current time as a seed

    Xoroshiro128Plus rng; // Seed it (you can change the seed)
    if (permutations.size() > 10) {
        // Use std::shuffle with the Xoroshiro128Plus generator
        permutations.resize(10);
    }

    ankerl::unordered_dense::set<uint64_t> processedSetsCache;
    ankerl::unordered_dense::set<uint64_t> processedPermutationsCache;
    std::atomic<int>                       cuts_count(0); // Thread-safe counter for cuts

    // Create tasks for each selected node to parallelize
    std::vector<int> tasks(selectedNodes.size());
    std::iota(tasks.begin(), tasks.end(), 0); // Filling tasks with indices 0 to selectedNodes.size()

    exec::static_thread_pool pool(std::thread::hardware_concurrency());
    auto                     sched = pool.get_scheduler();

    auto input_sender = stdexec::just();
#ifdef NSYNC
    nsync::nsync_mu cuts_mutex = NSYNC_MU_INIT;
#else
    std::mutex cuts_mutex; // Protect access to shared resources
#endif
    CutPriorityQueue cutQueue;
    auto            &cutCache = (T == CutType::FourRow) ? cutCache4 : cutCache5;

    std::vector<double>                            coefficients_aux(allPaths.size(), 0.0);
    std::vector<std::tuple<int, std::vector<int>>> task_data; // To hold task_id and setsOf45 for each task

    const int chunk_size = 10; // Adjust chunk size based on performance experiments

    // Emplace tasks and prepare setsOf45 for each task
    for (auto task_id : tasks) {
        auto &consumer = allPaths[selectedNodes[task_id]].route;

        if constexpr (T == CutType::FourRow) {
            if (consumer.size() < 4) { continue; } // Skip if not enough elements for CutType::FourRow
        } else if constexpr (T == CutType::FiveRow) {
            if (consumer.size() < 5) { continue; } // Skip if not enough elements for CutType::FiveRow
        }

        std::vector<std::vector<int>> setsOf45;

        constexpr int combination_size = (T == CutType::FourRow) ? 4 : 5;
        combinations(consumer, combination_size, 20, setsOf45);

        for (auto set45 : setsOf45) {
            // Emplace the task id and its setsOf45 into the task_data vector
            task_data.emplace_back(task_id, set45);
        }
    }

    // Parallel processing for setsOf45
    auto bulk_sender = stdexec::bulk(
        stdexec::just(), (task_data.size() + chunk_size - 1) / chunk_size, // Calculate number of chunks
        [this, &permutations, &processedSetsCache, &processedPermutationsCache, &cuts_mutex, &x, &selectedNodes,
         &cutQueue, &max_number_of_cuts, &max_generated_cuts, violation_threshold, &task_data, &cuts_count,
         &rng](std::size_t chunk_idx) {
            // Calculate the start and end index for this chunk
            size_t start_idx = chunk_idx * chunk_size;
            size_t end_idx   = std::min(start_idx + chunk_size, task_data.size());

            // Process each task in the chunk
            for (size_t idx = start_idx; idx < end_idx; ++idx) {
                auto [task_id, set45] = task_data[idx];
                std::vector<double> coefficients_aux(allPaths.size(), 0.0);

                uint64_t setHash = hashVector(set45);

                {
                    std::lock_guard<std::mutex> cache_lock(cuts_mutex);
                    if (processedSetsCache.find(setHash) != processedSetsCache.end()) {
                        return; // Skip already processed sets
                    }
                    processedSetsCache.insert(setHash);
                }
                std::array<uint64_t, num_words> AM      = {};
                std::array<uint64_t, num_words> baseSet = {};
                std::vector<int>                order(N_SIZE, 0);

                for (const auto &p : permutations) {
                    auto p_cat = p.num;
                    p_cat.push_back(p.den);
                    uint64_t pHash = hashVector(p_cat);
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
                        order[node] = ordering;
                        ordering++;
                    }

                    std::fill(coefficients_aux.begin(), coefficients_aux.end(), 0.0);
                    int    rhs             = p.getRHS();
                    double alpha           = 0;
                    bool   violation_found = false;

                    for (auto j = 0; j < selectedNodes.size(); ++j) {
                        // if (selectedNodes[j] == selectedNodes[task_idx]) {}
                        auto &consumer_inner = allPaths[selectedNodes[j]].route;

                        int              max_limit   = (T == CutType::FourRow) ? 3 : 4;
                        int              match_count = 0;
                        std::vector<int> match_indices;

                        for (int i = 0; i < consumer_inner.size(); ++i) {
                            if (std::ranges::find(set45, consumer_inner[i]) != set45.end()) {
                                match_indices.push_back(i);
                                if (++match_count == max_limit) { break; }
                            }
                        }

                        if (match_count < max_limit) continue; // Skip if not enough matches

                        // Extract elements between the first and last match indices
                        int start_idx = match_indices.front();
                        int end_idx   = match_indices.back();

                        for (int i = start_idx; i <= end_idx; ++i) {
                            AM[consumer_inner[i] >> 6] |= (1ULL << (consumer_inner[i] & 63));
                        }
                        for (auto c : set45) { AM[c >> 6] |= (1ULL << (c & 63)); }

                        double alpha_inner = computeLimitedMemoryCoefficient(baseSet, AM, p, consumer_inner, order);
                        alpha += alpha_inner;

                        coefficients_aux[selectedNodes[j]] = alpha_inner;

                        if (alpha > rhs + violation_threshold) { violation_found = true; }

                        if (violation_found) {
                            // set setOf45 to AM
                            for (auto c : set45) { AM[c >> 6] |= (1ULL << (c & 63)); }

                            // Recalculate alpha after applying the minimal set of nodes
                            alpha = 0;
                            for (auto j = 0; j < selectedNodes.size(); ++j) {
                                const auto &consumer_inner = allPaths[selectedNodes[j]].route;
                                alpha += computeLimitedMemoryCoefficient(baseSet, AM, p, consumer_inner, order);
                            }

                            // compute coefficients_aux for all the other nodes
                            for (auto k = 0; k < allPaths.size(); k++) {
                                if (k == selectedNodes[j]) continue;
                                auto &consumer_inner = allPaths[k];

                                std::vector<int> thePath(consumer_inner.begin(), consumer_inner.end());

                                double alpha_inner  = computeLimitedMemoryCoefficient(baseSet, AM, p, thePath, order);
                                coefficients_aux[k] = alpha_inner;
                            }

                            Cut cut(baseSet, AM, coefficients_aux, p);
                            cut.baseSetOrder = order;
                            cut.rhs          = rhs;

                            ViolatedCut vCut{alpha, cut}; // Pair: first is the violation, second is the cut
#ifndef NSYNC
                            std::lock_guard<std::mutex> cut_lock(cuts_mutex);
#endif
                            {
#ifdef NSYNC
                                nsync::nsync_mu_lock(&cuts_mutex);
#endif
                                if (cutQueue.size() < max_number_of_cuts) {
                                    cutQueue.push(vCut);
                                } else if (cutQueue.top().first < vCut.first) { // Compare violations
                                    cutQueue.pop();                             // Remove the least violated cut
                                    cutQueue.push(vCut);
                                }
// CachedCut newCachedCut{cut, alpha};
// cutCache.push_back(newCachedCut);
#ifdef NSYNC
                                nsync::nsync_mu_unlock(&cuts_mutex);
#endif
                            }
                            // increment the cuts_count
                            cuts_count.fetch_add(1);

                            if (cuts_count.load() > max_generated_cuts) { return; }
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
