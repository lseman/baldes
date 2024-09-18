/**
 * @file SRC.h
 * @brief Definitions for handling cuts and optimizations for the Vehicle Routing Problem with Time Windows (VRPTW).
 *
 * This header file contains the structure and class definitions required for the limited memory rank-1 cuts,
 * including the handling of cuts for optimization algorithms used in the Vehicle Routing Problem with Time Windows (VRPTW).
 *
 * The file defines the following key structures and classes:
 * 1. `VRPTW_SRC`: Holds the state and best sets for processing cuts.
 * 2. `Cut`: Represents an individual cut with coefficients and multipliers.
 * 3. `CutStorage`: Manages storage and operations related to cuts in the solver.
 * 4. `LimitedMemoryRank1Cuts`: Provides methods for separating and generating cuts, including heuristics like the 45 Heuristic.
 *
 * It also includes utility functions to compute coefficients, generate cuts, and work with sparse models. 
 * The file facilitates the optimization process by allowing computation of limited memory coefficients and 
 * the generation of cuts via heuristics. The file makes use of Gurobi for handling constraints in the solver.
 *
 * @note Several methods are optimized for parallel execution using thread pools.
 */

#pragma once

#include "../include/BucketGraph.h"
#include "../include/Definitions.h"

#include <random>

#define VRPTW_SRC_max_S_n 10000

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

enum class CutType { ThreeRow, FourRow, FiveRow };


/**
 * @struct Cut
 * @brief Represents a cut in the optimization problem.
 * 
 * The Cut structure holds information about a specific cut, including its base set,
 * neighbors, coefficients, multipliers, and other properties.
 * 
 * @var Cut::cutMaster
 * Integer representing the master cut.
 * 
 * @var Cut::baseSet
 * Bit-level base set represented as an array of uint64_t.
 * 
 * @var Cut::neighbors
 * Bit-level neighbors represented as an array of uint64_t.
 * 
 * @var Cut::baseSetOrder
 * Order for the base set represented as a vector of integers.
 * 
 * @var Cut::coefficients
 * Cut coefficients represented as a vector of doubles.
 * 
 * @var Cut::multipliers
 * Multipliers for the cut, initialized to {0.5, 0.5, 0.5}.
 * 
 * @var Cut::rhs
 * Right-hand side value of the cut, initialized to 1.
 * 
 * @var Cut::id
 * Identifier for the cut, initialized to -1.
 * 
 * @var Cut::added
 * Boolean flag indicating whether the cut has been added, initialized to false.
 * 
 * @var Cut::updated
 * Boolean flag indicating whether the cut has been updated, initialized to false.
 * 
 * @var Cut::type
 * Type of the cut, initialized to CutType::ThreeRow.
 * 
 * @var Cut::grbConstr
 * Gurobi constraint associated with the cut.
 * 
 * @fn Cut::Cut()
 * Default constructor.
 * 
 * @fn Cut::Cut(const std::array<uint64_t, num_words> baseSetInput, const std::array<uint64_t, num_words> &neighborsInput, const std::vector<double> &coefficients)
 * Constructor that initializes the base set, neighbors, and coefficients.
 * 
 * @fn Cut::Cut(const std::array<uint64_t, num_words> baseSetInput, const std::array<uint64_t, num_words> &neighborsInput, const std::vector<double> &coefficients, const std::vector<double> &multipliers)
 * Constructor that initializes the base set, neighbors, coefficients, and multipliers.
 * 
 * @fn size_t Cut::size() const
 * Returns the size of the cut, which is the size of the coefficients vector.
 * 
 */
struct Cut {
    int                             cutMaster;
    std::array<uint64_t, num_words> baseSet;      // Bit-level baseSet
    std::array<uint64_t, num_words> neighbors;    // Bit-level neighbors
    std::vector<int>                baseSetOrder; // Order for baseSet
    std::vector<double>             coefficients; // Cut coefficients
    std::vector<double>             multipliers = {0.5, 0.5, 0.5};
    double                          rhs         = 1;
    int                             id          = -1;
    bool                            added       = false;
    bool                            updated     = false;
    CutType                         type        = CutType::ThreeRow;
    GRBConstr                       grbConstr;

    // Default constructor
    Cut() = default;

    // constructor to receive array
    Cut(const std::array<uint64_t, num_words> baseSetInput, const std::array<uint64_t, num_words> &neighborsInput,
        const std::vector<double> &coefficients)
        : baseSet(baseSetInput), neighbors(neighborsInput), coefficients(coefficients) {}

    Cut(const std::array<uint64_t, num_words> baseSetInput, const std::array<uint64_t, num_words> &neighborsInput,
        const std::vector<double> &coefficients, const std::vector<double> &multipliers)
        : baseSet(baseSetInput), neighbors(neighborsInput), coefficients(coefficients), multipliers(multipliers) {}

    // Define size of the cut
    size_t size() const { return coefficients.size(); }
};

using Cuts = std::vector<Cut>;

/**
 * @class CutStorage
 * @brief Manages the storage and operations related to cuts in a solver.
 * 
 * The CutStorage class provides functionalities to add, manage, and query cuts.
 * It also allows setting dual values and computing coefficients with limited memory.
 * 
 * @var latest_column
 * The latest column index.
 * 
 * @var SRCDuals
 * A vector storing the dual values.
 * 
 * @fn void addCut(Cut &cut)
 * @brief Adds a cut to the storage.
 * @param cut The cut to be added.
 * 
 * @fn void setDuals(const std::vector<double> &duals)
 * @brief Sets the dual values.
 * @param duals A vector of dual values.
 * 
 * @fn size_t size() const noexcept
 * @brief Returns the number of cuts in the storage.
 * @return The number of cuts.
 * 
 * @fn auto begin() const noexcept
 * @brief Returns a const iterator to the beginning of the cuts.
 * @return A const iterator to the beginning.
 * 
 * @fn auto end() const noexcept
 * @brief Returns a const iterator to the end of the cuts.
 * @return A const iterator to the end.
 * 
 * @fn auto begin() noexcept
 * @brief Returns an iterator to the beginning of the cuts.
 * @return An iterator to the beginning.
 * 
 * @fn auto end() noexcept
 * @brief Returns an iterator to the end of the cuts.
 * @return An iterator to the end.
 * 
 * @fn bool empty() const noexcept
 * @brief Checks if the storage is empty.
 * @return True if the storage is empty, false otherwise.
 * 
 * @fn std::pair<int, std::vector<double>> cutExists(const std::size_t &cut_key) const
 * @brief Checks if a cut exists in the storage.
 * @param cut_key The key of the cut to check.
 * @return A pair containing the size of the cut and its coefficients if it exists, otherwise {-1, {}}.
 * 
 * @fn auto getCtr(int i) const
 * @brief Retrieves the constraint associated with a cut.
 * @param i The index of the cut.
 * @return The constraint associated with the cut.
 * 
 * @fn auto computeLimitedMemoryCoefficients(const std::vector<int> &P)
 * @brief Computes coefficients with limited memory.
 * @param P A vector of integers used in the computation.
 * @return A vector of computed coefficients.
 * 
 * @fn std::size_t generateCutKey(const int &cutMaster, const std::vector<bool> &baseSetStr) const
 * @brief Generates a key for a cut.
 * @param cutMaster The master identifier for the cut.
 * @param baseSetStr A vector representing the base set structure.
 * @return The generated key.
 * 
 * @var cutMaster_to_cut_map
 * A map from cut keys to their indices in the cuts vector.
 * 
 * @var cuts
 * A collection of cuts.
 * 
 * @var indexCuts
 * A map from cut keys to vectors of indices.
 */
class CutStorage {
public:
    int latest_column = 0;

    // Add a cut to the storage
    void addCut(Cut &cut);

    std::vector<double> SRCDuals = {};

    void setDuals(const std::vector<double> &duals) { SRCDuals = duals; }

    // Define size method
    size_t size() const noexcept { return cuts.size(); }

    // Define begin and end
    auto begin() const noexcept { return cuts.begin(); }
    auto end() const noexcept { return cuts.end(); }
    auto begin() noexcept { return cuts.begin(); }
    auto end() noexcept { return cuts.end(); }

    // Define empty method
    bool empty() const noexcept { return cuts.empty(); }

    /**
     * @brief Checks if a cut exists for the given cut key and returns its size and coefficients.
     *
     * This function searches for the specified cut key in the cutMaster_to_cut_map. If the cut key
     * is found, it retrieves the size and coefficients of the corresponding cut from the cuts vector.
     * If the cut key is not found, it returns a pair with -1 and an empty vector.
     *
     * @param cut_key The key of the cut to search for.
     * @return A pair where the first element is the size of the cut (or -1 if not found) and the second
     *         element is a vector of coefficients (empty if not found).
     */
    std::pair<int, std::vector<double>> cutExists(const std::size_t &cut_key) const {
        auto it = cutMaster_to_cut_map.find(cut_key);
        if (it != cutMaster_to_cut_map.end()) {
            auto tam    = cuts[it->second].size();
            auto coeffs = cuts[it->second].coefficients;
            return {tam, coeffs};
        }
        return {-1, {}};
    }

    /**
     * @brief Retrieves the constraint at the specified index.
     * 
     * This function returns the Gurobi constraint object associated with the 
     * cut at the given index.
     * 
     * @param i The index of the cut whose constraint is to be retrieved.
     * @return The Gurobi constraint object at the specified index.
     */
    auto getCtr(int i) const { return cuts[i].grbConstr; }

    /**
     * @brief Computes limited memory coefficients for a given set of cuts.
     *
     * This function iterates over a collection of cuts and computes a set of coefficients
     * based on the provided vector P. The computation involves checking membership of nodes
     * in specific sets and updating coefficients accordingly.
     *
     * @param P A vector of integers representing the nodes to be processed.
     * @return A vector of doubles containing the computed coefficients for each cut.
     */
    auto computeLimitedMemoryCoefficients(const std::vector<int> &P) {
        // iterate over cuts
        std::vector<double> alphas;
        alphas.reserve(cuts.size());
        for (auto c : cuts) {
            double alpha = 0;
            double S     = 0;
            auto   AM    = c.neighbors;
            auto   C     = c.baseSet;
            auto   p     = c.multipliers;
            auto   order = c.baseSetOrder;

            for (size_t j = 1; j < P.size() - 1; ++j) {
                int vj = P[j];

                // Check if the node vj is in AM (bitwise check)
                if (!(AM[vj / 64] & (1ULL << (vj % 64)))) {
                    S = 0; // Reset S if vj is not in AM
                } else if (C[vj / 64] & (1ULL << (vj % 64))) {
                    // Get the position of vj in C by counting the set bits up to vj
                    int pos = order[vj];
                    S += p[pos];
                    if (S >= 1) {
                        S -= 1;
                        alpha += 1;
                    }
                }
            }

            alphas.push_back(alpha);
        }
        return alphas;
    }

    std::size_t generateCutKey(const int &cutMaster, const std::vector<bool> &baseSetStr) const;

private:
    std::unordered_map<std::size_t, int>              cutMaster_to_cut_map;
    Cuts                                              cuts;
    std::unordered_map<std::size_t, std::vector<int>> indexCuts;
};

struct SparseModel;

/**
 * @class LimitedMemoryRank1Cuts
 * @brief A class for handling limited memory rank-1 cuts in optimization problems.
 *
 * This class provides methods for separating cuts by enumeration, generating cut coefficients,
 * and computing limited memory coefficients. It also includes various utility functions for
 * managing and printing base sets, inserting sets, and performing heuristics.
 *
 * @public
 * @fn std::vector<std::vector<double>> separateByEnumeration(const SparseModel &A, const std::vector<double> &x, int nC, double violation_threshold)
 * @brief Separates cuts by enumeration.
 * @param A The sparse model.
 * @param x The vector of variables.
 * @param nC The number of constraints.
 * @param violation_threshold The threshold for violation.
 * @return A vector of vectors containing the separated cuts.
 *
 */
class LimitedMemoryRank1Cuts {
public:
    std::vector<std::vector<double>> separateByEnumeration(const SparseModel &A, const std::vector<double> &x, int nC,
                                                           double violation_threshold);
    LimitedMemoryRank1Cuts(std::vector<VRPJob> &jobs, CutType cutType);

    CutStorage cutStorage;

    void              printBaseSets();
    std::vector<Path> allPaths;

    std::vector<std::vector<int>>    labels;
    int                              labels_counter = 0;
    std::vector<std::vector<double>> separate(const SparseModel &A, const std::vector<double> &x, int nC,
                                              double violation_threshold);
    void insertSet(VRPTW_SRC &cuts, int i, int j, int k, const std::vector<int> &buffer_int, int buffer_int_n,
                   double LHS_cut, double violation_threshold);

    void generateCutCoefficients(VRPTW_SRC &cuts, std::vector<std::vector<double>> &coefficients, int numNodes,
                                 const SparseModel &A, const std::vector<double> &x);

    template <CutType T>
    void the45Heuristic(const SparseModel &A, const std::vector<double> &x, int numNodes, int subsetSize);

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
            // fmt::print("P[j]: {}\n", P[j]);
            int vj = P[j];

            // Check if the node vj is in AM (bitwise check)
            if (!(AM[vj / 64] & (1ULL << (vj % 64)))) {
                S = 0; // Reset S if vj is not in AM
            } else if (C[vj / 64] & (1ULL << (vj % 64))) {
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

private:
    static std::mutex cache_mutex;

    static std::unordered_map<int, std::pair<std::vector<int>, std::vector<int>>> column_cache;

    std::vector<VRPJob>                       jobs;
    std::vector<std::vector<int>>             baseSets;
    std::unordered_map<int, std::vector<int>> neighborSets;
    CutType                                   cutType;

    // Function to create a unique key from an unordered set of strings
    std::string createKey(const std::unordered_set<std::string> &set);

    int alpha(const std::vector<int> &C, const std::vector<int> &M, const std::vector<double> &p,
              const std::vector<int> &r);
};

/**
 * @brief Generates all unique permutations for a predefined set of vectors of size 5.
 *
 * This function creates permutations for a predefined set of vectors, each containing
 * five double values. The permutations are generated in lexicographical order.
 *
 * @return A vector containing all unique permutations of the predefined vectors.
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

// Function to get all permutations of multipliers for size 4
/**
 * @brief Generates all unique permutations of a fixed-size vector.
 *
 * This function generates all unique permutations of a vector of size 4
 * containing the elements {2.0/3.0, 1.0/3.0, 1.0/3.0, 1.0/3.0}. The permutations
 * are generated in lexicographical order.
 *
 * @return A vector of vectors, where each inner vector is a unique permutation
 *         of the original vector.
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
 * @tparam T The type of the elements in the input vector.
 * @param elements The input vector containing the elements to combine.
 * @param k The number of elements in each combination.
 * @param result A reference to a vector where the resulting combinations will be stored.
 *
 * @note The order of elements in each combination is determined by their order in the input vector.
 *       The function uses `std::prev_permutation` to generate the combinations.
 */
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

/**
 * @brief Selects the indices of the highest coefficients from a given vector.
 *
 * This function takes a vector of doubles and an integer specifying the maximum number of nodes to select.
 * It filters out elements with coefficients less than or equal to 1e-2, sorts the remaining elements in 
 * descending order based on their coefficients, and returns the indices of the top elements up to the 
 * specified maximum number of nodes.
 *
 * @param x A vector of doubles representing the coefficients.
 * @param maxNodes An integer specifying the maximum number of nodes to select.
 * @return A vector of integers containing the indices of the selected nodes with the highest coefficients.
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

/**
 * @brief Finds the visiting nodes in a sparse model based on selected nodes.
 *
 * This function processes a sparse model to identify and return the visiting nodes
 * for the given selected nodes. It filters the columns of the sparse model and 
 * returns only those that are relevant to the selected nodes.
 *
 * @param A The sparse model represented by the SparseModel structure.
 * @param selectedNodes A vector of integers representing the selected nodes.
 * @return A vector of vectors of integers, where each inner vector contains the 
 *         visiting nodes corresponding to a selected node.
 */
inline std::vector<std::vector<int>> findVisitingNodes(const SparseModel &A, const std::vector<int> &selectedNodes) {
    std::vector<std::vector<int>> consumers;
    std::unordered_set<int>       selectedNodeSet(selectedNodes.begin(), selectedNodes.end());

    // Reserve space for consumers to prevent multiple allocations
    consumers.reserve(selectedNodes.size());

    // Preprocess the selected columns
    std::vector<std::vector<int>> col_to_rows(A.num_cols);
    for (int j = 0; j < A.row_indices.size(); ++j) {
        // print A.values[j]
        // fmt::print("A.values[j]: {}\n", A.values[j]);
        if (A.values[j] == -1) { col_to_rows[A.col_indices[j]].push_back(A.row_indices[j]); }
    }
    // Filter only the selected nodes
    for (int col : selectedNodes) {
        if (!col_to_rows[col].empty()) { consumers.push_back(std::move(col_to_rows[col])); }
    }

    return consumers;
}

/**
 * @brief Converts a vector of integers to a comma-separated string.
 *
 * This function takes a vector of integers and concatenates each integer
 * into a single string, separated by commas.
 *
 * @param vec The vector of integers to be converted.
 * @return A string representation of the vector, with each integer separated by a comma.
 */
inline std::string vectorToString(const std::vector<int> &vec) {
    std::string result;
    for (int num : vec) { result += std::to_string(num) + ","; }
    return result;
}

/**
 * @brief Converts a vector of doubles to a comma-separated string.
 *
 * This function takes a vector of doubles and concatenates each element
 * into a single string, with each number separated by a comma.
 *
 * @param vec The vector of doubles to be converted.
 * @return A string representation of the vector, with elements separated by commas.
 */
inline std::string vectorToString(const std::vector<double> &vec) {
    std::string result;
    for (double num : vec) { result += std::to_string(num) + ","; }
    return result;
}

/**
 * @brief Implements the 45 Heuristic for generating limited memory rank-1 cuts.
 *
 * This function generates limited memory rank-1 cuts using a heuristic approach.
 * It processes sets of customers and permutations to identify violations and generate cuts.
 * The function is parallelized to improve efficiency.
 *
 * @tparam T The type of cut to generate (FourRow or FiveRow).
 * @param A The sparse model.
 * @param x The vector of coefficients.
 * @param numNodes The number of nodes.
 * @param subsetSize The size of the subset.
 */
template <CutType T>
void LimitedMemoryRank1Cuts::the45Heuristic(const SparseModel &A, const std::vector<double> &x, int numNodes,
                                            int subsetSize) {
    double primal_violation    = 0.0;
    int    max_number_of_cuts  = 1; // Max number of cuts to generate
    double violation_threshold = 1e-3;
    int    max_important_nodes = 50;

    auto cuts         = VRPTW_SRC();
    auto coefficients = std::vector<std::vector<double>>();

    int m_max = std::min(cuts.S_n, max_number_of_cuts);
    coefficients.resize(m_max, std::vector<double>(numNodes, 0.0));

    // Ensure selectedNodes is valid
    std::vector<int> selectedNodes = selectHighestCoefficients(x, max_important_nodes);
    if (selectedNodes.empty()) {
        std::cerr << "Error: No selected nodes found!" << std::endl;
        return;
    }

    // Initialize coefficients_aux based on the size of x
    std::vector<double> coefficients_aux(x.size(), 0.0);

    std::vector<std::vector<double>> permutations;
    if constexpr (T == CutType::FourRow) {
        permutations = getPermutationsForSize4();
    } else if constexpr (T == CutType::FiveRow) {
        permutations = getPermutationsForSize5();
    }

    // Shuffle permutations and limit to 3 for efficiency
    std::random_device rd;
    std::mt19937       g(rd());
    if (permutations.size() > 3) {
        std::shuffle(permutations.begin(), permutations.end(), g);
        permutations.resize(3);
    }

    std::unordered_set<std::string> processedSetsCache;
    std::unordered_set<std::string> processedPermutationsCache;
    std::mutex                      cuts_mutex;    // Protect access to shared resources
    std::atomic<int>                cuts_count(0); // Thread-safe counter for cuts

    // Create tasks for each selected node to parallelize
    std::vector<int> tasks(selectedNodes.size());
    std::iota(tasks.begin(), tasks.end(), 0); // Filling tasks with indices 0 to selectedNodes.size()

    exec::static_thread_pool pool(std::thread::hardware_concurrency()); // Dynamically set based on system
    auto                     sched = pool.get_scheduler();

    auto input_sender = stdexec::just();

    // Define the bulk operation to process sets of customers
    auto bulk_sender = stdexec::bulk(
        input_sender, tasks.size(),
        [this, &permutations, &processedSetsCache, &processedPermutationsCache, &cuts_mutex, &cuts_count, &cuts, &x,
         &selectedNodes, &coefficients_aux, &numNodes, max_number_of_cuts, violation_threshold,
         subsetSize](std::size_t task_idx) {
            auto &consumer = allPaths[selectedNodes[task_idx]].route;

            std::vector<std::vector<int>> setsOf45;
            if constexpr (T == CutType::FourRow) {
                combinations(consumer, 4, setsOf45);
            } else if constexpr (T == CutType::FiveRow) {
                combinations(consumer, 5, setsOf45);
            }

            std::array<uint64_t, num_words> AM      = {};
            std::array<uint64_t, num_words> baseSet = {};
            std::vector<Cut>                threadCuts;

            for (const auto &set45 : setsOf45) {
                std::string setHash = vectorToString(set45);

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
                    std::string pHash = vectorToString(p);
                    // concatenate processed set and permutation
                    std::string setPermutationHash = setHash + pHash;

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

                    for (auto c : set45) { baseSet[c / 64] |= (1ULL << (c % 64)); }
                    int ordering = 0;
                    for (auto node : set45) {
                        AM[node / 64] |= (1ULL << (node % 64)); // Set the bit for node in AM
                        order[node] = ordering++;
                    }

                    std::fill(coefficients_aux.begin(), coefficients_aux.end(), 0.0);
                    int    rhs             = std::floor(std::accumulate(p.begin(), p.end(), 0.0));
                    double alpha           = 0;
                    bool   violation_found = false;

                    for (auto j = 0; j < selectedNodes.size(); ++j) {
                        if (selectedNodes[j] == selectedNodes[task_idx]) {}
                        auto &consumer_inner = allPaths[selectedNodes[j]];

                        int max_limit = 0;
                        if constexpr (T == CutType::FourRow) {
                            max_limit = 3;
                        } else if constexpr (T == CutType::FiveRow) {
                            max_limit = 4;
                        }

                        int match_count = 0;
                        for (auto &job : set45) {
                            if (std::ranges::find(consumer_inner, job) != consumer_inner.end()) {
                                if (++match_count == max_limit) { break; }
                            }
                        }

                        if (match_count < max_limit) continue;

                        std::vector<int> thePath(consumer_inner.begin(), consumer_inner.end());
                        for (auto c : thePath) { AM[c / 64] |= (1ULL << (c % 64)); }

                        double alpha_inner = computeLimitedMemoryCoefficient(baseSet, AM, p, thePath, order);
                        alpha += alpha_inner;

                        coefficients_aux[selectedNodes[j]] = alpha_inner;

                        if (alpha > rhs + violation_threshold) { violation_found = true; }

                        if (violation_found) {
                            for (int i = 0; i < numNodes; ++i) {
                                // Skip nodes that are part of baseSet (i.e., cannot be removed from AM)
                                if (!(baseSet[i / 64] & (1ULL << (i % 64)))) {

                                    // Check if the node is currently in AM
                                    if (AM[i / 64] & (1ULL << (i % 64))) {

                                        // Temporarily remove node i from AM
                                        AM[i / 64] &= ~(1ULL << (i % 64));

                                        // Recompute the alpha value with the reduced AM
                                        double reduced_alpha = 0;
                                        for (auto j = 0; j < selectedNodes.size(); ++j) {
                                            auto            &consumer_inner = allPaths[selectedNodes[j]];
                                            std::vector<int> thePath(consumer_inner.begin(), consumer_inner.end());

                                            reduced_alpha +=
                                                computeLimitedMemoryCoefficient(baseSet, AM, p, thePath, order);
                                        }

                                        // If the violation no longer holds, restore the node in AM
                                        if (reduced_alpha <= rhs + violation_threshold) {
                                            AM[i / 64] |= (1ULL << (i % 64)); // Restore node i in AM
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
                                for (auto &job : set45) {
                                    if (std::ranges::find(consumer_inner, job) != consumer_inner.end()) {
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
                            // threadCuts.push_back(cut);

                            std::lock_guard<std::mutex> cut_lock(cuts_mutex);
                            if (cuts_count.load() < max_number_of_cuts) {
                                cutStorage.addCut(cut);
                                cuts_count += 1;
                            }

                            break;
                        }
                    }

                    // if (violation_found && cuts_count.load() < max_number_of_cuts) {}

                    if (cuts_count.load() >= max_number_of_cuts) {
                        return; // Stop processing further if the limit is reached
                    }
                }
            }
        });

    auto work = stdexec::on(sched, bulk_sender);
    stdexec::sync_wait(std::move(work));
}
