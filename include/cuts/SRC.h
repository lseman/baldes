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

#include "miphandler/MIPHandler.h"

#include "HighOrderSRC.h"
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
    Xoroshiro128Plus rp; // Seed it (you can change the seed)
    using HighDimCutsGeneratorPtr     = std::shared_ptr<HighDimCutsGenerator>;
    HighDimCutsGeneratorPtr generator = std::make_shared<HighDimCutsGenerator>(N_SIZE, 5, 1e-6);

    void setDistanceMatrix(const std::vector<std::vector<double>> distances) {
        generator->setDistanceMatrix(distances);
    }
    LimitedMemoryRank1Cuts(std::vector<VRPNode> &nodes);

    LimitedMemoryRank1Cuts(const LimitedMemoryRank1Cuts &other)
        : rp(other.rp), generator(other.generator ? other.generator->clone() : nullptr), // Clone generator if it exists
          cutStorage(other.cutStorage), allPaths(other.allPaths), labels(other.labels),
          labels_counter(other.labels_counter), nodes(other.nodes) {
        other.generator->setDistanceMatrix(other.generator->cost_mat4_vertex);
    }

    void setDuals(const std::vector<double> &duals) {
        // print nodes.size
        for (size_t i = 1; i < N_SIZE - 1; ++i) { nodes[i].setDuals(duals[i - 1]); }
    }

    // default constructor
    LimitedMemoryRank1Cuts() = default;

    void setNodes(std::vector<VRPNode> &nodes) { this->nodes = nodes; }

    CutStorage cutStorage = CutStorage();

    void              printBaseSets();
    std::vector<Path> allPaths;

    std::vector<std::vector<int>>    labels;
    int                              labels_counter = 0;
    std::vector<std::vector<double>> separate(const SparseMatrix &A, const std::vector<double> &x);
    void insertSet(VRPTW_SRC &cuts, int i, int j, int k, const std::vector<int> &buffer_int, int buffer_int_n,
                   double LHS_cut);

    void insertSet(VRPTW_SRC &cuts, int i, const std::vector<int> &buffer_int, int buffer_int_n, double LHS_cut);
    void generateCutCoefficients(VRPTW_SRC &cuts, std::vector<std::vector<double>> &coefficients, int numNodes,
                                 const SparseMatrix &A, const std::vector<double> &x);

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

    std::pair<bool, bool> runSeparation(BNBNode *node, std::vector<Constraint *> &SRCconstraints);

private:
    static std::mutex cache_mutex;

    static ankerl::unordered_dense::map<int, std::pair<std::vector<int>, std::vector<int>>> column_cache;

    std::vector<VRPNode> nodes;
    CutType              cutType;
};
