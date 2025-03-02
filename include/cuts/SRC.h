/**
 * @file SRC.h
 * @brief Definitions for handling cuts and optimizations for the Vehicle
 * Routing Problem with Time Windows (VRPTW).
 *
 * This header file contains the structure and class definitions required for
 * the limited memory rank-1 cuts, including the handling of cuts for
 * optimization algorithms used in the Vehicle Routing Problem with Time Windows
 * (VRPTW).
 *
 *
 * It also includes utility functions to compute coefficients, generate cuts,
 * and work with sparse models. The file facilitates the optimization process by
 * allowing computation of limited memory coefficients and the generation of
 * cuts via heuristics. The file makes use of Gurobi for handling constraints in
 * the solver.
 *
 */

#pragma once

#include "../third_party/concurrentqueue.h"
#include "Cut.h"
#include "Definitions.h"
#include "Pools.h"
#include "SparseMatrix.h"
#include "ankerl/unordered_dense.h"
#include "bucket/BucketGraph.h"
#include "miphandler/LinExp.h"
#include "miphandler/MIPHandler.h"
// #include "xxhash.h"
//
#include <cstdint>

#include "HeuristicHighOrder.h"
#include "RNG.h"
// include nsync
#ifdef NSYNC
extern "C" {
#include "nsync_mu.h"
}
#endif

struct CachedCut {
    Cut cut;
    double violation;
};

class BNBNode;

struct VectorStringHash;
struct UnorderedSetStringHash;
struct PairHash;

struct SparseMatrix;

#include "RNG.h"

/**
 * @class LimitedMemoryRank1Cuts
 * @brief A class for handling limited memory rank-1 cuts in optimization
 * problems.
 *
 * This class provides methods for separating cuts by enumeration, generating
 * cut coefficients, and computing limited memory coefficients. It also includes
 * various utility functions for managing and printing base sets, inserting
 * sets, and performing heuristics.
 *
 */
using RCCManagerPtr = std::shared_ptr<RCCManager>;

class LimitedMemoryRank1Cuts {
   public:
    HighRankCuts high_rank_cuts;
#if defined(RCC) || defined(EXACT_RCC)
    ArcDuals arc_duals;
    void setArcDuals(const ArcDuals &arc_duals) { this->arc_duals = arc_duals; }
#endif
    int last_path_idx = 0;

    std::vector<std::vector<int>> vertex_route_map;
    void initializeVertexRouteMap() {
        // Initialize the vertex_route_map with N_SIZE rows and allPaths.size()
        // columns, all set to 0.
        if (last_path_idx == 0) {
            vertex_route_map.assign(N_SIZE,
                                    std::vector<int>(allPaths.size(), 0));
        } else {
            // assign 0 from last_path_idx to allPaths.size()
            for (size_t i = 0; i < N_SIZE; ++i) {
                vertex_route_map[i].resize(allPaths.size());
                for (size_t j = last_path_idx; j < allPaths.size(); ++j) {
                    vertex_route_map[i][j] = 0;
                }
            }
        }

        // Populate the map: for each path, count the appearance of each vertex.
        for (size_t r = last_path_idx; r < allPaths.size(); ++r) {
            for (const auto &vertex : allPaths[r].route) {
                // Only consider vertices that are not the depot (assumed at
                // indices 0 and N_SIZE-1).
                if (vertex > 0 && vertex < N_SIZE - 1) {
                    ++vertex_route_map[vertex][r];
                }
            }
        }
        last_path_idx = allPaths.size();
    }

    Xoroshiro128Plus rp;  // Seed it (you can change the seed)

    void setDistanceMatrix(const std::vector<std::vector<double>> distances) {
        high_rank_cuts.distances = distances;
    }
    LimitedMemoryRank1Cuts(std::vector<VRPNode> &nodes);

    LimitedMemoryRank1Cuts(const LimitedMemoryRank1Cuts &other)
        : rp(other.rp),
          cutStorage(other.cutStorage),
          allPaths(other.allPaths),
          labels(other.labels),
          labels_counter(other.labels_counter),
          nodes(other.nodes) {}

    void setDuals(const std::vector<double> &duals) {
        // print nodes.size
        for (size_t i = 1; i < N_SIZE - 1; ++i) {
            nodes[i].setDuals(duals[i - 1]);
        }
    }

    // default constructor
    LimitedMemoryRank1Cuts() { setTasks(); }

    void setNodes(std::vector<VRPNode> &nodes) { this->nodes = nodes; }

    std::vector<std::tuple<int, int, int>> tasks;

    void setTasks() {
        // Create tasks for each combination of (i, j, k)
        for (int i = 1; i < N_SIZE - 1; ++i) {
            for (int j = i + 1; j < N_SIZE - 1; ++j) {
                for (int k = j + 1; k < N_SIZE - 1; ++k) {
                    tasks.emplace_back(i, j, k);
                }
            }
        }
    }

    CutStorage cutStorage = CutStorage();

    void printBaseSets();
    std::vector<Path> allPaths;

    std::vector<std::vector<int>> labels;
    int labels_counter = 0;
    void separate(const SparseMatrix &A, const std::vector<double> &x);

    ankerl::unordered_dense::map<int, std::vector<int>> row_indices_map;
    /**
     * @brief Computes the limited memory coefficient based on the given
     * parameters.
     *
     * This function calculates the coefficient by iterating through the
     * elements of the vector P, checking their presence in the bitwise
     * arrays C and AM, and updating the coefficient based on the values in
     * the vector p and the order vector.
     *
     */
    double computeLimitedMemoryCoefficient(
        const std::array<uint64_t, num_words> &C,
        const std::array<uint64_t, num_words> &AM, const SRCPermutation &p,
        const std::vector<uint16_t> &P, std::vector<int> &order) noexcept;

    std::pair<bool, bool> runSeparation(
        BNBNode *node, std::vector<baldesCtrPtr> &SRCconstraints);

    exec::static_thread_pool pool =
        exec::static_thread_pool(std::thread::hardware_concurrency());
    exec::static_thread_pool::scheduler sched = pool.get_scheduler();

    void separateR1C1(const SparseMatrix &A, const std::vector<double> &x) {
        // === Step 1: Extract Potential Cuts from allPaths ===
        const size_t nPaths = allPaths.size();
        const int JOBS = std::thread::hardware_concurrency();
        const int chunk_size = (nPaths + JOBS - 1) / JOBS;
        const size_t num_chunks = (nPaths + chunk_size - 1) / chunk_size;

        // Reserve per-chunk storage to avoid locking.
        std::vector<std::vector<std::pair<double, int>>> chunk_cuts(num_chunks);
        for (auto &vec : chunk_cuts) {
            vec.reserve(10);  // Reserve an estimate; adjust as needed.
        }

        // Parallel execution in chunks over allPaths.
        auto bulk_sender = stdexec::bulk(
            stdexec::just(), num_chunks, [&](std::size_t chunk_idx) {
                size_t start_idx = chunk_idx * chunk_size;
                size_t end_idx = std::min(start_idx + chunk_size, nPaths);

                // Local vector for storing cuts computed in this chunk.
                std::vector<std::pair<double, int>> local_cuts;
                local_cuts.reserve(chunk_size);
                // Temporary map to count occurrences of nodes in a given path.

                // Process each path in this chunk.
                for (size_t i = start_idx; i < end_idx; ++i) {
                    for (auto node : allPaths[i].route) {
                        if (vertex_route_map[node][i] >= 2) {
                            // Calculate the lhs value for the current path.
                            double cut_value = x[i];
                            if (cut_value > 1e-3) {  // Tolerance check.
                                local_cuts.emplace_back(cut_value, node);
                            }
                        }
                    }
                }

                // Save the local cuts into the per-chunk vector.
                chunk_cuts[chunk_idx] = std::move(local_cuts);
            });

        // Submit the bulk work and wait for all tasks to complete.
        auto work = stdexec::starts_on(sched, bulk_sender);
        stdexec::sync_wait(std::move(work));

        // Merge results from all chunks.
        std::vector<std::pair<double, int>> tmp_cuts;
        for (const auto &vec : chunk_cuts) {
            tmp_cuts.insert(tmp_cuts.end(), vec.begin(), vec.end());
        }

        // Early return if no cuts were found.
        if (tmp_cuts.empty()) return;

        // === Step 2: Sort Extracted Cuts by Their Cut Value in Descending
        // Order ===
        pdqsort(
            tmp_cuts.begin(), tmp_cuts.end(),
            [](const std::pair<double, int> &a,
               const std::pair<double, int> &b) { return a.first > b.first; });

        // === Step 3: Generate and Add the Best Cut ===
        // Only generate coefficients for the top cut.
        const int cuts_to_apply =
            std::min(static_cast<int>(tmp_cuts.size()), 3);
        std::vector<double> coefficients_aux(allPaths.size(), 0.0);

        // Setup a simple SRCPermutation for the cut.
        SRCPermutation p;
        p.num = {1};
        p.den = 2;

        for (int i = 0; i < cuts_to_apply; ++i) {
            double cut_value = tmp_cuts[i].first;
            int node = tmp_cuts[i].second;

            // Initialize bitmask arrays for representing the cut.
            std::array<uint64_t, num_words> C = {};   // Base set bitmask.
            std::array<uint64_t, num_words> AM = {};  // Augmented mask.
            std::vector<int> order(N_SIZE, 0);

            // Define C: mark only the selected node.
            C[node / 64] |= (1ULL << (node % 64));

            // Define AM: for this simple cut, mark all nodes.
            for (int node_idx = 0; node_idx < N_SIZE; ++node_idx) {
                AM[node_idx / 64] |= (1ULL << (node_idx % 64));
            }

            // Compute coefficients for each path based on the current cut.
            coefficients_aux.assign(allPaths.size(), 0.0);
            for (size_t j = 0; j < allPaths.size(); ++j) {
                const auto &clients = allPaths[j].route;
                coefficients_aux[j] =
                    computeLimitedMemoryCoefficient(C, AM, p, clients, order);
            }

            // Create a new cut using the computed coefficients.
            Cut cut(C, AM, coefficients_aux, p);
            cut.baseSetOrder = order;
            cut.type = CutType::OneRow;

            // Add the cut to the cut storage.
            cutStorage.addCut(cut);
        }
    }

    bool cutCleaner(BNBNode *node, std::vector<baldesCtrPtr> &SRCconstraints);

   private:
    static std::mutex cache_mutex;

    static ankerl::unordered_dense::map<
        int, std::pair<std::vector<int>, std::vector<int>>>
        column_cache;

    std::vector<VRPNode> nodes;
    CutType cutType;
};
