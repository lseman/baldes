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
#include "HighOrderSRC.h"
#include "Pools.h"
#include "SparseMatrix.h"
#include "ankerl/unordered_dense.h"
#include "bucket/BucketGraph.h"
#include "miphandler/LinExp.h"
#include "miphandler/MIPHandler.h"
// #include "xxhash.h"

#include <cstdint>

#include "RNG.h"
// include nsync
#ifdef NSYNC
extern "C" {
#include "nsync_mu.h"
}
#endif

#define VRPTW_SRC_max_S_n 10000

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
class LimitedMemoryRank1Cuts {
   public:
    Xoroshiro128Plus rp;  // Seed it (you can change the seed)
    using HighDimCutsGeneratorPtr = std::shared_ptr<HighDimCutsGenerator>;
    HighDimCutsGeneratorPtr generator =
        std::make_shared<HighDimCutsGenerator>(N_SIZE, 5, 1e-6);

    void setDistanceMatrix(const std::vector<std::vector<double>> distances) {
        generator->setDistanceMatrix(distances);
    }
    LimitedMemoryRank1Cuts(std::vector<VRPNode> &nodes);

    LimitedMemoryRank1Cuts(const LimitedMemoryRank1Cuts &other)
        : rp(other.rp),
          generator(other.generator ? other.generator->clone()
                                    : nullptr),  // Clone generator if it exists
          cutStorage(other.cutStorage),
          allPaths(other.allPaths),
          labels(other.labels),
          labels_counter(other.labels_counter),
          nodes(other.nodes) {
        other.generator->setDistanceMatrix(other.generator->cost_mat4_vertex);
    }

    void setDuals(const std::vector<double> &duals) {
        // print nodes.size
        for (size_t i = 1; i < N_SIZE - 1; ++i) {
            nodes[i].setDuals(duals[i - 1]);
        }
    }

    // default constructor
    LimitedMemoryRank1Cuts() = default;

    void setNodes(std::vector<VRPNode> &nodes) { this->nodes = nodes; }

    CutStorage cutStorage = CutStorage();

    void printBaseSets();
    std::vector<Path> allPaths;

    std::vector<std::vector<int>> labels;
    int labels_counter = 0;
    void separate(const SparseMatrix &A, const std::vector<double> &x);

    /**
     * @brief Computes the limited memory coefficient based on the given
     * parameters.
     *
     * This function calculates the coefficient by iterating through the
     * elements of the vector P, checking their presence in the bitwise arrays C
     * and AM, and updating the coefficient based on the values in the vector p
     * and the order vector.
     *
     */
    double computeLimitedMemoryCoefficient(
        const std::array<uint64_t, num_words> &C,
        const std::array<uint64_t, num_words> &AM, const SRCPermutation &p,
        const std::vector<uint16_t> &P, std::vector<int> &order);

    std::pair<bool, bool> runSeparation(
        BNBNode *node, std::vector<baldesCtrPtr> &SRCconstraints);

    void separateR1C1(const SparseMatrix &A, const std::vector<double> &x) {
        const size_t num_paths = allPaths.size();

        // Pre-allocate vectors to avoid reallocations
        std::vector<std::pair<double, int>> tmp_cuts;
        tmp_cuts.reserve(num_paths);

        // Reuse map to avoid repeated allocations
        ankerl::unordered_dense::map<int, int> vis_map;
        vis_map.reserve(N_SIZE);  // Pre-allocate expected size

        // Thread pool for parallel processing
        const int JOBS = std::thread::hardware_concurrency();
        exec::static_thread_pool pool(JOBS);
        auto sched = pool.get_scheduler();

        struct LocalCuts {
            std::vector<std::pair<double, int>> cuts;
            ankerl::unordered_dense::map<int, int> vis_map;

            LocalCuts() {
                cuts.reserve(64);  // Reserve reasonable chunk size
                vis_map.reserve(N_SIZE);
            }
        };

        // Create thread-local storage for cuts
        std::vector<LocalCuts> thread_local_cuts(JOBS);
        std::mutex cuts_mutex;

        // Process paths in parallel chunks
        const size_t chunk_size = (num_paths + JOBS - 1) / JOBS;

        auto bulk_sender =
            stdexec::bulk(stdexec::just(), JOBS, [&](size_t thread_idx) {
                const size_t start = thread_idx * chunk_size;
                const size_t end = std::min(start + chunk_size, num_paths);

                auto &local_cuts = thread_local_cuts[thread_idx];

                for (size_t path_idx = start; path_idx < end; ++path_idx) {
                    const auto &r = allPaths[path_idx];
                    local_cuts.vis_map.clear();

                    // Count occurrences
                    for (const auto i : r.route) {
                        ++local_cuts.vis_map[i];
                    }

                    // Find violations
                    for (const auto &[v, times] : local_cuts.vis_map) {
                        if (times > 1) {
                            double cut_value =
                                std::floor(times / 2.0) * r.frac_x;
                            if (cut_value > 1e-3) {
                                local_cuts.cuts.emplace_back(cut_value, v);
                            }
                        }
                    }
                }

                // Merge local results if we found any cuts
                if (!local_cuts.cuts.empty()) {
                    std::lock_guard<std::mutex> lock(cuts_mutex);
                    tmp_cuts.insert(
                        tmp_cuts.end(),
                        std::make_move_iterator(local_cuts.cuts.begin()),
                        std::make_move_iterator(local_cuts.cuts.end()));
                }
            });

        // Execute parallel work
        auto work = stdexec::starts_on(sched, bulk_sender);
        stdexec::sync_wait(std::move(work));

        if (tmp_cuts.empty()) return;

        // Sort cuts by violation value
        pdqsort(tmp_cuts.begin(), tmp_cuts.end(),
                [](const auto &a, const auto &b) { return a.first > b.first; });

        // Pre-allocate vectors for cut generation
        std::vector<double> coefficients_aux(num_paths);
        const int cuts_to_apply =
            std::min(static_cast<int>(tmp_cuts.size()), 2);

        // Process top violations and generate cuts
        for (int i = 0; i < cuts_to_apply; ++i) {
            const auto &[cut_value, v] = tmp_cuts[i];

            // Initialize cut data structures
            std::array<uint64_t, num_words> C{};
            std::array<uint64_t, num_words> AM{};
            std::vector<int> order(N_SIZE);

            // Set up cut components
            C[v / 64] |= (1ULL << (v % 64));

            // Fill AM using optimized bit operations
            constexpr uint64_t all_ones = ~uint64_t(0);
            for (size_t j = 0; j < num_words - 1; ++j) {
                AM[j] = all_ones;
            }
            // Handle last word carefully to avoid setting bits beyond N_SIZE
            AM[num_words - 1] = (1ULL << (N_SIZE % 64)) - 1;

            // Initialize permutation
            SRCPermutation p;
            p.num = {1};
            p.den = 2;

            // Compute coefficients in parallel
            auto coef_sender = stdexec::bulk(
                stdexec::just(), (num_paths + chunk_size - 1) / chunk_size,
                [&](size_t chunk_idx) {
                    const size_t start = chunk_idx * chunk_size;
                    const size_t end = std::min(start + chunk_size, num_paths);

                    for (size_t j = start; j < end; ++j) {
                        coefficients_aux[j] = computeLimitedMemoryCoefficient(
                            C, AM, p, allPaths[j].route, order);
                    }
                });

            auto coef_work = stdexec::starts_on(sched, coef_sender);
            stdexec::sync_wait(std::move(coef_work));

            // Create and add cut
            Cut cut(C, AM, coefficients_aux, p);
            cut.baseSetOrder = std::move(order);
            cutStorage.addCut(cut);
        }
    }

   private:
    static std::mutex cache_mutex;

    static ankerl::unordered_dense::map<
        int, std::pair<std::vector<int>, std::vector<int>>>
        column_cache;

    std::vector<VRPNode> nodes;
    CutType cutType;
};
