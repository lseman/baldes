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
    bool cuts_harvested = false;
    int n_high_order = 0;

#if defined(RCC) || defined(EXACT_RCC)
    ArcDuals arc_duals;
    void setArcDuals(const ArcDuals &arc_duals) { this->arc_duals = arc_duals; }
#endif

    Xoroshiro128Plus rp;  // Seed it (you can change the seed)
    using HighDimCutsGeneratorPtr = std::shared_ptr<HighDimCutsGenerator>;
    HighDimCutsGeneratorPtr generator =
        std::make_shared<HighDimCutsGenerator>(N_SIZE);

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
        const std::vector<uint16_t> &P, const std::vector<int> &order) noexcept;

    std::pair<bool, bool> runSeparation(
        BNBNode *node, std::vector<baldesCtrPtr> &SRCconstraints);

    void separateR1C1(const SparseMatrix &A, const std::vector<double> &x) {
        const int JOBS = std::thread::hardware_concurrency();
        const size_t paths_size = allPaths.size();

        // Pre-allocate thread-local storage
        std::vector<std::vector<std::pair<double, int>>> thread_local_cuts(
            JOBS);
        for (auto &cuts : thread_local_cuts) {
            cuts.reserve(paths_size / JOBS);  // Estimate per thread
        }

        exec::static_thread_pool pool(JOBS);
        auto sched = pool.get_scheduler();

        const int chunk_size =
            std::max(1000, static_cast<int>((paths_size + JOBS - 1) / JOBS));

        // Thread-local buffers to avoid reallocations
        struct ThreadLocalBuffers {
            ankerl::unordered_dense::map<int, int> vis_map;

            ThreadLocalBuffers() : vis_map(64) {}  // Pre-reserve typical size
        };

        auto bulk_sender = stdexec::bulk(
            stdexec::just(), (paths_size + chunk_size - 1) / chunk_size,
            [this, &thread_local_cuts, paths_size,
             chunk_size](std::size_t chunk_idx) {
                ThreadLocalBuffers buffers;
                auto &local_cuts =
                    thread_local_cuts[chunk_idx % thread_local_cuts.size()];

                const size_t start_idx = chunk_idx * chunk_size;
                const size_t end_idx =
                    std::min(start_idx + chunk_size, paths_size);

                for (size_t i = start_idx; i < end_idx; ++i) {
                    const auto &route = allPaths[i].route;
                    const double frac_x = allPaths[i].frac_x;

                    buffers.vis_map.clear();

                    // Count occurrences - manual loop for better performance
                    const size_t route_size = route.size();
                    for (size_t j = 0; j < route_size; ++j) {
                        ++buffers.vis_map[route[j]];
                    }

                    // Process repeating nodes
                    for (const auto &[v, times] : buffers.vis_map) {
                        if (times > 1) {
                            const double cut_value =
                                (times >> 1) * frac_x;  // Integer division by 2
                            if (cut_value > 1e-3) {
                                local_cuts.emplace_back(cut_value, v);
                            }
                        }
                    }
                }
            });

        stdexec::sync_wait(stdexec::on(sched, std::move(bulk_sender)));

        // Merge cuts from all threads
        std::vector<std::pair<double, int>> final_cuts;
        size_t total_cuts = 0;
        for (const auto &thread_cuts : thread_local_cuts) {
            total_cuts += thread_cuts.size();
        }
        final_cuts.reserve(total_cuts);

        for (const auto &thread_cuts : thread_local_cuts) {
            final_cuts.insert(final_cuts.end(), thread_cuts.begin(),
                              thread_cuts.end());
        }

        if (final_cuts.empty()) return;

        // Sort cuts by value
        pdqsort(final_cuts.begin(), final_cuts.end(),
                [](const auto &a, const auto &b) { return a.first > b.first; });

        // Pre-allocate reusable buffers for cut generation
        std::vector<double> coefficients_aux(paths_size);
        const int cuts_to_apply =
            std::min(static_cast<int>(final_cuts.size()), 2);

        for (int i = 0; i < cuts_to_apply; ++i) {
            const int v = final_cuts[i].second;

            // Initialize C and AM more efficiently
            std::array<uint64_t, num_words> C = {};
            std::array<uint64_t, num_words> AM = {};

            // Set bit in C
            C[v >> 6] |= (1ULL << (v & 63));

            // Fill AM using memset for better performance
            std::memset(AM.data(), 0xFF, sizeof(AM));

            // Initialize order vector
            std::vector<int> order(N_SIZE);
            std::iota(order.begin(), order.end(), 0);

            SRCPermutation p;
            p.num = {1};
            p.den = 2;

// Compute coefficients
#pragma GCC ivdep
            for (size_t j = 0; j < paths_size; ++j) {
                coefficients_aux[j] = computeLimitedMemoryCoefficient(
                    C, AM, p, allPaths[j].route, order);
            }

            Cut cut(C, AM, coefficients_aux, p);
            cut.baseSetOrder = std::move(order);
            cutStorage.addCut(cut);
        }
    }

    void separateBG(BNBNode *node, std::vector<baldesCtrPtr> &SRCconstraints);
    bool getCutsBG();
    bool cutCleaner(BNBNode *node, std::vector<baldesCtrPtr> &SRCconstraints);

    std::function<int(std::vector<R1c> &)> processCuts =
        [&](std::vector<R1c> &cortes) {
            auto cuts = &cutStorage;
            auto multipliers = generator->map_rank1_multiplier;

            // Initialize paths with solution values
            auto cut_ctr = 0;

            // Process each cut
            for (auto &cut : cortes) {
                auto &cut_info = cut.info_r1c;

                // Skip if arc_mem is empty
                if (cut.arc_mem.empty()) {
                    continue;
                }

                // Pre-allocate cut_indices to avoid repeated resizing
                std::vector<int> cut_indices;
                cut_indices.reserve(cut_info.first.size());
                for (auto &node : cut_info.first) {
                    cut_indices.push_back(node);
                }

                // Use std::array with pre-initialization for bit arrays
                std::array<uint64_t, num_words> C = {};
                std::array<uint64_t, num_words> AM = {};

                // Set bits in C and AM for cut_indices
                for (auto &node : cut_indices) {
                    C[node / 64] |= (1ULL << (node % 64));
                    AM[node / 64] |= (1ULL << (node % 64));
                }
                // Set bits in AM for arcs in arc_mem
                for (auto &arc : cut.arc_mem) {
                    AM[arc / 64] |= (1ULL << (arc % 64));
                }

                // Retrieve multiplier information
                auto &mult =
                    multipliers[cut_info.first.size()][cut_info.second];
                SRCPermutation p;
                p.num = std::get<0>(mult);
                p.den = std::get<1>(mult);

                // Initialize order without an extra resize, only if N_SIZE
                // is constant
                std::vector<int> order(N_SIZE, 0);
                int ordering = 0;
                for (auto node : cut_indices) {
                    order[node] = ordering++;
                }

                // Pre-allocate coeffs to allPaths.size()
                std::vector<double> coeffs(allPaths.size(), 0.0);

#if defined(__cpp_lib_parallel_algorithm)
                std::atomic<bool> has_coeff{false};
                std::transform(allPaths.begin(), allPaths.end(), coeffs.begin(),
                               [&](const auto &path) {
                                   auto coeff = computeLimitedMemoryCoefficient(
                                       C, AM, p, path.route, order);
                                   if (coeff > 1e-3)
                                       has_coeff.store(
                                           true, std::memory_order_relaxed);
                                   return coeff;
                               });
#else
                bool has_coeff = false;
                std::transform(allPaths.begin(), allPaths.end(), coeffs.begin(),
                               [&](const auto &path) {
                                   auto coeff = computeLimitedMemoryCoefficient(
                                       C, AM, p, path.route, order);
                                   if (coeff > 1e-3) has_coeff = true;
                                   return coeff;
                               });
#endif
                // Skip adding cut if no coefficients met threshold
                if (!has_coeff) {
                    continue;
                }

                // Create and add new cut
                Cut corte(C, AM, coeffs, p);
                corte.baseSetOrder = order;
                cuts->addCut(corte);
                cut_ctr++;
            }
            return cut_ctr;
        };

   private:
    static std::mutex cache_mutex;

    static ankerl::unordered_dense::map<
        int, std::pair<std::vector<int>, std::vector<int>>>
        column_cache;

    std::vector<VRPNode> nodes;
    CutType cutType;
};
