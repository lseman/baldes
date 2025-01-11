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

    void separateCombined(const SparseMatrix &A, const std::vector<double> &x) {
        std::vector<std::pair<double, Cut>> tmp_normal_cuts;
        std::vector<std::pair<double, int>> tmp_r1c_cuts;

        const int JOBS = std::thread::hardware_concurrency();
        exec::static_thread_pool pool(JOBS);
        auto sched = pool.get_scheduler();

        // Define chunk size to reduce parallelization overhead
        const int chunk_size = (allPaths.size() + JOBS - 1) / JOBS;

        // Mutex to protect access to shared resources
        std::mutex cuts_mutex;

        // Create a map for non-zero entries by rows (for normal cuts)
        std::vector<std::vector<int>> row_indices_map(N_SIZE);
        for (int idx = 0; idx < A.values.size(); ++idx) {
            int row = A.rows[idx];
            if (row > N_SIZE - 2) {
                continue;
            }
            row_indices_map[row + 1].push_back(idx);
        }

        // Parallel execution in chunks
        auto bulk_sender = stdexec::bulk(
            stdexec::just(), (allPaths.size() + chunk_size - 1) / chunk_size,
            [&](std::size_t chunk_idx) {
                size_t start_idx = chunk_idx * chunk_size;
                size_t end_idx =
                    std::min(start_idx + chunk_size, allPaths.size());

                // Local storage for cuts in this chunk
                std::vector<std::pair<double, Cut>> local_normal_cuts;
                std::vector<std::pair<double, int>> local_r1c_cuts;

                ankerl::unordered_dense::map<int, int> vis_map;
                for (size_t i = start_idx; i < end_idx; ++i) {
                    const auto &r = allPaths[i];

                    // Compute R1C cuts
                    vis_map.clear();
                    for (const auto node : r.route) {
                        ++vis_map[node];
                    }
                    for (const auto &[v, times] : vis_map) {
                        if (times > 1) {
                            double cut_value =
                                std::floor(times / 2.) * r.frac_x;
                            if (cut_value > 1e-3) {
                                local_r1c_cuts.emplace_back(cut_value, v);
                            }
                        }
                    }

                    // Compute normal cuts
                    for (int i = 1; i < N_SIZE - 1; ++i) {
                        for (int j = i + 1; j < N_SIZE - 1; ++j) {
                            for (int k = j + 1; k < N_SIZE - 1; ++k) {
                                std::vector<int> expanded(A.num_cols, 0);
                                double lhs = 0.0;

                                // Combine the three updates into one loop for
                                // efficiency
                                for (int idx : row_indices_map[i])
                                    expanded[A.cols[idx]] += 1;
                                for (int idx : row_indices_map[j])
                                    expanded[A.cols[idx]] += 1;
                                for (int idx : row_indices_map[k])
                                    expanded[A.cols[idx]] += 1;

                                // Accumulate LHS cut values
                                for (int idx = 0; idx < A.num_cols; ++idx) {
                                    if (expanded[idx] >= 2) {
                                        lhs += std::floor(expanded[idx] * 0.5) *
                                               x[idx];
                                    }
                                }

                                // If lhs violation found, insert the cut
                                if (lhs > 1.0 + 1e-3) {
                                    std::array<uint64_t, num_words> C = {};
                                    std::array<uint64_t, num_words> AM = {};
                                    std::vector<int> order(N_SIZE, 0);

                                    std::vector<int> C_index = {i, j, k};
                                    for (auto node : C_index) {
                                        C[node / 64] |= (1ULL << (node % 64));
                                        AM[node / 64] |= (1ULL << (node % 64));
                                        order[node] = 1;  // Simplified ordering
                                    }

                                    SRCPermutation p;
                                    p.num = {1, 1, 1};
                                    p.den = 2;

                                    std::vector<double> cut_coefficients(
                                        allPaths.size(), 0.0);
                                    for (size_t j = 0; j < allPaths.size();
                                         ++j) {
                                        auto &clients = allPaths[j].route;
                                        cut_coefficients[j] =
                                            computeLimitedMemoryCoefficient(
                                                C, AM, p, clients, order);
                                    }

                                    Cut cut(C, AM, cut_coefficients, p);
                                    cut.baseSetOrder = order;
                                    local_normal_cuts.emplace_back(lhs, cut);
                                }
                            }
                        }
                    }
                }

                // Merge local cuts into global cuts
                std::lock_guard<std::mutex> lock(cuts_mutex);
                tmp_normal_cuts.insert(tmp_normal_cuts.end(),
                                       local_normal_cuts.begin(),
                                       local_normal_cuts.end());
                tmp_r1c_cuts.insert(tmp_r1c_cuts.end(), local_r1c_cuts.begin(),
                                    local_r1c_cuts.end());
            });

        // Submit work to the thread pool
        auto work = stdexec::starts_on(sched, bulk_sender);
        stdexec::sync_wait(std::move(work));

        // Process R1C cuts
        if (!tmp_r1c_cuts.empty()) {
            pdqsort(
                tmp_r1c_cuts.begin(), tmp_r1c_cuts.end(),
                [](const auto &a, const auto &b) { return a.first > b.first; });

            std::vector<double> coefficients_aux(allPaths.size(), 0.0);
            auto cuts_to_apply =
                std::min(static_cast<int>(tmp_r1c_cuts.size()), 2);

            for (int i = 0; i < cuts_to_apply; ++i) {
                auto cut_value = tmp_r1c_cuts[i].first;
                auto v = tmp_r1c_cuts[i].second;

                std::array<uint64_t, num_words> C = {};
                std::array<uint64_t, num_words> AM = {};
                std::vector<int> order(N_SIZE, 0);

                C[v / 64] |= (1ULL << (v % 64));
                for (int node = 0; node < N_SIZE; ++node) {
                    AM[node / 64] |= (1ULL << (node % 64));
                }

                SRCPermutation p;
                p.num = {1};
                p.den = 2;

                coefficients_aux.assign(allPaths.size(), 0.0);
                for (size_t j = 0; j < allPaths.size(); ++j) {
                    auto &clients = allPaths[j].route;
                    coefficients_aux[j] = computeLimitedMemoryCoefficient(
                        C, AM, p, clients, order);
                }

                Cut cut(C, AM, coefficients_aux, p);
                cut.baseSetOrder = order;
                cutStorage.addCut(cut);
            }
        }

        // Process normal cuts
        if (!tmp_normal_cuts.empty()) {
            pdqsort(
                tmp_normal_cuts.begin(), tmp_normal_cuts.end(),
                [](const auto &a, const auto &b) { return a.first > b.first; });

            auto max_cuts = 2;
            for (int i = 0;
                 i <
                 std::min(max_cuts, static_cast<int>(tmp_normal_cuts.size()));
                 ++i) {
                auto &cut = tmp_normal_cuts[i].second;
                cutStorage.addCut(cut);
            }
        }
    }

    void separateR1C1(const SparseMatrix &A, const std::vector<double> &x) {
        std::vector<std::pair<double, int>> tmp_cuts;
        tmp_cuts.reserve(
            allPaths.size());  // Conservative reservation based on sol size

        const int JOBS = std::thread::hardware_concurrency();
        exec::static_thread_pool pool(JOBS);
        auto sched = pool.get_scheduler();

        // Define chunk size to reduce parallelization overhead
        const int chunk_size = (allPaths.size() + JOBS - 1) / JOBS;

        // Mutex to protect access to tmp_cuts
        std::mutex cuts_mutex;

        // Parallel execution in chunks
        auto bulk_sender = stdexec::bulk(
            stdexec::just(), (allPaths.size() + chunk_size - 1) / chunk_size,
            [&](std::size_t chunk_idx) {
                size_t start_idx = chunk_idx * chunk_size;
                size_t end_idx =
                    std::min(start_idx + chunk_size, allPaths.size());

                // Local storage for cuts in this chunk
                std::vector<std::pair<double, int>> local_cuts;

                ankerl::unordered_dense::map<int, int> vis_map;
                for (size_t i = start_idx; i < end_idx; ++i) {
                    const auto &r = allPaths[i];
                    vis_map.clear();
                    for (const auto i : r.route) {
                        ++vis_map[i];
                    }
                    for (const auto &[v, times] : vis_map) {
                        if (times > 1) {
                            // Calculate fractional cut value with integer
                            // division instead of `floor`
                            double cut_value =
                                std::floor(times / 2.) * r.frac_x;
                            if (cut_value > 1e-3) {  // Apply tolerance check
                                local_cuts.emplace_back(cut_value, v);
                            }
                        }
                    }
                }

                // Merge local cuts into global tmp_cuts
                std::lock_guard<std::mutex> lock(cuts_mutex);
                tmp_cuts.insert(tmp_cuts.end(), local_cuts.begin(),
                                local_cuts.end());
            });

        // Submit work to the thread pool
        auto work = stdexec::starts_on(sched, bulk_sender);
        stdexec::sync_wait(std::move(work));

        // Early return if no cuts were found
        if (tmp_cuts.empty()) return;

        // Sort cuts by value in descending order
        pdqsort(
            tmp_cuts.begin(), tmp_cuts.end(),
            [](const std::pair<double, int> &a,
               const std::pair<double, int> &b) { return a.first > b.first; });

        // Generate cut coefficients for the top cuts
        std::vector<double> coefficients_aux(allPaths.size(), 0.0);
        auto cuts_to_apply = std::min(static_cast<int>(tmp_cuts.size()), 2);

        for (int i = 0; i < cuts_to_apply; ++i) {
            auto cut_value = tmp_cuts[i].first;
            auto v = tmp_cuts[i].second;

            // Create a new cut
            std::array<uint64_t, num_words> C = {};  // Reset C for each cut
            std::array<uint64_t, num_words> AM = {};
            std::vector<int> order(N_SIZE, 0);

            // Define C
            C[v / 64] |= (1ULL << (v % 64));

            // Define AM
            for (int node = 0; node < N_SIZE; ++node) {
                AM[node / 64] |= (1ULL << (node % 64));
            }

            SRCPermutation p;
            p.num = {1};
            p.den = 2;

            // Compute coefficients for all paths
            coefficients_aux.assign(allPaths.size(), 0.0);
            for (size_t j = 0; j < allPaths.size(); ++j) {
                auto &clients = allPaths[j].route;
                coefficients_aux[j] =
                    computeLimitedMemoryCoefficient(C, AM, p, clients, order);
            }

            // Create and store the cut
            Cut cut(C, AM, coefficients_aux, p);
            cut.baseSetOrder = order;
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
