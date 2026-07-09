/**
 * @file SRC.h
 * @brief Defines limited memory rank-1 cut handling for VRPTW.
 *
 */

#pragma once

#include "cuts/Cut.h"
#include "core/Definitions.h"
#include "core/Pools.h"
#include "SparseMatrix.h"
#include "ankerl/unordered_dense.h"
#include "bucket/BucketGraph.h"
#include "miphandler/LinExp.h"
#include "miphandler/MIPHandler.h"
// #include "xxhash.h"
//
#include <cstdint>

#include "cuts/HeuristicHighOrder.h"
#include "math/RNG.h"
// include nsync
#ifdef NSYNC
extern "C" {
#include "nsync_mu.h"
}
#endif

struct CachedCut {
    Cut    cut;
    double violation;
};

class BNBNode;

struct VectorStringHash;
struct UnorderedSetStringHash;
struct PairHash;

struct SparseMatrix;

#include "math/RNG.h"

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
    void     setArcDuals(const ArcDuals &arc_duals) { this->arc_duals = arc_duals; }
#endif
    int last_path_idx = 0;

    std::vector<std::vector<int>> vertex_route_map;
    static constexpr int          MAX_HEURISTIC_SEP_ROW_RANK1 = 8;

    std::vector<std::vector<int>> rank1_sep_heur_mem4_vertex;
    std::vector<std::vector<int>> map_rank1_multiplier_dominance;

    void initializeVertexRouteMap() {
        if (last_path_idx == 0) {
            row_indices_map.clear();
            row_indices_map.reserve(N_SIZE);
            map_rank1_multiplier_dominance.assign(N_SIZE, {});
        }

        // Initialize the vertex_route_map with N_SIZE rows and allPaths.size()
        // columns, all set to 0.
        if (last_path_idx == 0) {
            vertex_route_map.assign(N_SIZE, std::vector<int>(allPaths.size(), 0));
        } else {
            // assign 0 from last_path_idx to allPaths.size()
            for (size_t i = 0; i < N_SIZE; ++i) {
                vertex_route_map[i].resize(allPaths.size());
                for (size_t j = last_path_idx; j < allPaths.size(); ++j) { vertex_route_map[i][j] = 0; }
            }
        }

        // Populate the map: for each path, count the appearance of each vertex.
        std::vector<unsigned char> seen_in_path(N_SIZE, 0);
        std::vector<int>           touched_vertices;
        for (size_t r = last_path_idx; r < allPaths.size(); ++r) {
            for (const auto &vertex : allPaths[r].route) {
                // Only consider vertices that are not the depot (assumed at
                // indices 0 and N_SIZE-1).
                if (vertex > 0 && vertex < N_SIZE - 1) {
                    ++vertex_route_map[vertex][r];
                    if (!seen_in_path[vertex]) {
                        seen_in_path[vertex] = 1;
                        touched_vertices.push_back(vertex);
                        row_indices_map[vertex].push_back(static_cast<int>(r));
                    }
                }
            }
            for (int vertex : touched_vertices) { seen_in_path[vertex] = 0; }
            touched_vertices.clear();
        }
        last_path_idx = allPaths.size();
    }

    void initializeRank1HeuristicNeighbors() {
        rank1_sep_heur_mem4_vertex.clear();
        rank1_sep_heur_mem4_vertex.resize(N_SIZE);
        if (high_rank_cuts.distances.size() != static_cast<size_t>(N_SIZE)) return;

        for (int i = 1; i < N_SIZE - 1; ++i) {
            std::vector<std::pair<double, int>> cost;
            cost.reserve(N_SIZE - 2);
            for (int j = 1; j < N_SIZE - 1; ++j) {
                if (i == j) continue;
                cost.emplace_back(high_rank_cuts.distances[i][j], j);
            }
            std::stable_sort(cost.begin(), cost.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
            int limit = std::min(MAX_HEURISTIC_SEP_ROW_RANK1, static_cast<int>(cost.size()));
            for (int k = 0; k < limit; ++k) { rank1_sep_heur_mem4_vertex[i].push_back(cost[k].second); }
        }
    }

    Xoroshiro128Plus rp; // Seed it (you can change the seed)

    void setDistanceMatrix(const std::vector<std::vector<double>> distances) { high_rank_cuts.distances = distances; }
    LimitedMemoryRank1Cuts(std::vector<VRPNode> &nodes);

    LimitedMemoryRank1Cuts(const LimitedMemoryRank1Cuts &other)
        : rp(other.rp), last_path_idx(other.last_path_idx), vertex_route_map(other.vertex_route_map),
          rank1_sep_heur_mem4_vertex(other.rank1_sep_heur_mem4_vertex),
          map_rank1_multiplier_dominance(other.map_rank1_multiplier_dominance), cutStorage(other.cutStorage),
          allPaths(other.allPaths), labels(other.labels), labels_counter(other.labels_counter),
          row_indices_map(other.row_indices_map), nodes(other.nodes), cutType(other.cutType), tasks(other.tasks) {}

    void setDuals(const std::vector<double> &duals) {
        // print nodes.size
        for (size_t i = 1; i < N_SIZE - 1; ++i) { nodes[i].setDuals(duals[i - 1]); }
    }

    // default constructor
    LimitedMemoryRank1Cuts() { setTasks(); }

    void setNodes(std::vector<VRPNode> &nodes) { this->nodes = nodes; }

    std::vector<std::tuple<int, int, int>> tasks;

    void setTasks() {
        // Create tasks for each combination of (i, j, k)
        for (int i = 1; i < N_SIZE - 1; ++i) {
            for (int j = i + 1; j < N_SIZE - 1; ++j) {
                for (int k = j + 1; k < N_SIZE - 1; ++k) { tasks.emplace_back(i, j, k); }
            }
        }
    }

    CutStorage cutStorage = CutStorage();

    void              printBaseSets();
    std::vector<Path> allPaths;

    std::vector<std::vector<int>> labels;
    int                           labels_counter = 0;
    void                          separate(const SparseMatrix &A, const std::vector<double> &x);

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
    double computeLimitedMemoryCoefficient(const std::array<uint64_t, num_words> &C,
                                           const std::array<uint64_t, num_words> &AM, const SRCPermutation &p,
                                           const std::vector<uint16_t> &P, std::vector<int> &order) noexcept;

    std::pair<bool, bool> runSeparation(BNBNode *node, std::vector<baldesCtrPtr> &SRCconstraints);

    exec::static_thread_pool            pool  = exec::static_thread_pool(std::thread::hardware_concurrency());
    exec::static_thread_pool::scheduler sched = pool.get_scheduler();

    void separateR1C1(const SparseMatrix &A, const std::vector<double> &x) {
        if (allPaths.empty()) return;

        initializeRank1HeuristicNeighbors();

        struct CandidateVertex {
            double violation;
            int    node;
        };

        std::vector<CandidateVertex> tmp_cuts;
        tmp_cuts.reserve(N_SIZE);
        std::vector<double> vertex_violation(N_SIZE, 0.0);

        for (int node = 1; node < N_SIZE - 1; ++node) {
            if (row_indices_map[node].empty()) continue;

            double violation = 0.0;
            for (int path_idx : row_indices_map[node]) {
                int visit_count = vertex_route_map[node][path_idx];
                if (visit_count >= 2) {
                    int pair_count = visit_count / 2;
                    violation += pair_count * x[path_idx];
                }
            }
            vertex_violation[node] = violation;
            if (violation > 1e-3) { tmp_cuts.push_back({violation, node}); }
        }

        if (tmp_cuts.empty()) return;

        // === Step 2: Sort vertices by total cut violation in descending order ===
        pdqsort(tmp_cuts.begin(), tmp_cuts.end(),
                [](const CandidateVertex &a, const CandidateVertex &b) { return a.violation > b.violation; });

        std::vector<std::pair<double, int>> ordered_cuts;
        ordered_cuts.reserve(tmp_cuts.size());
        std::vector<char> seen(N_SIZE, 0);

        for (const auto &candidate : tmp_cuts) {
            if (static_cast<int>(ordered_cuts.size()) >= 3) break;
            if (seen[candidate.node]) continue;
            ordered_cuts.emplace_back(candidate.violation, candidate.node);
            seen[candidate.node] = 1;
        }
        pdqsort(ordered_cuts.begin(), ordered_cuts.end(),
                [](const std::pair<double, int> &a, const std::pair<double, int> &b) { return a.first > b.first; });
        const int cuts_to_apply = std::min(static_cast<int>(ordered_cuts.size()), 3);
        SRCPermutation p;
        p.num = {1};
        p.den = 2;

        for (int i = 0; i < cuts_to_apply; ++i) {
            int node = ordered_cuts[i].second;

            std::array<uint64_t, num_words> C  = {};
            std::array<uint64_t, num_words> AM = {};
            std::vector<int>                order(N_SIZE, -1);

            C[node / 64] |= (1ULL << (node % 64));
            order[node] = 0;
            const auto cut_key = cutStorage.compute_cut_key(C, p.num, p.den);
            if (cutStorage.cutExists(cut_key).first >= 0) { continue; }

            for (int node_idx = 0; node_idx < N_SIZE; ++node_idx) { AM[node_idx / 64] |= (1ULL << (node_idx % 64)); }

            std::vector<int>    coefficient_indices;
            std::vector<double> coefficient_values;
            const auto         &support_paths = row_indices_map[node];
            coefficient_indices.reserve(support_paths.size());
            coefficient_values.reserve(support_paths.size());

            double       exact_violation = 0.0;
            const double rhs             = p.getRHS();
            for (int path_idx : support_paths) {
                if (path_idx < 0 || static_cast<size_t>(path_idx) >= allPaths.size() ||
                    static_cast<size_t>(path_idx) >= x.size())
                    continue;
                const auto   &clients = allPaths[path_idx].route;
                const double coeff    = computeLimitedMemoryCoefficient(C, AM, p, clients, order);
                exact_violation += coeff * x[path_idx];
                if (!numericutils::isZero(coeff)) {
                    coefficient_indices.push_back(path_idx);
                    coefficient_values.push_back(coeff);
                }
            }

            if (exact_violation <= rhs + 1e-6) {
                continue; // Skip cuts that are not actually violated by the current solution.
            }

            Cut cut(C, AM, {}, p);
            cut.baseSetOrder        = std::move(order);
            cut.type                = CutType::OneRow;
            cut.coefficient_indices = std::move(coefficient_indices);
            cut.coefficient_values  = std::move(coefficient_values);

            cutStorage.addCut(cut);
        }
    }

    bool cutCleaner(BNBNode *node, std::vector<baldesCtrPtr> &SRCconstraints);

    /**
     * @brief Adjacency-based rank-3 SRC separation (RouteOpt-style).
     *
     * For each pair (i,j), candidate third nodes k are drawn from the
     * union (if i–j adjacent) or intersection (otherwise) of their
     * adjacency neighborhoods, mirroring RouteOpt's generateR1C3().
     * The fast floor-based violation screening avoids the O(n³) task
     * enumeration of separate(), focusing effort on structurally
     * promising triples.
     */
    void separateR1C3Adjacency(const SparseMatrix &A, const std::vector<double> &x);

private:
    static std::mutex cache_mutex;

    static ankerl::unordered_dense::map<int, std::pair<std::vector<int>, std::vector<int>>> column_cache;

    std::vector<VRPNode> nodes;
    CutType              cutType;
};
