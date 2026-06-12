/*
 * @file HeuristicHighOrder.h
 * @brief Declares HeuristicHighOrder interfaces and types used by the BALDES solver.
 *
 * This file declares the HeuristicHighOrder interfaces and helper functions used by the BALDES solver.
 *
 */

#pragma once
#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <unordered_map>

#include "Definitions.h"
#include "Dual.h"
#include "Path.h"
#include "RNG.h"
#include "VRPNode.h"
#include "utils/NumericUtils.h"

// include CutHelper
#include "CutHelper.h"
#include "CutIntelligence.h"

// namespace std {
// template <>
// struct hash<std::pair<int, int>> {
//     size_t operator()(const std::pair<int, int> &p) const {
//         // Combine the hash values of both integers
//         size_t h1 = std::hash<int>{}(p.first);
//         size_t h2 = std::hash<int>{}(p.second);
//         return h1 ^ (h2 << 1);
//     }
// };
// }  // namespace std

class HighRankCuts {
public:
    // Convenience alias — used by separate() and helpers.
    using SeedMap = std::unordered_map<std::vector<int>, std::vector<int>, std::hash<std::vector<int>>>;

private:
    static constexpr int MAX_COMBINATIONS     = 5;
    static constexpr int MAX_WORKING_SET_SIZE = 12;

    exec::static_thread_pool            pool  = exec::static_thread_pool(std::thread::hardware_concurrency());
    exec::static_thread_pool::scheduler sched = pool.get_scheduler();

    static void insertBestCandidate(
        ankerl::unordered_dense::set<CandidateSet, CandidateSetHasher, CandidateSetCompare> &candidates,
        CandidateSet candidate) {
        auto it = candidates.find(candidate);
        if (it == candidates.end()) {
            candidates.emplace(std::move(candidate));
            return;
        }
        if (candidate.violation > it->violation + 1e-9) {
            candidates.erase(it);
            candidates.emplace(std::move(candidate));
        }
    }

    // Add vertex-route mapping
    std::map<std::vector<int>, double> candidate_cache;
    int                                last_path_idx = 0;
    std::vector<std::vector<int>>      rank1_sep_heur_mem4_vertex;
    void                               initializeVertexRouteMap() {
        if (last_path_idx == 0) {
            row_indices_map.clear();
            row_indices_map.reserve(N_SIZE);
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

    void initializeRank1Memory(const std::vector<std::vector<NodeScore>> &scores) {
        rank1_sep_heur_mem4_vertex.clear();
        rank1_sep_heur_mem4_vertex.resize(N_SIZE);
        for (int i = 1; i < N_SIZE - 1; ++i) {
            std::vector<std::pair<double, int>> scored;
            scored.reserve(scores[i].size());
            for (const auto &ns : scores[i]) {
                double proximity = 0.0;
                if (distances.size() == static_cast<size_t>(N_SIZE) &&
                    distances[i].size() == static_cast<size_t>(N_SIZE)) {
                    proximity = distances[i][ns.node];
                }
                double combined_score = ns.cost_score + 0.1 * proximity;
                scored.emplace_back(combined_score, ns.node);
            }
            std::stable_sort(scored.begin(), scored.end(),
                             [](const auto &a, const auto &b) { return a.first < b.first; });
            const int limit = std::min(MAX_WORKING_SET_SIZE, static_cast<int>(scored.size()));
            for (int j = 0; j < limit; ++j) { rank1_sep_heur_mem4_vertex[i].push_back(scored[j].second); }
        }
    }

    SeedMap constructHighRankSeeds() {
        SeedMap seed_map;
        if (allPaths.empty()) return seed_map;

        // Build route appearances for each node.
        std::vector<std::unordered_map<int, int>> v_r_map(N_SIZE);
        for (int r = 0; r < static_cast<int>(allPaths.size()); ++r) {
            for (auto vertex : allPaths[r].route) {
                if (vertex > 0 && vertex < N_SIZE - 1) { ++v_r_map[vertex][r]; }
            }
        }

        std::vector<char> is_mem(N_SIZE);
        std::vector<int>  candidate_nodes;
        std::vector<int>  memory_nodes;
        for (int i = 1; i < N_SIZE - 1; ++i) {
            if (v_r_map[i].empty()) continue;

            std::fill(is_mem.begin(), is_mem.end(), 0);
            for (const auto &pr : v_r_map[i]) {
                const auto &route = allPaths[pr.first].route;
                for (auto vertex : route) {
                    if (vertex > 0 && vertex < N_SIZE - 1 && vertex != i) {
                        if (std::find(rank1_sep_heur_mem4_vertex[i].begin(), rank1_sep_heur_mem4_vertex[i].end(),
                                      vertex) != rank1_sep_heur_mem4_vertex[i].end()) {
                            is_mem[vertex] = 1;
                        }
                    }
                }
            }
            if (std::none_of(is_mem.begin(), is_mem.end(), [](char v) { return v; })) continue;

            for (int r = 0; r < static_cast<int>(allPaths.size()); ++r) {
                if (v_r_map[i].find(r) != v_r_map[i].end()) continue;
                candidate_nodes.clear();
                candidate_nodes.push_back(i);
                const auto &route = allPaths[r].route;
                for (auto vertex : route) {
                    if (vertex > 0 && vertex < N_SIZE - 1 && is_mem[vertex]) { candidate_nodes.push_back(vertex); }
                }
                std::sort(candidate_nodes.begin(), candidate_nodes.end());
                candidate_nodes.erase(std::unique(candidate_nodes.begin(), candidate_nodes.end()),
                                      candidate_nodes.end());
                if (candidate_nodes.size() < MIN_RANK || candidate_nodes.size() > MAX_RANK) continue;
                memory_nodes.clear();
                for (int vertex : rank1_sep_heur_mem4_vertex[i]) {
                    if (vertex > 0 && vertex < N_SIZE - 1 &&
                        !std::binary_search(candidate_nodes.begin(), candidate_nodes.end(), vertex)) {
                        memory_nodes.push_back(vertex);
                    }
                }
                if (!memory_nodes.empty()) {
                    std::sort(memory_nodes.begin(), memory_nodes.end());
                    memory_nodes.erase(std::unique(memory_nodes.begin(), memory_nodes.end()), memory_nodes.end());
                }
                auto &entry = seed_map[candidate_nodes];
                if (entry.empty()) {
                    entry = memory_nodes;
                } else {
                    for (auto v : memory_nodes) {
                        if (!std::binary_search(entry.begin(), entry.end(), v)) { entry.push_back(v); }
                    }
                    std::sort(entry.begin(), entry.end());
                }
            }
        }
        return seed_map;
    }

    ankerl::unordered_dense::set<CandidateSet, CandidateSetHasher, CandidateSetCompare>
    generateRouteSupportSeeds(const SparseMatrix &A, const std::vector<double> &x,
                              const std::vector<std::vector<NodeScore>> &scores) {
        ankerl::unordered_dense::set<CandidateSet, CandidateSetHasher, CandidateSetCompare> seeds;
        if (allPaths.empty() || x.empty()) return seeds;

        struct RouteSupport {
            double weight;
            int    path_idx;
        };
        std::vector<RouteSupport> route_supports;
        route_supports.reserve(nonzero_paths.size());
        for (int path_idx : nonzero_paths) {
            if (path_idx < 0 || static_cast<size_t>(path_idx) >= allPaths.size() || static_cast<size_t>(path_idx) >= x.size())
                continue;
            if (x[path_idx] > 1e-6) { route_supports.push_back({x[path_idx], path_idx}); }
        }
        pdqsort(route_supports.begin(), route_supports.end(),
                [](const auto &a, const auto &b) { return a.weight > b.weight; });

        constexpr int MAX_ROUTE_SEEDS       = 60;
        constexpr int MAX_ROUTE_NODE_POOL   = 10;
        constexpr int MAX_MEMORY_PER_SEED   = 10;
        constexpr int MAX_SEEDS_PER_ROUTE   = 8;
        const int     n_routes_to_scan = std::min(MAX_ROUTE_SEEDS, static_cast<int>(route_supports.size()));

        std::vector<char> in_candidate(N_SIZE, 0);
        std::vector<int>  route_nodes;
        std::vector<int>  candidate_nodes;
        std::vector<int>  memory_nodes;
        for (int route_pos = 0; route_pos < n_routes_to_scan; ++route_pos) {
            const auto &path = allPaths[route_supports[route_pos].path_idx];
            route_nodes.clear();
            for (int vertex : path.route) {
                if (vertex <= 0 || vertex >= N_SIZE - 1) continue;
                if (std::find(route_nodes.begin(), route_nodes.end(), vertex) == route_nodes.end()) {
                    route_nodes.push_back(vertex);
                }
            }
            if (static_cast<int>(route_nodes.size()) < MIN_RANK) continue;

            std::vector<std::pair<double, int>> scored_nodes;
            scored_nodes.reserve(route_nodes.size());
            for (int vertex : route_nodes) {
                double score = -route_supports[route_pos].weight;
                if (vertex < static_cast<int>(scores.size()) && !scores[vertex].empty()) {
                    score += 0.05 * scores[vertex].front().cost_score;
                }
                scored_nodes.emplace_back(score, vertex);
            }
            pdqsort(scored_nodes.begin(), scored_nodes.end(),
                    [](const auto &a, const auto &b) { return a.first < b.first; });
            if (static_cast<int>(scored_nodes.size()) > MAX_ROUTE_NODE_POOL) {
                scored_nodes.resize(MAX_ROUTE_NODE_POOL);
            }

            std::vector<int> node_pool;
            node_pool.reserve(scored_nodes.size());
            for (const auto &[_, vertex] : scored_nodes) { node_pool.push_back(vertex); }

            int seeds_for_route = 0;
            for (int rank = MAX_RANK; rank >= MIN_RANK && seeds_for_route < MAX_SEEDS_PER_ROUTE; --rank) {
                if (rank > static_cast<int>(node_pool.size())) continue;
                uint32_t mask = (1u << rank) - 1;
                const uint32_t limit = (1u << node_pool.size());
                while (mask < limit && seeds_for_route < MAX_SEEDS_PER_ROUTE) {
                    candidate_nodes.clear();
                    candidate_nodes.reserve(rank);
                    for (size_t bit = 0; bit < node_pool.size(); ++bit) {
                        if (mask & (1u << bit)) candidate_nodes.push_back(node_pool[bit]);
                    }
                    std::sort(candidate_nodes.begin(), candidate_nodes.end());

                    for (int node : candidate_nodes) { in_candidate[node] = 1; }
                    memory_nodes.clear();
                    memory_nodes.reserve(MAX_MEMORY_PER_SEED);
                    for (int node : route_nodes) {
                        if (static_cast<int>(memory_nodes.size()) >= MAX_MEMORY_PER_SEED) break;
                        if (!in_candidate[node]) memory_nodes.push_back(node);
                    }
                    for (int base : candidate_nodes) {
                        if (base >= static_cast<int>(rank1_sep_heur_mem4_vertex.size())) continue;
                        for (int neighbor : rank1_sep_heur_mem4_vertex[base]) {
                            if (static_cast<int>(memory_nodes.size()) >= MAX_MEMORY_PER_SEED) break;
                            if (neighbor <= 0 || neighbor >= N_SIZE - 1 || in_candidate[neighbor]) continue;
                            if (std::find(memory_nodes.begin(), memory_nodes.end(), neighbor) == memory_nodes.end()) {
                                memory_nodes.push_back(neighbor);
                            }
                        }
                    }
                    for (int node : candidate_nodes) { in_candidate[node] = 0; }

                    if (auto candidate = evaluateBaseThenMemory(candidate_nodes, A, x)) {
                        insertBestCandidate(seeds, std::move(*candidate));
                    }
                    ++seeds_for_route;

                    uint32_t c = mask & -mask;
                    uint32_t r = mask + c;
                    if (c == 0) break;
                    mask = (((r ^ mask) >> 2) / c) | r;
                }
            }
        }
        return seeds;
    }

    ankerl::unordered_dense::set<CandidateSet, CandidateSetHasher, CandidateSetCompare>
    generateLiftedCutSeeds(const SparseMatrix &A, const std::vector<double> &x,
                           const std::vector<std::vector<NodeScore>> &scores) {
        ankerl::unordered_dense::set<CandidateSet, CandidateSetHasher, CandidateSetCompare> seeds;
        if (cutStorage == nullptr || cutStorage->empty()) return seeds;

        constexpr int MAX_EXTENSION_POOL = 12;
        constexpr int MAX_LIFTS_PER_CUT  = 10;
        constexpr int MAX_MEMORY_NODES   = 12;

        std::vector<int>  base_nodes;
        std::vector<int>  extension_pool;
        std::vector<int>  candidate_nodes;
        std::vector<int>  memory_nodes;
        std::vector<char> in_candidate(N_SIZE, 0);
        for (const auto &cut : *cutStorage) {
            base_nodes.clear();
            memory_nodes.clear();
            for (int node = 1; node < N_SIZE - 1; ++node) {
                const size_t   segment = static_cast<size_t>(node) >> 6;
                const uint64_t bit     = 1ULL << (node & 63);
                if (cut.baseSet[segment] & bit) {
                    base_nodes.push_back(node);
                } else if (cut.neighbors[segment] & bit) {
                    memory_nodes.push_back(node);
                }
            }
            if (static_cast<int>(base_nodes.size()) < MIN_RANK ||
                static_cast<int>(base_nodes.size()) >= MAX_RANK)
                continue;

            for (int node : base_nodes) { in_candidate[node] = 1; }
            extension_pool.clear();
            for (int base : base_nodes) {
                if (base >= static_cast<int>(scores.size())) continue;
                for (const auto &score : scores[base]) {
                    const int candidate = score.node;
                    if (candidate <= 0 || candidate >= N_SIZE - 1 || in_candidate[candidate]) continue;
                    if (std::find(extension_pool.begin(), extension_pool.end(), candidate) == extension_pool.end()) {
                        extension_pool.push_back(candidate);
                        if (static_cast<int>(extension_pool.size()) >= MAX_EXTENSION_POOL) break;
                    }
                }
                if (static_cast<int>(extension_pool.size()) >= MAX_EXTENSION_POOL) break;
            }

            int lifts_for_cut = 0;
            const int max_extra = std::min(MAX_RANK - static_cast<int>(base_nodes.size()),
                                           static_cast<int>(extension_pool.size()));
            for (int extra = 1; extra <= max_extra && lifts_for_cut < MAX_LIFTS_PER_CUT; ++extra) {
                uint32_t mask = (1u << extra) - 1;
                const uint32_t limit = (1u << extension_pool.size());
                while (mask < limit && lifts_for_cut < MAX_LIFTS_PER_CUT) {
                    candidate_nodes = base_nodes;
                    for (size_t bit = 0; bit < extension_pool.size(); ++bit) {
                        if (mask & (1u << bit)) candidate_nodes.push_back(extension_pool[bit]);
                    }
                    std::sort(candidate_nodes.begin(), candidate_nodes.end());

                    ankerl::unordered_dense::set<int> memory_set;
                    memory_set.reserve(MAX_MEMORY_NODES);
                    for (int node : memory_nodes) {
                        if (static_cast<int>(memory_set.size()) >= MAX_MEMORY_NODES) break;
                        if (!std::binary_search(candidate_nodes.begin(), candidate_nodes.end(), node)) {
                            memory_set.insert(node);
                        }
                    }
                    for (int node : extension_pool) {
                        if (static_cast<int>(memory_set.size()) >= MAX_MEMORY_NODES) break;
                        if (!std::binary_search(candidate_nodes.begin(), candidate_nodes.end(), node)) {
                            memory_set.insert(node);
                        }
                    }

                    if (auto candidate = evaluateBaseThenMemory(candidate_nodes, A, x)) {
                        insertBestCandidate(seeds, std::move(*candidate));
                    }
                    ++lifts_for_cut;

                    uint32_t c = mask & -mask;
                    uint32_t r = mask + c;
                    if (c == 0) break;
                    mask = (((r ^ mask) >> 2) / c) | r;
                }
            }
            for (int node : base_nodes) { in_candidate[node] = 0; }
        }
        return seeds;
    }

    ankerl::unordered_dense::set<CandidateSet, CandidateSetHasher, CandidateSetCompare>
    generateFractionalCooccurrenceSeeds(const SparseMatrix &A, const std::vector<double> &x,
                                        const std::vector<std::vector<NodeScore>> &scores) {
        ankerl::unordered_dense::set<CandidateSet, CandidateSetHasher, CandidateSetCompare> seeds;
        if (allPaths.empty() || nonzero_paths.empty()) return seeds;

        constexpr int MAX_SUPPORT_NODES     = 48;
        constexpr int MAX_ANCHORS           = 24;
        constexpr int MAX_EXTENSION_POOL    = 12;
        constexpr int MAX_CANDIDATES_ANCHOR = 18;
        constexpr int MAX_MEMORY_NODES      = 12;

        std::vector<double> node_support(N_SIZE, 0.0);
        for (int path_idx : nonzero_paths) {
            if (path_idx < 0 || static_cast<size_t>(path_idx) >= allPaths.size() || static_cast<size_t>(path_idx) >= x.size())
                continue;
            const double x_val = x[path_idx];
            if (numericutils::isZero(x_val)) continue;
            for (int node : allPaths[path_idx].route) {
                if (node > 0 && node < N_SIZE - 1) { node_support[node] += x_val; }
            }
        }

        std::vector<int> support_nodes;
        support_nodes.reserve(N_SIZE);
        for (int node = 1; node < N_SIZE - 1; ++node) {
            if (node_support[node] > 1e-6) support_nodes.push_back(node);
        }
        pdqsort(support_nodes.begin(), support_nodes.end(),
                [&](int a, int b) { return node_support[a] > node_support[b]; });
        if (static_cast<int>(support_nodes.size()) > MAX_SUPPORT_NODES) support_nodes.resize(MAX_SUPPORT_NODES);
        if (static_cast<int>(support_nodes.size()) < MIN_RANK) return seeds;

        std::vector<int> top_index(N_SIZE, -1);
        for (int idx = 0; idx < static_cast<int>(support_nodes.size()); ++idx) { top_index[support_nodes[idx]] = idx; }

        const int top_count = static_cast<int>(support_nodes.size());
        std::vector<std::vector<double>> cooc(top_count, std::vector<double>(top_count, 0.0));
        std::vector<int>                 route_top_nodes;
        std::vector<char>                seen_top(top_count, 0);
        for (int path_idx : nonzero_paths) {
            if (path_idx < 0 || static_cast<size_t>(path_idx) >= allPaths.size() || static_cast<size_t>(path_idx) >= x.size())
                continue;
            const double x_val = x[path_idx];
            if (numericutils::isZero(x_val)) continue;
            route_top_nodes.clear();
            for (int node : allPaths[path_idx].route) {
                if (node <= 0 || node >= N_SIZE - 1) continue;
                const int idx = top_index[node];
                if (idx < 0 || seen_top[idx]) continue;
                seen_top[idx] = 1;
                route_top_nodes.push_back(idx);
            }
            for (size_t i = 0; i < route_top_nodes.size(); ++i) {
                for (size_t j = i + 1; j < route_top_nodes.size(); ++j) {
                    const int a = route_top_nodes[i];
                    const int b = route_top_nodes[j];
                    cooc[a][b] += x_val;
                    cooc[b][a] += x_val;
                }
            }
            for (int idx : route_top_nodes) { seen_top[idx] = 0; }
        }

        std::vector<int>  extension_pool;
        std::vector<int>  candidate_nodes;
        std::vector<char> in_candidate(N_SIZE, 0);
        const int         anchor_count = std::min(MAX_ANCHORS, top_count);
        for (int anchor_pos = 0; anchor_pos < anchor_count; ++anchor_pos) {
            const int anchor = support_nodes[anchor_pos];
            std::vector<std::pair<double, int>> neighbor_scores;
            neighbor_scores.reserve(top_count - 1);
            for (int pos = 0; pos < top_count; ++pos) {
                if (pos == anchor_pos) continue;
                const int    node  = support_nodes[pos];
                double score = cooc[anchor_pos][pos] + 0.25 * node_support[node];
                if (node < static_cast<int>(scores.size()) && !scores[node].empty()) {
                    score -= 0.01 * scores[node].front().cost_score;
                }
                if (score > 1e-6) neighbor_scores.emplace_back(score, node);
            }
            pdqsort(neighbor_scores.begin(), neighbor_scores.end(),
                    [](const auto &a, const auto &b) { return a.first > b.first; });
            if (neighbor_scores.empty()) continue;

            extension_pool.clear();
            for (const auto &[_, node] : neighbor_scores) {
                extension_pool.push_back(node);
                if (static_cast<int>(extension_pool.size()) >= MAX_EXTENSION_POOL) break;
            }

            int candidates_for_anchor = 0;
            for (int rank = MAX_RANK; rank >= 4 && candidates_for_anchor < MAX_CANDIDATES_ANCHOR; --rank) {
                const int choose = rank - 1;
                if (choose > static_cast<int>(extension_pool.size())) continue;
                uint32_t mask = (1u << choose) - 1;
                const uint32_t limit = (1u << extension_pool.size());
                while (mask < limit && candidates_for_anchor < MAX_CANDIDATES_ANCHOR) {
                    candidate_nodes.clear();
                    candidate_nodes.reserve(rank);
                    candidate_nodes.push_back(anchor);
                    for (size_t bit = 0; bit < extension_pool.size(); ++bit) {
                        if (mask & (1u << bit)) candidate_nodes.push_back(extension_pool[bit]);
                    }
                    std::sort(candidate_nodes.begin(), candidate_nodes.end());

                    for (int node : candidate_nodes) { in_candidate[node] = 1; }
                    ankerl::unordered_dense::set<int> memory_set;
                    memory_set.reserve(MAX_MEMORY_NODES);
                    for (int node : extension_pool) {
                        if (static_cast<int>(memory_set.size()) >= MAX_MEMORY_NODES) break;
                        if (!in_candidate[node]) memory_set.insert(node);
                    }
                    for (int node : candidate_nodes) {
                        if (node >= static_cast<int>(rank1_sep_heur_mem4_vertex.size())) continue;
                        for (int neighbor : rank1_sep_heur_mem4_vertex[node]) {
                            if (static_cast<int>(memory_set.size()) >= MAX_MEMORY_NODES) break;
                            if (neighbor <= 0 || neighbor >= N_SIZE - 1 || in_candidate[neighbor]) continue;
                            memory_set.insert(neighbor);
                        }
                    }
                    for (int node : candidate_nodes) { in_candidate[node] = 0; }

                    if (auto candidate = evaluateBaseThenMemory(candidate_nodes, A, x)) {
                        insertBestCandidate(seeds, std::move(*candidate));
                    }
                    ++candidates_for_anchor;

                    uint32_t c = mask & -mask;
                    uint32_t r = mask + c;
                    if (c == 0) break;
                    mask = (((r ^ mask) >> 2) / c) | r;
                }
            }
        }

        return seeds;
    }

    ankerl::unordered_dense::set<CandidateSet, CandidateSetHasher, CandidateSetCompare>
    generateSeedCandidates(const SparseMatrix &A, const std::vector<double> &x, const SeedMap &seed_map) {
        ankerl::unordered_dense::set<CandidateSet, CandidateSetHasher, CandidateSetCompare> seeds;
        for (auto &seed_pair : seed_map) {
            const auto                       &candidate_nodes = seed_pair.first;
            const auto                       &memory_nodes    = seed_pair.second;
            ankerl::unordered_dense::set<int> memory_set(memory_nodes.begin(), memory_nodes.end());
            auto [violation, perm, rhs] = computeViolationWithBestPerm(candidate_nodes, memory_set, A, x);
            if (violation > 1e-3) {
                insertBestCandidate(seeds, CandidateSet(candidate_nodes, violation, perm, memory_set, rhs));
            }
        }
        return seeds;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Node scoring
    ////////////////////////////////////////////////////////////////////////////

    AdaptiveNodeScorer                  ml_scorer;
    std::vector<std::vector<NodeScore>> computeNodeScores(const SparseMatrix &A, const std::vector<double> &x) {
        auto scores = ml_scorer.computeNodeScores(A, x, distances, nodes, arc_duals, *cutStorage);

        return scores;
    }
    ankerl::unordered_dense::set<CandidateSet, CandidateSetHasher, CandidateSetCompare>
    generateCandidates(const std::vector<std::vector<NodeScore>> &scores, const SparseMatrix &A,
                       const std::vector<double> &x) {
        struct ViolationResult {
            double         violation = 0.0;
            SRCPermutation perm;
            double         rhs = 0.0;
        };

        // Thread-local data for each thread.
        struct ThreadLocalData {
            ankerl::unordered_dense::set<CandidateSet, CandidateSetHasher, CandidateSetCompare> candidates;
            ankerl::unordered_dense::map<CandidateSet, ViolationResult, CandidateSetHasher>     candidate_cache;
            ThreadLocalData() {
                candidates.reserve(1000);
                candidate_cache.reserve(1000);
            }
        };

        const size_t                 numThreads = std::thread::hardware_concurrency();
        std::vector<ThreadLocalData> threadData;
        for (size_t t = 0; t < numThreads; ++t) { threadData.emplace_back(); }

        ankerl::unordered_dense::set<CandidateSet, CandidateSetHasher, CandidateSetCompare> finalCandidates;
        finalCandidates.reserve(numThreads * 1000);

        // Use a lambda that accepts an explicit thread index.
        auto process_vertex = [&](std::size_t i, size_t threadID) {
            // Skip first and last vertices.
            if (i == 0 || i == N_SIZE - 1) return;
            auto &localData = threadData[threadID % numThreads];

            // Build a lookup for heuristic memory from scores[i].
            const auto                       &heuristic_memory_scores = scores[i];
            ankerl::unordered_dense::set<int> heuristic_memory_lookup;
            heuristic_memory_lookup.reserve(heuristic_memory_scores.size());
            std::vector<double> node_score_lookup(N_SIZE, std::numeric_limits<double>::infinity());
            for (const auto &score : heuristic_memory_scores) {
                heuristic_memory_lookup.insert(score.node);
                node_score_lookup[score.node] = score.cost_score;
            }

            // Build the working set from routes that contain vertex i.
            std::vector<int> working_set;
            working_set.reserve(std::min<size_t>(MAX_WORKING_SET_SIZE, allPaths.size()));
            if (auto row_it = row_indices_map.find(static_cast<int>(i)); row_it != row_indices_map.end()) {
                for (int r : row_it->second) {
                    for (const auto &v : allPaths[r].route) {
                        if (v <= 0 || v >= N_SIZE - 1 || v == static_cast<int>(i) ||
                            !heuristic_memory_lookup.contains(v))
                            continue;
                        if (std::find(working_set.begin(), working_set.end(), v) == working_set.end()) {
                            working_set.push_back(v);
                        }
                    }
                }
            }

            // Add additional memory-based candidates for vertices with sparse
            // route support.
            if (working_set.size() < static_cast<size_t>(MIN_RANK - 1) &&
                i < static_cast<size_t>(rank1_sep_heur_mem4_vertex.size())) {
                for (int v : rank1_sep_heur_mem4_vertex[i]) {
                    if (static_cast<int>(working_set.size()) >= MAX_WORKING_SET_SIZE) break;
                    if (v == static_cast<int>(i) ||
                        std::find(working_set.begin(), working_set.end(), v) != working_set.end())
                        continue;
                    working_set.push_back(v);
                }
            }

            // Limit working_set size using a heuristic if necessary.
            if (working_set.size() > MAX_WORKING_SET_SIZE) {
                std::vector<int> working_set_vec(working_set.begin(), working_set.end());
                std::nth_element(working_set_vec.begin(), working_set_vec.begin() + MAX_WORKING_SET_SIZE,
                                 working_set_vec.end(),
                                 [&](int a, int b) { return node_score_lookup[a] < node_score_lookup[b]; });
                working_set_vec.resize(MAX_WORKING_SET_SIZE);
                working_set = std::move(working_set_vec);
            }

            // Helper to build candidate-specific memory from the working set.
            auto build_candidate_memory = [&](const std::vector<int> &candidate_nodes) {
                ankerl::unordered_dense::set<int> memory;
                memory.reserve(working_set.size());
                for (int v : working_set) {
                    if (std::find(candidate_nodes.begin(), candidate_nodes.end(), v) == candidate_nodes.end()) {
                        memory.insert(v);
                    }
                }
                return memory;
            };

            // Helper lambda to process a candidate set.
            auto process_candidate = [&](const std::vector<int> &candidate_nodes) {
                if (static_cast<int>(candidate_nodes.size()) < MIN_RANK ||
                    static_cast<int>(candidate_nodes.size()) > MAX_RANK)
                    return false;

                auto         candidate_memory = build_candidate_memory(candidate_nodes);
                CandidateSet temp_candidate(candidate_nodes,
                                            0.0,                   // dummy violation
                                            SRCPermutation({}, 0), // dummy permutation
                                            candidate_memory, 0.0);
                // Check cache to avoid duplicate work.
                auto            cache_it = localData.candidate_cache.find(temp_candidate);
                ViolationResult result;
                if (cache_it != localData.candidate_cache.end()) {
                    result = cache_it->second;
                } else {
                    std::tie(result.violation, result.perm, result.rhs) =
                        computeViolationWithBestPerm(candidate_nodes, candidate_memory, A, x);
                    localData.candidate_cache.emplace(temp_candidate, result);
                }
                if (result.violation > 1e-3) {
                    CandidateSet candidate_set(candidate_nodes, result.violation, result.perm, candidate_memory,
                                               result.rhs);
                    insertBestCandidate(localData.candidates, std::move(candidate_set));
                    return true;
                }
                return false;
            };

            // Build the working set vector sorted by heuristic score.
            std::vector<int> working_set_vec(working_set.begin(), working_set.end());
            std::sort(working_set_vec.begin(), working_set_vec.end(),
                      [&](int a, int b) { return node_score_lookup[a] < node_score_lookup[b]; });

            // --- 1. Generate promising root candidates from the best
            // scored working set neighbors.
            const int max_root_size = std::min(static_cast<int>(working_set_vec.size()), MAX_RANK - 1);
            for (int root_size = MIN_RANK - 1; root_size <= max_root_size; ++root_size) {
                std::vector<int> candidate_nodes;
                candidate_nodes.reserve(root_size + 1);
                candidate_nodes.push_back(static_cast<int>(i));
                for (int idx = 0; idx < root_size; ++idx) { candidate_nodes.push_back(working_set_vec[idx]); }
                std::sort(candidate_nodes.begin(), candidate_nodes.end());
                process_candidate(candidate_nodes);
            }

            // --- 2. Generate additional candidate sets by combining vertices
            // ---
            if (working_set.size() > 1) {
                size_t combination_count = 0;
                // Make sure the working set is sorted by score.
                // Generate combinations of size k.
                for (size_t k = 2; k <= MAX_RANK && combination_count < MAX_COMBINATIONS; ++k) {
                    if (k > working_set_vec.size()) break;
                    // Initialize combination bitmask: k ones.
                    uint32_t mask = (1u << k) - 1;
                    while (mask < (1u << working_set_vec.size()) && combination_count < MAX_COMBINATIONS) {
                        std::vector<int> candidate_nodes;
                        candidate_nodes.reserve(k + 1);
                        candidate_nodes.push_back(static_cast<int>(i));
                        // Insert vertices corresponding to bits set in mask.
                        for (size_t bit = 0; bit < working_set_vec.size(); ++bit) {
                            if (mask & (1u << bit)) candidate_nodes.push_back(working_set_vec[bit]);
                        }
                        std::sort(candidate_nodes.begin(), candidate_nodes.end());
                        if (process_candidate(candidate_nodes)) ++combination_count;

                        // Gosper’s hack to get next combination.
                        uint32_t c = mask & -mask;
                        uint32_t r = mask + c;
                        if (c == 0) break; // safeguard against infinite loop
                        mask = (((r ^ mask) >> 2) / c) | r;
                    }
                }
            }
        };

        // Parallelize over vertices. Here we assign explicit thread indices.
        std::vector<std::thread> threads;
        std::atomic_size_t       vertexIndex{1}; // starting from 1; skipping index 0
        for (size_t t = 0; t < numThreads; ++t) {
            threads.emplace_back([&, t]() {
                while (true) {
                    size_t i = vertexIndex.fetch_add(1);
                    if (i >= N_SIZE - 1) break;
                    process_vertex(i, t);
                }
            });
        }
        for (auto &th : threads) th.join();

        // Merge thread-local candidate sets into finalCandidates.
        for (const auto &data : threadData) {
            for (const auto &candidate : data.candidates) { insertBestCandidate(finalCandidates, candidate); }
        }
        return finalCandidates;
    }

    ankerl::unordered_dense::map<int, std::vector<SRCPermutation>> permutations_cache;

    // ── Plan-group infrastructure (RouteOpt-style per-plan local search) ─────
    // A "plan" is the equivalence class of all permutations sharing the same
    // sorted numerator vector and denominator.  Grouping allows us to restrict
    // local-search evaluation to a single plan rather than retrying ALL
    // permutations on every add/remove/swap step.
    struct PlanGroup {
        std::vector<int>                    sorted_num;
        int                                 den = 0;
        std::vector<const SRCPermutation *> perms; // all orderings of this plan
    };
    // plan_groups_map[rank] = ordered list of PlanGroups for that rank
    ankerl::unordered_dense::map<int, std::vector<PlanGroup>> plan_groups_map;

    void buildPlanGroupsMap() {
        plan_groups_map.clear();
        for (const auto &[rank, perm_vec] : permutations_cache) {
            // Use a sorted map so plan groups are in deterministic order.
            std::map<std::pair<std::vector<int>, int>, std::vector<const SRCPermutation *>> tmp;
            for (const auto &p : perm_vec) {
                std::vector<int> sorted = p.num;
                std::sort(sorted.begin(), sorted.end());
                tmp[{sorted, p.den}].push_back(&p);
            }
            auto &groups = plan_groups_map[rank];
            groups.reserve(tmp.size());
            for (const auto &[key, ptrs] : tmp) {
                PlanGroup pg;
                pg.sorted_num = key.first;
                pg.den        = key.second;
                pg.perms      = ptrs;
                groups.push_back(std::move(pg));
            }
        }
    }

    // Evaluate the best violation over all orderings of a SINGLE plan group.
    // C, AM, and node_order must already be populated for the current node set.
    // The caller must clean up node_order entries after the call.
    double evalPlanGroup(const std::array<uint64_t, num_words> &C,
                         const std::array<uint64_t, num_words> &AM,
                         std::vector<int>                      &node_order,
                         const std::vector<int>                &paths,
                         const PlanGroup                       &pg,
                         const std::vector<double>             &x,
                         const SRCPermutation                 *&best_perm_out) const {
        double                best_vio  = 0.0;
        const SRCPermutation *best_perm = nullptr;
        for (const auto *pp : pg.perms) {
            const double rhs = pp->getRHS();
            double       lhs = 0.0;
            for (int r : paths) {
                if (static_cast<size_t>(r) >= x.size()) continue;
                const double xv = x[r];
                if (numericutils::isZero(xv)) continue;
                lhs += static_cast<double>(
                           computeLimitedMemoryCoefficient(C, AM, *pp, allPaths[r].route, node_order)) *
                       xv;
            }
            const double vio = lhs - rhs;
            if (vio > best_vio) {
                best_vio  = vio;
                best_perm = pp;
            }
        }
        best_perm_out = best_perm;
        return best_vio;
    }

    // Find the plan group with the highest violation for a given node set.
    // Internally sets up C/AM/node_order, evaluates all plan groups, cleans up.
    // Returns {violation, plan_group_index(-1 if none), best_perm}.
    std::tuple<double, int, const SRCPermutation *>
    findBestPlanGroupIdx(const std::vector<int>                    &nodes,
                         const ankerl::unordered_dense::set<int>   &mem,
                         const std::vector<int>                    &paths,
                         const std::vector<double>                 &x) const {
        const int RANK = static_cast<int>(nodes.size());
        auto      it   = plan_groups_map.find(RANK);
        if (it == plan_groups_map.end()) return {0.0, -1, nullptr};
        const auto &groups = it->second;

        std::array<uint64_t, num_words> C{}, AM{};
        thread_local std::vector<int>   no;
        if (no.size() < N_SIZE) no.assign(N_SIZE, -1);

        int pos = 0;
        for (int n : nodes) {
            C[n / 64] |= bit_mask_lookup[n % 64];
            AM[n / 64] |= bit_mask_lookup[n % 64];
            no[n] = pos++;
        }
        for (int n : mem) AM[n / 64] |= bit_mask_lookup[n % 64];

        double                best_vio  = 0.0;
        int                   best_gi   = -1;
        const SRCPermutation *best_perm = nullptr;
        for (int gi = 0; gi < static_cast<int>(groups.size()); ++gi) {
            const SRCPermutation *perm = nullptr;
            double                vio  = evalPlanGroup(C, AM, no, paths, groups[gi], x, perm);
            if (vio > best_vio) {
                best_vio  = vio;
                best_gi   = gi;
                best_perm = perm;
            }
        }
        for (int n : nodes) no[n] = -1; // restore thread-local
        return {best_vio, best_gi, best_perm};
    }

    // ── RouteOpt-style greedy high-dim cut search ─────────────────────────────
    // For each seed the function runs a deterministic add/remove/swap greedy
    // local search restricted to one plan group at a time.
    //
    // Performance notes (vs. previous version):
    //  • Seeds are processed in parallel via stdexec::bulk.
    //  • SWAP step uses a precomputed C_base/AM_base and flips only 2 bits per
    //    candidate (O(1)) instead of rebuilding O(rank) from scratch.
    //  • Working neighbourhood is a sorted std::vector for cache-friendly scan.
    //  • Seeds are built exactly once and reused from separate().
    void getHighDimCutsRouteOpt(const std::vector<double> &x, const SeedMap &seed_map) {
        if (allPaths.empty() || plan_groups_map.empty() || seed_map.empty()) return;

        struct HarvestedCut {
            double                            vio  = 0.0;
            std::vector<int>                  nodes;
            ankerl::unordered_dense::set<int> mem;
            const SRCPermutation             *perm = nullptr;
        };

        // Flatten to vector so stdexec::bulk can address by index.
        using SeedPair = std::pair<std::vector<int>, std::vector<int>>;
        std::vector<SeedPair> seeds_vec(seed_map.begin(), seed_map.end());

        std::vector<HarvestedCut> all_harvested;
        all_harvested.reserve(seeds_vec.size());
        std::mutex harvest_mutex;

        // ── parallel over seeds ───────────────────────────────────────────────
        auto bulk_sender = stdexec::bulk(
            stdexec::just(), seeds_vec.size(),
            [&](std::size_t idx) {
                const auto &[nodes_vec, mem_vec] = seeds_vec[idx];
                if (nodes_vec.empty()) return;
                const int RANK = static_cast<int>(nodes_vec.size());
                if (RANK < MIN_RANK || RANK > MAX_RANK) return;
                if (plan_groups_map.find(RANK) == plan_groups_map.end()) return;

                // Thread-local scratch storage (one copy per OS thread).
                thread_local std::vector<int> cand_paths;
                thread_local std::vector<int> no2;
                if (no2.size() < N_SIZE) no2.assign(N_SIZE, -1);

                ankerl::unordered_dense::set<int> cur_mem(mem_vec.begin(), mem_vec.end());
                std::vector<int>                  cur_nodes = nodes_vec; // sorted

                collectCandidatePathUnion(cur_nodes, cand_paths);
                if (cand_paths.empty()) return;

                auto [cur_vio, cur_gi, cur_perm] = findBestPlanGroupIdx(cur_nodes, cur_mem, cand_paths, x);
                if (cur_vio <= 1e-4 || cur_gi < 0) return;

                // ── working neighbourhood as sorted small-vector ──────────────
                std::vector<int> w_no_c;
                w_no_c.reserve(32);
                const auto w_insert = [&](int n) {
                    auto it = std::lower_bound(w_no_c.begin(), w_no_c.end(), n);
                    if (it == w_no_c.end() || *it != n) w_no_c.insert(it, n);
                };
                const auto w_erase = [&](int n) {
                    auto it = std::lower_bound(w_no_c.begin(), w_no_c.end(), n);
                    if (it != w_no_c.end() && *it == n) w_no_c.erase(it);
                };
                for (int n : cur_nodes) {
                    if (n >= static_cast<int>(rank1_sep_heur_mem4_vertex.size())) continue;
                    for (int nb : rank1_sep_heur_mem4_vertex[n]) {
                        if (nb > 0 && nb < N_SIZE - 1 &&
                            !std::binary_search(cur_nodes.begin(), cur_nodes.end(), nb))
                            w_insert(nb);
                    }
                }
                for (int n : mem_vec) {
                    if (n > 0 && n < N_SIZE - 1 &&
                        !std::binary_search(cur_nodes.begin(), cur_nodes.end(), n))
                        w_insert(n);
                }

                // ── greedy loop ───────────────────────────────────────────────
                constexpr int MAX_GREEDY_STEPS = 8;
                for (int step = 0; step < MAX_GREEDY_STEPS; ++step) {
                    double best_delta    = 1e-5;
                    int    best_type     = 0; // 0=stay, 1=add, 2=remove, 3=swap
                    int    best_add      = -1, best_rem = -1;
                    int    best_swap_out = -1, best_swap_in = -1;
                    const int cur_rank   = static_cast<int>(cur_nodes.size());

                    // ── SWAP (rank unchanged → same plan group) ───────────────
                    // Precompute C_base/AM_base once; flip 2 bits per candidate.
                    {
                        auto pgit = plan_groups_map.find(cur_rank);
                        if (pgit != plan_groups_map.end() && cur_gi >= 0 &&
                            cur_gi < static_cast<int>(pgit->second.size())) {
                            const auto &pg = pgit->second[cur_gi];

                            std::array<uint64_t, num_words> C_base{}, AM_base{};
                            for (int i = 0; i < cur_rank; ++i) {
                                const int n = cur_nodes[i];
                                C_base[n / 64]  |= bit_mask_lookup[n % 64];
                                AM_base[n / 64] |= bit_mask_lookup[n % 64];
                            }
                            for (int n : cur_mem) AM_base[n / 64] |= bit_mask_lookup[n % 64];

                            for (int out_idx = 0; out_idx < cur_rank; ++out_idx) {
                                const int out_node = cur_nodes[out_idx];
                                for (const int in_node : w_no_c) {
                                    // Build swapped node set
                                    std::vector<int> new_nodes = cur_nodes;
                                    new_nodes[out_idx]         = in_node;
                                    std::sort(new_nodes.begin(), new_nodes.end());

                                    // C2/AM2: O(1) bit-flip from base
                                    std::array<uint64_t, num_words> C2  = C_base;
                                    C2[out_node / 64] &= ~bit_mask_lookup[out_node % 64];
                                    C2[in_node / 64]  |=  bit_mask_lookup[in_node  % 64];

                                    std::array<uint64_t, num_words> AM2 = AM_base;
                                    if (!cur_mem.count(out_node))
                                        AM2[out_node / 64] &= ~bit_mask_lookup[out_node % 64];
                                    AM2[in_node / 64] |= bit_mask_lookup[in_node % 64];

                                    // no2: rebuild for new_nodes (O(rank), necessary)
                                    int pos3 = 0;
                                    for (int n : new_nodes) no2[n] = pos3++;

                                    collectCandidatePathUnion(new_nodes, cand_paths);
                                    const SRCPermutation *pp  = nullptr;
                                    const double          vio2 =
                                        evalPlanGroup(C2, AM2, no2, cand_paths, pg, x, pp);
                                    for (int n : new_nodes) no2[n] = -1;

                                    if (vio2 - cur_vio > best_delta) {
                                        best_delta    = vio2 - cur_vio;
                                        best_type     = 3;
                                        best_swap_out = out_node;
                                        best_swap_in  = in_node;
                                    }
                                }
                            }
                            // No outer no2 cleanup needed — we never set it in the outer loop.
                        }
                    }

                    // ── ADD (rank increases → re-find plan) ───────────────────
                    if (cur_rank < MAX_RANK) {
                        for (const int nb : w_no_c) {
                            std::vector<int> new_nodes = cur_nodes;
                            new_nodes.push_back(nb);
                            std::sort(new_nodes.begin(), new_nodes.end());
                            collectCandidatePathUnion(new_nodes, cand_paths);
                            auto [vio2, gi2, pp2] =
                                findBestPlanGroupIdx(new_nodes, cur_mem, cand_paths, x);
                            if (vio2 - cur_vio > best_delta) {
                                best_delta = vio2 - cur_vio;
                                best_type  = 1;
                                best_add   = nb;
                            }
                        }
                    }

                    // ── REMOVE (rank decreases → re-find plan) ────────────────
                    if (cur_rank > MIN_RANK) {
                        for (int ri = 0; ri < cur_rank; ++ri) {
                            std::vector<int> new_nodes;
                            new_nodes.reserve(cur_rank - 1);
                            for (int j2 = 0; j2 < cur_rank; ++j2)
                                if (j2 != ri) new_nodes.push_back(cur_nodes[j2]);
                            collectCandidatePathUnion(new_nodes, cand_paths);
                            auto [vio2, gi2, pp2] =
                                findBestPlanGroupIdx(new_nodes, cur_mem, cand_paths, x);
                            // Small bonus: simpler cut is cheaper in pricing.
                            if (vio2 - cur_vio > best_delta + 5e-5) {
                                best_delta = vio2 - cur_vio;
                                best_type  = 2;
                                best_rem   = cur_nodes[ri];
                            }
                        }
                    }

                    if (best_type == 0) break; // converged

                    // ── apply best move ───────────────────────────────────────
                    if (best_type == 1) { // ADD
                        cur_nodes.push_back(best_add);
                        std::sort(cur_nodes.begin(), cur_nodes.end());
                        w_erase(best_add);
                        if (best_add < static_cast<int>(rank1_sep_heur_mem4_vertex.size())) {
                            for (int nb : rank1_sep_heur_mem4_vertex[best_add]) {
                                if (nb > 0 && nb < N_SIZE - 1 &&
                                    !std::binary_search(cur_nodes.begin(), cur_nodes.end(), nb))
                                    w_insert(nb);
                            }
                        }
                        collectCandidatePathUnion(cur_nodes, cand_paths);
                        auto [vio2, gi2, pp2] = findBestPlanGroupIdx(cur_nodes, cur_mem, cand_paths, x);
                        cur_vio = vio2; cur_gi = gi2; cur_perm = pp2;

                    } else if (best_type == 2) { // REMOVE
                        auto it2 = std::find(cur_nodes.begin(), cur_nodes.end(), best_rem);
                        if (it2 != cur_nodes.end()) {
                            w_insert(best_rem);
                            cur_nodes.erase(it2);
                        }
                        collectCandidatePathUnion(cur_nodes, cand_paths);
                        auto [vio2, gi2, pp2] = findBestPlanGroupIdx(cur_nodes, cur_mem, cand_paths, x);
                        cur_vio = vio2; cur_gi = gi2; cur_perm = pp2;

                    } else { // SWAP
                        auto it2 = std::find(cur_nodes.begin(), cur_nodes.end(), best_swap_out);
                        if (it2 != cur_nodes.end()) *it2 = best_swap_in;
                        std::sort(cur_nodes.begin(), cur_nodes.end());
                        w_erase(best_swap_in);
                        w_insert(best_swap_out);
                        // Same rank → stay in same plan group, re-eval incrementally
                        const int new_rank = static_cast<int>(cur_nodes.size());
                        auto      pgit2    = plan_groups_map.find(new_rank);
                        if (pgit2 != plan_groups_map.end() && cur_gi >= 0 &&
                            cur_gi < static_cast<int>(pgit2->second.size())) {
                            const auto &pg2 = pgit2->second[cur_gi];
                            std::array<uint64_t, num_words> C3{}, AM3{};
                            int pos3 = 0;
                            for (int n : cur_nodes) {
                                C3[n / 64]  |= bit_mask_lookup[n % 64];
                                AM3[n / 64] |= bit_mask_lookup[n % 64];
                                no2[n] = pos3++;
                            }
                            for (int n : cur_mem) AM3[n / 64] |= bit_mask_lookup[n % 64];
                            collectCandidatePathUnion(cur_nodes, cand_paths);
                            const SRCPermutation *pp2 = nullptr;
                            cur_vio  = evalPlanGroup(C3, AM3, no2, cand_paths, pg2, x, pp2);
                            cur_perm = pp2;
                            for (int n : cur_nodes) no2[n] = -1;
                        } else {
                            collectCandidatePathUnion(cur_nodes, cand_paths);
                            auto [vio2, gi2, pp2] =
                                findBestPlanGroupIdx(cur_nodes, cur_mem, cand_paths, x);
                            cur_vio = vio2; cur_gi = gi2; cur_perm = pp2;
                        }
                    }
                } // end greedy steps

                if (cur_vio > 1e-4 && cur_perm != nullptr) {
                    std::lock_guard<std::mutex> lock(harvest_mutex);
                    all_harvested.push_back({cur_vio, cur_nodes, std::move(cur_mem), cur_perm});
                }
            }); // end bulk lambda

        stdexec::sync_wait(stdexec::starts_on(sched, std::move(bulk_sender)));

        if (all_harvested.empty()) return;

        pdqsort(all_harvested.begin(), all_harvested.end(),
                [](const auto &a, const auto &b) { return a.vio > b.vio; });

        constexpr int MAX_GREEDY_CUTS = 6;
        int           added           = 0;
        for (const auto &hc : all_harvested) {
            if (added >= MAX_GREEDY_CUTS || hc.vio <= 1e-4) break;
            if (!hc.perm) continue;
            CandidateSet cs(hc.nodes, hc.vio, *hc.perm, hc.mem, hc.perm->getRHS());
            if (addCutToCutStorage(cs, x)) ++added;
        }
    }

    const std::vector<SRCPermutation> &getPermutations(const int RANK) {
        // check if permutations are already computed
        if (permutations_cache.find(RANK) == permutations_cache.end()) {
            std::throw_with_nested(
                std::runtime_error("Permutations for rank " + std::to_string(RANK) + " not found in cache"));
        }
        return permutations_cache[RANK];
        // return perms;
        // if (RANK == 3) {
        //     return getPermutationsForSize3();
        // } else if (RANK == 4) {
        //     return getPermutationsForSize4();
        // } else if (RANK == 5) {
        //     return getPermutationsForSize5();
        // }
        // return {};
    }

    ankerl::unordered_dense::set<int> buildFullMemory(const std::vector<int> &candidate_nodes) const {
        ankerl::unordered_dense::set<int> memory;
        memory.reserve(N_SIZE);
        std::vector<char> in_candidate(N_SIZE, 0);
        for (int node : candidate_nodes) {
            if (node > 0 && node < N_SIZE - 1) in_candidate[node] = 1;
        }
        for (int node = 1; node < N_SIZE - 1; ++node) {
            if (!in_candidate[node]) memory.insert(node);
        }
        return memory;
    }

    ankerl::unordered_dense::set<int> deriveLimitedMemory(const std::vector<int> &candidate_nodes,
                                                          const SRCPermutation &perm,
                                                          const std::vector<double> &x,
                                                          int max_memory_nodes = 18) const {
        ankerl::unordered_dense::set<int> memory;
        memory.reserve(max_memory_nodes);

        std::array<uint64_t, num_words> C  = {};
        std::array<uint64_t, num_words> AM = {};
        std::vector<int>                order(N_SIZE, -1);
        std::vector<char>               in_candidate(N_SIZE, 0);
        for (int pos = 0; pos < static_cast<int>(candidate_nodes.size()); ++pos) {
            const int node = candidate_nodes[pos];
            C[node / 64] |= bit_mask_lookup[node % 64];
            AM[node / 64] |= bit_mask_lookup[node % 64];
            order[node] = pos;
            in_candidate[node] = 1;
        }

        thread_local std::vector<int> candidate_paths;
        collectCandidatePathUnion(candidate_nodes, candidate_paths);
        pdqsort(candidate_paths.begin(), candidate_paths.end(), [&](int a, int b) {
            const double xa = (a >= 0 && static_cast<size_t>(a) < x.size()) ? x[a] : 0.0;
            const double xb = (b >= 0 && static_cast<size_t>(b) < x.size()) ? x[b] : 0.0;
            return xa > xb;
        });

        constexpr int MAX_PATHS_FOR_MEMORY = 40;
        int           paths_seen           = 0;
        std::vector<int> base_positions;
        for (int path_idx : candidate_paths) {
            if (paths_seen >= MAX_PATHS_FOR_MEMORY || static_cast<int>(memory.size()) >= max_memory_nodes) break;
            if (path_idx < 0 || static_cast<size_t>(path_idx) >= allPaths.size() || static_cast<size_t>(path_idx) >= x.size())
                continue;
            if (numericutils::isZero(x[path_idx])) continue;
            const auto &route = allPaths[path_idx].route;
            base_positions.clear();
            for (int pos = 1; pos < static_cast<int>(route.size()) - 1; ++pos) {
                const int node = route[pos];
                if (node > 0 && node < N_SIZE - 1 && in_candidate[node]) { base_positions.push_back(pos); }
            }
            if (base_positions.size() < 2) continue;
            ++paths_seen;

            for (size_t b = 1; b < base_positions.size(); ++b) {
                const int from = base_positions[b - 1];
                const int to   = base_positions[b];
                for (int pos = from + 1; pos < to && static_cast<int>(memory.size()) < max_memory_nodes; ++pos) {
                    const int node = route[pos];
                    if (node <= 0 || node >= N_SIZE - 1 || in_candidate[node]) continue;
                    memory.insert(node);
                }
            }
        }

        for (int node : candidate_nodes) {
            if (node >= static_cast<int>(rank1_sep_heur_mem4_vertex.size())) continue;
            for (int neighbor : rank1_sep_heur_mem4_vertex[node]) {
                if (static_cast<int>(memory.size()) >= max_memory_nodes) break;
                if (neighbor <= 0 || neighbor >= N_SIZE - 1 || in_candidate[neighbor]) continue;
                memory.insert(neighbor);
            }
        }
        return memory;
    }

    std::optional<CandidateSet> evaluateBaseThenMemory(const std::vector<int> &candidate_nodes,
                                                       const SparseMatrix &A, const std::vector<double> &x) {
        auto full_memory = buildFullMemory(candidate_nodes);
        auto [full_violation, full_perm, full_rhs] = computeViolationWithBestPerm(candidate_nodes, full_memory, A, x);
        if (full_violation <= 1e-3) return std::nullopt;

        auto limited_memory = deriveLimitedMemory(candidate_nodes, full_perm, x);
        auto [limited_violation, limited_perm, limited_rhs] =
            computeViolationWithBestPerm(candidate_nodes, limited_memory, A, x);
        if (limited_violation <= 1e-3) return std::nullopt;
        return CandidateSet(candidate_nodes, limited_violation, limited_perm, limited_memory, limited_rhs);
    }

    void collectCandidatePathUnion(const std::vector<int> &candidate_nodes, std::vector<int> &candidate_paths) const {
        thread_local std::vector<uint32_t> seen_epoch;
        thread_local uint32_t              epoch = 1;

        if (seen_epoch.size() < allPaths.size()) { seen_epoch.assign(allPaths.size(), 0); }
        if (++epoch == 0) {
            std::fill(seen_epoch.begin(), seen_epoch.end(), 0);
            epoch = 1;
        }

        candidate_paths.clear();
        for (const int node : candidate_nodes) {
            if (node <= 0 || node >= N_SIZE - 1) continue;
            auto row_it = row_indices_map.find(node);
            if (row_it == row_indices_map.end()) continue;
            for (const int path_idx : row_it->second) {
                if (path_idx < 0 || static_cast<size_t>(path_idx) >= allPaths.size()) continue;
                if (seen_epoch[path_idx] == epoch) continue;
                seen_epoch[path_idx] = epoch;
                candidate_paths.push_back(path_idx);
            }
        }
    }

public:
    std::vector<std::vector<int>> vertex_route_map;

    std::tuple<double, SRCPermutation, double>
    computeViolationWithBestPerm(const std::vector<int> &nodes, const ankerl::unordered_dense::set<int> &memory,
                                 const SparseMatrix &A, const std::vector<double> &x) {
        static constexpr double EPSILON = 1e-6;
        std::vector<int>        node_vec(nodes);
        std::sort(node_vec.begin(), node_vec.end());
        const int RANK = static_cast<int>(node_vec.size());

        // Initialize result variables.
        double                best_violation = 0.0;
        double                best_rhs       = 0.0;
        const SRCPermutation *best_perm      = nullptr;

        // Initialize bit arrays for efficient set operations.
        std::array<uint64_t, num_words> candidate_bits{};        // all bits initially 0
        std::array<uint64_t, num_words> augmented_memory_bits{}; // all bits initially 0

        // Create a reusable node order lookup vector; initialize with -1.
        thread_local std::vector<int> node_order;
        if (node_order.size() < N_SIZE) {
            node_order.assign(N_SIZE, -1);
        } else {
            std::fill(node_order.begin(), node_order.end(), -1);
        }

        // Populate candidate_bits and node_order based on candidate nodes.
        int i = 0;
        for (auto node : node_vec) {
            const int      word_idx = node / 64;
            const uint64_t bit_mask = bit_mask_lookup[node % 64];

            candidate_bits[word_idx] |= bit_mask;
            augmented_memory_bits[word_idx] |= bit_mask;
            node_order[node] = i++;
        }

        // Add memory nodes to augmented_memory_bits.
        for (const int node : memory) {
            int word_idx = node / 64;
            augmented_memory_bits[word_idx] |= (1ULL << (node % 64));
        }

        // Get all permutations for the given rank (compute once).
        const auto &permutations = getPermutations(RANK);

        thread_local std::vector<int> candidate_paths;
        collectCandidatePathUnion(node_vec, candidate_paths);
        if (candidate_paths.empty()) { return {0.0, SRCPermutation({}, 0), 0.0}; }

        // Evaluate each permutation.
        for (const auto &perm : permutations) {
            // Compute the RHS value from the permutation.
            double rhs = perm.getRHS();
            // Create an SRCPermutation from the current permutation.
            const SRCPermutation src_perm{perm.num, perm.den};

            // Reset coefficient vector and accumulator for LHS.
            // cut_coefficients.clear();
            double lhs = 0.0;

            // Only routes touching the base set can receive a nonzero
            // limited-memory coefficient.
            for (int path_idx : candidate_paths) {
                if (static_cast<size_t>(path_idx) >= x.size()) continue;
                const double x_val = x[path_idx];
                if (numericutils::isZero(x_val)) continue;

                // Compute the coefficient using candidate_bits,
                // augmented_memory_bits, current permutation, the route from
                // the path, and the node ordering.
                const int coeff = computeLimitedMemoryCoefficient(candidate_bits, augmented_memory_bits, src_perm,
                                                                  allPaths[path_idx].route, node_order);
                // cut_coefficients.push_back(coeff);
                lhs += coeff * x_val;
            }

            // Compute the violation: the difference between lhs and (rhs +
            // EPSILON).
            const double violation = lhs - (rhs + EPSILON);
            if (violation > best_violation) {
                best_violation = violation;
                best_perm      = &perm;
                best_rhs       = rhs;
            }
        }

        // Return the best violation, corresponding permutation (or default if
        // none), and best rhs.
        return {best_violation, best_perm ? *best_perm : SRCPermutation({}, 0), best_rhs};
    }

    std::tuple<double, SRCPermutation, double>
    computeViolationWithBestPerm(const ankerl::unordered_dense::set<int> &nodes,
                                 const ankerl::unordered_dense::set<int> &memory, const SparseMatrix &A,
                                 const std::vector<double> &x) {
        std::vector<int> node_vec(nodes.begin(), nodes.end());
        std::sort(node_vec.begin(), node_vec.end());
        return computeViolationWithBestPerm(node_vec, memory, A, x);
    }

    Xoroshiro128Plus rng;

    bool addCutToCutStorage(const CandidateSet &candidate, const std::vector<double> &x) {
        // Initialize bit arrays for the candidate (C) and augmented memory
        // (AM).
        std::array<uint64_t, num_words> C  = {};
        std::array<uint64_t, num_words> AM = {};

        // Build a deterministic node order for the candidate.
        std::vector<int> node_ordered(candidate.nodes.begin(), candidate.nodes.end());
        std::sort(node_ordered.begin(), node_ordered.end());
        std::vector<int> order(N_SIZE, -1);
        int              ordering = 0;
        for (int node : node_ordered) { order[node] = ordering++; }

        // Set bits for candidate nodes.
        for (int node : node_ordered) {
            C[node / 64] |= bit_mask_lookup[node % 64];
            AM[node / 64] |= bit_mask_lookup[node % 64];
        }
        // Set bits for candidate neighbors into the augmented memory.
        for (auto node : candidate.neighbor) { AM[node / 64] |= bit_mask_lookup[node % 64]; }

        // Prepare the permutation from the candidate.
        SRCPermutation p;
        p.num = candidate.perm.num;
        p.den = candidate.perm.den;

        // Store only the nonzero coefficients. This keeps high-order cuts
        // lighter in memory while still allowing us to build the initial
        // master constraint exactly.
        std::vector<int>    coefficient_indices;
        std::vector<double> coefficient_values;
        coefficient_indices.reserve(allPaths.size() / 8 + 1);
        coefficient_values.reserve(allPaths.size() / 8 + 1);

        thread_local std::vector<int> candidate_paths;
        collectCandidatePathUnion(node_ordered, candidate_paths);

        double       exact_violation = 0.0;
        const double rhs             = p.getRHS();
        for (const int path_idx : candidate_paths) {
            if (static_cast<size_t>(path_idx) >= allPaths.size() || static_cast<size_t>(path_idx) >= x.size())
                continue;
            const double coeff = computeLimitedMemoryCoefficient(C, AM, p, allPaths[path_idx].route, order);
            exact_violation += coeff * x[path_idx];
            if (!numericutils::isZero(coeff)) {
                coefficient_indices.push_back(path_idx);
                coefficient_values.push_back(coeff);
            }
        }

        if (exact_violation <= rhs + 1e-6) { return false; }

        // Construct the Cut object using the candidate data.
        Cut cut(C, AM, {}, p);
        cut.baseSetOrder        = order;
        cut.coefficient_indices = std::move(coefficient_indices);
        cut.coefficient_values  = std::move(coefficient_values);

        // Set the cut type based on the number of candidate nodes.
        const auto cut_rank = candidate.nodes.size();
        if (cut_rank == 4) {
            cut.type = CutType::FourRow;
            // return;
        } else if (cut_rank == 5) {
            cut.type = CutType::FiveRow;
        }

        // Add the cut to the global cut storage.
        const size_t previous_size = cutStorage->size();
        cutStorage->addCut(cut);
        return cutStorage->size() > previous_size || cut.updated;
    }

    // std::unique_ptr<AdaptiveNodeScorer> ml_scorer;

public:
    // HighRankCuts() { ml_scorer = std::make_unique<AdaptiveNodeScorer>();
    // }
    HighRankCuts() {
        for (int i = MIN_RANK; i <= MAX_RANK; ++i) {
            fmt::print("Generating hybrid permutations for rank {}\n", i);
            auto perms = generateExactPermutations(i);
            auto extra = generateGeneticPermutations(i);
            for (auto &perm : extra) {
                auto duplicate = std::find_if(perms.begin(), perms.end(), [&](const SRCPermutation &existing) {
                    return existing.den == perm.den && existing.num == perm.num;
                });
                if (duplicate == perms.end()) { perms.push_back(std::move(perm)); }
            }
            permutations_cache[i] = std::move(perms);
        }
        buildPlanGroupsMap();
    }

    CutStorage *cutStorage = nullptr;

    std::vector<Path> allPaths;

    ArcDuals arc_duals;

    std::vector<VRPNode>                                nodes;
    std::vector<std::vector<double>>                    distances;
    ankerl::unordered_dense::map<int, std::vector<int>> row_indices_map;
    std::vector<int>                                    nonzero_paths;

    std::vector<double> solution;

    void separate(const SparseMatrix &A, const std::vector<double> &x) {
        // Set the solution.
        solution = x;
        initializeVertexRouteMap();
        nonzero_paths.clear();
        nonzero_paths.reserve(x.size());
        for (size_t path_idx = 0; path_idx < x.size(); ++path_idx) {
            if (!numericutils::isZero(x[path_idx])) { nonzero_paths.push_back(static_cast<int>(path_idx)); }
        }

        // Compute node scores.
        auto node_scores = computeNodeScores(A, x);
        initializeRank1Memory(node_scores);

        // Generate candidate sets.
        // (Assuming generateCandidates returns a container type that is not
        // random access.)
        // Seeds are built once here and reused by generateSeedCandidates and
        // getHighDimCutsRouteOpt to avoid the expensive O(N×R) construction twice.
        auto seed_map                     = constructHighRankSeeds();
        auto candidates_set               = generateCandidates(node_scores, A, x);
        auto seed_candidates              = generateSeedCandidates(A, x, seed_map);
        auto route_seed_candidates        = generateRouteSupportSeeds(A, x, node_scores);
        auto lifted_seed_candidates       = generateLiftedCutSeeds(A, x, node_scores);
        auto cooccurrence_seed_candidates = generateFractionalCooccurrenceSeeds(A, x, node_scores);
        for (const auto &candidate : seed_candidates) { insertBestCandidate(candidates_set, candidate); }
        for (const auto &candidate : route_seed_candidates) { insertBestCandidate(candidates_set, candidate); }
        for (const auto &candidate : lifted_seed_candidates) { insertBestCandidate(candidates_set, candidate); }
        for (const auto &candidate : cooccurrence_seed_candidates) {
            insertBestCandidate(candidates_set, candidate);
        }
        std::vector<CandidateSet> candidates(candidates_set.begin(), candidates_set.end());

        // Prepare for parallel local search.
        const size_t num_candidates = candidates.size();
        std::mutex   candidates_mutex;
        ankerl::unordered_dense::set<CandidateSet, CandidateSetHasher, CandidateSetCompare> improved_candidates;

        // Parallelize the local search using stdexec::bulk.
        auto bulk_sender = stdexec::bulk(
            stdexec::just(), num_candidates,
            [this, &A, &x, &node_scores, &improved_candidates, &candidates, &candidates_mutex](std::size_t idx) {
                // Each thread creates its own LocalSearch object.
                LocalSearch local_search(this);

                // Retrieve candidate via index.
                const CandidateSet &current_candidate = candidates[idx];
                const auto          current_violation = current_candidate.violation;

                // Perform local search to get potential improvements.
                auto improved_list = local_search.solve(current_candidate, A, x, node_scores);

                // Collect improved candidates locally to reduce locking
                // frequency.
                std::vector<CandidateSet> local_improved;
                for (const auto &candidate : improved_list) {
                    if (candidate.violation > current_violation) { local_improved.push_back(candidate); }
                }
                if (local_improved.empty()) { local_improved.push_back(current_candidate); }

                // Insert all local improved candidates into the global set with
                // one lock.
                {
                    std::lock_guard<std::mutex> lock(candidates_mutex);
                    for (const auto &candidate : local_improved) { insertBestCandidate(improved_candidates, candidate); }
                }
            });

        // Execute the bulk sender on the scheduler and wait for completion.
        auto work = stdexec::starts_on(sched, std::move(bulk_sender));
        stdexec::sync_wait(std::move(work));

        // Provide feedback using the improved candidates.
        ml_scorer.provideFeedback(improved_candidates);

        // Process and sort unique candidates based on violation and diversity.
        int                       initial_cut_size = cutStorage->size();
        std::vector<CandidateSet> unique_candidates(improved_candidates.begin(), improved_candidates.end());
        std::vector<int>          vertex_budget(N_SIZE, 8);
        for (const auto &cut : *cutStorage) {
            for (int node = 1; node < N_SIZE - 1; ++node) {
                const size_t   segment = static_cast<size_t>(node) >> 6;
                const uint64_t bit     = 1ULL << (node & 63);
                if ((cut.baseSet[segment] & bit) || (cut.neighbors[segment] & bit)) {
                    vertex_budget[node] = std::max(0, vertex_budget[node] - 1);
                }
            }
        }

        auto candidate_score = [](const CandidateSet &candidate) {
            return candidate.violation - 1e-3 * static_cast<double>(candidate.neighbor.size()) -
                   5e-4 * static_cast<double>(candidate.nodes.size());
        };
        pdqsort(unique_candidates.begin(), unique_candidates.end(),
                [&](const CandidateSet &a, const CandidateSet &b) { return candidate_score(a) > candidate_score(b); });

        // Add cuts from top candidates while keeping the round diverse.
        const int max_cuts   = 10;
        int       cuts_added = 0;
        int       max_trials = 80;
        std::vector<int> touched_nodes;
        touched_nodes.reserve(MAX_RANK + MAX_WORKING_SET_SIZE);
        std::vector<char> touched_seen(N_SIZE, 0);
        for (const auto &candidate : unique_candidates) {
            if (cuts_added >= max_cuts || max_trials <= 0) break;

            touched_nodes.clear();
            for (const int node : candidate.nodes) {
                if (node <= 0 || node >= N_SIZE - 1 || touched_seen[node]) continue;
                touched_seen[node] = 1;
                touched_nodes.push_back(node);
            }
            for (const int node : candidate.neighbor) {
                if (node <= 0 || node >= N_SIZE - 1 || touched_seen[node]) continue;
                touched_seen[node] = 1;
                touched_nodes.push_back(node);
            }

            bool budget_available = true;
            for (const int node : touched_nodes) {
                if (vertex_budget[node] <= 0) {
                    budget_available = false;
                    break;
                }
            }
            for (const int node : touched_nodes) { touched_seen[node] = 0; }
            if (!budget_available) continue;

            --max_trials;
            if (!addCutToCutStorage(candidate, x)) continue;

            ++cuts_added;
            for (const int node : touched_nodes) { vertex_budget[node] = std::max(0, vertex_budget[node] - 1); }
        }
        int final_cut_size = cutStorage->size();

        // RouteOpt-style deterministic greedy search: runs AFTER the SA pass
        // so it can polish seeds that the stochastic search may have missed.
        // Reuses the seed_map built at the top of separate() — no extra construction.
        getHighDimCutsRouteOpt(x, seed_map);

        // Print summary of the candidate processing.
        print_cut("Candidates: {} | Improved Candidates: {} | Added {} SRC 3-4-5 "
                  "cuts\n",
                  candidates.size(), improved_candidates.size(), cutStorage->size() - initial_cut_size);
    }

    static constexpr std::array<uint64_t, 64> bit_mask_lookup = []() {
        std::array<uint64_t, 64> masks{};
        for (size_t i = 0; i < 64; ++i) { masks[i] = 1ULL << i; }
        return masks;
    }();

    int computeLimitedMemoryCoefficient(const std::array<uint64_t, num_words> &C,
                                        const std::array<uint64_t, num_words> &AM, const SRCPermutation &p,
                                        const std::vector<uint16_t> &P, std::vector<int> &order) const noexcept {
        int       alpha          = 0.0;
        int       cumulative_sum = 0;
        const int denominator    = p.den;

#if defined(SRC_MEMORY_MODE_ARC)
        // Arc memory mode: preserve state only across memory arcs; entering a
        // base vertex starts a new state if the previous state was reset.
        for (size_t j = 1; j < P.size() - 1; ++j) {
            const int vertex      = P[j];
            const int prev_vertex = P[j - 1];

            const size_t   word_index    = vertex >> 6;
            const uint64_t bit_mask      = 1ULL << (vertex & 63);
            const size_t   prev_word     = prev_vertex >> 6;
            const uint64_t prev_bit_mask = 1ULL << (prev_vertex & 63);

            if (!(AM[word_index] & bit_mask) || !(AM[prev_word] & prev_bit_mask)) {
                cumulative_sum = 0;
            }
            if (C[word_index] & bit_mask) {
                int pos = order[vertex];
                cumulative_sum += p.num[pos];
                if (cumulative_sum >= denominator) {
                    cumulative_sum -= denominator;
                    alpha += 1;
                }
            }
        }
#else
        for (size_t j = 1; j < P.size() - 1; ++j) {
            const int vertex = P[j];

            const size_t   word_index = vertex >> 6;
            const uint64_t bit_mask   = 1ULL << (vertex & 63);

            if (!(AM[word_index] & bit_mask)) {
                cumulative_sum = 0;
            } else if (C[word_index] & bit_mask) {
                int pos = order[vertex];
                cumulative_sum += p.num[pos];
                if (cumulative_sum >= denominator) {
                    cumulative_sum -= denominator;
                    alpha += 1;
                }
            }
        }
#endif

        return alpha;
    }
};
