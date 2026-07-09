/**
 * @file SRC.cpp
 * @brief Implementation of functions and classes for solving VRP problems using
 *
 */

#include "cuts/SRC.h"

#include "bnb/Node.h"
#include "core/Definitions.h"
#include "cuts/Cut.h"
#include "cuts/HeuristicHighOrder.h"
#ifdef IPM
#include "ipm/IPSolver.h"
#endif

#ifdef HIGHS
#include <Highs.h>
#endif
using Cuts = std::vector<Cut>;

#include "cuts/CutIntelligence.h"

namespace {
constexpr double SRC_SEPARATION_TOL          = 1e-3;
constexpr int    MAX_VERTEX_RELATED_SRC_CUTS = 8;

#ifndef MAX_RANK3_SRC_CUTS_PER_ROUND
#define MAX_RANK3_SRC_CUTS_PER_ROUND 150
#endif

constexpr int MAX_RANK3_CUTS_PER_ROUND = MAX_RANK3_SRC_CUTS_PER_ROUND;

std::vector<int> collect_cut_nodes(const Cut &cut, bool include_neighbors = false) {
    std::vector<int> nodes;
    nodes.reserve(cut.p.num.size());
    for (int node = 1; node < N_SIZE - 1; ++node) {
        const size_t   segment = static_cast<size_t>(node) >> 6;
        const uint64_t bit     = 1ULL << (node & 63);
        if ((cut.baseSet[segment] & bit) || (include_neighbors && (cut.neighbors[segment] & bit))) {
            nodes.push_back(node);
        }
    }
    return nodes;
}

void set_sparse_coefficients(Cut &cut, const std::vector<int> &indices, const std::vector<double> &values) {
    cut.coefficient_indices.clear();
    cut.coefficient_values.clear();
    cut.coefficient_indices.reserve(indices.size());
    cut.coefficient_values.reserve(values.size());
    for (size_t idx = 0; idx < indices.size(); ++idx) {
        if (!numericutils::isZero(values[idx])) {
            cut.coefficient_indices.push_back(indices[idx]);
            cut.coefficient_values.push_back(values[idx]);
        }
    }
}
} // namespace

/**
 * @brief Adds a cut to the CutStorage.
 *
 * This function adds a given cut to the CutStorage. It first computes a unique
 * key for the cut based on its base set and multipliers. If a cut with the same
 * key already exists in the storage, it updates the existing cut. Otherwise, it
 * adds the new cut to the storage and updates the necessary mappings.
 *
 */
void CutStorage::addCut(Cut &cut) {
    // Compute a unique key for the cut based on its base set and probability
    // data.
    cut.key = compute_cut_key(cut.baseSet, cut.p.num, cut.p.den);

    // Check if the cut already exists in our map.
    if (auto it = cutMaster_to_cut_map.find(cut.key); it != cutMaster_to_cut_map.end()) {
        // The cut already exists; update its id and merge neighbor information.
        cut.id                    = it->second;
        auto        old_cut       = cuts[cut.id];
        const auto &old_neighbors = old_cut.neighbors;
        for (size_t i = 0; i < num_words; ++i) { cut.neighbors[i] |= old_neighbors[i]; }
        // Preserve 'added' flag if the existing cut has been marked as added.
        if (old_cut.added) {
            cut.added   = true;
            cut.updated = true;
        }
        cut.separation_count        = old_cut.separation_count;
        cut.inclusion_count         = old_cut.inclusion_count;
        cut.active_count            = old_cut.active_count;
        cut.creation_epoch          = old_cut.creation_epoch;
        cut.last_separation_epoch   = old_cut.last_separation_epoch;
        cut.last_active_epoch       = old_cut.last_active_epoch;
        cut.last_nonzero_dual_epoch = old_cut.last_nonzero_dual_epoch;
        cut.last_violation          = old_cut.last_violation;
        cut.max_violation           = old_cut.max_violation;
        cut.total_violation         = old_cut.total_violation;
        cut.avg_dual_magnitude      = old_cut.avg_dual_magnitude;
        cuts[cut.id]                = cut;
    } else {
        // The cut is new; assign a new id and store it.
        cut.id = cuts.size();
        if (cut.creation_epoch == 0) { cut.creation_epoch = current_epoch_; }
        cuts.push_back(cut);
        cutMaster_to_cut_map[cut.key] = cut.id;
        indexCuts[cut.key].push_back(cut.id);
    }
}

LimitedMemoryRank1Cuts::LimitedMemoryRank1Cuts(std::vector<VRPNode> &nodes) : nodes(nodes) {}

/**
 * @brief Separates the given solution vector into cuts using Limited Memory
 * Rank-1 Cuts.
 *
 * This function generates cuts for the given solution vector `x` based on the
 * sparse model `A`. It uses a parallel approach to evaluate combinations of
 * indices and identify violations that exceed the specified threshold.
 *
 */
void LimitedMemoryRank1Cuts::separate(const SparseMatrix &A, const std::vector<double> &x) {
    if (tasks.empty() || A.num_cols == 0 || allPaths.empty()) return;

    // Determine parallel parameters.
    const int    JOBS       = std::max(1u, std::thread::hardware_concurrency());
    const int    chunk_size = (tasks.size() + JOBS - 1) / JOBS;
    const size_t num_chunks = (tasks.size() + chunk_size - 1) / chunk_size;

    // Reserve per-chunk storage to avoid locking.
    std::vector<std::vector<std::pair<double, Cut>>> chunk_cuts(num_chunks);
    for (auto &vec : chunk_cuts) {
        vec.reserve(20); // Reserve an estimate; adjust as needed.
    }

    // Initialize SRCPermutation 'p' and compute a right-hand-side value.
    SRCPermutation p;
    p.num      = {1, 1, 1};
    p.den      = 2;
    double rhs = p.getRHS();

    // === Parallel Processing of Tasks in Chunks ===
    auto bulk_sender = stdexec::bulk(
        stdexec::just(), num_chunks, [this, &A, &x, chunk_size, &p, rhs, &chunk_cuts](std::size_t chunk_idx) {
            // Each chunk gets its own temporary vector.
            std::vector<std::pair<double, Cut>> local_cuts;
            std::vector<int>                    expanded(allPaths.size(), 0);
            std::vector<int>                    touched_paths;
            std::vector<int>                    candidate_paths;
            std::vector<double>                 candidate_coefficients;
            touched_paths.reserve(256);
            candidate_paths.reserve(128);
            candidate_coefficients.reserve(128);
            size_t start_idx = chunk_idx * chunk_size;
            size_t end_idx   = std::min(start_idx + chunk_size, tasks.size());

            // Process each task in this chunk.
            for (size_t task_idx = start_idx; task_idx < end_idx; ++task_idx) {
                // Unpack task parameters (i, j, k).
                const auto &[i, j, k] = tasks[task_idx];
                double lhs            = 0.0;
                candidate_paths.clear();

                const auto accumulate_node_paths = [&](int customer) {
                    if (auto it = row_indices_map.find(customer); it != row_indices_map.end()) {
                        for (int path_idx : it->second) {
                            if (path_idx < 0 || static_cast<size_t>(path_idx) >= allPaths.size()) continue;
                            if (expanded[path_idx]++ == 0) { touched_paths.push_back(path_idx); }
                        }
                    }
                };

                accumulate_node_paths(i);
                accumulate_node_paths(j);
                accumulate_node_paths(k);

                for (int path_idx : touched_paths) {
                    if (expanded[path_idx] >= 2) {
                        lhs += x[path_idx];
                        candidate_paths.push_back(path_idx);
                    }
                }

                // Only proceed if lhs exceeds threshold.
                if (lhs > rhs + 1e-3) {
                    // Initialize bitmask arrays for cut representation.
                    std::array<uint64_t, num_words> C  = {};
                    std::array<uint64_t, num_words> AM = {};
                    std::vector<int>                order(N_SIZE, 0);
                    int                             ordering = 0;

                    // Build the base cut indices from the task parameters.
                    std::array<int, 3> C_index = {i, j, k};
                    for (int node : C_index) {
                        C[node / 64] |= (1ULL << (node % 64));
                        AM[node / 64] |= (1ULL << (node % 64));
                        order[node] = ordering++;
                    }
                    const auto is_base_node = [&](int customer) {
                        return customer == i || customer == j || customer == k;
                    };

                    // Update AM based on the positions of nodes in each
                    // consumer path.
                    for (int path_idx : candidate_paths) {
                        const auto &consumers = allPaths[path_idx].route;
                        int         first = -1, second = -1;
                        // Identify first and second occurrences of elements
                        // from C_set.
                        for (size_t pos = 1; pos < consumers.size() - 1; ++pos) {
                            if (is_base_node(consumers[pos])) {
                                if (first == -1) {
                                    first = pos;
                                } else {
                                    second = pos;
                                    for (int pos_inner = first + 1; pos_inner < second; ++pos_inner) {
                                        AM[consumers[pos_inner] / 64] |= (1ULL << (consumers[pos_inner] % 64));
                                    }
                                    break;
                                }
                            }
                        }
                    }
                    // set all AM vector to 1
                    // for (int node_idx = 0; node_idx < N_SIZE; ++node_idx) {
                    // AM[node_idx / 64] |= (1ULL << (node_idx % 64));
                    // }

                    double exact_lhs = 0.0;
                    candidate_coefficients.clear();
                    candidate_coefficients.reserve(candidate_paths.size());
                    for (int path_idx : candidate_paths) {
                        if (static_cast<size_t>(path_idx) >= x.size()) {
                            candidate_coefficients.push_back(0.0);
                            continue;
                        }
                        const auto  &clients = allPaths[path_idx].route;
                        const double coeff   = computeLimitedMemoryCoefficient(C, AM, p, clients, order);
                        candidate_coefficients.push_back(coeff);
                        exact_lhs += coeff * x[path_idx];
                    }

                    if (exact_lhs > rhs + SRC_SEPARATION_TOL) {
                        Cut cut(C, AM, {}, p);
                        cut.baseSetOrder = order;
                        set_sparse_coefficients(cut, candidate_paths, candidate_coefficients);
                        local_cuts.emplace_back(exact_lhs - rhs, cut);
                    }
                }

                for (int path_idx : touched_paths) { expanded[path_idx] = 0; }
                touched_paths.clear();
            }
            // Save the local cuts into the global vector for this chunk.
            chunk_cuts[chunk_idx] = std::move(local_cuts);
        });

    // Submit the bulk work and wait for all tasks to complete.
    auto work = stdexec::starts_on(sched, bulk_sender);
    stdexec::sync_wait(std::move(work));

    // === Merge results from all chunks ===
    std::vector<std::pair<double, Cut>> tmp_cuts;
    for (const auto &vec : chunk_cuts) { tmp_cuts.insert(tmp_cuts.end(), vec.begin(), vec.end()); }

    // === Post-Processing: Sort and Add Cuts ===
    pdqsort(tmp_cuts.begin(), tmp_cuts.end(), [](const auto &a, const auto &b) { return a.first > b.first; });

    std::vector<int> vertex_cut_budget(N_SIZE, MAX_VERTEX_RELATED_SRC_CUTS);
    for (const auto &existing_cut : cutStorage) {
        for (int node : collect_cut_nodes(existing_cut, false)) {
            if (node >= 0 && node < N_SIZE) { --vertex_cut_budget[node]; }
        }
    }

    int cuts_added = 0;
    int max_trials = std::max(15, MAX_RANK3_CUTS_PER_ROUND * 3);
    for (auto &candidate : tmp_cuts) {
        if (cuts_added >= MAX_RANK3_CUTS_PER_ROUND || max_trials <= 0) break;
        if (candidate.first <= SRC_SEPARATION_TOL) break;

        auto &cut = candidate.second;
        cut.key   = cutStorage.compute_cut_key(cut.baseSet, cut.p.num, cut.p.den);
        if (cutStorage.cutExists(cut.key).first >= 0) {
            cutStorage.markCutSeparated(cut, candidate.first);
            continue;
        }

        auto nodes = collect_cut_nodes(cut, false);
        bool keep  = true;
        for (int node : nodes) {
            if (node >= 0 && node < N_SIZE && vertex_cut_budget[node] <= 0) {
                keep = false;
                break;
            }
        }
        if (!keep) { continue; }

        --max_trials;
        cutStorage.addCut(cut);
        cutStorage.markCutSeparated(cut, candidate.first);
        for (int node : nodes) {
            if (node >= 0 && node < N_SIZE) { --vertex_cut_budget[node]; }
        }
        ++cuts_added;
    }

    // Enforce pool size limit after adding cuts
    cutStorage.enforcePoolSizeLimit();
}

std::pair<bool, bool> LimitedMemoryRank1Cuts::runSeparation(BNBNode *node, std::vector<baldesCtrPtr> &SRCconstraints) {
    // Pointer to our cut storage
    auto *cuts = &cutStorage;

    // Extract the sparse model data from the node
    ModelData matrix = node->extractModelDataSparse();

    // Record the number of cuts before separation
    const size_t cuts_before = cuts->size();

    // Obtain the solution from the node (macro or function)
    std::vector<double> solution;
    GET_SOL(node);

    // fmt::print("Starting separation phase...\n");
    initializeVertexRouteMap();

    // Perform separation routines for Rank-1 cuts.
    // fmt::print("Separating Rank-1 cuts...\n");
    separateR1C1(matrix.A_sparse, solution);
    // fmt::print("Separation Rank-3 cuts...\n");
    const size_t initial_cut_count = cuts->size();
    separate(matrix.A_sparse, solution);
    // Adjacency-based rank-3 separation (RouteOpt generateR1C3 style):
    // focuses on triples where nodes are structurally adjacent in routes.
    separateR1C3Adjacency(matrix.A_sparse, solution);

    // Record the cut count after the first separation phase.
    int rank3_cuts_size = static_cast<int>(cuts->size() - initial_cut_count);

    ////////////////////////////////////////////////////
    // High-Rank Cuts Separation
    ////////////////////////////////////////////////////
    high_rank_cuts.cutStorage       = &cutStorage;
    high_rank_cuts.vertex_route_map = vertex_route_map;
    high_rank_cuts.allPaths         = allPaths;
    high_rank_cuts.nodes            = nodes;
    high_rank_cuts.arc_duals        = arc_duals;
    // fmt::print("Separating High-Rank cuts...\n");
    high_rank_cuts.separate(matrix.A_sparse, solution);

    const size_t cuts_after_separation = cuts->size();

    ////////////////////////////////////////////////////
    // Optionally, clean non-violated cuts (currently disabled)
    ////////////////////////////////////////////////////
    bool cleared = cutCleaner(node, SRCconstraints);
    // bool cleared = false;

    // Calculate how many cuts were removed (if any)
    const size_t n_cuts_removed = cuts_after_separation - cuts->size();

    // Determine if any cuts have changed compared to before the separation.
    bool cuts_changed = (cuts_before != cuts->size());
    if (!cuts_changed) {
        for (const auto &cut : *cuts) {
            if (!cut.added || cut.updated) {
                cuts_changed = true;
                break;
            }
        }
    }
    return std::make_pair(cuts_changed, cleared);
}

/**
 * @brief Adjacency-based rank-3 SRC separator (RouteOpt generateR1C3 style).
 *
 * For each customer pair (i,j):
 *   - If (i,j) are consecutive in some route  → candidate k's = union of their adjacency lists
 *   - Otherwise                               → candidate k's = intersection of their adjacency lists
 * Violation is screened with the fast floor formula Σ_r ⌊(v_i+v_j+v_k)/2⌋·x_r.
 * Promising triples are then re-evaluated with the exact limited-memory coefficient
 * before being added to the cut storage.
 *
 * This is called in addition to separate() and typically finds cuts with strong
 * structural locality that the O(n³) enumeration may reach less efficiently.
 */
void LimitedMemoryRank1Cuts::separateR1C3Adjacency(const SparseMatrix &A, const std::vector<double> &x) {
    if (allPaths.empty() || x.empty()) return;

    const int all_num_routes = static_cast<int>(allPaths.size());

    // ── Build adjacency graph ────────────────────────────────────────────────
    // i_connections[v] = sorted list of customer nodes appearing consecutively
    // with v in any route.  Depot nodes (0 and N_SIZE-1) are excluded.
    std::vector<std::vector<int>> i_connections(N_SIZE);
    for (const auto &path : allPaths) {
        const auto &route = path.route;
        for (int pos = 0; pos + 1 < static_cast<int>(route.size()); ++pos) {
            const int u = route[pos], v = route[pos + 1];
            if (u > 0 && u < N_SIZE - 1 && v > 0 && v < N_SIZE - 1) {
                i_connections[u].push_back(v);
                i_connections[v].push_back(u);
            }
        }
    }
    for (int v = 0; v < N_SIZE; ++v) {
        std::sort(i_connections[v].begin(), i_connections[v].end());
        i_connections[v].erase(std::unique(i_connections[v].begin(), i_connections[v].end()), i_connections[v].end());
    }

    // ── Standard rank-3 permutation ─────────────────────────────────────────
    SRCPermutation p;
    p.num            = {1, 1, 1};
    p.den            = 2;
    const double rhs = p.getRHS(); // = 1.0

    // ── Screening pass: collect violated (i,j,k) triples ───────────────────
    std::vector<std::pair<double, std::array<int, 3>>> cut_candidates;
    cut_candidates.reserve(1024);

    std::vector<int>  v_tmp;                     // candidate k nodes
    std::vector<int>  v_tmp2(all_num_routes, 0); // visit-count accumulator
    std::vector<int>  touched_routes;            // routes with v_tmp2 > 0
    std::vector<bool> in_touched(all_num_routes, false);
    touched_routes.reserve(256);

    for (int i = 1; i < N_SIZE - 1; ++i) {
        if (i_connections[i].empty()) continue;
        for (int j = i + 1; j < N_SIZE - 1; ++j) {
            if (i_connections[j].empty()) continue;

            // Generate candidate k set
            v_tmp.clear();
            const bool ij_adjacent = std::binary_search(i_connections[i].begin(), i_connections[i].end(), j);
            if (ij_adjacent) {
                std::set_union(i_connections[i].begin(), i_connections[i].end(), i_connections[j].begin(),
                               i_connections[j].end(), std::back_inserter(v_tmp));
            } else {
                std::set_intersection(i_connections[i].begin(), i_connections[i].end(), i_connections[j].begin(),
                                      i_connections[j].end(), std::back_inserter(v_tmp));
            }
            if (v_tmp.empty()) continue;

            // Accumulate visit counts for i and j
            touched_routes.clear();
            if (auto it = row_indices_map.find(i); it != row_indices_map.end()) {
                for (int r : it->second) {
                    if (!in_touched[r]) {
                        in_touched[r] = true;
                        touched_routes.push_back(r);
                    }
                    v_tmp2[r] += vertex_route_map[i][r];
                }
            }
            if (auto it = row_indices_map.find(j); it != row_indices_map.end()) {
                for (int r : it->second) {
                    if (!in_touched[r]) {
                        in_touched[r] = true;
                        touched_routes.push_back(r);
                    }
                    v_tmp2[r] += vertex_route_map[j][r];
                }
            }

            for (int k : v_tmp) {
                if (k <= j || k <= 0 || k >= N_SIZE - 1) continue;

                // Temporarily add k's visits
                if (auto it = row_indices_map.find(k); it != row_indices_map.end()) {
                    for (int r : it->second) {
                        if (!in_touched[r]) {
                            in_touched[r] = true;
                            touched_routes.push_back(r);
                        }
                        v_tmp2[r] += vertex_route_map[k][r];
                    }
                }

                // Floor-formula violation screening
                double vio = -rhs;
                for (int r : touched_routes) {
                    if (v_tmp2[r] >= 2 && static_cast<size_t>(r) < x.size()) vio += (v_tmp2[r] / 2) * x[r];
                }

                // Remove k's visits (restore (i,j)-only state)
                if (auto it = row_indices_map.find(k); it != row_indices_map.end()) {
                    for (int r : it->second) v_tmp2[r] -= vertex_route_map[k][r];
                }

                if (vio > SRC_SEPARATION_TOL) { cut_candidates.emplace_back(vio, std::array<int, 3>{i, j, k}); }
            }

            // Reset accumulators for the next (i,j) pair
            for (int r : touched_routes) {
                v_tmp2[r]     = 0;
                in_touched[r] = false;
            }
        }
    }

    if (cut_candidates.empty()) return;

    // Sort by descending violation
    pdqsort(cut_candidates.begin(), cut_candidates.end(),
            [](const auto &a, const auto &b) { return a.first > b.first; });

    // ── Budget tracking (mirror the logic in separate()) ────────────────────
    std::vector<int> vertex_cut_budget(N_SIZE, MAX_VERTEX_RELATED_SRC_CUTS);
    for (const auto &existing_cut : cutStorage) {
        for (int node : collect_cut_nodes(existing_cut, false)) {
            if (node >= 0 && node < N_SIZE) --vertex_cut_budget[node];
        }
    }

    // ── Exact evaluation + insertion ────────────────────────────────────────
    int cuts_added = 0;
    int max_trials = std::max(15, MAX_RANK3_CUTS_PER_ROUND * 3);

    std::vector<int> exact_visit_count(all_num_routes, 0);
    std::vector<int> exact_touched_routes;
    std::vector<int> exact_candidate_paths;
    exact_touched_routes.reserve(256);
    exact_candidate_paths.reserve(256);

    for (const auto &[rough_vio, ijk] : cut_candidates) {
        if (cuts_added >= MAX_RANK3_CUTS_PER_ROUND || max_trials <= 0) break;
        if (rough_vio <= SRC_SEPARATION_TOL) break;

        const int i = ijk[0], j = ijk[1], k = ijk[2];

        exact_touched_routes.clear();
        exact_candidate_paths.clear();
        const auto accumulate_base_visits = [&](int node) {
            auto row_it = row_indices_map.find(node);
            if (row_it == row_indices_map.end() || node < 0 || node >= static_cast<int>(vertex_route_map.size()))
                return;
            for (int r : row_it->second) {
                if (r < 0 || r >= all_num_routes || static_cast<size_t>(r) >= vertex_route_map[node].size()) continue;
                if (exact_visit_count[r] == 0) exact_touched_routes.push_back(r);
                exact_visit_count[r] += vertex_route_map[node][r];
            }
        };

        accumulate_base_visits(i);
        accumulate_base_visits(j);
        accumulate_base_visits(k);

        for (int r : exact_touched_routes) {
            if (exact_visit_count[r] >= 2 && static_cast<size_t>(r) < x.size() && !numericutils::isZero(x[r])) {
                exact_candidate_paths.push_back(r);
            }
        }

        // Build base set C and initial memory AM = C
        std::array<uint64_t, num_words> C  = {};
        std::array<uint64_t, num_words> AM = {};
        std::vector<int>                order(N_SIZE, -1);
        int                             ordering = 0;
        for (int node : {i, j, k}) {
            C[node / 64] |= (1ULL << (node % 64));
            AM[node / 64] |= (1ULL << (node % 64));
            order[node] = ordering++;
        }

        // Grow AM: add intermediate nodes between the first two cut-vertex
        // visits in every route with x > 0.
        const auto is_base_node = [&](int n) noexcept { return n == i || n == j || n == k; };
        for (int r : exact_candidate_paths) {
            const auto &route = allPaths[r].route;
            int         first = -1, second = -1;
            for (int pos = 1; pos + 1 < static_cast<int>(route.size()); ++pos) {
                if (is_base_node(route[pos])) {
                    if (first < 0)
                        first = pos;
                    else {
                        second = pos;
                        break;
                    }
                }
            }
            if (second < 0) continue;
            for (int inner = first + 1; inner < second; ++inner) {
                const int node = route[inner];
                AM[node / 64] |= (1ULL << (node % 64));
            }
        }

        // Compute exact violation with the limited-memory coefficient
        double              exact_vio = -rhs;
        std::vector<int>    cand_paths;
        std::vector<double> cand_coeffs;
        for (int r : exact_candidate_paths) {
            const double coeff = computeLimitedMemoryCoefficient(C, AM, p, allPaths[r].route, order);
            if (!numericutils::isZero(coeff)) {
                exact_vio += coeff * x[r];
                cand_paths.push_back(r);
                cand_coeffs.push_back(coeff);
            }
        }

        for (int r : exact_touched_routes) exact_visit_count[r] = 0;

        if (exact_vio <= SRC_SEPARATION_TOL) {
            --max_trials;
            continue;
        }

        Cut cut(C, AM, {}, p);
        cut.baseSetOrder = order;
        cut.type         = CutType::ThreeRow;
        set_sparse_coefficients(cut, cand_paths, cand_coeffs);

        cut.key = cutStorage.compute_cut_key(cut.baseSet, cut.p.num, cut.p.den);
        if (cutStorage.cutExists(cut.key).first >= 0) {
            cutStorage.markCutSeparated(cut, exact_vio);
            --max_trials;
            continue;
        }

        const auto nodes_in_cut = collect_cut_nodes(cut, false);
        bool       keep         = true;
        for (int node : nodes_in_cut) {
            if (node >= 0 && node < N_SIZE && vertex_cut_budget[node] <= 0) {
                keep = false;
                break;
            }
        }
        if (!keep) {
            --max_trials;
            continue;
        }

        --max_trials;
        cutStorage.addCut(cut);
        cutStorage.markCutSeparated(cut, exact_vio);
        for (int node : nodes_in_cut) {
            if (node >= 0 && node < N_SIZE) --vertex_cut_budget[node];
        }
        ++cuts_added;
    }
}

bool LimitedMemoryRank1Cuts::cutCleaner(BNBNode *node, std::vector<baldesCtrPtr> &SRCconstraints) {
    // Pointer to the cut storage.
    auto               *cuts = &cutStorage;
    std::vector<double> solution;
    bool                cleaned = false;

    // if (cuts->busy) return false;

    GET_SOL(node); // Retrieves the solution into 'solution'.
    // print soltuino size

    // Traverse SRCconstraints in reverse.
    for (auto it = SRCconstraints.rbegin(); it != SRCconstraints.rend();) {
        auto   constr        = *it;
        int    current_index = constr->index();
        double slack         = node->getSlack(current_index, solution);

        // If the slack is positive (non-violated), remove the constraint.
        if (numericutils::gt(slack, 0.0)) {
            cleaned = true;
            node->remove(constr);

            // Convert the reverse iterator to a normal iterator.
            // it.base() points to the element *after* the one we want to
            // erase.
            auto normal_it = it.base();
            --normal_it; // Now 'normal_it' points to the element to be
                         // erased.
            // Compute the index of this element.
            int index = std::distance(SRCconstraints.begin(), normal_it);
            cuts->removeCut(cuts->getID(index));

            // Erase the element and update the reverse iterator.
            it = std::make_reverse_iterator(SRCconstraints.erase(normal_it));
        } else {
            ++it;
        }
    }
    return cleaned;
}

/*
 * @brief Computes the limited memory coefficient for a given set of nodes.
 *
 */
double LimitedMemoryRank1Cuts::computeLimitedMemoryCoefficient(const std::array<uint64_t, num_words> &C,
                                                               const std::array<uint64_t, num_words> &AM,
                                                               const SRCPermutation &p, const std::vector<uint16_t> &P,
                                                               std::vector<int> &order) noexcept {
    double alpha = 0.0;
    int    S     = 0;
    auto   den   = p.den;

#if defined(SRC_MEMORY_MODE_ARC)
    for (size_t j = 1; j < P.size() - 1; ++j) {
        const int      vj        = P[j];
        const int      vprev     = P[j - 1];
        const size_t   word      = vj >> 6;
        const uint64_t bit_mask  = 1ULL << (vj & 63);
        const size_t   prev_word = vprev >> 6;
        const uint64_t prev_mask = 1ULL << (vprev & 63);

        if (!(AM[word] & bit_mask) || !(AM[prev_word] & prev_mask)) {
            S = 0; // Reset S if vj or previous node is not in AM
        }
        if (C[word] & bit_mask) {
            const int pos = order[vj];
            S += p.num[pos];
            if (S >= den) {
                S -= den;
                alpha += 1.0;
            }
        }
    }
#else
    for (size_t j = 1; j < P.size() - 1; ++j) {
        const int vj = P[j];

        // Precompute bitshift values for reuse
        const size_t   word     = vj >> 6;
        const uint64_t bit_mask = 1ULL << (vj & 63);

        // Check if vj is in AM using precomputed values
        if (!(AM[word] & bit_mask)) {
            S = 0; // Reset S if vj is not in AM
        } else if (C[word] & bit_mask) {
            // Get the position of vj in C by counting the set bits up to vj
            const int pos = order[vj];
            S += p.num[pos];
            if (S >= den) {
                S -= den;
                alpha += 1.0;
            }
        }
    }
#endif

    return alpha;
}

std::vector<CandidateSet> LocalSearch::solve(const CandidateSet &initial, const SparseMatrix &A,
                                             const std::vector<double>                 &x,
                                             const std::vector<std::vector<NodeScore>> &node_scores,
                                             int                                        max_iterations) {
    // Initialize best and current candidate sets.
    CandidateSet              best    = initial;
    CandidateSet              current = initial;
    std::vector<CandidateSet> diverse_solutions;
    diverse_solutions.reserve(LocalSearchConfig::MAX_DIVERSE_SOLUTIONS + 1);

    // Lambda: Check if a candidate solution is "diverse" enough.
    auto isSolutionDiverse = [&](const CandidateSet &candidate) -> bool {
        // Convert candidate nodes into a sorted vector.
        std::vector<int> cand_nodes(candidate.nodes.begin(), candidate.nodes.end());
        std::sort(cand_nodes.begin(), cand_nodes.end());
        for (const auto &sol : diverse_solutions) {
            std::vector<int> sol_nodes(sol.nodes.begin(), sol.nodes.end());
            std::sort(sol_nodes.begin(), sol_nodes.end());
            double similarity = computeSimilarity(cand_nodes, sol_nodes);
            if (similarity > 0.9 ||
                std::abs(sol.violation - candidate.violation) < LocalSearchConfig::DIVERSITY_THRESHOLD) {
                return false;
            }
        }
        return true;
    };

    // Lambda: Update the list of diverse solutions.
    auto updateDiverseSolutions = [&](const CandidateSet &candidate) {
        if (isSolutionDiverse(candidate)) {
            diverse_solutions.push_back(candidate);
            if (diverse_solutions.size() > LocalSearchConfig::MAX_DIVERSE_SOLUTIONS) {
                auto min_it = std::min_element(
                    diverse_solutions.begin(), diverse_solutions.end(),
                    [](const CandidateSet &a, const CandidateSet &b) { return a.violation < b.violation; });
                diverse_solutions.erase(min_it);
            }
        }
    };

    // Lambda: Update segment statistics after each segment.
    auto updateSegmentStatistics = [&]() {
        segment_iterations++;
        if (segment_iterations == LocalSearchConfig::SEGMENT_SIZE) {
            history.push_back(current_segment);
            if (history.size() > 3) { history.pop_front(); }
            current_segment    = SegmentStats{};
            segment_iterations = 0;
        }
    };

    // Main iterative search loop.
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Backup the current solution.
        CandidateSet backup      = current;
        OperatorType selected_op = selectOperator();

        // Apply the selected operator.
        switch (selected_op) {
        case OperatorType::REMOVE_NODE: applyRemoveNode(current); break;
        case OperatorType::ADD_NODE: applyAddNode(current, node_scores); break;
        case OperatorType::REMOVE_NEIGHBORS: applyRemoveNeighbors(current, LocalSearchConfig::MAX_REMOVE_COUNT); break;
        case OperatorType::ADD_NEIGHBORS:
            applyAddNeighbors(current, node_scores, LocalSearchConfig::MAX_ADD_COUNT);
            break;
        case OperatorType::SWAP_NODE: applySwapNode(current, node_scores); break;
        default: applyUpdateNeighbors(current, node_scores); break;
        }

        // Evaluate the new candidate using parent's violation computation.
        auto [new_violation, new_perm, rhs] =
            parent->computeViolationWithBestPerm(current.nodes, current.neighbor, A, x);
        double delta = new_violation - backup.violation;

        // Accept or reject the move.
        if (acceptMove(delta, current)) {
            // Accept move: update candidate solution.
            current.violation = new_violation;
            current.perm      = std::move(new_perm);
            current.rhs       = std::move(rhs);
            updateStatistics(selected_op, delta, current, new_violation);
            if (new_violation > best.violation) {
                best = current;
                current_segment.improvements++;
                iterations_since_improvement   = 0;
                current_segment.best_violation = std::max(current_segment.best_violation, new_violation);
            } else {
                iterations_since_improvement++;
            }
            updateDiverseSolutions(current);
        } else {
            // Revert move and adjust operator score.
            current = backup;
            operators[static_cast<size_t>(selected_op)].score =
                std::max(LocalSearchConfig::MIN_WEIGHT, operators[static_cast<size_t>(selected_op)].score *
                                                            (1 - LocalSearchConfig::OPERATOR_LEARNING_RATE));
        }

        updateSegmentStatistics();
        updateTemperature();
        strategicRestart(current, best, diverse_solutions);
    }

    // Final processing: add the best solution and sort the diverse set.
    diverse_solutions.push_back(std::move(best));
    pdqsort(diverse_solutions.begin(), diverse_solutions.end(),
            [](const CandidateSet &a, const CandidateSet &b) { return a.violation > b.violation; });

    return diverse_solutions;
}
