#pragma once
#include "Definitions.h"
#include "SRC.h"
// Permutation structures and helper functions
struct Permutation {
    std::vector<int> num;
    int              den;

    Permutation(const std::vector<int> &n, int d) : num(n), den(d) {}
};

inline std::vector<Permutation> getPermutationsForSize5() {
    std::vector<Permutation> base_perms = {
        Permutation({2, 2, 1, 1, 1}, 4), // {0.5, 0.5, 0.25, 0.25, 0.25}
        Permutation({3, 1, 1, 1, 1}, 4), // {0.75, 0.25, 0.25, 0.25, 0.25}
        Permutation({3, 2, 2, 1, 1}, 5), // {0.6, 0.4, 0.4, 0.2, 0.2}
        Permutation({2, 2, 1, 1, 1},
                    3),                 // {0.6667, 0.6667, 0.3333, 0.3333, 0.3333}
        Permutation({3, 3, 2, 2, 1}, 4) // {0.75, 0.75, 0.5, 0.5, 0.25}
    };

    std::vector<Permutation> all_perms;
    for (const auto &base_perm : base_perms) {
        std::vector<int> p = base_perm.num;
        std::sort(p.begin(), p.end());
        do { all_perms.emplace_back(p, base_perm.den); } while (std::next_permutation(p.begin(), p.end()));
    }
    return all_perms;
}

inline std::vector<Permutation> getPermutationsForSize4() {
    std::vector<int>         p = {2, 1, 1, 1}; // 2/3, 1/3, 1/3, 1/3
    std::vector<Permutation> perms;
    std::sort(p.begin(), p.end());
    do { perms.emplace_back(p, 3); } while (std::next_permutation(p.begin(), p.end()));
    return perms;
}

template <typename T>
inline void combinations(const std::vector<T> &elements, int k, std::vector<std::vector<T>> &result) {
    std::vector<int> indices(k);
    std::iota(indices.begin(), indices.end(), 0);
    while (true) {
        std::vector<T> combination;
        for (int idx : indices) { combination.push_back(elements[idx]); }
        result.push_back(combination);

        int i = k - 1;
        while (i >= 0 && indices[i] == elements.size() - k + i) { --i; }
        if (i < 0) break;

        ++indices[i];
        for (int j = i + 1; j < k; ++j) { indices[j] = indices[j - 1] + 1; }
    }
}

template <size_t RANK>
class HighRankCuts {
private:
    CutStorage          &cutStorage;
    std::vector<Path>    allPaths;
    static constexpr int MAX_CANDIDATES_PER_NODE = 20; // Limit candidate combinations

    struct NodeScore {
        int    node            = 0;
        double violation_score = 0.0;
        double distance_score  = 0.0;
        double cost_score      = 0.0;
        double combined_score  = 0.0;

        NodeScore() = default;

        NodeScore(int n, double v, double d, double c) : node(n), violation_score(v), distance_score(d), cost_score(c) {
            // Combine scores with weights
            // We want higher violation_score, lower distance_score, and lower cost_score
            constexpr double VIOLATION_WEIGHT = 1.0;
            constexpr double DISTANCE_WEIGHT  = -0.3; // Negative because we want closer nodes
            constexpr double COST_WEIGHT      = -0.2; // Negative because we want lower costs

            combined_score =
                VIOLATION_WEIGHT * violation_score + DISTANCE_WEIGHT * distance_score + COST_WEIGHT * cost_score;
        }

        bool operator<(const NodeScore &other) const { return combined_score > other.combined_score; }
    };

    struct CandidateSet {
        std::vector<int> nodes;
        double           violation;
        Permutation      perm;

        CandidateSet(const std::vector<int> &n, double v, const Permutation &p) : nodes(n), violation(v), perm(p) {}
    };

    // Compute contribution scores for each node based on column overlap and x values
    std::vector<std::vector<NodeScore>> computeNodeScores(const SparseMatrix &A, const std::vector<double> &x) {

        std::vector<std::vector<NodeScore>> scores(N_SIZE);
        std::vector<std::vector<int>>       col_to_rows(A.num_cols);

        // Build column to rows mapping
        for (int idx = 0; idx < A.values.size(); ++idx) {
            int row = A.rows[idx];
            if (row > N_SIZE - 2) continue;
            col_to_rows[A.cols[idx]].push_back(row + 1);
        }

        // For each node, compute scores with other nodes
        for (int i = 1; i < N_SIZE - 1; ++i) {
            std::vector<double> node_contributions(N_SIZE, 0.0);

            // Find columns where node i appears
            for (int idx = 0; idx < A.values.size(); ++idx) {
                if (A.rows[idx] + 1 == i) {
                    int    col       = A.cols[idx];
                    double col_value = x[col];

                    // For each other node that shares this column
                    for (int other_row : col_to_rows[col]) {
                        if (other_row != i) { node_contributions[other_row] += col_value; }
                    }
                }
            }

            // Convert to scores incorporating distance and cost
            std::vector<NodeScore> node_scores;
            for (int j = 1; j < N_SIZE - 1; ++j) {
                if (j != i && node_contributions[j] > 0) {
                    double violation_score = node_contributions[j];
                    double distance_score  = distances[i][j];
                    double cost_score      = nodes[i].cost + nodes[j].cost;

                    // Additional cost-based filtering
                    if (cost_score < nodes[i].cost * 3) { // Only consider if total cost is reasonable
                        node_scores.emplace_back(j, violation_score, distance_score, cost_score);
                    }
                }
            }

            // Sort by combined score
            std::sort(node_scores.begin(), node_scores.end());

            // Keep only the top candidates
            if (node_scores.size() > MAX_CANDIDATES_PER_NODE) { node_scores.resize(MAX_CANDIDATES_PER_NODE); }

            scores[i] = std::move(node_scores);
        }

        return scores;
    }

    // Generate promising candidate sets based on node scores
    std::vector<CandidateSet> generateCandidates(const std::vector<std::vector<NodeScore>> &scores,
                                                 const std::vector<std::vector<int>>       &row_indices_map,
                                                 const SparseMatrix &A, const std::vector<double> &x) {

        std::vector<CandidateSet> candidates;
        std::vector<bool>         used(N_SIZE, false);

        for (int start_node = 1; start_node < N_SIZE - 1; ++start_node) {
            if (scores[start_node].empty()) continue;

            std::vector<int> current_set = {start_node};
            std::fill(used.begin(), used.end(), false);
            used[start_node] = true;

            // Build set of size RANK greedily
            while (current_set.size() < RANK) {
                double      best_violation = 0.0;
                int         best_node      = -1;
                Permutation best_perm      = getPermutations()[0]; // Default permutation

                // Try adding each candidate node
                for (const auto &score : scores[start_node]) {
                    int candidate = score.node;
                    if (!used[candidate]) {
                        current_set.push_back(candidate);
                        auto [violation, perm] = computeViolationWithBestPerm(current_set, row_indices_map, A, x);

                        if (violation > best_violation) {
                            best_violation = violation;
                            best_node      = candidate;
                            best_perm      = perm;
                        }
                        current_set.pop_back();
                    }
                }

                if (best_node == -1) break;
                current_set.push_back(best_node);
                used[best_node] = true;

                // If we have a complete set with good violation, add it
                if (current_set.size() == RANK && best_violation > 0.8) {
                    candidates.emplace_back(current_set, best_violation, best_perm);
                    if (candidates.size() >= MAX_CANDIDATES_PER_NODE) break;
                }
            }
        }

        return candidates;
    }

    std::vector<Permutation> getPermutations() const {
        if constexpr (RANK == 4) {
            return getPermutationsForSize4();
        } else if constexpr (RANK == 5) {
            return getPermutationsForSize5();
        }
        return {};
    }

    std::pair<double, Permutation> computeViolationWithBestPerm(const std::vector<int>              &nodes,
                                                                const std::vector<std::vector<int>> &row_indices_map,
                                                                const SparseMatrix &A, const std::vector<double> &x) {
        std::vector<int> expanded(A.num_cols, 0);

        for (int node : nodes) {
            for (int idx : row_indices_map[node]) { expanded[A.cols[idx]] += 1; }
        }

        double       best_violation = 0.0;
        Permutation *best_perm      = nullptr;
        auto         permutations   = getPermutations();

        for (auto &perm : permutations) {
            double lhs = 0.0;
            for (int idx = 0; idx < A.num_cols; ++idx) {
                if (expanded[idx] >= 2) {
                    double coef = 0.0;
                    for (size_t i = 0; i < RANK; ++i) {
                        if (expanded[idx] > i) { coef += static_cast<double>(perm.num[i]) / perm.den; }
                    }
                    lhs += coef * x[idx];
                }
            }

            if (lhs > best_violation) {
                best_violation = lhs;
                best_perm      = &perm;
            }
        }

        return {best_violation, best_perm ? *best_perm : permutations[0]};
    }

    CandidateSet localSearch(const CandidateSet &initial, const std::vector<std::vector<int>> &row_indices_map,
                             const SparseMatrix &A, const std::vector<double> &x, double temperature = 0.1,
                             int max_iterations = 100) {
        CandidateSet best                           = initial;
        CandidateSet current                        = initial;
        int          iterations_without_improvement = 0;

        std::random_device               rd;
        std::mt19937                     gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        for (int iter = 0; iter < max_iterations && iterations_without_improvement < 20; ++iter) {
            // Generate neighbor by swapping one node
            auto neighbor      = current;
            int  pos_to_swap   = gen() % RANK;
            int  original_node = neighbor.nodes[pos_to_swap];

            // Try to find a better neighbor
            bool improved = false;
            for (int new_node = 1; new_node < N_SIZE - 1; ++new_node) {
                if (std::find(neighbor.nodes.begin(), neighbor.nodes.end(), new_node) != neighbor.nodes.end()) {
                    continue;
                }

                neighbor.nodes[pos_to_swap]    = new_node;
                auto [new_violation, new_perm] = computeViolationWithBestPerm(neighbor.nodes, row_indices_map, A, x);

                // Accept if better or with probability based on temperature
                double delta = new_violation - current.violation;
                if (delta > 0 || (temperature > 0 && dis(gen) < std::exp(delta / temperature))) {
                    current.nodes     = neighbor.nodes;
                    current.violation = new_violation;
                    current.perm      = new_perm;
                    improved          = true;

                    if (new_violation > best.violation) {
                        best                           = current;
                        iterations_without_improvement = 0;
                    }
                    break;
                }
            }

            if (!improved) {
                neighbor.nodes[pos_to_swap] = original_node;
                iterations_without_improvement++;
            }

            temperature *= 0.95; // Cooling schedule
        }

        return best;
    }

    void addCutToCutStorage(const CandidateSet &candidate, std::vector<int> &order) {
        std::array<uint64_t, num_words> C  = {};
        std::array<uint64_t, num_words> AM = {};

        for (auto node : candidate.nodes) {
            C[node / 64] |= (1ULL << (node % 64));
            AM[node / 64] |= (1ULL << (node % 64));
        }

        SRCPermutation p;
        p.num = candidate.perm.num;
        p.den = candidate.perm.den;

        std::vector<double> cut_coefficients(allPaths.size());
        auto                z = 0;
        for (auto &path : allPaths) {
            cut_coefficients[z++] = computeLimitedMemoryCoefficient(C, AM, p, path.route, order);
        }

        Cut cut(C, AM, cut_coefficients, p);
        cut.baseSetOrder = order;
        cutStorage.addCut(cut);
    }

public:
    HighRankCuts(const std::vector<Path> &paths, CutStorage &storage) : allPaths(paths), cutStorage(storage) {
        static_assert(RANK == 4 || RANK == 5, "Only rank 4 and 5 are supported");
    }

    std::vector<VRPNode>             nodes;
    std::vector<std::vector<double>> distances;

    void separate(const SparseMatrix &A, const std::vector<double> &x) {
        // Compute promising node combinations based on column overlap
        auto node_scores = computeNodeScores(A, x);

        // Build row indices map for efficient access
        std::vector<std::vector<int>> row_indices_map(N_SIZE);
        for (int idx = 0; idx < A.values.size(); ++idx) {
            int row = A.rows[idx];
            if (row > N_SIZE - 2) continue;
            row_indices_map[row + 1].push_back(idx);
        }

        // Generate promising candidates
        auto candidates = generateCandidates(node_scores, row_indices_map, A, x);

        // Parallel local search
        const int                JOBS = std::thread::hardware_concurrency();
        exec::static_thread_pool pool(JOBS);
        auto                     sched = pool.get_scheduler();

        std::mutex                candidates_mutex;
        std::vector<CandidateSet> improved_candidates;

        auto bulk_sender = stdexec::bulk(
            stdexec::just(), candidates.size(),
            [this, &row_indices_map, &A, &x, &candidates_mutex, &improved_candidates, &candidates](std::size_t idx) {
                auto improved = localSearch(candidates[idx], row_indices_map, A, x);

                if (improved.violation > 1.0 + 1e-3) {
                    std::lock_guard<std::mutex> lock(candidates_mutex);
                    improved_candidates.push_back(improved);
                }
            });

        auto work = stdexec::starts_on(sched, bulk_sender);
        stdexec::sync_wait(std::move(work));

        // Sort and add best cuts
        std::sort(improved_candidates.begin(), improved_candidates.end(),
                  [](const auto &a, const auto &b) { return a.violation > b.violation; });

        const int max_cuts = 2;
        for (int i = 0; i < std::min(max_cuts, static_cast<int>(improved_candidates.size())); ++i) {
            std::vector<int> order(N_SIZE);
            int              ordering = 0;
            for (int node : improved_candidates[i].nodes) { order[node] = ordering++; }
            addCutToCutStorage(improved_candidates[i], order);
        }
    }

    double computeLimitedMemoryCoefficient(const std::array<uint64_t, num_words> &C,
                                           const std::array<uint64_t, num_words> &AM, const SRCPermutation &p,
                                           const std::vector<uint16_t> &P, std::vector<int> &order) noexcept {
        double alpha = 0.0;
        int    S     = 0;
        auto   den   = p.den;

        for (size_t j = 1; j < P.size() - 1; ++j) {
            int vj = P[j];

            // Precompute bitshift values for reuse
            uint64_t am_mask  = (1ULL << (vj & 63));
            uint64_t am_index = vj >> 6;

            // Check if vj is in AM using precomputed values
            if (!(AM[am_index] & am_mask)) {
                S = 0; // Reset S if vj is not in AM
            } else if (C[am_index] & am_mask) {
                // Get the position of vj in C by counting the set bits up to vj
                int pos = order[vj];

                S += p.num[pos];
                if (S >= den) {
                    S -= den;
                    alpha += 1;
                }
            }
        }

        return alpha;
    }
};
