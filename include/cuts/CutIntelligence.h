#pragma once
#include "Cut.h"
#include "CutHelper.h"
#include "Dual.h"
#include "RNG.h"
#include "SparseMatrix.h"
#include "VRPNode.h"

class HighRankCuts;  // Forward declaration

class LocalSearch {
   private:
    enum class OperatorType : uint8_t {
        SWAP_NODES,
        REMOVE_ADD_NODE,
        UPDATE_NEIGHBORS
    };

    struct OperatorStats {
        double score = 1.0;
        int32_t usage = 0;
        int32_t success = 0;
        double avg_improvement = 0.0;
    };

    struct SegmentStats {
        double avg_violation = 0.0;
        double best_violation = 0.0;
        int32_t improvements = 0;
        int32_t accepted_moves = 0;
    };

    // Operator statistics and search history.
    std::array<OperatorStats, 3> operators;
    std::deque<SegmentStats> history;
    SegmentStats current_segment{};

    // Random number generator for moves and operator selection.
    Xoroshiro128Plus rng;

    // Counters for iterations and search segments.
    int iterations_since_improvement = 0;
    int segment_iterations = 0;

    // Temperature for simulated annealing.
    double temperature;

    // Pointer to the parent HighRankCuts instance.
    HighRankCuts *parent;

    // Computes the similarity between two sets based on the size of their
    // intersection.
    double computeSimilarity(const std::vector<int> &set1,
                             const std::vector<int> &set2) const {
        std::vector<int> intersection;
        std::set_intersection(set1.begin(), set1.end(), set2.begin(),
                              set2.end(), std::back_inserter(intersection));
        return 2.0 * intersection.size() / (set1.size() + set2.size());
    }

    // Selects an operator based on the current operator scores.
    OperatorType selectOperator() {
        std::vector<double> weights;
        for (const auto &op : operators) {
            weights.push_back(op.score);
        }
        std::discrete_distribution<> dist(weights.begin(), weights.end());
        return static_cast<OperatorType>(dist(rng));
    }

    // Swap two nodes in the candidate set.
    void applySwapNodes(CandidateSet &current) {
        if (current.nodes.size() >= 2) {
            std::uniform_int_distribution<> node_dist(0,
                                                      current.nodes.size() - 1);
            int pos1 = node_dist(rng), pos2;
            do {
                pos2 = node_dist(rng);
            } while (pos2 == pos1);
            std::swap(current.nodes[pos1], current.nodes[pos2]);
        }
    }

    // Remove a node and add a new node from the candidate's neighbor list,
    // using neighbor scores to guide the choice.
    void applyRemoveAddNode(CandidateSet &current,
                            const std::vector<std::vector<int>> &node_scores) {
        if (current.nodes.size() > 1 && !current.neighbor.empty()) {
            std::uniform_int_distribution<> node_dist(0,
                                                      current.nodes.size() - 1);
            int remove_pos = node_dist(rng);
            current.nodes.erase(current.nodes.begin() + remove_pos);

            std::vector<std::pair<int, double>> scored_neighbors;
            for (const auto &n : current.neighbor) {
                double score = 0.0;
                for (const auto &node : current.nodes) {
                    if (std::binary_search(node_scores[node].begin(),
                                           node_scores[node].end(), n)) {
                        score += 1.0;
                    }
                }
                scored_neighbors.emplace_back(n, score);
            }

            if (!scored_neighbors.empty()) {
                pdqsort(scored_neighbors.begin(), scored_neighbors.end(),
                        [](const auto &a, const auto &b) {
                            return a.second > b.second;
                        });
                double temp_factor = std::max(
                    0.1, temperature / LocalSearchConfig::INITIAL_TEMPERATURE);
                std::exponential_distribution<> exp_dist(
                    1.0 / (temp_factor * (1.0 + iterations_since_improvement)));
                int rank = std::min<int>(scored_neighbors.size() - 1,
                                         static_cast<int>(exp_dist(rng)));
                current.nodes.push_back(scored_neighbors[rank].first);
            }
        }
    }

    // Update the candidate's neighbor list by removing some neighbors and
    // adding potential ones.
    void applyUpdateNeighbors(
        CandidateSet &current,
        const std::vector<std::vector<int>> &node_scores) {
        std::shuffle(current.neighbor.begin(), current.neighbor.end(), rng);
        current.neighbor.erase(
            current.neighbor.begin(),
            current.neighbor.begin() +
                std::min(LocalSearchConfig::MAX_REMOVE_COUNT,
                         static_cast<int>(current.neighbor.size())));

        std::vector<int> potential_neighbors;
        for (const int &node : current.nodes) {
            for (const int &neighbor : node_scores[node]) {
                if (!std::binary_search(current.nodes.begin(),
                                        current.nodes.end(), neighbor) &&
                    std::find(current.neighbor.begin(), current.neighbor.end(),
                              neighbor) == current.neighbor.end()) {
                    potential_neighbors.push_back(neighbor);
                }
            }
        }
        std::shuffle(potential_neighbors.begin(), potential_neighbors.end(),
                     rng);
        current.neighbor.insert(
            current.neighbor.end(), potential_neighbors.begin(),
            potential_neighbors.begin() +
                std::min(LocalSearchConfig::MAX_REMOVE_COUNT,
                         static_cast<int>(potential_neighbors.size())));
    }

    // Accept a move based on its improvement delta and current temperature.
    bool acceptMove(double delta, const CandidateSet &current) {
        if (delta > 0) return true;

        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double acceptance_prob = std::exp(delta / temperature);

        if (history.size() >= 2) {
            double recent_improvement_rate =
                static_cast<double>(history.back().improvements) /
                LocalSearchConfig::SEGMENT_SIZE;
            double acceptance_rate =
                std::clamp(0.4 * (1.0 - recent_improvement_rate),
                           LocalSearchConfig::MIN_ACCEPTANCE_RATE,
                           LocalSearchConfig::MAX_ACCEPTANCE_RATE);
            double avg_recent_violation =
                std::accumulate(history.begin(), history.end(), 0.0,
                                [](double sum, const auto &seg) {
                                    return sum + seg.avg_violation;
                                }) /
                history.size();

            if (current.violation < avg_recent_violation) {
                return (dist(rng) < acceptance_rate *
                                        LocalSearchConfig::IMPROVEMENT_BONUS) &&
                       (delta >
                        -std::abs(current.violation *
                                  LocalSearchConfig::MAX_DETERIORATION));
            }
            return (dist(rng) < acceptance_rate) &&
                   (delta > -std::abs(current.violation *
                                      LocalSearchConfig::MAX_DETERIORATION));
        }

        return dist(rng) < acceptance_prob;
    }

    // Update operator and segment statistics after a move.
    void updateStatistics(OperatorType op, double delta, CandidateSet &current,
                          double new_violation) {
        auto &op_stats = operators[static_cast<size_t>(op)];
        op_stats.usage++;

        if (delta > 0) {
            op_stats.success++;
            op_stats.avg_improvement =
                0.9 * op_stats.avg_improvement + 0.1 * delta;
        }

        op_stats.score = std::max(
            LocalSearchConfig::MIN_WEIGHT,
            op_stats.score * (1 - LocalSearchConfig::OPERATOR_LEARNING_RATE) +
                LocalSearchConfig::OPERATOR_LEARNING_RATE *
                    (1.0 + std::max(0.0, delta)));

        current_segment.accepted_moves++;
        current_segment.avg_violation =
            (current_segment.avg_violation * segment_iterations +
             new_violation) /
            (segment_iterations + 1);
    }

    // Update the temperature using a cooling schedule and reheating if
    // necessary.
    void updateTemperature() {
        temperature *= LocalSearchConfig::COOLING_RATE;
        if (iterations_since_improvement >=
            LocalSearchConfig::REHEAT_INTERVAL) {
            temperature *= LocalSearchConfig::REHEATING_FACTOR;
            iterations_since_improvement = 0;
        }
    }

    // Perform a strategic restart using diverse solutions when search
    // stagnates.
    void strategicRestart(CandidateSet &current, const CandidateSet &best,
                          std::vector<CandidateSet> &diverse_solutions) {
        if (iterations_since_improvement > LocalSearchConfig::SEGMENT_SIZE &&
            !diverse_solutions.empty()) {
            std::vector<std::pair<double, size_t>> restart_candidates;
            for (size_t i = 0; i < diverse_solutions.size(); ++i) {
                const auto &sol = diverse_solutions[i];
                double quality_score = sol.violation / best.violation;
                double diversity_score = 0.0;
                for (const auto &other : diverse_solutions) {
                    if (&other != &sol) {
                        double structural_div =
                            1.0 - computeSimilarity(sol.nodes, other.nodes);
                        double violation_div =
                            std::abs(other.violation - sol.violation);
                        diversity_score +=
                            0.5 * (structural_div + violation_div);
                    }
                }
                restart_candidates.emplace_back(
                    quality_score * LocalSearchConfig::QUALITY_WEIGHT +
                        diversity_score * LocalSearchConfig::DIVERSITY_WEIGHT,
                    i);
            }

            pdqsort(restart_candidates.begin(), restart_candidates.end());
            current = diverse_solutions[restart_candidates.back().second];
            iterations_since_improvement = 0;
            temperature = LocalSearchConfig::INITIAL_TEMPERATURE;
        }
    }

   public:
    // Constructor: initializes the random number generator and starting
    // temperature.
    explicit LocalSearch(HighRankCuts *p)
        : rng(std::random_device{}()),
          temperature(LocalSearchConfig::INITIAL_TEMPERATURE),
          parent(p) {}

    // The main local search solver function.
    std::vector<CandidateSet> solve(
        const CandidateSet &initial, const SparseMatrix &A,
        const std::vector<double> &x,
        const std::vector<std::vector<int>> &node_scores,
        int max_iterations = 100);
};

constexpr int MAX_CANDIDATES_PER_NODE = 25;
constexpr double LEARNING_RATE = 0.1;
constexpr double HISTORY_WEIGHT = 0.8;
constexpr double DECAY_FACTOR = 0.95;  // Decay old feedback

class AdaptiveNodeScorer {
   private:
    struct NodePairStats {
        double success_count = 0;
        double total_violation = 0;
        double avg_violation = 0;
        int occurrences = 0;
    };

    // Track historical pair statistics and neighbor relevance scores.
    ankerl::unordered_dense::map<std::pair<int, int>, NodePairStats> pair_stats;
    ankerl::unordered_dense::map<int, double> neighbor_scores;

    static constexpr double LEARNING_RATE = 0.1;
    static constexpr double HISTORY_WEIGHT = 0.8;
    static constexpr double DECAY_FACTOR =
        0.95;  // Decay factor for old feedback
    static constexpr double NEIGHBOR_WEIGHT =
        0.5;  // Weight for neighbor relevance

    // Retrieve historical score for a pair (node_i, node_j) based on prior
    // feedback.
    double getHistoricalScore(int node_i, int node_j) const {
        auto key =
            std::make_pair(std::min(node_i, node_j), std::max(node_i, node_j));
        auto it = pair_stats.find(key);
        if (it != pair_stats.end() && it->second.occurrences > 0) {
            const auto &stats = it->second;
            return stats.avg_violation * std::log1p(stats.success_count);
        }
        return 0.0;
    }

    // Retrieve a stored neighbor relevance score.
    double getNeighborScore(int node) const {
        auto it = neighbor_scores.find(node);
        return (it != neighbor_scores.end()) ? it->second : 0.0;
    }

   public:
    // Update feedback statistics from the provided candidate set.
    void provideFeedback(
        const ankerl::unordered_dense::set<CandidateSet, CandidateSetHasher,
                                           CandidateSetCompare> &candidates) {
        // Convert candidate set to vector and sort by violation in descending
        // order.
        std::vector<CandidateSet> candidate_vec(candidates.begin(),
                                                candidates.end());
        pdqsort(candidate_vec.begin(), candidate_vec.end(),
                [](const auto &a, const auto &b) {
                    return a.violation > b.violation;
                });

        // Use top 20% of candidates for feedback.
        const int max_feedback_candidates =
            std::min(static_cast<int>(candidate_vec.size()),
                     static_cast<int>(candidate_vec.size() * 0.2));

        for (int idx = 0; idx < max_feedback_candidates; ++idx) {
            const auto &candidate = candidate_vec[idx];

            // Update pair statistics for every pair in candidate.nodes.
            for (size_t i = 0; i < candidate.nodes.size(); ++i) {
                for (size_t j = i + 1; j < candidate.nodes.size(); ++j) {
                    auto key = std::make_pair(
                        std::min(candidate.nodes[i], candidate.nodes[j]),
                        std::max(candidate.nodes[i], candidate.nodes[j]));
                    auto &stats = pair_stats[key];
                    stats.success_count =
                        stats.success_count * DECAY_FACTOR + 1;
                    stats.total_violation =
                        stats.total_violation * DECAY_FACTOR +
                        candidate.violation;
                    stats.occurrences += 1;
                    stats.avg_violation =
                        stats.total_violation / stats.occurrences;
                }
            }

            // Update neighbor scores for candidate neighbors.
            for (int node : candidate.nodes) {
                for (int neighbor : candidate.neighbor) {
                    neighbor_scores[neighbor] =
                        neighbor_scores[neighbor] * DECAY_FACTOR + 1;
                }
            }
        }
    }

    // Compute node scores for each vertex using problem data and feedback.
    std::vector<std::vector<int>> computeNodeScores(
        const SparseMatrix &A, const std::vector<double> &x,
        const std::vector<std::vector<double>> &distances,
        const std::vector<VRPNode> &nodes, const ArcDuals &arc_duals,
        const CutStorage &cutStorage) {
        std::vector<std::vector<int>> scores(N_SIZE);
        const auto &active_cuts = cutStorage.getActiveCuts();

        // Pre-calculate adjustments from active cuts.
        std::vector<std::vector<double>> cut_adjustments(
            N_SIZE, std::vector<double>(N_SIZE, 0.0));
        for (const auto &cut : active_cuts) {
            if (cut.type == CutType::ThreeRow) {
                for (int i = 1; i < N_SIZE - 1; ++i) {
                    for (int j = i + 1; j < N_SIZE - 1; ++j) {
                        if (cut.isSRCset(i, j)) {
                            cut_adjustments[i][j] -= cut.dual_value;
                            cut_adjustments[j][i] -= cut.dual_value;
                        }
                    }
                }
            } else if (cut.type == CutType::OneRow) {
                for (int i = 1; i < N_SIZE - 1; ++i) {
                    if (cut.isSRCset(i)) {
                        for (int j = 1; j < N_SIZE - 1; ++j) {
                            if (i != j) {
                                cut_adjustments[i][j] -= cut.dual_value;
                            }
                        }
                    }
                }
            }
        }

        // Compute scores for each vertex i (ignoring depot vertices).
        for (int i = 1; i < N_SIZE - 1; ++i) {
            std::vector<std::pair<int, double>> node_scores;
            node_scores.reserve(N_SIZE - 2);

            for (int j = 1; j < N_SIZE - 1; ++j) {
                if (i == j) continue;

                // Base score incorporates distance, node costs, arc duals, and
                // cut adjustments.
                double base_score =
                    distances[i][j] - (nodes[i].cost + nodes[j].cost) / 2.0 -
                    arc_duals.getDual(i, j) + cut_adjustments[i][j];

                // // Incorporate historical performance and neighbor relevance.
                // double historical_score = getHistoricalScore(i, j);
                // double neighbor_score = getNeighborScore(j);

                // // Confidence factor based on feedback frequency.
                // double confidence = std::min(
                //     1.0,
                //     std::log1p(pair_stats[std::make_pair(i,
                //     j)].occurrences));
                // double final_score =
                //     (1 - HISTORY_WEIGHT * confidence) * base_score +
                //     HISTORY_WEIGHT * confidence * historical_score +
                //     NEIGHBOR_WEIGHT * neighbor_score;

                node_scores.emplace_back(j, base_score);
            }

            // Sort candidates for vertex i by score (lowest scores are best).
            std::partial_sort(
                node_scores.begin(),
                node_scores.begin() +
                    std::min(MAX_CANDIDATES_PER_NODE,
                             static_cast<int>(node_scores.size())),
                node_scores.end(), [](const auto &a, const auto &b) {
                    return a.second < b.second;
                });

            // Store top candidate node indices.
            auto &i_scores = scores[i];
            for (int j = 0; j < std::min(MAX_CANDIDATES_PER_NODE,
                                         static_cast<int>(node_scores.size()));
                 ++j) {
                i_scores.push_back(node_scores[j].first);
            }
        }

        return scores;
    }
};
