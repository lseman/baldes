#pragma once
#include "Cut.h"
#include "CutHelper.h"
#include "Dual.h"
#include "SparseMatrix.h"
#include "VRPNode.h"

constexpr int MAX_CANDIDATES_PER_NODE = 25;
class AdaptiveNodeScorer {
   private:
    // Store historical feedback about successful node pairings
    struct NodePairStats {
        double success_count = 0;
        double total_violation = 0;
        double avg_violation = 0;
        int occurrences = 0;
    };

    // Map to store statistics for node pairs
    std::unordered_map<std::pair<int, int>, NodePairStats> pair_stats;

    // Learning rate for updating statistics
    static constexpr double LEARNING_RATE = 0.1;
    static constexpr double HISTORY_WEIGHT = 0.8;

    // Score adjustment based on historical performance
    double getHistoricalScore(int node_i, int node_j) const {
        auto key =
            std::make_pair(std::min(node_i, node_j), std::max(node_i, node_j));
        auto it = pair_stats.find(key);
        if (it != pair_stats.end()) {
            const auto& stats = it->second;
            if (stats.occurrences > 0) {
                return stats.avg_violation * std::log1p(stats.success_count);
            }
        }
        return 0.0;
    }

   public:
    void provideFeedback(const std::vector<CandidateSet>& candidates) {
        for (const auto& candidate : candidates) {
            // Process each pair of nodes in successful candidates
            for (size_t i = 0; i < candidate.nodes.size(); ++i) {
                for (size_t j = i + 1; j < candidate.nodes.size(); ++j) {
                    auto key = std::make_pair(
                        std::min(candidate.nodes[i], candidate.nodes[j]),
                        std::max(candidate.nodes[i], candidate.nodes[j]));

                    auto& stats = pair_stats[key];
                    stats.success_count += 1;
                    stats.total_violation += candidate.violation;
                    stats.occurrences += 1;
                    stats.avg_violation =
                        stats.total_violation / stats.occurrences;
                }
            }
        }
    }

    void provideFeedbackFromCandidates(
        const ankerl::unordered_dense::set<CandidateSet, CandidateSetHasher,
                                           CandidateSetCompare>& candidates) {
        std::vector<CandidateSet> candidate_vec(candidates.begin(),
                                                candidates.end());

        // Sort candidates by violation to identify the most successful ones
        pdqsort(candidate_vec.begin(), candidate_vec.end(),
                [](const auto& a, const auto& b) {
                    return a.violation > b.violation;
                });

        // Take top performing candidates for feedback
        const int max_feedback_candidates =
            std::min(static_cast<int>(candidate_vec.size()),
                     static_cast<int>(candidate_vec.size() *
                                      0.2)  // Use top 20% of candidates
            );

        std::vector<CandidateSet> feedback_candidates(
            candidate_vec.begin(),
            candidate_vec.begin() + max_feedback_candidates);

        // Provide feedback to the adaptive scorer
        provideFeedback(feedback_candidates);
    }

    std::vector<std::vector<int>> computeNodeScores(
        const SparseMatrix& A, const std::vector<double>& x,
        const std::vector<std::vector<double>>& distances,
        const std::vector<VRPNode>& nodes, const ArcDuals& arc_duals,
        const CutStorage& cutStorage) {
        std::vector<std::vector<int>> scores(N_SIZE);
        const auto& active_cuts = cutStorage.getActiveCuts();

        // Pre-calculate cut adjustments
        std::vector<std::vector<double>> cut_adjustments(
            N_SIZE, std::vector<double>(N_SIZE, 0.0));
        for (const auto& cut : active_cuts) {
            if (cut.type == CutType::ThreeRow) {
                for (int i = 1; i < N_SIZE - 1; ++i) {
                    for (int j = i + 1; j < N_SIZE - 1; ++j) {
                        if (cut.isSRCset(i, j)) {
                            cut_adjustments[i][j] -= cut.dual_value;
                            cut_adjustments[j][i] -= cut.dual_value;
                        }
                    }
                }
            }
        }

        for (int i = 1; i < N_SIZE - 1; ++i) {
            std::vector<NodeScore> node_scores;
            node_scores.reserve(N_SIZE - 2);

            for (int j = 1; j < N_SIZE - 1; ++j) {
                if (i != j) {
                    // Basic score calculation
                    double base_score =
                        distances[i][j] - (nodes[i].cost + nodes[j].cost) / 2 -
                        arc_duals.getDual(i, j) + cut_adjustments[i][j];

                    // Add historical performance component
                    double historical_score = getHistoricalScore(i, j);

                    // Combine scores with weighting
                    double final_score = (1 - HISTORY_WEIGHT) * base_score +
                                         HISTORY_WEIGHT * historical_score;

                    node_scores.emplace_back(i, j, final_score);
                }
            }

            // Sort and select top candidates
            if (node_scores.size() > MAX_CANDIDATES_PER_NODE) {
                std::partial_sort(node_scores.begin(),
                                  node_scores.begin() + MAX_CANDIDATES_PER_NODE,
                                  node_scores.end(),
                                  [](const auto& a, const auto& b) {
                                      return a.cost_score < b.cost_score;
                                  });
            } else {
                std::sort(node_scores.begin(), node_scores.end(),
                          [](const auto& a, const auto& b) {
                              return a.cost_score < b.cost_score;
                          });
            }

            auto& i_scores = scores[i];
            for (int j = 0; j < std::min(MAX_CANDIDATES_PER_NODE,
                                         static_cast<int>(node_scores.size()));
                 ++j) {
                i_scores.push_back(node_scores[j].other_node);
            }
        }

        return scores;
    }
};
