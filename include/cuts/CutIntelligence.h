#pragma once

// Store historical scores for each (i,j) pair
struct ScoreHistory {
    std::deque<double> historical_scores;
    double prediction_weight =
        1.0;  // Adaptive weight based on prediction accuracy
    static constexpr size_t MAX_HISTORY = 5;  // Keep last 5 iterations
};

// Custom hash function for int pairs
struct PairHash {
    std::size_t operator()(const std::pair<int, int> &p) const {
        return std::hash<int>{}(p.first) ^ (std::hash<int>{}(p.second) << 1);
    }
};
