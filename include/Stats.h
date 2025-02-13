#pragma once

#include <algorithm>
#include <cmath>
#include <deque>
#include <numeric>
#include <random>
#include <vector>

#include "RNG.h"

#pragma once

#include <algorithm>
#include <cmath>
#include <deque>
#include <numeric>
#include <random>

class Stats {
   private:
    // Constants
    static constexpr double MIN_THRESHOLD = -10.0;
    static constexpr double MAX_THRESHOLD = -1.5;
    static constexpr double INITIAL_BASE = -15.0;
    static constexpr double MIN_EPSILON = 0.1;
    static constexpr double MOMENTUM_FACTOR = 0.9;
    static constexpr size_t MAX_HISTORY = 100;
    static constexpr double MIN_LEARNING_RATE = 0.001;
    static constexpr double MAX_LEARNING_RATE = 0.1;
    static constexpr double NUMERICAL_STABILITY_THRESHOLD = 1e-6;

    // Core state
    std::deque<double> obj_history;
    std::deque<bool> decision_outcomes;
    double learning_rate = 0.01;
    double epsilon = 1.0;

    // Threshold state
    double prev_threshold = INITIAL_BASE;
    double momentum = 0.0;

    // RNG for exploration
    std::mt19937 rng{std::random_device{}()};

   public:
    void addIteration(double obj_value, bool decision_successful = true) {
        // Validate input
        if (!std::isfinite(obj_value)) {
            obj_value = obj_history.empty() ? 0.0 : obj_history.back();
        }

        obj_history.push_back(obj_value);
        decision_outcomes.push_back(decision_successful);

        updateLearningRate(decision_successful);
        updateExploration();

        // Keep history bounded
        while (obj_history.size() > MAX_HISTORY) {
            obj_history.pop_front();
        }
        while (decision_outcomes.size() > MAX_HISTORY) {
            decision_outcomes.pop_front();
        }
    }

    double computeThreshold(int iteration, double current_obj) {
        // Validate input
        if (!std::isfinite(current_obj)) {
            current_obj = obj_history.empty() ? 0.0 : obj_history.back();
        }
        iteration = std::max(0, iteration);

        // Get recent statistics with protection
        size_t window = std::min(static_cast<size_t>(20), obj_history.size());
        double recent_variance = getRecentVariance(window);
        double success_rate = getSuccessRate(window);

        // Base threshold with decay
        double decay_rate = 0.003 * (1.0 + safeValue(recent_variance));
        double base = INITIAL_BASE * std::exp(-decay_rate * iteration);

        // Progress factor with protection
        double progress_factor = 1.0;
        if (hasHistory()) {
            double improvement = getRecentImprovement(window);
            double denominator = 1.0 + safeValue(recent_variance);
            if (std::isfinite(improvement) &&
                denominator > NUMERICAL_STABILITY_THRESHOLD) {
                progress_factor =
                    std::clamp(improvement / denominator, 0.5, 1.5);
            }
        }

        // Combine components safely
        double threshold = safeValue(base * progress_factor);

        // Apply momentum with adaptive factor
        double adaptive_momentum =
            MOMENTUM_FACTOR / (1.0 + safeValue(recent_variance));
        momentum =
            safeValue(adaptive_momentum * momentum +
                      (1.0 - adaptive_momentum) * (threshold - prev_threshold));
        threshold = safeValue(threshold + momentum);

        // Apply success rate adjustment
        double success_adjustment = (success_rate - 0.5) * 2.0;
        threshold = safeValue(threshold + success_adjustment * learning_rate);

        // Add exploration component
        if (std::uniform_real_distribution<>(0, 1)(rng) < epsilon) {
            double exploration_range = MAX_THRESHOLD - MIN_THRESHOLD;
            double random_factor =
                2.0 * std::uniform_real_distribution<>(0, 1)(rng) - 1.0;
            threshold =
                safeValue(threshold + random_factor * exploration_range * 0.2);
        }

        // Store current threshold for next iteration
        prev_threshold = threshold;

        return std::clamp(threshold, MIN_THRESHOLD, MAX_THRESHOLD);
    }

   private:
    void updateLearningRate(bool decision_successful) {
        if (hasHistory()) {
            double performance = safeValue(getRecentImprovement(10));
            double variance = safeValue(getRecentVariance(10));

            // Adjust based on performance and stability
            double rate_adjustment = (1.0 + performance) / (1.0 + variance);
            learning_rate *=
                std::isfinite(rate_adjustment) ? rate_adjustment : 1.0;

            // Add cyclical component
            int iteration = static_cast<int>(obj_history.size());
            const double cycle_length = 50.0;
            double cycle_position =
                std::fmod(iteration, static_cast<int>(cycle_length)) /
                cycle_length;
            double cyclical_factor =
                0.5 + 0.5 * std::cos(2.0 * M_PI * cycle_position);
            learning_rate *=
                std::isfinite(cyclical_factor) ? cyclical_factor : 1.0;

            // Bound the learning rate
            learning_rate =
                std::clamp(learning_rate, MIN_LEARNING_RATE, MAX_LEARNING_RATE);
        }
    }

    void updateExploration() {
        if (decision_outcomes.size() >= 20) {
            double success_rate = getSuccessRate(20);
            double variance = safeValue(getRecentVariance(20));

            // Reduce exploration when successful and stable
            double exploration_score =
                safeValue((1.0 - success_rate) * std::sqrt(variance));
            epsilon = MIN_EPSILON +
                      (1.0 - MIN_EPSILON) * std::exp(-exploration_score);
        }
    }

    bool hasHistory() const { return obj_history.size() >= 2; }

    double getRecentImprovement(size_t window) const {
        if (obj_history.size() < window) return 1.0;

        auto start = obj_history.end() - window;
        auto end = obj_history.end();

        double total_change = 0.0;
        size_t valid_changes = 0;
        double prev = *start;

        for (auto it = start + 1; it != end; ++it) {
            double curr = *it;
            if (std::isfinite(prev) && std::isfinite(curr)) {
                double denominator =
                    std::max(NUMERICAL_STABILITY_THRESHOLD, std::abs(prev));
                double change = (prev - curr) / denominator;
                if (std::isfinite(change)) {
                    total_change += change;
                    valid_changes++;
                }
            }
            prev = curr;
        }

        return valid_changes > 0 ? total_change / valid_changes : 0.0;
    }

    double getRecentVariance(size_t window) const {
        if (obj_history.size() < window) return 0.0;

        auto start = obj_history.end() - window;
        auto end = obj_history.end();

        // Calculate mean with protection
        double sum = 0.0;
        size_t valid_count = 0;
        for (auto it = start; it != end; ++it) {
            if (std::isfinite(*it)) {
                sum += *it;
                valid_count++;
            }
        }

        if (valid_count == 0) return 0.0;
        double mean = sum / valid_count;

        // Calculate variance with protection
        double variance_sum = 0.0;
        valid_count = 0;
        for (auto it = start; it != end; ++it) {
            if (std::isfinite(*it)) {
                double diff = *it - mean;
                variance_sum += diff * diff;
                valid_count++;
            }
        }

        return valid_count > 0 ? variance_sum / valid_count : 0.0;
    }

    double getSuccessRate(size_t window) const {
        if (decision_outcomes.empty()) return 0.5;

        auto start = decision_outcomes.end() -
                     std::min(window, decision_outcomes.size());
        return std::count(start, decision_outcomes.end(), true) /
               static_cast<double>(
                   std::distance(start, decision_outcomes.end()));
    }

    // Utility function to handle potential NaN/Inf values
    static double safeValue(double value) {
        return std::isfinite(value) ? value : 0.0;
    }

   public:
    // Accessors
    double getLearningRate() const { return learning_rate; }
    double getEpsilon() const { return epsilon; }
    size_t getIterationCount() const { return obj_history.size(); }

    // Debug helpers
    double getCurrentVariance() const {
        return obj_history.size() >= 2 ? getRecentVariance(obj_history.size())
                                       : 0.0;
    }

    double getCurrentImprovement() const {
        return obj_history.size() >= 2
                   ? getRecentImprovement(obj_history.size())
                   : 0.0;
    }
};
