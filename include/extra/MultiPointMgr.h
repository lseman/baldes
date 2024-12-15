// MultiPointManager.h
#pragma once

#include "Definitions.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <deque>
#include <limits>
#include <numeric>
#include <vector>

/**
 * @class MultiPointManager
 * @brief Enhanced manager for multi-point dual stabilization in column generation.
 *
 * This class implements an advanced stabilization strategy for dual variables
 * in column generation algorithms. It maintains a pool of stability points
 * and uses adaptive weighting strategies to improve convergence.
 */
class MultiPointManager {
public:
    /**
     * @struct StabilityPoint
     * @brief Represents a single stability point in the pool
     */
    struct StabilityPoint {
        DualSolution duals;           ///< Dual solution vector
        double       objective_value; ///< Objective value at this point
        int          age;             ///< Number of iterations since addition
        double       quality_score;   ///< Computed quality metric

        StabilityPoint() : objective_value(0.0), age(0), quality_score(0.0) {}

        StabilityPoint(const DualSolution &d, double obj)
            : duals(d), objective_value(obj), age(0), quality_score(0.0) {}
    };

    /**
     * @struct Metrics
     * @brief Tracks convergence metrics and stabilization parameters
     */
    struct Metrics {
        double last_improvement;     ///< Most recent relative improvement
        int    stagnant_iterations;  ///< Consecutive iterations with small improvement
        double average_improvement;  ///< Moving average of improvements
        double variance_improvement; ///< Variance in improvement values

        static constexpr int MAX_STAGNANT = 2; // Maximum allowed stagnant iterations

        Metrics()
            : last_improvement(1.0), stagnant_iterations(0), average_improvement(0.0), variance_improvement(0.0) {}
    };

    // Configuration constants
    static constexpr double EPSILON             = 1e-12; ///< Numerical precision threshold
    static constexpr double FAST_CONV_THRESHOLD = 5e-2;  ///< Base convergence threshold
    static constexpr size_t MIN_POOL_SIZE       = 3;     ///< Minimum stability points
    static constexpr size_t MAX_POOL_SIZE       = 10;     ///< Maximum stability points
    static constexpr double MIN_IMPROVEMENT     = 1e-2;  ///< Minimum meaningful improvement
    static constexpr size_t HISTORY_SIZE        = 10;     ///< Size of improvement history
    static constexpr double AGE_DECAY_RATE      = 0.5;   ///< Rate of age-based quality decay
    static constexpr double QUALITY_OBJ_WEIGHT  = 0.7;   ///< Weight for objective in quality score

private:
    std::vector<StabilityPoint> stability_points;    ///< Pool of stability points
    std::deque<double>          improvement_history; ///< Recent improvement values
    size_t                      current_pool_size;   ///< Current desired pool size
    Metrics                     metrics;             ///< Convergence metrics
    bool                        active;              ///< Stabilization active flag
    size_t                      solution_size;       ///< Size of dual solution vector
    double                      best_objective;      ///< Best objective seen
    double                      worst_objective;     ///< Worst objective seen
    double                      conv_threshold;      ///< Current convergence threshold

public:
    /**
     * @brief Constructor
     * @param n Size of dual solution vector
     */
    explicit MultiPointManager(size_t n)
        : current_pool_size(5), active(true), solution_size(n),
          best_objective(-std::numeric_limits<double>::infinity()),
          worst_objective(std::numeric_limits<double>::infinity()), conv_threshold(FAST_CONV_THRESHOLD) {
        assert(n > 0);
        // improvement_history.reserve(HISTORY_SIZE);
    }

    /**
     * @brief Updates the stability point pool with a new solution
     * @param new_point New dual solution
     * @param obj_value Objective value at new point
     */
    void updatePool(const DualSolution &new_point, double obj_value) {
        if (!active) return;

        assert(new_point.size() == solution_size);

        updateObjectiveBounds(obj_value);
        double relative_improvement = calculateImprovement(obj_value);
        updateMetrics(relative_improvement);

        if (shouldDeactivate()) {
            active = false;
            return;
        }

        adjustPoolSize(relative_improvement);

        if (shouldAddPoint(new_point, obj_value)) {
            addPoint(new_point, obj_value);
            updatePointQualities();
            maintainPoolSize();
        }
    }

    /**
     * @brief Computes stabilized dual solution
     * @return Weighted combination of stability points
     */
    DualSolution getWeightedSolution() const {
        if (stability_points.empty()) { return DualSolution(solution_size, 0.0); }

        if (isNearingConvergence()) { return getBestPoint().duals; }

        return computeWeightedSolution();
    }

    /**
     * @brief Computes adaptive weight for directional step
     * @param dir_sol Directional solution
     * @param mp_sol Master problem solution
     * @param subgradient Subgradient direction
     * @param subgradient_norm Norm of subgradient
     * @return Adaptive weight in [0.1, 0.8]
     */
    double computeAdaptiveWeight(const DualSolution &dir_sol, const DualSolution &mp_sol,
                                 const DualSolution &subgradient, double subgradient_norm) const {
        if (stability_points.empty() || subgradient_norm < EPSILON) { return 0.5; }

        if (metrics.stagnant_iterations > 0) {
            double stagnation_factor = std::min(0.8, 0.1 * metrics.stagnant_iterations);
            return 0.1 + stagnation_factor;
        }

        std::vector<double> diff_dir(solution_size);
        double              diff_norm = computeDirection(diff_dir, dir_sol, mp_sol);

        if (diff_norm < EPSILON) return 0.4;

        double alignment          = computeAlignment(diff_dir, diff_norm, subgradient, subgradient_norm);
        double improvement_factor = std::exp(-metrics.average_improvement / FAST_CONV_THRESHOLD);

        double base_weight     = 0.4 - alignment * 0.6;
        double adjusted_weight = base_weight * improvement_factor;

        return std::clamp(adjusted_weight, 0.1, 0.8);
    }

    /**
     * @brief Check if stabilization is active
     */
    bool isActive() const { return active; }

    /**
     * @brief Get current metrics
     */
    const Metrics &getMetrics() const { return metrics; }

    /**
     * @brief Deactivate stabilization
     */
    void deactivate() { active = false; }

    /**
     * @brief Get current pool size
     */
    size_t getPoolSize() const { return stability_points.size(); }

    /**
     * @brief Reset manager state
     */
    void clear() {
        stability_points.clear();
        improvement_history.clear();
        current_pool_size = 4;
        metrics           = Metrics();
        best_objective    = -std::numeric_limits<double>::infinity();
        worst_objective   = std::numeric_limits<double>::infinity();
        conv_threshold    = FAST_CONV_THRESHOLD;
        active            = true;
    }

private:
    /**
     * @brief Update best and worst objective bounds
     */
    void updateObjectiveBounds(double obj_value) {
        best_objective  = std::max(best_objective, obj_value);
        worst_objective = std::min(worst_objective, obj_value);
    }

    /**
     * @brief Calculate relative improvement from new solution
     */
    double calculateImprovement(double obj_value) {
        double best_known = getBestObjectiveValue();
        return std::abs((obj_value - best_known) / (std::abs(best_known) + EPSILON));
    }

    /**
     * @brief Update convergence metrics with new improvement
     */
    void updateMetrics(double relative_improvement) {
        metrics.last_improvement = relative_improvement;
        improvement_history.push_back(relative_improvement);

        if (improvement_history.size() > HISTORY_SIZE) { improvement_history.pop_front(); }

        // Update statistical metrics
        metrics.average_improvement =
            std::accumulate(improvement_history.begin(), improvement_history.end(), 0.0) / improvement_history.size();

        metrics.variance_improvement = std::accumulate(improvement_history.begin(), improvement_history.end(), 0.0,
                                                       [&](double acc, double imp) {
                                                           double diff = imp - metrics.average_improvement;
                                                           return acc + diff * diff;
                                                       }) /
                                       improvement_history.size();

        // Adapt convergence threshold
        conv_threshold = FAST_CONV_THRESHOLD * (1.0 + std::sqrt(metrics.variance_improvement));
    }

    /**
     * @brief Check if stabilization should be deactivated
     */
    bool shouldDeactivate() const {
        return metrics.stagnant_iterations >= Metrics::MAX_STAGNANT ||
               (improvement_history.size() == HISTORY_SIZE && allSmallImprovements() &&
                metrics.variance_improvement < EPSILON);
    }

    /**
     * @brief Adjust pool size based on convergence behavior
     */
    void adjustPoolSize(double relative_improvement) {
        if (relative_improvement < conv_threshold) {
            metrics.stagnant_iterations++;
            current_pool_size = std::max(MIN_POOL_SIZE, current_pool_size - 1);
        } else {
            metrics.stagnant_iterations = 0;
            current_pool_size           = std::min(MAX_POOL_SIZE, current_pool_size + 1);
        }
    }

    /**
     * @brief Update quality scores for all points
     */
    void updatePointQualities() {
        double obj_range = best_objective - worst_objective + EPSILON;

        for (auto &point : stability_points) {
            double obj_quality  = (point.objective_value - worst_objective) / obj_range;
            double age_quality  = std::exp(-AGE_DECAY_RATE * point.age);
            point.quality_score = QUALITY_OBJ_WEIGHT * obj_quality + (1.0 - QUALITY_OBJ_WEIGHT) * age_quality;
        }
    }

    /**
     * @brief Compute weighted combination of stability points
     */
    DualSolution computeWeightedSolution() const {
        DualSolution result(solution_size, 0.0);
        auto [weights, total_weight] = computeWeights();

        if (total_weight > EPSILON) {
            for (size_t i = 0; i < stability_points.size(); i++) {
                double normalized_weight = weights[i] / total_weight;
                for (size_t j = 0; j < solution_size; j++) {
                    result[j] += normalized_weight * stability_points[i].duals[j];
                }
            }
        }

        return result;
    }

    /**
     * @brief Check if approaching convergence
     */
    bool isNearingConvergence() const {
        return metrics.last_improvement < conv_threshold || metrics.stagnant_iterations > 0;
    }

    /**
     * @brief Check if all recent improvements are small
     */
    bool allSmallImprovements() const {
        return std::all_of(improvement_history.begin(), improvement_history.end(),
                           [this](double imp) { return imp < conv_threshold; });
    }

    /**
     * @brief Get best objective value in pool
     */
    double getBestObjectiveValue() const {
        if (stability_points.empty()) { return -std::numeric_limits<double>::infinity(); }
        return std::max_element(stability_points.begin(), stability_points.end(),
                                [](const auto &a, const auto &b) { return a.objective_value < b.objective_value; })
            ->objective_value;
    }

    /**
     * @brief Get stability point with best objective
     */
    const StabilityPoint &getBestPoint() const {
        assert(!stability_points.empty());
        return *std::max_element(stability_points.begin(), stability_points.end(),
                                 [](const auto &a, const auto &b) { return a.objective_value < b.objective_value; });
    }

    /**
     * @brief Check if new point should be added to pool
     */
    bool shouldAddPoint(const DualSolution &new_point, double obj_value) const {
        return std::none_of(stability_points.begin(), stability_points.end(), [&](const auto &point) {
            return vectorNorm(point.duals, new_point) < EPSILON ||
                   std::abs(point.objective_value - obj_value) < MIN_IMPROVEMENT;
        });
    }

    /**
     * @brief Add new point to stability pool
     */
    void addPoint(const DualSolution &new_point, double obj_value) {
        stability_points.emplace_back(new_point, obj_value);
        for (auto &point : stability_points) {
            if (&point != &stability_points.back()) { point.age++; }
        }
    }

    /**
     * @brief Maintain pool size by removing lowest quality points
     */
    void maintainPoolSize() {
        if (stability_points.size() > current_pool_size) {
            std::sort(stability_points.begin(), stability_points.end(),
                      [](const auto &a, const auto &b) { return a.quality_score > b.quality_score; });
            stability_points.resize(current_pool_size);
        }
    }

    /**
     * @brief Compute weights for stability points
     */
    std::pair<std::vector<double>, double> computeWeights() const {
        std::vector<double> weights(stability_points.size());
        double              total_weight = 0.0;

        for (size_t i = 0; i < stability_points.size(); i++) {
            weights[i] = stability_points[i].quality_score;
            total_weight += weights[i];
        }

        return {weights, total_weight};
    }

    /**
     * @brief Compute direction between solutions
     */
    double computeDirection(std::vector<double> &diff_dir, const DualSolution &dir_sol,
                            const DualSolution &mp_sol) const {
        double diff_norm = 0.0;
        for (size_t i = 0; i < solution_size; i++) {
            diff_dir[i] = mp_sol[i] - dir_sol[i];
            diff_norm += diff_dir[i] * diff_dir[i];
        }
        return std::sqrt(diff_norm + EPSILON);
    }

    /**
     * @brief Compute alignment between two directions
     */
    double computeAlignment(const std::vector<double> &diff_dir, double diff_norm, const DualSolution &subgradient,
                            double subgradient_norm) const {
        double dot_product = 0.0;
        for (size_t i = 0; i < solution_size; i++) {
            dot_product += (diff_dir[i] / diff_norm) * (subgradient[i] / subgradient_norm);
        }
        return std::clamp(dot_product, -1.0, 1.0);
    }

    /**
     * @brief Compute L2 norm between two vectors
     */
    double vectorNorm(const DualSolution &v1, const DualSolution &v2) const {
        assert(v1.size() == v2.size());
        double sum = 0.0;
        for (size_t i = 0; i < v1.size(); i++) {
            double diff = v2[i] - v1[i];
            sum += diff * diff;
        }
        return std::sqrt(sum + EPSILON);
    }

    /**
     * @brief Compute L2 norm of a single vector
     */
    double vectorNorm(const DualSolution &v) const {
        double sum = 0.0;
        for (double val : v) { sum += val * val; }
        return std::sqrt(sum + EPSILON);
    }

    /**
     * @brief Normalize a vector to unit length
     */
    void normalizeVector(DualSolution &v) const {
        double norm = vectorNorm(v);
        if (norm > EPSILON) {
            for (double &val : v) { val /= norm; }
        }
    }
};

// Optional: Helper functions for external use

/**
 * @brief Create stability manager with initial point
 * @param initial_duals Initial dual solution
 * @param initial_obj Initial objective value
 * @return Configured MultiPointManager
 */
inline MultiPointManager createStabilityManager(const DualSolution &initial_duals, double initial_obj) {
    MultiPointManager manager(initial_duals.size());
    manager.updatePool(initial_duals, initial_obj);
    return manager;
}

/**
 * @brief Compute relative gap between solutions
 * @param sol1 First solution
 * @param sol2 Second solution
 * @return Relative difference between solutions
 */
inline double computeRelativeGap(const DualSolution &sol1, const DualSolution &sol2) {
    assert(sol1.size() == sol2.size());
    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < sol1.size(); i++) {
        num += std::abs(sol1[i] - sol2[i]);
        den += std::abs(sol1[i]) + std::abs(sol2[i]);
    }
    return num / (den + MultiPointManager::EPSILON);
}