#pragma once

#include "Definitions.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

/**
 * @class MultiPointManager
 * @brief Manages multi-point stabilization aspects for column generation.
 */
class MultiPointManager {
public:
    struct StabilityPoint {
        DualSolution duals;
        double       objective_value;
        int          age;
        StabilityPoint() : objective_value(0.0), age(0) {}
        StabilityPoint(const DualSolution &d, double obj) : duals(d), objective_value(obj), age(0) {}
    };

    struct Metrics {
        double               last_improvement;
        int                  stagnant_iterations;
        static constexpr int MAX_STAGNANT = 1;  // Reduced for faster deactivation

        Metrics() : last_improvement(1.0), stagnant_iterations(0) {}
    };

    static constexpr double EPSILON              = 1e-10;
    static constexpr double FAST_CONV_THRESHOLD  = 1e-3;
    static constexpr size_t MIN_POOL_SIZE        = 2;
    static constexpr size_t MAX_POOL_SIZE        = 5;
    static constexpr double MIN_IMPROVEMENT      = 1e-4;
    static constexpr size_t HISTORY_SIZE         = 3;

private:
    std::vector<StabilityPoint> stability_points;
    std::vector<double>         improvement_history;
    size_t                      current_pool_size;
    Metrics                     metrics;
    bool                        active;
    size_t                      solution_size;
    double                      best_objective;
    double                      worst_objective;

public:
    MultiPointManager(size_t n) 
        : current_pool_size(3)
        , active(true)
        , solution_size(n)
        , best_objective(-std::numeric_limits<double>::infinity())
        , worst_objective(std::numeric_limits<double>::infinity()) {
        improvement_history.reserve(HISTORY_SIZE);
    }

    void updatePool(const DualSolution &new_point, double obj_value) {
        if (!active) return;

        // Update objective bounds
        best_objective = std::max(best_objective, obj_value);
        worst_objective = std::min(worst_objective, obj_value);

        // Calculate improvement
        double best_known = getBestObjectiveValue();
        double relative_improvement = std::abs((obj_value - best_known) / (std::abs(best_known) + EPSILON));
        
        // Track improvement history
        improvement_history.push_back(relative_improvement);
        if (improvement_history.size() > HISTORY_SIZE) {
            improvement_history.erase(improvement_history.begin());
        }

        // Update metrics and check convergence
        metrics.last_improvement = relative_improvement;
        if (relative_improvement < FAST_CONV_THRESHOLD) {
            metrics.stagnant_iterations++;
            if (metrics.stagnant_iterations >= Metrics::MAX_STAGNANT || 
                (improvement_history.size() == HISTORY_SIZE && 
                 allSmallImprovements())) {
                active = false;
                return;
            }
            current_pool_size = MIN_POOL_SIZE;
        } else {
            metrics.stagnant_iterations = 0;
            current_pool_size = std::min(MAX_POOL_SIZE, current_pool_size + 1);
        }

        if (shouldAddPoint(new_point, obj_value)) {
            addPoint(new_point, obj_value);
            maintainPoolSize();
        }
    }

    DualSolution getWeightedSolution() const {
        DualSolution result(solution_size, 0.0);
        
        if (stability_points.empty()) return result;

        if (metrics.last_improvement < FAST_CONV_THRESHOLD || 
            metrics.stagnant_iterations > 0) {
            return getBestPoint().duals;
        }

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

    double computeAdaptiveWeight(const DualSolution &dir_sol, 
                               const DualSolution &mp_sol,
                               const DualSolution &subgradient, 
                               double subgradient_norm) const {
        if (stability_points.empty() || subgradient_norm < EPSILON) {
            return 0.5;
        }

        if (metrics.stagnant_iterations > 0) {
            return 0.1;  // Favor directional component when stagnating
        }

        std::vector<double> diff_dir(solution_size);
        double diff_norm = computeDirection(diff_dir, dir_sol, mp_sol);

        if (diff_norm < EPSILON) return 0.3;

        double dot_product = computeAlignment(diff_dir, diff_norm, subgradient, subgradient_norm);
        
        double base_weight = 0.3 - dot_product * 0.5;
        return std::clamp(base_weight, 0.1, 0.7);
    }

    bool isActive() const { return active; }
    const Metrics& getMetrics() const { return metrics; }
    void deactivate() { active = false; }
    size_t getPoolSize() const { return stability_points.size(); }
    
    void clear() { 
        stability_points.clear();
        improvement_history.clear();
        current_pool_size = 3;
        metrics = Metrics();
        best_objective = -std::numeric_limits<double>::infinity();
        worst_objective = std::numeric_limits<double>::infinity();
        active = true;
    }

private:
    bool allSmallImprovements() const {
        return std::all_of(improvement_history.begin(), improvement_history.end(),
            [this](double imp) { return imp < FAST_CONV_THRESHOLD; });
    }

    double getBestObjectiveValue() const {
        if (stability_points.empty()) {
            return std::numeric_limits<double>::lowest();
        }
        return std::max_element(
            stability_points.begin(), 
            stability_points.end(),
            [](const auto &a, const auto &b) { return a.objective_value < b.objective_value; }
        )->objective_value;
    }

    const StabilityPoint& getBestPoint() const {
        return *std::max_element(
            stability_points.begin(), 
            stability_points.end(),
            [](const auto &a, const auto &b) { return a.objective_value < b.objective_value; }
        );
    }

    bool shouldAddPoint(const DualSolution &new_point, double obj_value) const {
        return std::none_of(stability_points.begin(), stability_points.end(),
            [&](const auto &point) {
                return norm(point.duals, new_point) < EPSILON ||
                       std::abs(point.objective_value - obj_value) < MIN_IMPROVEMENT;
            });
    }

    void addPoint(const DualSolution &new_point, double obj_value) {
        stability_points.emplace_back(new_point, obj_value);
        for (auto &point : stability_points) {
            if (&point != &stability_points.back()) {
                point.age++;
            }
        }
    }

    void maintainPoolSize() {
        if (stability_points.size() > current_pool_size) {
            std::sort(stability_points.begin(), stability_points.end(),
                [](const auto &a, const auto &b) {
                    if (std::abs(a.objective_value - b.objective_value) > MIN_IMPROVEMENT) {
                        return a.objective_value > b.objective_value;
                    }
                    return a.age < b.age;
                });
            stability_points.resize(current_pool_size);
        }
    }

    std::pair<std::vector<double>, double> computeWeights() const {
        std::vector<double> weights(stability_points.size());
        double total_weight = 0.0;

        double obj_range = best_objective - worst_objective;

        for (size_t i = 0; i < stability_points.size(); i++) {
            const auto &point = stability_points[i];
            double age_factor = 1.0 / (1.0 + 0.7 * point.age);  // Faster decay
            double obj_factor = obj_range > EPSILON ? 
                (point.objective_value - worst_objective) / obj_range : 1.0;
            
            weights[i] = age_factor * (0.2 + 0.8 * obj_factor);  // More emphasis on quality
            total_weight += weights[i];
        }

        return {weights, total_weight};
    }

    double computeDirection(std::vector<double>& diff_dir,
                          const DualSolution& dir_sol,
                          const DualSolution& mp_sol) const {
        double diff_norm = 0.0;
        for (size_t i = 0; i < solution_size; i++) {
            diff_dir[i] = mp_sol[i] - dir_sol[i];
            diff_norm += diff_dir[i] * diff_dir[i];
        }
        return std::sqrt(diff_norm + EPSILON);
    }

    double computeAlignment(const std::vector<double>& diff_dir,
                          double diff_norm,
                          const DualSolution& subgradient,
                          double subgradient_norm) const {
        double dot_product = 0.0;
        for (size_t i = 0; i < solution_size; i++) {
            dot_product += (diff_dir[i] / diff_norm) * (subgradient[i] / subgradient_norm);
        }
        return dot_product;
    }

    double norm(const DualSolution &v1, const DualSolution &v2) const {
        double sum = 0.0;
        for (size_t i = 0; i < v1.size(); i++) {
            double diff = v2[i] - v1[i];
            sum += diff * diff;
        }
        return std::sqrt(sum + EPSILON);
    }
};