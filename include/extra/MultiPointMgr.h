#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <deque>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "Definitions.h"

class MultiPointManager {
   public:
    // Configuration
    struct Config {
        static constexpr double EPSILON = 1e-12;
        static constexpr double FAST_CONV_THRESHOLD = 5e-2;
        static constexpr size_t MIN_POOL_SIZE = 3;
        static constexpr size_t MAX_POOL_SIZE = 10;
        static constexpr double MIN_IMPROVEMENT = 1e-2;
        static constexpr size_t HISTORY_SIZE = 10;
        static constexpr double AGE_DECAY_RATE = 0.5;
        static constexpr double QUALITY_OBJ_WEIGHT = 0.7;
    };

    // Internal structures
    struct StabilityPoint {
        DualSolution duals;
        double objective_value;
        int age;
        double quality_score;

        StabilityPoint() : objective_value(0.0), age(0), quality_score(0.0) {}

        StabilityPoint(const DualSolution& d, double obj)
            : duals(d), objective_value(obj), age(0), quality_score(0.0) {}
    };

    struct Metrics {
        double last_improvement;
        int stagnant_iterations;
        double average_improvement;
        double variance_improvement;

        static constexpr int MAX_STAGNANT = 2;

        Metrics()
            : last_improvement(1.0),
              stagnant_iterations(0),
              average_improvement(0.0),
              variance_improvement(0.0) {}
    };

    // Main class implementation
    explicit MultiPointManager(size_t n)
        : current_pool_size(Config::MIN_POOL_SIZE + 2),
          active(true),
          solution_size(n),
          best_objective(-std::numeric_limits<double>::infinity()),
          worst_objective(std::numeric_limits<double>::infinity()),
          conv_threshold(Config::FAST_CONV_THRESHOLD),
          improvement_history(Config::HISTORY_SIZE, 0.0),
          best_point(nullptr) {
        if (n == 0) {
            throw std::invalid_argument("Solution size must be positive");
        }
        stability_points.reserve(Config::MAX_POOL_SIZE);
    }

    // Move semantics
    MultiPointManager(MultiPointManager&& other) noexcept = default;
    MultiPointManager& operator=(MultiPointManager&& other) noexcept = default;

    // Core functionality
    void updatePool(const DualSolution& new_point, double obj_value) {
        if (!active) return;
        if (new_point.empty()) {
            throw std::invalid_argument("Empty solution vector");
        }
        if (new_point.size() != solution_size) {
            throw std::invalid_argument("Invalid solution size");
        }

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
            updateBestPoint();
        }
    }

    DualSolution getWeightedSolution() const {
        if (stability_points.empty()) {
            return DualSolution(solution_size, 0.0);
        }

        if (isNearingConvergence() && best_point) {
            return best_point->duals;
        }

        return computeWeightedSolution();
    }

    double computeAdaptiveWeight(const DualSolution& dir_sol,
                                 const DualSolution& mp_sol,
                                 const DualSolution& subgradient,
                                 double subgradient_norm) const {
        if (stability_points.empty() || subgradient_norm < Config::EPSILON) {
            return 0.5;
        }

        if (metrics.stagnant_iterations > 0) {
            double stagnation_factor =
                std::min(0.8, 0.1 * metrics.stagnant_iterations);
            return 0.1 + stagnation_factor;
        }

        std::vector<double> diff_dir(solution_size);
        double diff_norm = computeDirection(diff_dir, dir_sol, mp_sol);

        if (diff_norm < Config::EPSILON) return 0.4;

        double alignment = computeAlignment(diff_dir, diff_norm, subgradient,
                                            subgradient_norm);
        double improvement_factor = std::exp(-metrics.average_improvement /
                                             Config::FAST_CONV_THRESHOLD);

        double base_weight = 0.4 - alignment * 0.6;
        double adjusted_weight = base_weight * improvement_factor;

        return std::clamp(adjusted_weight, 0.1, 0.8);
    }

    // Utility methods
    bool isActive() const { return active; }
    const Metrics& getMetrics() const { return metrics; }
    void deactivate() { active = false; }
    size_t getPoolSize() const { return stability_points.size(); }

    void clear() {
        stability_points.clear();
        improvement_history.assign(Config::HISTORY_SIZE, 0.0);
        current_pool_size = Config::MIN_POOL_SIZE + 2;
        metrics = Metrics();
        best_objective = -std::numeric_limits<double>::infinity();
        worst_objective = std::numeric_limits<double>::infinity();
        conv_threshold = Config::FAST_CONV_THRESHOLD;
        active = true;
        best_point = nullptr;
    }

   private:
    // Member variables
    std::vector<StabilityPoint> stability_points;
    std::deque<double> improvement_history;
    size_t current_pool_size;
    Metrics metrics;
    bool active;
    size_t solution_size;
    double best_objective;
    double worst_objective;
    double conv_threshold;
    StabilityPoint* best_point;

    // Private methods
    void updateObjectiveBounds(double obj_value) {
        best_objective = std::max(best_objective, obj_value);
        worst_objective = std::min(worst_objective, obj_value);
    }

    double calculateImprovement(double obj_value) {
        double best_known = best_point
                                ? best_point->objective_value
                                : -std::numeric_limits<double>::infinity();
        return std::abs((obj_value - best_known) /
                        (std::abs(best_known) + Config::EPSILON));
    }

    void updateMetrics(double relative_improvement) {
        metrics.last_improvement = relative_improvement;
        improvement_history.push_back(relative_improvement);
        improvement_history.pop_front();

        metrics.average_improvement =
            std::accumulate(improvement_history.begin(),
                            improvement_history.end(), 0.0) /
            Config::HISTORY_SIZE;

        metrics.variance_improvement =
            std::inner_product(
                improvement_history.begin(), improvement_history.end(),
                improvement_history.begin(), 0.0, std::plus<>(),
                [avg = metrics.average_improvement](double a, double b) {
                    double diff = (a - avg);
                    return diff * diff;
                }) /
            Config::HISTORY_SIZE;

        conv_threshold = Config::FAST_CONV_THRESHOLD *
                         (1.0 + std::sqrt(metrics.variance_improvement));
    }

    bool shouldDeactivate() const {
        return metrics.stagnant_iterations >= Metrics::MAX_STAGNANT ||
               (improvement_history.back() != 0.0 &&  // Skip initial state
                allSmallImprovements() &&
                metrics.variance_improvement < Config::EPSILON);
    }

    void adjustPoolSize(double relative_improvement) {
        if (relative_improvement < conv_threshold) {
            metrics.stagnant_iterations++;
            current_pool_size =
                std::max(Config::MIN_POOL_SIZE, current_pool_size - 1);
        } else {
            metrics.stagnant_iterations = 0;
            current_pool_size =
                std::min(Config::MAX_POOL_SIZE, current_pool_size + 1);
        }
    }

    void updatePointQualities() {
        if (stability_points.empty()) return;

        double obj_range = best_objective - worst_objective + Config::EPSILON;
        double obj_factor = Config::QUALITY_OBJ_WEIGHT / obj_range;
        double age_factor = 1.0 - Config::QUALITY_OBJ_WEIGHT;

        for (auto& point : stability_points) {
            point.quality_score =
                obj_factor * (point.objective_value - worst_objective) +
                age_factor * std::exp(-Config::AGE_DECAY_RATE * point.age);
        }
    }

    void updateBestPoint() {
        if (!stability_points.empty()) {
            best_point = &*std::max_element(
                stability_points.begin(), stability_points.end(),
                [](const auto& a, const auto& b) {
                    return a.objective_value < b.objective_value;
                });
        } else {
            best_point = nullptr;
        }
    }

    DualSolution computeWeightedSolution() const {
        DualSolution result(solution_size, 0.0);
        auto [weights, total_weight] = computeWeights();

        if (total_weight > Config::EPSILON) {
            for (size_t i = 0; i < stability_points.size(); i++) {
                double normalized_weight = weights[i] / total_weight;
                std::transform(stability_points[i].duals.begin(),
                               stability_points[i].duals.end(), result.begin(),
                               result.begin(),
                               [w = normalized_weight](double x, double y) {
                                   return y + w * x;
                               });
            }
        }

        return result;
    }

    bool isNearingConvergence() const {
        return metrics.last_improvement < conv_threshold ||
               metrics.stagnant_iterations > 0;
    }

    bool allSmallImprovements() const {
        return std::all_of(improvement_history.begin(),
                           improvement_history.end(),
                           [this](double imp) { return imp < conv_threshold; });
    }

    bool shouldAddPoint(const DualSolution& new_point, double obj_value) const {
        return std::none_of(
            stability_points.begin(), stability_points.end(),
            [&](const auto& point) {
                return vectorNorm(point.duals, new_point) < Config::EPSILON ||
                       std::abs(point.objective_value - obj_value) <
                           Config::MIN_IMPROVEMENT;
            });
    }

    void addPoint(const DualSolution& new_point, double obj_value) {
        stability_points.emplace_back(new_point, obj_value);
        for (auto& point : stability_points) {
            if (&point != &stability_points.back()) {
                point.age++;
            }
        }
    }

    void maintainPoolSize() {
        if (stability_points.size() > current_pool_size) {
            std::partial_sort(stability_points.begin(),
                              stability_points.begin() + current_pool_size,
                              stability_points.end(),
                              [](const auto& a, const auto& b) {
                                  return a.quality_score > b.quality_score;
                              });
            stability_points.resize(current_pool_size);
        }
    }

    std::pair<std::vector<double>, double> computeWeights() const {
        std::vector<double> weights;
        weights.reserve(stability_points.size());

        double total_weight = 0.0;
        for (const auto& point : stability_points) {
            weights.push_back(point.quality_score);
            total_weight += point.quality_score;
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
        return std::sqrt(diff_norm + Config::EPSILON);
    }

    double computeAlignment(const std::vector<double>& diff_dir,
                            double diff_norm, const DualSolution& subgradient,
                            double subgradient_norm) const {
        double dot_product =
            std::inner_product(diff_dir.begin(), diff_dir.end(),
                               subgradient.begin(), 0.0) /
            (diff_norm * subgradient_norm);
        return std::clamp(dot_product, -1.0, 1.0);
    }

    double vectorNorm(const DualSolution& v1, const DualSolution& v2) const {
        assert(v1.size() == v2.size());
        return std::sqrt(std::inner_product(v1.begin(), v1.end(), v2.begin(),
                                            0.0, std::plus<>(),
                                            [](double a, double b) {
                                                double diff = b - a;
                                                return diff * diff;
                                            }) +
                         Config::EPSILON);
    }
};
