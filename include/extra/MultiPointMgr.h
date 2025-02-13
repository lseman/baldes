#pragma once

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "Definitions.h"

class MultiPointManager {
   private:
    static constexpr double EPSILON = 1e-12;
    static constexpr double FAST_CONV_THRESHOLD = 5e-2;
    static constexpr size_t MIN_POOL_SIZE = 3;
    static constexpr size_t MAX_POOL_SIZE = 10;
    static constexpr double MIN_IMPROVEMENT = 1e-2;
    static constexpr size_t HISTORY_SIZE = 10;
    static constexpr double AGE_DECAY_RATE = 0.5;
    static constexpr double QUALITY_OBJ_WEIGHT = 0.7;
    static constexpr double LEARNING_RATE = 0.1;
    static constexpr double CONVERGENCE_FACTOR = 0.9;

    struct StabilityPoint {
        DualSolution duals;
        double objective_value;
        int age;
        double quality_score;

        // Default constructor
        StabilityPoint()
            : duals(), objective_value(0.0), age(0), quality_score(0.0) {}

        StabilityPoint(const DualSolution& d, double obj)
            : duals(d), objective_value(obj), age(0), quality_score(0.0) {}
    };

    size_t current_pool_size;
    bool active;
    double best_objective;
    double worst_objective;
    double conv_threshold;
    size_t solution_size;
    std::vector<StabilityPoint> stability_points;
    std::deque<double> improvement_history;
    std::shared_ptr<StabilityPoint> best_point;

    void updateObjectiveBounds(double obj_value) {
        best_objective = std::max(best_objective, obj_value);
        worst_objective = std::min(worst_objective, obj_value);
    }

    void updateMetrics(double improvement) {
        improvement_history.push_back(improvement);
        improvement_history.pop_front();
    }

    void adjustPoolSize(double improvement) {
        if (improvement < conv_threshold) {
            current_pool_size = std::max(MIN_POOL_SIZE, current_pool_size - 1);
        } else {
            current_pool_size = std::min(MAX_POOL_SIZE, current_pool_size + 1);
        }
    }

    void updatePointQualities() {
        double obj_range = best_objective - worst_objective + EPSILON;
        double obj_factor = QUALITY_OBJ_WEIGHT / obj_range;
        double age_factor = 1.0 - QUALITY_OBJ_WEIGHT;

        for (auto& point : stability_points) {
            point.quality_score =
                obj_factor * (point.objective_value - worst_objective) +
                age_factor * std::exp(-AGE_DECAY_RATE * point.age);
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

    bool shouldAddPoint(const DualSolution& new_point, double obj_value) const {
        return std::none_of(
            stability_points.begin(), stability_points.end(),
            [&](const auto& point) {
                return vectorNorm(point.duals, new_point) < EPSILON ||
                       std::abs(point.objective_value - obj_value) <
                           MIN_IMPROVEMENT;
            });
    }

    double vectorNorm(const DualSolution& v1, const DualSolution& v2) const {
        return std::sqrt(std::transform_reduce(v1.begin(), v1.end(), v2.begin(),
                                               0.0, std::plus<>(),
                                               [](double a, double b) {
                                                   double diff = b - a;
                                                   return diff * diff;
                                               }) +
                         EPSILON);
    }

    void addPoint(const DualSolution& new_point, double obj_value) {
        stability_points.emplace_back(new_point, obj_value);
        for (auto& point : stability_points) {
            if (&point != &stability_points.back()) {
                point.age++;
            }
        }
    }

    void updateBestPoint() {
        if (!stability_points.empty()) {
            best_point = std::make_shared<StabilityPoint>(*std::max_element(
                stability_points.begin(), stability_points.end(),
                [](const auto& a, const auto& b) {
                    return a.objective_value < b.objective_value;
                }));
        }
    }

   public:
    explicit MultiPointManager(size_t n)
        : current_pool_size(MIN_POOL_SIZE + 2),
          active(true),
          best_objective(-std::numeric_limits<double>::infinity()),
          worst_objective(std::numeric_limits<double>::infinity()),
          conv_threshold(FAST_CONV_THRESHOLD),
          solution_size(n),
          improvement_history(HISTORY_SIZE, 0.0),
          best_point(nullptr) {
        if (n == 0)
            throw std::invalid_argument("Solution size must be positive");
        stability_points.reserve(MAX_POOL_SIZE);
    }

    void updatePool(const DualSolution& new_point, double obj_value) {
        if (!active || new_point.empty() || new_point.size() != solution_size)
            return;

        updateObjectiveBounds(obj_value);
        double improvement =
            std::abs((obj_value -
                      (best_point ? best_point->objective_value
                                  : -std::numeric_limits<double>::infinity())) /
                     (std::abs(best_point ? best_point->objective_value : 0.0) +
                      EPSILON));
        updateMetrics(improvement);

        if (improvement < conv_threshold && improvement_history.back() != 0.0 &&
            std::all_of(
                improvement_history.begin(), improvement_history.end(),
                [this](double imp) { return imp < conv_threshold; })) {
            active = false;
            return;
        }

        adjustPoolSize(improvement);

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

        if ((improvement_history.back() < conv_threshold) && best_point) {
            return best_point->duals;
        }

        std::vector<double> weights;
        double total_weight = 0.0;
        for (const auto& point : stability_points) {
            weights.push_back(point.quality_score);
            total_weight += point.quality_score;
        }

        if (total_weight <= EPSILON) {
            return DualSolution(solution_size, 0.0);
        }

        DualSolution result(solution_size, 0.0);
        for (size_t i = 0; i < stability_points.size(); i++) {
            double normalized_weight = weights[i] / total_weight;
            std::transform(stability_points[i].duals.begin(),
                           stability_points[i].duals.end(), result.begin(),
                           result.begin(),
                           [normalized_weight](double x, double y) {
                               return y + normalized_weight * x;
                           });
        }
        return result;
    }

    bool isActive() const { return active; }
    void deactivate() { active = false; }
    size_t getPoolSize() const { return stability_points.size(); }

    void clear() {
        stability_points.clear();
        improvement_history.assign(HISTORY_SIZE, 0.0);
        current_pool_size = MIN_POOL_SIZE + 2;
        active = true;
        best_objective = -std::numeric_limits<double>::infinity();
        worst_objective = std::numeric_limits<double>::infinity();
        conv_threshold = FAST_CONV_THRESHOLD;
        best_point = nullptr;
    }
};
