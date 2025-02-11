#pragma once

#include <algorithm>
#include <cassert>
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
    struct Config {
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
    };

    struct State {
        size_t current_pool_size;
        bool active;
        double best_objective;
        double worst_objective;
        double conv_threshold;

        State(size_t initial_pool_size)
            : current_pool_size(initial_pool_size),
              active(true),
              best_objective(-std::numeric_limits<double>::infinity()),
              worst_objective(std::numeric_limits<double>::infinity()),
              conv_threshold(Config::FAST_CONV_THRESHOLD) {}
    };

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

    class WeightedSolutionCalculator {
       private:
        const std::vector<StabilityPoint>& points;
        const size_t solution_size;

       public:
        WeightedSolutionCalculator(const std::vector<StabilityPoint>& p,
                                   size_t size)
            : points(p), solution_size(size) {}

        DualSolution compute() const {
            if (points.empty()) {
                return DualSolution(solution_size, 0.0);
            }

            auto [weights, total_weight] = computeWeights();
            if (total_weight <= Config::EPSILON) {
                return DualSolution(solution_size, 0.0);
            }

            return computeWeightedAverage(weights, total_weight);
        }

       private:
        std::pair<std::vector<double>, double> computeWeights() const {
            std::vector<double> weights;
            weights.reserve(points.size());
            double total = 0.0;

            for (const auto& point : points) {
                weights.push_back(point.quality_score);
                total += point.quality_score;
            }

            return {weights, total};
        }

        DualSolution computeWeightedAverage(const std::vector<double>& weights,
                                            double total_weight) const {
            DualSolution result(solution_size, 0.0);
            for (size_t i = 0; i < points.size(); i++) {
                double normalized_weight = weights[i] / total_weight;
                addWeightedPoint(result, points[i].duals, normalized_weight);
            }
            return result;
        }

        void addWeightedPoint(DualSolution& result,
                              const DualSolution& point_duals,
                              double weight) const {
            std::transform(
                point_duals.begin(), point_duals.end(), result.begin(),
                result.begin(),
                [w = weight](double x, double y) { return y + w * x; });
        }
    };

    State state;
    const size_t solution_size;
    std::vector<StabilityPoint> stability_points;
    std::deque<double> improvement_history;
    Metrics metrics;
    std::shared_ptr<StabilityPoint> best_point;

    void updateObjectiveBounds(double obj_value) {
        state.best_objective = std::max(state.best_objective, obj_value);
        state.worst_objective = std::min(state.worst_objective, obj_value);
    }

    void updateMetrics(double improvement) {
        metrics.last_improvement = improvement;
        improvement_history.push_back(improvement);
        improvement_history.pop_front();

        updateAverageAndVariance();
        state.conv_threshold = Config::FAST_CONV_THRESHOLD *
                               (1.0 + std::sqrt(metrics.variance_improvement));
    }

    void updateAverageAndVariance() {
        metrics.average_improvement =
            std::accumulate(improvement_history.begin(),
                            improvement_history.end(), 0.0) /
            Config::HISTORY_SIZE;

        metrics.variance_improvement =
            std::transform_reduce(
                improvement_history.begin(), improvement_history.end(),
                improvement_history.begin(), 0.0, std::plus<>(),
                [avg = metrics.average_improvement](double a, double b) {
                    double diff = (a - avg);
                    return diff * diff;
                }) /
            Config::HISTORY_SIZE;
    }

    void adjustPoolSize(double improvement) {
        if (improvement < state.conv_threshold) {
            metrics.stagnant_iterations++;
            state.current_pool_size =
                std::max(Config::MIN_POOL_SIZE, state.current_pool_size - 1);
        } else {
            metrics.stagnant_iterations = 0;
            state.current_pool_size =
                std::min(Config::MAX_POOL_SIZE, state.current_pool_size + 1);
        }
    }

    void updatePointQualities() {
        if (stability_points.empty()) return;

        double obj_range =
            state.best_objective - state.worst_objective + Config::EPSILON;
        double obj_factor = Config::QUALITY_OBJ_WEIGHT / obj_range;
        double age_factor = 1.0 - Config::QUALITY_OBJ_WEIGHT;

        for (auto& point : stability_points) {
            point.quality_score =
                obj_factor * (point.objective_value - state.worst_objective) +
                age_factor * std::exp(-Config::AGE_DECAY_RATE * point.age);
        }
    }

    void maintainPoolSize() {
        if (stability_points.size() <= state.current_pool_size) return;

        std::partial_sort(stability_points.begin(),
                          stability_points.begin() + state.current_pool_size,
                          stability_points.end(),
                          [](const auto& a, const auto& b) {
                              return a.quality_score > b.quality_score;
                          });
        stability_points.resize(state.current_pool_size);
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

    double vectorNorm(const DualSolution& v1, const DualSolution& v2) const {
        return std::sqrt(std::transform_reduce(v1.begin(), v1.end(), v2.begin(),
                                               0.0, std::plus<>(),
                                               [](double a, double b) {
                                                   double diff = b - a;
                                                   return diff * diff;
                                               }) +
                         Config::EPSILON);
    }

    static double computeAlignment(const std::vector<double>& dir,
                                   double dir_norm, const DualSolution& subgrad,
                                   double subgrad_norm) {
        double dot_product =
            std::inner_product(dir.begin(), dir.end(), subgrad.begin(), 0.0) /
            (dir_norm * subgrad_norm);
        return std::clamp(dot_product, -1.0, 1.0);
    }

   public:
    explicit MultiPointManager(size_t n)
        : state(Config::MIN_POOL_SIZE + 2),
          solution_size(n),
          improvement_history(Config::HISTORY_SIZE, 0.0),
          best_point(nullptr) {
        if (n == 0)
            throw std::invalid_argument("Solution size must be positive");
        stability_points.reserve(Config::MAX_POOL_SIZE);
    }

    void updatePool(const DualSolution& new_point, double obj_value) {
        if (!state.active || new_point.empty() ||
            new_point.size() != solution_size)
            return;

        updateObjectiveBounds(obj_value);
        double improvement = calculateImprovement(obj_value);
        updateMetrics(improvement);

        if (shouldDeactivate()) {
            state.active = false;
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

        if (isNearingConvergence() && best_point) {
            return best_point->duals;
        }

        return WeightedSolutionCalculator(stability_points, solution_size)
            .compute();
    }

    double computeAdaptiveWeight(const DualSolution& dir_sol,
                                 const DualSolution& mp_sol,
                                 const DualSolution& subgradient,
                                 double subgradient_norm) const {
        if (stability_points.empty() || subgradient_norm < Config::EPSILON)
            return 0.5;

        if (metrics.stagnant_iterations > 0) {
            return 0.1 + std::min(0.8, 0.1 * metrics.stagnant_iterations);
        }

        std::vector<double> diff_dir(solution_size);
        double diff_norm = computeDirection(diff_dir, dir_sol, mp_sol);
        if (diff_norm < Config::EPSILON) return 0.4;

        double alignment = computeAlignment(diff_dir, diff_norm, subgradient,
                                            subgradient_norm);
        double improvement_factor = std::exp(-metrics.average_improvement /
                                             Config::FAST_CONV_THRESHOLD);

        return std::clamp((0.4 - alignment * 0.6) * improvement_factor, 0.1,
                          0.8);
    }

    bool isActive() const { return state.active; }
    const Metrics& getMetrics() const { return metrics; }
    void deactivate() { state.active = false; }
    size_t getPoolSize() const { return stability_points.size(); }

    void clear() {
        stability_points.clear();
        improvement_history.assign(Config::HISTORY_SIZE, 0.0);
        state = State(Config::MIN_POOL_SIZE + 2);
        metrics = Metrics();
        best_point = nullptr;
    }

   private:
    double calculateImprovement(double obj_value) const {
        double best_known = best_point
                                ? best_point->objective_value
                                : -std::numeric_limits<double>::infinity();
        return std::abs((obj_value - best_known) /
                        (std::abs(best_known) + Config::EPSILON));
    }

    bool shouldDeactivate() const {
        return metrics.stagnant_iterations >= Metrics::MAX_STAGNANT ||
               (improvement_history.back() != 0.0 &&
                std::all_of(improvement_history.begin(),
                            improvement_history.end(),
                            [this](double imp) {
                                return imp < state.conv_threshold;
                            }) &&
                metrics.variance_improvement < Config::EPSILON);
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
        if (stability_points.empty()) {
            best_point = nullptr;
            return;
        }

        best_point = std::make_shared<StabilityPoint>(
            *std::max_element(stability_points.begin(), stability_points.end(),
                              [](const auto& a, const auto& b) {
                                  return a.objective_value < b.objective_value;
                              }));
    }

    bool isNearingConvergence() const {
        return metrics.last_improvement < state.conv_threshold ||
               metrics.stagnant_iterations > 0;
    }

    double computeDirection(std::vector<double>& diff_dir,
                            const DualSolution& dir_sol,
                            const DualSolution& mp_sol) const {
        std::transform(mp_sol.begin(), mp_sol.end(), dir_sol.begin(),
                       diff_dir.begin(), std::minus<>());

        return std::sqrt(std::inner_product(diff_dir.begin(), diff_dir.end(),
                                            diff_dir.begin(), 0.0) +
                         Config::EPSILON);
    }
};
