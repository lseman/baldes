/**
 * @file TR.h
 * @brief Defines the TrustRegion class for trust region management in
 *optimization problems.
 *
 * This file implements the TrustRegion class, which is responsible for handling
 *the trust region management during optimization in large-scale optimization
 *problems, such as vehicle routing and resource-constrained shortest path
 *problems (RCSPP).
 *
 **/
#pragma once

#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "bnb/Node.h"
#include "config.h"

class TrustRegion {
   private:
    static constexpr double EPSILON = 1e-12;
    static constexpr double INIT_V = 100.0;
    static constexpr double MIN_V = 10.0;
    static constexpr double CURVATURE_RATE =
        0.1;  // Similar to StabilFuncCurvatureAdvanceRate
    static constexpr double MAX_V = 1000.0;
    static constexpr double MAX_DUAL_CHANGE = 1000.0;
    static constexpr double MAX_EPSILON = 1000.0;
    static constexpr int HISTORY_SIZE = 5;

    bool isInsideTrustRegion;
    int numConstrs;
    std::vector<baldesVarPtr> w;     // Inner piece variables (w_i)
    std::vector<baldesVarPtr> zeta;  // Outer piece variables (zeta_i)
    std::vector<double> delta1;      // Lower bounds
    std::vector<double> delta2;      // Upper bounds
    std::vector<double> prevDuals;   // Previous dual values
    std::vector<double>
        diffHistory;  // History of differences for adaptive adjustments

    double v;      // Trust region size
    double c;      // Scaling factor
    double kappa;  // Growth control parameter
    bool TRstop;   // Trust region termination flag

    // Variables for penalty function
    double savedInnerInterval;
    double innerHalfInterval;
    double averDifference;
    double epsilon1;  // Inner bound
    double epsilon2;  // Outer bound
    double prev_inner_obj;
    double inner_obj;  // Current inner objective value

   public:
    TrustRegion(int numConstrs)
        : numConstrs(numConstrs),
          v(INIT_V),
          kappa(0.9),
          TRstop(false),
          isInsideTrustRegion(false),
          savedInnerInterval(0.0),
          innerHalfInterval(0.0),
          averDifference(0.0),
          epsilon1(10.0),
          epsilon2(10.0) {
        // Initialize vectors
        w.resize(numConstrs);
        zeta.resize(numConstrs);
        delta1.resize(numConstrs);
        delta2.resize(numConstrs);
        prevDuals.resize(numConstrs);
    }

    void adjustParameters(double progress) {
        constexpr double GOOD_PROGRESS = 0.25;
        constexpr double POOR_PROGRESS = 0.05;

        if (progress > GOOD_PROGRESS) {
            // Very good progress - be more aggressive
            kappa = std::max(0.3, kappa * 0.9);
            v = v * 1.5;
        } else if (progress > POOR_PROGRESS) {
            // Moderate progress - slight adjustment
            kappa = std::max(0.4, kappa * 0.95);
            v = v * 1.1;
        } else if (progress > 0) {
            // Minimal progress - be conservative
            kappa = std::min(0.95, kappa * 1.02);
        } else {
            // No progress - contract trust region
            kappa = std::min(0.98, kappa * 1.1);
            v = v * 0.5;
        }

        // Safeguard bounds
        v = std::max(MIN_V, std::min(v, MAX_V));
        c = v / kappa;
    }

    void setup(BNBNode *node, const std::vector<double> &nodeDuals) {
        // Calculate initial center point
        v = std::accumulate(nodeDuals.begin(), nodeDuals.end(), 0.0) /
            numConstrs;
        c = v / kappa;

        // Create variables for 3-piece function
        for (int i = 0; i < numConstrs; i++) {
            // Create w_i variables (inner piece)
            std::string w_name = "w_" + std::to_string(i + 1);
            w[i] = node->addVar(w_name, VarType::Continuous, 0, epsilon1, -1.0);

            // Create zeta_i variables (outer piece)
            std::string zeta_name = "zeta_" + std::to_string(i + 1);
            zeta[i] =
                node->addVar(zeta_name, VarType::Continuous, 0, epsilon2, 1.0);

            // Setup constraints coefficients
            auto constrs = node->getConstrs();
            node->chgCoeff(constrs[i]->index(), w[i]->index(), -1);
            node->chgCoeff(constrs[i]->index(), zeta[i]->index(), 1);
        }

        // Initialize bounds
        updateBounds(nodeDuals);

        // Set initial coefficients
        updateCoefficients(nodeDuals);

        // Initial optimization
        node->optimize();
        prevDuals = nodeDuals;

        print_info("Starting trust region with v = {}\n", v);
    }

    double calculateAverageDifference(const std::vector<double> &nodeDuals) {
        double differenceSum = 0.0;
        int numStabConstraints = 0;

        for (size_t i = 0; i < numConstrs; i++) {
            if (!prevDuals.empty()) {  // Check if we have previous duals
                double value = std::abs(nodeDuals[i] - prevDuals[i]);
                differenceSum += value;
                numStabConstraints++;
            }
        }

        return numStabConstraints > 0 ? differenceSum / numStabConstraints
                                      : 0.0;
    }

    double calculateTRScaling() {
        // Base scaling of 1.0
        double scaling = 1.0;

        // Adjust based on history stability
        if (diffHistory.size() >= 2) {
            double recentVariation = std::abs(
                diffHistory.back() - diffHistory[diffHistory.size() - 2]);
            double historyAvg =
                std::accumulate(diffHistory.begin(), diffHistory.end(), 0.0) /
                diffHistory.size();

            // If recent variation is high, increase scaling to be more
            // conservative
            if (recentVariation > historyAvg * 1.5) {
                scaling *= 1.2;
            }
            // If variation is low, decrease scaling to be more aggressive
            else if (recentVariation < historyAvg * 0.5) {
                scaling *= 0.9;
            }
        }

        // Ensure scaling stays in reasonable range
        return std::max(0.5, std::min(scaling, 2.0));
    }

    void updatePenaltyFunction(const std::vector<double> &nodeDuals) {
        // Calculate current difference and update history
        double currentDiff = calculateAverageDifference(nodeDuals);
        diffHistory.push_back(currentDiff);
        if (diffHistory.size() > HISTORY_SIZE) {
            diffHistory.erase(diffHistory.begin());
        }

        // Calculate moving average for more stable updates
        double avgDiff =
            std::accumulate(diffHistory.begin(), diffHistory.end(), 0.0) /
            diffHistory.size();

        // Smoother transition in penalty updates
        if (savedInnerInterval == 0.0) {
            savedInnerInterval = avgDiff;
        } else {
            double alpha = calculateAdaptiveRate();
            savedInnerInterval =
                savedInnerInterval + (avgDiff - savedInnerInterval) * alpha;
        }

        // More robust interval calculations
        if (savedInnerInterval < std::numeric_limits<double>::infinity()) {
            innerHalfInterval =
                savedInnerInterval * calculateAdaptiveHalfInterval();
            v = savedInnerInterval / (c * calculateTRScaling());

            // Add safeguards for v
            v = std::max(MIN_V, std::min(v, MAX_V));
        } else {
            innerHalfInterval = 0.0;
            v = MIN_V;
        }
    }

    double calculateAdaptiveRate() {
        if (std::abs(prev_inner_obj) < EPSILON) {
            return CURVATURE_RATE;  // Default rate if prev_inner_obj is close
                                    // to zero
        }

        // Adjust rate based on optimization progress
        return std::min(CURVATURE_RATE *
                            (1.0 + std::abs(prev_inner_obj - inner_obj) /
                                       std::max(std::abs(prev_inner_obj), 1.0)),
                        0.5);
    }

    double calculateAdaptiveHalfInterval() {
        // Start with base factor of 0.5
        double factor = 0.5;

        // Adjust based on history if available
        if (!diffHistory.empty()) {
            double avgDiff =
                std::accumulate(diffHistory.begin(), diffHistory.end(), 0.0) /
                diffHistory.size();
            double latestDiff = diffHistory.back();

            // If recent changes are smaller than average, reduce interval
            if (latestDiff < avgDiff * 0.8) {
                factor *= 0.9;
            }
            // If recent changes are larger than average, increase interval
            else if (latestDiff > avgDiff * 1.2) {
                factor *= 1.1;
            }
        }

        // Ensure factor stays in reasonable range
        return std::max(0.3, std::min(factor, 0.7));
    }

    void updateBounds(const std::vector<double> &nodeDuals) {
        for (int i = 0; i < numConstrs; i++) {
            delta1[i] = nodeDuals[i] - v / 2;
            delta2[i] = nodeDuals[i] + v / 2;
        }
    }

    void updateCoefficients(const std::vector<double> &nodeDuals) {
        for (int i = 0; i < numConstrs; i++) {
            // Update w_i coefficients (inner piece)
            w[i]->setOBJ(std::max(0.0, nodeDuals[i] - v));

            // Update zeta_i coefficients (outer piece)
            zeta[i]->setOBJ(nodeDuals[i] + v);
        }
    }

    double iterate(BNBNode *node, std::vector<double> &nodeDuals,
                   double inner_obj, int stage) {
        isInsideTrustRegion = true;
        TRstop = false;
        inner_obj = inner_obj;

        // Check if solution is inside trust region
        for (int i = 0; i < numConstrs; i++) {
            if (nodeDuals[i] < delta1[i] || nodeDuals[i] > delta2[i]) {
                isInsideTrustRegion = false;
                break;
            }
        }

        double dualDiffSum = 0.0;
        for (int i = 0; i < numConstrs; i++) {
            dualDiffSum += std::abs(nodeDuals[i] - prevDuals[i]);
        }

        if (isInsideTrustRegion && dualDiffSum > EPSILON) {
            double avg_diff = dualDiffSum / numConstrs;

            // More conservative v updates
            if (avg_diff < v) {
                v = std::max(v * 0.9, avg_diff);
            } else {
                v = std::min(v * 1.1, avg_diff);
            }

            // Adjust kappa based on progress
            double progress = (inner_obj - prev_inner_obj) /
                              std::abs(prev_inner_obj + EPSILON);
            adjustParameters(progress);

            // More robust epsilon calculation
            double new_eps = std::max(MIN_V, std::min(v / c, MAX_EPSILON));

            // Update variable bounds and coefficients
            updateVariables(nodeDuals, new_eps);
        }

        updateBounds(nodeDuals);

        if (v <= MIN_V || stage == 4) {
            cleanup(node);
            return 0.0;
        }

        prevDuals = nodeDuals;
        prev_inner_obj = inner_obj;
        return v;
    }

    double safeguardValue(double value, double min_val, double max_val) {
        return std::max(min_val, std::min(max_val, value));
    }

    void updateVariables(const std::vector<double> &nodeDuals, double new_eps) {
        for (int i = 0; i < numConstrs; i++) {
            // Safeguard the dual values
            double safeDual =
                safeguardValue(nodeDuals[i], -MAX_DUAL_CHANGE, MAX_DUAL_CHANGE);

            // Update bounds with safeguards
            w[i]->setUB(new_eps);
            zeta[i]->setUB(new_eps);

            // Update coefficients with safeguards
            w[i]->setOBJ(
                std::max(0.0, safeDual - safeguardValue(v, 0.0, MAX_V)));
            zeta[i]->setOBJ(safeDual + safeguardValue(v, 0.0, MAX_V));
        }
    }

    bool stop() { return TRstop; }

   private:
    void cleanup(BNBNode *node) {
        v = 0;
        int counter = 0;
        for (size_t i = 0; i < w.size(); i++) {
            node->remove(w[i]);
            node->remove(zeta[i]);
            counter++;
        }
        // fmt::print("HERE: Removed {} variables\n", counter);
        TRstop = true;
        node->update();
        node->optimize();
    }
};
