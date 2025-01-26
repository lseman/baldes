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
    struct Parameters {
        static constexpr double EPSILON = 1e-12;
        static constexpr double INIT_V = 100.0;
        static constexpr double MIN_V = 10.0;
        static constexpr double CURVATURE_RATE = 0.1;
        static constexpr double MAX_V = 1000.0;
        static constexpr double MAX_DUAL_CHANGE = 1000.0;
        static constexpr double MAX_EPSILON = 1000.0;
        static constexpr int HISTORY_SIZE = 5;
        static constexpr double GOOD_PROGRESS = 0.25;
        static constexpr double POOR_PROGRESS = 0.05;
    };

    struct State {
        double v = Parameters::INIT_V;  // Trust region size
        double kappa = 0.9;             // Growth control parameter
        double c = v / kappa;           // Scaling factor
        bool TRstop = false;            // Trust region termination flag
        bool isInsideTrustRegion = false;

        // Penalty function state
        double savedInnerInterval = 0.0;
        double innerHalfInterval = 0.0;
        double epsilon1 = 10.0;  // Inner bound
        double epsilon2 = 10.0;  // Outer bound
        double prev_inner_obj = 0.0;
        double inner_obj = 0.0;  // Current inner objective value
    };

    const int numConstrs;
    State state;
    std::vector<baldesVarPtr> w;      // Inner piece variables
    std::vector<baldesVarPtr> zeta;   // Outer piece variables
    std::vector<double> delta1;       // Lower bounds
    std::vector<double> delta2;       // Upper bounds
    std::vector<double> prevDuals;    // Previous dual values
    std::vector<double> diffHistory;  // History for adaptive adjustments

    void adjustParameters(double progress) {
        if (progress > Parameters::GOOD_PROGRESS) {
            state.kappa = std::max(0.3, state.kappa * 0.9);
            state.v *= 1.5;
        } else if (progress > Parameters::POOR_PROGRESS) {
            state.kappa = std::max(0.4, state.kappa * 0.95);
            state.v *= 1.1;
        } else if (progress > 0) {
            state.kappa = std::min(0.95, state.kappa * 1.02);
        } else {
            state.kappa = std::min(0.98, state.kappa * 1.1);
            state.v *= 0.5;
        }

        state.v = std::clamp(state.v, Parameters::MIN_V, Parameters::MAX_V);
        state.c = state.v / state.kappa;
    }

    double calculateAverageDifference(
        const std::vector<double>& nodeDuals) const {
        if (prevDuals.empty()) return 0.0;

        double differenceSum = 0.0;
        for (size_t i = 0; i < numConstrs; i++) {
            differenceSum += std::abs(nodeDuals[i] - prevDuals[i]);
        }
        return differenceSum / numConstrs;
    }

    double calculateTRScaling() const {
        if (diffHistory.size() < 2) return 1.0;

        double recentVariation =
            std::abs(diffHistory.back() - diffHistory[diffHistory.size() - 2]);
        double historyAvg =
            std::accumulate(diffHistory.begin(), diffHistory.end(), 0.0) /
            diffHistory.size();

        double scaling = 1.0;
        if (recentVariation > historyAvg * 1.5) {
            scaling *= 1.2;
        } else if (recentVariation < historyAvg * 0.5) {
            scaling *= 0.9;
        }

        return std::clamp(scaling, 0.5, 2.0);
    }

    double calculateAdaptiveRate() const {
        if (std::abs(state.prev_inner_obj) < Parameters::EPSILON) {
            return Parameters::CURVATURE_RATE;
        }
        return std::min(
            Parameters::CURVATURE_RATE *
                (1.0 + std::abs(state.prev_inner_obj - state.inner_obj) /
                           std::max(std::abs(state.prev_inner_obj), 1.0)),
            0.5);
    }

    double calculateAdaptiveHalfInterval() const {
        double factor = 0.5;

        if (!diffHistory.empty()) {
            double avgDiff =
                std::accumulate(diffHistory.begin(), diffHistory.end(), 0.0) /
                diffHistory.size();
            double latestDiff = diffHistory.back();

            if (latestDiff < avgDiff * 0.8) {
                factor *= 0.9;
            } else if (latestDiff > avgDiff * 1.2) {
                factor *= 1.1;
            }
        }

        return std::clamp(factor, 0.3, 0.7);
    }

    void updateDiffHistory(double currentDiff) {
        diffHistory.push_back(currentDiff);
        if (diffHistory.size() > Parameters::HISTORY_SIZE) {
            diffHistory.erase(diffHistory.begin());
        }
    }

    void updateInnerInterval(double avgDiff) {
        if (state.savedInnerInterval == 0.0) {
            state.savedInnerInterval = avgDiff;
        } else {
            double alpha = calculateAdaptiveRate();
            state.savedInnerInterval +=
                (avgDiff - state.savedInnerInterval) * alpha;
        }
    }

    void updateTrustRegionSize() {
        if (state.savedInnerInterval <
            std::numeric_limits<double>::infinity()) {
            state.innerHalfInterval =
                state.savedInnerInterval * calculateAdaptiveHalfInterval();
            state.v =
                state.savedInnerInterval / (state.c * calculateTRScaling());
            state.v = std::clamp(state.v, Parameters::MIN_V, Parameters::MAX_V);
        } else {
            state.innerHalfInterval = 0.0;
            state.v = Parameters::MIN_V;
        }
    }

    void updatePenaltyFunction(const std::vector<double>& nodeDuals) {
        double currentDiff = calculateAverageDifference(nodeDuals);
        updateDiffHistory(currentDiff);

        double avgDiff =
            std::accumulate(diffHistory.begin(), diffHistory.end(), 0.0) /
            diffHistory.size();

        updateInnerInterval(avgDiff);
        updateTrustRegionSize();
    }

    void updateBounds(const std::vector<double>& nodeDuals) {
        for (int i = 0; i < numConstrs; i++) {
            delta1[i] = nodeDuals[i] - state.v / 2;
            delta2[i] = nodeDuals[i] + state.v / 2;
        }
    }

    void updateCoefficients(const std::vector<double>& nodeDuals) {
        for (int i = 0; i < numConstrs; i++) {
            w[i]->setOBJ(std::max(0.0, nodeDuals[i] - state.v));
            zeta[i]->setOBJ(nodeDuals[i] + state.v);
        }
    }

    void updateVariables(const std::vector<double>& nodeDuals, double new_eps) {
        for (int i = 0; i < numConstrs; i++) {
            double safeDual =
                std::clamp(nodeDuals[i], -Parameters::MAX_DUAL_CHANGE,
                           Parameters::MAX_DUAL_CHANGE);

            w[i]->setUB(new_eps);
            zeta[i]->setUB(new_eps);

            w[i]->setOBJ(std::max(
                0.0, safeDual - std::clamp(state.v, 0.0, Parameters::MAX_V)));
            zeta[i]->setOBJ(safeDual +
                            std::clamp(state.v, 0.0, Parameters::MAX_V));
        }
    }

    void cleanup(BNBNode* node) {
        state.v = 0;
        for (auto& var : w) node->remove(var);
        for (auto& var : zeta) node->remove(var);
        state.TRstop = true;
        node->update();
        node->optimize();
    }

   public:
    explicit TrustRegion(int numConstrs) : numConstrs(numConstrs) {
        w.resize(numConstrs);
        zeta.resize(numConstrs);
        delta1.resize(numConstrs);
        delta2.resize(numConstrs);
        prevDuals.resize(numConstrs);
    }

    void setup(BNBNode* node, const std::vector<double>& nodeDuals) {
        state.v = std::accumulate(nodeDuals.begin(), nodeDuals.end(), 0.0) /
                  numConstrs;
        state.c = state.v / state.kappa;

        for (int i = 0; i < numConstrs; i++) {
            w[i] = node->addVar("w_" + std::to_string(i + 1),
                                VarType::Continuous, 0, state.epsilon1, -1.0);
            zeta[i] = node->addVar("zeta_" + std::to_string(i + 1),
                                   VarType::Continuous, 0, state.epsilon2, 1.0);

            auto constrs = node->getConstrs();
            node->chgCoeff(constrs[i]->index(), w[i]->index(), -1);
            node->chgCoeff(constrs[i]->index(), zeta[i]->index(), 1);
        }

        updateBounds(nodeDuals);
        updateCoefficients(nodeDuals);

        node->optimize();
        prevDuals = nodeDuals;
    }

    double iterate(BNBNode* node, std::vector<double>& nodeDuals,
                   double inner_obj, int stage, bool transition) {
        state.isInsideTrustRegion = true;
        state.TRstop = false;
        state.inner_obj = inner_obj;

        state.isInsideTrustRegion =
            std::all_of(nodeDuals.begin(), nodeDuals.end(),
                        [this, i = 0](double dual) mutable {
                            return dual >= delta1[i] && dual <= delta2[i++];
                        });

        double dualDiffSum = std::transform_reduce(
            nodeDuals.begin(), nodeDuals.end(), prevDuals.begin(), 0.0,
            std::plus<>(), [](double a, double b) { return std::abs(a - b); });

        if (state.isInsideTrustRegion && dualDiffSum > Parameters::EPSILON) {
            double avg_diff = dualDiffSum / numConstrs;
            state.v = avg_diff < state.v ? std::max(state.v * 0.9, avg_diff)
                                         : std::min(state.v * 1.1, avg_diff);

            double progress =
                (inner_obj - state.prev_inner_obj) /
                std::abs(state.prev_inner_obj + Parameters::EPSILON);
            adjustParameters(progress);

            double new_eps = std::clamp(state.v / state.c, Parameters::MIN_V,
                                        Parameters::MAX_EPSILON);
            updateVariables(nodeDuals, new_eps);
        }

        updateBounds(nodeDuals);

        if (state.v <= Parameters::MIN_V || stage == 4 || transition) {
            cleanup(node);
            return 0.0;
        }

        prevDuals = nodeDuals;
        state.prev_inner_obj = inner_obj;
        return state.v;
    }

    bool stop() const { return state.TRstop; }
};
