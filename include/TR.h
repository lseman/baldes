#pragma once

#include "bnb/Node.h"
#include "config.h"
#include <iostream>
#include <numeric>

class TrustRegion {
private:
    static constexpr double EPSILON        = 1e-12;
    static constexpr double INIT_V         = 100.0;
    static constexpr double MIN_V          = 10.0;
    static constexpr double CURVATURE_RATE = 0.1; // Similar to StabilFuncCurvatureAdvanceRate

public:
    bool                      isInsideTrustRegion;
    int                       numConstrs;
    std::vector<baldesVarPtr> w;         // Inner piece variables (w_i)
    std::vector<baldesVarPtr> zeta;      // Outer piece variables (zeta_i)
    std::vector<double>       delta1;    // Lower bounds
    std::vector<double>       delta2;    // Upper bounds
    std::vector<double>       prevDuals; // Previous dual values

    double v;      // Trust region size
    double c;      // Scaling factor
    double kappa;  // Growth control parameter
    bool   TRstop; // Trust region termination flag

    // Variables for penalty function
    double savedInnerInterval;
    double innerHalfInterval;
    double averDifference;
    double epsilon1; // Inner bound
    double epsilon2; // Outer bound
    double prev_inner_obj;

    TrustRegion(int numConstrs)
        : numConstrs(numConstrs), v(INIT_V), kappa(0.9), TRstop(false), isInsideTrustRegion(false),
          savedInnerInterval(0.0), innerHalfInterval(0.0), averDifference(0.0), epsilon1(10.0), epsilon2(10.0) {

        // Initialize vectors
        w.resize(numConstrs);
        zeta.resize(numConstrs);
        delta1.resize(numConstrs);
        delta2.resize(numConstrs);
        prevDuals.resize(numConstrs);
    }

    void adjustParameters(double progress) {
        // Add dynamic kappa adjustment
        if (progress > 0) {
            // If making progress, be more aggressive
            kappa = std::max(0.5, kappa * 0.95);
        } else {
            // If not making progress, be more conservative
            kappa = std::min(0.95, kappa * 1.05);
        }

        // Update c based on new kappa
        c = v / kappa;
    }

    void setup(BNBNode *node, const std::vector<double> &nodeDuals) {
        // Calculate initial center point
        v = std::accumulate(nodeDuals.begin(), nodeDuals.end(), 0.0) / numConstrs;
        c = v / kappa;

        // Create variables for 3-piece function
        for (int i = 0; i < numConstrs; i++) {
            // Create w_i variables (inner piece)
            std::string w_name = "w_" + std::to_string(i + 1);
            w[i]               = node->addVar(w_name, VarType::Continuous, 0, epsilon1, -1.0);

            // Create zeta_i variables (outer piece)
            std::string zeta_name = "zeta_" + std::to_string(i + 1);
            zeta[i]               = node->addVar(zeta_name, VarType::Continuous, 0, epsilon2, 1.0);

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

        fmt::print("Starting trust region with v = {}\n", v);
    }

    void updatePenaltyFunction(const std::vector<double> &nodeDuals) {
        // Calculate average difference
        double differenceSum      = 0.0;
        int    numStabConstraints = 0;

        for (size_t i = 0; i < numConstrs; i++) {
            double value = std::abs(nodeDuals[i] - prevDuals[i]);
            differenceSum += value;
            numStabConstraints++;
        }

        averDifference = numStabConstraints > 0 ? differenceSum / numStabConstraints : 0.0;

        // Three-piece penalty function update
        if (savedInnerInterval == 0.0) {
            savedInnerInterval = averDifference;
        } else {
            savedInnerInterval = savedInnerInterval + (averDifference - savedInnerInterval) * CURVATURE_RATE;
        }

        if (savedInnerInterval < std::numeric_limits<double>::infinity()) {
            innerHalfInterval = savedInnerInterval * 0.5;
            v                 = savedInnerInterval / c;
        } else {
            innerHalfInterval = 0.0;
            v                 = 0.0;
        }
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

    double iterate(BNBNode *node, std::vector<double> &nodeDuals, double inner_obj, int stage) {
        isInsideTrustRegion = true;
        TRstop              = false;

        // Check if solution is inside trust region
        for (int i = 0; i < numConstrs; i++) {
            if (nodeDuals[i] < delta1[i] || nodeDuals[i] > delta2[i]) {
                isInsideTrustRegion = false;
                break;
            }
        }

        double dualDiffSum = 0.0;
        for (int i = 0; i < numConstrs; i++) { dualDiffSum += std::abs(nodeDuals[i] - prevDuals[i]); }

        if (isInsideTrustRegion && dualDiffSum > EPSILON) {
            double avg_diff = dualDiffSum / numConstrs;

            // More conservative v updates
            if (avg_diff < v) {
                v = std::max(v * 0.9, avg_diff);
            } else {
                v = std::min(v * 1.1, avg_diff);
            }

            // Adjust kappa based on progress
            double progress = (inner_obj - prev_inner_obj) / std::abs(prev_inner_obj + EPSILON);
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

        prevDuals      = nodeDuals;
        prev_inner_obj = inner_obj;
        return v;
    }

    double safeguardValue(double value, double min_val, double max_val) {
        return std::max(min_val, std::min(max_val, value));
    }

    void updateVariables(const std::vector<double> &nodeDuals, double new_eps) {
        for (int i = 0; i < numConstrs; i++) {
            // Safeguard the dual values
            double safeDual = safeguardValue(nodeDuals[i], -MAX_DUAL_CHANGE, MAX_DUAL_CHANGE);

            // Update bounds with safeguards
            w[i]->setUB(new_eps);
            zeta[i]->setUB(new_eps);

            // Update coefficients with safeguards
            w[i]->setOBJ(std::max(0.0, safeDual - safeguardValue(v, 0.0, MAX_V)));
            zeta[i]->setOBJ(safeDual + safeguardValue(v, 0.0, MAX_V));
        }
    }

    bool stop() { return TRstop; }

private:
    static constexpr double MAX_V           = 1000.0;
    static constexpr double MAX_DUAL_CHANGE = 1000.0;
    static constexpr double MAX_EPSILON     = 1000.0;

    void cleanup(BNBNode *node) {
        v = 0;
        for (size_t i = 0; i < w.size(); i++) {
            node->remove(w[i]);
            node->remove(zeta[i]);
        }
        TRstop = true;
        node->update();
        node->optimize();
    }
};
