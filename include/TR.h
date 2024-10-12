#pragma once

#include "bnb/Node.h"
#include "config.h"
#include <iostream>
#include <utility>

class TrustRegion {
public:
    bool                isInsideTrustRegion;
    int                 numConstrs;
    std::vector<GRBVar> w;
    std::vector<GRBVar> zeta;
    std::vector<double> delta1;
    std::vector<double> delta2;
    int                 epsilon1 = 10;
    int                 epsilon2 = 10;
    double              v        = 100;
    bool                TRstop   = false;

    TrustRegion(int numConstrs) : numConstrs(numConstrs) {}

    bool stop() { return TRstop; }

    void setup(BNBNode *node, std::vector<double> nodeDuals) {

        w.resize(numConstrs);
        zeta.resize(numConstrs);

        std::vector<int> tr_vars_idx;
        // Create w_i and zeta_i variables for each constraint
        for (int i = 0; i < numConstrs; i++) {
            w[i]    = node->addVar(0, epsilon1, -1.0, GRB_CONTINUOUS, "w_" + std::to_string(i + 1));
            zeta[i] = node->addVar(0, epsilon2, 1.0, GRB_CONTINUOUS, "zeta_" + std::to_string(i + 1));
            tr_vars_idx.push_back(w[i].index());
            tr_vars_idx.push_back(zeta[i].index());
        }

        const GRBConstr *constrs = node->getConstrs();
        for (int i = 0; i < numConstrs; i++) {
            node->chgCoeff(constrs[i], w[i], -1);
            node->chgCoeff(constrs[i], zeta[i], 1);
        }
        bool isInsideTrustRegion = false;

        for (auto dual : nodeDuals) {
            delta1.push_back(dual - v);
            delta2.push_back(dual + v);
        }
        for (int i = 0; i < numConstrs; i++) {
            // Update coefficients for w[i] and zeta[i] in the objective function
            w[i].set(GRB_DoubleAttr_Obj, std::max(0.0, nodeDuals[i] - v)); // Set w[i]'s coefficient to -delta1
            zeta[i].set(GRB_DoubleAttr_Obj, nodeDuals[i] + v);             // Set zeta[i]'s coefficient to delta2
            w[i].set(GRB_DoubleAttr_UB, epsilon1);
            zeta[i].set(GRB_DoubleAttr_UB, epsilon2);
        }
        fmt::print("Starting trust region with v = {}\n", v);
        node->optimize();
    }

    int iterate(BNBNode *node, std::vector<double> nodeDuals, double inner_obj) {
        isInsideTrustRegion = true;
        TRstop              = false;

        for (int i = 0; i < numConstrs; i++) {
            if (nodeDuals[i] < delta1[i] || nodeDuals[i] > delta2[i]) { isInsideTrustRegion = false; }
        }
        // if (isInsideTrustRegion) { fmt::print("Fall inside trust region\n"); }
        if (isInsideTrustRegion) {
            v -= 1;
            // print v
            // fmt::print("Reducing v to {}\n", v);
            if (v >= 80) {
                epsilon1 += 100;
                epsilon2 += 100;

            } else {
                epsilon1 += 500;
                epsilon2 += 500;
            }
            for (int i = 0; i < numConstrs; i++) {
                // Update coefficients for w[i] and zeta[i] in the objective function
                w[i].set(GRB_DoubleAttr_Obj, nodeDuals[i]);        // Set w[i]'s coefficient to -delta1
                zeta[i].set(GRB_DoubleAttr_Obj, nodeDuals[i] + v); // Set zeta[i]'s coefficient to delta2
                w[i].set(GRB_DoubleAttr_UB, epsilon1);
                zeta[i].set(GRB_DoubleAttr_UB, epsilon2);
            }
        }
        for (int i = 0; i < numConstrs; i++) {
            delta1[i] = nodeDuals[i] - v;
            delta2[i] = nodeDuals[i] + v;
        }
        if (v <= 25 || inner_obj > -10) {
            v = 0;
            for (int i = 0; i < w.size(); i++) {
                node->remove(w[i]);
                node->remove(zeta[i]);
            }
            TRstop = true;
            node->update();
            node->optimize();
        }
        return v;
    }
};