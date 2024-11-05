#pragma once

#include "bnb/Node.h"
#include "config.h"
#include <iostream>
#include <utility>

class TrustRegion {
public:
    bool                    isInsideTrustRegion;
    int                     numConstrs;
    std::vector<baldesVarPtr > w;
    std::vector<baldesVarPtr > zeta;
    std::vector<double>     delta1;
    std::vector<double>     delta2;
    int                     epsilon1 = 10;
    int                     epsilon2 = 10;
    double                  v        = 100;
    bool                    TRstop   = false;
    std::vector<double>     prevDuals;
    double                  c = 0;

    TrustRegion(int numConstrs) : numConstrs(numConstrs) {}

    bool stop() { return TRstop; }

    void setup(BNBNode *node, std::vector<double> nodeDuals) {

        w.resize(numConstrs);
        zeta.resize(numConstrs);
        v = std::accumulate(nodeDuals.begin(), nodeDuals.end(), 0.0) / numConstrs;
        c = v / 100;
        std::vector<int> tr_vars_idx;
        // Create w_i and zeta_i variables for each constraint
        for (int i = 0; i < numConstrs; i++) {
            auto w_name    = std::string("w_") + std::to_string(i + 1);
            auto zeta_name = std::string("zeta_") + std::to_string(i + 1);
            w[i]           = node->addVar(w_name, VarType::Continuous, 0, epsilon1, -1.0);
            zeta[i]        = node->addVar(zeta_name, VarType::Continuous, 0, epsilon2, 1.0);

            tr_vars_idx.push_back(w[i]->index());
            tr_vars_idx.push_back(zeta[i]->index());
        }

        auto constrs = node->getConstrs();
        for (int i = 0; i < numConstrs; i++) {
            node->chgCoeff(constrs[i]->index(), w[i]->index(), -1);
            node->chgCoeff(constrs[i]->index(), zeta[i]->index(), 1);
        }
        bool isInsideTrustRegion = false;

        for (auto dual : nodeDuals) {
            delta1.push_back(dual - v);
            delta2.push_back(dual + v);
        }
        for (int i = 0; i < numConstrs; i++) {
            // Update coefficients for w[i] and zeta[i] in the objective function
            w[i]->setOBJ(std::max(0.0, nodeDuals[i] - v)); // Set w[i]'s coefficient to -delta1
            zeta[i]->setOBJ(nodeDuals[i] + v);             // Set zeta[i]'s coefficient to delta2
            w[i]->setUB(epsilon1);
            zeta[i]->setUB(epsilon2);
        }
        fmt::print("Starting trust region with v = {}\n", v);
        node->optimize();
        prevDuals.assign(nodeDuals.size(), 0.0);
    }

    double update_v(std::vector<double> &nodeDuals) {

        double dualDiffSum = 0.0;
        for (int i = 0; i < numConstrs; i++) { dualDiffSum += std::abs(nodeDuals[i] - prevDuals[i]); }

        double avgDualDiff = dualDiffSum / numConstrs;
        prevDuals          = nodeDuals;
        return avgDualDiff;
    }
    double iterate(BNBNode *node, std::vector<double> nodeDuals, double inner_obj, int stage) {
        isInsideTrustRegion = true;
        TRstop              = false;
        auto kappa          = 10;

        for (int i = 0; i < numConstrs; i++) {
            if (nodeDuals[i] < delta1[i] || nodeDuals[i] > delta2[i]) { isInsideTrustRegion = false; }
        }
        if (isInsideTrustRegion) {
            // v -= 1;
            v            = update_v(nodeDuals);
            auto new_eps = v / c;

            if (v >= 80) {
                epsilon1 += 100;
                epsilon2 += 100;

            } else {
                epsilon1 += 500;
                epsilon2 += 500;
            }
            for (int i = 0; i < numConstrs; i++) {
                // Update coefficients for w[i] and zeta[i] in the objective function
                w[i]->setOBJ(nodeDuals[i]);        // Set w[i]'s coefficient to -delta1
                zeta[i]->setOBJ(nodeDuals[i] + v); // Set zeta[i]'s coefficient to delta2
                w[i]->setUB(epsilon1);
                zeta[i]->setUB(epsilon2);
            }
        }
        for (int i = 0; i < numConstrs; i++) {
            delta1[i] = nodeDuals[i] - v / 2;
            delta2[i] = nodeDuals[i] + v / 2;
        }
        if (v <= 5) {
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
