/**
 * @file RCC.h
 * @brief Definitions for generating and separating Rounded Capacity Cuts (RCC) in the context of vehicle routing
 * problems.
 *
 * This header file contains the structure and function definitions required for separating Rounded Capacity Cuts (RCCs)
 * using Gurobi's optimization model. RCC separation is an important aspect of optimization algorithms, particularly
 * in vehicle routing problems where capacity constraints must be enforced.
 *
 * The file leverages Gurobi for optimization and constraint management.
 *
 */

#pragma once

#include "Definitions.h"

#include "Arc.h"
#include "Hashes.h"

#include <cmath>
#include <iostream>
#include <numeric>
#include <set>
#include <vector>

#include "Dual.h"

#include <iostream>

#include "ankerl/unordered_dense.h"
#include "miphandler/MIPHandler.h"

// Define the RawArc struct
struct RCCut {
    std::vector<RawArc> arcs; // Stores the arcs involved in the cut
    int                 rhs;  // Right-hand side value of the constraint
    Constraint         *ctr;  // Gurobi constraint object
};

class BNBNode;

class RCCManager {
public:
    RCCManager() = default;
    int cut_ctr  = 0;

    // define size as the size of the cuts vector
    int size() { return cuts_.size(); }

    std::vector<Constraint *> getConstraints() {
        std::vector<Constraint *> constraints;
        for (const auto &cut : cuts_) { constraints.push_back(cut.ctr); }
        return constraints;
    }

    std::vector<double> computeRCCCoefficients(const std::vector<int> &label) {
        std::vector<double> coeffs(cuts_.size(), 0);

        // Precompute the pairs of consecutive nodes in the label
        ankerl::unordered_dense::set<std::pair<int, int>> label_arcs;
        for (int j = 0; j < label.size() - 1; j++) { label_arcs.insert({label[j], label[j + 1]}); }

        // Iterate over cuts
        for (int i = 0, n_cuts = cuts_.size(); i < n_cuts; i++) {
            int coeff = 0;

            // Iterate over arcs in the cut and check if they exist in the precomputed label arcs
            for (const auto &arc : cuts_[i].arcs) { coeff += label_arcs.count({arc.from, arc.to}); }

            coeffs[i] = coeff;
        }

        return coeffs;
    }

    // Method to add a new cut
    void addCut(const std::vector<RawArc> &arcs, int rhs, Constraint *ctr) {
        // std::lock_guard<std::mutex> lock(mutex_);
        cuts_.emplace_back(RCCut{arcs, rhs, ctr});
        cut_ctr++;
    }

    // Retrieve all the cuts for further processing
    const std::vector<RCCut> &getCuts() const { return cuts_; }

    // Optionally, clear cuts if needed
    void clearCuts() {
        // std::lock_guard<std::mutex> lock(mutex_);
        cuts_.clear();
    }

    // Compute the dual values for each arc by summing the duals of cuts passing through the arc
    ArcDuals computeDuals(BNBNode *model, double threshold = 1e-3);

    // define remove cut method
    void removeCut(RCCut &cut) {
        // std::lock_guard<std::mutex> lock(mutex_);
        cuts_.erase(std::remove_if(cuts_.begin(), cuts_.end(),
                                   [&](const RCCut &c) { return c.ctr == cut.ctr; }),
                    cuts_.end());
    }
    ArcDuals computeDuals(std::vector<double> dualValues, double threshold = 1e-3) {
        ArcDuals arcDuals;
        // First pass: Compute dual values and store them
        for (int i = 0; i < cuts_.size(); ++i) {
            const auto &cut       = cuts_[i];
            double      index     = cut.ctr->index();
            auto        dualValue = -dualValues[index];

            if (std::abs(dualValue) < 1e-3) { continue; }

            // Sum the dual values for all arcs in this cut
            for (const auto &arc : cut.arcs) {
                arcDuals.setOrIncrementDual(arc, dualValue); // Update arc duals
            }
        }

        // Second pass: Remove cuts with dual values near zero
        /*
        cuts_.erase(std::remove_if(cuts_.begin(), cuts_.end(),
                                   [&](const RCCut &cut, size_t i = 0) mutable {
                                       if (std::abs(dualValues[i]) < threshold) {
                                           model->remove(cut.ctr); // Remove the constraint from the model
                                           return true;            // Mark this cut for removal
                                       }
                                       ++i;
                                       return false; // Keep this cut
                                   }),
                    cuts_.end());
    */
        return arcDuals;
    }

private:
    std::vector<RCCut> cuts_; // Vector to hold all the RCCuts
    // std::mutex         mutex_; // Mutex to protect access to the cuts
};
