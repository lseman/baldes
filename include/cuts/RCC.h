/**
 * @file RCC.h
 * @brief Definitions for generating and separating Rounded Capacity Cuts (RCC) in the context of vehicle routing
 * problems.
 *
 * This header file contains the structure and function definitions required for separating Rounded Capacity Cuts (RCCs)
 * using Gurobi's optimization model. RCC separation is an important aspect of optimization algorithms, particularly
 * in vehicle routing problems where capacity constraints must be enforced.
 *
 * Key components of the file include:
 * - `separate_Rounded_Capacity_cuts`: A function that identifies and separates RCCs by solving the relaxed Gurobi model
 *   and searching for violated constraints in the solution space. The function generates multiple RCC solutions.
 *
 * The file leverages Gurobi for optimization and constraint management.
 *
 * @note Several parts of the function rely on setting up and solving a Gurobi optimization model to identify capacity
 * violations and generate RCCs.
 */

#pragma once

#include "Definitions.h"

#include "Arc.h"
#include "Hashes.h"

#include <cmath>
#include <iostream>
#include <numeric>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <iostream>
#include <unordered_map>

// Define the RCCArc struct
struct RCCut {
    std::vector<RCCArc> arcs; // Stores the arcs involved in the cut
    int                 rhs;  // Right-hand side value of the constraint
    GRBConstr           ctr;  // Gurobi constraint object
};

// Structure to store and manage dual values for arcs
class ArcDuals {
public:
    // Add or update the dual value for an arc
    void setDual(const RCCArc &arc, double dualValue) { arcDuals_[arc] = dualValue; }

    // Retrieve the dual value for an arc; returns 0 if the arc does not have a dual
    double getDual(int i, int j) const {
        RCCArc arc(i, j);
        auto   it = arcDuals_.find(arc);
        if (it != arcDuals_.end()) {
            return it->second; // Return the dual if found
        }
        return 0.0; // Default to 0 if not found
    }

    double getDual(RCCArc arc) const {
        auto it = arcDuals_.find(arc);
        if (it != arcDuals_.end()) {
            return it->second; // Return the dual if found
        }
        return 0.0; // Default to 0 if not found
    }

    void setOrIncrementDual(const RCCArc &arc, double dualValue) {
        auto it = arcDuals_.find(arc);
        if (it != arcDuals_.end()) {
            it->second += dualValue; // Increment the dual if the arc already has a dual
        } else {
            arcDuals_[arc] = dualValue; // Set the dual if the arc does not have a dual
        }
    }

private:
    std::unordered_map<RCCArc, double, RCCArcHash> arcDuals_; // Map for storing arc duals
};

class RCCManager {
public:
    RCCManager() = default;
    int cut_ctr  = 0;

    // define size as the size of the cuts vector
    int size() { return cuts_.size(); }

    std::vector<GRBConstr> getConstraints() {
        std::vector<GRBConstr> constraints;
        for (const auto &cut : cuts_) { constraints.push_back(cut.ctr); }
        return constraints;
    }

    std::vector<double> computeRCCCoefficients(const std::vector<int> &label) {
        std::vector<double> coeffs(cuts_.size(), 0);

        // Precompute the pairs of consecutive nodes in the label
        std::unordered_set<std::pair<int, int>> label_arcs;
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
    void addCut(const std::vector<RCCArc> &arcs, int rhs, GRBConstr &ctr) {
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
    ArcDuals computeDuals(GRBModel *model, double threshold = 1e-3) {
        ArcDuals arcDuals;
        // First pass: Compute dual values and store them
        for (int i = 0; i < cuts_.size(); ++i) {
            const auto &cut       = cuts_[i];
            double      dualValue = cut.ctr.get(GRB_DoubleAttr_Pi);
            // fmt::print("Dual value for cut {}: {}\n", i, dualValue);

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
