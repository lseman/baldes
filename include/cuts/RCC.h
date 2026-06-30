/**
 * @file RCC.h
 * @brief Defines Rounded Capacity Cut separation for vehicle routing.
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
    baldesCtrPtr        ctr;  // Gurobi constraint object

    // Cut pool management metadata
    int64_t separation_count = 0; // Times this cut was separated
    int64_t inclusion_count  = 0; // Times selected into active set
    int64_t creation_epoch   = 0; // B&B node epoch when created
    double  max_violation    = 0; // Peak violation observed
    double  avg_dual         = 0; // Running average dual magnitude
    double  dual_magnitude   = 0; // Current dual magnitude
};

class BNBNode;

class RCCManager {
public:
    RCCManager() = default;
    int cut_ctr  = 0;

    // Cut pool management configuration
    size_t  max_pool_size_   = 500;  // Max RCC cuts in pool
    int64_t current_epoch_   = 0;    // B&B node epoch counter
    double  age_decay_alpha_ = 0.95; // Exponential decay per epoch

    /**
     * @brief Mark cut as separated, updating metadata.
     */
    void markCutSeparated(RCCut &cut, double violation, double dual_mag) {
        cut.separation_count++;
        cut.dual_magnitude = dual_mag;
        cut.avg_dual       = (cut.avg_dual * (cut.separation_count - 1) + dual_mag) / cut.separation_count;
        if (violation > cut.max_violation) cut.max_violation = violation;
        if (cut.creation_epoch == 0) cut.creation_epoch = current_epoch_;
    }

    void markCutInclusion(RCCut &cut) { cut.inclusion_count++; }

    /**
     * @brief Compute selection score for probabilistic cut selection.
     * Score = dual_strength * age_penalty * inclusion_bonus
     */
    double computeSelectionScore(const RCCut &cut) const {
        double  strength        = std::log1p(cut.avg_dual) * std::log1p(cut.avg_dual);
        int64_t age             = current_epoch_ - cut.creation_epoch;
        double  age_penalty     = (age == 0) ? 1.0 : std::pow(age_decay_alpha_, static_cast<double>(age));
        double  inclusion_bonus = 1.0 + 0.1 * std::log1p(static_cast<double>(cut.inclusion_count));
        return strength * age_penalty * inclusion_bonus;
    }

    /**
     * @brief Enforce pool size by removing lowest-scoring cuts.
     */
    void enforcePoolSizeLimit() {
        if (cuts_.size() <= max_pool_size_) return;

        std::vector<std::pair<size_t, double>> scored;
        scored.reserve(cuts_.size());
        for (size_t i = 0; i < cuts_.size(); ++i) { scored.emplace_back(i, computeSelectionScore(cuts_[i])); }

        std::sort(scored.begin(), scored.end(), [](const auto &a, const auto &b) { return a.second < b.second; });

        size_t excess  = cuts_.size() - max_pool_size_;
        size_t removed = 0;
        for (auto &[idx, score] : scored) {
            if (removed >= excess) break;
            if (score > 0) {
                cuts_.erase(cuts_.begin() + static_cast<int>(idx));
                removed++;
            }
        }
    }

    /**
     * @brief Update active cuts with probabilistic scoring.
     * Returns indices of selected cuts sorted by score.
     */
    std::vector<size_t> selectActiveCuts(size_t max_active = 64) {
        std::vector<std::pair<size_t, double>> scored;
        scored.reserve(cuts_.size());
        for (size_t i = 0; i < cuts_.size(); ++i) {
            if (cuts_[i].dual_magnitude > 1e-6) { scored.emplace_back(i, computeSelectionScore(cuts_[i])); }
        }

        std::sort(scored.begin(), scored.end(), [](const auto &a, const auto &b) { return a.second > b.second; });

        if (scored.size() > max_active) scored.resize(max_active);

        std::vector<size_t> active;
        active.reserve(scored.size());
        for (auto &[idx, score] : scored) {
            active.push_back(idx);
            markCutInclusion(cuts_[idx]);
        }
        return active;
    }

    // define size as the size of the cuts vector
    int size() { return cuts_.size(); }

    std::vector<baldesCtrPtr> getbaldesCtrs() {
        std::vector<baldesCtrPtr> constraints;
        for (const auto &cut : cuts_) { constraints.push_back(cut.ctr); }
        return constraints;
    }

    std::shared_ptr<RCCManager> clone() {
        auto clone   = std::make_shared<RCCManager>();
        clone->cuts_ = cuts_; // First copy the vector structure

        // Clone each cut's constraint using the baldesCtr clone method
        for (auto &cut : clone->cuts_) {
            // Use the constraint's clone method instead of creating new one
            cut.ctr = nullptr;
        }
        return clone;
    }

    void setCutCtrs(std::vector<baldesCtrPtr> ctrs) {
        for (int i = 0; i < ctrs.size(); i++) { cuts_[i].ctr = ctrs[i]; }
    }

    std::vector<double> computeRCCCoefficients(const std::vector<uint16_t> &label) {
        std::vector<double> coeffs(cuts_.size(), 0);
        if (label.size() < 2 || cuts_.empty()) { return coeffs; }

        // Precompute the pairs of consecutive nodes in the label
        ankerl::unordered_dense::set<std::pair<int, int>> label_arcs;
        label_arcs.reserve(label.size() - 1);
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
    void addCut(const std::vector<RawArc> &arcs, int rhs, baldesCtrPtr ctr) {
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
        cuts_.erase(std::remove_if(cuts_.begin(), cuts_.end(), [&](const RCCut &c) { return c.ctr == cut.ctr; }),
                    cuts_.end());
    }
    ArcDuals computeDuals(std::vector<double> dualValues, BNBNode *node, double threshold = 1e-3);

private:
    std::vector<RCCut> cuts_; // Vector to hold all the RCCuts
    // std::mutex         mutex_; // Mutex to protect access to the cuts
};
