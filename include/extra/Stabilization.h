/**
 * @file Stabilization.h
 * @brief Defines the Stabilization class for dual stabilization in column
 * generation.
 *
 * This file implements the Stabilization class, which is responsible for
 * handling the stabilization process during column generation in large-scale
 * optimization problems, such as vehicle routing and resource-constrained
 * shortest path problems (RCSPP).
 *
 * The Stabilization class is crucial for improving the convergence of column
 * generation algorithms by stabilizing the dual values and avoiding large
 * oscillations in the dual space.
 *
 */

#pragma once

#include "Definitions.h"
#include "Pools.h"

#ifdef GUROBI
#include "gurobi_c++.h"
#include "gurobi_c.h"
#endif

#include <algorithm>  // For std::transform
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>  // For std::iota
#include <vector>

#include "MultiPointMgr.h"
#define NORM_TOLERANCE 1e-4

/**
 * @class Stabilization
 * @brief A class to handle stabilization in optimization problems.
 *
 * This class implements various methods to manage and compute stabilization
 * parameters and dual solutions in optimization problems.
 */
class Stabilization {
   public:
    double alpha;  // Current alpha parameter
    int t;         // Iteration counter

    double base_alpha;  // "global" alpha parameter
    double cur_alpha;   // alpha parameter during the current misprice sequence
    int nb_misprices =
        0;  // number of misprices during the current misprice sequence
    double pseudo_dual_bound;      // pseudo dual bound, may be non-valid
    double valid_dual_bound;       // valid dual bound
    DualSolution cur_stab_center;  // current stability center
    DualSolution stab_center_for_next_iteration;  // stability center for the
                                                  // next iteration

    bool stabilization_active = true;

    DualSolution phi_in;
    DualSolution phi_out;
    DualSolution prev_dual;
    bool misprice = false;

    DualSolution duals_in;
    DualSolution duals_out;
    DualSolution duals_sep;
    DualSolution duals_g;
    double beta;
    DualSolution rho;
    DualSolution master_dual;
    DualSolution smooth_dual_sol;

    double subgradient_norm = 0.0;
    DualSolution subgradient;
    std::vector<double> new_rows;

    double lag_gap = 0.0;
    double lag_gap_prev = -std::numeric_limits<double>::infinity();

    int numK = 10;

    int sizeDual;

    double lp_obj = 0.0;

    ReducedCostResult rc;

    MultiPointManager mp_manager;
    std::vector<double> stab_constraint_values;

    bool cut_added = false;

    void update_stabilization_after_misprice() {
        nb_misprices++;
        alpha = _misprice_schedule(nb_misprices, base_alpha);
        cur_alpha = alpha;
        beta = 0.0;
    }

    void update_stabilization_after_iter(const DualSolution &new_center) {
        if (!stab_center_for_next_iteration.empty()) {
            cur_stab_center = stab_center_for_next_iteration;
            stab_center_for_next_iteration.clear();
        }
    }

    bool update_stabilization_after_master_optim(
        const DualSolution &new_center) {
        nb_misprices = 0;
        cur_alpha = 0;

        if (cur_stab_center.empty()) {
            cur_stab_center = new_center;
            return false;
        }
        cur_alpha = base_alpha;
        return cur_alpha > 0;
    }

    void reset_misprices() { nb_misprices = 0; }

    double _misprice_schedule(int nb_misprices, double base_alpha) {
        double alpha = 1.0 - (nb_misprices + 1) * (1 - base_alpha);
        if (nb_misprices > 10 || alpha <= 1e-3) {
            alpha = 0.0;  // Deactivate stabilization
        }
        duals_in = duals_sep;
        return alpha;
    }

    Stabilization(double base_alpha, DualSolution &mast_dual_sol)
        : alpha(base_alpha),
          t(0),
          base_alpha(base_alpha),
          cur_alpha(base_alpha),
          nb_misprices(0),
          cur_stab_center(mast_dual_sol),
          mp_manager(mast_dual_sol.size()) {
        pseudo_dual_bound = std::numeric_limits<double>::infinity();
        valid_dual_bound = std::numeric_limits<double>::infinity();
        beta = 0.0;
        sizeDual = mast_dual_sol.size();
    }

    DualSolution getStabDualSol(const DualSolution &input_duals) {
        std::vector<double> master_dual;
        master_dual.assign(input_duals.begin(), input_duals.begin() + sizeDual);
        if (cur_stab_center.empty()) {
            return master_dual;
        }
        DualSolution stab_dual_sol(master_dual.size());
        for (size_t i = 0; i < master_dual.size(); ++i) {
            stab_dual_sol[i] =
                std::max(0.0, cur_alpha * cur_stab_center[i] +
                                  (1 - cur_alpha) * master_dual[i]);
        }
        smooth_dual_sol = stab_dual_sol;
        return stab_dual_sol;
    }

    inline double norm(const std::vector<double> &vector) {
        double res = 0;
        for (int i = 0; i < vector.size(); ++i) {
            res += vector[i] * vector[i];
        }
        return std::sqrt(res + 1e-6);
    }

    inline double norm(const std::vector<double> &vector_1,
                       const std::vector<double> &vector_2) {
        double res = 0.0;
        for (auto i = 0; i < vector_1.size(); ++i) {
            res += (vector_2[i] - vector_1[i]) * (vector_2[i] - vector_1[i]);
        }
        return std::sqrt(res + 1e-8);
    }

    DualSolution getStabDualSolAdvanced(const DualSolution &input_duals) {
        constexpr double EPSILON = 1e-12;

        // Use only the first sizeDual elements from input_duals.
        DualSolution nodeDuals(input_duals.begin(),
                               input_duals.begin() + sizeDual);

        // If stabilization center or subgradient are not available, return the
        // input.
        if (cur_stab_center.empty() || subgradient.empty()) {
            return input_duals;
        }

        const size_t n = nodeDuals.size();

        // Step 1: pi_tilde = cur_stab_center + (1 - cur_alpha)*(nodeDuals -
        // cur_stab_center)
        DualSolution pi_tilde(n);
        for (size_t i = 0; i < n; ++i) {
            pi_tilde[i] =
                cur_stab_center[i] +
                (1.0 - cur_alpha) * (nodeDuals[i] - cur_stab_center[i]);
        }

        // Step 2: Compute norm_in_out = ||nodeDuals - cur_stab_center||
        double norm_in_out =
            std::sqrt(std::inner_product(nodeDuals.begin(), nodeDuals.end(),
                                         cur_stab_center.begin(), 0.0,
                                         std::plus<double>(),
                                         [](double a, double b) {
                                             double d = a - b;
                                             return d * d;
                                         }) +
                      EPSILON);

        // Compute pi_g = cur_stab_center +
        // (subgradient/subgradient_norm)*norm_in_out
        DualSolution pi_g(n);
        for (size_t i = 0; i < n; ++i) {
            pi_g[i] = cur_stab_center[i] +
                      (subgradient[i] / subgradient_norm) * norm_in_out;
        }

        // Step 3: rho = beta*pi_g + (1 - beta)*nodeDuals
        DualSolution rho(n);
        for (size_t i = 0; i < n; ++i) {
            rho[i] = beta * pi_g[i] + (1.0 - beta) * nodeDuals[i];
        }

        // Compute norm_tilde_in = ||pi_tilde - cur_stab_center||
        double norm_tilde_in =
            std::sqrt(std::inner_product(pi_tilde.begin(), pi_tilde.end(),
                                         cur_stab_center.begin(), 0.0,
                                         std::plus<double>(),
                                         [](double a, double b) {
                                             double d = a - b;
                                             return d * d;
                                         }) +
                      EPSILON);

        // Compute norm_rho_in = ||rho - cur_stab_center||
        double norm_rho_in = std::sqrt(
            std::inner_product(rho.begin(), rho.end(), cur_stab_center.begin(),
                               0.0, std::plus<double>(),
                               [](double a, double b) {
                                   double d = a - b;
                                   return d * d;
                               }) +
            EPSILON);

        // Step 4: new_duals = cur_stab_center + (norm_tilde_in / norm_rho_in) *
        // (rho - cur_stab_center)
        DualSolution new_duals(n);
        for (size_t i = 0; i < n; ++i) {
            new_duals[i] =
                cur_stab_center[i] +
                (norm_tilde_in / norm_rho_in) * (rho[i] - cur_stab_center[i]);
            new_duals[i] = std::max(
                0.0, new_duals[i]);  // Project onto the positive orthant.
        }

        // Store the new dual solutions in smoothing variables.
        smooth_dual_sol = new_duals;
        duals_sep = new_duals;
        return new_duals;
    }

    static constexpr double EPSILON = 1e-12;

    bool dynamic_alpha_schedule(const ModelData &dados) {
        constexpr double DOT_TOLERANCE = 1e-3;
        const size_t n = cur_stab_center.size();

        // Compute relative distance: ||smooth_dual_sol - cur_stab_center|| /
        // |lp_obj|
        double rel_distance = norm(smooth_dual_sol, cur_stab_center) /
                              (std::abs(lp_obj) + EPSILON);
        if (rel_distance < NORM_TOLERANCE) {
            alpha = 0.0;
            return false;
        }

        // Compute the difference vector (direction = smooth_dual_sol -
        // cur_stab_center)
        std::vector<double> direction(n);
        for (size_t i = 0; i < n; ++i) {
            direction[i] = smooth_dual_sol[i] - cur_stab_center[i];
        }

        // Compute norm of direction vector using inner_product
        double dir_norm =
            std::sqrt(std::inner_product(direction.begin(), direction.end(),
                                         direction.begin(), 0.0) +
                      EPSILON);
        if (dir_norm < EPSILON || subgradient_norm < EPSILON) {
            return false;
        }

        // Normalize the direction and the subgradient vectors.
        std::vector<double> normalized_direction(n);
        std::vector<double> normalized_subgradient(n);
        for (size_t i = 0; i < n; ++i) {
            normalized_direction[i] = direction[i] / dir_norm;
            normalized_subgradient[i] = subgradient[i] / subgradient_norm;
        }

        // Compute the cosine of the angle between normalized_direction and
        // normalized_subgradient.
        double cos_angle = std::inner_product(
            normalized_direction.begin(), normalized_direction.end(),
            normalized_subgradient.begin(), 0.0);

        // If the cosine of the angle is very close to zero, then the vectors
        // are nearly orthogonal.
        return cos_angle < DOT_TOLERANCE;
    }

    void update_subgradient(const ModelData &dados,
                            const DualSolution &nodeDuals,
                            const std::vector<Label *> &best_pricing_cols) {
        size_t number_of_rows = nodeDuals.size();
        new_rows.assign(number_of_rows, 0.0);

        auto cols_to_check =
            std::min(numK, static_cast<int>(best_pricing_cols.size()));
        for (size_t i = 0; i < cols_to_check; ++i) {
            const auto &col = best_pricing_cols[i];
            for (const auto &node : col->nodes_covered) {
                if (node > 0 && node != N_SIZE - 1) {
                    new_rows[node - 1] += 1.0;
                }
            }
        }

        subgradient.assign(number_of_rows, 0.0);
        for (size_t i = 0; i < number_of_rows; ++i) {
            // If the relation is '<', treat it as a "≤" constraint.
            if (dados.sense[i] == '<') {
                subgradient[i] = new_rows[i] - dados.b[i];
            } else {
                // For '>' (i.e., "≥") or '=', compute as before.
                subgradient[i] = dados.b[i] - new_rows[i];
            }
        }

        subgradient_norm = norm(subgradient);
    }

    void set_pseudo_dual_bound(double bound) { pseudo_dual_bound = bound; }

    int no_progress_count = 0;

    void setObj(double obj) { lp_obj = obj; }

    double lp_obj_prev = 0.0;
    const int NO_PROGRESS_THRESHOLD = 50;

    void update_stabilization_after_pricing_optim(
        const ModelData &dados, const DualSolution &input_duals,
        const double &lag_gap, std::vector<Label *> best_pricing_cols) {
        std::vector<double> nodeDuals(input_duals.begin(),
                                      input_duals.begin() + sizeDual);

        mp_manager.updatePool(nodeDuals, -lag_gap);
        DualSolution historical_avg = mp_manager.getWeightedSolution();

        // check for progress within a threshold
        if (std::abs(lp_obj - lp_obj_prev) < 1e-3) {
            no_progress_count++;
        } else {
            no_progress_count = 0;
        }

        if (!historical_avg.empty() && nb_misprices == 0) {
            // Dynamic weighting based on optimization progress
            double pw = std::min(
                0.9, std::max(0.1, std::abs(lag_gap) /
                                       (std::abs(lag_gap_prev) + 1e-10)));

            // fmt::print("Progress weight: {}\n", pw);

            // Scale historical weight based on
            // pool quality
            // double historical_weight = (1.0
            // - progress_weight);

            // Blend solutions with adaptive
            // weights
            for (size_t i = 0; i < nodeDuals.size(); ++i) {
                nodeDuals[i] = pw * nodeDuals[i] + (1 - pw) * historical_avg[i];
            }
        }

        if (nb_misprices == 0) {
            update_subgradient(dados, nodeDuals, best_pricing_cols);
            bool should_increase = dynamic_alpha_schedule(dados);
            constexpr double ALPHA_FACTOR = 0.1;

            alpha = should_increase
                        ? std::min(0.99, alpha + (1.0 - alpha) * ALPHA_FACTOR)
                        : std::max(0.0, alpha / 1.1);

            if (!std::isnan(alpha) && !std::isinf(alpha)) {
                base_alpha = alpha;
            }
        }

        // if (no_progress_count > NO_PROGRESS_THRESHOLD) {
        //     alpha = 0.0;
        //     base_alpha = 0.0;
        // }

        // if (lag_gap <= lag_gap_prev || cut_added) {
        stab_center_for_next_iteration = smooth_dual_sol;
        cut_added = false;
        // } else {
        // stab_center_for_next_iteration = cur_stab_center;
        // }

        lag_gap_prev = lag_gap;

        if (std::isnan(alpha) || std::isinf(alpha)) {
            stabilization_active = false;
            cleanup();
        }
    }

    bool shouldExit() { return cur_alpha < 1e-3; }

    void cleanup() {
        mp_manager.clear();
        stab_constraint_values.clear();
        smooth_dual_sol.clear();
        subgradient.clear();
        duals_sep.clear();
        beta = 0.0;
        alpha = base_alpha;
    }

    bool ipm_active = false;
    void define_smooth_dual_sol(const DualSolution &nodeDuals) {
        smooth_dual_sol.assign(nodeDuals.begin(), nodeDuals.begin() + sizeDual);
        ipm_active = true;
    }

    void updateNumK(int numK) { this->numK = numK; }

    void clearAlpha() { alpha = 0.0; }
};
