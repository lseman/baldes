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

    int numK = 8;

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

    inline std::vector<double> mult(const std::vector<double> &v,
                                    const std::vector<double> &w) {
        std::vector<double> res(v.size(), 0.0);
        for (size_t i = 0; i < v.size(); ++i) {
            res[i] = v[i] * w[i];
        }
        return res;
    }

    inline std::vector<double> sub(const std::vector<double> &v,
                                   const std::vector<double> &w) {
        std::vector<double> res(v.size(), 0.0);
        for (size_t i = 0; i < v.size(); ++i) {
            res[i] = v[i] - w[i];
        }
        return res;
    }

    DualSolution getStabDualSolAdvanced(const DualSolution &input_duals) {
        constexpr double EPSILON = 1e-12;

        DualSolution nodeDuals(input_duals.begin(),
                               input_duals.begin() + sizeDual);

        if (cur_stab_center.empty() || subgradient.empty()) {
            return getStabDualSol(input_duals);
        }

        const size_t n = nodeDuals.size();

        // Step 1: π˜ = πin + (1−α)(πout − πin)
        DualSolution pi_tilde(n);
        for (size_t i = 0; i < n; ++i) {
            pi_tilde[i] =
                cur_stab_center[i] +
                (1.0 - cur_alpha) * (nodeDuals[i] - cur_stab_center[i]);
        }

        // Step 2: πg = πin + (gin/||gin||) * ||πout − πin||
        double norm_in_out = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double diff = nodeDuals[i] - cur_stab_center[i];
            norm_in_out += diff * diff;
        }
        norm_in_out = std::sqrt(norm_in_out + EPSILON);

        DualSolution pi_g(n);
        for (size_t i = 0; i < n; ++i) {
            pi_g[i] = cur_stab_center[i] +
                      (subgradient[i] / subgradient_norm) * norm_in_out;
        }

        // Step 3: ρ = βπg + (1−β)πout
        DualSolution rho(n);
        for (size_t i = 0; i < n; ++i) {
            rho[i] = beta * pi_g[i] + (1.0 - beta) * nodeDuals[i];
        }

        // Calculate ||π˜ − πin||
        double norm_tilde_in = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double diff = pi_tilde[i] - cur_stab_center[i];
            norm_tilde_in += diff * diff;
        }
        norm_tilde_in = std::sqrt(norm_tilde_in + EPSILON);

        // Calculate ||ρ − πin||
        double norm_rho_in = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double diff = rho[i] - cur_stab_center[i];
            norm_rho_in += diff * diff;
        }
        norm_rho_in = std::sqrt(norm_rho_in + EPSILON);

        // Step 4: πsep = πin + (||π˜ − πin||/||ρ − πin||)(ρ − πin)
        DualSolution new_duals(n);
        for (size_t i = 0; i < n; ++i) {
            new_duals[i] =
                cur_stab_center[i] +
                (norm_tilde_in / norm_rho_in) * (rho[i] - cur_stab_center[i]);
            new_duals[i] =
                std::max(0.0, new_duals[i]);  // Project onto positive orthant
        }

        smooth_dual_sol = new_duals;
        duals_sep = new_duals;
        return new_duals;
    }

    static constexpr double LARGE_NUMBER = 1e+6;
    inline double safeAdd(double a, double b) {
        if (std::abs(a) > LARGE_NUMBER || std::abs(b) > LARGE_NUMBER) {
            return std::copysign(LARGE_NUMBER, a + b);
        }
        return a + b;
    }

    inline double safeMult(double a, double b) {
        if (std::abs(a) < EPSILON || std::abs(b) < EPSILON) {
            return 0.0;
        }
        if (std::abs(a) > LARGE_NUMBER || std::abs(b) > LARGE_NUMBER) {
            return std::copysign(LARGE_NUMBER, a * b);
        }
        return a * b;
    }

    inline double safeDiv(double a, double b) {
        if (std::abs(b) < EPSILON) {
            return (std::abs(a) < EPSILON) ? 0.0
                                           : std::copysign(LARGE_NUMBER, a);
        }
        return a / b;
    }

    inline double safeNorm(const std::vector<double> &v1,
                           const std::vector<double> &v2) {
        double sum = 0.0;
        for (size_t i = 0; i < v1.size(); ++i) {
            double diff = v2[i] - v1[i];
            sum = safeAdd(sum, safeMult(diff, diff));
        }
        return std::sqrt(sum + EPSILON);
    }

    inline double safeNorm(const std::vector<double> &v) {
        double sum = 0.0;
        for (size_t i = 0; i < v.size(); ++i) {
            sum = safeAdd(sum, safeMult(v[i], v[i]));
        }
        return std::sqrt(sum + EPSILON);
    }

    static constexpr double EPSILON = 1e-12;

    bool dynamic_alpha_schedule(const ModelData &dados) {
        const size_t n = cur_stab_center.size();

        double relative_distance = norm(smooth_dual_sol, cur_stab_center) /
                                   (std::abs(lp_obj) + EPSILON);
        if (relative_distance < NORM_TOLERANCE) {
            alpha = 0.0;
            return false;
        }

        std::vector<double> direction(n);
        double direction_norm = 0.0;

        for (size_t i = 0; i < n; i++) {
            direction[i] = smooth_dual_sol[i] - cur_stab_center[i];
            direction_norm += direction[i] * direction[i];
        }
        direction_norm = std::sqrt(direction_norm + EPSILON);

        if (direction_norm < EPSILON || subgradient_norm < EPSILON) {
            return false;
        }

        std::vector<double> normalized_direction(n);
        std::vector<double> normalized_subgradient(n);

        for (size_t i = 0; i < n; i++) {
            normalized_direction[i] = direction[i] / direction_norm;
            normalized_subgradient[i] = subgradient[i] / subgradient_norm;
        }

        double cos_angle = 0.0;
        for (size_t i = 0; i < n; i++) {
            cos_angle += normalized_direction[i] * normalized_subgradient[i];
        }

        return cos_angle < 0;
    }

    void update_subgradient(const ModelData &dados,
                            const DualSolution &nodeDuals,
                            const std::vector<Label *> &best_pricing_cols) {
        size_t number_of_rows = nodeDuals.size();

        new_rows.assign(number_of_rows, 0.0);

        auto best_pricing_col = best_pricing_cols[0];

        for (const auto &node : best_pricing_col->nodes_covered) {
            if (node > 0 && node != N_SIZE - 1) {
                new_rows[node - 1] += 1.0;
            }
        }

        subgradient.assign(number_of_rows, 0.0);

        std::vector<double> most_reduced_cost_column = new_rows;

        std::transform(dados.b.begin(), dados.b.end(),
                       most_reduced_cost_column.begin(), subgradient.begin(),
                       [this](double a, double b) { return a - numK * b; });

        subgradient_norm = norm(subgradient);
    }

    void set_pseudo_dual_bound(double bound) { pseudo_dual_bound = bound; }

    int no_progress_count = 0;

    void setObj(double obj) { lp_obj = obj; }

    double lp_obj_prev = 0.0;
    const int NO_PROGRESS_THRESHOLD = 5;

    void update_stabilization_after_pricing_optim(
        const ModelData &dados, const DualSolution &input_duals,
        const double &lag_gap, std::vector<Label *> best_pricing_cols) {
        std::vector<double> nodeDuals(input_duals.begin(),
                                      input_duals.begin() + sizeDual);

        double lp_obj_rounded = std::round(lp_obj);
        // Check if lag_gap has changed
        if (lp_obj_rounded == lp_obj_prev) {
            no_progress_count++;
        } else {
            no_progress_count = 0;  // Reset the counter if lag_gap changes
        }
        lp_obj_prev = lp_obj_rounded;

        // If lag_gap remains constant for 3 iterations, set alpha to zero
        // if (no_progress_count >= NO_PROGRESS_THRESHOLD) {
        //     alpha = 0.0;
        //     base_alpha = 0.0;
        //     no_progress_count =
        //         0;  // Reset the counter after setting alpha to zero
        //     update_subgradient(dados, nodeDuals, best_pricing_cols);

        // } else {
        if (nb_misprices == 0) {
            update_subgradient(dados, nodeDuals, best_pricing_cols);
            bool should_increase = dynamic_alpha_schedule(dados);

            constexpr double ALPHA_FACTOR = 0.1;
            if (should_increase) {
                alpha = std::min(0.99, alpha + (1.0 - alpha) * ALPHA_FACTOR);
            } else {
                alpha = std::max(0.0, alpha / 1.1);
            }

            if (!std::isnan(alpha) && !std::isinf(alpha)) {
                base_alpha = alpha;
            }
        }
        // }

        DualSolution stab_sol = smooth_dual_sol;

        if (lag_gap < lag_gap_prev || cut_added) {
            stab_center_for_next_iteration = smooth_dual_sol;
            cut_added = false;
        } else {
            stab_center_for_next_iteration = cur_stab_center;
        }

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

    void define_smooth_dual_sol(const DualSolution &nodeDuals) {
        smooth_dual_sol.assign(nodeDuals.begin(), nodeDuals.begin() + sizeDual);
    }

    void updateNumK(int numK) { this->numK = numK; }

    void clearAlpha() { alpha = 0.0; }
};
