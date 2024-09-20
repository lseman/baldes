/**
 * @file Stabilization.h
 * @brief Defines the Stabilization class for dual stabilization in column generation.
 *
 * This file implements the Stabilization class, which is responsible for handling the
 * stabilization process during column generation in large-scale optimization problems,
 * such as vehicle routing and resource-constrained shortest path problems (RCSPP).
 *
 * The class maintains various dual solution parameters, including stability centers and
 * misprice sequences, and it adjusts dual values based on a set of rules to improve
 * convergence and avoid oscillations during the optimization process.
 *
 * Key Features:
 * - Maintains and updates dual solutions during the optimization.
 * - Handles the stabilization of dual values by adjusting the alpha parameter.
 * - Provides methods for computing the norms and updates for misprice sequences.
 * - Calculates subgradients and adjusts duals using norms and angles between vectors.
 * - Implements a misprice schedule to control the alpha value during stabilization.
 *
 * The Stabilization class is crucial for improving the convergence of column generation
 * algorithms by stabilizing the dual values and avoiding large oscillations in the dual space.
 */

#pragma once

#include "Definitions.h"
#include "gurobi_c++.h"
#include <algorithm> // For std::transform
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric> // For std::iota
#include <vector>

/**
 * @class Stabilization
 * @brief A class to handle stabilization in optimization problems.
 *
 * This class implements various methods to manage and compute stabilization parameters
 * and dual solutions in optimization problems.
 */
class Stabilization {
public:
    double alpha; // Current alpha parameter
    int    t;     // Iteration counter

    double       base_alpha;                     // "global" alpha parameter
    double       cur_alpha;                      // alpha parameter during the current misprice sequence
    int          nb_misprices = 0;               // number of misprices during the current misprice sequence
    double       pseudo_dual_bound;              // pseudo dual bound, may be non-valid
    double       valid_dual_bound;               // valid dual bound
    DualSolution cur_stab_center;                // current stability center
    DualSolution stab_center_for_next_iteration; // stability center for the next iteration

    DualSolution phi_in;
    DualSolution phi_out;
    DualSolution prev_dual;
    bool         misprice = false;

    DualSolution duals_in;
    DualSolution duals_out;
    DualSolution duals_sep;
    DualSolution duals_g;
    double       beta;
    DualSolution rho;
    DualSolution master_dual;
    DualSolution smooth_dual_sol;

    double lag_gap      = 0.0;
    double lag_gap_prev = std::numeric_limits<double>::infinity();

    /**
     * @brief Increments the misprice counter if the alpha value is positive.
     *
     * This function checks if the alpha value is greater than zero. If it is,
     * the function increments the misprice counter (nb_misprices).
     */
    void add_misprice() {
        nb_misprices++;
        cur_alpha = _misprice_schedule(nb_misprices, base_alpha);
    }

    /**
     * @brief Updates the stabilization center after an iteration.
     *
     * This function updates the current stabilization center with the new center
     * if the `stab_center_for_next_iteration` is not empty. After updating, it clears
     * the `stab_center_for_next_iteration`.
     *
     * @param new_center The new dual solution center to be considered for stabilization.
     */
    void update_stabilization_after_iter(const DualSolution &new_center) {
        if (!stab_center_for_next_iteration.empty()) {
            cur_stab_center = stab_center_for_next_iteration;
            stab_center_for_next_iteration.clear();
        }
    }

    /**
     * @brief Updates the stabilization parameters after the master optimization.
     *
     * This function updates the stabilization parameters based on the new center
     * provided by the master optimization. It resets the mispricing counter and
     * the current alpha value. If the current stabilization center is empty, it
     * sets it to the new center and returns false. Otherwise, it updates the 
     * current alpha to the base alpha and returns whether the current alpha is 
     * greater than zero.
     *
     * @param new_center The new center provided by the master optimization.
     * @return true if the current alpha is greater than zero, false otherwise.
     */
    bool update_stabilization_after_master_optim(const DualSolution &new_center) {
        nb_misprices = 0;
        cur_alpha    = 0;

        if (cur_stab_center.empty()) {
            cur_stab_center = new_center;
            return false;
        }
        cur_alpha = base_alpha;
        return cur_alpha > 0;
    }

    /**
     * @brief Resets the mispricing counter to zero.
     *
     * This function sets the number of misprices (`nb_misprices`) to zero,
     * effectively resetting any previously recorded mispricing events.
     */
    void reset_misprices() { nb_misprices = 0; }

    /**
     * Calculates the stabilization factor based on the number of misprices.
     *
     * @param nb_misprices The number of misprices encountered.
     * @param base_alpha The base alpha value used for calculation.
     * @return The calculated alpha value. If the number of misprices is greater than 10 or the calculated alpha is less
     * than or equal to 0.001, stabilization is deactivated and alpha is set to 0.0.
     */
    double _misprice_schedule(int nb_misprices, double base_alpha) {
        double alpha = 1.0 - (nb_misprices + 1) * (1 - base_alpha);
        if (nb_misprices > 10 || alpha <= 1e-3) {
            alpha = 0.0; // Deactivate stabilization
        }
        return alpha;
    }

    // Constructor
    /**
     * @brief Constructs a Stabilization object with the given base alpha value and master dual solution.
     * 
     * @param base_alpha The base alpha value used for stabilization.
     * @param mast_dual_sol A reference to the master dual solution.
     */
    Stabilization(double base_alpha, DualSolution &mast_dual_sol)
        : alpha(base_alpha), t(0), base_alpha(base_alpha), cur_alpha(base_alpha), nb_misprices(0),
          cur_stab_center(mast_dual_sol) {
        pseudo_dual_bound = std::numeric_limits<double>::infinity();
        valid_dual_bound  = std::numeric_limits<double>::infinity();
        beta              = 0.0;
    }

    /**
     * @brief Computes the stabilized dual solution.
     *
     * This function calculates a stabilized dual solution by combining the
     * input and output dual values using a weighted average. The weights are
     * determined by the current alpha value (`cur_alpha`).
     *
     * @return DualSolution The stabilized dual solution.
     */
    DualSolution getStabDualSol(const DualSolution &master_dual) {
        if (cur_stab_center.empty()) { return master_dual; }
        DualSolution stab_dual_sol(master_dual.size());
        std::transform(master_dual.begin(), master_dual.end(), cur_stab_center.begin(), stab_dual_sol.begin(),
                       [this](double out, double in) { return cur_alpha * in + (1 - cur_alpha) * out; });
        smooth_dual_sol = stab_dual_sol;
        return stab_dual_sol;
    }


    /**
     * @brief Updates the stabilization parameter alpha based on the provided model data, dual solutions, and lagrangian gap.
     *
     * This function performs several steps to update the stabilization parameter alpha:
     * 1. Calculates the direction of separation between the smooth dual solution and the current stabilization center.
     * 2. Computes the norm of the smooth dual solution.
     * 3. Checks if the norm is zero and returns early if it is.
     * 4. Calculates the subgradient as the difference between the right-hand side vector and the product of the constraint matrix and the primal solution.
     * 5. Accumulates values of the constraint matrix multiplied by the primal solution.
     * 6. Updates the subgradient based on the best pricing columns.
     * 7. Computes the norm of the subgradient and normalizes it if non-zero.
     * 8. Calculates the cosine of the angle between the separation direction and the subgradient.
     * 9. Adjusts the current alpha based on the cosine angle and a threshold.
     * 10. Updates the stabilization center for the next iteration if the lagrangian gap has decreased.
     *
     * @param dados The model data containing the constraint matrix and other relevant information.
     * @param jobDuals The dual solution for the job constraints.
     * @param master_primal The primal solution for the master problem.
     * @param lag_gap The current lagrangian gap.
     * @param best_pricing_cols A vector of pointers to the best pricing columns.
     */
    void updateAlpha(const ModelData &dados, const DualSolution &jobDuals, const DualSolution &master_primal,
                     const double &lag_gap, std::vector<Label *> best_pricing_cols) {

        // calculate smooth_dual_sol - cur_stab_center
        std::vector<double> in_sep_direction(jobDuals.size());
        std::transform(smooth_dual_sol.begin(), smooth_dual_sol.end(), cur_stab_center.begin(),
                       in_sep_direction.begin(), [](double a, double b) { return a - b; });

        // calculate norm of smooth_dual_sol
        double norm_smooth_dual_sol = std::transform_reduce(in_sep_direction.begin(), in_sep_direction.end(), 0.0,
                                                            std::plus<>(), [](double a) { return a * a; });

        // check if norm is 0, if so return
        if (norm_smooth_dual_sol == 0) { return; }

        // calculate subgradient as b - A * primal_sol
        std::vector<double> subgradient(jobDuals.size());

        // Step 5: Accumulate values of A * x in lagrangian_constraint_values
        std::vector<double> lagrangian_constraint_values(jobDuals.size(), 0.0);
        /*
        for (size_t idx = 0; idx < dados.A_sparse.values.size(); ++idx) {
            int row_id = dados.A_sparse.row_indices[idx];
            if (row_id >= N_SIZE - 1) { continue; }
            int col_id = dados.A_sparse.col_indices[idx];
            lagrangian_constraint_values[row_id] = dados.A_sparse.values[idx] * master_primal[col_id];
        }
        */

        // Update subgradient based on best_pricing_cols
        for (size_t row_id = 0; row_id < subgradient.size(); ++row_id) {
            char sense = dados.sense[row_id];

            // Apply contribution from best_pricing_col if non-zero
            for (auto best_pricing_col : best_pricing_cols) {
                if (best_pricing_col->cost > 0) { continue; }
                for (auto row : best_pricing_col->jobs_covered) {
                    if (row > 0 && row != N_SIZE - 1) { subgradient[row - 1] += 1; }
                }
            }

            // Check participation in stabilization
            if (sense == '<') {
                subgradient[row_id] = std::min(
                    0.0, subgradient[row_id] - lagrangian_constraint_values[row_id]); // For less-than constraints
            } else if (sense == '>') {
                subgradient[row_id] = std::max(
                    0.0, subgradient[row_id] - lagrangian_constraint_values[row_id]); // For greater-than constraints
            } else {                                                                  // For equality constraints
                subgradient[row_id] = subgradient[row_id] - lagrangian_constraint_values[row_id];
            }
        }

        // Compute the norm of the subgradient
        double subgradient_norm = std::transform_reduce(subgradient.begin(), subgradient.end(), 0.0, std::plus<>(),
                                                        [](double a) { return a * a; });

        // normalize subgradient if non-zero
        if (subgradient_norm > 0) {
            subgradient_norm = std::sqrt(subgradient_norm);
            for (auto &value : subgradient) { value /= subgradient_norm; }
        }
        double cos_angle =
            std::inner_product(in_sep_direction.begin(), in_sep_direction.end(), subgradient.begin(), 0.0) /
            (norm_smooth_dual_sol * subgradient_norm);

        cos_angle = cos_angle * 1;
        // Check if cos_angle is less than threshold (1e-12), and adjust cur_alpha accordingly
        if (cos_angle > 1e-12) {
            cur_alpha = std::min(0.99, cur_alpha + (1.0 - cur_alpha) * 0.1);
        } else {
            cur_alpha = std::max(0.0, cur_alpha - 0.1);
        }

        if (lag_gap < lag_gap_prev) { stab_center_for_next_iteration = smooth_dual_sol; }
        base_alpha = cur_alpha;
    }
};