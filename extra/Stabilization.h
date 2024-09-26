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

#include "DataClasses.h"
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

    double              subgradient_norm = 0.0;
    DualSolution        subgradient;
    std::vector<double> new_rows;

    double lag_gap      = 0.0;
    double lag_gap_prev = std::numeric_limits<double>::infinity();

    /**
     * @brief Increments the misprice counter if the alpha value is positive.
     *
     * This function checks if the alpha value is greater than zero. If it is,
     * the function increments the misprice counter (nb_misprices).
     */
    void update_stabilization_after_misprice() {
        nb_misprices++;
        alpha = _misprice_schedule(nb_misprices, base_alpha);
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
     * @brief Computes the Euclidean norm (L2 norm) of a given vector.
     *
     * This function calculates the square root of the sum of the squares of the elements
     * in the input vector, with a small epsilon added to avoid division by zero.
     *
     * @param vector A constant reference to a std::vector<double> containing the elements.
     * @return The Euclidean norm of the vector.
     */
    inline double norm(const std::vector<double> &vector) {
        double res = 0;
        for (int i = 0; i < vector.size(); ++i) { res += vector[i] * vector[i]; }
        return std::sqrt(res + 1e-6);
    }

    /**
     * @brief Computes the Euclidean norm (distance) between two vectors.
     *
     * This function calculates the Euclidean distance between two vectors by summing the squared differences
     * of their corresponding elements and then taking the square root of the result, with a small epsilon added
     * to avoid division by zero.
     *
     * @param vector_1 The first vector.
     * @param vector_2 The second vector.
     * @return The Euclidean norm (distance) between vector_1 and vector_2.
     */
    inline double norm(const std::vector<double> &vector_1, const std::vector<double> &vector_2) {
        double res = 0.0;
        for (auto i = 0; i < vector_1.size(); ++i) { res += (vector_2[i] - vector_1[i]) * (vector_2[i] - vector_1[i]); }
        return std::sqrt(res + 1e-6);
    }

    /**
     * @brief Computes an advanced stabilized dual solution.
     *
     * This function computes a stabilized dual solution based on the provided dual solution (`jobDuals`).
     * It uses various internal states and parameters to compute the stabilized solution, including
     * `cur_stab_center`, `smooth_dual_sol`, `subgradient`, and `duals_sep`. The function follows a series
     * of steps to compute intermediate values such as `duals_tilde`, `duals_g`, and `rho`, and uses these
     * to update the `duals_sep` and `smooth_dual_sol`.
     *
     * @param jobDuals The input dual solution to be stabilized.
     * @return The stabilized dual solution.
     */
    // TODO: we need to check the implementation of this method
    DualSolution getStabDualSolAdvanced(const DualSolution &jobDuals) {
        // If there's no stabilization center, return the input duals
        if (cur_stab_center.empty()) { return jobDuals; }

        // Initialize subgradient if it hasn't been initialized yet
        if (subgradient.empty()) { subgradient.assign(jobDuals.size(), 0.0); }

        // Initialize the separation duals and smooth dual solution if they are empty
        if (duals_sep.empty()) {
            duals_sep       = jobDuals;  // Start with the input duals
            smooth_dual_sol = duals_sep; // Initialize the smoothed dual solution
            return jobDuals;             // Return the input duals for now
        }

        // Get the current stabilization center (duals_in) and the current job duals (duals_out)
        const auto &duals_in  = cur_stab_center;
        const auto &duals_out = jobDuals;

        const size_t        n = jobDuals.size();
        std::vector<double> duals_tilde(n, 0.0);
        std::vector<double> duals_g(n, 0.0);
        std::vector<double> rho(n, 0.0);

        // Precompute norms that are used multiple times
        double norm_in_out      = norm(duals_in, duals_out);
        double norm_subgradient = norm(subgradient);

        // Compute π_tilde: a convex combination of duals_in and duals_out
        double one_minus_alpha = 1 - base_alpha;
        for (size_t row_id = 0; row_id < n; ++row_id) {
            duals_tilde[row_id] = base_alpha * duals_in[row_id] + one_minus_alpha * duals_out[row_id];
        }

        // Compute the coefficient for π_g based on the norm of duals_in and duals_out
        double coef_g = (norm_subgradient != 0.0) ? norm_in_out / norm_subgradient : 0.0;

        // Compute π_g: duals_in + coef_g * subgradient
        for (size_t row_id = 0; row_id < n; ++row_id) {
            duals_g[row_id] = duals_in[row_id] + coef_g * subgradient[row_id];
        }

        // Compute β: a weight factor that depends on the alignment of duals_g and duals_out
        double dot_product = 0.0;
        for (size_t row_id = 0; row_id < n; ++row_id) {
            dot_product += (duals_out[row_id] - duals_in[row_id]) * (duals_g[row_id] - duals_in[row_id]);
        }

        double norm_in_g = norm(duals_in, duals_g);
        beta = (norm_in_out * norm_in_g != 0.0) ? std::max(0.0, dot_product / (norm_in_out * norm_in_g)) : 0.0;

        // Compute ρ: a combination of duals_g and duals_out based on β
        double one_minus_beta = 1 - beta;
        for (size_t row_id = 0; row_id < n; ++row_id) {
            rho[row_id] = beta * duals_g[row_id] + one_minus_beta * duals_out[row_id];
        }

        // Compute the coefficient for dual separation (coef_sep)
        double norm_in_tilde = norm(duals_in, duals_tilde);
        double norm_in_rho   = norm(duals_in, rho);
        double coef_sep      = (norm_in_rho != 0.0) ? norm_in_tilde / norm_in_rho : 0.0;

        // Update the duals_sep by adjusting duals_in towards ρ
        for (size_t row_id = 0; row_id < n; ++row_id) {
            duals_sep[row_id] = duals_in[row_id] + coef_sep * (rho[row_id] - duals_in[row_id]);
        }

        // Update the smooth dual solution and return it
        smooth_dual_sol = duals_sep;
        return duals_sep;
    }

    /**
     * @brief Computes the dynamic alpha schedule for stabilization.
     *
     * This function calculates the dynamic alpha schedule based on the provided model data, dual solution,
     * and best pricing columns. It adjusts the current alpha value based on the angle between the
     * separation direction and the subgradient.
     *
     * @param dados The model data containing problem-specific information.
     * @param jobDuals The dual solution vector.
     * @param best_pricing_cols A vector of pointers to the best pricing columns.
     * @return The updated alpha value based on the dynamic schedule.
     */
    double dynamic_alpha_schedule(const ModelData &dados, const DualSolution &jobDuals,
                                  const std::vector<Label *> &best_pricing_cols) {

        // Get the number of rows from the size of the jobDuals
        size_t number_of_rows = jobDuals.size();

        // Calculate the direction of separation between the current smoothed dual solution and stabilization center
        std::vector<double> in_sep_direction(number_of_rows);
        std::transform(smooth_dual_sol.begin(), smooth_dual_sol.end(), cur_stab_center.begin(),
                       in_sep_direction.begin(), std::minus<>());

        // Calculate the norm of the smoothed dual solution (accumulating squared differences)
        double norm_smooth_dual_sol = std::transform_reduce(in_sep_direction.begin(), in_sep_direction.end(), 0.0,
                                                            std::plus<>(), [](double a) { return a * a; });

        // If the norm is zero, return the base alpha (no adjustment necessary)
        if (norm_smooth_dual_sol == 0.0) { return base_alpha; }

        // Initialize new_rows and apply contributions from best_pricing_cols, skipping positive-cost columns
        new_rows.assign(number_of_rows, 0.0);
        for (const auto *best_pricing_col : best_pricing_cols) {
            if (best_pricing_col->cost > 0) { continue; }
            for (const auto &row : best_pricing_col->jobs_covered) {
                if (row > 0 && row != N_SIZE - 1) { // Ignore invalid rows and the last job
                    new_rows[row - 1] += 1;
                }
            }
        }

        // Define and set row bounds based on the problem constraints (dados.sense)
        // Reuse vectors to avoid reallocations
        static std::vector<double> new_row_lower_bounds, new_row_upper_bounds;
        if (new_row_lower_bounds.size() != number_of_rows) {
            new_row_lower_bounds.resize(number_of_rows);
            new_row_upper_bounds.resize(number_of_rows);
        }

        // Set row bounds
        for (size_t row_id = 0; row_id < number_of_rows; ++row_id) {
            if (dados.sense[row_id] == '<') {
                new_row_upper_bounds[row_id] = dados.b[row_id];
                new_row_lower_bounds[row_id] = -std::numeric_limits<double>::infinity();
            } else if (dados.sense[row_id] == '>') {
                new_row_upper_bounds[row_id] = std::numeric_limits<double>::infinity();
                new_row_lower_bounds[row_id] = dados.b[row_id];
            } else {
                new_row_upper_bounds[row_id] = dados.b[row_id];
                new_row_lower_bounds[row_id] = dados.b[row_id];
            }
        }

        // Update the subgradient based on new row values and bounds
        subgradient.assign(number_of_rows, 0.0);
        for (size_t row_id = 0; row_id < number_of_rows; ++row_id) {
            subgradient[row_id] = std::min(0.0, new_row_upper_bounds[row_id] - new_rows[row_id]) +
                                  std::max(0.0, new_row_lower_bounds[row_id] - new_rows[row_id]);
        }

        // Compute the norm of the subgradient
        subgradient_norm = norm(subgradient);

        // Compute the cosine of the angle between in_sep_direction and subgradient
        double cos_angle =
            std::inner_product(in_sep_direction.begin(), in_sep_direction.end(), subgradient.begin(), 0.0) /
            (norm_smooth_dual_sol * subgradient_norm);

        // Negate the cosine angle to invert the direction
        cos_angle = -cos_angle;

        // Adjust cur_alpha based on the cosine angle and a threshold
        if (cos_angle > 1e-12) {
            cur_alpha = std::min(0.99, cur_alpha + (1.0 - cur_alpha) * 0.1); // Increase cur_alpha
        } else {
            cur_alpha = std::max(0.0, cur_alpha - 0.1); // Decrease cur_alpha
        }

        // Return the updated cur_alpha
        return cur_alpha;
    }

    /**
     * @brief Updates the stabilization parameters after the pricing optimization step.
     *
     * This function adjusts the stabilization parameters based on the results of the
     * pricing optimization. It updates the alpha value using a dynamic schedule if
     * there have been no misprices. Additionally, it updates the stabilization center
     * for the next iteration if the current lagrangian gap is smaller than the previous one.
     *
     * @param dados The model data containing relevant information for the optimization.
     * @param jobDuals The dual solution obtained from the pricing optimization.
     * @param lag_gap The current lagrangian gap.
     * @param best_pricing_cols A vector of pointers to the best pricing columns.
     */
    void update_stabilization_after_pricing_optim(const ModelData &dados, const DualSolution &jobDuals,
                                                  const double &lag_gap, std::vector<Label *> best_pricing_cols) {
        if (nb_misprices == 0) { alpha = dynamic_alpha_schedule(dados, jobDuals, best_pricing_cols); }
        if (lag_gap < lag_gap_prev) { stab_center_for_next_iteration = smooth_dual_sol; }
        base_alpha = alpha;
    }

    /**
     * @brief Determines whether the process should exit based on the current alpha value.
     *
     * This function checks the value of `cur_alpha`. If `cur_alpha` is zero, it returns true,
     * indicating that the process should exit. Otherwise, it returns false.
     *
     * @return true if `cur_alpha` is zero, otherwise false.
     */
    bool shouldExit() {
        if (base_alpha == 0) { return true; }
        return false;
    }
};
