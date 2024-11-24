/**
 * @file Stabilization.h
 * @brief Defines the Stabilization class for dual stabilization in column generation.
 *
 * This file implements the Stabilization class, which is responsible for handling the
 * stabilization process during column generation in large-scale optimization problems,
 * such as vehicle routing and resource-constrained shortest path problems (RCSPP).
 *
 * The Stabilization class is crucial for improving the convergence of column generation
 * algorithms by stabilizing the dual values and avoiding large oscillations in the dual space.
 *
 */

#pragma once

#include "Definitions.h"
#include "Pools.h"

#ifdef GUROBI
#include "gurobi_c++.h"
#include "gurobi_c.h"
#endif

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
    double lag_gap_prev = -std::numeric_limits<double>::infinity();

    int sizeDual;

    /**
     * @brief Increments the misprice counter if the alpha value is positive.
     *
     * This function checks if the alpha value is greater than zero. If it is,
     * the function increments the misprice counter (nb_misprices).
     */
    void update_stabilization_after_misprice() {
        nb_misprices++;
        alpha     = _misprice_schedule(nb_misprices, base_alpha);
        cur_alpha = alpha;
        beta      = 0.0;
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
        duals_in = duals_sep;
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
        sizeDual          = mast_dual_sol.size();
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
    DualSolution getStabDualSol(const DualSolution &input_duals) {
        std::vector<double> master_dual;
        master_dual.assign(input_duals.begin(), input_duals.begin() + sizeDual);
        if (cur_stab_center.empty()) { return master_dual; }
        DualSolution stab_dual_sol(master_dual.size());
        for (size_t i = 0; i < master_dual.size(); ++i) {
            stab_dual_sol[i] = std::max(0.0, cur_alpha * cur_stab_center[i] + (1 - cur_alpha) * master_dual[i]);
        }
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
        return std::sqrt(res + 1e-8);
    }

    inline std::vector<double> mult(const std::vector<double> &v, const std::vector<double> &w) {
        std::vector<double> res(v.size(), 0.0);
        for (size_t i = 0; i < v.size(); ++i) { res[i] = v[i] * w[i]; }
        return res;
    }

    inline std::vector<double> sub(const std::vector<double> &v, const std::vector<double> &w) {
        std::vector<double> res(v.size(), 0.0);
        for (size_t i = 0; i < v.size(); ++i) { res[i] = v[i] - w[i]; }
        return res;
    }

    /**
     * @brief Computes an advanced stabilized dual solution.
     *
     * This function computes a stabilized dual solution based on the provided dual solution (`nodeDuals`).
     * It uses various internal states and parameters to compute the stabilized solution, including
     * `cur_stab_center`, `smooth_dual_sol`, `subgradient`, and `duals_sep`. The function follows a series
     * of steps to compute intermediate values such as `duals_tilde`, `duals_g`, and `rho`, and uses these
     * to update the `duals_sep` and `smooth_dual_sol`.
     *
     */

    std::vector<double> duals_tilde;

    // TODO: we need to check the implementation of this method
    DualSolution getStabDualSolAdvanced(const DualSolution &input_duals) {
        // Constants for numerical stability
        constexpr double EPSILON = 1e-12;

        // Extract relevant dual values
        DualSolution nodeDuals;
        nodeDuals.assign(input_duals.begin(), input_duals.begin() + sizeDual);

        // Base case: no stabilization center
        if (cur_stab_center.empty()) { return nodeDuals; }

        // Initialize subgradient if needed
        if (subgradient.empty()) { subgradient.assign(nodeDuals.size(), 0.0); }

        // Initialize separation duals on first call
        if (duals_sep.empty()) {
            duals_sep       = nodeDuals;
            smooth_dual_sol = duals_sep;
            return nodeDuals;
        }

        // Set up references for clearer notation matching the paper
        const auto  &duals_in  = cur_stab_center; // π_in (stability center)
        const auto  &duals_out = nodeDuals;       // π_out (current duals)
        const size_t n         = nodeDuals.size();

        // Resize working vectors
        duals_tilde.resize(n);
        duals_g.resize(n);
        rho.resize(n);

        // Compute norms with numerical stability
        double norm_in_out      = safeNorm(duals_in, duals_out); // ||π_out - π_in||
        double norm_subgradient = safeNorm(subgradient);         // ||g||

        if (norm_subgradient < EPSILON) {
            return getStabDualSol(input_duals); // Fall back to basic smoothing
        }

        // Step 1: Compute π_tilde (Wentges smoothing)
        for (size_t i = 0; i < n; ++i) {
            duals_tilde[i] = safeAdd(safeMult(cur_alpha, duals_in[i]), safeMult((1.0 - cur_alpha), duals_out[i]));
        }

        // Step 2: Compute π_g using normalized subgradient
        double coef_g = safeDiv(norm_in_out, norm_subgradient);
        for (size_t i = 0; i < n; ++i) { duals_g[i] = safeAdd(duals_in[i], safeMult(coef_g, subgradient[i])); }

        // Step 3: Compute β (combination coefficient)
        if (nb_misprices > 0) {
            beta = 0.0; // Reset during mispricing
        } else {
            // Compute angle between (π_out - π_in) and (π_g - π_in)
            double dot_product = 0.0;
            for (size_t i = 0; i < n; ++i) {
                dot_product = safeAdd(dot_product, safeMult((duals_out[i] - duals_in[i]), (duals_g[i] - duals_in[i])));
            }

            double norm_in_g = safeNorm(duals_in, duals_g);
            beta             = safeDiv(dot_product, safeMult(norm_in_out, norm_in_g));
        }
        beta = std::max(0.0, beta);

        // Step 4: Compute ρ (combined direction)
        for (size_t i = 0; i < n; ++i) {
            rho[i] = safeAdd(safeMult(beta, duals_g[i]), safeMult((1.0 - beta), duals_out[i]));
        }

        // Step 5: Compute step size coefficient
        double norm_in_tilde = safeNorm(duals_in, duals_tilde);
        double norm_in_rho   = safeNorm(duals_in, rho);
        double coef_sep      = safeDiv(norm_in_tilde, norm_in_rho);

        // Step 6: Update separation point
        for (size_t i = 0; i < n; ++i) {
            // Take step from π_in in direction of ρ
            duals_sep[i] = safeAdd(duals_in[i], safeMult(coef_sep, (rho[i] - duals_in[i])));
            // Ensure dual feasibility
            duals_sep[i] = std::max(0.0, duals_sep[i]);
        }

        // Update tracking and return
        smooth_dual_sol = duals_sep;
        return duals_sep;
    }

    static constexpr double LARGE_NUMBER = 1e+6;
    // Safe numerical operations
    inline double safeAdd(double a, double b) {
        if (std::abs(a) > LARGE_NUMBER || std::abs(b) > LARGE_NUMBER) { return std::copysign(LARGE_NUMBER, a + b); }
        return a + b;
    }

    inline double safeMult(double a, double b) {
        if (std::abs(a) < EPSILON || std::abs(b) < EPSILON) { return 0.0; }
        if (std::abs(a) > LARGE_NUMBER || std::abs(b) > LARGE_NUMBER) { return std::copysign(LARGE_NUMBER, a * b); }
        return a * b;
    }

    inline double safeDiv(double a, double b) {
        if (std::abs(b) < EPSILON) { return (std::abs(a) < EPSILON) ? 0.0 : std::copysign(LARGE_NUMBER, a); }
        return a / b;
    }

    inline double safeNorm(const std::vector<double> &v1, const std::vector<double> &v2) {
        double sum = 0.0;
        for (size_t i = 0; i < v1.size(); ++i) {
            double diff = v2[i] - v1[i];
            sum         = safeAdd(sum, safeMult(diff, diff));
        }
        return std::sqrt(sum + EPSILON);
    }

    inline double safeNorm(const std::vector<double> &v) {
        double sum = 0.0;
        for (size_t i = 0; i < v.size(); ++i) { sum = safeAdd(sum, safeMult(v[i], v[i])); }
        return std::sqrt(sum + EPSILON);
    }

    static constexpr double EPSILON = 1e-12;

    /**
     * @brief Computes the dynamic alpha schedule for stabilization.
     *
     * This function calculates the dynamic alpha schedule based on the provided model data, dual solution,
     * and best pricing columns. It adjusts the current alpha value based on the angle between the
     * separation direction and the subgradient.
     *
     * @param dados The model data containing problem-specific information.
     * @param nodeDuals The dual solution vector.
     * @param best_pricing_cols A vector of pointers to the best pricing columns.
     * @return The updated alpha value based on the dynamic schedule.
     */
    bool dynamic_alpha_schedule(const ModelData &dados) {
        // Constants for numerical stability

        // Get dimensionality from stability center
        size_t number_of_rows = cur_stab_center.size();

        // Compute separation direction (π_sep - π_in)
        std::vector<double> in_sep_direction(number_of_rows, 0.0);
        in_sep_direction = sub(smooth_dual_sol, cur_stab_center);

        // Compute norm of separation direction
        double in_sep_dir_norm = norm(cur_stab_center, smooth_dual_sol);
        if (in_sep_dir_norm < EPSILON) {
            return false; // No meaningful direction
        }

        // Guard against zero subgradient norm
        if (subgradient_norm < EPSILON) { return false; }

        // Compute cosine of angle between subgradient and separation direction
        double dot_product = 0.0;
        for (size_t row_id = 0; row_id < number_of_rows; ++row_id) {
            dot_product += subgradient[row_id] * in_sep_direction[row_id];
        }

        // Compute cos(angle) with numerical stability
        double cos_angle = safeDiv(dot_product, safeMult(in_sep_dir_norm, subgradient_norm));

        // Return true if angle is close to 90 degrees (cos close to 0)
        // This indicates gradient is nearly perpendicular to separation direction
        return std::abs(cos_angle) < EPSILON;
    }

    void update_subgradient(const ModelData &dados, const DualSolution &nodeDuals,
                            const std::vector<Label *> &best_pricing_cols) {
        size_t number_of_rows = nodeDuals.size();

        // Initialize new_rows to track column contributions
        new_rows.assign(number_of_rows, 0.0);

        // Step 1: Calculate new_rows from columns brought by pricing
        for (const auto *best_pricing_col : best_pricing_cols) {
            // Skip positive reduced cost columns as they won't improve the solution
            if (best_pricing_col->cost >= 0) { continue; }

            // Update rows based on column contributions
            for (const auto &node : best_pricing_col->nodes_covered) {
                // Skip artificial nodes (0 and last node)
                if (node > 0 && node != N_SIZE - 1) { new_rows[node - 1] += 1.0; }
            }
        }

        // Step 2: Calculate the subgradient based on constraint types
        subgradient.assign(number_of_rows, 0.0);

        for (size_t row_id = 0; row_id < number_of_rows; ++row_id) {
            // For each constraint type, calculate appropriate component as per paper:
            // g = min(0, b - ax) for ≤ constraints
            // g = max(0, b - ax) for ≥ constraints
            // g = (b - ax) for = constraints
            if (dados.sense[row_id] == '<') {
                // ax ≤ b  =>  b - ax ≥ 0
                subgradient[row_id] = std::min(0.0, dados.b[row_id] - new_rows[row_id]);
            } else if (dados.sense[row_id] == '>') {
                // ax ≥ b  =>  b - ax ≤ 0
                subgradient[row_id] = std::max(0.0, dados.b[row_id] - new_rows[row_id]);
            } else { // dados.sense[row_id] == '='
                // ax = b
                subgradient[row_id] = dados.b[row_id] - new_rows[row_id];
            }
        }

        // Update subgradient norm
        subgradient_norm = norm(subgradient);
    }

    double compute_pseudo_dual_bound(const ModelData &dados, const DualSolution &nodeDuals,
                                     const std::vector<Label *> &best_pricing_cols) {
        double pseudo_dual_bound = 0.0;

        // Compute contribution from master dual solution
        for (size_t row_id = 0; row_id < nodeDuals.size(); ++row_id) {
            pseudo_dual_bound += dados.b[row_id] * nodeDuals[row_id];
        }

        return pseudo_dual_bound;
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
     * @param nodeDuals The dual solution obtained from the pricing optimization.
     * @param lag_gap The current lagrangian gap.
     * @param best_pricing_cols A vector of pointers to the best pricing columns.
     */
    void update_stabilization_after_pricing_optim(const ModelData &dados, const DualSolution &input_duals,
                                                  const double &lag_gap, std::vector<Label *> best_pricing_cols) {
        // Extract relevant dual values
        std::vector<double> nodeDuals;
        nodeDuals.assign(input_duals.begin(), input_duals.begin() + sizeDual);

        // Only update parameters if we're not in a mis-pricing sequence
        if (nb_misprices == 0) {
            // Update subgradient based on new pricing information
            update_subgradient(dados, nodeDuals, best_pricing_cols);

            // Get direction from dynamic schedule
            bool should_increase = dynamic_alpha_schedule(dados);

            // Adjust alpha based on gradient information
            constexpr double ALPHA_INCREASE_FACTOR = 0.1;
            constexpr double ALPHA_DECREASE_FACTOR = 0.1;
            constexpr double MAX_ALPHA             = 0.99;
            constexpr double MIN_ALPHA             = 0.0;

            if (should_increase) {
                // Increase alpha when the angle between subgradient and direction is large
                alpha = std::min(MAX_ALPHA, base_alpha + (1.0 - base_alpha) * ALPHA_INCREASE_FACTOR);
            } else {
                // Decrease alpha when the gradient indicates better progress
                alpha = std::max(MIN_ALPHA, base_alpha - ALPHA_DECREASE_FACTOR);
            }
        }

        // Update stability center based on gap improvement
        if (lag_gap < lag_gap_prev) {
            // If we made progress, use the current smoothed solution
            stab_center_for_next_iteration = smooth_dual_sol;
        } else {
            // Otherwise, keep the current center
            stab_center_for_next_iteration = cur_stab_center;
        }

        // Update tracking values for next iteration
        lag_gap_prev = lag_gap;
        base_alpha   = alpha;

        // Safeguard against numerical issues
        if (std::isnan(alpha) || std::isinf(alpha)) {
            alpha      = 0.0;
            base_alpha = 0.0;
        }
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

    void define_smooth_dual_sol(const DualSolution &nodeDuals) {
        smooth_dual_sol.assign(nodeDuals.begin(), nodeDuals.begin() + sizeDual);
    }
};