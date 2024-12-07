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

#include "MultiPointMgr.h"

#include <algorithm> // For std::transform
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric> // For std::iota
#include <vector>
#define NORM_TOLERANCE 1e-4

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

    bool stabilization_active = true;

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

    int numK = 8;

    int sizeDual;

    double lp_obj = 0.0;

    ReducedCostResult rc;

    MultiPointManager   mp_manager;
    std::vector<double> stab_constraint_values;

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
          cur_stab_center(mast_dual_sol), mp_manager(mast_dual_sol.size()) {
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

    DualSolution getStabDualSolAdvanced(const DualSolution &input_duals) {
        constexpr double EPSILON = 1e-12;
        DualSolution     nodeDuals(input_duals.begin(), input_duals.begin() + sizeDual);

        if (cur_stab_center.empty() || subgradient.empty() || duals_sep.empty()) { return getStabDualSol(input_duals); }

        const size_t        n = nodeDuals.size();
        std::vector<double> direction(n);

        // 1. Calculate in-out direction and its norm
        double norm_in_out = 0.0;
        for (size_t i = 0; i < n; ++i) {
            direction[i] = nodeDuals[i] - cur_stab_center[i];
            norm_in_out += direction[i] * direction[i];
        }
        norm_in_out = std::sqrt(norm_in_out + EPSILON);

        if (norm_in_out < EPSILON || subgradient_norm < EPSILON) { return getStabDualSol(input_duals); }

        // 2. Compute angle between subgradient and in-out direction
        double dot_product = 0.0;
        for (size_t i = 0; i < n; ++i) { dot_product += subgradient[i] * direction[i]; }
        double cos_angle = safeDiv(dot_product, safeMult(norm_in_out, subgradient_norm));

        // 3. Update beta based on angle
        double beta = (nb_misprices > 0) ? 0.0 : std::max(0.0, cos_angle);

        // 4. Combine directions with proper normalization
        for (size_t i = 0; i < n; ++i) {
            direction[i] =
                safeMult(beta / subgradient_norm, subgradient[i]) + safeMult((1.0 - beta) / norm_in_out, direction[i]);
        }

        // 5. Normalize combined direction
        double norm_direction = 0.0;
        for (size_t i = 0; i < n; ++i) { norm_direction += direction[i] * direction[i]; }
        norm_direction = std::sqrt(norm_direction + EPSILON);

        // 6. Compute adaptive step size
        double wentges_step = std::min(norm_in_out, subgradient_norm) * cur_alpha;

        // 7. Update separation point
        DualSolution new_duals(n);
        for (size_t i = 0; i < n; ++i) {
            new_duals[i] = safeAdd(cur_stab_center[i], safeMult(wentges_step / norm_direction, direction[i]));
            new_duals[i] = std::max(0.0, new_duals[i]);
        }

        smooth_dual_sol = new_duals;
        duals_sep       = new_duals;
        return new_duals;
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
     */
    bool dynamic_alpha_schedule(const ModelData &dados) {
        const size_t n = cur_stab_center.size();

        // 1. Check if stabilization is needed at all
        double relative_distance = norm(smooth_dual_sol, cur_stab_center) / (std::abs(lp_obj) + EPSILON);
        if (relative_distance < NORM_TOLERANCE) {
            alpha = 0.0;
            return false;
        }

        // 2. Calculate and normalize the in-out direction
        std::vector<double> direction(n);
        double              direction_norm = 0.0;

        for (size_t i = 0; i < n; i++) {
            direction[i] = smooth_dual_sol[i] - cur_stab_center[i];
            direction_norm += direction[i] * direction[i];
        }
        direction_norm = std::sqrt(direction_norm + EPSILON);

        if (direction_norm < EPSILON || subgradient_norm < EPSILON) { return false; }

        // 3. Normalize both vectors
        std::vector<double> normalized_direction(n);
        std::vector<double> normalized_subgradient(n);

        for (size_t i = 0; i < n; i++) {
            normalized_direction[i]   = direction[i] / direction_norm;
            normalized_subgradient[i] = subgradient[i] / subgradient_norm;
        }

        // 4. Calculate cosine of angle between normalized vectors
        double cos_angle = 0.0;
        for (size_t i = 0; i < n; i++) { cos_angle += normalized_direction[i] * normalized_subgradient[i]; }

        // 5. Logic for alpha adjustment
        // If cos_angle < 0: vectors point in opposite directions -> increase alpha
        // If cos_angle > 0: vectors point in similar directions -> decrease alpha
        return cos_angle < 0;
    }

    void update_subgradient(const ModelData &dados, const DualSolution &nodeDuals,
                            const std::vector<Label *> &best_pricing_cols) {
        size_t number_of_rows = nodeDuals.size();

        // Initialize new_rows to track column contributions
        new_rows.assign(number_of_rows, 0.0);

        // Step 1: Calculate new_rows from columns brought by pricing
        auto best_pricing_col = best_pricing_cols[0];

        // Update rows based on column contributions
        for (const auto &node : best_pricing_col->nodes_covered) {
            // Skip artificial nodes (0 and last node)
            if (node > 0 && node != N_SIZE - 1) { new_rows[node - 1] += 1.0; }
        }

        // Step 2: Calculate the subgradient based on constraint types
        subgradient.assign(number_of_rows, 0.0);

        /*
                for (size_t row_id = 0; row_id < number_of_rows; ++row_id) {
                    // For each constraint type, calculate appropriate component as per paper:
                    // g = min(0, b - ax) for ≤ constraints
                    // g = max(0, b - ax) for ≥ constraints
                    // g = (b - ax) for = constraints
                    if (dados.sense[row_id] == '<') {
                        // ax ≤ b  =>  b - ax ≥ 0
                        subgradient[row_id] = std::min(0.0, dados.b[row_id] - numK * new_rows[row_id]);
                    } else if (dados.sense[row_id] == '>') {
                        // ax ≥ b  =>  b - ax ≤ 0
                        subgradient[row_id] = std::max(0.0, dados.b[row_id] - numK * new_rows[row_id]);
                    } else { // dados.sense[row_id] == '='
                        // ax = b
                        subgradient[row_id] = dados.b[row_id] - numK * new_rows[row_id];
                    }
                }
        */
        // get column of the most reduced cost
        std::vector<double> most_reduced_cost_column = new_rows;

        // Update subgradient with most reduced cost column
        // std::vector<double> subgradient(pi_out.size());
        std::transform(dados.b.begin(), dados.b.end(), most_reduced_cost_column.begin(), subgradient.begin(),
                       [this](double a, double b) { return a - numK * b; }); // num_vehicle

        // Update subgradient norm
        subgradient_norm = norm(subgradient);
    }

    void set_pseudo_dual_bound(double bound) { pseudo_dual_bound = bound; }

    /**
     * @brief Updates the stabilization parameters after the pricing optimization step.
     *
     * This function adjusts the stabilization parameters based on the results of the
     * pricing optimization. It updates the alpha value using a dynamic schedule if
     * there have been no misprices. Additionally, it updates the stabilization center
     * for the next iteration if the current lagrangian gap is smaller than the previous one.
     *
     */
    void update_stabilization_after_pricing_optim(const ModelData &dados, const DualSolution &input_duals,
                                                  const double &lag_gap, std::vector<Label *> best_pricing_cols) {
        // Early exit checks and cycling detection
        static int    no_progress_count = 0;
        static double last_gap          = lag_gap;

        if (std::abs(lag_gap - last_gap) < EPSILON) {
            no_progress_count++;
            if (no_progress_count > 4) {
                stabilization_active = false;
                cleanup();
                return;
            }
        } else {
            no_progress_count = 0;
            last_gap          = lag_gap;
        }

        if (!stabilization_active) return;

        std::vector<double> nodeDuals(input_duals.begin(), input_duals.begin() + sizeDual);

        // Update stabilization parameters if no misprices
        if (nb_misprices == 0) {
            update_subgradient(dados, nodeDuals, best_pricing_cols);
            bool should_increase = dynamic_alpha_schedule(dados);

            constexpr double ALPHA_FACTOR = 0.15;
            if (should_increase) {
                alpha = std::min(0.99, alpha + (1.0 - alpha) * ALPHA_FACTOR);
            } else {
                alpha = std::max(0.1, alpha * (1.0 - ALPHA_FACTOR));
            }

            if (!std::isnan(alpha) && !std::isinf(alpha)) { base_alpha = alpha; }
        }

        // Get current stabilized solution
        DualSolution stab_sol = smooth_dual_sol;

        // Update multi-point manager
        mp_manager.updatePool(stab_sol, -lag_gap); // Negative because we're maximizing

        // Calculate relative improvement for stability center update
        double relative_improvement    = std::abs(lag_gap - lag_gap_prev) / (std::abs(lag_gap_prev) + EPSILON);
        bool   significant_improvement = relative_improvement > mp_manager.MIN_IMPROVEMENT;

        if (significant_improvement) {
            stab_center_for_next_iteration = stab_sol;
        } else {
            // Conservative update using weighted average with multi-point prediction
            double conservative_weight = 0.2;
            stab_center_for_next_iteration.resize(sizeDual);

            DualSolution mp_sol = mp_manager.getWeightedSolution();

            for (size_t i = 0; i < sizeDual; i++) {
                stab_center_for_next_iteration[i] = 0.5 * cur_stab_center[i] + conservative_weight * stab_sol[i] +
                                                    (0.5 - conservative_weight) * mp_sol[i];
            }
        }

        // Update metrics
        lag_gap_prev = lag_gap;

        // Safety checks
        if (std::isnan(alpha) || std::isinf(alpha)) {
            stabilization_active = false;
            cleanup();
        }
    }

    DualSolution getStabDualSolAdvanceHybrid(const DualSolution &input_duals) {
        // Constants and initialization
        constexpr double EPSILON = 1e-12;
        DualSolution     nodeDuals(input_duals.begin(), input_duals.begin() + sizeDual);

        // Base cases
        if (cur_stab_center.empty() || subgradient.empty()) { return getStabDualSol(input_duals); }

        const size_t n = nodeDuals.size();

        // Get multi-point prediction
        DualSolution mp_sol = mp_manager.getWeightedSolution();

        // Calculate directional solution
        std::vector<double> direction(n);
        double              norm_in_out = 0.0;

        // Use multi-point solution to adjust direction
        double dir_weight = mp_manager.computeAdaptiveWeight(nodeDuals, mp_sol, subgradient, subgradient_norm);

        for (size_t i = 0; i < n; ++i) {
            direction[i] = (1.0 - dir_weight) * (nodeDuals[i] - cur_stab_center[i]) +
                           dir_weight * (mp_sol[i] - cur_stab_center[i]);
            norm_in_out += direction[i] * direction[i];
        }
        norm_in_out = std::sqrt(norm_in_out + EPSILON);

        if (norm_in_out < EPSILON || subgradient_norm < EPSILON) { return getStabDualSol(input_duals); }

        // Compute angle with subgradient
        double dot_product = 0.0;
        for (size_t i = 0; i < n; ++i) { dot_product += subgradient[i] * direction[i]; }
        double cos_angle = safeDiv(dot_product, safeMult(norm_in_out, subgradient_norm));

        // Update beta based on angle
        beta = nb_misprices > 0 ? 0.0 : std::max(0.0, cos_angle);

        // Combine directions
        double norm_direction = 0.0;
        for (size_t i = 0; i < n; ++i) {
            direction[i] = safeMult(beta, subgradient[i]) + safeMult(1.0 - beta, direction[i]);
            norm_direction += direction[i] * direction[i];
        }
        norm_direction = std::sqrt(norm_direction + EPSILON);

        // Compute Wentges step
        double wentges_step = norm_in_out * cur_alpha;

        // Compute new duals
        DualSolution new_duals(n);
        for (size_t i = 0; i < n; ++i) {
            new_duals[i] = safeAdd(cur_stab_center[i], safeMult(wentges_step / norm_direction, direction[i]));
            new_duals[i] = std::max(0.0, new_duals[i]);
        }

        smooth_dual_sol = new_duals;
        duals_sep       = new_duals;
        return new_duals;
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
        return !stabilization_active ||
               (mp_manager.getMetrics().last_improvement < mp_manager.FAST_CONV_THRESHOLD &&
                mp_manager.getMetrics().stagnant_iterations >= MultiPointManager::Metrics::MAX_STAGNANT);
    }

    void cleanup() {
        mp_manager.clear();
        stab_constraint_values.clear();
        smooth_dual_sol.clear();
        subgradient.clear();
        duals_sep.clear();
        beta  = 0.0;
        alpha = base_alpha;
    }

    void define_smooth_dual_sol(const DualSolution &nodeDuals) {
        smooth_dual_sol.assign(nodeDuals.begin(), nodeDuals.begin() + sizeDual);
    }

    void updateNumK(int numK) { this->numK = numK; }
};
