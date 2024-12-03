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

    // Structure for stability points
    struct StabilityPoint {
        DualSolution duals;
        double       objective_value;
        int          age;
        StabilityPoint() : objective_value(0.0), age(0) {}

        StabilityPoint(const DualSolution &d, double obj) : duals(d), objective_value(obj), age(0) {}
    };

    std::vector<StabilityPoint> stability_points;
    static constexpr size_t     MAX_POINTS      = 5;
    static constexpr size_t     MAX_AGE         = 5;
    static constexpr double     MIN_IMPROVEMENT = 1e-6;

    // Track stabilization constraints
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
        // Constants and initialization
        constexpr double EPSILON = 1e-12;
        DualSolution     nodeDuals(input_duals.begin(), input_duals.begin() + sizeDual);

        // Base cases
        if (cur_stab_center.empty() || subgradient.empty() || duals_sep.empty()) {
            return getStabDualSol(input_duals); // Fall back to basic smoothing
        }

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
        if (nb_misprices > 0) {
            beta = 0.0; // No directional component during mispricing
        } else {
            beta = std::max(0.0, cos_angle); // Only use positive correlation
        }

        // 4. Combine directions (in original space, not normalized)
        for (size_t i = 0; i < n; ++i) {
            direction[i] = safeMult(beta, subgradient[i]) + safeMult(1.0 - beta, direction[i]);
        }

        // 5. Normalize combined direction
        double norm_direction = 0.0;
        for (size_t i = 0; i < n; ++i) { norm_direction += direction[i] * direction[i]; }
        norm_direction = std::sqrt(norm_direction + EPSILON);

        // 6. Compute Wentges step and apply combined direction
        double wentges_step = norm_in_out * cur_alpha;

        // 7. Update separation point
        DualSolution new_duals(n);
        for (size_t i = 0; i < n; ++i) {
            // Take step from stability center
            new_duals[i] = safeAdd(cur_stab_center[i], safeMult(wentges_step / norm_direction, direction[i]));
            // Ensure dual feasibility
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
     * @param dados The model data containing problem-specific information.
     * @param nodeDuals The dual solution vector.
     * @param best_pricing_cols A vector of pointers to the best pricing columns.
     * @return The updated alpha value based on the dynamic schedule.
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
     * @param dados The model data containing relevant information for the optimization.
     * @param nodeDuals The dual solution obtained from the pricing optimization.
     * @param lag_gap The current lagrangian gap.
     * @param best_pricing_cols A vector of pointers to the best pricing columns.
     */
    void update_stabilization_after_pricing_optim(const ModelData &dados, const DualSolution &input_duals,
                                                  const double &lag_gap, std::vector<Label *> best_pricing_cols) {
        // Detect cycling - if we've made no progress for too long
        static int    no_progress_count = 0;
        static double last_gap          = lag_gap;

        if (std::abs(lag_gap - last_gap) < EPSILON) {
            no_progress_count++;
            if (no_progress_count > 10) { // Adjust threshold as needed
                stabilization_active = false;
                return;
            }
        } else {
            no_progress_count = 0;
            last_gap          = lag_gap;
        }

        // Only proceed with stabilization if it's still active
        if (!stabilization_active) { return; }

        // Rest of the existing update logic...
        std::vector<double> nodeDuals(input_duals.begin(), input_duals.begin() + sizeDual);

        if (nb_misprices == 0) {
            update_subgradient(dados, nodeDuals, best_pricing_cols);
            bool should_increase = dynamic_alpha_schedule(dados);

            constexpr double ALPHA_FACTOR = 0.1;
            constexpr double MAX_ALPHA    = 0.99;
            constexpr double MIN_ALPHA    = 0.0;

            if (should_increase) {
                alpha = std::min(MAX_ALPHA, base_alpha + (1.0 - base_alpha) * ALPHA_FACTOR);
            } else {
                alpha = std::max(MIN_ALPHA, base_alpha - ALPHA_FACTOR);
            }

            if (!std::isnan(alpha) && !std::isinf(alpha)) { base_alpha = alpha; }
        }

        // For now, let's disable multi-point and use only directional
        DualSolution stab_sol = smooth_dual_sol;

        // Update based on gap
        double gap_diff = lag_gap - lag_gap_prev;
        if (gap_diff < -EPSILON) {
            // Improvement
            stab_center_for_next_iteration = stab_sol;
        } else {
            // No improvement - conservative update
            double weight = 0.3;
            stab_center_for_next_iteration.resize(sizeDual);
            for (size_t i = 0; i < sizeDual; i++) {
                stab_center_for_next_iteration[i] = weight * stab_sol[i] + (1.0 - weight) * cur_stab_center[i];
            }
        }

        smooth_dual_sol = stab_sol;
        lag_gap_prev    = lag_gap;

        // Safety check
        if (std::isnan(alpha) || std::isinf(alpha)) {
            stabilization_active = false;
            alpha                = 0.0;
            base_alpha           = 0.0;
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
        // Exit if:
        // 1. Alpha is effectively zero
        // 2. No progress is being made (cycling detection)
        // 3. Stabilization is explicitly deactivated
        if (base_alpha < EPSILON || !stabilization_active) { return true; }
        return false;
    }

    void define_smooth_dual_sol(const DualSolution &nodeDuals) {
        smooth_dual_sol.assign(nodeDuals.begin(), nodeDuals.begin() + sizeDual);
    }

    void updateNumK(int numK) { this->numK = numK; }
    // Main hybrid stabilization method combining directional and multi-point
    DualSolution getHybridStabDualSol(const DualSolution &input_duals) {
        try {
            const size_t n = sizeDual;
            if (n == 0 || input_duals.size() < n) { return input_duals; }

            // Get directional solution
            DualSolution dir_sol = getStabDualSolAdvanced(input_duals);

            // Use only directional if not enough stability points
            if (stability_points.size() < 2) {
                smooth_dual_sol = dir_sol;
                return dir_sol;
            }

            // Get multi-point solution
            DualSolution mp_sol = getMultiPointSol(n);

            // Ensure feasibility of multi-point solution
            for (double &val : mp_sol) { val = std::max(0.0, val); }

            // Calculate weight with bounds checking
            double mp_weight = calculateMultiPointWeight(dir_sol, mp_sol);
            mp_weight        = std::max(0.0, std::min(1.0, mp_weight));

            // Combine solutions
            DualSolution result(n);
            for (size_t i = 0; i < n; i++) {
                result[i] = std::max(0.0, mp_weight * mp_sol[i] + (1.0 - mp_weight) * dir_sol[i]);
            }

            smooth_dual_sol = result;
            return result;
        } catch (const std::exception &e) {
            // Fallback to directional on any error
            return getStabDualSolAdvanced(input_duals);
        }
    }
    // Update stability points with new solution
    void updateStabilityPoints(const DualSolution &new_point, double obj_value) {
        // Check if point provides meaningful improvement
        bool should_add = true;
        for (const auto &point : stability_points) {
            if (norm(point.duals, new_point) < EPSILON ||
                std::abs(point.objective_value - obj_value) < MIN_IMPROVEMENT) {
                should_add = false;
                break;
            }
        }

        if (should_add) {
            // Add new point
            stability_points.emplace_back(new_point, obj_value);

            // Age existing points
            for (auto &point : stability_points) { point.age++; }

            // Remove old points
            stability_points.erase(std::remove_if(stability_points.begin(), stability_points.end(),
                                                  [this](const auto &p) { return p.age >= MAX_AGE; }),
                                   stability_points.end());

            // Keep only best points if we exceed maximum
            if (stability_points.size() > MAX_POINTS) {
                // Sort by objective value (descending)
                std::sort(stability_points.begin(), stability_points.end(),
                          [](const auto &a, const auto &b) { return a.objective_value > b.objective_value; });
                stability_points.resize(MAX_POINTS);
            }
        }
    }

private:
    // Helper method to compute multi-point solution
    DualSolution getMultiPointSol(size_t n) const {
        DualSolution result(n, 0.0);
        double       total_weight = 0.0;

        // Compute age-based weights
        for (const auto &point : stability_points) {
            double weight = std::max(0.0, 1.0 - static_cast<double>(point.age) / MAX_AGE);
            total_weight += weight;

            for (size_t i = 0; i < n; i++) { result[i] += weight * point.duals[i]; }
        }

        // Normalize
        if (total_weight > EPSILON) {
            for (double &val : result) { val /= total_weight; }
        }

        return result;
    }

    // Helper method to compute weight between multi-point and directional solutions
    double calculateMultiPointWeight(const DualSolution &dir_sol, const DualSolution &mp_sol) const {
        if (subgradient.empty() || subgradient_norm < EPSILON) {
            return 0.5; // Default to equal weight if no subgradient info
        }

        // Compute direction between solutions
        std::vector<double> diff_dir(dir_sol.size());
        double              diff_norm = 0.0;

        for (size_t i = 0; i < dir_sol.size(); i++) {
            diff_dir[i] = mp_sol[i] - dir_sol[i];
            diff_norm += diff_dir[i] * diff_dir[i];
        }
        diff_norm = std::sqrt(diff_norm + EPSILON);

        if (diff_norm < EPSILON) {
            return 0.5; // Solutions are very close
        }

        // Compute angle with subgradient
        double dot_product = 0.0;
        for (size_t i = 0; i < dir_sol.size(); i++) {
            dot_product += (diff_dir[i] / diff_norm) * (subgradient[i] / subgradient_norm);
        }

        // Adjust weight based on angle
        if (dot_product < -0.5) {
            return 0.7; // Multi-point solution appears better
        } else if (dot_product > 0.5) {
            return 0.3; // Directional solution appears better
        }
        return 0.5; // No strong preference
    }

    void cleanup() {
        stability_points.clear();
        stab_constraint_values.clear();
        smooth_dual_sol.clear();
        subgradient.clear();
        duals_sep.clear();
        beta  = 0.0;
        alpha = base_alpha;
    }
};
