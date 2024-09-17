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

    /**
     * @brief Increments the misprice counter if the alpha value is positive.
     *
     * This function checks if the alpha value is greater than zero. If it is,
     * the function increments the misprice counter (nb_misprices).
     */
    void add_misprice() {
        if (alpha > 0) { nb_misprices++; }
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
     * @return The calculated alpha value. If the number of misprices is greater than 10 or the calculated alpha is less than or equal to 0.001, stabilization is deactivated and alpha is set to 0.0.
     */
    double _misprice_schedule(int nb_misprices, double base_alpha) {
        double alpha = 1.0 - (nb_misprices + 1) * (1 - base_alpha);
        if (nb_misprices > 10 || alpha <= 1e-3) {
            alpha = 0.0; // Deactivate stabilization
        }
        return alpha;
    }

    // Constructor
    Stabilization(double base_alpha, DualSolution &mast_dual_sol)
        : alpha(base_alpha), t(0), base_alpha(base_alpha), cur_alpha(base_alpha), nb_misprices(0),
          cur_stab_center(mast_dual_sol) {
        pseudo_dual_bound = std::numeric_limits<double>::infinity();
        valid_dual_bound  = std::numeric_limits<double>::infinity();
        beta              = 0.0;
    }

    /**
     * @brief Computes the norm of a subset of elements from a given vector.
     *
     * This function calculates the Euclidean norm (L2 norm) of the elements in the 
     * provided vector that are indexed by the values in the new_rows vector. It 
     * adds a small epsilon value (1e-6) to the result to avoid issues with zero 
     * norms.
     *
     * @param new_rows A vector of indices specifying which elements of the vector 
     *                 to include in the norm calculation.
     * @param vector   The vector containing the elements to be used in the norm 
     *                 calculation.
     * @return The Euclidean norm of the specified elements in the vector.
     */
    double norm(const std::vector<int> &new_rows, const std::vector<double> &vector) const {
        double res = std::transform_reduce(new_rows.begin(), new_rows.end(), 0.0, std::plus<>(),
                                           [&vector](int i) { return vector[i] * vector[i]; });
        return std::sqrt(res + 1e-6);
    }

    /**
     * @brief Computes the Euclidean norm (L2 norm) of the difference between two vectors
     *        for the specified indices.
     *
     * This function calculates the Euclidean norm of the difference between elements
     * of two vectors, `vector_1` and `vector_2`, at the positions specified by `new_rows`.
     * A small constant (1e-6) is added to the result to avoid division by zero or other
     * numerical issues.
     *
     * @param new_rows A vector of indices specifying which elements to consider.
     * @param vector_1 The first vector of double values.
     * @param vector_2 The second vector of double values.
     * @return The Euclidean norm of the difference between the specified elements of
     *         `vector_1` and `vector_2`.
     */
    double norm(const std::vector<int> &new_rows, const std::vector<double> &vector_1,
                const std::vector<double> &vector_2) const {
        double res =
            std::transform_reduce(new_rows.begin(), new_rows.end(), 0.0, std::plus<>(), [&vector_1, &vector_2](int i) {
                return (vector_2[i] - vector_1[i]) * (vector_2[i] - vector_1[i]);
            });
        return std::sqrt(res + 1e-6);
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
    DualSolution getStabDualSol() const {
        DualSolution stab_dual_sol(duals_out.size());
        std::transform(duals_out.begin(), duals_out.end(), duals_in.begin(), stab_dual_sol.begin(),
                       [this](double out, double in) { return cur_alpha * in + (1 - cur_alpha) * out; });
        return stab_dual_sol;
    }

    /**
     * @brief Computes the value of beta based on the cosine of the angle between two vectors.
     *
     * This function calculates the cosine of the angle between two vectors defined by the differences
     * between `duals_out` and `duals_in`, and `duals_g` and `duals_in`. The result is stored in the
     * member variable `beta`.
     *
     * The steps involved are:
     * 1. Calculate the dot product of the vectors (duals_out - duals_in) and (duals_g - duals_in).
     * 2. Calculate the norm (magnitude) of the vector (duals_out - duals_in).
     * 3. Calculate the norm (magnitude) of the vector (duals_g - duals_in).
     * 4. Compute the cosine of the angle between the two vectors using the dot product and norms.
     * 5. Ensure that `beta` is non-negative.
     *
     * @param new_rows A vector of integers representing the indices of the rows to be used in the calculations.
     */
    void compute_b(const std::vector<int> &new_rows) {
        // Calculate the dot product of (duals_out - duals_in) and (duals_g - duals_in)
        double dot_product =
            std::transform_reduce(new_rows.begin(), new_rows.end(), 0.0, std::plus<>(), [this](int row_id) {
                double a = duals_out[row_id] - duals_in[row_id]; // Vector A component
                double b = duals_g[row_id] - duals_in[row_id];   // Vector B component
                return a * b;                                    // Dot product of A and B
            });

        // Calculate the norm of (duals_out - duals_in) i.e., norm of vector A
        double norm_a = std::transform_reduce(new_rows.begin(), new_rows.end(), 0.0, std::plus<>(), [this](int row_id) {
            double a = duals_out[row_id] - duals_in[row_id];
            return a * a;
        });
        norm_a        = std::sqrt(norm_a);

        // Calculate the norm of (duals_g - duals_in) i.e., norm of vector B
        double norm_b = std::transform_reduce(new_rows.begin(), new_rows.end(), 0.0, std::plus<>(), [this](int row_id) {
            double b = duals_g[row_id] - duals_in[row_id];
            return b * b;
        });
        norm_b        = std::sqrt(norm_b);

        // Calculate the cosine of the angle
        if (norm_a > 0 && norm_b > 0) {
            beta = dot_product / (norm_a * norm_b);
        } else {
            beta = 0.0; // Handle the case where one of the norms is zero to avoid division by zero
        }

        beta = std::max(0.0, beta); // Ensure beta is non-negative
    }

    /**
     * @brief Computes the rho values for the given new rows.
     *
     * This function updates the `rho` vector by applying a transformation to each element
     * in the `new_rows` vector. The transformation is defined as a weighted sum of the 
     * corresponding elements in the `duals_g` and `duals_out` vectors, where the weights 
     * are given by `beta` and `1 - beta`, respectively.
     *
     * @param new_rows A vector of integers representing the new row indices for which 
     *                 the rho values need to be computed.
     */
    void compute_rho(const std::vector<int> &new_rows) {
        std::transform(new_rows.begin(), new_rows.end(), rho.begin(),
                       [this](int row_id) { return beta * duals_g[row_id] + (1 - beta) * duals_out[row_id]; });
    }

    /**
     * @brief Computes the subgradient for the given model data and updates the Lagrangian constraint values.
     *
     * This function calculates the subgradient for the provided model data (`dados`) and updates the 
     * Lagrangian constraint values based on the new rows and the sparse matrix representation of the constraints.
     *
     * @param dados The model data containing the constraint information and sparse matrix representation.
     * @param new_rows A vector of indices representing the new rows to be considered.
     * @param lagrangian_constraint_values A vector to be updated with the computed Lagrangian constraint values.
     * @param subgradient A vector to be updated with the computed subgradient values.
     */
    void subgradient_call(const ModelData &dados, const std::vector<int> &new_rows,
                          std::vector<double> &lagrangian_constraint_values, std::vector<double> &subgradient) const {

        auto                number_of_rows = dados.b.size();
        std::vector<double> new_row_lower_bounds(number_of_rows);
        std::vector<double> new_row_upper_bounds(number_of_rows);

        for (int row_id = 0; row_id < number_of_rows; ++row_id) {
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

        // Initialize lagrangian_constraint_values to zero
        std::fill(lagrangian_constraint_values.begin(), lagrangian_constraint_values.end(), 0.0);

        // Iterate over non-zero elements in the sparse matrix A
        for (size_t idx = 0; idx < dados.A_sparse.values.size(); ++idx) {
            int row_id = dados.A_sparse.row_indices[idx];
            int col_id = dados.A_sparse.col_indices[idx];
            lagrangian_constraint_values[row_id] += dados.A_sparse.values[idx] * duals_sep[row_id];
        }

        std::transform(new_rows.begin(), new_rows.end(), subgradient.begin(),
                       [&new_row_upper_bounds, &new_row_lower_bounds, &lagrangian_constraint_values](int row_id) {
                           return std::min(0.0, new_row_upper_bounds[row_id] - lagrangian_constraint_values[row_id]) +
                                  std::max(0.0, new_row_lower_bounds[row_id] - lagrangian_constraint_values[row_id]);
                       });
    }

    /**
     * @brief Executes the stabilization algorithm for dual solutions.
     *
     * This function performs a series of operations to stabilize the dual solutions
     * for a given model. It adjusts the dual solutions based on subgradients and 
     * other parameters to ensure convergence and stability.
     *
     * @param dados The model data containing necessary information for the algorithm.
     * @param jobDuals The initial dual solutions to be stabilized.
     * @return The stabilized dual solutions.
     *
     * The function follows these steps:
     * 1. Initializes or resizes vectors for dual solutions and related parameters.
     * 2. Computes coefficients for updating dual solutions.
     * 3. Updates the dual solutions using the computed coefficients.
     * 4. Computes subgradients and adjusts the stabilization parameters.
     * 5. Returns the stabilized dual solutions.
     */
    DualSolution run(const ModelData &dados, const DualSolution &jobDuals) {
        duals_out = jobDuals; // current stab center

        if (duals_sep.empty()) {
            duals_g.assign(dados.b.size(), 0);
            duals_sep.assign(dados.b.size(), 0);
            rho.assign(dados.b.size(), 0);
        }
        if (duals_sep.size() < duals_out.size()) { duals_sep.resize(duals_out.size(), 0); }
        if (duals_g.size() < duals_out.size()) { duals_g.resize(duals_out.size(), 0); }
        if (rho.size() < duals_out.size()) { rho.resize(duals_out.size(), 0); }
        duals_in = duals_sep;

        // print duals_out size and duals_in size
        int                 number_of_rows = dados.b.size();
        std::vector<double> lagrangian_constraint_values(number_of_rows, 0);
        std::vector<double> subgradient(number_of_rows, 0);
        std::vector<int>    new_rows(number_of_rows);
        std::iota(new_rows.begin(), new_rows.end(), 0);

        auto   duals_tilde = getStabDualSol();
        double coef_g      = norm(new_rows, duals_in, duals_out);
        double coef_g_sub  = norm(new_rows, subgradient);
        coef_g             = coef_g / coef_g_sub;

        if (std::isnan(coef_g)) { coef_g = 0; }

        std::transform(new_rows.begin(), new_rows.end(), duals_g.begin(), [this, &subgradient, coef_g](int row_id) {
            return duals_in[row_id] + coef_g * subgradient[row_id];
        });

        compute_b(new_rows);
        compute_rho(new_rows);

        double coef_sep     = norm(new_rows, duals_in, duals_tilde);
        double coef_sep_den = norm(new_rows, duals_in, rho);
        coef_sep            = coef_sep / coef_sep_den;

        if (std::isnan(coef_sep)) { coef_sep = 0.0; }

        if (coef_sep == 0.0) {
            std::copy(duals_in.begin(), duals_in.end(), duals_sep.begin());
        } else {
            std::transform(new_rows.begin(), new_rows.end(), duals_sep.begin(), [this, coef_sep](int row_id) {
                return std::max(0.0, duals_in[row_id] + coef_sep * (rho[row_id] - duals_in[row_id]));
            });
        }

        subgradient_call(dados, new_rows, lagrangian_constraint_values, subgradient);
        // Calculate the in-sep direction
        std::vector<double> in_sep_direction(duals_sep.size());
        std::transform(duals_sep.begin(), duals_sep.end(), duals_in.begin(), in_sep_direction.begin(),
                       [](double sep, double in) { return sep - in; });
        /*
        // Calculate the norms
        double in_sep_norm      = norm(new_rows, in_sep_direction);
        double subgradient_norm = norm(new_rows, subgradient);

        // Compute the cosine of the angle between in-sep direction and subgradient
        double cos_angle =
            std::inner_product(in_sep_direction.begin(), in_sep_direction.end(), subgradient.begin(), 0.0) /
            (in_sep_norm * subgradient_norm);

        // Check if cos_angle is less than threshold (1e-12), and adjust cur_alpha accordingly
        if (cos_angle < 1e-12) {
            cur_alpha = std::min(0.99, cur_alpha + (1.0 - cur_alpha) * 0.1);
        } else {
            cur_alpha = std::max(0.0, cur_alpha - 0.1);
        }

        cur_alpha = _misprice_schedule(nb_misprices, base_alpha);

        return duals_sep;
*/
        double v = std::transform_reduce(
            new_rows.begin(), new_rows.end(), 0.0, std::plus<>(),
            [&subgradient, this](int row_id) { return subgradient[row_id] * (duals_out[row_id] - duals_in[row_id]); });

        if (v > 0) {
            // inc
            cur_alpha = std::min(0.99, cur_alpha + (1.0 - cur_alpha) * 0.1);

        } else {
            // dec
            cur_alpha = std::max(0.0, cur_alpha - 0.1);
        }
        // std::print(fg(fmt::color::yellow), "alpha: {}\n", cur_alpha);
        cur_alpha = _misprice_schedule(nb_misprices, base_alpha);

        return duals_sep;
    }
};
