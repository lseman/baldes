/**
 * @file Stabilization.h
 * @brief Defines the Stabilization class for dual stabilization in column
 *
 */

#pragma once

#include "core/Definitions.h"
#include "core/Pools.h"

#ifdef GUROBI
#include "gurobi_c++.h"
#include "gurobi_c.h"
#endif

#include <algorithm> // For std::transform
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric> // For std::iota
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @class Stabilization
 * @brief A class to handle stabilization in optimization problems.
 *
 * This class implements various methods to manage and compute stabilization
 * parameters and dual solutions in optimization problems.
 */
class Stabilization {
public:
    static constexpr double kAlphaMin      = 0.02;
    static constexpr double kAlphaMax      = 0.90;
    static constexpr double kAlphaEps      = 1e-3;
    static constexpr double kNormTolerance = 1e-4;
    static constexpr double kBoundTolerance = 1e-9;

    double alpha; // Current alpha parameter
    int    t;     // Iteration counter

    double       base_alpha;                     // "global" alpha parameter
    double       cur_alpha;                      // alpha parameter during the current misprice sequence
    int          nb_misprices = 0;               // number of misprices during the current misprice sequence
    double       pseudo_dual_bound;              // pseudo dual bound, may be non-valid
    double       valid_dual_bound;               // valid dual bound
    DualSolution cur_stab_center;                // current stability center
    DualSolution stab_center_for_next_iteration; // stability center for the
                                                 // next iteration

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
    DualSolution        subgradient; // subgradient at the last separation point
    std::vector<double> new_rows;

    // Subgradient at the stability center. Directional smoothing uses this
    // one: pi_g = pi_in + g_in/||g_in|| * ||pi_out - pi_in||. It moves only
    // when the center moves.
    DualSolution subgradient_in;
    double       subgradient_in_norm = 0.0;
    DualSolution pending_subgradient_in;
    double       pending_subgradient_in_norm = 0.0;

    // Set by set_pricing_subgradient() for the current round; otherwise the
    // route-count estimate in update_subgradient() is used.
    bool subgradient_from_caller = false;

    // Lagrangian bound at the current stability center; the center only
    // moves to a separation point whose bound beats it (Wentges).
    double center_bound = -std::numeric_limits<double>::infinity();

    int numK = 10;

    int sizeDual;

    double lp_obj = 0.0;

    ReducedCostResult   rc;
    std::vector<double> stab_constraint_values;
    std::vector<char>   constraint_senses;

    bool cut_added = false;

    void align_with_master_duals(const DualSolution &master_duals) {
        const std::size_t new_size = master_duals.size();

        auto align = [&](DualSolution &duals) {
            const std::size_t old_size = duals.size();
            duals.resize(new_size);
            for (std::size_t i = old_size; i < new_size; ++i) { duals[i] = master_duals[i]; }
        };

        align(cur_stab_center);
        align(smooth_dual_sol);
        if (!duals_sep.empty()) align(duals_sep);

        if (subgradient.size() != new_size) {
            subgradient.clear();
            subgradient_norm = 0.0;
            beta             = 0.0;
        }
        if (subgradient_in.size() != new_size) {
            subgradient_in.clear();
            subgradient_in_norm = 0.0;
        }
        if (pending_subgradient_in.size() != new_size) {
            pending_subgradient_in.clear();
            pending_subgradient_in_norm = 0.0;
        }

        sizeDual = static_cast<int>(new_size);
    }

    void project_onto_dual_domain(DualSolution &duals) const {
        const std::size_t count = std::min(duals.size(), constraint_senses.size());
        for (std::size_t i = 0; i < count; ++i) {
            // BALDES solves a minimization master. Its '<' row duals are
            // non-positive, '>' row duals are non-negative, and equality-row
            // duals are unrestricted.
            if (constraint_senses[i] == '<') {
                duals[i] = std::min(0.0, duals[i]);
            } else if (constraint_senses[i] == '>') {
                duals[i] = std::max(0.0, duals[i]);
            }
        }
    }

    void update_stabilization_after_misprice() {
        nb_misprices++;
        cur_alpha = _misprice_schedule(nb_misprices, base_alpha);
        alpha     = cur_alpha;
        // As proposed in the paper for mis-pricing sequences, disable
        // directional smoothing after a misprice.
        beta = 0.0;
    }

    void update_stabilization_after_iter(const DualSolution &new_center) {
        if (!new_center.empty() && cur_stab_center.empty()) { cur_stab_center = new_center; }
        if (!stab_center_for_next_iteration.empty()) {
            cur_stab_center = stab_center_for_next_iteration;
            stab_center_for_next_iteration.clear();
            // The center moved to a separation point: its subgradient is the
            // one computed there.
            subgradient_in      = std::move(pending_subgradient_in);
            subgradient_in_norm = pending_subgradient_in_norm;
            pending_subgradient_in.clear();
            pending_subgradient_in_norm = 0.0;
        }
    }

    bool update_stabilization_after_master_optim(const DualSolution &new_center) {
        // New cuts change the dual problem, so bounds at the old center are
        // no longer comparable.
        if (cut_added) {
            center_bound = -std::numeric_limits<double>::infinity();
            cut_added    = false;
        }
        nb_misprices = 0;
        cur_alpha    = std::clamp(base_alpha, kAlphaMin, kAlphaMax);
        alpha        = cur_alpha;

        if (cur_stab_center.empty()) {
            cur_stab_center = new_center;
            smooth_dual_sol = new_center;
            sizeDual        = static_cast<int>(new_center.size());
            return false;
        }
        align_with_master_duals(new_center);
        return cur_alpha > 0;
    }

    void reset_misprices() { nb_misprices = 0; }

    double _misprice_schedule(int nb_misprices, double base_alpha) {
        // Table 1 (left): alpha_tilde = [1 - k * (1 - alpha)]_+
        const int k     = std::max(1, nb_misprices);
        double    alpha = std::max(0.0, 1.0 - k * (1.0 - base_alpha));
        if (nb_misprices > 20 || alpha <= kAlphaEps) {
            alpha = 0.0; // Deactivate stabilization
        }
        duals_in = duals_sep;
        return alpha;
    }

    Stabilization(double base_alpha, DualSolution &mast_dual_sol)
        : alpha(std::clamp(base_alpha, kAlphaMin, kAlphaMax)), t(0),
          base_alpha(std::clamp(base_alpha, kAlphaMin, kAlphaMax)),
          cur_alpha(std::clamp(base_alpha, kAlphaMin, kAlphaMax)), nb_misprices(0), cur_stab_center(mast_dual_sol) {
        pseudo_dual_bound = std::numeric_limits<double>::infinity();
        valid_dual_bound  = std::numeric_limits<double>::infinity();
        beta              = 0.0;
        sizeDual          = mast_dual_sol.size();
        smooth_dual_sol   = mast_dual_sol;
    }

    DualSolution getStabDualSol(const DualSolution &input_duals) {
        if (input_duals.empty()) { return input_duals; }
        align_with_master_duals(input_duals);
        DualSolution pi_out = input_duals;
        if (cur_stab_center.empty()) { return pi_out; }
        if (cur_alpha <= 0.0) {
            smooth_dual_sol = pi_out;
            duals_sep       = pi_out;
            return pi_out;
        }

        // Non-directional smoothing:
        // pi_tilde = alpha * pi_in + (1-alpha) * pi_out
        const size_t n = pi_out.size();
        DualSolution pi_tilde(n);
        for (size_t i = 0; i < n; ++i) { pi_tilde[i] = cur_alpha * cur_stab_center[i] + (1.0 - cur_alpha) * pi_out[i]; }

        // If directional components are unavailable, return the convex
        // combination. Since both endpoints are dual-feasible, the combination
        // respects each row's dual domain.
        // Directional smoothing needs the subgradient at the center; beta is
        // the cosine between g_in and pi_out - pi_in, and it stays off during a misprice
        // sequence (beta == 0 there).
        if (beta > 0.0 && subgradient_in.size() == n && subgradient_in_norm > EPSILON) {
            double dot = 0.0, dir_sq = 0.0;
            for (size_t i = 0; i < n; ++i) {
                const double d = pi_out[i] - cur_stab_center[i];
                dot += d * subgradient_in[i];
                dir_sq += d * d;
            }
            const double dir_norm = std::sqrt(dir_sq);
            beta = dir_norm > EPSILON ? std::clamp(dot / (dir_norm * subgradient_in_norm), 0.0, 1.0) : 0.0;
        }
        if (subgradient_in.size() != n || subgradient_in_norm <= EPSILON || beta <= 0.0) {
            smooth_dual_sol = pi_tilde;
            duals_sep       = pi_tilde;
            return pi_tilde;
        }

        // Directional smoothing (Table 2):
        // pi_g = pi_in + (g_in / ||g_in||) * ||pi_out - pi_in||
        // rho = beta*pi_g + (1-beta)*pi_out
        // pi_sep = (pi_in + ||pi_tilde-pi_in||/||rho-pi_in|| * (rho-pi_in))_+
        const double norm_in_out = std::sqrt(std::inner_product(pi_out.begin(), pi_out.end(), cur_stab_center.begin(),
                                                                0.0, std::plus<double>(),
                                                                [](double a, double b) {
                                                                    const double d = a - b;
                                                                    return d * d;
                                                                }) +
                                             EPSILON);

        DualSolution pi_g(n);
        for (size_t i = 0; i < n; ++i) {
            pi_g[i] = cur_stab_center[i] + (subgradient_in[i] / subgradient_in_norm) * norm_in_out;
        }

        DualSolution rho(n);
        for (size_t i = 0; i < n; ++i) { rho[i] = beta * pi_g[i] + (1.0 - beta) * pi_out[i]; }

        const double norm_tilde_in = std::sqrt(std::inner_product(pi_tilde.begin(), pi_tilde.end(),
                                                                  cur_stab_center.begin(), 0.0, std::plus<double>(),
                                                                  [](double a, double b) {
                                                                      const double d = a - b;
                                                                      return d * d;
                                                                  }) +
                                               EPSILON);
        const double norm_rho_in =
            std::sqrt(std::inner_product(rho.begin(), rho.end(), cur_stab_center.begin(), 0.0, std::plus<double>(),
                                         [](double a, double b) {
                                             const double d = a - b;
                                             return d * d;
                                         }) +
                      EPSILON);

        DualSolution pi_sep(n);
        for (size_t i = 0; i < n; ++i) {
            pi_sep[i] = cur_stab_center[i] + (norm_tilde_in / norm_rho_in) * (rho[i] - cur_stab_center[i]);
        }
        project_onto_dual_domain(pi_sep);
        smooth_dual_sol = pi_sep;
        duals_sep       = pi_sep;
        return pi_sep;
    }

    inline double norm(const std::vector<double> &vector) const {
        return std::sqrt(std::inner_product(vector.begin(), vector.end(), vector.begin(), 0.0));
    }

    inline double norm(const std::vector<double> &vector_1, const std::vector<double> &vector_2) const {
        if (vector_1.size() != vector_2.size()) {
            throw std::invalid_argument("Cannot compute a norm for vectors with different dimensions");
        }
        return std::sqrt(std::inner_product(vector_1.begin(), vector_1.end(), vector_2.begin(), 0.0,
                                            std::plus<double>(), [](double a, double b) {
                                                const double difference = a - b;
                                                return difference * difference;
                                            }));
    }

    DualSolution getStabDualSolAdvanced(const DualSolution &input_duals) { return getStabDualSol(input_duals); }

    static constexpr double EPSILON = 1e-12;

    bool dynamic_alpha_schedule(const ModelData &dados) {
        constexpr double DOT_TOLERANCE = 1e-3;
        const size_t     n             = cur_stab_center.size();

        // Compute relative distance: ||smooth_dual_sol - cur_stab_center|| /
        // |lp_obj|
        double rel_distance = norm(smooth_dual_sol, cur_stab_center) / (std::abs(lp_obj) + EPSILON);
        if (rel_distance < kNormTolerance) {
            alpha = cur_alpha = 0.0;
            return false;
        }

        // Compute the difference vector (direction = smooth_dual_sol -
        // cur_stab_center)
        std::vector<double> direction(n);
        for (size_t i = 0; i < n; ++i) { direction[i] = smooth_dual_sol[i] - cur_stab_center[i]; }

        // Compute norm of direction vector using inner_product
        double dir_norm =
            std::sqrt(std::inner_product(direction.begin(), direction.end(), direction.begin(), 0.0) + EPSILON);
        if (dir_norm < EPSILON || subgradient_norm < EPSILON) { return false; }

        // Normalize the direction and the subgradient vectors.
        std::vector<double> normalized_direction(n);
        std::vector<double> normalized_subgradient(n);
        for (size_t i = 0; i < n; ++i) {
            normalized_direction[i]   = direction[i] / dir_norm;
            normalized_subgradient[i] = subgradient[i] / subgradient_norm;
        }

        // Compute the cosine of the angle between normalized_direction and
        // normalized_subgradient.
        double cos_angle = std::inner_product(normalized_direction.begin(), normalized_direction.end(),
                                              normalized_subgradient.begin(), 0.0);

        // If the cosine of the angle is very close to zero, then the vectors
        // are nearly orthogonal.
        return cos_angle < DOT_TOLERANCE;
    }

    void update_subgradient(const ModelData &model, const DualSolution &node_duals,
                            const std::vector<Label *> &pricing_columns) {
        const std::size_t number_of_rows = node_duals.size();
        if (model.b.size() != number_of_rows || model.sense.size() != number_of_rows) {
            throw std::invalid_argument("Master RHS (" + std::to_string(model.b.size()) + "), senses (" +
                                        std::to_string(model.sense.size()) + "), and dual solution (" +
                                        std::to_string(number_of_rows) + ") have inconsistent row counts");
        }

        // Without pricing columns the Lagrangian solution is unknown, so
        // there is no direction to adapt alpha or beta with.
        if (pricing_columns.empty()) {
            subgradient.clear();
            subgradient_norm = 0.0;
            return;
        }

        new_rows.assign(number_of_rows, 0.0);
        const std::size_t customer_rows = std::min<std::size_t>(number_of_rows, N_SIZE - 2);
        const std::size_t columns_to_use =
            std::min<std::size_t>(static_cast<std::size_t>(std::max(0, numK)), pricing_columns.size());
        for (std::size_t column_index = 0; column_index < columns_to_use; ++column_index) {
            const Label *column = pricing_columns[column_index];
            if (column == nullptr) continue;
            for (uint16_t node : column->getRoute()) {
                if (node > 0 && node < N_SIZE - 1) {
                    const std::size_t row = static_cast<std::size_t>(node - 1);
                    if (row < customer_rows) new_rows[row] += 1.0;
                }
            }
        }

        // Pricing routes expose customer-row coefficients. Extra cut and
        // branching-row coefficients cannot be reconstructed from a route
        // alone, so leave those directional components at zero.
        subgradient.assign(number_of_rows, 0.0);
        for (std::size_t row = 0; row < customer_rows; ++row) { subgradient[row] = model.b[row] - new_rows[row]; }
        subgradient_norm = norm(subgradient);
    }

    // Exact Lagrangian subgradient at this round's separation point,
    // g = b - U * a(best column) (computed by the caller, which knows every
    // master row). Takes precedence over update_subgradient() for this round.
    void set_pricing_subgradient(DualSolution g) {
        subgradient             = std::move(g);
        subgradient_norm        = norm(subgradient);
        subgradient_from_caller = true;
    }

    void set_pseudo_dual_bound(double bound) { pseudo_dual_bound = bound; }

    int no_progress_count = 0;

    void setObj(double obj) { lp_obj = obj; }

    double    lp_obj_prev           = 0.0;
    const int NO_PROGRESS_THRESHOLD = 50;

    /**
     * Updates the stabilization state after one pricing round.
     *
     * @param master_duals      pi_out that this round's separation point was
     *                          built from (not the duals of a later re-solve).
     * @param lagrangian_bound  Lagrangian bound at the separation point, or
     *                          NaN when pricing was not exact.
     * @param mispriced         true when no column has negative reduced cost
     *                          at pi_out.
     */
    void update_stabilization_after_pricing_optim(const ModelData &dados, const DualSolution &master_duals,
                                                  double lagrangian_bound,
                                                  const std::vector<Label *> &best_pricing_cols,
                                                  bool                        mispriced = false) {
        align_with_master_duals(master_duals);
        constraint_senses = dados.sense;

        // The alpha and beta rules use the subgradient of a successful round;
        // during a misprice sequence alpha follows the misprice schedule.
        if (!mispriced && nb_misprices == 0) {
            if (!subgradient_from_caller) update_subgradient(dados, master_duals, best_pricing_cols);

            // Dynamic alpha schedule (Table 1 right, with the paper's sign
            // error corrected, as in GCG): if g_sep · (pi_out - pi_in) > 0
            // the Lagrangian bound still increases toward pi_out, so the step
            // was too cautious -> fdecr(alpha); otherwise -> fincr(alpha).
            double     g_dot_dir = 0.0;
            const bool has_direction =
                subgradient_norm > EPSILON && subgradient.size() == master_duals.size() && !cur_stab_center.empty();
            if (has_direction) {
                for (size_t i = 0; i < master_duals.size(); ++i) {
                    g_dot_dir += subgradient[i] * (master_duals[i] - cur_stab_center[i]);
                }
            }

            // f_incr / f_decr of the paper.
            auto fincr = [](double a) { return a + (1.0 - a) * 0.1; };
            auto fdecr = [](double a) { return std::max(0.0, a - 0.1); };
            if (has_direction) {
                alpha      = (g_dot_dir > 0.0) ? fdecr(alpha) : fincr(alpha);
                alpha      = std::clamp(alpha, 0.0, kAlphaMax);
                base_alpha = std::clamp(alpha, kAlphaMin, kAlphaMax);
                cur_alpha  = alpha;
            }

            // Directional smoothing is re-enabled after a successful round;
            // getStabDualSol sets beta = cos(g_in, pi_out - pi_in).
            beta = 1.0;
        } else if (mispriced && nb_misprices == 0) {
            // A misprice at the first pricing of the round means the step was
            // too cautious for this center: alpha is lowered as well.
            alpha      = std::max(0.0, alpha - 0.1);
            base_alpha = std::clamp(alpha, kAlphaMin, kAlphaMax);
        }
        subgradient_from_caller = false;

        // Stability center (Wentges): move to the separation point only when
        // its Lagrangian bound beats the center's. Heuristic pricing gives no
        // valid bound, so then the center follows each successful round.
        const DualSolution &separation_point = smooth_dual_sol.empty() ? master_duals : smooth_dual_sol;
        if (std::isfinite(lagrangian_bound)) {
            if (lagrangian_bound > center_bound + kBoundTolerance) {
                center_bound                   = lagrangian_bound;
                stab_center_for_next_iteration = separation_point;
                pending_subgradient_in         = subgradient;
                pending_subgradient_in_norm    = subgradient_norm;
            }
        } else if (!mispriced) {
            stab_center_for_next_iteration = separation_point;
            pending_subgradient_in         = subgradient;
            pending_subgradient_in_norm    = subgradient_norm;
        }

        if (std::isnan(alpha) || std::isinf(alpha)) {
            stabilization_active = false;
            cleanup();
        }
    }

    bool shouldExit() const { return cur_alpha < kAlphaEps; }

    void cleanup() {
        stab_constraint_values.clear();
        smooth_dual_sol.clear();
        subgradient.clear();
        subgradient_in.clear();
        pending_subgradient_in.clear();
        subgradient_norm            = 0.0;
        subgradient_in_norm         = 0.0;
        pending_subgradient_in_norm = 0.0;
        duals_sep.clear();
        beta      = 0.0;
        alpha     = base_alpha;
        cur_alpha = base_alpha;
    }

    bool ipm_active = false;
    void define_smooth_dual_sol(const DualSolution &nodeDuals) {
        align_with_master_duals(nodeDuals);
        smooth_dual_sol = nodeDuals;
        ipm_active      = true;
    }

    void updateNumK(int numK) { this->numK = numK; }

    void clearAlpha() { alpha = cur_alpha = 0.0; }
};
