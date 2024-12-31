#include "Definitions.h"
#include "PDHG.h"

#define EIGEN_INITIALIZE_MATRICES_BY_ZERO

#include <omp.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

#include "ipm/IPSolver.h"
/**
 * @brief Performs gradient-based equilibration scaling.
 * This scaling method considers both the objective and constraints
 * to produce better-balanced optimality conditions.
 */
ScalingFactors IPSolver::ruiz_scaling(Eigen::SparseMatrix<double> &A,
                                      Eigen::VectorXd &b, Eigen::VectorXd &c,
                                      Eigen::VectorXd &lo,
                                      Eigen::VectorXd &hi) {
    const int m = A.rows();
    const int n = A.cols();
    const int n_bounds = lo.size();

    ScalingFactors factors;
    factors.row_scaling = Eigen::VectorXd::Ones(m);
    factors.col_scaling = Eigen::VectorXd::Ones(n);

    // Parameters
    const int max_iter = 10;
    const double thresh = 1e-3;  // Convergence threshold
    const double alpha = 0.1;    // Step size for updates

    // Store initial norms for reference
    double init_objective_norm = c.norm();
    double init_constraint_norm = b.norm();

    // Compute column norms for A and objective scaling
    Eigen::VectorXd col_norms = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd row_norms = Eigen::VectorXd::Zero(m);

    // Initial norms computation
    for (int k = 0; k < A.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
            double val = std::abs(it.value());
            col_norms(it.col()) += val;
            row_norms(it.row()) += val;
        }
    }

    // Initial scaling to roughly same magnitude
    double mean_norm =
        (col_norms.sum() / (n * m) + std::abs(c.mean()) + std::abs(b.mean())) /
        3.0;
    if (mean_norm > 0) {
        double log_scale = -std::log10(mean_norm);
        double scale = std::pow(10.0, std::round(log_scale));

        // Apply initial scaling
        if (scale != 1.0 && scale > 1e-4 && scale < 1e4) {
            A *= scale;
            b *= scale;
            c *= scale;

            factors.row_scaling *= std::sqrt(scale);
            factors.col_scaling *= std::sqrt(scale);
        }
    }

    // Iterative equilibration
    for (int iter = 0; iter < max_iter; iter++) {
        // Compute gradient of the scaled problem
        Eigen::VectorXd grad_scaling = Eigen::VectorXd::Zero(n);

        // Consider objective contribution
        grad_scaling = c.cwiseAbs();

        // Add constraint contributions
        for (int k = 0; k < A.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it;
                 ++it) {
                grad_scaling(it.col()) += std::abs(it.value());
            }
        }

        // Compute scaling updates
        double max_update = 0.0;
        for (int j = 0; j < n; j++) {
            if (grad_scaling(j) > 0) {
                double target = 1.0;
                double current = grad_scaling(j);
                double log_ratio = std::log(target / current);

                // Limit the update
                double update =
                    std::min(std::max(log_ratio * alpha, -0.1), 0.1);
                double scale = std::exp(update);

                // Apply column scaling
                if (scale != 1.0) {
                    for (Eigen::SparseMatrix<double>::InnerIterator it(A, j);
                         it; ++it) {
                        A.coeffRef(it.row(), j) *= scale;
                    }
                    c(j) *= scale;
                    factors.col_scaling(j) *= scale;

                    max_update = std::max(max_update, std::abs(update));
                }
            }
        }

        // Check convergence
        if (max_update < thresh) break;
    }

    // Scale bounds
    for (int j = 0; j < n_bounds && j < n; ++j) {
        if (std::isfinite(lo(j))) {
            lo(j) *= factors.col_scaling(j);
        }
        if (std::isfinite(hi(j))) {
            hi(j) *= factors.col_scaling(j);
        }
    }

    // Compute and print statistics
    double scaled_objective_norm = c.norm();
    double scaled_constraint_norm = b.norm();

    std::cout << "\nGradient-based scaling statistics:\n";
    std::cout << "Objective norm - Before: " << init_objective_norm
              << ", After: " << scaled_objective_norm << "\n";
    std::cout << "Constraint norm - Before: " << init_constraint_norm
              << ", After: " << scaled_constraint_norm << "\n";
    std::cout << "Column scaling range: [" << factors.col_scaling.minCoeff()
              << ", " << factors.col_scaling.maxCoeff() << "]\n\n";

    return factors;
}

/**
 * @brief Unscales the optimization solution using the scaling factors.
 *
 * This function reverses the scaling applied by ruiz_scaling to recover
 * the solution in the original problem space.
 *
 * @param x Primal solution vector to be unscaled
 * @param lambda Dual solution vector to be unscaled
 * @param factors The scaling factors used in the scaling phase
 */
void IPSolver::unscale_solution(Eigen::VectorXd &x, Eigen::VectorXd &lambda,
                                const ScalingFactors &factors) {
    // Unscale primal solution
    x = x.cwiseQuotient(factors.col_scaling);

    // Unscale dual solution
    lambda = lambda.cwiseQuotient(factors.row_scaling);
}

/**
 * @brief Converts a dense vector to a sparse diagonal matrix.
 *
 * This function takes a dense vector and converts it into a sparse diagonal
 * matrix. The resulting sparse matrix has non-zero values only on its diagonal,
 * where the values are taken from the input vector.
 */
Eigen::SparseMatrix<double> IPSolver::convertToSparseDiagonal(
    const Eigen::VectorXd &vec) {
    Eigen::SparseMatrix<double> mat(vec.size(), vec.size());
    mat = vec.asDiagonal();
    return mat;
}

/**
 * @brief Converts the given linear programming problem to its standard form.
 *
 * This function transforms the input linear programming problem defined by the
 * matrices and vectors A, b, c, lb, ub, and sense into its standard form. The
 * standard form is characterized by having all variables non-negative and all
 * constraints as equalities.
 *
 */
void IPSolver::convert_to_standard_form(
    const Eigen::SparseMatrix<double> &A, const Eigen::VectorXd &b,
    const Eigen::VectorXd &c, const Eigen::VectorXd &lb,
    const Eigen::VectorXd &ub, const Eigen::VectorXd &sense,
    Eigen::SparseMatrix<double> &As, Eigen::VectorXd &bs, Eigen::VectorXd &cs) {
    constexpr double infty = std::numeric_limits<double>::infinity();
    const int n = A.rows();
    const int m = A.cols();

    if (b.size() != n || c.size() != m) {
        fmt::print("b.size(): {}, n: {}, c.size(): {}, m: {}\n", b.size(), n,
                   c.size(), m);
        throw std::invalid_argument(
            "Size of b or c does not match the matrix A dimensions.");
    }

    Eigen::VectorXd lo = lb;
    Eigen::VectorXd hi = ub;

    int n_free = 0, n_ubounds = 0, nv = A.cols();

    // Precompute bound categories
    std::vector<bool> is_free(lo.size()), is_bounded(lo.size()),
        is_upper(lo.size()), is_lower(lo.size());
    for (int i = 0; i < lo.size(); ++i) {
        if (lo[i] == -infty && hi[i] == infty) {
            is_free[i] = true;
            ++n_free;
        } else if (std::isfinite(lo[i]) && std::isfinite(hi[i])) {
            is_bounded[i] = true;
            ++n_ubounds;
        } else if (lo[i] == -infty && std::isfinite(hi[i])) {
            is_upper[i] = true;
        } else if (std::isfinite(lo[i]) && hi[i] == infty) {
            is_lower[i] = true;
        } else {
            throw std::runtime_error("unexpected bounds");
        }
    }

    // Precompute slack variables and reserve memory
    int num_slacks = n - sense.sum();
    cs.resize(c.size() + n_free + num_slacks);
    cs.setZero();
    cs.head(m) = c;

    bs = b;  // Direct assignment

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(A.nonZeros() +
                     num_slacks);  // Reserve space to avoid reallocations

    std::vector<int> ind_ub;
    std::vector<double> val_ub;

    auto lb_extended = lb;
    auto ub_extended = ub;
    lb_extended.conservativeResize(nv + num_slacks + n_free);
    ub_extended.conservativeResize(nv + num_slacks + n_free);
    lb_extended.tail(num_slacks + n_free).setZero();
    ub_extended.tail(num_slacks + n_free).setConstant(1.0);

    int free = 0, ubi = 0;
    for (int j = 0; j < lo.size(); ++j) {
        double l = lo[j];
        double h = hi[j];

        for (Eigen::SparseMatrix<double>::InnerIterator it(A, j); it; ++it) {
            int i = it.row();
            double v = it.value();

            if (is_free[j]) {
                cs[j] = c[j];
                cs[nv + free] = -c[j];
                triplets.emplace_back(i, j, v);
                triplets.emplace_back(i, nv + free, -v);
                lb_extended[nv + free] = lb[j];
                ub_extended[nv + free] = ub[j];
                ++free;
            } else if (is_bounded[j]) {
                cs[j] = c[j];
                bs[i] -= v * l;
                triplets.emplace_back(i, j, v);
                ind_ub.push_back(j);
                val_ub.push_back(h - l);
                ++ubi;
            } else if (is_upper[j]) {
                cs[j] = -c[j];
                bs[i] -= (-v * h);
                triplets.emplace_back(i, j, -v);
            } else if (is_lower[j]) {
                cs[j] = c[j];
                bs[i] -= v * l;
                triplets.emplace_back(i, j, v);
            }
        }
    }

    // Add slack variables
    int slack_counter = 0;
    for (int i = 0; i < sense.size(); ++i) {
        if (sense(i) == 0) {
            // print nv + n_free + slack_counter
            triplets.emplace_back(i, nv + n_free + slack_counter, 1.0);
            lb_extended[nv + n_free + slack_counter] = 0.0;
            ub_extended[nv + n_free + slack_counter] = infty;
            ++slack_counter;
        }
    }

    // Construct sparse matrix As
    As.resize(bs.size(), cs.size());
    As.setFromTriplets(triplets.begin(), triplets.end());
    As.makeCompressed();

    n_slacks = num_slacks;
}

/**
 * @brief Updates the residuals for the interior point method solver.
 *
 * This function calculates and updates the primal residual (rp), upper bound
 * residual (ru), dual residual (rd), and gap residual (rg) along with their
 * norms.
 *
 */
void IPSolver::update_residuals(
    Residuals &res, const Eigen::VectorXd &x, const Eigen::VectorXd &lambda,
    const Eigen::VectorXd &s, const Eigen::VectorXd &v,
    const Eigen::VectorXd &w, const Eigen::SparseMatrix<double> &A,
    const Eigen::VectorXd &b, const Eigen::VectorXd &c,
    const Eigen::VectorXd &ubv, const Eigen::VectorXi &ubi, double tau,
    double kappa) {
    // Pre-compute tau-scaled vectors to avoid repeated multiplications
    const Eigen::VectorXd tau_b = tau * b;
    const Eigen::VectorXd tau_c = tau * c;

    // Calculate primal residual (rp)
    res.rp.noalias() = tau_b - A * x;
    res.rpn = res.rp.norm();

    // Update residual for upper bounds (ru)
    res.ru = -v;  // Direct assignment for better performance

    // Use raw pointers for faster access in the loop
    double *ru_data = res.ru.data();
    const double *ubv_data = ubv.data();
    const double *x_data = x.data();
    const int *ubi_data = ubi.data();
    const int ubi_size = ubi.size();

    // Parallelize the loop for updating ru
    for (int i = 0; i < ubi_size; ++i) {
        const int idx = ubi_data[i];
        ru_data[idx] += tau * ubv_data[i] - x_data[idx];
    }

    // Calculate dual residual (rd)
    res.rd.noalias() = tau_c - A.transpose() * lambda - s;

    // Update rd with w values
    double *rd_data = res.rd.data();
    const double *w_data = w.data();

    // Parallelize the loop for updating rd
    for (int i = 0; i < ubi_size; ++i) {
        const int idx = ubi_data[i];
        rd_data[idx] += w_data[i];
    }

    // Calculate gap residual (rg) using dot products
    const double cx = c.dot(x);
    const double blambda = b.dot(lambda);
    const double wubv = ubv.dot(w);

    res.rg = kappa + cx - blambda + wubv;
    res.rgn = std::abs(res.rg);
}

/**
 * @brief Solves the augmented system for the given right-hand side vectors.
 *
 * This function solves the augmented system using either the augmented approach
 * or the regularized approach based on the preprocessor directive `AUGMENTED`.
 *
 */
void IPSolver::solve_augmented_system(Eigen::VectorXd &dx, Eigen::VectorXd &dy,
                                      SparseSolver &ls,
                                      const Eigen::VectorXd &xi_p,
                                      const Eigen::VectorXd &xi_d) {
    // Set-up right-hand side with preserved order
    Eigen::VectorXd xi(xi_d.size() + xi_p.size());
    xi << xi_d, xi_p;

    // Solve augmented system
    Eigen::VectorXd d = ls.solve(xi);

    // Recover dx, dy in original order
    dx = d.head(xi_d.size());  // Gets the first n elements
    dy = d.tail(xi_p.size());  // Gets the last m elements
}

void IPSolver::solve_augsys(Eigen::VectorXd &delta_x, Eigen::VectorXd &delta_y,
                            Eigen::VectorXd &delta_z, SparseSolver &ls,
                            const Eigen::VectorXd &theta_vw,
                            const Eigen::VectorXi &ubi,
                            const Eigen::VectorXd &xi_p,
                            const Eigen::VectorXd &xi_d,
                            const Eigen::VectorXd &xi_u) {
    // Static allocation for frequently used vectors
    static Eigen::VectorXd xi_d_mod;
    if (xi_d_mod.size() != xi_d.size()) {
        xi_d_mod.resize(xi_d.size());
    }

    // Use direct memory copy instead of assignment
    std::memcpy(xi_d_mod.data(), xi_d.data(), xi_d.size() * sizeof(double));

    // Get raw pointers for faster access
    double *xi_d_data = xi_d_mod.data();
    const double *theta_vw_data = theta_vw.data();
    const double *xi_u_data = xi_u.data();
    const int *ubi_data = ubi.data();
    const int ubi_size = ubi.size();

// Use SIMD vectorization for the update loop
#pragma omp simd
    for (int i = 0; i < ubi_size; ++i) {
        const int idx = ubi_data[i];
        xi_d_data[idx] -= xi_u_data[i] * theta_vw_data[i];
    }

    // Solve augmented system with optimized xi_d_mod
    solve_augmented_system(delta_x, delta_y, ls, xi_p, xi_d_mod);

    // Update delta_z efficiently
    delta_z.resize(ubi_size);
    double *delta_z_data = delta_z.data();
    const double *delta_x_data = delta_x.data();

// Use SIMD vectorization for final update
#pragma omp simd
    for (int i = 0; i < ubi_size; ++i) {
        const int idx = ubi_data[i];
        delta_z_data[i] = (delta_x_data[idx] - xi_u_data[i]) * theta_vw_data[i];
    }
}
/**
 * @brief Solves the Newton system for the interior point method.
 *
 * This function updates the provided solution vectors (Delta_x, Delta_lambda,
 * Delta_w, Delta_s, Delta_v) and scalars (Delta_tau, Delta_kappa) by solving
 * the augmented system using the provided sparse solver.
 *
 */
void IPSolver::solve_newton_system(
    Eigen::VectorXd &Delta_x, Eigen::VectorXd &Delta_lambda,
    Eigen::VectorXd &Delta_w, Eigen::VectorXd &Delta_s,
    Eigen::VectorXd &Delta_v, double &Delta_tau, double &Delta_kappa,
    SparseSolver &ls, const Eigen::VectorXd &theta_vw, const Eigen::VectorXd &b,
    const Eigen::VectorXd &c, const Eigen::VectorXi &ubi,
    const Eigen::VectorXd &ubv, const Eigen::VectorXd &delta_x,
    const Eigen::VectorXd &delta_y, const Eigen::VectorXd &delta_w,
    double delta_0, const Eigen::VectorXd &iter_x,
    const Eigen::VectorXd &iter_lambda, const Eigen::VectorXd &iter_w,
    const Eigen::VectorXd &iter_s, const Eigen::VectorXd &iter_v,
    double iter_tau, double iter_kappa, const Eigen::VectorXd &xi_p,
    const Eigen::VectorXd &xi_u, const Eigen::VectorXd &xi_d, double xi_g,
    const Eigen::VectorXd &xi_xs, const Eigen::VectorXd &xi_vw,
    double xi_tau_kappa) {
    Eigen::VectorXd xi_d_copy =
        xi_d - (xi_xs.array() / iter_x.array()).matrix();
    Eigen::VectorXd xi_u_copy =
        xi_u - (xi_vw.array() / iter_w.array()).matrix();

    // Pre-compute frequently used values
    const double inv_tau = 1.0 / iter_tau;
    const double inv_kappa = 1.0 / iter_kappa;

    // Pre-allocate vectors to avoid reallocations
    static Eigen::VectorXd xi_d_mod;
    if (xi_d_mod.size() != xi_d.size()) {
        xi_d_mod.resize(xi_d.size());
    }

// Use vectorized operations for division
#pragma omp parallel sections
    {
#pragma omp section
        {
            // Compute xi_d_mod = xi_d - xi_xs./iter_x efficiently
            xi_d_mod = xi_d.array() - (xi_xs.array() / iter_x.array());
        }
#pragma omp section
        {
            // Pre-compute xi_u_copy = xi_u - xi_vw./iter_w
            Eigen::VectorXd xi_u_copy =
                xi_u - (xi_vw.array() / iter_w.array()).matrix();

            // Call solve_augsys with pre-computed values
            solve_augsys(Delta_x, Delta_lambda, Delta_w, ls, theta_vw, ubi,
                         xi_p, xi_d_mod, xi_u_copy);
        }
    }

    // Compute Delta_tau using efficient dot products
    const double c_dot_Delta_x = c.dot(Delta_x);
    const double b_dot_Delta_lambda = b.dot(Delta_lambda);
    const double ubv_dot_Delta_w = ubv.dot(Delta_w);

    Delta_tau = (xi_g + xi_tau_kappa * inv_tau + c_dot_Delta_x -
                 b_dot_Delta_lambda + ubv_dot_Delta_w) /
                delta_0;
    Delta_kappa = (xi_tau_kappa - iter_kappa * Delta_tau) * inv_tau;

// Update vectors using vectorized operations
#pragma omp parallel sections
    {
#pragma omp section
        {
            Delta_x.array() += Delta_tau * delta_x.array();
            Delta_lambda.array() += Delta_tau * delta_y.array();
            Delta_w.array() += Delta_tau * delta_w.array();
        }
#pragma omp section
        {
            // Compute Delta_s and Delta_v using vectorized operations
            Delta_s = (xi_xs.array() - iter_s.array() * Delta_x.array()) /
                      iter_x.array();
            Delta_v = (xi_vw.array() - iter_v.array() * Delta_w.array()) /
                      iter_w.array();
        }
    }
}
/**
 * @brief Computes the maximum step size (alpha) for a single direction vector.
 *
 * This function calculates the maximum allowable step size (alpha) such that
 * the updated vector (v + alpha * dv) remains non-negative. It iterates through
 * each element of the input vectors `v` and `dv`, and for each negative element
 * in `dv`, it computes a potential alpha value. The minimum of these potential
 * alpha values is returned as the result.
 *
 */
double IPSolver::max_alpha_single(const Eigen::VectorXd &v,
                                  const Eigen::VectorXd &dv) {
    double alpha = std::numeric_limits<double>::infinity();

    // Get direct access to data
    const double *v_data = v.data();
    const double *dv_data = dv.data();
    const Eigen::Index size = v.size();

    // Process in chunks for better cache utilization
    constexpr Eigen::Index CHUNK_SIZE = 4;
    Eigen::Index i = 0;

    // Process main chunks
    for (; i + CHUNK_SIZE <= size; i += CHUNK_SIZE) {
        if (dv_data[i] < 0) alpha = std::min(alpha, -v_data[i] / dv_data[i]);
        if (dv_data[i + 1] < 0)
            alpha = std::min(alpha, -v_data[i + 1] / dv_data[i + 1]);
        if (dv_data[i + 2] < 0)
            alpha = std::min(alpha, -v_data[i + 2] / dv_data[i + 2]);
        if (dv_data[i + 3] < 0)
            alpha = std::min(alpha, -v_data[i + 3] / dv_data[i + 3]);
    }

    // Handle remaining elements
    for (; i < size; ++i) {
        if (dv_data[i] < 0) {
            alpha = std::min(alpha, -v_data[i] / dv_data[i]);
        }
    }

    return alpha;
}
/**
 * @brief Computes the maximum step size (alpha) that can be taken along the
 * direction of the search vectors.
 *
 * This function calculates the maximum allowable step size (alpha) that can be
 * taken along the direction of the search vectors (dx, dv, ds, dw) without
 * violating certain constraints. It considers the current values of the
 * variables (x, v, s, w) and their respective search directions. Additionally,
 * it takes into account the step sizes for tau and kappa.
 *
 */
double IPSolver::max_alpha(const Eigen::VectorXd &x, const Eigen::VectorXd &dx,
                           const Eigen::VectorXd &v, const Eigen::VectorXd &dv,
                           const Eigen::VectorXd &s, const Eigen::VectorXd &ds,
                           const Eigen::VectorXd &w, const Eigen::VectorXd &dw,
                           double tau, double dtau, double kappa,
                           double dkappa) {
    // Initialize alpha with first scalar check
    double alpha = (dtau < 0) ? (-tau / dtau) : 1.0;

    // Check kappa condition and update alpha if needed
    if (dkappa < 0) {
        double alpha_kappa = -kappa / dkappa;
        if (alpha_kappa < alpha) {
            alpha = alpha_kappa;
        }
    }

    // Efficiently compute and update minimum alpha for each vector pair
    double potential_alpha = max_alpha_single(x, dx);
    if (potential_alpha < alpha) alpha = potential_alpha;

    potential_alpha = max_alpha_single(v, dv);
    if (potential_alpha < alpha) alpha = potential_alpha;

    potential_alpha = max_alpha_single(s, ds);
    if (potential_alpha < alpha) alpha = potential_alpha;

    potential_alpha = max_alpha_single(w, dw);
    if (potential_alpha < alpha) alpha = potential_alpha;

    return alpha;
}

/**
 * @brief Runs the optimization process on the given model data.
 *
 * This function performs an optimization using an interior point method (IPM)
 * on the provided model data. It converts the model data to a standard form,
 * initializes necessary variables, and iteratively solves the optimization
 * problem until convergence or the maximum number of iterations is reached.
 *
 */
void IPSolver::run_optimization(ModelData &model, const double tol) {
    auto componentes = convertToOptimizationData(model);

    Eigen::SparseMatrix<double> As = componentes.As;
    Eigen::VectorXd bs = componentes.bs;
    Eigen::VectorXd cs = componentes.cs;
    Eigen::VectorXd lo = componentes.lo;
    Eigen::VectorXd hi = componentes.hi;
    Eigen::VectorXd sense = componentes.sense;

    int nv_orig = cs.size();

    // Convert to standard form
    Eigen::SparseMatrix<double> A;
    Eigen::VectorXd b;
    Eigen::VectorXd c;
    convert_to_standard_form(As, bs, cs, lo, hi, sense, A, b, c);

    int n = A.cols();
    int m = A.rows();

    // Tolerance and maximum iterations
    int max_iter = 50;

    // Initialize vectors and scalars
    Eigen::VectorXd x = Eigen::VectorXd::Ones(n);
    Eigen::VectorXd lambda = Eigen::VectorXd::Zero(m);
    Eigen::VectorXd s = Eigen::VectorXd::Ones(n);

    warm_start = false;
    n_slacks_old = n_slacks;

    // Initialize ubi and ubv
    Eigen::VectorXi ubi;
    Eigen::VectorXd ubv;
    int count = 0;
    for (int i = 0; i < hi.size(); i++) {
        if (hi[i] != std::numeric_limits<double>::infinity()) {
            count++;
        }
    }

    Eigen::VectorXi tempUbi(count);
    Eigen::VectorXd tempUbv(count);

    double infty = std::numeric_limits<double>::infinity();
    count = 0;
    for (int i = 0; i < hi.size(); i++) {
        if (hi[i] != infty) {
            tempUbi[count] = i;
            tempUbv[count] = hi[i];
            count++;
        }
    }
    ubi = tempUbi;
    ubv = tempUbv;

    Eigen::VectorXd v = Eigen::VectorXd::Ones(ubv.size());
    Eigen::VectorXd w = Eigen::VectorXd::Ones(ubv.size());

    double tau = 1.0;
    double kappa = 1.0;

    Eigen::VectorXd regP = Eigen::VectorXd::Ones(n);
    Eigen::VectorXd regD = Eigen::VectorXd::Ones(m);
    double regG = 1.0;

    ls.reset();
    start_linear_solver(ls, A);

    int nc = A.rows();
    int nv = A.cols();
    int nu = ubi.size();

    Eigen::VectorXd delta_x(nv), delta_y(nc), delta_z(nu);
    Residuals res;

    double r_min = std::sqrt(std::numeric_limits<double>::epsilon());
    int attempt = 0, ncor = 0;
    double _p, _d, _g, mu;
    double alpha, alpha_c, alpha_;
    double beta, gamma, damping, oneMinusAlpha;
    double mu_l, mu_u, taukappa, t0;
    Eigen::VectorXd theta_vw, theta_xs;
    Eigen::VectorXd xi_p, xi_d, xi_u, xi_xs, xi_vw;
    Eigen::VectorXd Delta_x(x.size()), Delta_lambda(lambda.size()),
        Delta_w(w.size()), Delta_s(s.size()), Delta_v(v.size());
    double Delta_tau, Delta_kappa;
    Eigen::VectorXd Delta_x_c(x.size()), Delta_lambda_c(lambda.size()),
        Delta_w_c(w.size()), Delta_s_c(s.size()), Delta_v_c(v.size());
    double Delta_tau_c, Delta_kappa_c;
    Eigen::VectorXd xs, vw, t_xs, t_vw;
    Eigen::ArrayXd t_xs_lower, t_xs_upper, t_vw_lower, t_vw_upper;
    double delta_0, bl_dot_lambda, correction;
    bool saved_interior_solution_bool = false;

    // Adaptive tolerance parameters
    const double initial_tol = 1e-6;  // Initial tolerance
    const double min_tol = 1e-10;     // Minimum tolerance
    const double scale_factor = 0.1;  // Scaling factor for adaptive tolerance

    for (int k = 0; k < max_iter; ++k) {
        ncor = 0;
        beta = 0.1;

        // Batch zero initialization
        delta_x.setZero();
        delta_y.setZero();
        delta_z.setZero();
        Delta_x.setZero();
        Delta_lambda.setZero();
        Delta_w.setZero();
        Delta_s.setZero();
        Delta_v.setZero();
        Delta_tau = Delta_kappa = 0.0;

        // Update residuals without vbv and vbi
        update_residuals(res, x, lambda, s, v, w, A, b, c, ubv, ubi, tau,
                         kappa);
        mu = (tau * kappa + x.dot(s)) / (n + nu + 1.0);

        // Efficient residual calculations
        double rp_norm = res.rp.lpNorm<Eigen::Infinity>();
        double ru_norm = res.ru.lpNorm<Eigen::Infinity>();
        double rd_norm = res.rd.lpNorm<Eigen::Infinity>();
        double b_norm = b.lpNorm<Eigen::Infinity>();
        double ubv_norm = ubv.lpNorm<Eigen::Infinity>();
        double c_norm = c.lpNorm<Eigen::Infinity>();

        // Adaptive tolerance computation
        double adaptive_tol = std::max(min_tol, initial_tol * scale_factor);

        // Combined residual computations
        bl_dot_lambda = b.dot(lambda) - ubv.dot(w);
        _p = std::fmax(rp_norm / (tau * (1.0 + b_norm)),
                       ru_norm / (tau * (1.0 + ubv_norm)));
        _d = rd_norm / (tau * (1.0 + c_norm));
        _g = std::abs(c.dot(x) - bl_dot_lambda) /
             (tau + std::abs(bl_dot_lambda));

        // Save interior solution if residuals are small
        if (!saved_interior_solution_bool &&
            (_d <= adaptive_tol && _g <= adaptive_tol * 2)) {
            save_interior_solution(x, lambda, w, s, v, tau, kappa);
            saved_interior_solution_bool = true;
            warm_start = true;
        }

        // Check convergence with adaptive tolerance
        if (_p <= adaptive_tol && _d <= adaptive_tol && _g <= tol) break;

        // Update adaptive tolerance for the next iteration
        adaptive_tol = std::max(min_tol, adaptive_tol * scale_factor);

        // Efficient scaling computations
        theta_vw = w.cwiseQuotient(v);
        theta_xs = s.cwiseQuotient(x);
        for (int i = 0; i < ubi.size(); ++i) {
            theta_xs[ubi[i]] += theta_vw[i];
        }

        // Update regularizations dynamically
        for (int attempt = 0; attempt < 3; ++attempt) {
            // Adjust regularization based on solver performance
            regP = (regP / 10.0).cwiseMax(r_min);
            regD = (regD / 10.0).cwiseMax(r_min);
            regG = std::max(r_min, regG / 10.0);

            // Update linear solver with new regularization
            if (update_linear_solver(ls, theta_xs, regP, regD) == 0) break;

            // Increase regularization if solver fails
            regP *= 100.0;
            regD *= 100.0;
            regG *= 100.0;
        }

        solve_augsys(delta_x, delta_y, delta_z, ls, theta_vw, ubi, b, c, ubv);
        delta_0 = regG + kappa / tau - delta_x.dot(c) + delta_y.dot(b) -
                  delta_z.dot(ubv);

        // First Newton solve
        solve_newton_system(Delta_x, Delta_lambda, Delta_w, Delta_s, Delta_v,
                            Delta_tau, Delta_kappa, ls, theta_vw, b, c, ubi,
                            ubv, delta_x, delta_y, delta_z, delta_0, x, lambda,
                            w, s, v, tau, kappa, res.rp, res.ru, res.rd, res.rg,
                            -x.cwiseProduct(s), -v.cwiseProduct(w),
                            -tau * kappa);

        alpha = max_alpha(x, Delta_x, v, Delta_v, s, Delta_s, w, Delta_w, tau,
                          Delta_tau, kappa, Delta_kappa);

        oneMinusAlpha = 1.0 - alpha;
        gamma = std::fmax(
            oneMinusAlpha * oneMinusAlpha * std::fmin(beta, oneMinusAlpha),
            0.1);
        damping = 1.0 - gamma;

        // Second Newton solve with damping
        solve_newton_system(
            Delta_x, Delta_lambda, Delta_w, Delta_s, Delta_v, Delta_tau,
            Delta_kappa, ls, theta_vw, b, c, ubi, ubv, delta_x, delta_y,
            delta_z, delta_0, x, lambda, w, s, v, tau, kappa, damping * res.rp,
            damping * res.ru, damping * res.rd, damping * res.rg,
            (-x.cwiseProduct(s)).array() + (gamma * mu) -
                Delta_x.cwiseProduct(Delta_s).array(),
            (-v.cwiseProduct(w)).array() + (gamma * mu) -
                Delta_v.cwiseProduct(Delta_w).array(),
            (-tau * kappa) + (gamma * mu) - Delta_tau * Delta_kappa);

        alpha = max_alpha(x, Delta_x, v, Delta_v, s, Delta_s, w, Delta_w, tau,
                          Delta_tau, kappa, Delta_kappa);

        // High order corrections
        while (ncor <= 2 && alpha < 0.9995) {
            ncor++;
            alpha_ = std::min(1.0, 2.0 * alpha);
            mu_l = beta * mu * gamma;
            mu_u = gamma * mu / beta;

            // Compute products and corrections
            xs = x + alpha_ * Delta_x;
            xs.array() *= (s + alpha_ * Delta_s).array();
            vw = v + alpha_ * Delta_v;
            vw.array() *= (w + alpha_ * Delta_w).array();

            t_xs =
                (xs.array() < mu_l)
                    .select(mu_l - xs.array(),
                            (xs.array() > mu_u).select(mu_u - xs.array(), 0.0));
            t_vw =
                (vw.array() < mu_l)
                    .select(mu_l - vw.array(),
                            (vw.array() > mu_u).select(mu_u - vw.array(), 0.0));

            taukappa =
                (tau + alpha_ * Delta_tau) * (kappa + alpha_ * Delta_kappa);
            t0 = std::clamp(taukappa, mu_l, mu_u) - taukappa;

            const double sum_correction =
                (t_xs.sum() + t_vw.sum() + t0) / (nv + nu + 1);
            t_xs.array() -= sum_correction;
            t_vw.array() -= sum_correction;
            t0 -= sum_correction;

            // Save current directions
            Delta_x_c = Delta_x;
            Delta_lambda_c = Delta_lambda;
            Delta_w_c = Delta_w;
            Delta_s_c = Delta_s;
            Delta_v_c = Delta_v;
            Delta_tau_c = Delta_tau;
            Delta_kappa_c = Delta_kappa;

            // Solve correction system
            solve_newton_system(
                Delta_x_c, Delta_lambda_c, Delta_w_c, Delta_s_c, Delta_v_c,
                Delta_tau_c, Delta_kappa_c, ls, theta_vw, b, c, ubi, ubv,
                delta_x, delta_y, delta_z, delta_0, x, lambda, w, s, v, tau,
                kappa, Eigen::VectorXd::Zero(res.rp.size()),
                Eigen::VectorXd::Zero(res.ru.size()),
                Eigen::VectorXd::Zero(res.rd.size()), 0, -t_xs, -t_vw, -t0);

            alpha_c =
                max_alpha(x, Delta_x_c, v, Delta_v_c, s, Delta_s_c, w,
                          Delta_w_c, tau, Delta_tau_c, kappa, Delta_kappa_c);

            if (alpha_c > alpha_) {
                Delta_x = Delta_x_c;
                Delta_lambda = Delta_lambda_c;
                Delta_w = Delta_w_c;
                Delta_s = Delta_s_c;
                Delta_v = Delta_v_c;
                Delta_tau = Delta_tau_c;
                Delta_kappa = Delta_kappa_c;
                alpha = alpha_c;
            }

            if (alpha_c < 1.1 * alpha_) break;
        }

        // Final update step
        alpha *= 0.9995;
        x += alpha * Delta_x;
        lambda += alpha * Delta_lambda;
        s += alpha * Delta_s;
        v += alpha * Delta_v;
        w += alpha * Delta_w;
        tau += alpha * Delta_tau;
        kappa += alpha * Delta_kappa;
    }

    // Final solution recovery
    const double inv_tau = 1.0 / tau;
    Eigen::VectorXd original_x(As.cols());
    int free_var = 0;

    for (int j = 0; j < lo.size(); ++j) {
        const double l = lo[j];
        const double h = hi[j];

        if (l == -infty && h == infty) {
            original_x[j] = (x[j + free_var] - x[nv_orig + free_var]) * inv_tau;
            ++free_var;
        } else if (std::isfinite(l) && std::isfinite(h)) {
            original_x[j] = l + x[j] * inv_tau;
        } else if (l == -infty && std::isfinite(h)) {
            original_x[j] = h - x[j] * inv_tau;
        } else if (std::isfinite(l) && h == infty) {
            original_x[j] = l + x[j] * inv_tau;
        }
    }

    // Final objective computation and output
    objVal = cs.dot(original_x);
    lambda *= inv_tau;
    dual_vals.assign(lambda.data(), lambda.data() + lambda.size());
    primal_vals.assign(original_x.data(),
                       original_x.data() + original_x.size());
}

#ifdef GUROBI
OptimizationData IPSolver::extractOptimizationComponents(GRBModel &model) {
    OptimizationData data;
    int numConstrs = model.get(GRB_IntAttr_NumConstrs);
    int numVars = model.get(GRB_IntAttr_NumVars);

    std::vector<Eigen::Triplet<double>> triplets;
    data.bs.resize(numConstrs);
    data.cs.resize(numVars);
    data.lo.resize(numVars);
    data.hi.resize(numVars);
    data.sense.resize(numConstrs);

    // Extract the objective function
    GRBVar *vars = model.getVars();
    for (int j = 0; j < numVars; ++j) {
        data.cs(j) = vars[j].get(GRB_DoubleAttr_Obj);
        data.lo(j) = vars[j].get(GRB_DoubleAttr_LB);

        if (vars[j].get(GRB_DoubleAttr_UB) == GRB_INFINITY) {
            data.hi(j) = std::numeric_limits<double>::infinity();
        } else {
            data.hi(j) = vars[j].get(GRB_DoubleAttr_UB);
        }
    }

    // Extract constraint coefficients and senses
    for (int i = 0; i < numConstrs; ++i) {
        GRBConstr constr = model.getConstr(i);
        GRBLinExpr expr = model.getRow(constr);
        data.bs(i) = constr.get(GRB_DoubleAttr_RHS);

        //
        data.sense(i) = (constr.get(GRB_CharAttr_Sense) == '=') ? 1 : 0;
        for (int j = 0; j < expr.size(); ++j) {
            // if constr.get(GRB_CharAttr_Sense) > flip the sign
            double coef;
            if (constr.get(GRB_CharAttr_Sense) == '>') {
                coef = -expr.getCoeff(j);
                data.bs(i) = -data.bs(i);
            } else {
                coef = expr.getCoeff(j);
            }
            GRBVar var = expr.getVar(j);
            if (coef != 0.0) {
                triplets.push_back(
                    Eigen::Triplet<double>(i, var.index(), coef));
            }
        }
    }

    // Build the sparse matrix As
    data.As.resize(numConstrs, numVars);
    data.As.setFromTriplets(triplets.begin(), triplets.end());

    // save the model to a file, in the matricial form
    data.As.makeCompressed();

    return data;
}
#endif

OptimizationData IPSolver::convertToOptimizationData(
    const ModelData &modelData) {
    OptimizationData optData;

    // Convert SparseMatrix to Eigen::SparseMatrix using Eigen::Triplet
    std::vector<Eigen::Triplet<double>> triplets;
    auto sparseMatrix = modelData.A_sparse;

    optData.As = modelData.A_sparse.toEigenSparseMatrix();

    // Iterate over the CRS format of SparseMatrix to build triplets
    // for (int i = 0; i < sparseMatrix.outerSize(); ++i) {
    //    for (Eigen::SparseMatrix<double>::InnerIterator it(sparseMatrix, i);
    //    it; ++it) {
    //        triplets.push_back(Eigen::Triplet<double>(it.row(), it.col(),
    //        it.value()));
    //    }
    //}
    // Resize the Eigen sparse matrix
    // optData.As.resize(sparseMatrix.num_rows, sparseMatrix.num_cols);

    // Set the values from the triplets
    // optData.As.setFromTriplets(triplets.begin(), triplets.end());

    // Make the matrix compressed for efficient operations
    // optData.As.makeCompressed();

    // Convert b to Eigen::VectorXd
    optData.bs = Eigen::VectorXd::Map(modelData.b.data(), modelData.b.size());

    // Convert c to Eigen::VectorXd
    optData.cs = Eigen::VectorXd::Map(modelData.c.data(), modelData.c.size());

    // Convert lb to Eigen::VectorXd
    optData.lo = Eigen::VectorXd::Map(modelData.lb.data(), modelData.lb.size());

    // Convert ub to Eigen::VectorXd
    optData.hi = Eigen::VectorXd::Map(modelData.ub.data(), modelData.ub.size());

    // Convert sense to Eigen::VectorXd (mapping '<' to 0, '=' to 1, '>' to 0
    // and flipping the corresponding row)
    optData.sense.resize(modelData.sense.size());
    for (size_t i = 0; i < modelData.sense.size(); ++i) {
        if (modelData.sense[i] == '<') {
            optData.sense[i] = 0.0;
        } else if (modelData.sense[i] == '=') {
            optData.sense[i] = 1.0;
        } else if (modelData.sense[i] == '>') {
            optData.sense[i] = 0.0;
            optData.bs[i] = -optData.bs[i];
            optData.As.row(i) *= -1;  // Flip the row for '>'
        }
    }

    return optData;
}

int IPSolver::update_linear_solver(SparseSolver &ls,
                                   const Eigen::VectorXd &theta,
                                   const Eigen::VectorXd &regP,
                                   const Eigen::VectorXd &regD) {
    // Update internal data
    ls.theta = theta;
    ls.regP = regP;
    ls.regD = regD;

    // Update S. S is stored as upper-triangular and only its diagonal changes.
    Eigen::VectorXd combinedValues(ls.n + ls.m);
    combinedValues.head(ls.n) = -theta - regP;
    combinedValues.tail(ls.m) = regD;

    // Efficiently update diagonal elements
    for (int i = 0; i < combinedValues.size(); i++) {
        ls.S.coeffRef(i, i) = combinedValues[i];
    }

    // Refactorize
    ls.factorizeMatrix(ls.S);

    return ls.info();
}

/**
 * Starts the linear solver by initializing the necessary data structures and
 * performing factorization.
 *
 */
void IPSolver::start_linear_solver(SparseSolver &ls,
                                   const Eigen::SparseMatrix<double> A) {
    ls.A = A;
    ls.m = A.rows();
    ls.n = A.cols();
    // print ls.A size

    ls.theta = Eigen::VectorXd::Ones(ls.n);
    ls.regP = Eigen::VectorXd::Ones(ls.n);
    ls.regD = Eigen::VectorXd::Ones(ls.m);

    Eigen::SparseMatrix<double> topRight = ls.A.transpose();
    Eigen::SparseMatrix<double> bottomLeft = ls.A;
    Eigen::SparseMatrix<double> topLeft =
        convertToSparseDiagonal(-ls.theta - ls.regP);
    Eigen::SparseMatrix<double> bottomRight = convertToSparseDiagonal(ls.regD);

    // S_ is known, reserve space for it
    Eigen::SparseMatrix<double> S_(ls.n + ls.m, ls.n + ls.m);

    // Reserving space for tripletList
    int estimated_nonzeros =
        topLeft.nonZeros() + 2 * topRight.nonZeros() + bottomRight.nonZeros();
    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(estimated_nonzeros);

    // Insert topLeft, topRight, bottomLeft, bottomRight matrices
    auto insertBlock = [&](const Eigen::SparseMatrix<double> &block,
                           int startRow, int startCol) {
        for (int k = 0; k < block.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(block, k); it;
                 ++it) {
                tripletList.emplace_back(it.row() + startRow,
                                         it.col() + startCol, it.value());
            }
        }
    };

    insertBlock(topLeft, 0, 0);
    insertBlock(topRight, 0, ls.n);
    insertBlock(bottomLeft, ls.n, 0);
    insertBlock(bottomRight, ls.n, ls.n);

    // Finally, set the values from the triplets
    S_.setFromTriplets(tripletList.begin(), tripletList.end());
    // S_.makeCompressed();

    ls.S = S_;
    // Factorize
    ls.factorizeMatrix(ls.S);
}
