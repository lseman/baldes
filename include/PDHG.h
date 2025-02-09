#pragma once
#include <Eigen/Dense>
#include <iostream>

class PDLP {
   public:
    PDLP(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
         const Eigen::VectorXd& c, const Eigen::VectorXd& lower_bounds,
         const Eigen::VectorXd& upper_bounds)
        : A(A),
          b(b),
          c(c),
          lower_bounds(lower_bounds),
          upper_bounds(upper_bounds) {
        x = Eigen::VectorXd::Zero(A.cols());
        y = Eigen::VectorXd::Zero(A.rows());
    }

    Eigen::VectorXd proximal_operator(const Eigen::VectorXd& x,
                                      const Eigen::VectorXd& lower,
                                      const Eigen::VectorXd& upper) {
        return x.cwiseMax(lower).cwiseMin(upper);
    }

    Eigen::VectorXd dual_residuals(const Eigen::VectorXd& x,
                                   const Eigen::VectorXd& y) {
        return A.transpose() * y + c;
    }

    double corrected_dual_objective(const Eigen::VectorXd& x,
                                    const Eigen::VectorXd& y) {
        return (b.transpose() * y - c.transpose() * x).value();
    }

    std::pair<bool, bool> check_infeasibility(const Eigen::VectorXd& y) {
        bool primal_inf =
            (A.transpose() * y).isZero(1e-6) && (b.transpose() * y < 0);

        Eigen::VectorXd d = Eigen::VectorXd::Zero(x.size());
        bool dual_inf = (A * d).isZero(1e-6) && (c.transpose() * d < 0);

        return {primal_inf, dual_inf};
    }

    Eigen::VectorXd iterate(double tau, double sigma, int max_iter = 1000,
                            double tol = 1e-5) {
        for (int i = 0; i < max_iter; ++i) {
            Eigen::VectorXd x_new = proximal_operator(
                x - tau * (A.transpose() * y + c), lower_bounds, upper_bounds);
            Eigen::VectorXd y_new = y + sigma * (A * (2 * x_new - x) - b);

            double primal_residual = (x_new - x).norm();
            double dual_residual = (y_new - y).norm();
            double corrected_dual_obj = corrected_dual_objective(x_new, y_new);

            x = x_new;
            y = y_new;

            /*
            if (i % 100 == 0) {
                std::cout << "Iteration " << i
                          << ": Primal Residual = " << primal_residual
                          << ", Dual Residual = " << dual_residual
                          << ", Dual Objective = " << corrected_dual_obj
                          << std::endl;
            }
            */
            if (primal_residual < tol && dual_residual < tol) {
                std::cout << "Converged in " << i + 1 << " iterations."
                          << std::endl;
                break;
            }
        }
        return x;
    }

    const Eigen::VectorXd& get_primal_solution() const { return x; }
    const Eigen::VectorXd& get_dual_variables() const { return y; }

   private:
    Eigen::MatrixXd A;
    Eigen::VectorXd b, c, lower_bounds, upper_bounds, x, y;
};
