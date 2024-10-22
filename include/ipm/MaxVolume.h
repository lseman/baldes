#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>

#include <fmt/core.h>
using Int = int;

class MaxVolume {
public:
    // Constructor with default values for control parameters
    MaxVolume(int maxpasses = 100, double volume_tol = 1e-6, int maxskip_updates = 5)
        : maxpasses_(maxpasses), volume_tol_(volume_tol), maxskip_updates_(maxskip_updates) {
        Reset();
    }

    double max_alpha(const Eigen::VectorXd &x, const Eigen::VectorXd &dx, const Eigen::VectorXd &v,
                     const Eigen::VectorXd &dv, const Eigen::VectorXd &s, const Eigen::VectorXd &ds,
                     const Eigen::VectorXd &w, const Eigen::VectorXd &dw, double tau, double dtau, double kappa,
                     double dkappa) {
        double alpha_tau   = (dtau < 0) ? (-tau / dtau) : 1.0;
        double alpha_kappa = (dkappa < 0) ? (-kappa / dkappa) : 1.0;

        double alpha = std::min({1.0, max_alpha_single(x, dx), max_alpha_single(v, dv), max_alpha_single(s, ds),
                                 max_alpha_single(w, dw), alpha_tau, alpha_kappa});

        return alpha;
    }

    double max_alpha_single(const Eigen::VectorXd &v, const Eigen::VectorXd &dv) {

        double alpha = std::numeric_limits<double>::infinity();
        // #pragma omp parallel for reduction(min : alpha)
        for (int i = 0; i < v.size(); ++i) {
            if (dv(i) < 0) {
                double potential_alpha = -v(i) / dv(i);
                alpha                  = std::min(alpha, potential_alpha);
            }
        }

        return alpha;
    }

    // Main function to run MaxVolume within an IPM solver
    // Main function to run MaxVolume within an IPM solver
    Int RunIPM(Eigen::SparseMatrix<double> &A, Eigen::VectorXd &x, Eigen::VectorXd &lambda, Eigen::VectorXd &s,
               Eigen::VectorXd &v, Eigen::VectorXd &w, const Eigen::VectorXi &ubi, const Eigen::VectorXd &ubv,
               double &tau, double &kappa) {
        const Int n = A.cols(); // Number of columns in A (size of x, s)
        const Int m = A.rows(); // Number of rows in A (size of lambda)

        Eigen::VectorXd colscale(n); // Scaling factors for columns
        Eigen::VectorXd invscale(m); // Inverse scaling factors for rows

        // Initialize scaling factors for rows and columns
        colscale.setOnes();
        invscale.setOnes();

        // Reset internal state
        Reset();

        // Loop over iterations
        for (Int pass = 0; pass < maxpasses_; ++pass) {
            passes_++;
            // Step 1: Compute update directions (Primal-Dual)
            Eigen::VectorXd primal_update = A.transpose() * lambda + s; // Gradient w.r.t. primal
            // Efficiently calculate the dual update
            Eigen::VectorXd dual_update = A * x; // Start with A * x

            // Subtract v at the indices where upper bounds apply (ubi)
            for (int i = 0; i < ubi.size(); ++i) {
                int idx = ubi(i);         // Get the index from ubi
                dual_update(idx) -= v(i); // Subtract v at the index
            }

            // Step 2: Incorporate tau and kappa into the primal and dual updates
            primal_update /= tau; // Scale the primal update by tau
            dual_update /= tau;   // Scale the dual update by tau
            // Step 3: Select the column that maximizes the volume increase
            Int selected_col = SelectMaxVolumeColumn(A, colscale, invscale, primal_update);
            if (selected_col == -1) break; // No valid column found, terminate
            // Step 4: Update scaling factors based on the selected column
            colscale[selected_col] = std::max(colscale[selected_col], 1.0);
            invscale[selected_col] = 1.0 / colscale[selected_col];


            // ** Compute the step size alpha to prevent infeasibility **
            double alpha =
                max_alpha(x, primal_update, v, dual_update, s, primal_update, w, dual_update, tau, 1.0, kappa, 1.0);
            // ** Apply the scaled updates to prevent infeasibility **
            x += alpha * primal_update; // Update primal variables
            lambda += alpha * dual_update; // Update dual variables
            s += alpha * primal_update; // Ensure feasibility of the slack variables
            for (int i = 0; i < ubi.size(); ++i) {
                int idx = ubi(i);                   // Upper-bound index
                v(i) += alpha * (dual_update(idx)); // Update v only where there are upper bounds
                w(i) += alpha * (dual_update(idx)); // Update w based on upper-bound feasibility
            }

            updates_++;

            // Step 5: Adjust tau and kappa to maintain stability
            double primal_dual_gap = (x.dot(s) + v.dot(w) + tau * kappa) / (n + ubi.size() + 1.0);
            tau += primal_dual_gap;              // Adjust tau based on the primal-dual gap
            kappa = std::max(kappa / 2.0, 1e-8); // Gradually reduce kappa for convergence

            // Step 6: Volume Check
            double vol_increase = primal_update.norm() * dual_update.norm();
            if (vol_increase <= volume_tol_) {
                skipped_++;
                if (skipped_ > maxskip_updates_) break;
                continue;
            }

            volinc_ += std::log(vol_increase); // Accumulate volume increase
        }

        return updates_;
    }

private:
    // Function to select the column with the maximum volume increase
    // Function to select the column with the maximum volume increase
    Int SelectMaxVolumeColumn(const Eigen::SparseMatrix<double> &A, const Eigen::VectorXd &colscale,
                              const Eigen::VectorXd &invscale, Eigen::VectorXd &update_dir) {
        Int    max_col = -1;
        double max_vol = 0.0;

        // Ensure colscale and invscale have correct sizes
        assert(colscale.size() == A.cols());
        assert(invscale.size() == A.rows());

        // Loop over all columns to find the one with the maximum volume increase
        for (Int j = 0; j < A.cols(); ++j) {
            double col_norm = 0.0;

            // Iterate over the non-zero elements of the j-th column of the sparse matrix A
            for (Eigen::SparseMatrix<double>::InnerIterator it(A, j); it; ++it) {
                int    row   = it.row();
                double value = it.value();

                // Compute the norm of the column considering invscale and colscale
                col_norm += std::pow(value * invscale[row], 2); // Update the norm using invscale
            }

            col_norm = std::sqrt(col_norm); // Compute the final norm

            // Compute the volume contribution of the column
            double col_vol = col_norm * colscale[j];
            // fmt::print("Column {} Volume: {}\n", j, col_vol);
            //  Check if this column has the maximum volume increase
            if (col_vol > max_vol) {
                max_vol = col_vol;
                max_col = j;
            }
        }

        return (max_vol > volume_tol_) ? max_col : -1;
    }

    // Reset internal counters and values
    void Reset() {
        updates_          = 0;
        skipped_          = 0;
        passes_           = 0;
        volinc_           = 0.0;
        time_             = 0.0;
        frobnorm_squared_ = 0.0;
    }

    // Control-like parameters
    int    maxpasses_;       // Maximum number of passes (iterations)
    double volume_tol_;      // Tolerance for volume increase
    int    maxskip_updates_; // Maximum number of updates that can be skipped

    // Internal tracking variables
    Int    updates_, skipped_, passes_;
    double volinc_, time_;
    double frobnorm_squared_;
};
