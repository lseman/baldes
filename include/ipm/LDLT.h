#pragma once

#include "Definitions.h"
#include "fmt/base.h"
#include <Eigen/Dense>
#include <Eigen/OrderingMethods>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <iostream>
#include <new>
#include <stdexcept>
#include <vector>

// Custom preconditioner using LDLT factorization (use a pointer to avoid reference issues)
class LDLTPreconditioner {
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Lower, Eigen::AMDOrdering<int>> *ldltSolver;
    Eigen::ComputationInfo                                                                     m_info;
    bool                                                                                       patternAnalyzed = false;
    int                                                                                        nonZeroElements = 0;
    double regularizationFactor = 1e-5; // Regularization factor
public:
    // Default constructor (required for Eigen)
    LDLTPreconditioner() : ldltSolver(nullptr), m_info(Eigen::Success) {}

    // Constructor with LDLT solver pointer
    LDLTPreconditioner(
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Lower, Eigen::AMDOrdering<int>> *solver)
        : ldltSolver(solver), m_info(Eigen::Success) {}

    // Set the LDLT solver (in case we use the default constructor)
    void
    setLDLTSolver(Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Lower, Eigen::AMDOrdering<int>> *solver) {
        ldltSolver = solver;
    }

    // Preconditioner compute function (required by Eigen)
    template <typename MatrixType>
    void compute(const MatrixType &matrix) {

        if (!patternAnalyzed) {
            ldltSolver->analyzePattern(matrix); // Analyze the sparsity pattern
            patternAnalyzed = true;
        }
        // First try factorizing the matrix without modifying it
        ldltSolver->factorize(matrix);
        if (ldltSolver->info() != Eigen::Success) {

            // Convert the matrix to a modifiable SparseMatrix for regularization
            Eigen::SparseMatrix<double> regMatrix      = matrix; // Copy of the matrix
            double                      regularization = 1e-5;   // Initial regularization term

            // Attempt to regularize and factorize up to 3 times
            const int maxAttempts = 1;
            for (int attempt = 1; attempt <= maxAttempts; ++attempt) {
                applySelectiveRegularization(regMatrix, regularization); // Apply increasing regularization

                ldltSolver->factorize(matrix); // Analyze the sparsity pattern and factorize

                if (ldltSolver->info() == Eigen::Success) {
                    m_info = Eigen::Success; // Successfully factorized after regularization
                    return;
                }

                // Increase the regularization term for the next attempt
                regularization *= 100;
            }

            // If all attempts fail, throw an exception
            throw std::runtime_error("Matrix factorization failed after 3 attempts with LDLT and regularization.");
        }

        m_info = Eigen::Success; // Successfully factorized without regularization
    }

    // The preconditioner operation: Apply the LDLT solve step
    Eigen::VectorXd solve(const Eigen::VectorXd &b) const {
        if (ldltSolver == nullptr) { throw std::runtime_error("LDLT solver is not initialized."); }
        return ldltSolver->solve(b).eval(); // Ensure we return a copy of the result
    }

    // Provide info about the preconditioner status (required by Eigen)
    Eigen::ComputationInfo info() const { return m_info; }

    // Apply regularization to the diagonal of the matrix
    // Helper function to apply regularization (add small value to the diagonal)
    void applyRegularization(Eigen::SparseMatrix<double> &matrix, double regularization) {
        // Ensure the matrix is square
        if (matrix.rows() != matrix.cols()) {
            throw std::runtime_error("Matrix is not square, cannot apply regularization to the diagonal.");
        }

        for (int i = 0; i < matrix.rows(); ++i) {
            if (i < matrix.outerSize()) {                // Ensure the index is valid in the sparse matrix
                matrix.coeffRef(i, i) += regularization; // Add regularization to the diagonal
            } else {
                throw std::runtime_error("Invalid diagonal index encountered during regularization.");
            }
        }

        // Check if matrix remains valid after regularization
        if (!matrix.isCompressed()) {
            matrix.makeCompressed(); // Ensure the matrix is compressed after modification
        }
    }

    void applySelectiveRegularization(Eigen::SparseMatrix<double> &matrix, double regularization,
                                      double threshold = 1e-6) {
        // Ensure the matrix is square
        if (matrix.rows() != matrix.cols()) {
            throw std::runtime_error("Matrix is not square, cannot apply regularization to the diagonal.");
        }

        for (int i = 0; i < matrix.outerSize(); ++i) {
            if (i < matrix.outerSize()) {
                // Check if the diagonal element is below the threshold
                double diagValue = matrix.coeff(i, i);
                if (std::abs(diagValue) < threshold) {
                    matrix.coeffRef(i, i) += regularization; // Apply regularization if below threshold
                }
            } else {
                throw std::runtime_error("Invalid diagonal index encountered during regularization.");
            }
        }

        // Ensure the matrix is compressed after modifications
        if (!matrix.isCompressed()) { matrix.makeCompressed(); }
    }
};

class ConjugateResidual {
    int    maxIterations;
    double tolerance;
    int    resetFrequency = 10; // Reset the search direction every 10 iterations
public:
    Eigen::ComputationInfo m_info; // Store computation status
    LDLTPreconditioner    *preconditioner;
    ConjugateResidual(int maxIter = 100, double tol = 1e-4)
        : maxIterations(maxIter), tolerance(tol), m_info(Eigen::Success) {}

    void setMaxIterations(int maxIter) { maxIterations = maxIter; }
    void setTolerance(double tol) { tolerance = tol; }
    void setPreconditioner(LDLTPreconditioner *precond) { preconditioner = precond; }

    template <typename MatrixType>
    Eigen::VectorXd solve(const MatrixType &A, const Eigen::VectorXd &b) {
        Eigen::VectorXd x = Eigen::VectorXd::Zero(b.size()); // Initialize x = 0
        Eigen::VectorXd r = b;                               // Initial residual r = b - A*x, but x=0 so r = b
        Eigen::VectorXd p = preconditioner->solve(r);        // Apply preconditioner to residual
        Eigen::VectorXd z = p;                               // z = p for the first iteration
        Eigen::VectorXd Ap(b.size());
        double          resNorm          = r.norm();
        double          initialResNorm   = resNorm;
        double          currentTolerance = tolerance; // Start with initial tolerance

        m_info = Eigen::Success; // Initialize info to success at the beginning

        for (int i = 0; i < maxIterations; ++i) {
            if (resNorm < currentTolerance) {
                m_info = Eigen::Success; // Successfully converged
                break;
            }

            Ap           = A * p;                // Compute A*p
            double alpha = r.dot(z) / p.dot(Ap); // Compute alpha
            x += alpha * p;                      // Update solution
            Eigen::VectorXd r_old = r;           // Save old residual
            r -= alpha * Ap;                     // Update residual
            resNorm = r.norm();

            // Dynamically adapt tolerance based on residual progress
            if (resNorm / initialResNorm < 0.1) {
                currentTolerance *= 0.5; // Tighten tolerance if significant progress is made
            }

            // Check for divergence or numerical issues
            if (std::isnan(resNorm) || std::isinf(resNorm)) {
                m_info = Eigen::NumericalIssue; // Numerical issues (NaN or Inf) detected
                break;
            }

            z           = preconditioner->solve(r);                           // Apply preconditioner to new residual
            double beta = r.dot(z) / r_old.dot(preconditioner->solve(r_old)); // Compute beta
            p           = z + beta * p;                                       // Update search direction
        }

        // If we reached the maximum number of iterations and haven't converged
        if (resNorm >= currentTolerance) {
            m_info = Eigen::NoConvergence; // Did not converge within maxIterations
        }

        return x; // Return the solution
    }

    // define compute method which calls preconditioner compute
    template <typename MatrixType>
    void compute(const MatrixType &matrix) {
        preconditioner->compute(matrix);
    }

    // Provide computation info after solving
    Eigen::ComputationInfo info() const { return m_info; }
};

/*
 * @class SparseCholesky
 * @brief A class to perform sparse Cholesky factorization.
 *
 * This class provides an interface to perform sparse Cholesky factorization
 * on a given sparse matrix. It uses Eigen's SimplicialLDLT decomposition
 * to factorize the matrix and solve linear systems efficiently.
 */
class LDLTSolver {
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Lower, Eigen::AMDOrdering<int>> solver;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Lower, Eigen::AMDOrdering<int>> solverBackup;

    bool                        initialized          = false;
    bool                        patternAnalyzed      = false;
    Eigen::Index                nonZeroElements      = 0;
    double                      regularizationFactor = 1e-5; // Regularization factor
    Eigen::SparseMatrix<double> matrixToFactorize;

    LDLTPreconditioner preconditioner;
    ConjugateResidual  cr; // Conjugate Residual solver

public:
    // Factorize the matrix using LDLT, which can handle indefinite matrices
    void factorizeMatrix(const Eigen::SparseMatrix<double> &matrix, int maxIterations = 50, double tolerance = 1e-3) {
        matrixToFactorize = matrix;

        if (!initialized) {
            preconditioner.setLDLTSolver(&solver);
            solverBackup.analyzePattern(matrixToFactorize);
            cr.setMaxIterations(maxIterations);
            cr.setTolerance(tolerance);
            cr.setPreconditioner(&preconditioner);
            initialized = true;
        }
        cr.compute(matrixToFactorize);
    }

    // Solve the system using the factorized matrix
    Eigen::VectorXd solve(const Eigen::VectorXd &b) {
        if (!initialized) { throw std::runtime_error("Matrix is not factorized."); }

        // Perform CG solve using the LDLT-preconditioned system
        Eigen::VectorXd x;

        try {
            x = cr.solve(matrixToFactorize, b);

            if (cr.info() != Eigen::Success) {
                // fmt::print("CR failed, attempting direct solve\n");

                // Attempt direct solve as a fallback
                solverBackup.factorize(matrixToFactorize);

                x = solverBackup.solve(b);
                if (solverBackup.info() != Eigen::Success) {
                    throw std::runtime_error("Direct solve failed after CG failure.");
                }
            }
        } catch (const std::exception &e) { throw std::runtime_error(fmt::format("Error during solve: {}", e.what())); }

        return x;
    }
};
