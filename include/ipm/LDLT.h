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

#include "LDLTSimp.h"

class DiagonalPreconditioner {
public:
    Eigen::VectorXd invDiagonal; // Store the inverse of the diagonal elements

    // Constructor
    DiagonalPreconditioner() = default;

    // Compute the preconditioner by inverting the diagonal elements of the matrix A
    template <typename MatrixType>
    void compute(const MatrixType &A) {
        Eigen::VectorXd diagonal = A.diagonal();            // Get the diagonal elements of A
        invDiagonal              = diagonal.cwiseInverse(); // Store their inverses

        // Optionally: Add a safeguard for zero/near-zero diagonal elements
        const double epsilon = 1e-10;
        for (int i = 0; i < diagonal.size(); ++i) {
            if (std::abs(diagonal[i]) < epsilon) {
                invDiagonal[i] = 1.0 / epsilon; // Replace with a small inverse value
            }
        }
    }

    // Apply the preconditioner (solve Mz = r where M is the diagonal matrix)
    Eigen::VectorXd solve(const Eigen::VectorXd &r) const {
        return invDiagonal.asDiagonal() * r; // Multiply r by the inverse diagonal elements
    }
};

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
            const int maxAttempts = 0;
            for (int attempt = 1; attempt <= maxAttempts; ++attempt) {
                applySelectiveRegularization(regMatrix, regularization); // Apply increasing regularization

                ldltSolver->factorize(regMatrix); // Analyze the sparsity pattern and factorize

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

class SchulzPreconditioner {
public:
    Eigen::MatrixXd inverseApprox; // Approximate inverse of matrix A

    // Constructor: takes number of iterations to improve the inverse
    SchulzPreconditioner(int iterations = 5) : maxIterations(iterations) {}

    // Compute the preconditioner using Schulz iteration
    template <typename MatrixType>
    void compute(const MatrixType &A) {
        // Start with the inverse of the diagonal as an initial guess
        Eigen::MatrixXd D_inv = A.diagonal().cwiseInverse().asDiagonal();
        inverseApprox         = D_inv; // Initial approximation of the inverse

        // Identity matrix
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(A.rows(), A.cols());

        // Schulz iteration
        for (int i = 0; i < maxIterations; ++i) { inverseApprox = inverseApprox * (2.0 * I - A * inverseApprox); }
    }

    // Apply the preconditioner (solve the system M*z = r where M is the approximate inverse)
    Eigen::VectorXd solve(const Eigen::VectorXd &r) const { return inverseApprox * r; }

private:
    int maxIterations; // Number of Schulz iterations
};

class ConjugateResidual {
    int    maxIterations;
    double tolerance;
    int    resetFrequency    = 10;
    bool   usePreconditioner = false;

public:
    Eigen::ComputationInfo  m_info;
    DiagonalPreconditioner *preconditioner = nullptr;

    ConjugateResidual(int maxIter = 100, double tol = 1e-4)
        : maxIterations(maxIter), tolerance(tol), m_info(Eigen::Success) {}

    void setMaxIterations(int maxIter) { maxIterations = maxIter; }
    void setTolerance(double tol) { tolerance = tol; }

    // Set diagonal preconditioner
    void setPreconditioner(DiagonalPreconditioner *precond) {
        preconditioner    = precond;
        usePreconditioner = true;
    }

    // Initial residual
    template <typename MatrixType>
    Eigen::VectorXd solve(const MatrixType &A, const Eigen::VectorXd &b) {
        Eigen::VectorXd x = Eigen::VectorXd::Zero(b.size()); // Initialize solution
        Eigen::VectorXd r = b;                               // Initial residual
        Eigen::VectorXd p;                                   // Search direction
        Eigen::VectorXd z;                                   // Preconditioned residual
        Eigen::VectorXd Ap(b.size());                        // A*p product
        double          resNorm              = r.norm();     // Residual norm
        double          initialResNorm       = resNorm;      // Initial residual norm
        double          currentTolerance     = tolerance;    // Dynamic tolerance
        double          prevResNorm          = resNorm;      // Track previous residual norm for adaptive check
        const double    regularizationFactor = 1e-8;         // Small regularization term
        int             stagnationCounter    = 0;            // Track stagnation over iterations

        m_info = Eigen::Success; // Initialize success

        // Apply preconditioner if available
        if (usePreconditioner && preconditioner != nullptr) {
            z = preconditioner->solve(r);
            p = z;
        } else {
            p = r;
        }

        for (int i = 0; i < maxIterations; ++i) {
            // Check if we've already reached the desired tolerance
            if (resNorm < currentTolerance) {
                m_info = Eigen::Success;
                break;
            }

            // Add small regularization term to stabilize the solution
            Ap = A * p + regularizationFactor * p;

            // Compute alpha (with safety checks for numerical stability)
            double denom = p.dot(Ap);
            if (denom == 0 || !std::isfinite(denom)) {
                m_info = Eigen::NumericalIssue;
                break;
            }
            double alpha = r.dot(p) / denom;

            if (!std::isfinite(alpha)) {
                m_info = Eigen::NumericalIssue;
                break;
            }

            // Update solution and residual
            x += alpha * p;
            r -= alpha * Ap;
            resNorm = r.norm();

            // Dynamically adjust tolerance based on residual progress
            if (resNorm / initialResNorm < 0.1) { currentTolerance *= 0.5; }

            // Check for NaN/Inf in residual and terminate early if detected
            if (std::isnan(resNorm) || std::isinf(resNorm)) {
                m_info = Eigen::NumericalIssue;
                break;
            }

            // Reapply the preconditioner if available
            if (usePreconditioner && preconditioner != nullptr) {
                z = preconditioner->solve(r);
            } else {
                z = r;
            }

            // Compute beta (with safety checks for numerical stability)
            double betaNumerator   = r.dot(z);
            double betaDenominator = r.dot(p);
            if (betaDenominator == 0 || !std::isfinite(betaDenominator)) {
                m_info = Eigen::NumericalIssue;
                break;
            }
            double beta = betaNumerator / betaDenominator;

            if (!std::isfinite(beta)) {
                m_info = Eigen::NumericalIssue;
                break;
            }

            // Update search direction
            p = z + beta * p;

            // Check for stagnation or divergence
            if (resNorm / initialResNorm > 1e2) {
                m_info = Eigen::NumericalIssue;
                break;
            }

            // Detect stagnation: if residual norm changes very little over a few iterations
            if (std::abs(prevResNorm - resNorm) / prevResNorm < 1e-6) {
                stagnationCounter++;
            } else {
                stagnationCounter = 0; // Reset counter if significant progress is made
            }

            // Terminate if stagnation persists for too long
            if (stagnationCounter >= 5) {
                m_info = Eigen::NoConvergence;
                break;
            }

            prevResNorm = resNorm; // Update previous residual norm for next iteration
        }

        if (resNorm >= currentTolerance) {
            m_info = Eigen::NoConvergence; // Indicate that we didn't converge
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
    Eigen::CustomSimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Lower, Eigen::AMDOrdering<int>> solver;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Lower, Eigen::AMDOrdering<int>>       solverBackup;

    bool                        initialized          = false;
    bool                        patternAnalyzed      = false;
    Eigen::Index                nonZeroElements      = 0;
    double                      regularizationFactor = 1e-5; // Regularization factor
    Eigen::SparseMatrix<double> matrixToFactorize;
    Eigen::SparseMatrix<double> originalMatrix;

    DiagonalPreconditioner preconditioner;
    ConjugateResidual      cr; // Conjugate Residual solver
    // Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solveBIC;

public:
    // Factorize the matrix using LDLT, which can handle indefinite matrices
    void factorizeMatrix(const Eigen::SparseMatrix<double> &matrix, int maxIterations = 50, double tolerance = 1e-3) {

        /*
        preconditioner.compute(matrix); // Compute the inverse of diagonal of A

        matrixToFactorize = matrix;
        originalMatrix    = matrix;

        if (!initialized) {
            // cr.setPreconditioner(&preconditioner);
            //  solverBackup.analyzePattern(matrixToFactorize);
            // cr.setMaxIterations(maxIterations);
            // cr.setTolerance(tolerance);
            //  cr.setPreconditioner(&preconditioner);
            initialized = true;
        }
        // solveBIC.compute(matrixToFactorize);
        */
        matrixToFactorize = matrix;

        if (!patternAnalyzed) {
            solver.analyzePattern(matrixToFactorize); // Analyze the sparsity pattern
            patternAnalyzed = true;
        }
        // First try factorizing the matrix without modifying it
        solver.factorize(matrixToFactorize);
        if (solver.info() != Eigen::Success) {

            // Convert the matrix to a modifiable SparseMatrix for regularization
            Eigen::SparseMatrix<double> regMatrix      = matrixToFactorize; // Copy of the matrix
            double                      regularization = 1e-5;              // Initial regularization term

            // Attempt to regularize and factorize up to 3 times
            const int maxAttempts = 3;
            for (int attempt = 1; attempt <= maxAttempts; ++attempt) {
                applySelectiveRegularization(regMatrix, regularization); // Apply increasing regularization

                solver.factorize(regMatrix); // Analyze the sparsity pattern and factorize

                if (solver.info() == Eigen::Success) { return; }

                // Increase the regularization term for the next attempt
                regularization *= 10;
            }

            // If all attempts fail, throw an exception
            throw std::runtime_error("Matrix factorization failed after 3 attempts with LDLT and regularization.");
        }
        initialized = true;
    }

    // Solve the system using the factorized matrix
    Eigen::VectorXd solve(const Eigen::VectorXd &b) {
        if (!initialized) { throw std::runtime_error("Matrix is not factorized."); }

        Eigen::VectorXd rhs = b;
        // Perform CG solve using the LDLT-preconditioned system
        Eigen::VectorXd x;
        /*
                x = solveBIC.solve(b);

                if (solveBIC.info() != Eigen::Success) {
                    // fmt::print("CR failed, attempting direct solve\n");

                    // Attempt direct solve as a fallback
                    solverBackup.compute(originalMatrix);

                    x = solverBackup.solve(b);
                    if (solverBackup.info() != Eigen::Success) {
                        throw std::runtime_error("Direct solve failed after CG failure.");
                    }
                }
        */
        x = solver.solve(rhs);
        return x;
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
