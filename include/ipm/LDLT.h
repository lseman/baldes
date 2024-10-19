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

    bool                        initialized          = false;
    bool                        patternAnalyzed      = false;
    Eigen::Index                nonZeroElements      = 0;
    double                      regularizationFactor = 1e-5; // Regularization factor
    Eigen::SparseMatrix<double> matrixToFactorize;
    Eigen::SparseMatrix<double> originalMatrix;

public:
    // Factorize the matrix using LDLT, which can handle indefinite matrices
    void factorizeMatrix(const Eigen::SparseMatrix<double> &matrix, int maxIterations = 50, double tolerance = 1e-3) {

        matrixToFactorize = matrix;

        if (!patternAnalyzed) {
            solver.analyzePattern(matrixToFactorize); // Analyze the sparsity pattern
            patternAnalyzed = true;
        }
        // First try factorizing the matrix without modifying it
        solver.factorize(matrixToFactorize);
        if (solver.info() != Eigen::Success) {

            // Use a reference to the original matrix to avoid copying unnecessarily
            Eigen::SparseMatrix<double> &regMatrix      = matrixToFactorize;
            double                       regularization = 1e-5; // Initial regularization term

            // Attempt to regularize and factorize up to 3 times
            const int maxAttempts = 3;
            for (int attempt = 1; attempt <= maxAttempts; ++attempt) {
                // Apply increasing regularization to the diagonal selectively
                applySelectiveRegularization(regMatrix, regularization);

                solver.factorize(regMatrix); // Factorize the modified matrix

                if (solver.info() == Eigen::Success) {
                    initialized = true;
                    return; // Exit early if successful
                }

                // Increase the regularization term for the next attempt
                regularization *= 10;
            }

            // If all attempts fail, throw an exception
            throw std::runtime_error("Matrix factorization failed after 3 attempts with LDLT and regularization.");
        }
        initialized = true;
    }

    void updateFactorization(const Eigen::SparseMatrix<double> &matrix, const Eigen::VectorXd &updatedDiag) {
        if (!initialized) { throw std::runtime_error("Matrix is not factorized."); }

        matrixToFactorize = matrix;
        // Attempt to update the factorization
        solver.update_factorization(matrixToFactorize, updatedDiag);
        if (solver.info() != Eigen::Success) {

            // Use a reference to the original matrix to avoid copying unnecessarily
            Eigen::SparseMatrix<double> &regMatrix       = matrixToFactorize;
            Eigen::VectorXd              updatedDiagCopy = updatedDiag;
            double                       regularization  = 1e-5; // Initial regularization term

            // Attempt to regularize and factorize up to 3 times
            const int maxAttempts = 3;
            for (int attempt = 1; attempt <= maxAttempts; ++attempt) {
                // Apply increasing regularization to the diagonal selectively
                applySelectiveRegularizationDiag(regMatrix, regularization, updatedDiagCopy); // Regularize diagonal

                solver.update_factorization(regMatrix, updatedDiagCopy); // Factorize the modified matrix
                if (solver.info() == Eigen::Success) {
                    initialized = true;
                    return; // Exit early if successful
                }

                // Increase the regularization term for the next attempt
                regularization *= 10;
            }

            // If all attempts fail, throw an exception
            throw std::runtime_error("Matrix factorization failed after 3 attempts with LDLT and regularization.");
        }
    }

    void applySelectiveRegularizationDiag(Eigen::SparseMatrix<double> &matrix, double regularization,
                                          Eigen::VectorXd &updatedDiag, double threshold = 1e-6) {
        // Ensure the matrix is square
        if (matrix.rows() != matrix.cols()) {
            throw std::runtime_error("Matrix is not square, cannot apply regularization to the diagonal.");
        }

        for (int i = 0; i < matrix.rows(); ++i) {

            // Additionally, add regularization to the updatedDiag as well
            if (std::abs(updatedDiag[i]) < threshold) { updatedDiag[i] += regularization; }
        }

        // Ensure the matrix is compressed after modifications
    }
    // Solve the system using the factorized matrix
    Eigen::VectorXd solve(const Eigen::VectorXd &b) {
        if (!initialized) { throw std::runtime_error("Matrix is not factorized."); }

        Eigen::VectorXd rhs = b;
        // Perform CG solve using the LDLT-preconditioned system
        Eigen::VectorXd x = solver.solve(rhs);
        return x;
    }

    void applySelectiveRegularization(Eigen::SparseMatrix<double> &matrix, double regularization,
                                      double threshold = 1e-6) {
        // Ensure the matrix is square
        if (matrix.rows() != matrix.cols()) {
            throw std::runtime_error("Matrix is not square, cannot apply regularization to the diagonal.");
        }

        // Directly access and modify diagonal elements
        for (int i = 0; i < matrix.rows(); ++i) {
            double &diagValue = matrix.coeffRef(i, i);
            if (std::abs(diagValue) < threshold) {
                diagValue += regularization; // Apply regularization if below threshold
            }
        }

        // Only compress if significant changes were made (optional, depending on matrix structure)
        // matrix.makeCompressed();
    }
};
