#pragma once

#include "Definitions.h"
#include "fmt/base.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <gurobi_c++.h>
#include <iostream>
#include <new>
#include <stdexcept>
#include <vector>

/*
 * @class SparseCholesky
 * @brief A class to perform sparse Cholesky factorization.
 *
 * This class provides an interface to perform sparse Cholesky factorization
 * on a given sparse matrix. It uses Eigen's SimplicialLDLT decomposition
 * to factorize the matrix and solve linear systems efficiently.
 */
class LDLTSolver {
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver; // LDLT decomposition
    bool                                               initialized          = false;
    bool                                               patternAnalyzed      = false;
    Eigen::Index                                       nonZeroElements      = 0;
    double                                             regularizationFactor = 1e-5; // Regularization factor

    Eigen::SparseMatrix<double>                        preconditionerMatrix; // Preconditioner matrix
    Eigen::SparseMatrix<double>                        Kpp, Kuu, Kup, Kpu;   // Submatrices for Schur complement
    Eigen::SparseMatrix<double>                        schurComplement;      // Schur complement matrix
    std::vector<char>                                  mask;
    int                                                sizeKuu = 0, sizeKpp = 0;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> KuuSolver; // Cache the factorization
    Eigen::SparseMatrix<double>                        matrixToFactorize;

public:
    // Factorize the matrix using LDLT, which can handle indefinite matrices
    void factorizeMatrix(const Eigen::SparseMatrix<double> &matrix, bool useReordering = true,
                         bool usePreconditioner = false) {
        if (matrix.rows() != matrix.cols()) { throw std::invalid_argument("Matrix must be square for decomposition."); }

        matrixToFactorize = matrix;

        // Apply Cuthill-McKee reordering if requested
        if (useReordering) {
            Eigen::VectorXi perm;
            reorderCuthillMcKee(matrix, perm);
            matrixToFactorize = applyPermutation(matrix, perm);
        }

        // Infer block sizes and create a mask dynamically if preconditioner is enabled
        if (usePreconditioner) {
            std::vector<char> mask = createMaskFromSparsity(matrixToFactorize);

            // Infer block sizes using the mask
            auto sizesKs = inferBlockSizes(mask);
            sizeKuu      = sizesKs.first;
            sizeKpp      = sizesKs.second;

            // Create the Schur complement preconditioner using the inferred block sizes and mask
            createSchurComplementPreconditioner(matrixToFactorize);
        }

        // Analyze sparsity pattern if needed
        if (!patternAnalyzed || matrixToFactorize.nonZeros() != nonZeroElements) {
            solver.analyzePattern(matrixToFactorize);
            patternAnalyzed = true;
            nonZeroElements = matrixToFactorize.nonZeros();
        }

        // Factorize matrix, and apply regularization if needed
        if (!tryFactorize(matrixToFactorize)) {
            Eigen::SparseMatrix<double> regMatrix = matrixToFactorize;
            applyRegularization(regMatrix);
            if (!tryFactorize(regMatrix)) {
                throw std::runtime_error("Matrix factorization failed, even with LDLT and regularization.");
            }
        }

        initialized = true;
    }

    // Solve the system using the factorized matrix
    Eigen::VectorXd solve(const Eigen::VectorXd &b, bool usePreconditioner = false) {
        if (!initialized) { throw std::runtime_error("Matrix is not factorized."); }

        Eigen::VectorXd rhs = b;

        // Apply Schur complement preconditioner if enabled
        if (usePreconditioner) { rhs = applySchurComplementPreconditioner(b, sizeKuu, sizeKpp); }

        Eigen::VectorXd x = solver.solve(rhs);
        if (solver.info() != Eigen::Success) { throw std::runtime_error("Solving the system failed."); }

        return x;
    }

    void applyJacobiPreconditioner(const Eigen::SparseMatrix<double> &matrix, Eigen::VectorXd &rhs) {
        Eigen::VectorXd diag    = matrix.diagonal();
        const double    epsilon = 1e-8; // Regularization term for small/zero diagonals

        // Safely apply Jacobi scaling to rhs
        rhs.array() /= (diag.array().abs() > epsilon).select(diag.array(), epsilon);
    }

private:
    // Try to factorize the matrix, return success or failure
    bool tryFactorize(const Eigen::SparseMatrix<double> &matrix) {
        solver.factorize(matrix);
        return solver.info() == Eigen::Success;
    }

    // Apply regularization to the diagonal of the matrix
    void applyRegularization(Eigen::SparseMatrix<double> &matrix) const {
        matrix.diagonal() += Eigen::VectorXd::Constant(matrix.rows(), regularizationFactor);
    }

    //////////////////////////////////////////
    // Cuthill-McKee Reordering
    //////////////////////////////////////////

    // Apply Cuthill-McKee reordering to reduce bandwidth
    void reorderCuthillMcKee(const Eigen::SparseMatrix<double> &matrix, Eigen::VectorXi &perm) {
        const Eigen::Index n = matrix.rows();
        perm.resize(n);
        perm.setZero(); // Initialize permutation vector

        // Implementing Cuthill-McKee based on the provided code template
        // Initialize vectors for degrees, level set, etc.
        std::vector<Eigen::Index> degree(n, 0);
        std::vector<Eigen::Index> levelSet(n, 0);
        std::vector<Eigen::Index> nextSameDegree(n, -1);
        Eigen::Index              initialNode = 0;
        Eigen::Index              maxDegree   = 0;

        // Compute degrees of the graph nodes
        for (Eigen::Index i = 0; i < n; ++i) {
            degree[i] = matrix.outerIndexPtr()[i + 1] - matrix.outerIndexPtr()[i];
            maxDegree = std::max(maxDegree, degree[i]);
        }

        // Initialize the first level set
        perm[0]                                 = initialNode;
        Eigen::Index currentLevelSet            = 1;
        levelSet[initialNode]                   = currentLevelSet;
        Eigen::Index maxDegreeInCurrentLevelSet = degree[initialNode];

        // Main loop to process the level sets and fill the permutation
        for (Eigen::Index next = 1; next < n;) {
            Eigen::Index              nMDICLS = 0;
            std::vector<Eigen::Index> nFirstWithDegree(maxDegree + 1, -1);

            for (Eigen::Index node = 0; node < n; ++node) {
                if (levelSet[node] == 0) {
                    levelSet[node] = currentLevelSet + 1;
                    perm[next]     = node;
                    ++next;
                    nextSameDegree[node]           = nFirstWithDegree[degree[node]];
                    nFirstWithDegree[degree[node]] = node;
                    nMDICLS                        = std::max(nMDICLS, degree[node]);
                }
            }

            ++currentLevelSet;
        }
    }

    // Apply the permutation to reorder the matrix
    Eigen::SparseMatrix<double> applyPermutation(const Eigen::SparseMatrix<double> &matrix,
                                                 const Eigen::VectorXi             &perm) {
        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> p(perm);
        return p.transpose() * matrix * p;
    }

    //////////////////////////////////////////
    // Preconditioner
    //////////////////////////////////////////

    // Infer the block sizes dynamically based on the mask
    std::pair<Eigen::Index, Eigen::Index> inferBlockSizes(const std::vector<char> &mask) const {
        Eigen::Index sizeKuu = 0;
        Eigen::Index sizeKpp = 0;

        // Count the number of entries for Kuu and Kpp based on the mask
        for (const auto &m : mask) {
            if (m == 0) {
                ++sizeKuu; // Count entries belonging to Kuu block
            } else {
                ++sizeKpp; // Count entries belonging to Kpp block
            }
        }

        // Validate block sizes
        if (sizeKuu == 0 || sizeKpp == 0) {
            throw std::runtime_error(
                "Invalid block sizes inferred from the mask. Both Kuu and Kpp must have non-zero sizes.");
        }

        return {sizeKuu, sizeKpp};
    }

    // Create the Schur complement preconditioner
    void createSchurComplementPreconditioner(Eigen::SparseMatrix<double> &matrix) {
        // Decompose matrix into submatrices: Kuu, Kup, Kpu, Kpp
        decomposeMatrixIntoBlocks(matrix, sizeKuu, sizeKpp);

        // Factorize Kuu and compute Kuu^-1 * Kup
        KuuSolver.compute(Kuu);
        if (KuuSolver.info() != Eigen::Success) { throw std::runtime_error("Kuu factorization failed."); }

        // Solve for Kuu^-1 * Kup
        Eigen::SparseMatrix<double> KuuInvKup = KuuSolver.solve(Kup);

        // Compute the Schur complement: S = Kpp - Kpu * Kuu^-1 * Kup
        schurComplement = Kpp - Kpu * KuuInvKup;

        // Ensure Schur complement matrix size matches Kpp
        if (schurComplement.rows() != Kpp.rows() || schurComplement.cols() != Kpp.cols()) {
            throw std::runtime_error("Schur complement matrix size mismatch.");
        }

        // Store the Schur complement as the preconditioner
        preconditionerMatrix = schurComplement;
    }

    // Decompose the matrix into the submatrices Kuu, Kup, Kpu, Kpp
    void decomposeMatrixIntoBlocks(const Eigen::SparseMatrix<double> &matrix, Eigen::Index sizeKuu,
                                   Eigen::Index sizeKpp) {
        // Ensure the matrix has enough rows and columns to decompose
        if (matrix.rows() != sizeKuu + sizeKpp || matrix.cols() != sizeKuu + sizeKpp) {
            throw std::runtime_error("Matrix size does not match the expected block sizes.");
        }

        // Decompose into blocks
        Kuu = matrix.topLeftCorner(sizeKuu, sizeKuu);     // Kuu: top-left block
        Kup = matrix.topRightCorner(sizeKuu, sizeKpp);    // Kup: top-right block
        Kpu = matrix.bottomLeftCorner(sizeKpp, sizeKuu);  // Kpu: bottom-left block
        Kpp = matrix.bottomRightCorner(sizeKpp, sizeKpp); // Kpp: bottom-right block
    }

    // Apply Schur complement preconditioner
    // Apply Schur complement preconditioner and reconstruct the full solution
    Eigen::VectorXd applySchurComplementPreconditioner(const Eigen::VectorXd &b, Eigen::Index sizeKuu,
                                                       Eigen::Index sizeKpp) {
        if (b.size() != sizeKuu + sizeKpp) {
            throw std::runtime_error("Dimension mismatch between matrix blocks and vector b.");
        }

        // Split b into parts corresponding to Kuu and Kpp
        Eigen::VectorXd bu = b.head(sizeKuu);
        Eigen::VectorXd bp = b.tail(sizeKpp);

        // Solve Kuu * u = bu
        Eigen::VectorXd u = KuuSolver.solve(bu);
        if (KuuSolver.info() != Eigen::Success) { throw std::runtime_error("Kuu solve failed."); }

        // Compute rhs_p = bp - Kpu * u
        Eigen::VectorXd rhs_p = bp - Kpu * u;

        // Solve the Schur complement system
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> internal_solver;
        internal_solver.compute(schurComplement);
        if (internal_solver.info() != Eigen::Success) {
            throw std::runtime_error("Schur complement factorization failed.");
        }

        Eigen::VectorXd p = internal_solver.solve(rhs_p);
        if (internal_solver.info() != Eigen::Success) {
            throw std::runtime_error("Solving Schur complement system failed.");
        }

        // Reconstruct the full solution (combine u and p)
        Eigen::VectorXd fullSolution(b.size());
        fullSolution.head(sizeKuu) = u; // Assign solution for Kuu-part
        fullSolution.tail(sizeKpp) = p; // Assign solution for Kpp-part

        return fullSolution; // Return the full solution with the same size as the original system
    }

    std::vector<char> createMaskFromSparsity(const Eigen::SparseMatrix<double> &matrix) {
        Eigen::Index      totalSize = matrix.rows();
        std::vector<char> mask(totalSize, 0);       // Default to Kuu (0)
        Eigen::Index      halfSize = totalSize / 2; // Assuming the split is roughly halfway

        for (Eigen::Index i = 0; i < totalSize; ++i) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(matrix, i); it; ++it) {
                if ((it.row() < halfSize && it.col() >= halfSize) || (it.row() >= halfSize && it.col() < halfSize)) {
                    mask[it.row()] = 1; // Mark row for Kpp
                    mask[it.col()] = 1; // Mark column for Kpp
                }
            }
        }
        return mask;
    }
};
