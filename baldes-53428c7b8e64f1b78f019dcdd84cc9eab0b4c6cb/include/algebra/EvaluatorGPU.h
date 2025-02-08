// SparseSolveTriangularGPU.h

#pragma once
#include <Eigen/Sparse>

using Scalar = double;
// Template declaration for the solver
template <typename Lhs, typename Rhs, int Mode>
void SparseSolveTriangularGPU(const Lhs &lhs, Rhs &other, typename Lhs::Scalar regularizationFactor) {
    constexpr bool isLower    = (Mode & Eigen::Lower) != 0;
    constexpr bool isRowMajor = (int(Lhs::Flags) & Eigen::RowMajorBit) != 0;

    if (isRowMajor) {
        if (isLower) {
            // Row-major, lower triangular solve
            sparseLowerTriangularSolveGPU_RowMajor(lhs, other, regularizationFactor);
        } else {
            // Row-major, upper triangular solve
            sparseUpperTriangularSolveGPU_RowMajor(lhs, other, regularizationFactor);
        }
    } else {
        if (isLower) {
            // Column-major, lower triangular solve
            sparseLowerTriangularSolveGPU_ColMajor(lhs, other, regularizationFactor);
        } else {
            // Column-major, upper triangular solve
            sparseUpperTriangularSolveGPU_ColMajor(lhs, other, regularizationFactor);
        }
    }
}
