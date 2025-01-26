// EvaluatorGPU.h
#pragma once
#include <Eigen/Sparse>

void sparseTriangularSolveGPU(const Eigen::SparseMatrix<double>& lhs,
                              Eigen::Matrix<double, -1, 1>& rhs, bool isLower,
                              double regularizationFactor);

template <typename Lhs, typename Rhs, int Mode>
void SparseSolveTriangularGPU(const Lhs& lhs, Rhs& other,
                              typename Lhs::Scalar regularizationFactor) {
    constexpr bool isLower = (Mode & Eigen::Lower) != 0;
    sparseTriangularSolveGPU(lhs.nestedExpression(), other, isLower,
                             regularizationFactor);
}
