#pragma once
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <queue>

template <typename MatrixType>
class CustomMatIterator {
public:
    using Scalar       = typename MatrixType::Scalar;
    using StorageIndex = typename MatrixType::StorageIndex;

    // Constructor: takes the matrix and the column to iterate over
    CustomMatIterator(const MatrixType &mat, StorageIndex col)
        : matrix_(mat), outerIndex_(matrix_.outerIndexPtr()[col]), // Start of the column in outer index
          end_(matrix_.outerIndexPtr()[col + 1]),                  // End of the column in outer index
          innerIndex_(matrix_.innerIndexPtr() + outerIndex_),      // Inner indices start for the column
          valuePtr_(matrix_.valuePtr() + outerIndex_)              // Non-zero values start for the column
    {}

    // Moves to the next non-zero element
    CustomMatIterator &operator++() {
        ++outerIndex_;
        ++innerIndex_;
        ++valuePtr_;

        // Prefetch a few steps ahead to improve cache performance
        __builtin_prefetch(innerIndex_ + 4);
        __builtin_prefetch(valuePtr_ + 4);

        return *this;
    }

    // Returns true if there are more elements in the current column
    operator bool() const { return outerIndex_ < end_; }

    // Access to the current non-zero element's row index
    StorageIndex row() const { return *innerIndex_; }

    // Access to the current non-zero element's value
    Scalar value() const { return *valuePtr_; }

    // Access to the current element's index
    StorageIndex index() const { return *innerIndex_; }

private:
    const MatrixType   &matrix_;     // Reference to the matrix
    StorageIndex        outerIndex_; // Current element in the column
    StorageIndex        end_;        // End of the current column
    const StorageIndex *innerIndex_; // Pointer to the current inner index
    const Scalar       *valuePtr_;   // Pointer to the current value
};

template <typename Lhs, typename Rhs, int Mode, bool IsLower, bool IsRowMajor>
struct SparseSolveTriangular {
    typedef typename Rhs::Scalar                                    Scalar;
    typedef Eigen::internal::evaluator<Lhs>                         LhsEval;
    typedef typename Eigen::internal::evaluator<Lhs>::InnerIterator LhsIterator;
    typedef typename Lhs::Index                                     Index;

    template <typename Scalar>
    static typename std::enable_if<std::is_floating_point<Scalar>::value, bool>::type is_exactly_zero(Scalar value) {
        return std::abs(value) < std::numeric_limits<Scalar>::epsilon();
    }

    // For integral types, check for exact zero.
    template <typename Scalar>
    static typename std::enable_if<std::is_integral<Scalar>::value, bool>::type is_exactly_zero(Scalar value) {
        return value == Scalar(0);
    }
    static void run(const Lhs &lhs, Rhs &other, Scalar regularizationFactor) {
        LhsEval lhsEval(lhs);

        for (Index col = 0; col < other.cols(); ++col) {
            if constexpr (IsRowMajor) {
                if constexpr (IsLower) {
                    for (Index i = 0; i < lhs.rows(); ++i) {
                        Scalar tmp       = other.coeff(i, col);
                        Scalar lastVal   = 0;
                        Index  lastIndex = 0;

                        for (LhsIterator it(lhsEval, i); it; ++it) {
                            lastVal   = it.value();
                            lastIndex = it.index();
                            if (lastIndex == i) break;
                            tmp -= lastVal * other.coeff(lastIndex, col);
                        }

                        if (Mode & Eigen::UnitDiag)
                            other.coeffRef(i, col) = tmp;
                        else {
                            if (std::abs(lastVal) < regularizationFactor) {
                                lastVal = (lastVal >= 0 ? regularizationFactor : -regularizationFactor);
                            }
                            other.coeffRef(i, col) = tmp / lastVal;
                        }
                    }
                } else {
                    for (Index i = lhs.rows() - 1; i >= 0; --i) {
                        Scalar tmp  = other.coeff(i, col);
                        Scalar l_ii = 0;

                        LhsIterator it(lhsEval, i);
                        while (it && it.index() < i) ++it;
                        if (!(Mode & Eigen::UnitDiag)) {
                            eigen_assert(it && it.index() == i);
                            l_ii = it.value();
                            ++it;
                        } else if (it && it.index() == i)
                            ++it;

                        for (; it; ++it) { tmp -= it.value() * other.coeff(it.index(), col); }

                        if (Mode & Eigen::UnitDiag)
                            other.coeffRef(i, col) = tmp;
                        else {
                            if (std::abs(l_ii) < regularizationFactor) {
                                l_ii = (l_ii >= 0 ? regularizationFactor : -regularizationFactor);
                            }
                            other.coeffRef(i, col) = tmp / l_ii;
                        }
                    }
                }
            } else { // Column-major case
                if constexpr (IsLower) {
                    for (Index i = 0; i < lhs.cols(); ++i) {
                        Scalar &tmp = other.coeffRef(i, col);
                        if (!is_exactly_zero(tmp)) {
                            LhsIterator it(lhsEval, i);
                            while (it && it.index() < i) ++it;
                            if (!(Mode & Eigen::UnitDiag)) {
                                constexpr Scalar epsilon = Scalar(1e-12); // Regularization threshold
                                tmp /= (std::abs(it.value()) < epsilon) ? it.value() + epsilon : it.value();
                            }
                            if (it && it.index() == i) ++it;
                            for (; it; ++it) { other.coeffRef(it.index(), col) -= tmp * it.value(); }
                        }
                    }
                } else {
                    for (Index i = lhs.cols() - 1; i >= 0; --i) {
                        Scalar &tmp = other.coeffRef(i, col);
                        if (!is_exactly_zero(tmp)) {
                            LhsIterator it(lhsEval, i);
                            if (!(Mode & Eigen::UnitDiag)) {
                                constexpr Scalar epsilon = Scalar(1e-12); // Regularization threshold
                                while (it && it.index() != i) ++it;
                                tmp /= (std::abs(it.value()) < epsilon) ? it.value() + epsilon : it.value();
                            }
                            for (; it && it.index() < i; ++it) { other.coeffRef(it.index(), col) -= tmp * it.value(); }
                        }
                    }
                }
            }
        }
    }
};

// Dispatch to the correct specialization
template <typename Lhs, typename Rhs, int Mode>
void solveTriangular(const Lhs &lhs, Rhs &other, typename Lhs::Scalar regularizationFactor) {
    constexpr bool isLower    = (Mode & Eigen::Lower) != 0;
    constexpr bool isRowMajor = (int(Lhs::Flags) & Eigen::RowMajorBit) != 0;

    SparseSolveTriangular<Lhs, Rhs, Mode, isLower, isRowMajor>::run(lhs, other, regularizationFactor);
}
