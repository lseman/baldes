#pragma once
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <queue>

template <typename MatrixType>
class CustomMatIterator {
public:
    using Scalar = typename MatrixType::Scalar;
    using StorageIndex = typename MatrixType::StorageIndex;
    
    // Prefetch distance tuned for common L1 cache line sizes
    static constexpr int PREFETCH_DISTANCE = 8;
    
    // Constructor with immediate prefetching of first elements
    EIGEN_STRONG_INLINE CustomMatIterator(const MatrixType& mat, StorageIndex col)
        : matrix_(mat)
        , outerIndex_(matrix_.outerIndexPtr()[col])
        , end_(matrix_.outerIndexPtr()[col + 1])
        , innerIndex_(matrix_.innerIndexPtr() + outerIndex_)
        , valuePtr_(matrix_.valuePtr() + outerIndex_)
    {
        // Prefetch the first chunk of data
        prefetchNext();
    }

    // Optimized increment with smart prefetching
    EIGEN_STRONG_INLINE CustomMatIterator& operator++() {
        ++outerIndex_;
        ++innerIndex_;
        ++valuePtr_;
        
        // Only prefetch if we're not near the end
        if (outerIndex_ + PREFETCH_DISTANCE < end_) {
            prefetchNext();
        }
        
        return *this;
    }

    // Fast validity check
    EIGEN_STRONG_INLINE operator bool() const { 
        return outerIndex_ < end_; 
    }

    // Direct accessors marked for inlining
    EIGEN_STRONG_INLINE StorageIndex row() const { 
        return *innerIndex_; 
    }
    
    EIGEN_STRONG_INLINE Scalar value() const { 
        return *valuePtr_; 
    }
    
    EIGEN_STRONG_INLINE StorageIndex index() const { 
        return *innerIndex_; 
    }

private:
    // Efficient prefetching strategy
    EIGEN_STRONG_INLINE void prefetchNext() {
        // Prefetch both indices and values with temporal locality hint
        #ifdef __GNUC__
            __builtin_prefetch(innerIndex_ + PREFETCH_DISTANCE, 0, 3);  // Read, high temporal locality
            __builtin_prefetch(valuePtr_ + PREFETCH_DISTANCE, 0, 3);    // Read, high temporal locality
            
            // If we're not near the end, prefetch the next cache line too
            if (outerIndex_ + 2 * PREFETCH_DISTANCE < end_) {
                __builtin_prefetch(innerIndex_ + 2 * PREFETCH_DISTANCE, 0, 2);  // Read, moderate temporal locality
                __builtin_prefetch(valuePtr_ + 2 * PREFETCH_DISTANCE, 0, 2);    // Read, moderate temporal locality
            }
        #endif
    }

    const MatrixType& matrix_;
    StorageIndex outerIndex_;
    StorageIndex end_;
    const StorageIndex* innerIndex_;
    const Scalar* valuePtr_;
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

    // Main function to solve the triangular system
    static void run(const Lhs &lhs, Rhs &other, Scalar regularizationFactor) {
        LhsEval     lhsEval(lhs);
        const Index n = other.cols();
        const Index m = lhs.rows();

        if constexpr (IsRowMajor) {
            for (Index col = 0; col < n; ++col) {
                if constexpr (IsLower) {
                    // Forward substitution
                    for (Index i = 0; i < m; ++i) {
                        Scalar tmp      = other.coeff(i, col);
                        Scalar diag_val = 0;

                        // Gather phase - accumulate off-diagonal contributions
                        for (LhsIterator it(lhsEval, i); it; ++it) {
                            const Index  idx = it.index();
                            const Scalar val = it.value();

                            if (idx == i) {
                                diag_val = val;
                                break;
                            }
                            tmp -= val * other.coeff(idx, col);
                        }

                        // Handle diagonal
                        if (!(Mode & Eigen::UnitDiag)) {
                            if (std::abs(diag_val) < regularizationFactor) {
                                diag_val = (diag_val >= 0) ? regularizationFactor : -regularizationFactor;
                            }
                            tmp /= diag_val;
                        }
                        other.coeffRef(i, col) = tmp;
                    }
                } else {
                    // Backward substitution
                    for (Index i = m - 1; i >= 0; --i) {
                        Scalar tmp        = other.coeff(i, col);
                        Scalar diag_val   = 0;
                        bool   found_diag = false;

                        // Find diagonal and accumulate
                        for (LhsIterator it(lhsEval, i); it; ++it) {
                            const Index  idx = it.index();
                            const Scalar val = it.value();

                            if (!found_diag) {
                                if (idx == i) {
                                    diag_val   = val;
                                    found_diag = true;
                                    continue;
                                }
                            } else {
                                tmp -= val * other.coeff(idx, col);
                            }
                        }

                        // Handle diagonal
                        if (!(Mode & Eigen::UnitDiag)) {
                            if (std::abs(diag_val) < regularizationFactor) {
                                diag_val = (diag_val >= 0) ? regularizationFactor : -regularizationFactor;
                            }
                            tmp /= diag_val;
                        }
                        other.coeffRef(i, col) = tmp;
                    }
                }
            }
        } else {
            // Column-major optimization
            for (Index col = 0; col < n; ++col) {
                if constexpr (IsLower) {
                    for (Index i = 0; i < m; ++i) {
                        Scalar &tmp = other.coeffRef(i, col);
                        if (!is_exactly_zero(tmp)) {
                            // Find diagonal and handle in one pass
                            LhsIterator it(lhsEval, i);
                            while (it && it.index() < i) ++it;

                            if (!(Mode & Eigen::UnitDiag)) {
                                const Scalar diag_val = it.value();
                                const Scalar reg_val =
                                    std::abs(diag_val) < regularizationFactor
                                        ? (diag_val >= 0 ? regularizationFactor : -regularizationFactor)
                                        : diag_val;
                                tmp /= reg_val;
                                ++it;
                            }

                            // Scatter phase - update remaining entries
                            for (; it; ++it) { other.coeffRef(it.index(), col) -= tmp * it.value(); }
                        }
                    }
                } else {
                    for (Index i = m - 1; i >= 0; --i) {
                        Scalar &tmp = other.coeffRef(i, col);
                        if (!is_exactly_zero(tmp)) {
                            if (!(Mode & Eigen::UnitDiag)) {
                                // Find diagonal efficiently
                                LhsIterator it(lhsEval, i);
                                while (it && it.index() != i) ++it;

                                const Scalar diag_val = it.value();
                                const Scalar reg_val =
                                    std::abs(diag_val) < regularizationFactor
                                        ? (diag_val >= 0 ? regularizationFactor : -regularizationFactor)
                                        : diag_val;
                                tmp /= reg_val;
                                ++it;

                                // Update remaining entries
                                for (; it; ++it) { other.coeffRef(it.index(), col) -= tmp * it.value(); }
                            } else {
                                // No diagonal division needed
                                for (LhsIterator it(lhsEval, i); it && it.index() < i; ++it) {
                                    other.coeffRef(it.index(), col) -= tmp * it.value();
                                }
                            }
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
