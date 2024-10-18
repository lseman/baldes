#ifndef EIGEN_CUSTOM_SIMPLICIAL_LDLT_H
#define EIGEN_CUSTOM_SIMPLICIAL_LDLT_H

#define EIGEN_USE_MKL_ALL

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <limits>

namespace Eigen {

template <typename MatrixType_, int UpLo_ = Lower, typename Ordering_ = AMDOrdering<typename MatrixType_::StorageIndex>>
class CustomSimplicialLDLT {
public:
    typedef MatrixType_ MatrixType;
    enum { UpLo = UpLo_ };
    typedef typename MatrixType::Scalar                  Scalar;
    typedef typename MatrixType::RealScalar              RealScalar;
    typedef typename MatrixType::StorageIndex            StorageIndex;
    typedef SparseMatrix<Scalar, ColMajor, StorageIndex> CholMatrixType;
    typedef Matrix<Scalar, Dynamic, 1>                   VectorType;

    typedef TriangularView<const CholMatrixType, Eigen::UnitLower>                             MatrixL;
    typedef TriangularView<const typename CholMatrixType::AdjointReturnType, Eigen::UnitUpper> MatrixU;
    typedef CholMatrixType const                                                              *ConstCholMatrixPtr;

public:
    CustomSimplicialLDLT()
        : m_isInitialized(false), m_info(Success), m_P(0), m_Pinv(0), m_matrix(1, 1), m_L(m_matrix),
          m_U(m_matrix.adjoint()), m_epsilon(1e-9), m_factorizationIsOk(false) {}

    template <typename Lhs, typename Rhs, int Mode, bool IsLower, bool IsRowMajor>
    struct SparseSolveTriangular;

    template <typename Scalar>
    static typename std::enable_if<std::is_floating_point<Scalar>::value, bool>::type is_exactly_zero(Scalar value) {
        return std::abs(value) < std::numeric_limits<Scalar>::epsilon();
    }

    // For integral types, check for exact zero.
    template <typename Scalar>
    static typename std::enable_if<std::is_integral<Scalar>::value, bool>::type is_exactly_zero(Scalar value) {
        return value == Scalar(0);
    }

    template <typename Lhs, typename Rhs, int Mode, bool IsLower, bool IsRowMajor>
    struct SparseSolveTriangular {
        typedef typename Rhs::Scalar                             Scalar;
        typedef internal::evaluator<Lhs>                         LhsEval;
        typedef typename internal::evaluator<Lhs>::InnerIterator LhsIterator;

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

                            if (Mode & UnitDiag)
                                other.coeffRef(i, col) = tmp;
                            else {
                                eigen_assert(lastIndex == i);
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
                            if (!(Mode & UnitDiag)) {
                                eigen_assert(it && it.index() == i);
                                l_ii = it.value();
                                ++it;
                            } else if (it && it.index() == i)
                                ++it;

                            for (; it; ++it) { tmp -= it.value() * other.coeff(it.index(), col); }

                            if (Mode & UnitDiag)
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
                                if (!(Mode & UnitDiag)) {
                                    eigen_assert(it && it.index() == i);
                                    tmp /= it.value();
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
                                if (!(Mode & UnitDiag)) {
                                    while (it && it.index() != i) ++it;
                                    eigen_assert(it && it.index() == i);
                                    tmp /= it.value();
                                }
                                for (; it && it.index() < i; ++it) {
                                    other.coeffRef(it.index(), col) -= tmp * it.value();
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
    void solveTriangular(const Lhs &lhs, Rhs &other, typename Lhs::Scalar regularizationFactor) const {
        constexpr bool isLower    = (Mode & Lower) != 0;
        constexpr bool isRowMajor = (int(Lhs::Flags) & RowMajorBit) != 0;

        SparseSolveTriangular<Lhs, Rhs, Mode, isLower, isRowMajor>::run(lhs, other, regularizationFactor);
    }

    explicit CustomSimplicialLDLT(const MatrixType &matrix) : m_isInitialized(false), m_info(Success), m_epsilon(1e-9) {
        compute(matrix);
    }

    CustomSimplicialLDLT &compute(const MatrixType &matrix) {
        analyzePattern(matrix);
        factorize_preordered<true, false>(m_matrix);
        return *this;
    }

    void analyzePattern_preordered(const CholMatrixType &ap, bool doLDLT) {
        const StorageIndex size = StorageIndex(ap.rows());
        m_matrix.resize(size, size);
        m_parent.resize(size);
        m_nonZerosPerCol.resize(size);

        // Using std::vector instead of VLAs for dynamic memory allocation
        std::vector<StorageIndex> tags(size, 0); // dynamically sized array for tags

        for (StorageIndex k = 0; k < size; ++k) {
            m_parent[k]         = -1; // Parent of k is not yet known
            tags[k]             = k;  // Mark node k as visited
            m_nonZerosPerCol[k] = 0;  // Count of nonzeros in column k of L

            for (typename CholMatrixType::InnerIterator it(ap, k); it; ++it) {
                StorageIndex i = it.index();
                if (i < k) {
                    // Follow path from i to root of the etree, stop at flagged node
                    for (; tags[i] != k; i = m_parent[i]) {
                        if (m_parent[i] == -1) m_parent[i] = k; // Set parent of i
                        m_nonZerosPerCol[i]++;                  // L(k,i) is nonzero
                        tags[i] = k;                            // Mark i as visited
                    }
                }
            }
        }

        // Construct Lp index array from m_nonZerosPerCol column counts
        StorageIndex *Lp = m_matrix.outerIndexPtr();
        Lp[0]            = 0;
        for (StorageIndex k = 0; k < size; ++k) { Lp[k + 1] = Lp[k] + m_nonZerosPerCol[k] + (doLDLT ? 0 : 1); }

        // Resize non-zeros for the matrix
        m_matrix.resizeNonZeros(Lp[size]);

        m_isInitialized     = true;
        m_info              = Success;
        m_analysisIsOk      = true;
        m_factorizationIsOk = false;
    }

    typedef typename MatrixType::RealScalar DiagonalScalar;
    static inline DiagonalScalar            getDiag(Scalar x) { return numext::real(x); }
    static inline Scalar                    getSymm(Scalar x) { return numext::conj(x); }
    template <bool DoLDLT, bool NonHermitian>
    void factorize_preordered(const CholMatrixType &ap) {
        using std::sqrt;

        eigen_assert(m_analysisIsOk && "You must first call analyzePattern()");
        eigen_assert(ap.rows() == ap.cols());
        eigen_assert(m_parent.size() == ap.rows());
        eigen_assert(m_nonZerosPerCol.size() == ap.rows());

        const StorageIndex  size = StorageIndex(ap.rows());
        const StorageIndex *Lp   = m_matrix.outerIndexPtr();
        StorageIndex       *Li   = m_matrix.innerIndexPtr();
        Scalar             *Lx   = m_matrix.valuePtr();

        // Use aligned arrays for SIMD optimization
        std::vector<Scalar, Eigen::aligned_allocator<Scalar>>             y(size, Scalar(0));
        std::vector<StorageIndex, Eigen::aligned_allocator<StorageIndex>> pattern(size, 0);
        std::vector<StorageIndex, Eigen::aligned_allocator<StorageIndex>> tags(size, 0);

        bool ok = true;
        m_diag.resize(DoLDLT ? size : 0);

// Parallelization strategy (optional, but experiment carefully)
#pragma omp parallel for
        for (StorageIndex k = 0; k < size; ++k) {
            y[k]                = Scalar(0);
            StorageIndex top    = size;
            tags[k]             = k;
            m_nonZerosPerCol[k] = 0;

            for (typename CholMatrixType::InnerIterator it(ap, k); it; ++it) {
                StorageIndex i = it.index();
                if (i <= k) {
                    y[i] += getSymm(it.value());

                    Index len;
#pragma GCC unroll 4 // Unroll the loop for cache-friendly access
                    for (len = 0; tags[i] != k; i = m_parent[i]) {
                        pattern[len++] = i;
                        tags[i]        = k;
                    }

                    while (len > 0) { pattern[--top] = pattern[--len]; }
                }
            }

            // Prefetch next iteration for y, Li, Lx
            if (k + 1 < size) {
                __builtin_prefetch(&y[k + 1], 1, 1);
                __builtin_prefetch(&Li[Lp[k + 1]], 1, 1);
                __builtin_prefetch(&Lx[Lp[k + 1]], 1, 1);
            }

            // Numerical solve for the current row
            DiagonalScalar d = getDiag(y[k]) * m_shiftScale + m_shiftOffset;
            y[k]             = Scalar(0); // Reset Y(k)

            for (; top < size; ++top) {
                Index  i  = pattern[top];
                Scalar yi = y[i];
                y[i]      = Scalar(0);

                Scalar l_ki = yi / getDiag(m_diag[i]);

                Index p2 = Lp[i] + m_nonZerosPerCol[i];
                for (Index p = Lp[i]; p < p2; ++p) { y[Li[p]] -= getSymm(Lx[p]) * yi; }

                d -= getDiag(l_ki * getSymm(yi));
                Li[p2] = k;
                Lx[p2] = l_ki;
                ++m_nonZerosPerCol[i];
            }

            m_diag[k] = d;
            if (d == RealScalar(0)) {
                ok = false;
                break;
            }
        }

        m_info              = ok ? Success : NumericalIssue;
        m_factorizationIsOk = true;
    }

    template <int SrcMode_, int DstMode_, bool NonHermitian, typename MatrixType, int DstOrder>
    void
    permute_symm_to_symm(const MatrixType                                                                       &mat,
                         SparseMatrix<typename MatrixType::Scalar, DstOrder, typename MatrixType::StorageIndex> &_dest,
                         const typename MatrixType::StorageIndex                                                *perm) {
        using StorageIndex = typename MatrixType::StorageIndex;
        using Scalar       = typename MatrixType::Scalar;
        SparseMatrix<Scalar, DstOrder, StorageIndex> &dest(_dest.derived());
        using VectorI     = Matrix<StorageIndex, Dynamic, 1>;
        using MatEval     = internal::evaluator<MatrixType>;
        using MatIterator = typename internal::evaluator<MatrixType>::InnerIterator;

        enum {
            SrcOrder          = MatrixType::IsRowMajor ? RowMajor : ColMajor,
            StorageOrderMatch = int(SrcOrder) == int(DstOrder),
            DstMode           = DstOrder == RowMajor ? (DstMode_ == Upper ? Lower : Upper) : DstMode_,
            SrcMode           = SrcOrder == RowMajor ? (SrcMode_ == Upper ? Lower : Upper) : SrcMode_
        };

        MatEval matEval(mat);
        Index   size = mat.rows();
        VectorI count(size);
        count.setZero();
        dest.resize(size, size);

        // First pass: Count the non-zero elements for each column/row
        for (StorageIndex j = 0; j < size; ++j) {
            StorageIndex jp = perm ? perm[j] : j;
            __builtin_prefetch(&jp, 0, 1); // Prefetch the index for permutation

            for (MatIterator it(matEval, j); it; ++it) {
                StorageIndex i = it.index();
                if ((int(SrcMode) == int(Lower) && i < j) || (int(SrcMode) == int(Upper) && i > j)) continue;

                StorageIndex ip = perm ? perm[i] : i;
                __builtin_prefetch(&ip, 0, 1); // Prefetch the index for permutation

                count[int(DstMode) == int(Lower) ? std::min(ip, jp) : std::max(ip, jp)]++;
            }
        }

        // Allocate space based on the counted non-zero entries
        dest.outerIndexPtr()[0] = 0;
        for (Index j = 0; j < size; ++j) { dest.outerIndexPtr()[j + 1] = dest.outerIndexPtr()[j] + count[j]; }
        dest.resizeNonZeros(dest.outerIndexPtr()[size]);

        // Reset counts for actual filling
        for (Index j = 0; j < size; ++j) { count[j] = dest.outerIndexPtr()[j]; }

        // Main loop: Populate the destination sparse matrix
        for (StorageIndex j = 0; j < size; ++j) {
            StorageIndex jp = perm ? perm[j] : j;
            __builtin_prefetch(&dest.innerIndexPtr()[count[j]], 1, 1); // Prefetch for insertion

            for (MatIterator it(matEval, j); it; ++it) {
                StorageIndex i = it.index();
                if ((int(SrcMode) == int(Lower) && i < j) || (int(SrcMode) == int(Upper) && i > j)) continue;

                StorageIndex jp = perm ? perm[j] : j;
                StorageIndex ip = perm ? perm[i] : i;

                Index k                 = count[int(DstMode) == int(Lower) ? std::min(ip, jp) : std::max(ip, jp)]++;
                dest.innerIndexPtr()[k] = int(DstMode) == int(Lower) ? std::max(ip, jp) : std::min(ip, jp);

                // Prefetch values for efficient memory access
                __builtin_prefetch(&dest.valuePtr()[k], 1, 1);

                if (!StorageOrderMatch) std::swap(ip, jp);
                if ((int(DstMode) == int(Lower) && ip < jp) || (int(DstMode) == int(Upper) && ip > jp)) {
                    dest.valuePtr()[k] = NonHermitian ? it.value() : numext::conj(it.value());
                } else {
                    dest.valuePtr()[k] = it.value();
                }
            }
        }
    }

    template <int Mode, bool NonHermitian, typename MatrixType, int DestOrder>
    void permute_symm_to_fullsymm(
        const MatrixType                                                                        &mat,
        SparseMatrix<typename MatrixType::Scalar, DestOrder, typename MatrixType::StorageIndex> &_dest,
        const typename MatrixType::StorageIndex                                                 *perm) {

        using StorageIndex = typename MatrixType::StorageIndex;
        using Scalar       = typename MatrixType::Scalar;
        using Dest         = SparseMatrix<Scalar, DestOrder, StorageIndex>;
        using VectorI      = Matrix<StorageIndex, Dynamic, 1>;
        using MatEval      = internal::evaluator<MatrixType>;
        using MatIterator  = typename internal::evaluator<MatrixType>::InnerIterator;

        MatEval matEval(mat);
        Dest   &dest(_dest.derived());

        enum { StorageOrderMatch = int(Dest::IsRowMajor) == int(MatrixType::IsRowMajor) };

        Index   size = mat.rows();
        VectorI count(size);
        count.setZero();
        dest.resize(size, size);

        // Cache-aligned iteration and prefetching for optimal memory access
        for (Index j = 0; j < size; ++j) {
            Index jp = perm ? perm[j] : j;
            __builtin_prefetch(&jp, 0, 1); // Prefetch perm[j]

            for (MatIterator it(matEval, j); it; ++it) {
                Index i  = it.index();
                Index r  = it.row();
                Index c  = it.col();
                Index ip = perm ? perm[i] : i;
                __builtin_prefetch(&ip, 0, 1); // Prefetch perm[i]

                if (Mode == int(Upper | Lower)) {
                    count[StorageOrderMatch ? jp : ip]++;
                } else if (r == c) {
                    count[ip]++;
                } else if ((Mode == Lower && r > c) || (Mode == Upper && r < c)) {
                    count[ip]++;
                    count[jp]++;
                }
            }
        }

        Index nnz = count.sum();

        // Reserve space and fill the outer index
        dest.resizeNonZeros(nnz);
        dest.outerIndexPtr()[0] = 0;
        for (Index j = 0; j < size; ++j) { dest.outerIndexPtr()[j + 1] = dest.outerIndexPtr()[j] + count[j]; }

        // Reset count for actual insertion
        for (Index j = 0; j < size; ++j) { count[j] = dest.outerIndexPtr()[j]; }

        // Copy data into destination matrix
        for (StorageIndex j = 0; j < size; ++j) {
            for (MatIterator it(matEval, j); it; ++it) {
                StorageIndex i = internal::convert_index<StorageIndex>(it.index());
                Index        r = it.row();
                Index        c = it.col();

                StorageIndex jp = perm ? perm[j] : j;
                StorageIndex ip = perm ? perm[i] : i;

                if (Mode == int(Upper | Lower)) {
                    Index k                 = count[StorageOrderMatch ? jp : ip]++;
                    dest.innerIndexPtr()[k] = StorageOrderMatch ? ip : jp;
                    dest.valuePtr()[k]      = it.value();
                    __builtin_prefetch(&dest.innerIndexPtr()[k + 1], 1, 1); // Prefetch next insertion
                } else if (r == c) {
                    Index k                 = count[ip]++;
                    dest.innerIndexPtr()[k] = ip;
                    dest.valuePtr()[k]      = it.value();
                    __builtin_prefetch(&dest.innerIndexPtr()[k + 1], 1, 1); // Prefetch next insertion
                } else if (((Mode & Lower) == Lower && r > c) || ((Mode & Upper) == Upper && r < c)) {
                    if (!StorageOrderMatch) std::swap(ip, jp);
                    Index k                 = count[jp]++;
                    dest.innerIndexPtr()[k] = ip;
                    dest.valuePtr()[k]      = it.value();
                    k                       = count[ip]++;
                    dest.innerIndexPtr()[k] = jp;
                    dest.valuePtr()[k]      = (NonHermitian ? it.value() : numext::conj(it.value()));
                    __builtin_prefetch(&dest.innerIndexPtr()[k + 1], 1, 1); // Prefetch next insertion
                }
            }
        }
    }

    template <bool NonHermitian>
    void ordering_local(const MatrixType &a, ConstCholMatrixPtr &pmat, CholMatrixType &ap) {
        eigen_assert(a.rows() == a.cols());
        const Index size = a.rows();
        pmat             = &ap;
        // Note that ordering methods compute the inverse permutation
        if (!internal::is_same<Ordering_, NaturalOrdering<Index>>::value) {
            {
                CholMatrixType C;
                permute_symm_to_fullsymm<UpLo, NonHermitian>(a, C, NULL);

                Ordering_ ordering;
                ordering(C, m_Pinv);
            }

            if (m_Pinv.size() > 0)
                m_P = m_Pinv.inverse();
            else
                m_P.resize(0);

            ap.resize(size, size);
            permute_symm_to_symm<UpLo, Upper, false>(a, ap, m_P.indices().data());
        } else {
            m_Pinv.resize(0);
            m_P.resize(0);
            if (int(UpLo) == int(Lower) || MatrixType::IsRowMajor) {
                // we have to transpose the lower part to to the upper one
                ap.resize(size, size);
                permute_symm_to_symm<UpLo, Upper, false>(a, ap, NULL);
            } else
                internal::simplicial_cholesky_grab_input<CholMatrixType, MatrixType>::run(a, pmat, ap);
        }
    }

    void analyzePattern(const MatrixType &a) {
        eigen_assert(a.rows() == a.cols());
        Index              size = a.cols();
        CholMatrixType     tmp(size, size);
        ConstCholMatrixPtr pmat;
        ordering_local<false>(a, pmat, tmp);
        analyzePattern_preordered(*pmat, true);
    }

    ComputationInfo info() const { return m_info; }

    const MatrixL matrixL() const { return m_L; }

    const MatrixU matrixU() const { return m_U; }
    template <typename Rhs>
    Eigen::VectorXd solve(const MatrixBase<Rhs> &b) const {
        eigen_assert(m_isInitialized && "Decomposition not initialized");

        Eigen::VectorXd dest;

        // Apply forward permutation (if needed)
        if (m_P.size() > 0) {
            dest = m_P * b;
        } else {
            dest = b;
        }

        const Scalar regularizationFactor = Scalar(1e-5); // Small regularization term

        // Solve L * y = P * b using SparseSolveTriangular for lower triangular matrix
        solveTriangular<decltype(matrixL()), decltype(dest), Lower>(matrixL(), dest, regularizationFactor);

        // Solve D * z = y (element-wise division)
        // Use Eigen's array-based division for vectorized operations
        dest.array() /= m_diag.array();

        // Solve U * x = z using SparseSolveTriangular for upper triangular matrix
        solveTriangular<decltype(matrixU()), decltype(dest), Upper>(matrixU(), dest, regularizationFactor);

        // Apply backward permutation (if needed)
        if (m_Pinv.size() > 0) { dest = m_Pinv * dest; }

        return dest;
    }

    Scalar determinant() const {
        eigen_assert(m_isInitialized && "Decomposition not initialized");
        Scalar det = Scalar(1);
        for (int i = 0; i < m_diag.size(); ++i) {
            det *= m_diag(i);
            if (std::abs(m_diag(i)) < m_epsilon) {
                throw std::runtime_error("Near-zero value encountered while computing determinant.");
            }
        }
        return det;
    }

    void setRegularization(Scalar epsilon) { m_epsilon = epsilon; }

    void factorize(const MatrixType &a) {
        bool DoLDLT       = true;
        bool NonHermitian = false;
        eigen_assert(a.rows() == a.cols());
        Index              size = a.cols();
        CholMatrixType     tmp(size, size);
        ConstCholMatrixPtr pmat;

        if (m_P.size() == 0 && (int(UpLo) & int(Upper)) == Upper) {
            // If there is no ordering, try to directly use the input matrix without any copy
            internal::simplicial_cholesky_grab_input<CholMatrixType, MatrixType>::run(a, pmat, tmp);
        } else {
            permute_symm_to_symm<UpLo, Upper, false>(a, tmp, m_P.indices().data());
            pmat = &tmp;
        }

        factorize_preordered<true, false>(*pmat);
    }

private:
    CholMatrixType                                    m_matrix;
    VectorType                                        m_diag;
    MatrixL                                           m_L;
    MatrixU                                           m_U;
    PermutationMatrix<Dynamic, Dynamic, StorageIndex> m_P, m_Pinv;
    std::vector<StorageIndex>                         m_nonZerosPerCol;
    std::vector<StorageIndex>                         m_parent;
    bool                                              m_isInitialized;
    bool                                              m_analysisIsOk;
    bool                                              m_factorizationIsOk;
    ComputationInfo                                   m_info;
    Scalar                                            m_epsilon; // Regularization parameter for numerical stability
    Scalar                                            m_shiftScale  = Scalar(1);
    Scalar                                            m_shiftOffset = Scalar(0);
};

} // namespace Eigen

#endif // EIGEN_CUSTOM_SIMPLICIAL_LDLT_H
