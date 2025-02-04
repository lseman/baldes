#ifndef EIGEN_CUSTOM_SIMPLICIAL_LDLT_H
#define EIGEN_CUSTOM_SIMPLICIAL_LDLT_H

#include <vector>
#define EIGEN_USE_MKL_ALL

#include "Evaluator.h"
// #include "EvaluatorGPU.h"

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <execution>
#include <experimental/simd>
#include <limits>
#include <set>
#include <stdexec/execution.hpp>

#include "AMD.h"
#include "EvaluatorGPU.h"
template <typename T>
using BVector = std::vector<T>;

class Permutation {
   public:
    std::vector<int> perm;

    Permutation(int size) : perm(size) {
        for (int i = 0; i < size; ++i) perm[i] = i;
    }

    // Apply the permutation to a vector
    template <typename T>
    std::vector<T> apply(const std::vector<T> &vec) const {
        std::vector<T> result(vec.size());
        for (int i = 0; i < vec.size(); ++i) {
            result[i] = vec[perm[i]];
        }
        return result;
    }

    // Invert the permutation
    Permutation inverse() const {
        Permutation inv_perm(perm.size());
        for (int i = 0; i < perm.size(); ++i) {
            inv_perm.perm[perm[i]] = i;
        }
        return inv_perm;
    }
};

namespace Eigen {

template <typename MatrixType_, int UpLo_ = Lower,
          typename Ordering_ = AMDOrdering<typename MatrixType_::StorageIndex>>
class CustomSimplicialLDLT {
   public:
    typedef MatrixType_ MatrixType;
    enum { UpLo = UpLo_ };
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::RealScalar RealScalar;
    typedef typename MatrixType::StorageIndex StorageIndex;
    typedef SparseMatrix<Scalar, ColMajor, StorageIndex> CholMatrixType;
    typedef Matrix<Scalar, Dynamic, 1> VectorType;

    typedef TriangularView<const CholMatrixType, Eigen::UnitLower> MatrixL;
    typedef TriangularView<const typename CholMatrixType::AdjointReturnType,
                           Eigen::UnitUpper>
        MatrixU;
    typedef CholMatrixType const *ConstCholMatrixPtr;

    CustomSimplicialLDLT()
        : m_isInitialized(false),
          m_info(Success),
          m_P(0),
          m_Pinv(0),
          m_matrix(1, 1),
          m_L(m_matrix),
          m_U(m_matrix.adjoint()),
          m_epsilon(1e-9),
          m_factorizationIsOk(false) {}

    template <typename Lhs, typename Rhs, int Mode, bool IsLower,
              bool IsRowMajor>
    struct SparseSolveTriangular;

    explicit CustomSimplicialLDLT(const MatrixType &matrix)
        : m_isInitialized(false), m_info(Success), m_epsilon(1e-9) {
        // compute(matrix);
    }

    exec::static_thread_pool pool =
        exec::static_thread_pool(std::thread::hardware_concurrency());
    exec::static_thread_pool::scheduler sched = pool.get_scheduler();

    CustomSimplicialLDLT &compute(const MatrixType &matrix) {
        analyzePattern(matrix);
        factorize_preordered<true, false>(m_matrix);
        return *this;
    }

    void analyzePattern_preordered(const CholMatrixType &ap, bool doLDLT) {
        const StorageIndex size = StorageIndex(ap.rows());

        // Pre-allocate all vectors at once with appropriate sizes
        m_matrix.resize(size, size);
        m_parent.resize(size);
        m_nonZerosPerCol.resize(size);
        std::vector<StorageIndex> tags(size);

        // Initialize arrays - use memset for better performance on POD types
        std::memset(m_parent.data(), -1, size * sizeof(StorageIndex));
        std::memset(m_nonZerosPerCol.data(), 0, size * sizeof(StorageIndex));

        // Compute elimination tree and count nonzeros per column
        for (StorageIndex k = 0; k < size; ++k) {
            tags[k] = k;  // Mark node k as visited

            // Traverse column k using iterator
            for (typename CholMatrixType::InnerIterator it(ap, k); it; ++it) {
                StorageIndex i = it.index();
                if (i < k) {
                    // Use local variables to reduce memory access
                    StorageIndex current = i;
                    StorageIndex parent;

                    // Follow path from i to root of etree
                    while (tags[current] != k) {
                        parent = m_parent[current];
                        if (parent == -1) {
                            m_parent[current] = k;
                            parent = k;
                        }
                        m_nonZerosPerCol[current]++;
                        tags[current] = k;
                        current = parent;
                    }
                }
            }
        }

        // Build column pointers array
        StorageIndex *Lp = m_matrix.outerIndexPtr();
        StorageIndex running_total = 0;

        // Use prefix sum for better cache utilization
        Lp[0] = 0;
        for (StorageIndex k = 0; k < size; ++k) {
            running_total += m_nonZerosPerCol[k];
            Lp[k + 1] = running_total;
        }

        // Allocate space for non-zeros
        m_matrix.resizeNonZeros(running_total);

        // Set status flags
        m_isInitialized = true;
        m_info = Success;
        m_analysisIsOk = true;
        m_factorizationIsOk = false;
    }
    typedef typename MatrixType::RealScalar DiagonalScalar;
    static inline DiagonalScalar getDiag(Scalar x) { return numext::real(x); }
    static inline Scalar getSymm(Scalar x) { return numext::conj(x); }

    void reset() {
        m_isInitialized = false;
        m_analysisIsOk = false;
        m_factorizationIsOk = false;
        m_info = Success;
        patternAnalyzed = false;
    }

    bool patternAnalyzed = false;
    void factorizeMatrix(const MatrixType &matrix) {
        auto matrixToFactorize = matrix;
        if (!patternAnalyzed) {
            analyzePattern(matrixToFactorize);  // Analyze the sparsity pattern
            patternAnalyzed = true;
        }
        factorize(matrixToFactorize);
    }

    template <bool DoLDLT, bool NonHermitian>
    void factorize_preordered(const CholMatrixType &ap) {
        const StorageIndex size = StorageIndex(ap.rows());
        const StorageIndex *Lp = m_matrix.outerIndexPtr();
        StorageIndex *Li = m_matrix.innerIndexPtr();
        Scalar *Lx = m_matrix.valuePtr();

        // Aligned vectors for better cache performance
        alignas(64) std::vector<Scalar> y(size, Scalar(0));
        alignas(64) std::vector<StorageIndex> pattern(size, 0);
        alignas(64) std::vector<StorageIndex> tags(size, 0);

        m_diag.resize(size);

        // Preallocate thread-local storage
        const int hardware_threads = std::thread::hardware_concurrency();
        std::vector<std::vector<Scalar>> y_local_storage(
            hardware_threads, std::vector<Scalar>(size, Scalar(0)));

        // Precompute patterns for each column
        std::vector<std::vector<StorageIndex>> column_patterns(size);
        for (StorageIndex k = 0; k < size; ++k) {
            StorageIndex top = size;
            tags[k] = k;

            for (typename CholMatrixType::InnerIterator it(ap, k); it; ++it) {
                StorageIndex i = it.index();
                if (i <= k) {
                    StorageIndex len = 0;
                    for (; tags[i] != k; i = m_parent[i]) {
                        pattern[len++] = i;
                        tags[i] = k;
                        if (m_parent[i] == -1) break;
                    }
                    while (len > 0) {
                        pattern[--top] = pattern[--len];
                    }
                }
            }
            column_patterns[k].assign(pattern.begin() + top, pattern.end());
        }
        std::vector<bool> thread_ok(hardware_threads, true);
        // Use bulk sender for parallelism
        auto bulk_sender = stdexec::bulk(
            stdexec::just(), size,
            [this, &y, &tags, Lp, Li, Lx, size, &ap, &y_local_storage,
             hardware_threads, &thread_ok, &column_patterns](std::size_t k) {
                size_t thread_id =
                    std::hash<std::thread::id>{}(std::this_thread::get_id()) %
                    hardware_threads;
                auto &y_local = y_local_storage[thread_id];

                tags[k] = k;
                m_nonZerosPerCol[k] = 0;

                // Process column k using local buffer
                for (typename CholMatrixType::InnerIterator it(ap, k); it;
                     ++it) {
                    StorageIndex i = it.index();
                    if (i <= k) {
                        y_local[i] += getSymm(it.value());
                    }
                }

                DiagonalScalar d =
                    getDiag(y_local[k]) * m_shiftScale + m_shiftOffset;
                y_local[k] = Scalar(0);

                const auto &pattern = column_patterns[k];
                for (StorageIndex i : pattern) {
                    const Scalar yi = y_local[i];
                    y_local[i] = Scalar(0);

                    if (yi != Scalar(0)) {
                        const Scalar diag_i =
                            getDiag(m_diag[i]);  // Cache this value
                        const Scalar l_ki = yi / diag_i;
                        const StorageIndex p2 = Lp[i] + m_nonZerosPerCol[i];

                        // Sparse update with potential SIMD optimization
                        for (StorageIndex p = Lp[i]; p < p2; ++p) {
                            y_local[Li[p]] -= getSymm(Lx[p]) * yi;
                        }

                        d -= getDiag(l_ki * getSymm(yi));
                        Li[p2] = k;
                        Lx[p2] = l_ki;
                        ++m_nonZerosPerCol[i];
                    }
                }

                m_diag[k] = d;
                if (d == RealScalar(0)) {
                    thread_ok[thread_id] = false;
                }
            });

        stdexec::sync_wait(stdexec::when_all(bulk_sender));
        // Aggregate thread results
        bool all_ok = true;
        for (bool t_ok : thread_ok) {
            if (!t_ok) {
                all_ok = false;
                break;
            }
        }

        m_info = all_ok ? Success : NumericalIssue;
        m_factorizationIsOk = true;
    }

    template <int SrcMode_, int DstMode_, bool NonHermitian,
              typename MatrixType, int DstOrder>
    void permute_symm_to_symm(
        const MatrixType &mat,
        SparseMatrix<typename MatrixType::Scalar, DstOrder,
                     typename MatrixType::StorageIndex> &_dest,
        const typename MatrixType::StorageIndex *perm) {
        using StorageIndex = typename MatrixType::StorageIndex;
        using Scalar = typename MatrixType::Scalar;
        SparseMatrix<Scalar, DstOrder, StorageIndex> &dest(_dest.derived());
        using VectorI = Matrix<StorageIndex, Dynamic, 1>;
        using MatEval = internal::evaluator<MatrixType>;
        using MatIterator = CustomMatIterator<MatrixType>;

        enum {
            SrcOrder = MatrixType::IsRowMajor ? RowMajor : ColMajor,
            StorageOrderMatch = int(SrcOrder) == int(DstOrder),
            DstMode = DstOrder == RowMajor ? (DstMode_ == Upper ? Lower : Upper)
                                           : DstMode_,
            SrcMode = SrcOrder == RowMajor ? (SrcMode_ == Upper ? Lower : Upper)
                                           : SrcMode_
        };

        MatEval matEval(mat);
        Index size = mat.rows();
        VectorI count(size);
        count.setZero();
        dest.resize(size, size);

        const bool isLower = int(SrcMode) == int(Lower);
        const bool isUpper = int(SrcMode) == int(Upper);
        const bool isDstLower = int(DstMode) == int(Lower);

        // Precompute permutation
        std::vector<StorageIndex> perm_cache(size);
        for (StorageIndex j = 0; j < size; ++j) {
            perm_cache[j] = perm ? perm[j] : j;
        }

        // First pass: Count the non-zero elements for each column/row
        for (StorageIndex j = 0; j < size; ++j) {
            StorageIndex jp = perm_cache[j];

            for (MatIterator it(matEval, j); it; ++it) {
                StorageIndex i = it.index();
                if ((isLower && i < j) || (isUpper && i > j)) continue;

                StorageIndex ip = perm_cache[i];

                // Minimize conditional checks
                StorageIndex min_ip_jp = std::min(ip, jp);
                StorageIndex max_ip_jp = std::max(ip, jp);

                count[isDstLower ? min_ip_jp : max_ip_jp]++;
            }
        }

        // Allocate space based on the counted non-zero entries
        dest.outerIndexPtr()[0] = 0;
        for (Index j = 0; j < size; ++j) {
            dest.outerIndexPtr()[j + 1] = dest.outerIndexPtr()[j] + count[j];
        }
        dest.resizeNonZeros(dest.outerIndexPtr()[size]);

        // Reset counts for actual filling
        for (Index j = 0; j < size; ++j) {
            count[j] = dest.outerIndexPtr()[j];
        }

        // Main loop: Populate the destination sparse matrix
        for (StorageIndex j = 0; j < size; ++j) {
            StorageIndex jp = perm_cache[j];

            for (MatIterator it(matEval, j); it; ++it) {
                StorageIndex i = it.index();
                if ((isLower && i < j) || (isUpper && i > j)) continue;

                StorageIndex ip = perm_cache[i];

                // Minimize conditional checks
                StorageIndex min_ip_jp = std::min(ip, jp);
                StorageIndex max_ip_jp = std::max(ip, jp);

                Index k = count[isDstLower ? min_ip_jp : max_ip_jp]++;
                dest.innerIndexPtr()[k] = isDstLower ? max_ip_jp : min_ip_jp;

                if (!StorageOrderMatch) std::swap(ip, jp);
                if ((isDstLower && ip < jp) || (!isDstLower && ip > jp)) {
                    dest.valuePtr()[k] =
                        NonHermitian ? it.value() : numext::conj(it.value());
                } else {
                    dest.valuePtr()[k] = it.value();
                }
            }
        }
    }

    template <int Mode, bool NonHermitian, typename MatrixType, int DestOrder>
    void permute_symm_to_fullsymm(
        const MatrixType &mat,
        SparseMatrix<typename MatrixType::Scalar, DestOrder,
                     typename MatrixType::StorageIndex> &_dest,
        const typename MatrixType::StorageIndex *perm) {
        using StorageIndex = typename MatrixType::StorageIndex;
        using Scalar = typename MatrixType::Scalar;
        using Dest = SparseMatrix<Scalar, DestOrder, StorageIndex>;
        using VectorI = Matrix<StorageIndex, Dynamic, 1>;
        using MatEval = internal::evaluator<MatrixType>;
        using MatIterator =
            typename internal::evaluator<MatrixType>::InnerIterator;

        MatEval matEval(mat);
        Dest &dest(_dest.derived());

        enum {
            StorageOrderMatch =
                int(Dest::IsRowMajor) == int(MatrixType::IsRowMajor)
        };

        Index size = mat.rows();
        VectorI count(size);
        count.setZero();
        dest.resize(size, size);

        // Precompute permutation
        std::vector<StorageIndex> perm_cache(size);
        for (Index j = 0; j < size; ++j) {
            perm_cache[j] = perm ? perm[j] : j;
        }

        // First pass: Count non-zeros for each column
        for (Index j = 0; j < size; ++j) {
            Index jp = perm_cache[j];

            for (MatIterator it(matEval, j); it; ++it) {
                Index i = it.index();
                Index r = it.row();
                Index c = it.col();
                Index ip = perm_cache[i];

                if constexpr (Mode == int(Upper | Lower)) {
                    count[StorageOrderMatch ? jp : ip]++;
                } else if (r == c) {
                    count[ip]++;
                } else if ((Mode == Lower && r > c) ||
                           (Mode == Upper && r < c)) {
                    count[ip]++;
                    count[jp]++;
                }
            }
        }

        Index nnz = count.sum();

        // Resize for non-zeros and fill outer index
        dest.resizeNonZeros(nnz);
        dest.outerIndexPtr()[0] = 0;

        // Unrolling this loop for small loop performance gain
        for (Index j = 0; j < size; ++j) {
            dest.outerIndexPtr()[j + 1] = dest.outerIndexPtr()[j] + count[j];
        }

        // Reset count for actual insertion
        for (Index j = 0; j < size; ++j) {
            count[j] = dest.outerIndexPtr()[j];
        }

        // Second pass: Copy data into destination matrix
        for (StorageIndex j = 0; j < size; ++j) {
            for (MatIterator it(matEval, j); it; ++it) {
                StorageIndex i =
                    internal::convert_index<StorageIndex>(it.index());
                Index r = it.row();
                Index c = it.col();

                StorageIndex jp = perm_cache[j];
                StorageIndex ip = perm_cache[i];

                if constexpr (Mode == int(Upper | Lower)) {
                    Index k = count[StorageOrderMatch ? jp : ip]++;
                    dest.innerIndexPtr()[k] = StorageOrderMatch ? ip : jp;
                    dest.valuePtr()[k] = it.value();
                } else if (r == c) {
                    Index k = count[ip]++;
                    dest.innerIndexPtr()[k] = ip;
                    dest.valuePtr()[k] = it.value();
                } else if (((Mode & Lower) == Lower && r > c) ||
                           ((Mode & Upper) == Upper && r < c)) {
                    if (!StorageOrderMatch) std::swap(ip, jp);
                    Index k = count[jp]++;
                    dest.innerIndexPtr()[k] = ip;
                    dest.valuePtr()[k] = it.value();
                    k = count[ip]++;
                    dest.innerIndexPtr()[k] = jp;
                    dest.valuePtr()[k] =
                        (NonHermitian ? it.value() : numext::conj(it.value()));
                }
            }
        }
    }

    template <bool NonHermitian>
    void ordering_local(const MatrixType &a, ConstCholMatrixPtr &pmat,
                        CholMatrixType &ap) {
        const Index size = a.rows();
        pmat = &ap;

        // Step 1: Permute the input matrix to full symmetric form
        CholMatrixType C;
        permute_symm_to_fullsymm<UpLo, NonHermitian>(a, C, nullptr);

        // Step 2: Compute the symmetric pattern A + A^T
        CholMatrixType symm;
        internal::ordering_helper_at_plus_a(C, symm);

        // Step 3: Compute the minimum degree ordering
        internal::minimum_degree_ordering(symm, m_Pinv);

        // Step 4: Compute the inverse permutation
        m_P = m_Pinv.inverse();

        // Step 5: Permute the original matrix using the computed ordering
        ap.resize(size, size);
        permute_symm_to_symm<UpLo, Upper, false>(a, ap, m_P.indices().data());
    }

    void analyzePattern(const MatrixType &a) {
        Index size = a.cols();

        CholMatrixType tmp(size, size);
        ConstCholMatrixPtr pmat;
        ordering_local<false>(a, pmat, tmp);
        analyzePattern_preordered(*pmat, true);
    }

    StorageIndex m_size;
    ComputationInfo info() const { return m_info; }
    const MatrixL matrixL() const { return m_L; }
    const MatrixU matrixU() const { return m_U; }

    template <typename Rhs>
    Eigen::VectorXd solve(const MatrixBase<Rhs> &b) const {
        eigen_assert(m_isInitialized && "Decomposition not initialized.");
        Eigen::VectorXd dest;

        // Apply forward permutation
        if (m_P.size() > 0) {
            dest.noalias() = m_P * b;
        } else {
            dest = b;
        }

        // Compute adaptive regularization based on matrix properties
        const Scalar condition_estimate =
            m_diag.maxCoeff() /
            (m_diag.minCoeff() + std::numeric_limits<Scalar>::epsilon());

        // Base regularization scaled by condition number and problem size
        const Scalar base_reg = std::numeric_limits<Scalar>::epsilon() *
                                std::sqrt(static_cast<double>(b.size()));

        // Adaptive regularization increases with condition number
        Scalar adaptiveRegularization = base_reg;
        if (condition_estimate > 1e6) {
            adaptiveRegularization *= std::log10(condition_estimate);
        }

        // Create regularized diagonal
        Eigen::VectorXd diagCopy = m_diag;
        if (condition_estimate > 1e6) {
            Scalar tikhonov_param =
                base_reg * std::pow(condition_estimate, 0.1);
            diagCopy.array() += tikhonov_param;
        }

        // Forward substitution with regularization
        dest = (matrixL().template triangularView<Lower>()).solve(dest);
        dest.array() *=
            (Scalar(1.0) / (diagCopy.array() + adaptiveRegularization));

        // Backward substitution with regularization
        dest = (matrixU().template triangularView<Upper>()).solve(dest);

        // Apply backward permutation
        if (m_Pinv.size() > 0) {
            dest.noalias() = m_Pinv * dest;
        }

        return dest;
    }
    // Eigen::VectorXd solve(const MatrixBase<Rhs> &b) const {
    //     eigen_assert(m_isInitialized && "Decomposition not initialized.");
    //     Eigen::VectorXd dest;

    //     if (m_P.size() > 0) {
    //         dest.noalias() = m_P * b;
    //     } else {
    //         dest = b;
    //     }

    //     static Scalar cachedDiagonalNorm = m_diag.norm();
    //     const Scalar residualNorm = (matrixL() * dest - b).norm();
    //     const Scalar adaptiveRegularization = std::max(
    //         Scalar(1e-12),
    //         std::min(Scalar(1e-6),
    //                  residualNorm / (cachedDiagonalNorm + residualNorm)));

    //     // GPU triangular solves
    //     SparseSolveTriangularGPU<decltype(matrixL()), decltype(dest), Lower>(
    //         matrixL(), dest, adaptiveRegularization);

    //     dest.array() /= m_diag.array();

    //     SparseSolveTriangularGPU<decltype(matrixU()), decltype(dest), Upper>(
    //         matrixU(), dest, adaptiveRegularization);

    //     if (m_Pinv.size() > 0) {
    //         dest.noalias() = m_Pinv * dest;
    //     }

    //     return dest;
    // }

    void setRegularization(Scalar epsilon) { m_epsilon = epsilon; }

    bool isFactorized = false;
    void factorize(const MatrixType &a) {
        bool DoLDLT = true;
        bool NonHermitian = false;
        eigen_assert(a.rows() == a.cols());
        Index size = a.cols();
        CholMatrixType tmp(size, size);
        ConstCholMatrixPtr pmat;

        permute_symm_to_symm<UpLo, Upper, false>(a, tmp, m_P.indices().data());
        pmat = &tmp;

        // if (!isFactorized) {
        factorize_preordered<true, false>(*pmat);
        isFactorized = true;
    }

   private:
    CholMatrixType m_matrix;
    VectorType m_diag;
    MatrixL m_L;
    MatrixU m_U;
    PermutationMatrix<Dynamic, Dynamic, StorageIndex> m_P, m_Pinv;
    std::vector<StorageIndex> m_nonZerosPerCol;
    std::vector<StorageIndex> m_parent;
    bool m_isInitialized;
    bool m_analysisIsOk;
    bool m_factorizationIsOk;
    ComputationInfo m_info;
    Scalar m_epsilon;  // Regularization parameter for numerical stability
    Scalar m_shiftScale = Scalar(1);
    Scalar m_shiftOffset = Scalar(0);
};

}  // namespace Eigen

#endif  // EIGEN_CUSTOM_SIMPLICIAL_LDLT_H
