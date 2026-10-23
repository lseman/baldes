#ifndef BALDES_CLP_MATRIX_KERNELS_HPP
#define BALDES_CLP_MATRIX_KERNELS_HPP

#include <cstddef>
#include <memory>
#if defined(__SSE2__) && !defined(BALDES_CLP_FORCE_SCALAR)
#include <emmintrin.h>
#endif

namespace baldes_clp {
// Repack the two dense row vectors once per tableau update. Each SIMD lane
// then accumulates one dot product in the original sparse-entry order.
class PairedRows {
  public:
    PairedRows(int rows, int columns) : rows_(rows), data_(nullptr) {
#if defined(__SSE2__) && !defined(BALDES_CLP_FORCE_SCALAR)
        if (rows > 0 && columns >= 128) {
            if (rows <= 256) {
                data_ = local_;
            } else {
                heap_.reset(new double[2 * static_cast<std::size_t>(rows)]);
                data_ = heap_.get();
            }
        }
#else
        (void)columns;
#endif
    }

    const double* pack(const double* pi, const double* weight) {
        if (!data_)
            return nullptr;
        for (int row = 0; row < rows_; ++row) {
            data_[2 * static_cast<std::size_t>(row)] = -pi[row];
            data_[2 * static_cast<std::size_t>(row) + 1] = weight[row];
        }
        return data_;
    }

  private:
    int rows_;
    alignas(16) double local_[512];
    std::unique_ptr<double[]> heap_;
    double* data_;
};

template <class Index>
inline void transpose_pair(Index start, Index end, const int* row, const double* element,
                           const double* paired, double& value, double& modification) {
#if defined(__SSE2__) && !defined(BALDES_CLP_FORCE_SCALAR)
    __m128d sums = _mm_setzero_pd();
    for (Index j = start; j < end; ++j) {
        const __m128d inputs = _mm_loadu_pd(paired + 2 * static_cast<std::size_t>(row[j]));
        const __m128d product = _mm_mul_pd(inputs, _mm_set1_pd(element[j]));
        sums = _mm_add_pd(sums, product);
    }
    alignas(16) double result[2];
    _mm_store_pd(result, sums);
    value = result[0];
    modification = result[1];
#else
    value = modification = 0.;
    for (Index j = start; j < end; ++j) {
        const std::size_t offset = 2 * static_cast<std::size_t>(row[j]);
        value += paired[offset] * element[j];
        modification += paired[offset + 1] * element[j];
    }
#endif
}
} // namespace baldes_clp
#endif
