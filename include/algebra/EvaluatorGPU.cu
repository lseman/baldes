// EvaluatorGPU.cu
#include "EvaluatorGPU.h"
#include <cuda_runtime.h>

template<typename T>
__device__ bool is_exactly_zero(T value) {
    return abs(value) < 1e-15;
}

class CustomMatIterator {
public:
    __device__ CustomMatIterator(const int* outerIndexPtr, const int* innerIndexPtr,
                                const double* valuePtr, int col)
        : outerIndexPtr_(outerIndexPtr),
          innerIndexPtr_(innerIndexPtr),
          valuePtr_(valuePtr),
          col_(col) {
        outerIndex_ = outerIndexPtr_[col];
        end_ = outerIndexPtr_[col + 1];
    }

    __device__ bool isValid() const { return outerIndex_ < end_; }
    __device__ void operator++() { ++outerIndex_; }
    __device__ int row() const { return innerIndexPtr_[outerIndex_]; }
    __device__ double value() const { return valuePtr_[outerIndex_]; }
    __device__ int index() const { return innerIndexPtr_[outerIndex_]; }

private:
    const int* outerIndexPtr_;
    const int* innerIndexPtr_;
    const double* valuePtr_;
    int col_;
    int outerIndex_;
    int end_;
};

template<bool IsLower>
__device__ void processRow(const int* outerIndexPtr, const int* innerIndexPtr,
                          const double* valuePtr, double* other,
                          int row, int col, int n, double regularizationFactor) {
    double& tmp = other[row * n + col];
    if (is_exactly_zero(tmp)) return;

    CustomMatIterator it(outerIndexPtr, innerIndexPtr, valuePtr, row);

    if constexpr (IsLower) {
        while (it.isValid() && it.index() < row) {
            tmp -= it.value() * other[it.index() * n + col];
            ++it;
        }

        if (it.isValid() && it.index() == row) {
            double diag_val = it.value();
            if (abs(diag_val) < regularizationFactor) {
                diag_val = (diag_val >= 0) ? regularizationFactor : -regularizationFactor;
            }
            tmp /= diag_val;
        }
    } else {
        double diag_val = 0;
        bool found_diag = false;

        while (it.isValid()) {
            const int idx = it.index();
            const double val = it.value();

            if (!found_diag) {
                if (idx == row) {
                    diag_val = val;
                    found_diag = true;
                }
            } else if (idx > row) {
                tmp -= val * other[idx * n + col];
            }
            ++it;
        }

        if (found_diag) {
            if (abs(diag_val) < regularizationFactor) {
                diag_val = (diag_val >= 0) ? regularizationFactor : -regularizationFactor;
            }
            tmp /= diag_val;
        }
    }
}

template<bool IsLower>
__global__ void sparseSolveTriangularKernel(const int* outerIndexPtr,
                                           const int* innerIndexPtr,
                                           const double* valuePtr,
                                           double* other, int n, int m,
                                           double regularizationFactor) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;

    if constexpr (IsLower) {
        for (int i = 0; i < m; ++i) {
            __syncthreads();
            processRow<true>(outerIndexPtr, innerIndexPtr, valuePtr,
                           other, i, col, n, regularizationFactor);
        }
    } else {
        for (int i = m - 1; i >= 0; --i) {
            __syncthreads();
            processRow<false>(outerIndexPtr, innerIndexPtr, valuePtr,
                            other, i, col, n, regularizationFactor);
        }
    }
}

void sparseTriangularSolveGPU(const Eigen::SparseMatrix<double>& mat,
                             Eigen::Matrix<double, -1, 1>& other,
                             bool isLower,
                             double regularizationFactor) {
    int m = mat.rows();
    int n = other.cols();

    int* d_outerIndexPtr;
    int* d_innerIndexPtr;
    double* d_valuePtr;
    double* d_other;

    cudaMalloc(&d_outerIndexPtr, (m + 1) * sizeof(int));
    cudaMalloc(&d_innerIndexPtr, mat.nonZeros() * sizeof(int));
    cudaMalloc(&d_valuePtr, mat.nonZeros() * sizeof(double));
    cudaMalloc(&d_other, m * n * sizeof(double));

    cudaMemcpy(d_outerIndexPtr, mat.outerIndexPtr(), (m + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_innerIndexPtr, mat.innerIndexPtr(), mat.nonZeros() * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_valuePtr, mat.valuePtr(), mat.nonZeros() * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_other, other.data(), m * n * sizeof(double),
               cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    if (isLower) {
        sparseSolveTriangularKernel<true><<<gridSize, blockSize>>>(
            d_outerIndexPtr, d_innerIndexPtr, d_valuePtr,
            d_other, n, m, regularizationFactor);
    } else {
        sparseSolveTriangularKernel<false><<<gridSize, blockSize>>>(
            d_outerIndexPtr, d_innerIndexPtr, d_valuePtr,
            d_other, n, m, regularizationFactor);
    }

    cudaMemcpy(other.data(), d_other, m * n * sizeof(double),
               cudaMemcpyDeviceToHost);

    cudaFree(d_outerIndexPtr);
    cudaFree(d_innerIndexPtr);
    cudaFree(d_valuePtr);
    cudaFree(d_other);
}
