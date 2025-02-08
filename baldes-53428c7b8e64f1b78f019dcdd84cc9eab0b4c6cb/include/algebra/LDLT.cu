#include "Eigen/Core"
#include "Eigen/Sparse"
#include "LDLTCuda.h"
#include "ipm/LDLTSimp.h"

// Type definitions (adjust as needed)
using CholMatrixType = Eigen::SparseMatrix<double>; // Example
using Scalar         = double;
using StorageIndex   = int;
using DiagonalScalar = double;
using RealScalar     = double;

using Index          = int; // Adjust depending on your needs
Scalar m_shiftScale  = Scalar(1);
Scalar m_shiftOffset = Scalar(0);

// Helper function to get symmetric values (replace this with the correct logic)
__device__ __host__ inline Scalar getSymm(Scalar x) {
    return x; // Modify this based on your actual requirement (e.g., conjugate for complex)
}

// Helper function to get diagonal values
__device__ __host__ inline DiagonalScalar getDiag(Scalar x) {
    return x; // Modify this based on your actual requirement (e.g., conjugate for complex)
}

// CUDA kernel implementation
template <bool DoLDLT, bool NonHermitian>
__global__ void factorizeKernel(const StorageIndex *Lp, StorageIndex *Li, Scalar *Lx, DiagonalScalar *diag, Scalar *y,
                                StorageIndex *pattern, StorageIndex *tags, StorageIndex *nonZerosPerCol,
                                StorageIndex *parent, Scalar shiftScale, Scalar shiftOffset, StorageIndex size,
                                bool *ok, const CholMatrixType ap) {
    int task_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (task_idx < size) {
        StorageIndex k = task_idx;

        y[k]              = Scalar(0);
        StorageIndex top  = size;
        tags[k]           = k;
        nonZerosPerCol[k] = 0;

        for (typename CholMatrixType::InnerIterator it(ap, k); it; ++it) {
            StorageIndex i = it.index();
            if (i <= k) {
                y[i] += getSymm(it.value());

                Index len;
                for (len = 0; tags[i] != k; i = parent[i]) {
                    pattern[len++] = i;
                    tags[i]        = k;
                }

                while (len > 0) { pattern[--top] = pattern[--len]; }
            }
        }

        DiagonalScalar d = getDiag(y[k]) * shiftScale + shiftOffset;
        y[k]             = Scalar(0);

        // Sparse reductions
        for (; top < size; ++top) {
            Index  i  = pattern[top];
            Scalar yi = y[i];
            y[i]      = Scalar(0);

            Scalar l_ki = yi / getDiag(diag[i]);
            Index  p2   = Lp[i] + nonZerosPerCol[i];

            for (Index p = Lp[i]; p < p2; ++p) { y[Li[p]] -= getSymm(Lx[p]) * yi; }

            d -= getDiag(l_ki * getSymm(yi));
            Li[p2] = k;
            Lx[p2] = l_ki;
            ++nonZerosPerCol[i];
        }

        diag[k] = d;
        if (d == RealScalar(0)) { *ok = false; }
    }
}

// Function definition
template <bool DoLDLT, bool NonHermitian, typename CholMatrixType, typename Scalar, typename StorageIndex,
          typename DiagonalScalar>
void CustomSimplicialLDLT::factorize_preordered_cuda(const CholMatrixType &ap) {
    const StorageIndex  size = StorageIndex(ap.rows());
    const StorageIndex *Lp   = m_matrix.outerIndexPtr();
    StorageIndex       *Li   = m_matrix.innerIndexPtr();
    Scalar             *Lx   = m_matrix.valuePtr();

    // Allocate device memory
    Scalar         *d_y;
    StorageIndex   *d_pattern, *d_tags, *d_Lp, *d_Li, *d_nonZerosPerCol, *d_parent;
    DiagonalScalar *d_diag;
    Scalar         *d_Lx;
    bool           *d_ok;

    CUDA_CHECK(cudaMalloc((void **)&d_y, sizeof(Scalar) * size));
    CUDA_CHECK(cudaMalloc((void **)&d_pattern, sizeof(StorageIndex) * size));
    CUDA_CHECK(cudaMalloc((void **)&d_tags, sizeof(StorageIndex) * size));
    CUDA_CHECK(cudaMalloc((void **)&d_diag, sizeof(DiagonalScalar) * size));
    CUDA_CHECK(cudaMalloc((void **)&d_Lp, sizeof(StorageIndex) * size));
    CUDA_CHECK(cudaMalloc((void **)&d_Li, sizeof(StorageIndex) * size));
    CUDA_CHECK(cudaMalloc((void **)&d_nonZerosPerCol, sizeof(StorageIndex) * size));
    CUDA_CHECK(cudaMalloc((void **)&d_parent, sizeof(StorageIndex) * size));
    CUDA_CHECK(cudaMalloc((void **)&d_Lx, sizeof(Scalar) * size));
    CUDA_CHECK(cudaMalloc((void **)&d_ok, sizeof(bool)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_Lp, Lp, sizeof(StorageIndex) * size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Li, Li, sizeof(StorageIndex) * size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Lx, Lx, sizeof(Scalar) * size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nonZerosPerCol, m_nonZerosPerCol, sizeof(StorageIndex) * size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_parent, m_parent, sizeof(StorageIndex) * size, cudaMemcpyHostToDevice));

    bool ok = true;
    CUDA_CHECK(cudaMemcpy(d_ok, &ok, sizeof(bool), cudaMemcpyHostToDevice));

    // Define CUDA grid and block dimensions
    int blockSize = 128;
    int numBlocks = (size + blockSize - 1) / blockSize;

    // Launch the kernel
    factorizeKernel<DoLDLT, NonHermitian><<<numBlocks, blockSize>>>(d_Lp, d_Li, d_Lx, d_diag, d_y, d_pattern, d_tags,
                                                                    d_nonZerosPerCol, d_parent, m_shiftScale,
                                                                    m_shiftOffset, size, d_ok, ap);

    // Synchronize and check for errors
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy the result back to host
    CUDA_CHECK(cudaMemcpy(&ok, d_ok, sizeof(bool), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(m_diag.data(), d_diag, sizeof(DiagonalScalar) * size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Li, d_Li, sizeof(StorageIndex) * size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Lx, d_Lx, sizeof(Scalar) * size, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_pattern));
    CUDA_CHECK(cudaFree(d_tags));
    CUDA_CHECK(cudaFree(d_diag));
    CUDA_CHECK(cudaFree(d_Lp));
    CUDA_CHECK(cudaFree(d_Li));
    CUDA_CHECK(cudaFree(d_nonZerosPerCol));
    CUDA_CHECK(cudaFree(d_parent));
    CUDA_CHECK(cudaFree(d_Lx));
    CUDA_CHECK(cudaFree(d_ok));
}

// Explicit instantiation
template void factorize_preordered_cuda<true, true>(const CholMatrixType &ap);
template void factorize_preordered_cuda<true, false>(const CholMatrixType &ap);
template void factorize_preordered_cuda<false, true>(const CholMatrixType &ap);
template void factorize_preordered_cuda<false, false>(const CholMatrixType &ap);
