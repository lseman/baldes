// SparseSolveTriangularGPU.h

#pragma once
#include <cuda_runtime.h>
#include <Eigen/Sparse>
#include <cmath>

// CUDA kernel for lower triangular solve (row-major)
__global__ void sparseLowerTriangularSolveKernelRowMajor(const int* rowPtr, const int* colIdx, const double* values, double* other, int numRows, int numCols, double regularizationFactor) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;  // Parallelize over rows
    if (row >= numRows) return;  // Out of bounds check

    for (int col = 0; col < numCols; ++col) {
        double tmp = other[row * numCols + col];  // Current RHS value

        int start = rowPtr[row];     // Row start in the compressed matrix
        int end = rowPtr[row + 1];   // Row end in the compressed matrix

        double lastVal = 0;
        for (int idx = start; idx < end; ++idx) {
            int currentCol = colIdx[idx];  // Get the column index
            double val = values[idx];      // Get the matrix value

            if (currentCol == row) {
                lastVal = val;  // Diagonal element
                break;
            }

            // Subtract the contribution of already solved elements
            tmp -= val * other[currentCol * numCols + col];
        }

        // Diagonal processing
        if (fabs(lastVal) < regularizationFactor) {
            lastVal = (lastVal >= 0 ? regularizationFactor : -regularizationFactor);
        }
        other[row * numCols + col] = tmp / lastVal;
    }
}

// CUDA kernel for upper triangular solve (row-major)
__global__ void sparseUpperTriangularSolveKernelRowMajor(const int* rowPtr, const int* colIdx, const double* values, double* other, int numRows, int numCols, double regularizationFactor) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;  // Parallelize over rows
    if (row >= numRows) return;  // Out of bounds check

    for (int col = 0; col < numCols; ++col) {
        double tmp = other[row * numCols + col];  // Current RHS value

        int start = rowPtr[row];     // Row start in the compressed matrix
        int end = rowPtr[row + 1];   // Row end in the compressed matrix

        double lastVal = 0;
        for (int idx = end - 1; idx >= start; --idx) {
            int currentCol = colIdx[idx];  // Get the column index
            double val = values[idx];      // Get the matrix value

            if (currentCol == row) {
                lastVal = val;  // Diagonal element
                break;
            }

            // Subtract the contribution of already solved elements
            tmp -= val * other[currentCol * numCols + col];
        }

        // Diagonal processing
        if (fabs(lastVal) < regularizationFactor) {
            lastVal = (lastVal >= 0 ? regularizationFactor : -regularizationFactor);
        }
        other[row * numCols + col] = tmp / lastVal;
    }
}

// CUDA kernel for lower triangular solve (column-major)
__global__ void sparseLowerTriangularSolveKernelColMajor(const int* rowPtr, const int* colIdx, const double* values, double* other, int numRows, int numCols, double regularizationFactor) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Parallelize over columns
    if (col >= numCols) return;  // Out of bounds check

    for (int row = 0; row < numRows; ++row) {
        double tmp = other[row * numCols + col];  // Current RHS value

        int start = rowPtr[row];     // Row start in the compressed matrix
        int end = rowPtr[row + 1];   // Row end in the compressed matrix

        double lastVal = 0;
        for (int idx = start; idx < end; ++idx) {
            int currentCol = colIdx[idx];  // Get the column index
            double val = values[idx];      // Get the matrix value

            if (currentCol == row) {
                lastVal = val;  // Diagonal element
                break;
            }

            // Subtract the contribution of already solved elements
            tmp -= val * other[currentCol * numCols + col];
        }

        // Diagonal processing
        if (fabs(lastVal) < regularizationFactor) {
            lastVal = (lastVal >= 0 ? regularizationFactor : -regularizationFactor);
        }
        other[row * numCols + col] = tmp / lastVal;
    }
}

// CUDA kernel for upper triangular solve (column-major)
__global__ void sparseUpperTriangularSolveKernelColMajor(const int* rowPtr, const int* colIdx, const double* values, double* other, int numRows, int numCols, double regularizationFactor) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Parallelize over columns
    if (col >= numCols) return;  // Out of bounds check

    for (int row = numRows - 1; row >= 0; --row) {
        double tmp = other[row * numCols + col];  // Current RHS value

        int start = rowPtr[row];     // Row start in the compressed matrix
        int end = rowPtr[row + 1];   // Row end in the compressed matrix

        double lastVal = 0;
        for (int idx = end - 1; idx >= start; --idx) {
            int currentCol = colIdx[idx];  // Get the column index
            double val = values[idx];      // Get the matrix value

            if (currentCol == row) {
                lastVal = val;  // Diagonal element
                break;
            }

            // Subtract the contribution of already solved elements
            tmp -= val * other[currentCol * numCols + col];
        }

        // Diagonal processing
        if (fabs(lastVal) < regularizationFactor) {
            lastVal = (lastVal >= 0 ? regularizationFactor : -regularizationFactor);
        }
        other[row * numCols + col] = tmp / lastVal;
    }
}

// Host function for row-major lower triangular solve
void sparseLowerTriangularSolveGPU_RowMajor(const Eigen::SparseMatrix<double>& lhs, Eigen::MatrixXd& rhs, double regularizationFactor) {
    const int* rowPtr = lhs.outerIndexPtr();
    const int* colIdx = lhs.innerIndexPtr();
    const double* values = lhs.valuePtr();
    int numRows = lhs.rows();
    int numCols = rhs.cols();

    int* d_rowPtr;
    int* d_colIdx;
    double* d_values;
    double* d_rhs;

    cudaMalloc(&d_rowPtr, (numRows + 1) * sizeof(int));
    cudaMalloc(&d_colIdx, lhs.nonZeros() * sizeof(int));
    cudaMalloc(&d_values, lhs.nonZeros() * sizeof(double));
    cudaMalloc(&d_rhs, numRows * numCols * sizeof(double));

    cudaMemcpy(d_rowPtr, rowPtr, (numRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, colIdx, lhs.nonZeros() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values, lhs.nonZeros() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rhs, rhs.data(), numRows * numCols * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (numRows + blockSize - 1) / blockSize;
    sparseLowerTriangularSolveKernelRowMajor<<<numBlocks, blockSize>>>(d_rowPtr, d_colIdx, d_values, d_rhs, numRows, numCols, regularizationFactor);

    cudaMemcpy(rhs.data(), d_rhs, numRows * numCols * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_rowPtr);
    cudaFree(d_colIdx);
    cudaFree(d_values);
    cudaFree(d_rhs);
}

// Host function for row-major upper triangular solve
void sparseUpperTriangularSolveGPU_RowMajor(const Eigen::SparseMatrix<double>& lhs, Eigen::MatrixXd& rhs, double regularizationFactor) {
    const int* rowPtr = lhs.outerIndexPtr();
    const int* colIdx = lhs.innerIndexPtr();
    const double* values = lhs.valuePtr();
    int numRows = lhs.rows();
    int numCols = rhs.cols();

    int* d_rowPtr;
    int* d_colIdx;
    double* d_values;
    double* d_rhs;

    cudaMalloc(&d_rowPtr, (numRows + 1) * sizeof(int));
    cudaMalloc(&d_colIdx, lhs.nonZeros() * sizeof(int));
    cudaMalloc(&d_values, lhs.nonZeros() * sizeof(double));
    cudaMalloc(&d_rhs, numRows * numCols * sizeof(double));

    cudaMemcpy(d_rowPtr, rowPtr, (numRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, colIdx, lhs.nonZeros() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values, lhs.nonZeros() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rhs, rhs.data(), numRows * numCols * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (numRows + blockSize - 1) / blockSize;
    sparseUpperTriangularSolveKernelRowMajor<<<numBlocks, blockSize>>>(d_rowPtr, d_colIdx, d_values, d_rhs, numRows, numCols, regularizationFactor);

    cudaMemcpy(rhs.data(), d_rhs, numRows * numCols * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_rowPtr);
    cudaFree(d_colIdx);
    cudaFree(d_values);
    cudaFree(d_rhs);
}

// Host function for column-major lower triangular solve
void sparseLowerTriangularSolveGPU_ColMajor(const Eigen::SparseMatrix<double>& lhs, Eigen::MatrixXd& rhs, double regularizationFactor) {
    const int* rowPtr = lhs.outerIndexPtr();
    const int* colIdx = lhs.innerIndexPtr();
    const double* values = lhs.valuePtr();
    int numRows = lhs.rows();
    int numCols = rhs.cols();

    int* d_rowPtr;
    int* d_colIdx;
    double* d_values;
    double* d_rhs;

    cudaMalloc(&d_rowPtr, (numRows + 1) * sizeof(int));
    cudaMalloc(&d_colIdx, lhs.nonZeros() * sizeof(int));
    cudaMalloc(&d_values, lhs.nonZeros() * sizeof(double));
    cudaMalloc(&d_rhs, numRows * numCols * sizeof(double));

    cudaMemcpy(d_rowPtr, rowPtr, (numRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, colIdx, lhs.nonZeros() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values, lhs.nonZeros() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rhs, rhs.data(), numRows * numCols * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (numCols + blockSize - 1) / blockSize;
    sparseLowerTriangularSolveKernelColMajor<<<numBlocks, blockSize>>>(d_rowPtr, d_colIdx, d_values, d_rhs, numRows, numCols, regularizationFactor);

    cudaMemcpy(rhs.data(), d_rhs, numRows * numCols * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_rowPtr);
    cudaFree(d_colIdx);
    cudaFree(d_values);
    cudaFree(d_rhs);
}

// Host function for column-major upper triangular solve
void sparseUpperTriangularSolveGPU_ColMajor(const Eigen::SparseMatrix<double>& lhs, Eigen::MatrixXd& rhs, double regularizationFactor) {
    const int* rowPtr = lhs.outerIndexPtr();
    const int* colIdx = lhs.innerIndexPtr();
    const double* values = lhs.valuePtr();
    int numRows = lhs.rows();
    int numCols = rhs.cols();

    int* d_rowPtr;
    int* d_colIdx;
    double* d_values;
    double* d_rhs;

    cudaMalloc(&d_rowPtr, (numRows + 1) * sizeof(int));
    cudaMalloc(&d_colIdx, lhs.nonZeros() * sizeof(int));
    cudaMalloc(&d_values, lhs.nonZeros() * sizeof(double));
    cudaMalloc(&d_rhs, numRows * numCols * sizeof(double));

    cudaMemcpy(d_rowPtr, rowPtr, (numRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, colIdx, lhs.nonZeros() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values, lhs.nonZeros() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rhs, rhs.data(), numRows * numCols * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (numCols + blockSize - 1) / blockSize;
    sparseUpperTriangularSolveKernelColMajor<<<numBlocks, blockSize>>>(d_rowPtr, d_colIdx, d_values, d_rhs, numRows, numCols, regularizationFactor);

    cudaMemcpy(rhs.data(), d_rhs, numRows * numCols * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_rowPtr);
    cudaFree(d_colIdx);
    cudaFree(d_values);
    cudaFree(d_rhs);
}
