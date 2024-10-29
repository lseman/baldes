#ifndef factorize_preordered_cuda_H
#define factorize_preordered_cuda_H

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <cuda_runtime.h>
#include <iostream>
using CholMatrixType = Eigen::SparseMatrix<double>; // Example
using Scalar         = double;
using StorageIndex   = int;
using DiagonalScalar = double;
using Index          = int; // Adjust depending on your needs

// Function to check for CUDA errors
#define CUDA_CHECK(call)                                                                              \
    {                                                                                                 \
        const cudaError_t error = call;                                                               \
        if (error != cudaSuccess) {                                                                   \
            std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", ";                            \
            std::cerr << "code: " << error << ", reason: " << cudaGetErrorString(error) << std::endl; \
            exit(1);                                                                                  \
        }                                                                                             \
    }

#endif // factorize_preordered_cuda_H
