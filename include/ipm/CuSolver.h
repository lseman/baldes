#include "Definitions.h"
#include "fmt/base.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cudss.h> // cuDSS header
#include <vector>

class CuSolver {
public:
    int  n;
    int  m;
    bool firstFactorization = true;

    // cuDSS related members
    cudssHandle_t handle;
    cudssConfig_t config;
    cudssData_t   data;
    cudssMatrix_t A_cudss;
    cudssMatrix_t b_cudss;
    cudssMatrix_t x_cudss;

    int    *d_rowOffsets;
    int    *d_colIndices;
    double *d_values;
    double *d_bvalues;
    double *d_xvalues;

    CuSolver()
        : d_rowOffsets(nullptr), d_colIndices(nullptr), d_values(nullptr), d_bvalues(nullptr), d_xvalues(nullptr),
          A_cudss(nullptr), b_cudss(nullptr), x_cudss(nullptr), handle(nullptr), config(nullptr), data(nullptr) {
        if (cudssCreate(&handle) != CUDSS_STATUS_SUCCESS) {
            std::cerr << "Failed to create cuDSS handle" << std::endl;
            std::exit(EXIT_FAILURE);
        }
        if (cudssConfigCreate(&config) != CUDSS_STATUS_SUCCESS) {
            std::cerr << "Failed to create cuDSS config" << std::endl;
            std::exit(EXIT_FAILURE);
        }
        if (cudssDataCreate(handle, &data) != CUDSS_STATUS_SUCCESS) {
            std::cerr << "Failed to create cuDSS data" << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    ~CuSolver() {
        if (d_rowOffsets) {
            cudaFree(d_rowOffsets);
            d_rowOffsets = nullptr;
        }
        if (d_colIndices) {
            cudaFree(d_colIndices);
            d_colIndices = nullptr;
        }
        if (d_values) {
            cudaFree(d_values);
            d_values = nullptr;
        }
        if (d_bvalues) {
            cudaFree(d_bvalues);
            d_bvalues = nullptr;
        }
        if (d_xvalues) {
            cudaFree(d_xvalues);
            d_xvalues = nullptr;
        }

        if (A_cudss) {
            cudssMatrixDestroy(A_cudss);
            A_cudss = nullptr;
        }
        if (b_cudss) {
            cudssMatrixDestroy(b_cudss);
            b_cudss = nullptr;
        }
        if (x_cudss) {
            cudssMatrixDestroy(x_cudss);
            x_cudss = nullptr;
        }

        if (config) {
            cudssConfigDestroy(config);
            config = nullptr;
        }
        if (data) {
            cudssDataDestroy(handle, data);
            data = nullptr;
        }
        if (handle) {
            cudssDestroy(handle);
            handle = nullptr;
        }
    }

    void initializeCuDSS(const Eigen::SparseMatrix<double> &matrix, const Eigen::VectorXd &b, Eigen::VectorXd &x) {
        n = matrix.rows();
        m = matrix.cols();

        // Allocate device memory
        if (cudaMalloc(&d_rowOffsets, (n + 1) * sizeof(int)) != cudaSuccess ||
            cudaMalloc(&d_colIndices, matrix.nonZeros() * sizeof(int)) != cudaSuccess ||
            cudaMalloc(&d_values, matrix.nonZeros() * sizeof(double)) != cudaSuccess ||
            cudaMalloc(&d_bvalues, b.size() * sizeof(double)) != cudaSuccess ||
            cudaMalloc(&d_xvalues, x.size() * sizeof(double)) != cudaSuccess) {
            std::cerr << "CUDA memory allocation failed" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        if (cudaMalloc(&d_rowOffsets, (n + 1) * sizeof(int)) != cudaSuccess ||
            cudaMalloc(&d_colIndices, matrix.nonZeros() * sizeof(int)) != cudaSuccess ||
            cudaMalloc(&d_values, matrix.nonZeros() * sizeof(double)) != cudaSuccess ||
            cudaMalloc(&d_bvalues, b.size() * sizeof(double)) != cudaSuccess ||
            cudaMalloc(&d_xvalues, x.size() * sizeof(double)) != cudaSuccess) {
            std::cerr << "CUDA memory allocation failed" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        // Copy data to device
        if (cudaMemcpy(d_rowOffsets, matrix.outerIndexPtr(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice) !=
                cudaSuccess ||
            cudaMemcpy(d_colIndices, matrix.innerIndexPtr(), matrix.nonZeros() * sizeof(int), cudaMemcpyHostToDevice) !=
                cudaSuccess ||
            cudaMemcpy(d_values, matrix.valuePtr(), matrix.nonZeros() * sizeof(double), cudaMemcpyHostToDevice) !=
                cudaSuccess ||
            cudaMemcpy(d_bvalues, b.data(), b.size() * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
            std::cerr << "CUDA memory copy failed" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        if (cudssMatrixCreateCsr(&A_cudss, n, m, matrix.nonZeros(), d_rowOffsets, nullptr, d_colIndices, d_values,
                                 CUDA_R_32I, CUDA_R_64F, CUDSS_MTYPE_SPD, CUDSS_MVIEW_UPPER,
                                 CUDSS_BASE_ZERO) != CUDSS_STATUS_SUCCESS ||
            cudssMatrixCreateDn(&b_cudss, n, 1, n, d_bvalues, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR) !=
                CUDSS_STATUS_SUCCESS ||
            cudssMatrixCreateDn(&x_cudss, n, 1, n, d_xvalues, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR) !=
                CUDSS_STATUS_SUCCESS) {
            std::cerr << "cuDSS matrix creation failed" << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    void factorizeMatrix(const Eigen::SparseMatrix<double> &matrix) {
        if (firstFactorization) {
            Eigen::VectorXd zeroVec = Eigen::VectorXd::Zero(matrix.rows());
            initializeCuDSS(matrix, zeroVec, zeroVec);

            if (cudssExecute(handle, CUDSS_PHASE_ANALYSIS, config, data, A_cudss, x_cudss, b_cudss) !=
                CUDSS_STATUS_SUCCESS) {
                std::cerr << "cuDSS analysis phase failed" << std::endl;
                std::exit(EXIT_FAILURE);
            }
            firstFactorization = false;
        }

        if (cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, config, data, A_cudss, x_cudss, b_cudss) !=
            CUDSS_STATUS_SUCCESS) {
            std::cerr << "cuDSS factorization phase failed" << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    Eigen::VectorXd solve(const Eigen::VectorXd &rhs) {
        Eigen::VectorXd x(rhs.size());
        if (cudaMemcpy(d_bvalues, rhs.data(), rhs.size() * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
            std::cerr << "CUDA memory copy failed" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        if (cudssExecute(handle, CUDSS_PHASE_SOLVE, config, data, A_cudss, x_cudss, b_cudss) != CUDSS_STATUS_SUCCESS) {
            std::cerr << "cuDSS solve phase failed" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        if (cudaMemcpy(x.data(), d_xvalues, x.size() * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
            std::cerr << "CUDA memory copy failed" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        return x;
    }
};