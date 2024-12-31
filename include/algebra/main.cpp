#include <Eigen/Sparse>
#include <iomanip>
#include <iostream>

#include "Supernodal.h"

// Helper function to create a test matrix with a different pattern
Eigen::SparseMatrix<double> createTestMatrix(int size) {
    Eigen::SparseMatrix<double> matrix(size, size);
    std::vector<Eigen::Triplet<double>> triplets;

    // Create a positive definite matrix with a different structure
    // This creates a "pentadiagonal" matrix with stronger diagonal dominance
    for (int i = 0; i < size; i++) {
        // Diagonal entry
        triplets.push_back(Eigen::Triplet<double>(i, i, 6.0));

        // First off-diagonal entries
        if (i < size - 1) {
            triplets.push_back(Eigen::Triplet<double>(i + 1, i, -1.0));
            triplets.push_back(Eigen::Triplet<double>(i, i + 1, -1.0));
        }

        // Second off-diagonal entries with different pattern
        if (i < size - 2) {
            triplets.push_back(Eigen::Triplet<double>(i + 2, i, -0.8));
            triplets.push_back(Eigen::Triplet<double>(i, i + 2, -0.8));
        }

        // Add some "long-range" connections
        if (i < size - 3) {
            triplets.push_back(Eigen::Triplet<double>(i + 3, i, -0.3));
            triplets.push_back(Eigen::Triplet<double>(i, i + 3, -0.3));
        }
    }

    matrix.setFromTriplets(triplets.begin(), triplets.end());
    return matrix;
}

// Modified pattern printing to handle new matrix structure
void printPattern(const Eigen::SparseMatrix<double>& matrix) {
    std::cout << "\nMatrix pattern:" << std::endl;
    for (int i = 0; i < matrix.rows(); i++) {
        for (int j = 0; j < matrix.cols(); j++) {
            double val = matrix.coeff(i, j);
            if (val != 0) {
                if (val == 6.0)
                    std::cout << "D ";  // Diagonal
                else if (std::abs(val - (-1.0)) < 1e-10)
                    std::cout << "1 ";  // First off-diagonal
                else if (std::abs(val - (-0.8)) < 1e-10)
                    std::cout << "2 ";  // Second off-diagonal
                else
                    std::cout << "3 ";  // Third off-diagonal
            } else {
                std::cout << ". ";
            }
        }
        std::cout << std::endl;
    }
}

// Rest of the helper functions remain the same
void printVector(const Eigen::VectorXd& v, const std::string& name) {
    std::cout << name << ":" << std::endl;
    for (int i = 0; i < v.size(); i++) {
        std::cout << std::setw(12) << std::setprecision(6) << v(i);
        if ((i + 1) % 6 == 0 || i == v.size() - 1) std::cout << std::endl;
    }
}

int main() {
    std::cout << std::fixed << std::setprecision(6);

    // Create test matrix
    const int size = 6;  // Increased size for better testing
    auto matrix = createTestMatrix(size);

    std::cout << "Testing Supernodal Solver" << std::endl;
    std::cout << "=========================" << std::endl;

    // Print original matrix pattern
    printPattern(matrix);

    try {
        // Create solver
        Supernodal<double, int> solver(matrix);

        // Print elimination tree and supernodes
        solver.printEliminationTree();
        solver.debugSupernodes();

        // Create right-hand side: b = A*ones
        Eigen::VectorXd ones = Eigen::VectorXd::Ones(size);
        Eigen::VectorXd b = matrix * ones;

        std::cout << "\nRight-hand side vector (b = A*ones):" << std::endl;
        printVector(b, "b");

        // Solve the system
        Eigen::VectorXd x = solver.solve(b);

        std::cout << "\nSolution vector (should be close to ones):"
                  << std::endl;
        printVector(x, "x");

        // Compute residual
        Eigen::VectorXd residual = matrix * x - b;
        double rel_error = residual.norm() / b.norm();

        std::cout << "\nVerification:" << std::endl;
        std::cout << "Relative error: " << rel_error << std::endl;
        std::cout << "Maximum deviation from 1.0: "
                  << (x - ones).lpNorm<Eigen::Infinity>() << std::endl;

        if (rel_error < 1e-10) {
            std::cout << "Test PASSED!" << std::endl;
        } else {
            std::cout << "Test FAILED! Residual too large." << std::endl;
        }

    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
