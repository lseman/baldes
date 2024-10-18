#include <Eigen/Dense>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

// LU Decomposition class for basis matrix inversion
#include <Eigen/Dense>
#include <iostream>
#include <stdexcept>
#include <vector>

class LU_Decomposition {
public:
    Eigen::MatrixXd  LU;   // Combined L and U matrices stored in one matrix
    std::vector<int> perm; // Permutation vector to track row swaps
    int              n;    // Size of the square matrix

    LU_Decomposition(const Eigen::MatrixXd &A) { decompose(A); }

    // Perform LU decomposition with partial pivoting
    void decompose(const Eigen::MatrixXd &A) {
        if (A.rows() != A.cols()) {
            std::cerr << "Error: LU decomposition requires a square matrix, but received a " << A.rows() << "x"
                      << A.cols() << " matrix." << std::endl;
            throw std::invalid_argument("Matrix must be square for LU decomposition.");
        }

        n  = A.rows();
        LU = A; // Combined L and U in the same matrix
        perm.resize(n);

        // Initialize permutation vector
        for (int i = 0; i < n; ++i) perm[i] = i;

        for (int k = 0; k < n; ++k) {
            // Partial Pivoting: Find the row with the largest absolute value in the current column
            int    pivotRow = k;
            double maxVal   = std::abs(LU(perm[k], k)); // Compare on the permuted row

            for (int i = k + 1; i < n; ++i) {
                if (std::abs(LU(perm[i], k)) > maxVal) {
                    maxVal   = std::abs(LU(perm[i], k));
                    pivotRow = i;
                }
            }

            // Swap rows in the permutation vector if necessary
            if (pivotRow != k) { std::swap(perm[k], perm[pivotRow]); }

            // Numerical stability check (to prevent division by near-zero values)
            if (std::abs(LU(perm[k], k)) < 1e-12) { std::cerr << "Warning: Near-zero pivot encountered!" << std::endl; }

            // Perform the elimination (on the original matrix, no perm needed in LU)
            for (int i = k + 1; i < n; ++i) {
                LU(perm[i], k) /= LU(perm[k], k); // Use perm[k] for pivot row
                for (int j = k + 1; j < n; ++j) { LU(perm[i], j) -= LU(perm[i], k) * LU(perm[k], j); }
            }
        }
    }

    // Solve the system Ax = b using the computed LU decomposition
    Eigen::VectorXd solve(const Eigen::VectorXd &b) {
        if (b.size() != n) {
            std::cerr << "Error: The vector b size does not match the LU matrix dimensions." << std::endl;
            throw std::invalid_argument("Dimension mismatch between LU matrix and b vector.");
        }

        Eigen::VectorXd y = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd x = Eigen::VectorXd::Zero(n);

        // Apply the permutation vector to the right-hand side
        Eigen::VectorXd b_permuted(n);
        for (int i = 0; i < n; ++i) { b_permuted(i) = b(perm[i]); }

        // Forward substitution to solve Ly = Pb
        for (int i = 0; i < n; ++i) {
            y(i) = b_permuted(i);
            for (int j = 0; j < i; ++j) {
                y(i) -= LU(perm[i], j) * y(j); // Use perm[i] to access the permuted row
            }
        }

        // Backward substitution to solve Ux = y
        for (int i = n - 1; i >= 0; --i) {
            x(i) = y(i);
            for (int j = i + 1; j < n; ++j) {
                x(i) -= LU(perm[i], j) * x(j); // Use perm[i] to access the permuted row
            }
            x(i) /= LU(perm[i], i);
        }

        return x;
    }
};
class SimplexRevised {
public:
    const double pivotTolerance = 1e-10;

    Eigen::MatrixXd     A;          // Constraint matrix
    Eigen::VectorXd     b;          // Right-hand side vector
    Eigen::VectorXd     c;          // Cost vector
    Eigen::VectorXd     original_c; // Original cost vector
    Eigen::VectorXd     lb;         // Lower bounds
    Eigen::VectorXd     ub;         // Upper bounds
    std::vector<int>    basis;      // Indices of basic variables
    Eigen::MatrixXd     B_inv;      // Inverse of the basis matrix
    LU_Decomposition    luBasis;    // LU Decomposition for the basis matrix
    std::vector<double> columnNormCache;

    SimplexRevised(const Eigen::MatrixXd &A, const Eigen::VectorXd &b, const Eigen::VectorXd &c,
                   const Eigen::VectorXd &lb, const Eigen::VectorXd &ub)
        : A(A), b(b), c(c), original_c(c), lb(lb), ub(ub), basis(A.rows()), B_inv(A.rows(), A.rows()), luBasis(A) {
        for (int i = 0; i < A.rows(); ++i) {
            basis[i] = i + A.cols(); // Initialize basis to slack variables
        }
    }

    void solve() {
        initializeLU();
        solvePrimalSimplex();
    }

    void solvePrimalSimplex() {
        while (!isOptimal()) {
            int pivotCol = findPivotColumn();
            if (pivotCol == -1) {
                std::cerr << "Problem may be unbounded.\n";
                break;
            }

            int pivotRow = findPivotRow(pivotCol);
            if (pivotRow == -1) {
                std::cerr << "Problem may be infeasible.\n";
                break;
            }

            performPivot(pivotRow, pivotCol);
        }

        printSolution();
    }

private:
    // Initialize LU decomposition for the basis matrix
    void initializeLU() {
        int m = A.rows(); // Number of constraints
        int n = A.cols(); // Number of variables

        // Print the size of A
        std::cout << "Size of A: " << A.rows() << "x" << A.cols() << std::endl;
        std::cout.flush(); // Ensure the output is printed

        // Augment A with slack variables (Identity matrix for m constraints)
        Eigen::MatrixXd A_aug(m, n + m);
        A_aug.leftCols(n)  = A;                               // Original matrix A
        A_aug.rightCols(m) = Eigen::MatrixXd::Identity(m, m); // Identity for slack variables

        // Print the size of A_aug after augmentation
        std::cout << "Size of A_aug (with slack variables): " << A_aug.rows() << "x" << A_aug.cols() << std::endl;
        std::cout.flush(); // Ensure the output is printed

        // Initialize the basis with slack variables
        basis.resize(m);
        for (int i = 0; i < m; ++i) {
            basis[i] = n + i; // Slack variables in the basis (from column n to n + m - 1 in A_aug)
        }

        // Print the basis indices
        std::cout << "Basis indices: ";
        for (int i = 0; i < m; ++i) { std::cout << basis[i] << " "; }
        std::cout << std::endl;
        std::cout.flush(); // Ensure the output is printed

        // Create the basis matrix B from the identity matrix (slack variable columns) in A_aug
        Eigen::MatrixXd B(m, m); // Basis matrix must be square (m x m)
        for (int i = 0; i < m; ++i) {
            B.col(i) = A_aug.col(basis[i]); // Select the slack variable columns (n to n+m-1)
        }

        // Print the size of the basis matrix B
        std::cout << "Size of B (basis matrix): " << B.rows() << "x" << B.cols() << std::endl;
        std::cout.flush(); // Ensure the output is printed

        // Ensure B is square before performing LU decomposition
        if (B.rows() == B.cols()) {
            std::cout << "Performing LU decomposition on B..." << std::endl;
            luBasis.decompose(B); // Perform LU decomposition
        } else {
            std::cerr << "Error: Basis matrix is not square: " << B.rows() << "x" << B.cols() << std::endl;
            throw std::invalid_argument("Matrix must be square for LU decomposition.");
        }
    }

    // Check if the current solution is optimal
    bool isOptimal() const {
        Eigen::VectorXd reducedCosts = computeReducedCosts();
        return (reducedCosts.array() >= -pivotTolerance).all(); // Ensure all reduced costs are non-negative
    }

    // Compute the reduced costs for non-basic variables
    Eigen::VectorXd computeReducedCosts() const {
        Eigen::VectorXd lambda = B_inv.transpose() * c(basis); // Dual prices
        return c - A.transpose() * lambda;                     // Reduced costs
    }

    // Find the pivot column based on reduced costs
    int findPivotColumn() const {
        Eigen::VectorXd reducedCosts   = computeReducedCosts();
        double          minReducedCost = std::numeric_limits<double>::max();
        int             pivotCol       = -1;

        for (int j = 0; j < reducedCosts.size(); ++j) {
            if (reducedCosts(j) < -pivotTolerance) {
                if (lb(j) < std::numeric_limits<double>::infinity() &&
                    ub(j) > -std::numeric_limits<double>::infinity()) {
                    double columnNorm        = columnNormCache[j];
                    double steepestEdgeRatio = -reducedCosts(j) / std::sqrt(columnNorm);

                    if (steepestEdgeRatio < minReducedCost) {
                        minReducedCost = steepestEdgeRatio;
                        pivotCol       = j;
                    }
                }
            }
        }

        return pivotCol;
    }

    // Find the pivot row based on the ratio test
    int findPivotRow(int pivotCol) const {
        Eigen::VectorXd A_pivotCol = A.col(pivotCol);
        Eigen::VectorXd u          = B_inv * A_pivotCol;

        double minRatio = std::numeric_limits<double>::max();
        int    pivotRow = -1;

        for (int i = 0; i < u.size(); ++i) {
            if (u(i) > pivotTolerance) {
                double ratio = b(i) / u(i);
                if (ratio < minRatio && i >= 0 && i < A.rows()) {
                    minRatio = ratio;
                    pivotRow = i;
                }
            }
        }

        return pivotRow;
    }

    // Perform the pivot operation
    void performPivot(int pivotRow, int pivotCol) {
        if (pivotRow < 0 || pivotRow >= A.rows() || pivotCol < 0 || pivotCol >= A.cols()) {
            std::cerr << "Invalid pivot row or column: " << pivotRow << ", " << pivotCol << std::endl;
            return; // Prevent invalid access
        }

        double pivotValue = A(pivotRow, pivotCol);

        if (std::abs(pivotValue) < pivotTolerance) {
            std::cerr << "Numerical instability detected: pivot value is too small: " << pivotValue << "\n";
            return;
        }

        basis[pivotRow] = pivotCol;

        Eigen::VectorXd A_pivotCol = A.col(pivotCol);
        Eigen::VectorXd u          = B_inv * A_pivotCol;
        Eigen::VectorXd e          = Eigen::VectorXd::Zero(B_inv.rows());
        e(pivotRow)                = 1.0;

        for (int i = 0; i < B_inv.rows(); ++i) {
            if (i != pivotRow && i >= 0 && i < B_inv.rows()) { // Ensure valid row access
                B_inv.row(i) = B_inv.row(i) - (u(i) / u(pivotRow)) * B_inv.row(pivotRow);
            }
        }

        B_inv.row(pivotRow) /= u(pivotRow);
        columnNormCache[pivotCol] = A.col(pivotCol).squaredNorm();
    }

    // Print the final solution
    void printSolution() const {
        Eigen::VectorXd solution = Eigen::VectorXd::Zero(A.cols());
        Eigen::VectorXd xB       = B_inv * b;

        for (int i = 0; i < basis.size(); ++i) {
            if (basis[i] < solution.size()) { solution(basis[i]) = xB(i); }
        }

        // Ensure solution is within bounds
        for (int i = 0; i < solution.size(); ++i) {
            solution(i) = std::min(std::max(solution(i), lb(i)), ub(i)); // Clip the solution to [lb, ub]
        }

        double objective_value = original_c.transpose() * solution;
        std::cout << "Optimal solution found!\n";
        std::cout << "Solution: " << solution.transpose() << "\n";
        std::cout << "Objective value: " << objective_value << "\n";
    }
};
