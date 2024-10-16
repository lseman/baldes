#include <Eigen/Dense>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

// LU Decomposition class for basis matrix inversion
class LU_Decomposition {
public:
    Eigen::MatrixXd L, U, P; // P is the permutation matrix for row swaps

    LU_Decomposition(const Eigen::MatrixXd &A) { decompose(A); }

    void decompose(const Eigen::MatrixXd &A) {
        int n = A.rows();
        L     = Eigen::MatrixXd::Identity(n, n);
        U     = A;
        P     = Eigen::MatrixXd::Identity(n, n);

        for (int k = 0; k < n; ++k) {
            // Partial Pivoting: Find the row with the largest absolute value in the current column
            int    pivotRow = k;
            double maxVal   = std::abs(U(k, k));
            for (int i = k + 1; i < n; ++i) {
                if (std::abs(U(i, k)) > maxVal) {
                    maxVal   = std::abs(U(i, k));
                    pivotRow = i;
                }
            }

            // Swap rows if necessary
            if (pivotRow != k) {
                U.row(k).swap(U.row(pivotRow));
                P.row(k).swap(P.row(pivotRow)); // Track row swaps in P
            }

            // Perform the elimination
            for (int i = k + 1; i < n; ++i) {
                L(i, k) = U(i, k) / U(k, k);
                U.row(i) -= L(i, k) * U.row(k);
            }
        }
    }

    // Solve the system Ax = b using the computed LU decomposition
    Eigen::VectorXd solve(const Eigen::VectorXd &b) {
        int             n = L.rows();
        Eigen::VectorXd y = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd x = Eigen::VectorXd::Zero(n);

        // Apply the permutation matrix to the right-hand side
        Eigen::VectorXd b_permuted = P * b;

        // Forward substitution to solve Ly = Pb
        for (int i = 0; i < n; ++i) {
            y(i) = b_permuted(i);
            for (int j = 0; j < i; ++j) { y(i) -= L(i, j) * y(j); }
        }

        // Backward substitution to solve Ux = y
        for (int i = n - 1; i >= 0; --i) {
            x(i) = y(i);
            for (int j = i + 1; j < n; ++j) { x(i) -= U(i, j) * x(j); }
            x(i) /= U(i, i);
        }

        return x;
    }
};

// SimplexRevised class with Dual Simplex implementation
class SimplexRevised {
public:
    const double pivotTolerance = 1e-10;

    Eigen::MatrixXd     A;          // Constraint matrix
    Eigen::VectorXd     b;          // Right-hand side vector
    Eigen::VectorXd     c;          // Cost vector
    Eigen::VectorXd     original_c; // Original cost vector
    std::vector<int>    basis;      // Indices of basic variables
    Eigen::MatrixXd     B_inv;      // Inverse of the basis matrix
    LU_Decomposition    luBasis;    // LU Decomposition for the basis matrix
    std::vector<double> columnNormCache;

    SimplexRevised(const Eigen::MatrixXd &A, const Eigen::VectorXd &b, const Eigen::VectorXd &c)
        : A(A), b(b), c(c), original_c(c), basis(A.rows()), B_inv(A.rows(), A.rows()), luBasis(A) {
        for (int i = 0; i < A.rows(); ++i) {
            basis[i] = i + A.cols(); // Initialize basis to slack variables
        }
    }

    void solve() {
        initializeLU();

        // Start with Dual Simplex if the primal solution is infeasible
        if (isDualFeasible()) {
            std::cout << "Starting with Dual Simplex...\n";
            solveDualSimplex();
        } else {
            std::cerr << "Dual Simplex is not applicable, as the solution is not dual feasible.\n";
        }
    }

    void solveDualSimplex() {
        while (!isPrimalFeasible()) {
            int pivotRow = findDualPivotRow();
            if (pivotRow == -1) {
                std::cerr << "No valid pivot row found, the problem may be infeasible.\n";
                break;
            }

            int pivotCol = findDualPivotColumn(pivotRow);
            if (pivotCol == -1) {
                std::cerr << "No valid pivot column found, the problem may be unbounded.\n";
                break;
            }

            performPivot(pivotRow, pivotCol);
        }

        printSolution();
    }

private:
    void initializeLU() {
        int m = A.rows();
        int n = A.cols();

        // Create an augmented matrix with slack variables (Identity matrix)
        Eigen::MatrixXd A_aug(m, n + m);
        A_aug.leftCols(n)  = A;
        A_aug.rightCols(m) = Eigen::MatrixXd::Identity(m, m);

        basis.resize(m);
        for (int i = 0; i < m; ++i) { basis[i] = n + i; }

        Eigen::MatrixXd B(m, m);
        for (int i = 0; i < m; ++i) { B.col(i) = A_aug.col(basis[i]); }

        columnNormCache.resize(n);
        for (int i = 0; i < n; ++i) { columnNormCache[i] = A.col(i).squaredNorm(); }

        luBasis.decompose(B);
        B_inv = Eigen::MatrixXd::Identity(m, m);
        for (int i = 0; i < m; ++i) { B_inv.col(i) = luBasis.solve(B_inv.col(i)); }
    }

    bool isPrimalFeasible() const {
        return (b.array() >= 0).all(); // Check if all b values are non-negative
    }

    bool isDualFeasible() const {
        Eigen::VectorXd reducedCosts = computeReducedCosts();
        return (reducedCosts.array() >= 0).all(); // Ensure all reduced costs are non-negative
    }

    Eigen::VectorXd computeReducedCosts() const {
        Eigen::VectorXd lambda = B_inv.transpose() * c(basis); // Dual prices
        return c - A.transpose() * lambda;                     // Reduced costs
    }

    int findDualPivotRow() {
        double mostNegativeB = 0.0;
        int    pivotRow      = -1;

        // Find the row with the most negative b value (primal infeasibility)
        for (int i = 0; i < b.size(); ++i) {
            if (b(i) < 0 && std::abs(b(i)) > mostNegativeB) {
                mostNegativeB = std::abs(b(i));
                pivotRow      = i;
            }
        }

        return pivotRow;
    }

    int findDualPivotColumn(int pivotRow) const {
        Eigen::VectorXd row                  = A.row(pivotRow);
        double          maxSteepestEdgeRatio = 0.0;
        int             pivotCol             = -1;

        for (int j = 0; j < row.size(); ++j) {
            if (row(j) < 0) { // Only consider negative entries in the pivot row
                double columnNorm        = columnNormCache[j];
                double steepestEdgeRatio = -c(j) / std::sqrt(columnNorm);

                if (steepestEdgeRatio > maxSteepestEdgeRatio) {
                    maxSteepestEdgeRatio = steepestEdgeRatio;
                    pivotCol             = j;
                }
            }
        }

        return pivotCol;
    }

    void performPivot(int pivotRow, int pivotCol) {
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
            if (i != pivotRow) { B_inv.row(i) = B_inv.row(i) - (u(i) / u(pivotRow)) * B_inv.row(pivotRow); }
        }

        B_inv.row(pivotRow) /= u(pivotRow);
        columnNormCache[pivotCol] = A.col(pivotCol).squaredNorm();
    }

    void printSolution() const {
        Eigen::VectorXd solution = Eigen::VectorXd::Zero(A.cols());
        Eigen::VectorXd xB       = B_inv * b;

        for (int i = 0; i < basis.size(); ++i) {
            if (basis[i] < solution.size()) { solution(basis[i]) = xB(i); }
        }

        double objective_value = original_c.transpose() * solution;
        std::cout << "Optimal solution found!\n";
        std::cout << "Solution: " << solution.transpose() << "\n";
        std::cout << "Objective value: " << objective_value << "\n";
    }
};
