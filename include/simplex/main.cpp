#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

class DualSimplex {
private:
    struct SteepestEdgeWeights {
        std::vector<double> weights;

        void initialize(size_t size) { weights.resize(size, 1.0); }

        void update(const std::vector<double> &pivot_column, size_t pivot_row, const std::vector<double> &weights) {
            // Update steepest edge weights using the DEVEX pricing rules
            double pivot_weight = weights[pivot_row];
            for (size_t i = 0; i < weights.size(); ++i) {
                if (i != pivot_row) {
                    double pivot_multiplier = pivot_column[i] / pivot_column[pivot_row];
                    this->weights[i]        = weights[i] - 2.0 * pivot_multiplier * pivot_weight +
                                       pivot_multiplier * pivot_multiplier * pivot_weight;
                }
            }
            this->weights[pivot_row] = pivot_weight / (pivot_column[pivot_row] * pivot_column[pivot_row]);
        }
    };

    std::vector<std::vector<double>> tableau;
    std::vector<int>                 basis;
    std::vector<double>              costs;
    size_t                           num_rows;
    size_t                           num_cols;
    SteepestEdgeWeights              edge_weights;
    double                           epsilon = 1e-10;

public:
    DualSimplex(const std::vector<std::vector<double>> &A, const std::vector<double> &b, const std::vector<double> &c) {
        initialize(A, b, c);
    }

    void initialize(const std::vector<std::vector<double>> &A, const std::vector<double> &b,
                    const std::vector<double> &c) {
        if (A.empty() || A[0].empty() || b.empty() || c.empty()) { throw std::invalid_argument("Empty input data"); }

        num_rows = A.size();
        num_cols = A[0].size();

        // Initialize tableau with slack variables
        tableau.resize(num_rows + 1);
        for (auto &row : tableau) { row.resize(num_cols + num_rows + 1, 0.0); }

        // Copy constraint matrix
        for (size_t i = 0; i < num_rows; ++i) {
            for (size_t j = 0; j < num_cols; ++j) { tableau[i][j] = A[i][j]; }
        }

        // Add slack variables
        for (size_t i = 0; i < num_rows; ++i) {
            tableau[i][num_cols + i] = 1.0;
            tableau[i].back()        = b[i];
        }

        // Set objective function coefficients
        costs = c;
        for (size_t j = 0; j < num_cols; ++j) { tableau.back()[j] = -c[j]; }

        // Initialize basis
        basis.resize(num_rows);
        for (size_t i = 0; i < num_rows; ++i) { basis[i] = num_cols + i; }

        // Initialize steepest edge weights
        edge_weights.initialize(num_rows);

        // Perform Phase 1 if necessary
        if (needsPhase1()) { performPhase1(); }
    }

    bool needsPhase1() const {
        for (size_t i = 0; i < num_rows; ++i) {
            if (tableau[i].back() < -epsilon) { return true; }
        }
        return false;
    }

    void performPhase1() {
        // Save original objective
        std::vector<double> original_obj = tableau.back();

        // Create Phase 1 objective
        for (size_t j = 0; j < tableau.back().size() - 1; ++j) { tableau.back()[j] = 0.0; }

        for (size_t i = 0; i < num_rows; ++i) {
            if (tableau[i].back() < -epsilon) {
                for (size_t j = 0; j < tableau.back().size() - 1; ++j) { tableau.back()[j] += tableau[i][j]; }
            }
        }

        // Solve Phase 1
        solve();

        // Check if Phase 1 found a feasible solution
        if (std::abs(tableau.back().back()) > epsilon) { throw std::runtime_error("Problem is infeasible"); }

        // Restore original objective
        tableau.back() = original_obj;
    }

    void solve() {
        while (true) {
            // Find leaving variable (most negative RHS)
            int    leaving_row = -1;
            double min_rhs     = -epsilon;

            for (size_t i = 0; i < num_rows; ++i) {
                if (tableau[i].back() < min_rhs) {
                    min_rhs     = tableau[i].back();
                    leaving_row = i;
                }
            }

            if (leaving_row == -1) {
                break; // Optimal solution found
            }

            // Find entering variable using steepest edge pricing
            int    entering_col = -1;
            double best_ratio   = -std::numeric_limits<double>::max();

            for (size_t j = 0; j < tableau[0].size() - 1; ++j) {
                if (std::abs(tableau[leaving_row][j]) > epsilon) {
                    double ratio = -tableau.back()[j] / tableau[leaving_row][j];
                    ratio *= edge_weights.weights[leaving_row];

                    if (ratio > best_ratio) {
                        best_ratio   = ratio;
                        entering_col = j;
                    }
                }
            }

            if (entering_col == -1) { throw std::runtime_error("Problem is unbounded"); }

            // Pivot
            pivot(leaving_row, entering_col);
        }
    }

    void pivot(size_t leaving_row, size_t entering_col) {
        // Store pivot column for steepest edge update
        std::vector<double> pivot_column;
        for (size_t i = 0; i < num_rows; ++i) { pivot_column.push_back(tableau[i][entering_col]); }

        // Perform pivot
        double pivot_element = tableau[leaving_row][entering_col];

        for (size_t j = 0; j < tableau[0].size(); ++j) { tableau[leaving_row][j] /= pivot_element; }

        for (size_t i = 0; i < tableau.size(); ++i) {
            if (i != leaving_row) {
                double multiplier = tableau[i][entering_col];
                for (size_t j = 0; j < tableau[0].size(); ++j) {
                    tableau[i][j] -= multiplier * tableau[leaving_row][j];
                }
            }
        }

        // Update basis
        basis[leaving_row] = entering_col;

        // Update steepest edge weights
        edge_weights.update(pivot_column, leaving_row, edge_weights.weights);
    }

    std::vector<double> getSolution() const {
        std::vector<double> solution(num_cols, 0.0);
        for (size_t i = 0; i < num_rows; ++i) {
            if (basis[i] < num_cols) { solution[basis[i]] = tableau[i].back(); }
        }
        return solution;
    }

    double getObjectiveValue() const { return -tableau.back().back(); }
};

int main() {
    // Example usage
    std::vector<std::vector<double>> A = {{1, 1}, {2, -1}};
    std::vector<double>              b = {4, 2};
    std::vector<double>              c = {1, 1};

    DualSimplex solver(A, b, c);
    solver.solve();

    auto   solution      = solver.getSolution();
    double optimal_value = solver.getObjectiveValue();
    std::cout << "Optimal value: " << optimal_value << "\n";
    std::cout << "Solution: ";
    for (double val : solution) { std::cout << val << " "; }
    std::cout << "\n";
}