#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

namespace baldes::bcp {

inline constexpr double kVarIntegralityTol = 1e-6;
inline constexpr double kBinaryTol         = 1e-5;
inline constexpr double kRowFeasTol        = 1e-5;
inline constexpr double kPricingTol        = 1e-6;

[[nodiscard]] inline bool hasNegativeReducedCost(double reduced_cost, double tolerance = kPricingTol) noexcept {
    return std::isfinite(reduced_cost) && reduced_cost < -tolerance;
}

[[nodiscard]] inline bool isIntegerSolution(const std::vector<double> &solution,
                                            double tolerance = kVarIntegralityTol) noexcept {
    return std::ranges::all_of(solution, [tolerance](double value) {
        return std::isfinite(value) && std::abs(value - std::round(value)) <= tolerance;
    });
}

template <typename Matrix>
[[nodiscard]] bool detectIntegerMasterSolution(const std::vector<double> &solution, const Matrix &matrix,
                                               double &integer_objective, double binary_tolerance = kBinaryTol,
                                               double row_tolerance = kRowFeasTol) {
    const std::size_t column_count = std::min(solution.size(), matrix.c.size());
    if (column_count == 0) return false;

    std::vector<double> binary_solution(column_count, 0.0);
    for (std::size_t column = 0; column < column_count; ++column) {
        const double value = solution[column];
        if (!std::isfinite(value) || value < -binary_tolerance || value > 1.0 + binary_tolerance) return false;

        if (std::abs(value) <= binary_tolerance) {
            binary_solution[column] = 0.0;
        } else if (std::abs(value - 1.0) <= binary_tolerance) {
            binary_solution[column] = 1.0;
        } else {
            return false;
        }
    }

    const auto lhs = matrix.A_sparse.multiply(binary_solution);
    if (lhs.size() != matrix.b.size() || lhs.size() != matrix.sense.size()) return false;

    for (std::size_t row = 0; row < lhs.size(); ++row) {
        const double activity = lhs[row];
        const double bound    = matrix.b[row];
        switch (matrix.sense[row]) {
        case '<':
            if (activity > bound + row_tolerance) return false;
            break;
        case '>':
            if (activity < bound - row_tolerance) return false;
            break;
        default:
            if (std::abs(activity - bound) > row_tolerance) return false;
            break;
        }
    }

    integer_objective = 0.0;
    for (std::size_t column = 0; column < column_count; ++column) {
        integer_objective += matrix.c[column] * binary_solution[column];
    }
    return std::isfinite(integer_objective);
}

} // namespace baldes::bcp
