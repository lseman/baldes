#pragma once

// define numericutils namespace that have eps, comparison within threshold, etc
#include <cstdlib>
namespace numericutils {
// Define the epsilon value for floating-point comparisons
constexpr double eps = 1e-9;

constexpr double half = 0.5;

// Define greater than: returns true if a is greater than b by more than
// threshold.
inline bool gt(double a, double b, double threshold = eps) {
    return (a - b) > threshold;
}

// Define less than: returns true if a is less than b by more than threshold.
inline bool lt(double a, double b, double threshold = eps) {
    return (a - b) < -threshold;
}

// Define greater than or equal: returns true if a is either nearly equal to b
// or greater than b (using threshold for near equality).
inline bool gte(double a, double b, double threshold = eps) {
    return a > b - threshold;
}

// Define less than or equal: returns true if a is either nearly equal to b
// or less than b (using threshold for near equality).
inline bool lte(double a, double b, double threshold = eps) {
    return a < b + threshold;
}

inline bool isZero(double a, double threshold = eps) {
    return std::abs(a) < threshold;
}

inline bool exact_lte(double a, double b, double threshold = eps) {
    return a <= b;
}  // namespace numericutils

inline bool exact_gte(double a, double b, double threshold = eps) {
    return a >= b;
}

inline bool exact_lt(double a, double b, double threshold = eps) {
    return a < b;
}

inline bool exact_gt(double a, double b, double threshold = eps) {
    return a > b;
}
};  // namespace numericutils
