#pragma once

// define numericutils namespace that have eps, comparison within threshold, etc
#include <cstdlib>
namespace numericutils {
// Define the epsilon value for floating-point comparisons
constexpr double eps = 1e-3;

constexpr double half = 0.5;

// Define a function to compare two floating-point values within a threshold
inline bool compare_within_threshold(double a, double b, double threshold = eps) { return std::abs(a - b) < threshold; }

// Define a function to compare two floating-point values within a threshold
inline bool compare_within_threshold(float a, float b, float threshold = eps) { return std::abs(a - b) < threshold; }

// Define a function to compare two floating-point values within a threshold
inline bool compare_within_threshold(int a, int b, int threshold = 0) { return std::abs(a - b) < threshold; }

// Define a function to compare two floating-point values within a threshold
inline bool compare_within_threshold(long a, long b, long threshold = 0) { return std::abs(a - b) < threshold; }

// Define a function to compare two floating-point values within a threshold
inline bool compare_within_threshold(long long a, long long b, long long threshold = 0) {
    return std::abs(a - b) < threshold;
}

// define greater than
inline bool greater_than(double a, double b, double threshold = eps) { return a - b > threshold; }

// define less than
inline bool less_than(double a, double b, double threshold = eps) { return a - b < -threshold; }

// define greater than or equal
inline bool greater_than_or_equal(double a, double b, double threshold = eps) { return a - b > -threshold; }

// define less than or equal
inline bool less_than_or_equal(double a, double b, double threshold = eps) { return a - b < threshold; }

} // namespace numericutils