#ifndef MATRIX_H
#define MATRIX_H

#include <vector>

class Matrix {
    int              cols_; // The number of columns of the matrix
    std::vector<int> data_; // The vector where all the data is stored (this represents the matrix)

public:
    // Empty constructor: with zero columns and a vector of size zero
    Matrix() : cols_(0), data_(std::vector<int>(0)) {}

    // Constructor: create a matrix of size dimension by dimension, using a C++ vector of size dimension * dimension
    Matrix(const int dimension) : cols_(dimension) { data_ = std::vector<int>(dimension * dimension); }

    // Set a value val at position (row, col) in the matrix
    void set(const int row, const int col, const int val) { data_[cols_ * row + col] = val; }

    // Get the value at position (row, col) in the matrix
    int get(const int row, const int col) const { return data_[cols_ * row + col]; }

    // Overload the subscript operator to access matrix rows
    int* operator[](const int row) {
        return &data_[cols_ * row]; // Return a pointer to the start of the row
    }

    // Const version for read-only access
    const int* operator[](const int row) const {
        return &data_[cols_ * row]; // Return a pointer to the start of the row (const version)
    }
};

#endif
