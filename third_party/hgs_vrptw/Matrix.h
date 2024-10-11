#ifndef MATRIX_H
#define MATRIX_H

#include <vector>

class Matrix {
    int              cols_; // The number of columns of the matrix
    std::vector<int> data_; // The vector where all the data is stored

public:
    // Default constructor: zero columns, empty data vector
    Matrix() : cols_(0), data_() {}

    // Constructor: create a square matrix of size `dimension x dimension`
    Matrix(const int dimension) : cols_(dimension), data_(dimension * dimension) {}

    // Set a value at (row, col) in the matrix
    inline void set(const int row, const int col, const int val) { data_[cols_ * row + col] = val; }

    // Get the value at (row, col) in the matrix
    inline int get(const int row, const int col) const { return data_[cols_ * row + col]; }

    // Overload the subscript operator to access rows (non-const version)
    inline int *operator[](const int row) { return data_.data() + (cols_ * row); }

    // Overload the subscript operator to access rows (const version)
    inline const int *operator[](const int row) const { return data_.data() + (cols_ * row); }
};

#endif
