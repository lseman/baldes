/**
 * @file SparseMatrix.h
 * @brief Sparse matrix in CRS format with optimized row-wise iteration.
 *
 * This file defines a `SparseMatrix` struct that represents a sparse matrix in Compressed Row Storage (CRS) format.
 * It supports efficient row-wise access to non-zero elements and can convert the sparse matrix to a dense format.
 *
 * The struct uses an optimized build function to convert a list of sparse elements into CRS format, enabling
 * fast row-wise traversal using an iterator.
 *
 */

#pragma once

#include "../third_party/small_vector.hpp"
#include <algorithm>
#include <functional>
#include <numeric> // for std::partial_sum
#include <ranges>  // for C++23 range algorithms
#include <utility> // for std::move
#include <vector>

#include "../third_party/pdqsort.h"
/**
 * @struct SparseElement
 * @brief Represents a non-zero element in a sparse matrix.
 *
 * Each `SparseElement` stores the row and column indices, as well as the value of a matrix element.
 */
struct SparseElement {
    int    row;
    int    col;
    double value;
};

/**
 * @struct SparseMatrix
 * @brief A sparse matrix in Compressed Row Storage (CRS) format.
 *
 * The sparse matrix stores non-zero elements and allows efficient row-wise access through iterators.
 * The `SparseMatrix` struct also provides methods to insert elements, build the CRS structure,
 * and convert the matrix to a dense representation.
 */
struct SparseMatrix {
    gch::small_vector<int>    rows;      ///< Row indices of the matrix elements
    gch::small_vector<int>    cols;      ///< Column indices of the matrix elements
    gch::small_vector<double> values;    ///< Values of the matrix elements
    gch::small_vector<int>    row_start; ///< Starting index of each row in `elements`
    int                       num_rows;  ///< Total number of rows in the matrix
    int                       num_cols;  ///< Total number of columns in the matrix
    bool                      is_dirty = true;

    // Default constructor
    SparseMatrix() : num_rows(0), num_cols(0) {}

    // Constructor with num_rows and num_cols
    SparseMatrix(int num_rows, int num_cols) : num_rows(num_rows), num_cols(num_cols) {}

    // Insert a non-zero element into the sparse matrix
    void insert(int row, int col, double value) {
        // Expand the number of rows if needed
        if (row >= num_rows) {
            num_rows = row + 1;
            row_start.resize(num_rows + 1, 0); // Expand the row_start structure
        }

        if (col >= num_cols) {
            num_cols = col + 1; // Expand the number of columns if necessary
        }

        rows.push_back(row);
        cols.push_back(col);
        values.push_back(value);
        is_dirty = true; // Mark as dirty to rebuild CRS later
    }

    // Delete a column (variable) from the matrix
    void delete_column(int col_to_delete) {
        size_t write_index = 0;
        for (size_t i = 0; i < cols.size(); ++i) {
            if (cols[i] != col_to_delete) {
                // Move the retained element to the new position
                if (write_index != i) {
                    rows[write_index]   = rows[i];
                    cols[write_index]   = cols[i];
                    values[write_index] = values[i];
                }
                ++write_index;
            }
        }

        // Resize the vectors to remove the deleted column's elements
        rows.resize(write_index);
        cols.resize(write_index);
        values.resize(write_index);

        // Adjust column indices for elements beyond the deleted column
        for (auto &col : cols) {
            if (col > col_to_delete) { --col; }
        }

        // Update the column count
        --num_cols;
        is_dirty = true; // Mark as dirty to rebuild CRS later
    }

    // Delete a row (constraint) from the matrix
    void delete_row(int row_to_delete) {
        size_t write_index = 0;
        for (size_t i = 0; i < rows.size(); ++i) {
            if (rows[i] != row_to_delete) {
                // Move the retained element to the new position
                if (write_index != i) {
                    rows[write_index]   = rows[i];
                    cols[write_index]   = cols[i];
                    values[write_index] = values[i];
                }
                ++write_index;
            }
        }

        // Resize the vectors to remove the deleted row's elements
        rows.resize(write_index);
        cols.resize(write_index);
        values.resize(write_index);

        // Adjust row indices for elements beyond the deleted row
        for (auto &row : rows) {
            if (row > row_to_delete) { --row; }
        }

        // Update the row count
        --num_rows;
        is_dirty = true; // Mark as dirty to rebuild CRS later
    }

    // Modify or delete an element at a given row and column
    void modify_or_delete(int row, int col, double value) {
        // Find the element in the vectors
        for (size_t i = 0; i < rows.size(); ++i) {
            if (rows[i] == row && cols[i] == col) {
                if (value != 0.0) {
                    // Modify the element's value
                    values[i] = value;
                } else {
                    // Remove the element if value is zero
                    rows.erase(rows.begin() + i);
                    cols.erase(cols.begin() + i);
                    values.erase(values.begin() + i);
                }
                is_dirty = true; // Mark as dirty to rebuild CRS later
                return;
            }
        }

        // If the element does not exist and the value is non-zero, insert a new element
        if (value != 0.0) { insert(row, col, value); }
    }

    // Build the CRS structure by computing row start positions
    void buildRowStart() {
        if (!is_dirty) return;

        // Step 1: Sort rows, columns, and values together by (row, col)
        std::vector<size_t> indices(rows.size());
        std::iota(indices.begin(), indices.end(), 0); // Initialize indices with 0, 1, ..., N-1

        pdqsort(indices.begin(), indices.end(),
                [&](size_t a, size_t b) { return std::tie(rows[a], cols[a]) < std::tie(rows[b], cols[b]); });

        gch::small_vector<int>    sorted_rows(rows.size());
        gch::small_vector<int>    sorted_cols(cols.size());
        gch::small_vector<double> sorted_values(values.size());

        for (size_t i = 0; i < indices.size(); ++i) {
            sorted_rows[i]   = rows[indices[i]];
            sorted_cols[i]   = cols[indices[i]];
            sorted_values[i] = values[indices[i]];
        }

        rows   = std::move(sorted_rows);
        cols   = std::move(sorted_cols);
        values = std::move(sorted_values);

        // Step 2: Build the row_start structure
        row_start.assign(num_rows + 1, 0);

        for (const auto &row : rows) { ++row_start[row + 1]; }

        // Accumulate the counts to get the starting index for each row
        std::partial_sum(row_start.begin(), row_start.end(), row_start.begin());

        is_dirty = false;
    }

    /**
     * @struct RowIterator
     * @brief Iterator for efficiently traversing non-zero elements of a specific row.
     */
    struct RowIterator {
        const SparseMatrix &matrix;
        size_t              index;
        size_t              end;

        /**
         * @brief Constructor for `RowIterator`.
         *
         * @param matrix Reference to the `SparseMatrix` being traversed.
         * @param row The row index to iterate over.
         */
        RowIterator(const SparseMatrix &matrix, int row)
            : matrix(matrix), index(matrix.row_start[row]), end(matrix.row_start[row + 1]) {}

        /**
         * @brief Check if the iterator is valid (i.e., it has more elements to traverse).
         * @return True if there are more elements, false otherwise.
         */
        inline bool valid() const { return index < end; }

        /**
         * @brief Move to the next element in the row.
         */
        inline void next() { ++index; }

        /**
         * @brief Get the value of the current element.
         * @return The value of the current element.
         */
        inline double value() const { return matrix.values[index]; }

        /**
         * @brief Get the column index of the current element.
         * @return The column index of the current element.
         */
        inline int col() const { return matrix.cols[index]; }

        /**
         * @brief Prefetch the next row's memory to improve cache locality.
         */
        inline void prefetch() const {
            __builtin_prefetch(&matrix.values[index + 1], 0, 1);
            __builtin_prefetch(&matrix.cols[index + 1], 0, 1);
        }
    };
    /**
     * @brief Get an iterator for a specific row.
     *
     * @param row The row index to iterate over.
     * @return A `RowIterator` for the specified row.
     */
    RowIterator rowIterator(int row) const {
        if (row_start.empty()) {
            const_cast<SparseMatrix *>(this)->buildRowStart(); // Lazy build if not yet done
        }
        return RowIterator(*this, row);
    }

    /**
     * @brief Perform a function for each non-zero element in every row.
     *
     * This function iterates over each row in the matrix and applies the given function
     * to each non-zero element in the row.
     *
     * @tparam Func A callable that accepts three arguments: row index, column index, and value of the element.
     * @param func The function to apply to each non-zero element.
     */
    template <typename Func>
    void forEachRow(Func &&func) const {
        // Assume row_start is already built for faster access; if not, ensure it's built once elsewhere.
        assert(!row_start.empty());

        // Iterate over rows and columns, directly invoking the callback for better inlining
        for (int row_idx = 0; row_idx < num_rows; ++row_idx) {
            for (RowIterator it = rowIterator(row_idx); it.valid(); it.next()) {
                func(row_idx, it.col(), it.value()); // Directly call the function without std::invoke
            }
        }
    }

    /**
     * @brief Convert the sparse matrix to a dense matrix format.
     *
     * The dense matrix has the same dimensions as the sparse matrix, with zero values
     * where there are no non-zero elements.
     *
     * @return A 2D vector representing the dense matrix.
     */
    std::vector<std::vector<double>> toDense() const {
        std::vector<std::vector<double>> dense(num_rows, std::vector<double>(num_cols, 0.0));
        forEachRow([&dense](int row, int col, double value) { dense[row][col] = value; });
        return dense;
    }
    std::vector<double> multiply(const std::vector<double> &x) {
        // Initialize the result vector with zeros
        std::vector<double> result(num_rows, 0.0);

        // Check if matrix is dirty and build row starts if necessary
        if (is_dirty) {
            buildRowStart(); // Ensure row start is built for CRS
        }

        // Perform the matrix-vector multiplication (A * x)
        for (int row = 0; row < num_rows; ++row) {
            double row_value = 0.0; // Store the result of each row * x
            for (RowIterator it = rowIterator(row); it.valid(); it.next()) {
                int    col_index = it.col();   // Column index of the matrix element
                double value     = it.value(); // Matrix value at (row, col_index)

                row_value += value * x[col_index]; // Perform dot product for this row
            }
            result[row] = row_value; // Store the result of the dot product in the result vector
        }

        return result;
    }

    std::vector<double> violation(const std::vector<double> &x, const std::vector<double> &b) {
        // Initialize the result vector for violations
        std::vector<double> violation_vec(num_rows, 0.0);

        // Perform matrix-vector multiplication (A * x) and compute violation in a single step
        forEachRow([&](int row_idx, int col_idx, double value) {
            rowIterator(row_idx).prefetch();

            violation_vec[row_idx] += value * x[col_idx]; // Perform dot product
        });

        // Compute violation: Ax - b in place
        for (int row_idx = 0; row_idx < num_rows; ++row_idx) {
            violation_vec[row_idx] -= b[row_idx]; // Compute Ax - b
        }

        return violation_vec;
    }

    const std::vector<int> getRowStart() {
        if (is_dirty) { buildRowStart(); }
        auto row_start_copy = std::vector<int>(row_start.begin(), row_start.end());
        return row_start_copy;
    }
    /**
     * @brief Get the column indices of non-zero elements.
     *
     * @return A reference to the vector of column indices for the non-zero elements of the matrix.
     */
    const std::vector<int> getIndices() const {
        std::vector<int> indices(cols.begin(), cols.end());
        return indices;
    }

    /**
     * @brief Get the values of non-zero elements.
     *
     * @return A reference to the vector of values for the non-zero elements of the matrix.
     */
    const std::vector<double> getValues() const {
        std::vector<double> values(this->values.begin(), this->values.end());
        return values;
    }

    // define getRowLength
    int getRowLength(int row) const { return row_start[row + 1] - row_start[row]; }
};
