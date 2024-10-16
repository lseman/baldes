#pragma once

#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <vector>

struct SparseMatrix {
    // COO format (always the default)
    std::vector<int>    rows;   // Row indices of the matrix elements
    std::vector<int>    cols;   // Column indices of the matrix elements
    std::vector<double> values; // Non-zero values of the matrix elements

    // CRS format (constructed on-demand)
    std::vector<int>    crs_row_start;
    std::vector<int>    crs_cols;
    std::vector<double> crs_values;

    int  num_rows = 0;
    int  num_cols = 0;
    bool coo_mode = true; // Always starts in COO mode

    SparseMatrix() = default;
    SparseMatrix(int num_rows, int num_cols) : num_rows(num_rows), num_cols(num_cols) {}

    // Insert a non-zero element into the sparse matrix (COO mode)
    void insert(int row, int col, double value) {
        // Dynamically resize rows and columns if the element is out of bounds
        if (row >= num_rows) {
            num_rows = row + 1; // Expand the number of rows
        }
        if (col >= num_cols) {
            num_cols = col + 1; // Expand the number of columns
        }

        rows.push_back(row);
        cols.push_back(col);
        values.push_back(value);
    }

    void insert_batch(const std::vector<int> &batch_rows, const std::vector<int> &batch_cols,
                      const std::vector<double> &batch_values) {
        assert(batch_rows.size() == batch_cols.size() && batch_cols.size() == batch_values.size());

        // Find the maximum row and column in the batch to resize if necessary
        int max_row = *std::max_element(batch_rows.begin(), batch_rows.end());
        int max_col = *std::max_element(batch_cols.begin(), batch_cols.end());

        // Dynamically resize the matrix dimensions if needed
        if (max_row >= num_rows) {
            num_rows = max_row + 1; // Expand the number of rows
        }
        if (max_col >= num_cols) {
            num_cols = max_col + 1; // Expand the number of columns
        }

        // Perform the batch insertion
        rows.insert(rows.end(), batch_rows.begin(), batch_rows.end());
        cols.insert(cols.end(), batch_cols.begin(), batch_cols.end());
        values.insert(values.end(), batch_values.begin(), batch_values.end());
    }

    // Convert COO to CRS (on-demand)
    void convertToCRS() {
        if (!coo_mode) return; // Already in CRS mode, no need to convert

        crs_row_start.assign(num_rows + 1, 0);
        crs_cols.resize(values.size());
        crs_values.resize(values.size());

        // Count non-zero elements per row
        for (int row : rows) { ++crs_row_start[row + 1]; }

        // Compute row start indices using cumulative sum
        std::partial_sum(crs_row_start.begin(), crs_row_start.end(), crs_row_start.begin());

        // Temporary array to keep track of the position for each row
        std::vector<int> row_position = crs_row_start;

        // Fill CRS data from COO format
        for (size_t i = 0; i < values.size(); ++i) {
            int row          = rows[i];
            int dest         = row_position[row]++;
            crs_cols[dest]   = cols[i];
            crs_values[dest] = values[i];
        }

        coo_mode = false; // Conversion complete
    }

    // Switch back to COO mode (clear CRS data)
    void switchToCOO() {
        if (coo_mode) return;
        coo_mode = true; // Now ready for COO operations
    }

    // RowIterator for CRS mode
    struct RowIterator {
        const SparseMatrix &matrix;
        size_t              index;
        size_t              end;

        RowIterator(const SparseMatrix &matrix, int row)
            : matrix(matrix), index(matrix.crs_row_start[row]),
              end((row + 1 < matrix.num_rows) ? matrix.crs_row_start[row + 1] : matrix.crs_values.size()) {}

        bool valid() const { return index < end; }

        void next() { ++index; }

        double value() const { return matrix.crs_values[index]; }

        int col() const { return matrix.crs_cols[index]; }
    };

    // Get a RowIterator for a specific row (convert to CRS if needed)
    RowIterator rowIterator(int row) const {
        if (coo_mode) const_cast<SparseMatrix *>(this)->convertToCRS(); // Convert to CRS if in COO mode
        return RowIterator(*this, row);
    }

    // Delete a column in COO mode
    void delete_column(int col_to_delete) {
        if (!coo_mode) switchToCOO();
        size_t write_index = 0;
        for (size_t i = 0; i < cols.size(); ++i) {
            if (cols[i] != col_to_delete) {
                rows[write_index]   = rows[i];
                cols[write_index]   = cols[i];
                values[write_index] = values[i];

                // Shift column indices to the left if greater than the deleted column
                if (cols[i] > col_to_delete) {
                    cols[write_index]--; // Shift left by one
                }

                ++write_index;
            }
        }
        rows.resize(write_index);
        cols.resize(write_index);
        values.resize(write_index);
        --num_cols; // Decrement the number of columns
    }

    // Delete a row in COO mode and shift subsequent rows
    void delete_row(int row_to_delete) {
        if (!coo_mode) switchToCOO(); // Ensure we're in COO mode

        size_t write_index = 0;

        for (size_t i = 0; i < rows.size(); ++i) {
            if (rows[i] != row_to_delete) {
                // Copy non-deleted row information
                rows[write_index]   = rows[i];
                cols[write_index]   = cols[i];
                values[write_index] = values[i];

                // If the row is greater than the deleted row, shift it down by 1
                if (rows[i] > row_to_delete) { rows[write_index]--; }

                ++write_index;
            }
        }

        // Resize the vectors to remove the deleted row's entries
        rows.resize(write_index);
        cols.resize(write_index);
        values.resize(write_index);

        // Decrement the total number of rows
        --num_rows;
    }

    // Matrix-vector multiplication in CRS mode
    std::vector<double> multiply(const std::vector<double> &x) {
        if (coo_mode) convertToCRS(); // Convert to CRS before multiplication

        std::vector<double> result(num_rows, 0.0);
        for (int row = 0; row < num_rows; ++row) {
            double sum = 0.0;
            for (int i = crs_row_start[row]; i < crs_row_start[row + 1]; ++i) { sum += crs_values[i] * x[crs_cols[i]]; }
            result[row] = sum;
        }
        return result;
    }

    // Modify or delete an element in COO mode
    void modify_or_delete(int row, int col, double value) {
        if (!coo_mode) switchToCOO(); // Ensure we're working in COO mode

        // Modify or delete element in COO format
        for (size_t i = 0; i < values.size(); ++i) {
            if (rows[i] == row && cols[i] == col) {
                if (value == 0.0) {
                    // Delete element
                    rows.erase(rows.begin() + i);
                    cols.erase(cols.begin() + i);
                    values.erase(values.begin() + i);

                    // Check if we've deleted the last element from the matrix
                    if (row == num_rows - 1 &&
                        std::none_of(rows.begin(), rows.end(), [row](int r) { return r == row; })) {
                        --num_rows; // Decrement num_rows if the last row is now empty
                    }
                } else {
                    // Modify element
                    values[i] = value;
                }
                return;
            }
        }

        // If the element doesn't exist and value is non-zero, insert it
        if (value != 0.0) insert(row, col, value);
    }

    // Convert to dense format (for testing/debugging purposes)
    std::vector<std::vector<double>> toDense() const {
        std::vector<std::vector<double>> dense(num_rows, std::vector<double>(num_cols, 0.0));
        for (size_t i = 0; i < values.size(); ++i) { dense[rows[i]][cols[i]] = values[i]; }
        return dense;
    }

    // Inline accessors for row_start (const version)
    const std::vector<int> &row_start() const {
        if (coo_mode) { throw std::runtime_error("row_start is only available in CRS mode."); }
        return crs_row_start;
    }

    // Inline accessors for row_start (non-const version)
    std::vector<int> &row_start() {
        if (coo_mode) { throw std::runtime_error("row_start is only available in CRS mode."); }
        return crs_row_start;
    }

    // Build the row start indices for CRS mode
    void buildRowStart() {
        if (coo_mode) convertToCRS();
    }
};
