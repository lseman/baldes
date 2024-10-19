#pragma once

#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <vector>

#ifdef IPM
#include <Eigen/Sparse>
#endif

struct SparseMatrix {
    // COO format (always the default)
    std::vector<int>    rows;   // Row indices of the matrix elements
    std::vector<int>    cols;   // Column indices of the matrix elements
    std::vector<double> values; // Non-zero values of the matrix elements

    // CRS format (constructed on-demand)
    mutable std::vector<int>    crs_row_start;
    mutable std::vector<int>    crs_cols;
    mutable std::vector<double> crs_values;
    mutable bool                coo_mode = true; // Always starts in COO mode

    int num_rows = 0;
    int num_cols = 0;

    SparseMatrix() = default;
    SparseMatrix(int num_rows, int num_cols) : num_rows(num_rows), num_cols(num_cols) {}

    // Insert a non-zero element into the sparse matrix (COO mode)
    void insert(int row, int col, double value) {
        if (!coo_mode) switchToCOO(); // Ensure we're in COO mode

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
        if (!coo_mode) switchToCOO(); // Ensure we're in COO mode

        for (size_t i = 0; i < batch_values.size(); ++i) { insert(batch_rows[i], batch_cols[i], batch_values[i]); }
    }

    // Convert to CRS format (only if in COO mode)
    void convertToCRS() const {
        if (!coo_mode) return;

        crs_row_start.assign(num_rows + 1, 0); // CRS row starts initialized
        crs_cols.resize(values.size());
        crs_values.resize(values.size());

        // Count non-zero elements per row
        for (int row : rows) ++crs_row_start[row + 1];

        // Compute row start indices using cumulative sum
        std::partial_sum(crs_row_start.begin(), crs_row_start.end(), crs_row_start.begin());

        // Temporary row position tracker
        std::vector<int> row_position = crs_row_start;

        // Fill CRS data
        for (size_t i = 0; i < values.size(); ++i) {
            int row          = rows[i];
            int dest         = row_position[row]++;
            crs_cols[dest]   = cols[i];
            crs_values[dest] = values[i];
        }

        coo_mode = false; // Conversion complete
    }

    // Switch to COO mode
    void switchToCOO() {
        if (coo_mode) return;
        coo_mode = true;
        crs_row_start.clear();
        crs_cols.clear();
        crs_values.clear();
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
                cols[write_index]   = (cols[i] > col_to_delete) ? cols[i] - 1 : cols[i];
                values[write_index] = values[i];
                ++write_index;
            }
        }

        rows.resize(write_index);
        cols.resize(write_index);
        values.resize(write_index);
        --num_cols;
    }

    // Delete a row in COO mode and shift subsequent rows
    void delete_row(int row_to_delete) {
        if (!coo_mode) {
            switchToCOO(); // Ensure the matrix is in COO mode
        }

        // We use the write_index to overwrite the row being deleted and any subsequent rows.
        size_t write_index = 0;
        for (size_t i = 0; i < rows.size(); ++i) {
            // Skip the row to be deleted
            if (rows[i] == row_to_delete) { continue; }

            // Copy valid rows to the new index, adjusting rows that are after the deleted row
            rows[write_index]   = (rows[i] > row_to_delete) ? rows[i] - 1 : rows[i];
            cols[write_index]   = cols[i];
            values[write_index] = values[i];
            ++write_index;
        }

        // Resize to shrink the containers to the actual new size after deletion
        rows.resize(write_index);
        cols.resize(write_index);
        values.resize(write_index);

        // Decrease the number of rows in the matrix
        --num_rows;
    }

    // Matrix-vector multiplication in CRS mode
    std::vector<double> multiply(const std::vector<double> &x) {
        if (coo_mode) convertToCRS(); // Convert to CRS before multiplication

        std::vector<double> result(num_rows, 0.0);
        for (int row = 0; row < num_rows; ++row) {
            double sum = 0.0;
            for (int i = crs_row_start[row]; i < (row + 1 < num_rows) ? crs_row_start[row + 1] : crs_values.size();
                 ++i) {
                sum += crs_values[i] * x[crs_cols[i]];
            }
            result[row] = sum;
        }
        return result;
    }

    // Modify or delete an element in COO mode
    void modify_or_delete(int row, int col, double value) {
        if (!coo_mode) switchToCOO(); // Ensure we're working in COO mode

        // Search for the element in the COO format
        for (size_t i = 0; i < values.size(); ++i) {
            if (rows[i] == row && cols[i] == col) {

                // Modify the element value
                values[i] = value;

                return;
            }
        }

        // If the element doesn't exist and value is non-zero, insert it
        if (value != 0.0) insert(row, col, value);
    }

    void modify_or_delete_batch(const std::vector<int> &rows, const std::vector<int> &cols,
                                const std::vector<double> &values) {
        if (rows.size() != cols.size() || cols.size() != values.size()) {
            throw std::invalid_argument("The input vectors must have the same length.");
        }

        if (!coo_mode) switchToCOO(); // Ensure we're working in COO mode

        // Iterate through the batch of values to modify or delete
        for (size_t i = 0; i < values.size(); ++i) { modify_or_delete(rows[i], cols[i], values[i]); }
    }

    // Compact function to clean up the marked deletions
    void compact() {
        if (!coo_mode) switchToCOO(); // Ensure we're in COO mode

        size_t write_index = 0;
        for (size_t i = 0; i < values.size(); ++i) {
            if (rows[i] != -1) { // Skip marked deletions
                if (write_index != i) {
                    rows[write_index]   = rows[i];
                    cols[write_index]   = cols[i];
                    values[write_index] = values[i];
                }
                ++write_index;
            }
        }

        // Resize vectors to remove the deleted elements
        rows.resize(write_index);
        cols.resize(write_index);
        values.resize(write_index);
    }

    // Convert to dense format (for testing/debugging purposes)
    std::vector<std::vector<double>> toDense() const {
        std::vector<std::vector<double>> dense(num_rows, std::vector<double>(num_cols, 0.0));
        for (size_t i = 0; i < values.size(); ++i) { dense[rows[i]][cols[i]] = values[i]; }
        return dense;
    }

    // Inline accessors for row_start (const version)
    const std::vector<int> &getRowStart() const {
        if (coo_mode) { throw std::runtime_error("row_start is only available in CRS mode."); }
        return crs_row_start;
    }

    // Inline accessors for row_start (non-const version)
    std::vector<int> &getRowStart() {
        if (coo_mode) { throw std::runtime_error("row_start is only available in CRS mode."); }
        return crs_row_start;
    }

    std::vector<int> &getIndices() {
        if (coo_mode) { throw std::runtime_error("indices are only available in CRS mode."); }
        return crs_cols;
    }

    std::vector<double> &getValues() {
        if (coo_mode) { throw std::runtime_error("values are only available in CRS mode."); }
        return crs_values;
    }

    // Build the row start indices for CRS mode
    void buildRowStart() {
        if (coo_mode) convertToCRS();
    }

    // Returns the outer size (number of rows)
    int outerSize() const { return num_rows; }

    void compress() {
        if (!coo_mode) {
            std::cerr << "Compress is only applicable in COO mode." << std::endl;
            return;
        }

        // Check if the matrix has any elements to compress
        if (values.empty()) { return; }

        // Step 1: Create a vector of tuples (row, col, value) for sorting
        std::vector<std::tuple<int, int, double>> triplets;
        triplets.reserve(values.size());

        for (size_t i = 0; i < values.size(); ++i) { triplets.emplace_back(rows[i], cols[i], values[i]); }

        // Step 2: Sort the triplets by (row, col)
        std::sort(triplets.begin(), triplets.end(), [](const auto &a, const auto &b) {
            return std::tie(std::get<0>(a), std::get<1>(a)) < std::tie(std::get<0>(b), std::get<1>(b));
        });

        // Step 3: Remove duplicates by summing their values
        size_t write_index = 0;
        for (size_t i = 1; i < triplets.size(); ++i) {
            if (std::get<0>(triplets[write_index]) == std::get<0>(triplets[i]) &&
                std::get<1>(triplets[write_index]) == std::get<1>(triplets[i])) {
                // Same (row, col) as the previous, sum the values
                std::get<2>(triplets[write_index]) += std::get<2>(triplets[i]);
            } else {
                // Move the current triplet to the next position in the compressed result
                ++write_index;
                triplets[write_index] = triplets[i];
            }
        }

        // Step 4: Resize the triplets to remove any unused space due to merging duplicates
        triplets.resize(write_index + 1);

        // Step 5: Rebuild the COO vectors from the compressed triplets
        rows.clear();
        cols.clear();
        values.clear();
        rows.reserve(triplets.size());
        cols.reserve(triplets.size());
        values.reserve(triplets.size());

        for (const auto &triplet : triplets) {
            rows.push_back(std::get<0>(triplet));
            cols.push_back(std::get<1>(triplet));
            values.push_back(std::get<2>(triplet));
        }

        // After compression, the matrix is now in an optimized state
    }

#ifdef IPM
    // Modify convertToCRS to be const
    Eigen::SparseMatrix<double> toEigenSparseMatrix() const {
        Eigen::SparseMatrix<double>         eigenMatrix(num_rows, num_cols);
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(values.size());

        for (size_t i = 0; i < values.size(); ++i) {
            // Check if row and column indices are within bounds
            if (rows[i] < 0 || rows[i] >= num_rows || cols[i] < 0 || cols[i] >= num_cols) {
                std::cerr << "Error: Index out of bounds - row: " << rows[i] << ", col: " << cols[i] << std::endl;
                throw std::out_of_range("Row or column index out of bounds in COO matrix");
            }

            triplets.emplace_back(rows[i], cols[i], values[i]);
        }

        eigenMatrix.setFromTriplets(triplets.begin(), triplets.end());
        return eigenMatrix;
    }

    // Declare mutable members that are modified in const functions

#endif
};
