#pragma once

#include "Common.h"

#if defined(IPM) || defined(IPM_ACEL)
#include <Eigen/Sparse>
#endif

// include for cerr
#include <iostream>

struct SparseMatrix {
    // COO format (always the default)
    std::vector<int>    rows;   // Row indices of the matrix elements
    std::vector<int>    cols;   // Column indices of the matrix elements
    std::vector<double> values; // Non-zero values of the matrix elements

    // CRS format (constructed on-demand)
    struct CRSData {
        std::vector<int>    row_start;
        std::vector<int>    cols;
        std::vector<double> values;

        // Add copy constructor for CRSData
        CRSData() = default;
        CRSData(const CRSData &other) : row_start(other.row_start), cols(other.cols), values(other.values) {}
    };

    mutable bool                     coo_mode = true; // Always starts in COO mode
    mutable std::shared_ptr<CRSData> crs_data;

    int num_rows = 0;
    int num_cols = 0;

    SparseMatrix() = default;
    SparseMatrix(int num_rows, int num_cols) : num_rows(num_rows), num_cols(num_cols) {}

    // Copy constructor
    SparseMatrix(const SparseMatrix &other)
        : rows(other.rows), cols(other.cols), values(other.values),
          crs_data(other.crs_data) // shared_ptr handles copying
          ,
          coo_mode(other.coo_mode), num_rows(other.num_rows), num_cols(other.num_cols) {}

    // Copy assignment operator
    SparseMatrix &operator=(const SparseMatrix &other) {
        if (this != &other) {
            rows     = other.rows;
            cols     = other.cols;
            values   = other.values;
            crs_data = other.crs_data; // shared_ptr handles copying
            coo_mode = other.coo_mode;
            num_rows = other.num_rows;
            num_cols = other.num_cols;
        }
        return *this;
    }

    // Move constructor
    SparseMatrix(SparseMatrix &&) noexcept = default;

    // Move assignment
    SparseMatrix &operator=(SparseMatrix &&) noexcept = default;

    // Insert a non-zero element into the sparse matrix (COO mode)
    void insert(int row, int col, double value) {
        if (!coo_mode) switchToCOO();

        num_rows = std::max(num_rows, row + 1);
        num_cols = std::max(num_cols, col + 1);

        // Try to combine with existing element
        for (size_t i = 0; i < rows.size(); ++i) {
            if (rows[i] == row && cols[i] == col) {
                values[i] += value;
                return;
            }
        }

        rows.push_back(row);
        cols.push_back(col);
        values.push_back(value);
    }

    // Optimized batch insertion with sorting and deduplication
    void insert_batch(const std::vector<int> &batch_rows, const std::vector<int> &batch_cols,
                      const std::vector<double> &batch_values) {
        if (!coo_mode) switchToCOO();

        const size_t batch_size = batch_values.size();
        if (batch_size == 0) return;

        // Update matrix dimensions
        num_rows = std::max(num_rows, *std::max_element(batch_rows.begin(), batch_rows.end()) + 1);
        num_cols = std::max(num_cols, *std::max_element(batch_cols.begin(), batch_cols.end()) + 1);

        // Create a map for combining duplicates
        std::unordered_map<uint64_t, double> element_map;
        element_map.reserve(rows.size() + batch_size);

        // Helper function to create key from row and column
        auto make_key = [](int row, int col) -> uint64_t {
            return (static_cast<uint64_t>(row) << 32) | static_cast<uint64_t>(col);
        };

        // Add existing elements to map
        for (size_t i = 0; i < rows.size(); ++i) { element_map[make_key(rows[i], cols[i])] = values[i]; }

        // Add batch elements to map, combining duplicates
        for (size_t i = 0; i < batch_size; ++i) {
            uint64_t key = make_key(batch_rows[i], batch_cols[i]);
            element_map[key] += batch_values[i];
        }

        // Rebuild vectors from map
        rows.clear();
        cols.clear();
        values.clear();
        rows.reserve(element_map.size());
        cols.reserve(element_map.size());
        values.reserve(element_map.size());

        for (const auto &[key, value] : element_map) {
            rows.push_back(static_cast<int>(key >> 32));
            cols.push_back(static_cast<int>(key & 0xFFFFFFFF));
            values.push_back(value);
        }
    }

    // Convert to CRS format (only if in COO mode)
    void convertToCRS() const {
        if (!coo_mode) return;

        if (!crs_data) { crs_data = std::make_shared<CRSData>(); }
        auto &crs = *crs_data;

        // Initialize row starts
        crs.row_start.assign(num_rows + 1, 0);

        // Count elements per row
        for (int row : rows) { ++crs.row_start[row + 1]; }

        // Compute prefix sum for row starts
        std::partial_sum(crs.row_start.begin(), crs.row_start.end(), crs.row_start.begin());

        // Allocate CRS arrays
        const size_t nnz = values.size();
        crs.cols.resize(nnz);
        crs.values.resize(nnz);

        // Fill CRS arrays
        std::vector<int> row_position = crs.row_start;
        for (size_t i = 0; i < nnz; ++i) {
            const int pos   = row_position[rows[i]]++;
            crs.cols[pos]   = cols[i];
            crs.values[pos] = values[i];
        }

        coo_mode = false;
    }

    void switchToCOO() {
        if (coo_mode) return;

        // Only create new vectors if we're the only one using this CRS data
        if (crs_data.use_count() == 1) {
            const auto &crs = *crs_data;
            rows.clear();
            cols.clear();
            values.clear();

            const size_t nnz = crs.values.size();
            rows.reserve(nnz);
            cols.reserve(nnz);
            values.reserve(nnz);

            for (int row = 0; row < num_rows; ++row) {
                const int row_start = crs.row_start[row];
                const int row_end   = (row + 1 < num_rows) ? crs.row_start[row + 1] : crs.values.size();

                for (int i = row_start; i < row_end; ++i) {
                    rows.push_back(row);
                    cols.push_back(crs.cols[i]);
                    values.push_back(crs.values[i]);
                }
            }

            crs_data.reset();
        } else {
            // If others are using the CRS data, we need to make a copy
            const auto &crs = *crs_data;
            rows.clear();
            cols.clear();
            values.clear();

            const size_t nnz = crs.values.size();
            rows.reserve(nnz);
            cols.reserve(nnz);
            values.reserve(nnz);

            for (int row = 0; row < num_rows; ++row) {
                const int row_start = crs.row_start[row];
                const int row_end   = (row + 1 < num_rows) ? crs.row_start[row + 1] : crs.values.size();

                for (int i = row_start; i < row_end; ++i) {
                    rows.push_back(row);
                    cols.push_back(crs.cols[i]);
                    values.push_back(crs.values[i]);
                }
            }
        }

        coo_mode = true;
    }

    // RowIterator for CRS mode
    class RowIterator {
    private:
        const SparseMatrix &matrix;
        size_t              current_index;
        size_t              row_end;

    public:
        // STL iterator traits
        using iterator_category = std::forward_iterator_tag;
        using value_type        = std::pair<int, double>;
        using difference_type   = std::ptrdiff_t;
        using pointer           = const value_type *;
        using reference         = const value_type &;

        RowIterator(const SparseMatrix &matrix_, int row) : matrix(matrix_) {
            if (matrix.coo_mode) { matrix.convertToCRS(); }

            if (!matrix.crs_data) { throw std::runtime_error("CRS data not initialized"); }

            const auto &crs = *matrix.crs_data;

            if (row < 0 || row >= matrix.num_rows) { throw std::out_of_range("Row index out of bounds"); }

            current_index = crs.row_start[row];
            row_end       = (row + 1 < matrix.num_rows) ? crs.row_start[row + 1] : crs.values.size();
        }

        // STL-compatible iterator operations
        bool operator==(const RowIterator &other) const { return current_index == other.current_index; }

        bool operator!=(const RowIterator &other) const { return !(*this == other); }

        RowIterator &operator++() {
            if (current_index < row_end) { ++current_index; }
            return *this;
        }

        RowIterator operator++(int) {
            RowIterator tmp = *this;
            ++(*this);
            return tmp;
        }

        value_type operator*() const {
            if (current_index >= row_end) { throw std::out_of_range("Iterator out of range"); }
            const auto &crs = *matrix.crs_data;
            return {crs.cols[current_index], crs.values[current_index]};
        }

        // Additional helper methods
        bool valid() const { return current_index < row_end; }

        int col() const {
            if (!valid()) { throw std::out_of_range("Invalid iterator access"); }
            return matrix.crs_data->cols[current_index];
        }

        double value() const {
            if (!valid()) { throw std::out_of_range("Invalid iterator access"); }
            return matrix.crs_data->values[current_index];
        }
    };

    // Iterator factory methods
    RowIterator row_begin(int row) const { return RowIterator(*this, row); }

    RowIterator row_end(int row) const {
        auto it = RowIterator(*this, row);
        while (it.valid()) { ++it; }
        return it;
    }

    // Example usage of the iterator in a method
    std::vector<double> get_row_values(int row) const {
        std::vector<double> result;
        for (auto it = row_begin(row); it.valid(); ++it) { result.push_back(it.value()); }
        return result;
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
    std::vector<double> multiply(const std::vector<double> &x) const {
        if (coo_mode) convertToCRS();

        std::vector<double> result(num_rows, 0.0);
        const auto         &crs = *crs_data;

        for (int row = 0; row < num_rows; ++row) {
            const int row_start = crs.row_start[row];
            const int row_end   = (row + 1 < num_rows) ? crs.row_start[row + 1] : crs.values.size();

            double sum = 0.0;
            for (int i = row_start; i < row_end; ++i) { sum += crs.values[i] * x[crs.cols[i]]; }
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
        return crs_data->row_start;
    }

    // Inline accessors for row_start (non-const version)
    std::vector<int> &getRowStart() {
        if (coo_mode) { throw std::runtime_error("row_start is only available in CRS mode."); }
        return crs_data->row_start;
    }

    std::vector<int> &getIndices() {
        if (coo_mode) { throw std::runtime_error("indices are only available in CRS mode."); }
        return crs_data->cols;
    }

    std::vector<double> &getValues() {
        if (coo_mode) { throw std::runtime_error("values are only available in CRS mode."); }
        return crs_data->values;
    }

    // Build the row start indices for CRS mode
    void buildRowStart() {
        if (coo_mode) convertToCRS();
    }

    // Returns the outer size (number of rows)
    int outerSize() const { return num_rows; }

    double sparsity() const {
        if (coo_mode) convertToCRS(); // Convert to CRS before computing sparsity

        int num_zeros = num_rows * num_cols - values.size();
        return 1.0 - (static_cast<double>(num_zeros) / (num_rows * num_cols));
    }

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

#if defined(IPM) || defined(IPM_ACEL)
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
