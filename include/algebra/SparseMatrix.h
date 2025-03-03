#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <future>
#include <iostream>
#include <memory>
#include <numeric>
#include <numeric>  // for std::partial_sum
#include <span>     // C++20
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "Common.h"  // Your common definitions

#if defined(IPM) || defined(IPM_ACEL)
#include <Eigen/Sparse>
#endif

// SparseMatrix in COO format with on-demand conversion to CRS
struct SparseMatrix {
    // COO storage
    std::vector<int> rows;       // Row indices for non-zero values
    std::vector<int> cols;       // Column indices for non-zero values
    std::vector<double> values;  // Non-zero values

    // CRS data structure (constructed on-demand)
    struct CRSData {
        std::vector<int> row_start;
        std::vector<int> cols;
        std::vector<double> values;

        CRSData() = default;
        CRSData(const CRSData &other)
            : row_start(other.row_start),
              cols(other.cols),
              values(other.values) {}
    };

    mutable bool coo_mode = true;  // Always starts in COO mode
    mutable std::shared_ptr<CRSData> crs_data;

    int num_rows = 0;
    int num_cols = 0;

    // Constructors
    SparseMatrix() = default;
    SparseMatrix(int num_rows, int num_cols)
        : num_rows(num_rows), num_cols(num_cols) {}

    // Copy constructor and assignment (using default semantics for shared_ptr)
    SparseMatrix(const SparseMatrix &other)
        : rows(other.rows),
          cols(other.cols),
          values(other.values),
          crs_data(other.crs_data),
          coo_mode(other.coo_mode),
          num_rows(other.num_rows),
          num_cols(other.num_cols) {}

    SparseMatrix &operator=(const SparseMatrix &other) {
        if (this != &other) {
            rows = other.rows;
            cols = other.cols;
            values = other.values;
            crs_data = other.crs_data;
            coo_mode = other.coo_mode;
            num_rows = other.num_rows;
            num_cols = other.num_cols;
        }
        return *this;
    }

    // Move semantics (default)
    SparseMatrix(SparseMatrix &&) noexcept = default;
    SparseMatrix &operator=(SparseMatrix &&) noexcept = default;

    // Insert a non-zero element (COO mode). Combines duplicate entries.
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

    // Batch insertion with deduplication via an unordered_map.
    void insert_batch(const std::vector<int> &batch_rows,
                      const std::vector<int> &batch_cols,
                      const std::vector<double> &batch_values) {
        if (!coo_mode) switchToCOO();
        const size_t batch_size = batch_values.size();
        if (batch_size == 0) return;

        num_rows = std::max(
            num_rows,
            *std::max_element(batch_rows.begin(), batch_rows.end()) + 1);
        num_cols = std::max(
            num_cols,
            *std::max_element(batch_cols.begin(), batch_cols.end()) + 1);

        std::unordered_map<uint64_t, double> element_map;
        element_map.reserve(rows.size() + batch_size);

        auto make_key = [](int row, int col) -> uint64_t {
            return (static_cast<uint64_t>(row) << 32) |
                   static_cast<uint64_t>(col);
        };

        // Add existing elements.
        for (size_t i = 0; i < rows.size(); ++i) {
            element_map[make_key(rows[i], cols[i])] = values[i];
        }

        // Add batch elements (combining duplicates).
        for (size_t i = 0; i < batch_size; ++i) {
            uint64_t key = make_key(batch_rows[i], batch_cols[i]);
            element_map[key] += batch_values[i];
        }

        // Rebuild COO arrays.
        rows.clear();
        cols.clear();
        values.clear();
        rows.reserve(element_map.size());
        cols.reserve(element_map.size());
        values.reserve(element_map.size());
        for (const auto &[key, val] : element_map) {
            rows.push_back(static_cast<int>(key >> 32));
            cols.push_back(static_cast<int>(key & 0xFFFFFFFF));
            values.push_back(val);
        }
    }

    // Convert from COO to CRS format (if currently in COO mode)
    void convertToCRS() const {
        if (!coo_mode) return;
        if (!crs_data) {
            crs_data = std::make_shared<CRSData>();
        }
        auto &crs = *crs_data;
        crs.row_start.assign(num_rows + 1, 0);

        // Count non-zeros per row.
        for (int row : rows) {
            ++crs.row_start[row + 1];
        }

        // Compute prefix sum to determine row start positions.
        std::partial_sum(crs.row_start.begin(), crs.row_start.end(),
                         crs.row_start.begin());

        size_t nnz = values.size();
        crs.cols.resize(nnz);
        crs.values.resize(nnz);

        std::vector<int> row_position = crs.row_start;
        for (size_t i = 0; i < nnz; ++i) {
            int pos = row_position[rows[i]]++;
            crs.cols[pos] = cols[i];
            crs.values[pos] = values[i];
        }
        coo_mode = false;
    }

    // Helper lambda to copy CRS data back to COO, used in switchToCOO.
    void copyCRSToCOO(const SparseMatrix::CRSData &crs) {
        rows.clear();
        cols.clear();
        values.clear();
        size_t nnz = crs.values.size();
        rows.reserve(nnz);
        cols.reserve(nnz);
        values.reserve(nnz);
        for (int row = 0; row < num_rows; ++row) {
            int row_start = crs.row_start[row];
            int row_end = (row + 1 < num_rows) ? crs.row_start[row + 1]
                                               : static_cast<int>(nnz);
            for (int i = row_start; i < row_end; ++i) {
                rows.push_back(row);
                cols.push_back(crs.cols[i]);
                values.push_back(crs.values[i]);
            }
        }
    }

    // Switch from CRS back to COO mode.
    void switchToCOO() {
        if (coo_mode) return;
        if (crs_data && crs_data.use_count() == 1) {
            copyCRSToCOO(*crs_data);
            crs_data.reset();
        } else if (crs_data) {
            CRSData temp = *crs_data;
            copyCRSToCOO(temp);
        }
        coo_mode = true;
    }

    // RowIterator for iterating over CRS rows.
    class RowIterator {
       public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = std::pair<int, double>;
        using difference_type = std::ptrdiff_t;
        using pointer = const value_type *;
        using reference = const value_type &;

       private:
        std::span<const int> cols;
        std::span<const double> values;
        size_t current_index;
        size_t end_index;

       public:
        RowIterator(std::span<const int> cols_, std::span<const double> values_,
                    size_t start, size_t end)
            : cols(cols_),
              values(values_),
              current_index(start),
              end_index(end) {}

        bool operator==(const RowIterator &other) const {
            return current_index == other.current_index;
        }
        bool operator!=(const RowIterator &other) const {
            return !(*this == other);
        }
        RowIterator &operator++() {
            if (current_index < end_index) ++current_index;
            return *this;
        }
        RowIterator operator++(int) {
            RowIterator tmp = *this;
            ++(*this);
            return tmp;
        }
        value_type operator*() const {
            if (current_index >= end_index)
                throw std::out_of_range("Iterator out of range");
            return {cols[current_index], values[current_index]};
        }
    };

    // Iterator factory methods (assuming CRS mode)
    RowIterator row_begin(int row) const {
        if (coo_mode)
            throw std::runtime_error("CRS data not available in COO mode.");
        const auto &crs = *crs_data;
        if (row < 0 || row >= num_rows)
            throw std::out_of_range("Row index out of bounds");
        size_t start = crs.row_start[row];
        size_t end =
            (row + 1 < num_rows) ? crs.row_start[row + 1] : crs.values.size();
        return RowIterator(crs.cols, crs.values, start, end);
    }

    RowIterator row_end(int row) const {
        if (coo_mode)
            throw std::runtime_error("CRS data not available in COO mode.");
        const auto &crs = *crs_data;
        size_t end =
            (row + 1 < num_rows) ? crs.row_start[row + 1] : crs.values.size();
        return RowIterator(crs.cols, crs.values, end, end);
    }

    // Example method using the iterator
    std::vector<double> get_row_values(int row) const {
        std::vector<double> result;
        for (auto it = row_begin(row); it != row_end(row); ++it) {
            result.push_back((*it).second);
        }
        return result;
    }

    // Delete a column in COO mode
    void delete_column(int col_to_delete) {
        if (!coo_mode) switchToCOO();
        size_t write_index = 0;
        for (size_t i = 0; i < cols.size(); ++i) {
            if (cols[i] == col_to_delete) continue;
            rows[write_index] = rows[i];
            cols[write_index] =
                (cols[i] > col_to_delete) ? cols[i] - 1 : cols[i];
            values[write_index] = values[i];
            ++write_index;
        }
        rows.resize(write_index);
        cols.resize(write_index);
        values.resize(write_index);
        --num_cols;
    }

    // Delete a row in COO mode and shift subsequent rows
    void delete_row(int row_to_delete) {
        if (!coo_mode) switchToCOO();
        size_t write_index = 0;
        for (size_t i = 0; i < rows.size(); ++i) {
            if (rows[i] == row_to_delete) continue;
            rows[write_index] =
                (rows[i] > row_to_delete) ? rows[i] - 1 : rows[i];
            cols[write_index] = cols[i];
            values[write_index] = values[i];
            ++write_index;
        }
        rows.resize(write_index);
        cols.resize(write_index);
        values.resize(write_index);
        --num_rows;
    }

    // Matrix-vector multiplication in CRS mode
    std::vector<double> multiply(const std::vector<double> &x) const {
        if (coo_mode) convertToCRS();
        std::vector<double> result(num_rows, 0.0);
        const auto &crs = *crs_data;
        for (int row = 0; row < num_rows; ++row) {
            int row_start = crs.row_start[row];
            int row_end = (row + 1 < num_rows) ? crs.row_start[row + 1]
                                               : crs.values.size();
            double sum = 0.0;
            for (int i = row_start; i < row_end; ++i) {
                sum += crs.values[i] * x[crs.cols[i]];
            }
            result[row] = sum;
        }
        return result;
    }

    // Modify or delete a single element in COO mode
    void modify_or_delete(int row, int col, double value) {
        if (!coo_mode) switchToCOO();
        for (size_t i = 0; i < values.size(); ++i) {
            if (rows[i] == row && cols[i] == col) {
                values[i] = value;
                return;
            }
        }
        if (value != 0.0) insert(row, col, value);
    }

    // Batch modification or deletion.
    void modify_or_delete_batch(const std::vector<int> &r,
                                const std::vector<int> &c,
                                const std::vector<double> &vals) {
        if (r.size() != c.size() || c.size() != vals.size())
            throw std::invalid_argument(
                "Input vectors must have the same length.");
        if (!coo_mode) switchToCOO();
        for (size_t i = 0; i < vals.size(); ++i) {
            modify_or_delete(r[i], c[i], vals[i]);
        }
    }

    // Compact the COO storage by removing marked deletions (assume rows marked
    // -1 are deleted)
    void compact() {
        if (!coo_mode) switchToCOO();
        size_t write_index = 0;
        for (size_t i = 0; i < values.size(); ++i) {
            if (rows[i] != -1) {
                if (write_index != i) {
                    rows[write_index] = rows[i];
                    cols[write_index] = cols[i];
                    values[write_index] = values[i];
                }
                ++write_index;
            }
        }
        rows.resize(write_index);
        cols.resize(write_index);
        values.resize(write_index);
    }

    // Convert to a dense matrix (for debugging/testing)
    std::vector<std::vector<double>> toDense() const {
        std::vector<std::vector<double>> dense(
            num_rows, std::vector<double>(num_cols, 0.0));
        for (size_t i = 0; i < values.size(); ++i) {
            dense[rows[i]][cols[i]] = values[i];
        }
        return dense;
    }

    // Accessors for CRS data (const and non-const)
    const std::vector<int> &getRowStart() const {
        if (coo_mode)
            throw std::runtime_error("CRS data not available in COO mode.");
        return crs_data->row_start;
    }
    std::vector<int> &getRowStart() {
        if (coo_mode)
            throw std::runtime_error("CRS data not available in COO mode.");
        return crs_data->row_start;
    }
    std::vector<int> &getIndices() {
        if (coo_mode)
            throw std::runtime_error("Indices only available in CRS mode.");
        return crs_data->cols;
    }
    std::vector<double> &getValues() {
        if (coo_mode)
            throw std::runtime_error("Values only available in CRS mode.");
        return crs_data->values;
    }

    // Build row start indices if needed (force conversion)
    void buildRowStart() {
        if (coo_mode) convertToCRS();
    }

    // Return number of rows (outer size)
    int outerSize() const { return num_rows; }

    // Compute sparsity as a fraction of non-zero elements.
    double sparsity() const {
        if (coo_mode) convertToCRS();
        int num_zeros = num_rows * num_cols - values.size();
        return 1.0 - static_cast<double>(num_zeros) / (num_rows * num_cols);
    }

    // Compress the matrix in COO mode by merging duplicate entries.
    void compress() {
        if (!coo_mode) switchToCOO();
        if (values.empty()) return;
        std::vector<std::tuple<int, int, double>> triplets;
        triplets.reserve(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            triplets.emplace_back(rows[i], cols[i], values[i]);
        }
        std::sort(triplets.begin(), triplets.end(),
                  [](const auto &a, const auto &b) {
                      return std::tie(std::get<0>(a), std::get<1>(a)) <
                             std::tie(std::get<0>(b), std::get<1>(b));
                  });
        size_t write_index = 0;
        for (size_t i = 1; i < triplets.size(); ++i) {
            if (std::get<0>(triplets[write_index]) ==
                    std::get<0>(triplets[i]) &&
                std::get<1>(triplets[write_index]) ==
                    std::get<1>(triplets[i])) {
                std::get<2>(triplets[write_index]) += std::get<2>(triplets[i]);
            } else {
                ++write_index;
                triplets[write_index] = triplets[i];
            }
        }
        triplets.resize(write_index + 1);
        rows.clear();
        cols.clear();
        values.clear();
        rows.reserve(triplets.size());
        cols.reserve(triplets.size());
        values.reserve(triplets.size());
        for (const auto &t : triplets) {
            rows.push_back(std::get<0>(t));
            cols.push_back(std::get<1>(t));
            values.push_back(std::get<2>(t));
        }
    }

#if defined(IPM) || defined(IPM_ACEL)
    // Convert to Eigen's SparseMatrix (const version) for IPM routines.
    Eigen::SparseMatrix<double> toEigenSparseMatrix() const {
        Eigen::SparseMatrix<double> eigenMatrix(num_rows, num_cols);
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            if (rows[i] < 0 || rows[i] >= num_rows || cols[i] < 0 ||
                cols[i] >= num_cols) {
                std::cerr << "Error: Index out of bounds - row: " << rows[i]
                          << ", col: " << cols[i] << std::endl;
                throw std::out_of_range(
                    "Row or column index out of bounds in COO matrix");
            }
            triplets.emplace_back(rows[i], cols[i], values[i]);
        }
        eigenMatrix.setFromTriplets(triplets.begin(), triplets.end());
        return eigenMatrix;
    }
#endif

};  // end SparseMatrix
