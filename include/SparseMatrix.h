#pragma once

#include <algorithm>
#include <numeric> // for std::partial_sum
#include <vector>

struct SparseElement {
    int    row;
    int    col;
    double value;
};

/*
 * SparseMatrix class
 * Represents a sparse matrix in CRS format
 *
 * @brief SparseMatrix class
 */
struct SparseMatrix {
    std::vector<SparseElement> elements;
    std::vector<int>           row_start;
    int                        num_rows;
    int                        num_cols;

    // Optimized buildRowStart function
    void buildRowStart() {
        // Ensure row_start has enough space and is reset
        row_start.assign(num_rows + 1, 0);

        // Sort the elements by row first, then by column
        std::sort(elements.begin(), elements.end(), [](const SparseElement &a, const SparseElement &b) {
            return (a.row < b.row) || (a.row == b.row && a.col < b.col);
        });

        // Count occurrences of elements per row
        for (const auto &el : elements) { ++row_start[el.row + 1]; }

        // Accumulate counts to get starting index for each row
        std::partial_sum(row_start.begin(), row_start.end(), row_start.begin());
    }

    // Optimized row-wise iterator using indices
    struct RowIterator {
        const SparseMatrix &matrix;
        size_t              index;
        size_t              end;

        RowIterator(const SparseMatrix &matrix, int row)
            : matrix(matrix), index(matrix.row_start[row]), end(matrix.row_start[row + 1]) {}

        bool valid() const { return index < end; }

        void next() { ++index; }

        double value() const { return matrix.elements[index].value; }
        int    col() const { return matrix.elements[index].col; }
    };

    RowIterator rowIterator(int row) const { return RowIterator(*this, row); }

    // Batch processing for each row
    template <typename Func>
    void forEachRow(Func func) const {
        for (int row_idx = 0; row_idx < num_rows; ++row_idx) {
            RowIterator it = rowIterator(row_idx);
            while (it.valid()) {
                func(it.col(), it.value());
                it.next();
            }
        }
    }
};
