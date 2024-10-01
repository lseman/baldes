#pragma once

#include "Definitions.h"
#include <algorithm>
#include <bitset>
#include <execution>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>

// Custom dense vector replacement
using DenseVector = std::vector<double>;

struct RowActivity {
    double min     = 0.0;
    double max     = 0.0;
    int    ninfmin = 0;
    int    ninfmax = 0;
};

enum class BoundChange { kLower, kUpper };

struct RowInfo {
    enum class RowFlag : uint8_t {
        None    = 0,
        kLhsInf = 1 << 0, // 0b00000001
        kRhsInf = 1 << 1  // 0b00000010
    };

    uint8_t rowFlag; // Store flags as a bitmask using uint8_t
    double  lhs;
    double  rhs;
    double  scale = 1.0;

    // Constructor accepting flags directly as bitmask or combined flags
    RowInfo(RowFlag flag, double lhs, double rhs, double scale = 1.0)
        : rowFlag(static_cast<uint8_t>(flag)), lhs(lhs), rhs(rhs), scale(scale) {}

    // Helper function to set a flag
    void setFlag(RowFlag flag) { rowFlag |= static_cast<uint8_t>(flag); }

    // Helper function to check if a specific flag is set
    bool hasFlag(RowFlag flag) const { return rowFlag & static_cast<uint8_t>(flag); }

    // Helper function to clear a flag
    void clearFlag(RowFlag flag) { rowFlag &= ~static_cast<uint8_t>(flag); }
};

// Operator overloads to combine flags
inline RowInfo::RowFlag operator|(RowInfo::RowFlag lhs, RowInfo::RowFlag rhs) {
    return static_cast<RowInfo::RowFlag>(static_cast<uint8_t>(lhs) | static_cast<uint8_t>(rhs));
}

struct PresolveResult {
    std::map<int, std::tuple<int, double, int>> fixCol;
    std::map<int, std::tuple<int, double, int>> changeColLB;
    std::map<int, std::tuple<int, double, int>> changeColUB;
    std::map<int, int>                          redundantRow;

    void clear() {
        fixCol.clear();
        changeColLB.clear();
        changeColUB.clear();
        redundantRow.clear();
    }
};

struct NumericUtils {
    static constexpr double hugeval                 = 1e8;
    static constexpr double feastol                 = 1e-6;
    static constexpr double weaken_bounds           = 1e-2;
    static constexpr double bound_tightening_offset = 1e-1;
    static constexpr double epsilon                 = 1e-12;
    static constexpr double infty                   = std::numeric_limits<double>::infinity();

    template <typename R1, typename R2>
    static bool isFeasLE(const R1 &a, const R2 &b) {
        return a - b <= NumericUtils::feastol;
    }

    template <typename R1, typename R2>
    static bool isFeasGE(const R1 &a, const R2 &b) {
        return a - b >= -NumericUtils::feastol;
    }
};

class Preprocessor {
public:
    std::vector<double>      lhs_values;
    std::vector<double>      rhs_values;
    std::vector<RowInfo>     rowInfoVector;
    std::vector<RowActivity> activities;

    SparseMatrix A; // Custom sparse matrix
    DenseVector  b;
    DenseVector  c;
    DenseVector  ub;
    DenseVector  lb;

    std::vector<char> sense;
    std::vector<char> vtype;

    PresolveResult result;

    // Constructor initializes the class by converting the input ModelData
    Preprocessor(ModelData &modelData) {
        A     = modelData.A_sparse;
        b     = modelData.b;
        c     = modelData.c;
        ub    = modelData.ub;
        lb    = modelData.lb;
        sense = modelData.sense;
        vtype = modelData.vtype;
    }
    void calculateRowActivities() {
        activities.resize(A.num_rows);
        for (size_t i = 0; i < activities.size(); ++i) {
            RowActivity &activity = activities[i];
            for (SparseMatrix::RowIterator it = A.rowIterator(i); it.valid(); it.next()) {
                double coef = it.value();
                int    j    = it.col();

                if (ub[j] != NumericUtils::infty) {
                    coef < 0 ? activity.min += coef * ub[j] : activity.max += coef * ub[j];
                } else {
                    coef < 0 ? ++activity.ninfmin : ++activity.ninfmax;
                }

                if (lb[j] != -NumericUtils::infty) {
                    coef < 0 ? activity.max += coef * lb[j] : activity.min += coef * lb[j];
                } else {
                    coef < 0 ? ++activity.ninfmax : ++activity.ninfmin;
                }
            }
        }
    }

    void processRowInformation() {
        rowInfoVector.clear();
        lhs_values.clear();
        rhs_values.clear();

        auto processBound = [&](char sense, double value) {
            switch (sense) {
            case '<':
                // LHS is -inf, so set the kLhsInf flag
                rowInfoVector.emplace_back(RowInfo::RowFlag::kLhsInf, -NumericUtils::infty, value);
                lhs_values.push_back(-NumericUtils::infty);
                rhs_values.push_back(value);
                break;
            case '>':
                // RHS is +inf, so set the kRhsInf flag
                rowInfoVector.emplace_back(RowInfo::RowFlag::kRhsInf, value, NumericUtils::infty);
                lhs_values.push_back(value);
                rhs_values.push_back(NumericUtils::infty);
                break;
            case '=':
                // Neither LHS nor RHS are infinite, so no flags
                rowInfoVector.emplace_back(RowInfo::RowFlag::None, value, value);
                lhs_values.push_back(value);
                rhs_values.push_back(value);
                break;
            default: std::cerr << "Invalid sense: " << sense << std::endl; return false;
            }
            return true;
        };

        // Iterate through the sense vector and process each bound
        for (size_t i = 0; i < sense.size(); ++i) { processBound(sense[i], b[i]); }
    }

    void flipRowSigns(int row) {
        for (SparseMatrix::RowIterator it = A.rowIterator(row); it.valid(); it.next()) {
            A.elements[it.index].value *= -1;
        }
    }

    void convert2LE() {
        for (int i = 0; i < A.num_rows; ++i) {
            if (sense[i] == '>') {
                lhs_values[i] = -NumericUtils::infty;
                rhs_values[i] = -lhs_values[i];
                sense[i]      = '<';
                flipRowSigns(i);
            }
        }
    }

    // Convert a row to a knapsack form
    void convert2Knapsack(int row) {
        std::cout << "Converting row " << row << " to knapsack\n";
        SparseMatrix::RowIterator rowIter = A.rowIterator(row);
        std::vector<double>       binCoeffs;

        if (sense[row] != '<') {
            return; // Only convert if the constraint is less-than-or-equal-to
        }

        std::cout << "Row " << row << " has rhs " << rhs_values[row] << "\n";

        for (; rowIter.valid(); rowIter.next()) {
            int    j     = rowIter.col();
            double coeff = rowIter.value();

            if (vtype[j] != 'B') {
                // Move non-binary variables to the RHS
                if (coeff < 0 && lb[j] != -NumericUtils::infty) {
                    rhs_values[row] -= coeff * lb[j];
                } else if (ub[j] != NumericUtils::infty) {
                    rhs_values[row] -= coeff * ub[j];
                }
            } else {
                // Handle binary variables
                if (coeff > 0) {
                    binCoeffs.push_back(coeff);
                } else {
                    rhs_values[row] -= coeff;
                    binCoeffs.push_back(-coeff);
                }
            }
        }

        std::cout << "Row " << row << " has new rhs " << rhs_values[row] << "\n";
    }

    void boundTightening() {
        std::for_each(rowInfoVector.begin(), rowInfoVector.end(), [&](const RowInfo &rowInfo) {
            size_t                    i        = &rowInfo - &rowInfoVector[0]; // Get row index
            SparseMatrix::RowIterator rowIter  = A.rowIterator(i);
            double                    alphaMax = 0.0, alphaMin = 0.0;

            // Calculate min/max activities
            for (; rowIter.valid(); rowIter.next()) {
                double coeff = rowIter.value();
                int    j     = rowIter.col();
                alphaMin += coeff > 0 ? coeff * lb[j] : coeff * ub[j];
                alphaMax += coeff > 0 ? coeff * ub[j] : coeff * lb[j];
            }

            if (lhs_values[i] > alphaMax || alphaMin > rhs_values[i]) {
                std::cout << "Constraint " << i << " is infeasible.\n";
            }
        });
    }

    // The method you requested to include:
    void findKnapsackRows() {
        std::mutex knapsack_mutex;

        std::for_each(rowInfoVector.begin(), rowInfoVector.end(), [&](const RowInfo &rowInfo) {
            if (rowInfo.hasFlag(RowInfo::RowFlag::kLhsInf) && !rowInfo.hasFlag(RowInfo::RowFlag::kRhsInf)) {
                size_t rowIndex  = &rowInfo - &rowInfoVector[0];
                int    nbinaries = 0;

                for (SparseMatrix::RowIterator it = A.rowIterator(rowIndex); it.valid(); it.next()) {
                    if (vtype[it.col()] == 'B') ++nbinaries;
                }

                if (nbinaries > 2) {
                    // std::lock_guard<std::mutex> lock(knapsack_mutex);
                    knapsackRows.push_back(rowIndex);
                }
            }
        });
        std::cout << "Number of knapsack rows: " << knapsackRows.size() << "\n";
    }

    ModelData getModelData() {
        ModelData data;
        data.A_sparse = A;

        for (int i = 0; i < sense.size(); ++i) { data.b.push_back(sense[i] == '<' ? rhs_values[i] : lhs_values[i]); }

        data.c     = c;
        data.ub    = ub;
        data.lb    = lb;
        data.sense = sense;
        data.vtype = vtype;
        return data;
    }

    std::vector<int> knapsackRows;
};
