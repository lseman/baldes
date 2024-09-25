#pragma once

#include "../include/Definitions.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <execution>
#include <bitset>
#include <mutex>

struct RowActivity {
    double min     = 0.0;
    double max     = 0.0;
    int    ninfmin = 0;
    int    ninfmax = 0;
};

enum class BoundChange { kLower, kUpper };


/**
 * @struct RowInfo
 * @brief Stores information about a row in a matrix, including flags for 
 *        infinite bounds, left-hand side (LHS) and right-hand side (RHS) values, 
 *        and a scaling factor.
 */
struct RowInfo {
    enum class RowFlag {
        kLhsInf, // Indicates LHS is -inf
        kRhsInf  // Indicates RHS is +inf
    };

    std::bitset<2> rowFlag; // Efficiently store boolean flags

    // Constructor
    RowInfo(std::bitset<2> flag, double lhs, double rhs, double scale = 1.0)
        : rowFlag(flag), lhs(lhs), rhs(rhs), scale(scale) {}

    double lhs;
    double rhs;
    double scale = 1.0;
};

/**
 * @struct PresolveResult
 * @brief A structure to store the results of the presolve process.
 *
 */
struct PresolveResult {
    std::map<int, std::tuple<int, double, int>> fixCol;
    std::map<int, std::tuple<int, double, int>> changeColLB;
    std::map<int, std::tuple<int, double, int>> changeColUB;
    std::map<int, int> redundantRow;

    void clear() {
        fixCol.clear();
        changeColLB.clear();
        changeColUB.clear();
        redundantRow.clear();
    }
};

/**
 * @struct NumericUtils
 * @brief A utility structure for numeric operations and constants.
 *
 * This structure provides a set of constants and utility functions for
 * numerical operations, particularly useful in optimization and solver
 * contexts.
 *
 */
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
        return a - b >= -feastol;
    }
};

/**
 * @class Preprocessor
 * @brief A class for preprocessing optimization model data.
 *
 * The Preprocessor class is responsible for converting input model data into a format suitable for optimization,
 * calculating row activities, processing row information, converting constraints, and identifying knapsack rows.
 */
class Preprocessor {
public:
    std::vector<double> lhs_values;
    std::vector<double> rhs_values;
    std::vector<RowInfo> rowInfoVector;
    std::vector<RowActivity> activities;

    Eigen::SparseMatrix<double> A;
    Eigen::VectorXd b;
    Eigen::VectorXd c;
    Eigen::VectorXd ub;
    Eigen::VectorXd lb;

    std::vector<char> sense;
    std::vector<char> vtype;

    PresolveResult result;

    // Constructor initializes the class by converting the input ModelData
    Preprocessor(ModelData &modelData) {
        A = convertToEigenSparseMatrix(modelData.A_sparse);
        b = Eigen::Map<Eigen::VectorXd>(modelData.b.data(), modelData.b.size());
        c = Eigen::Map<Eigen::VectorXd>(modelData.c.data(), modelData.c.size());
        ub = Eigen::Map<Eigen::VectorXd>(modelData.ub.data(), modelData.ub.size());
        lb = Eigen::Map<Eigen::VectorXd>(modelData.lb.data(), modelData.lb.size());
        sense = modelData.sense;
        vtype = modelData.vtype;
    }

    // Optimized row activity calculation
    void calculateRowActivities() {
        size_t rows = A.rows();
        activities.resize(rows); // Resize once for all rows

        std::for_each(std::execution::par, activities.begin(), activities.end(), [&](RowActivity &activity) {
            size_t i = &activity - &activities[0]; // Calculate row index
            for (Eigen::SparseMatrix<double>::InnerIterator it(A, i); it; ++it) {
                double coef = it.value();
                int j = it.col();

                if (ub[j] != NumericUtils::infty) {
                    if (coef < 0) {
                        activity.min += coef * ub[j];
                    } else {
                        activity.max += coef * ub[j];
                    }
                } else {
                    coef < 0 ? ++activity.ninfmin : ++activity.ninfmax;
                }

                if (lb[j] != -NumericUtils::infty) {
                    if (coef < 0) {
                        activity.max += coef * lb[j];
                    } else {
                        activity.min += coef * lb[j];
                    }
                } else {
                    coef < 0 ? ++activity.ninfmax : ++activity.ninfmin;
                }
            }
        });
    }

    void processRowInformation() {
        rowInfoVector.clear();
        lhs_values.clear();
        rhs_values.clear();

        auto processBound = [&](char sense, double value) {
            switch (sense) {
            case '<':
                rowInfoVector.emplace_back(std::bitset<2>(std::string("01")), -NumericUtils::infty, value);
                lhs_values.push_back(-NumericUtils::infty);
                rhs_values.push_back(value);
                break;
            case '>':
                rowInfoVector.emplace_back(std::bitset<2>(std::string("10")), value, NumericUtils::infty);
                lhs_values.push_back(value);
                rhs_values.push_back(NumericUtils::infty);
                break;
            case '=':
                rowInfoVector.emplace_back(std::bitset<2>(std::string("00")), value, value);
                lhs_values.push_back(value);
                rhs_values.push_back(value);
                break;
            default:
                std::cerr << "Invalid sense: " << sense << std::endl;
                return false;
            }
            return true;
        };

        for (size_t i = 0; i < sense.size(); ++i) {
            processBound(sense[i], b[i]);
        }
    }

    // Convert inequality constraints to less-than-or-equal-to form
    void convert2LE() {
        for (int i = 0; i < A.rows(); ++i) {
            if (sense[i] == '>') {
                lhs_values[i] = -NumericUtils::infty;
                rhs_values[i] = -lhs_values[i];
                sense[i] = '<';
                A.row(i) *= -1; // Flip the row's sign to convert '>' to '<'
            }
        }
    }

    // Convert a row to a knapsack form
    void convert2Knapsack(int row) {
        fmt::print("Converting row {} to knapsack\n", row);
        Eigen::VectorXd rowCoeffs = A.row(row);
        std::vector<double> binCoeffs;

        if (sense[row] != '<') {
            return; // Only convert if the constraint is less-than-or-equal-to
        }

        fmt::print("Row {} has rhs {}\n", row, rhs_values[row]);

        for (int j = 0; j < A.cols(); ++j) {
            if (vtype[j] != 'B') {
                // Move non-binary variables to the RHS
                if (rowCoeffs[j] < 0 && lb[j] != -NumericUtils::infty) {
                    rhs_values[row] -= rowCoeffs[j] * lb[j];
                } else if (ub[j] != NumericUtils::infty) {
                    rhs_values[row] -= rowCoeffs[j] * ub[j];
                }
            } else {
                // Handle binary variables
                if (rowCoeffs[j] > 0) {
                    binCoeffs.push_back(rowCoeffs[j]);
                } else {
                    A.row(row) *= -1;
                    rhs_values[row] -= rowCoeffs[j];
                    binCoeffs.push_back(-rowCoeffs[j]);
                }
            }
        }

        fmt::print("Row {} has new rhs {}\n", row, rhs_values[row]);
    }

    void bboundTightening() {
        std::for_each(std::execution::par, rowInfoVector.begin(), rowInfoVector.end(), [&](const RowInfo &rowInfo) {
            size_t i = &rowInfo - &rowInfoVector[0]; // Get row index
            const Eigen::VectorXd &rowCoeffs = A.row(i);
            double alphaMax = 0.0, alphaMin = 0.0;

            // Calculate min/max activities
            for (int j = 0; j < rowCoeffs.size(); ++j) {
                double coeff = rowCoeffs[j];
                alphaMin += coeff > 0 ? coeff * lb[j] : coeff * ub[j];
                alphaMax += coeff > 0 ? coeff * ub[j] : coeff * lb[j];
            }

            if (lhs_values[i] > alphaMax || alphaMin > rhs_values[i]) {
                fmt::print("Constraint {} is infeasible.\n", i);
            }
        });
    }

    void findKnapSackRows() {
        std::mutex knapsack_mutex;

        std::for_each(std::execution::par, rowInfoVector.begin(), rowInfoVector.end(), [&](const RowInfo &rowInfo) {
            if (rowInfo.rowFlag[static_cast<int>(RowInfo::RowFlag::kLhsInf)] &&
                !rowInfo.rowFlag[static_cast<int>(RowInfo::RowFlag::kRhsInf)]) {
                size_t rowIndex = &rowInfo - &rowInfoVector[0];
                int nbinaries = 0;

                for (Eigen::SparseMatrix<double>::InnerIterator it(A, rowIndex); it; ++it) {
                    if (vtype[it.col()] == 'B') ++nbinaries;
                }

                if (nbinaries > 2) {
                    std::lock_guard<std::mutex> lock(knapsack_mutex);
                    knapsackRows.push_back(rowIndex);
                }
            }
        });
        fmt::print("Number of knapsack rows: {}\n", knapsackRows.size());
    }

    // Conversion and helper functions
    Eigen::SparseMatrix<double> convertToEigenSparseMatrix(const SparseModel &sparseModel) {
        Eigen::SparseMatrix<double> eigenSparseMatrix(sparseModel.num_rows, sparseModel.num_cols);
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(sparseModel.values.size());

        for (size_t i = 0; i < sparseModel.values.size(); ++i) {
            triplets.emplace_back(sparseModel.row_indices[i], sparseModel.col_indices[i], sparseModel.values[i]);
        }

        eigenSparseMatrix.setFromTriplets(triplets.begin(), triplets.end());
        return eigenSparseMatrix;
    }

    SparseModel convertToSparseModel(const Eigen::SparseMatrix<double> &eigenSparseMatrix) {
        SparseModel sparseModel;
        sparseModel.num_rows = eigenSparseMatrix.rows();
        sparseModel.num_cols = eigenSparseMatrix.cols();

        for (int k = 0; k < eigenSparseMatrix.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(eigenSparseMatrix, k); it; ++it) {
                sparseModel.row_indices.push_back(it.row());
                sparseModel.col_indices.push_back(it.col());
                sparseModel.values.push_back(it.value());
            }
        }
        return sparseModel;
    }

    ModelData getModelData() {
        ModelData data;
        data.A_sparse = convertToSparseModel(A);

        for (int i = 0; i < sense.size(); ++i) {
            data.b.push_back(sense[i] == '<' ? rhs_values[i] : lhs_values[i]);
        }

        data.c = std::vector<double>(c.data(), c.data() + c.size());
        data.ub = std::vector<double>(ub.data(), ub.data() + ub.size());
        data.lb = std::vector<double>(lb.data(), lb.data() + lb.size());
        data.sense = sense;
        data.vtype = vtype;
        return data;
    }

    std::vector<int> knapsackRows;
};
