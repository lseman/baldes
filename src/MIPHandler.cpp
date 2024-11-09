#include "../include/miphandler/MIPHandler.h"
#include <iostream>
#include <memory>

void MIPProblem::addVars(const double *lb, const double *ub, const double *obj, const VarType *vtypes,
                         const std::string *names, const MIPColumn *cols, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        // Add each variable with its bounds, objective, and type
        add_variable(names[i], vtypes[i], lb[i], ub[i], obj[i]);

        // Get the index of the newly added variable (last variable)
        int col_index = variables.size() - 1;

        // Get the column information (non-zero coefficients) for the new variable
        auto column = cols[i];
        auto terms  = column.getTerms();

        // Prepare batch vectors for sparse matrix insertion
        std::vector<int>    batch_rows;
        std::vector<int>    batch_cols;
        std::vector<double> batch_values;

        // Prepare updates for each constraint
        std::vector<std::pair<int, double>> constraint_updates[constraints.size()];

        // Gather all terms for this variable
        for (const auto &[row_index, value] : terms) {

            // Batch the sparse matrix insertion
            batch_rows.push_back(row_index);
            batch_cols.push_back(col_index);
            batch_values.push_back(value);

            // Batch the constraint updates
            constraint_updates[row_index].emplace_back(col_index, value);
        }

        // Perform batch insertion into the sparse matrix
        sparse_matrix.insert_batch(batch_rows, batch_cols, batch_values);

        // Perform batch updates to the constraints
        for (size_t row = 0; row < constraints.size(); ++row) {
            for (const auto &[col_index, value] : constraint_updates[row]) {
                constraints[row]->addTerm(variables[col_index], value);
            }
        }
    }
}

void MIPProblem::addVars(const double *lb, const double *ub, const double *obj, const VarType *vtypes,
                         const std::string *names, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        // Add each variable with its bounds, objective, and type
        add_variable(names[i], vtypes[i], lb[i], ub[i], obj[i]);
    }
}

// Addition of two variables with an implicit coefficient of 1.0
std::vector<std::pair<baldesVarPtr, double>> operator+(const baldesVarPtr &var1, const baldesVarPtr &var2) {
    return {{var1, 1.0}, {var2, 1.0}};
}

// Subtraction of two variables with coefficients 1.0 and -1.0
std::vector<std::pair<baldesVarPtr, double>> operator-(const baldesVarPtr &var1, const baldesVarPtr &var2) {
    return {{var1, 1.0}, {var2, -1.0}};
}

// Addition of a baldesVarPtr with a LinearExpression
LinearExpression operator+(const baldesVarPtr &var, const LinearExpression &expr) {
    LinearExpression result = expr;
    result += var;
    return result;
}

// Subtraction of a baldesVarPtr with a LinearExpression
LinearExpression operator-(const baldesVarPtr &var, const LinearExpression &expr) {
    LinearExpression result = expr;
    result -= var;
    return result;
}

// Addition of a LinearExpression with a baldesVarPtr
LinearExpression operator+(const LinearExpression &expr, const baldesVarPtr &var) {
    LinearExpression result = expr;
    result += var;
    return result;
}

// Subtraction of a LinearExpression with a baldesVarPtr
LinearExpression operator-(const LinearExpression &expr, const baldesVarPtr &var) {
    LinearExpression result = expr;
    result -= var;
    return result;
}
