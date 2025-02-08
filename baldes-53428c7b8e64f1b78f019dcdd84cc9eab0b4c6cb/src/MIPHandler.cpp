#include "../include/miphandler/MIPHandler.h"

#include <iostream>
#include <memory>

void MIPProblem::addVars(const double *lb, const double *ub, const double *obj,
                         const VarType *vtypes, const std::string *names,
                         const MIPColumn *cols, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        // Add each variable with its bounds, objective, and type
        add_variable(names[i], vtypes[i], lb[i], ub[i], obj[i]);
        gurobiCache->addColumn(cols[i]);

        // Get the index of the newly added variable (last variable)
        int col_index = variables.size() - 1;

        // Get the column information (non-zero coefficients) for the new
        // variable
        auto column = cols[i];
        auto terms = column.getTerms();

        // Prepare batch vectors for sparse matrix insertion
        std::vector<int> batch_rows;
        std::vector<int> batch_cols;
        std::vector<double> batch_values;

        // Prepare updates for each constraint
        std::vector<std::pair<int, double>>
            constraint_updates[constraints.size()];

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

void MIPProblem::addVars(const double *lb, const double *ub, const double *obj,
                         const VarType *vtypes, const std::string *names,
                         size_t count) {
    // Pre-allocate space in containers
    variables.reserve(variables.size() + count);
    var_name_to_index.reserve(var_name_to_index.size() + count);

    size_t start_index = variables.size();

    // Batch create all variables
    for (size_t i = 0; i < count; ++i) {
        auto newVar = std::make_shared<baldesVar>(names[i], vtypes[i], lb[i],
                                                  ub[i], obj[i]);
        newVar->set_index(start_index + i);
        variables.emplace_back(newVar);
        var_name_to_index.emplace(names[i], start_index + i);
    }
}

// Variable-Variable operators - these are correct
std::vector<std::pair<baldesVarPtr, double>> operator+(
    const baldesVarPtr &var1, const baldesVarPtr &var2) {
    return {{var1, 1.0}, {var2, 1.0}};
}

std::vector<std::pair<baldesVarPtr, double>> operator-(
    const baldesVarPtr &var1, const baldesVarPtr &var2) {
    return {{var1, 1.0}, {var2, -1.0}};
}

// Variable-Expression operators
LinearExpression operator+(const baldesVarPtr &var,
                           const LinearExpression &expr) {
    LinearExpression result = expr;
    result.addTerm(var, 1.0);
    return result;
}

LinearExpression operator-(const baldesVarPtr &var,
                           const LinearExpression &expr) {
    LinearExpression result;
    result.addTerm(var, 1.0);
    for (auto &[v, coeff] : expr.get_terms()) {
        result.addTerm(v, -coeff);
    }
    return result;
}

// Expression-Variable operators
LinearExpression operator+(const LinearExpression &expr,
                           const baldesVarPtr &var) {
    LinearExpression result = expr;
    result.addTerm(var, 1.0);
    return result;
}

LinearExpression operator-(const LinearExpression &expr,
                           const baldesVarPtr &var) {
    LinearExpression result = expr;
    result.addTerm(var, -1.0);
    return result;
}

// Expression-Expression operators
LinearExpression operator+(const LinearExpression &expr1,
                           const LinearExpression &expr2) {
    LinearExpression result = expr1;
    for (auto &[var, coeff] : expr2.get_terms()) {
        if (result.get_terms().find(var) != result.get_terms().end()) {
            result.get_terms()[var] += coeff;
        } else {
            result.get_terms()[var] = coeff;
        }
    }
    return result;
}

LinearExpression operator-(const LinearExpression &expr1,
                           const LinearExpression &expr2) {
    LinearExpression result = expr1;
    for (auto &[var, coeff] : expr2.get_terms()) {
        if (result.get_terms().find(var) != result.get_terms().end()) {
            result.get_terms()[var] -= coeff;
        } else {
            result.get_terms()[var] = -coeff;
        }
    }
    return result;
}

//
