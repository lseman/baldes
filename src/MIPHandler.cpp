#include "../include/miphandler/MIPHandler.h"

#include <iostream>
#include <memory>

void MIPProblem::addVars(const double *lb, const double *ub, const double *obj,
                         const VarType *vtypes, const std::string *names,
                         const MIPColumn *cols, size_t count) {
    // Loop over each variable to add.
    for (size_t i = 0; i < count; ++i) {
        // Add variable with its bounds, objective, and type.
        add_variable(names[i], vtypes[i], lb[i], ub[i], obj[i]);
        gurobiCache->addColumn(cols[i]);

        // The new variable's index is the last element in 'variables'.
        int col_index = variables.size() - 1;

        // Get the nonzero terms from the MIP column.
        auto terms = cols[i].getTerms();

        // Reserve space for batch insertion based on the number of terms.
        std::vector<int> batch_rows;
        std::vector<int> batch_cols;
        std::vector<double> batch_values;
        batch_rows.reserve(terms.size());
        batch_cols.reserve(terms.size());
        batch_values.reserve(terms.size());

        // Cache the number of constraints.
        const size_t numCons = constraints.size();
        // Create a vector of vectors for constraint updates.
        std::vector<std::vector<std::pair<int, double>>> constraint_updates(
            numCons);

        // Collect all terms for this variable.
        for (const auto &[row_index, value] : terms) {
            batch_rows.push_back(row_index);
            batch_cols.push_back(col_index);
            batch_values.push_back(value);

            if (row_index < numCons) {  // Safety check.
                constraint_updates[row_index].emplace_back(col_index, value);
            }
        }

        // Perform batch insertion into the sparse matrix.
        sparse_matrix.insert_batch(batch_rows, batch_cols, batch_values);

        // Update constraints with the new variable's coefficients.
        for (size_t row = 0; row < numCons; ++row) {
            for (const auto &[col_idx, val] : constraint_updates[row]) {
                constraints[row]->addTerm(variables[col_idx], val);
            }
        }
    }
}

void MIPProblem::addVars(const double *lb, const double *ub, const double *obj,
                         const VarType *vtypes, const std::string *names,
                         size_t count) {
    // Reserve additional space in the containers to avoid reallocations.
    variables.reserve(variables.size() + count);
    var_name_to_index.reserve(var_name_to_index.size() + count);

    size_t start_index = variables.size();

    // Create and store each new variable.
    for (size_t i = 0; i < count; ++i) {
        // Create the variable using a shared pointer.
        auto newVar = std::make_shared<baldesVar>(names[i], vtypes[i], lb[i],
                                                  ub[i], obj[i]);
        newVar->set_index(start_index + i);
        variables.push_back(newVar);
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
