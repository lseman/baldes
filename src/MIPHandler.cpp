#include "../include/MIPHandler/MIPHandler.h"
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

        // Add the sparse column (non-zero coefficients) for the new variable
        for (const auto &[row_index, value] : terms) {
            // Update the sparse matrix with the new variable's coefficient in the given constraint (row_index)
            sparse_matrix.insert(row_index, col_index, value);

            // Update the corresponding constraint linear expression
            constraints[row_index].addTerm(variables[col_index],
                                           value); // Assuming constraints is a vector of linear expressions
        }
    }

    // Rebuild the sparse matrix structure
    sparse_matrix.buildRowStart();
}

void MIPProblem::addVars(const double *lb, const double *ub, const double *obj, const VarType *vtypes,
                         const std::string *names, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        // Add each variable with its bounds, objective, and type
        add_variable(names[i], vtypes[i], lb[i], ub[i], obj[i]);
    }

    // Rebuild the sparse matrix structure
    sparse_matrix.buildRowStart();
}

Constraint LinearExpression::operator>=(double rhs) const {
    return Constraint(*this, rhs, '>'); // '>' represents '>='
}
Constraint LinearExpression::operator<=(double rhs) const {
    return Constraint(*this, rhs, '<'); // '<' represents '<='
}
Constraint LinearExpression::operator==(double rhs) const {
    return Constraint(*this, rhs, '='); // '=' represents '=='
}

// This overload allows an int to be multiplied by a Variable
std::pair<Variable, double> operator*(int coeff, Variable &var) {
    return {var, static_cast<double>(coeff)}; // Convert int to double if necessary
}
std::pair<Variable, double> operator*(double coeff, const Variable &var) { return {var, coeff}; }
