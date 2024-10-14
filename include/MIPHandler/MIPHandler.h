#pragma once
#include "Definitions.h"
#include "SparseMatrix.h" // Include your SparseMatrix class

#ifdef GUROBI
#include "gurobi_c++.h"
#endif

#include <iostream>
#include <optional>
#include <ranges>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "Constraint.h"
#include "LinExp.h"
#include "Variable.h"

#include <fmt/core.h>

std::pair<Variable, double> operator*(int coeff, Variable &var);

enum class ObjectiveType { Minimize, Maximize };

class MIPColumn;
// Class representing the MIP Problem
class MIPProblem {

public:
    MIPProblem(const std::string &name, int num_rows, int num_cols) : name(name), sparse_matrix(num_rows, num_cols) {}

    // Add a variable to the problem
    void add_variable(const std::string &var_name, VarType type, double lb = 0.0, double ub = 1.0,
                      double obj_coeff = 0.0) {
        variables.emplace_back(var_name, type, lb, ub, obj_coeff);
    }

    // Set the objective function
    void set_objective(const LinearExpression &expr, ObjectiveType obj_type) {
        objective      = expr;
        objective_type = obj_type;
    }

    void setObjectiveSense(ObjectiveType obj_type) { objective_type = obj_type; }

    // Get the objective type (minimize or maximize)
    ObjectiveType get_objective_type() const { return objective_type; }

    // Get the objective expression
    const LinearExpression &get_objective() const { return objective; }

    // Delete a variable (column) from the problem
    void delete_variable(int var_index) {
        if (var_index >= 0 && var_index < variables.size()) {
            // Delete the column from the sparse matrix
            sparse_matrix.delete_column(var_index);

            // Remove the variable from the variables list
            variables.erase(variables.begin() + var_index);
        } else {
            throw std::out_of_range("Invalid variable index");
        }
    }

    // Add a constraint from a LinearExpression
    void add_constraint(const LinearExpression &expression, double rhs, char relation) {
        // Add the constraint to the list of constraints
        constraints.emplace_back(expression, rhs, relation);

        // Convert the LinearExpression to sparse row format and add to the sparse matrix
        int row_index = sparse_matrix.num_rows;
        for (const auto &[var_name, coeff] : expression.get_terms()) {
            // Find the variable index corresponding to var_name
            auto var_it = std::find_if(variables.begin(), variables.end(),
                                       [&var_name](const Variable &var) { return var.get_name() == var_name; });

            if (var_it != variables.end()) {
                int col_index = std::distance(variables.begin(), var_it);
                sparse_matrix.insert(row_index, col_index, coeff);
            }
        }

        // Update row start for CRS
        sparse_matrix.buildRowStart();
    }

    // Delete a constraint (row) from the problem
    void delete_constraint(int constraint_index) {
        if (constraint_index >= 0 && constraint_index < constraints.size()) {
            // Delete the row from the sparse matrix
            sparse_matrix.delete_row(constraint_index);

            // Remove the constraint from the constraints list
            constraints.erase(constraints.begin() + constraint_index);
        } else {
            throw std::out_of_range("Invalid constraint index");
        }
    }

    void delete_constraint(const Constraint &constraint) {
        // Find the index of the given constraint in the constraints vector
        auto it = std::find_if(constraints.begin(), constraints.end(),
                               [&constraint](const Constraint &c) { return &c == &constraint; });

        if (it != constraints.end()) {
            // Get the index of the constraint
            int constraintIndex = std::distance(constraints.begin(), it);

            // Delete the row from the sparse matrix
            sparse_matrix.delete_row(constraintIndex);

            // Remove the constraint from the constraints list
            constraints.erase(it);
        } else {
            throw std::invalid_argument("Constraint not found.");
        }
    }

    void delete_variable(const Variable &variable) {
        // Find the index of the given variable in the variables vector
        auto it = std::find_if(variables.begin(), variables.end(),
                               [&variable](const Variable &v) { return &v == &variable; });

        if (it != variables.end()) {
            // Get the index of the variable
            int variableIndex = std::distance(variables.begin(), it);

            // Delete the column from the sparse matrix
            sparse_matrix.delete_column(variableIndex);

            // Remove the variable from the variables list
            variables.erase(it);
        } else {
            throw std::invalid_argument("Variable not found.");
        }
    }

    // Add a sparse coefficient to the constraint matrix
    void add_sparse_coefficient(int row, int col, double value) { sparse_matrix.insert(row, col, value); }

    // Build the CRS structure for the sparse matrix after all coefficients are added
    void build_coefficient_matrix() { sparse_matrix.buildRowStart(); }

    // Multiply the sparse matrix by a vector
    std::vector<double> multiply_with_vector(const std::vector<double> &x) const { return sparse_matrix.multiply(x); }

    // Print sparse matrix as dense (for debugging)
    void print_dense_matrix() const {
        auto dense = sparse_matrix.toDense();
        std::cout << "Dense Matrix Representation:\n";
        for (const auto &row : dense) {
            for (const auto &val : row) { std::cout << val << " "; }
            std::cout << "\n";
        }
        // print number of rows and columns
        std::cout << "Number of rows: " << dense.size() << "\n";
        std::cout << "Number of columns: " << dense[0].size() << "\n";
    }

    // Get a variable by index
    Variable &getVar(size_t index) {
        if (index >= variables.size()) { throw std::out_of_range("Variable index out of range"); }
        return variables[index];
    }

    // Change multiple coefficients for a specific constraint
    void chgCoeff(int constraintIndex, const std::vector<double> &values) {
        if (constraintIndex < 0 || constraintIndex >= constraints.size()) {
            throw std::out_of_range("Invalid constraint index");
        }
        for (int i = 0; i < values.size(); ++i) { sparse_matrix.insert(constraintIndex, i, values[i]); }
        sparse_matrix.buildRowStart();
    }

    // Change a single coefficient for a specific constraint and variable
    void chgCoeff(int constraintIndex, int variableIndex, double value) {
        if (constraintIndex < 0 || constraintIndex >= constraints.size() || variableIndex < 0 ||
            variableIndex >= variables.size()) {
            throw std::out_of_range("Invalid constraint or variable index");
        }
        sparse_matrix.insert(constraintIndex, variableIndex, value);
        sparse_matrix.buildRowStart();
    }

    void addVars(const double *lb, const double *ub, const double *obj, const VarType *vtypes, const std::string *names,
                 const MIPColumn *cols, size_t count);

    void addVars(const double *lb, const double *ub, const double *obj, const VarType *vtypes, const std::string *names,
                 size_t count);
    // Get all variables
    std::vector<Variable> &getVars() { return variables; }
    // Get all constraints
    std::vector<Constraint> &getConstraints() { return constraints; }

    void add_variable_with_sparse_column(const std::string &var_name, VarType type, double lb, double ub,
                                         const std::vector<std::pair<int, double>> &sparse_column) {
        // Add the variable to the problem
        variables.emplace_back(var_name, type, lb, ub);

        // Get the index of the newly added variable (last variable)
        int col_index = variables.size() - 1;

        // Add each non-zero element from the sparse column into the sparse matrix
        for (const auto &[row_index, value] : sparse_column) {
            sparse_matrix.insert(row_index, col_index, value); // Insert (row, col, value) into the matrix
        }

        // Ensure the matrix structure is updated
        sparse_matrix.buildRowStart(); // Rebuild row starts for CRS format
    }

    void add_constraint_with_sparse_row(const std::vector<std::pair<int, double>> &sparse_row) {
        // Get the index of the newly added row (new constraint)
        int row_index = sparse_matrix.num_rows; // The new row will be the current number of rows

        // Add each non-zero element from the sparse row into the sparse matrix
        for (const auto &[col_index, value] : sparse_row) {
            sparse_matrix.insert(row_index, col_index, value); // Insert (row, col, value) into the matrix
        }

        // Ensure the matrix structure is updated
        sparse_matrix.buildRowStart(); // Rebuild row starts for CRS format
    }

    // Method to get the b vector (RHS values of all constraints)
    std::vector<double> get_b_vector() const {
        std::vector<double> b;
        for (const auto &constraint : constraints) { b.push_back(constraint.get_rhs()); }
        return b;
    }

    void chgCoeff(const Constraint &constraint, const std::vector<double> &new_coeffs) {
        // Find the index of the constraint
        auto it = std::find_if(constraints.begin(), constraints.end(),
                               [&constraint](const Constraint &c) { return &c == &constraint; });

        if (it != constraints.end()) {
            // Get the index of the constraint
            int constraintIndex = std::distance(constraints.begin(), it);

            // Change the coefficients for the constraint
            chgCoeff(constraintIndex, new_coeffs);
        } else {
            throw std::invalid_argument("Constraint not found.");
        }
    }

    Constraint &addConstr(const LinearExpression &lhs, char relation, double rhs, const std::string &name) {
        Constraint new_constraint(lhs, rhs, relation);
        constraints.emplace_back(new_constraint);
        return constraints.back(); // Return a reference to the newly added constraint
    }

    Constraint &addConstr(const Constraint &constraint, const std::string &name) {
        constraints.push_back(constraint);
        return constraints.back(); // Return a reference to the newly added constraint
    }

#ifdef GUROBI
    // Function to populate a Gurobi model from this MIPProblem instance
    GRBModel toGurobiModel(GRBEnv &env) {
        // Create a new Gurobi model
        GRBModel gurobiModel(env);

        // Map to store Gurobi variables
        ankerl::unordered_dense::map<std::string, GRBVar> gurobiVars;

        // check if we have repeted variable name
        std::unordered_map<std::string, int> var_count;
        for (const auto &var : variables) {
            if (var_count.find(var.get_name()) == var_count.end()) {
                var_count[var.get_name()] = 1;
            } else {
                var_count[var.get_name()]++;
                fmt::print("Variable name {} is repeated {} times\n", var.get_name(), var_count[var.get_name()]);
            }
        }
        // Step 1: Add variables to the Gurobi model
        for (const auto &var : variables) {
            // Add each variable to the Gurobi model, according to its type and bounds
            GRBVar gurobiVar           = gurobiModel.addVar(var.get_lb(), var.get_ub(), var.get_objective_coefficient(),
                                                            toGRBVarType(var.get_type()), var.get_name());
            gurobiVars[var.get_name()] = gurobiVar;
        }

        // Step 2: Add constraints to the Gurobi model
        for (const auto &constraint : constraints) {
            GRBLinExpr gurobiExpr = convertToGurobiExpr(constraint, gurobiVars);
            if (constraint.get_relation() == '<') {
                gurobiModel.addConstr(gurobiExpr <= constraint.get_rhs(), constraint.get_name());
            } else if (constraint.get_relation() == '>') {
                gurobiModel.addConstr(gurobiExpr >= constraint.get_rhs(), constraint.get_name());
            } else if (constraint.get_relation() == '=') {
                gurobiModel.addConstr(gurobiExpr == constraint.get_rhs(), constraint.get_name());
            }
        }

        // Step 3: Set objective if needed (assuming a linear objective function)
        GRBLinExpr objective;
        for (const auto &var : variables) { objective += gurobiVars[var.get_name()] * var.get_objective_coefficient(); }
        gurobiModel.setObjective(objective, GRB_MINIMIZE); // Assume minimization problem
        gurobiModel.update();
        return gurobiModel;
    }

    // Helper function to convert MIP variable type to Gurobi variable type
    char toGRBVarType(VarType varType) {
        switch (varType) {
        case VarType::Continuous: return GRB_CONTINUOUS;
        case VarType::Integer: return GRB_INTEGER;
        case VarType::Binary: return GRB_BINARY;
        default: throw std::invalid_argument("Unknown variable type");
        }
    }

    // Helper function to convert MIP constraints into a Gurobi linear expression
    GRBLinExpr convertToGurobiExpr(const Constraint                                        &constraint,
                                   const ankerl::unordered_dense::map<std::string, GRBVar> &gurobiVars) {
        GRBLinExpr expr;
        for (const auto &term : constraint.get_terms()) {
            const std::string &varName = term.first;
            double             coeff   = term.second;
            expr += gurobiVars.at(varName) * coeff;
        }
        return expr;
    }
#endif

    std::vector<double> get_lbs() const {
        std::vector<double> lbs;
        for (const auto &var : variables) { lbs.push_back(var.get_lb()); }
        return lbs;
    }

    std::vector<double> get_ubs() const {
        std::vector<double> ubs;
        for (const auto &var : variables) { ubs.push_back(var.get_ub()); }
        return ubs;
    }

    std::vector<double> get_c() const {
        std::vector<double> c;
        for (const auto &var : variables) { c.push_back(var.get_objective_coefficient()); }
        return c;
    }

    std::vector<char> get_senses() const {
        std::vector<char> senses;
        for (const auto &constraint : constraints) { senses.push_back(constraint.get_relation()); }
        return senses;
    }

    std::vector<char> get_vtypes() const {
        std::vector<char> vtypes;
        for (const auto &var : variables) {
            if (var.get_type() == VarType::Continuous) {
                vtypes.push_back('C');
            } else if (var.get_type() == VarType::Integer) {
                vtypes.push_back('I');
            } else if (var.get_type() == VarType::Binary) {
                vtypes.push_back('B');
            }
        }
        return vtypes;
    }

    ModelData extractModelDataSparse() {
        ModelData data;
        data.A_sparse = sparse_matrix;
        data.b        = get_b_vector();
        data.c        = get_c();
        data.lb       = get_lbs();
        data.ub       = get_ubs();
        data.vtype    = get_vtypes();
        data.sense    = get_senses();

        return data;
    }

private:
    std::string             name;
    std::vector<Variable>   variables;
    std::vector<Constraint> constraints;    // Store the constraints
    LinearExpression        objective;      // Store the objective function
    ObjectiveType           objective_type; // Minimize or Maximize
    SparseMatrix            sparse_matrix;  // Use SparseMatrix for coefficient storage
};

class MIPColumn {
public:
    // Add a term to the column (row index and coefficient)
    void addTerm(int row_index, double value) {
        if (value != 0.0) { terms.push_back({row_index, value}); }
    }

    // Add all terms from this column to the MIPProblem sparse matrix for the given variable index
    void addToMatrix(MIPProblem &mip, int var_index) const {
        for (const auto &[row_index, value] : terms) { mip.add_sparse_coefficient(row_index, var_index, value); }
    }

    // Clear the column for reuse
    void                                clear() { terms.clear(); }
    std::vector<std::pair<int, double>> getTerms() const { return terms; }

private:
    std::vector<std::pair<int, double>> terms; // Pairs of row index and value
};
