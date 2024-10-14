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

#ifdef HIGHS
#include <Highs.h>
#endif

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

    Constraint add_constraint(const LinearExpression &expression, double rhs, char relation) {
        int constraint_index = constraints.size(); // Get the current index

        // Add the constraint to the list of constraints and set its index
        Constraint new_constraint(expression, rhs, relation);
        new_constraint.set_index(constraint_index); // Set the constraint's index

        constraints.push_back(new_constraint); // Add to constraints list

        // Convert the LinearExpression to sparse row format and add to the sparse matrix
        int row_index = sparse_matrix.num_rows;
        constraint_row_indices.push_back(row_index); // Track the new row index

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
        return new_constraint;
    }

    Constraint &add_constraint(Constraint &constraint, const std::string &name) {
        int constraint_index = constraints.size(); // Get the current index

        // Set the index and name of the constraint
        constraint.set_index(constraint_index);
        constraint.set_name(name);

        constraints.push_back(constraint); // Add the constraint to the list
        auto relation = constraint.get_relation();
        // print rhs
        // Convert the LinearExpression inside the constraint to sparse row format and add it to the sparse matrix
        int row_index = sparse_matrix.num_rows;
        constraint_row_indices.push_back(row_index); // Track the new row index

        const LinearExpression &expression = constraint.get_expression();

        for (const auto &[var_name, coeff] : expression.get_terms()) {
            // Find the variable index corresponding to var_name
            auto var_it = std::find_if(variables.begin(), variables.end(),
                                       [&var_name](const Variable &var) { return var.get_name() == var_name; });

            if (var_it != variables.end()) {
                int col_index = std::distance(variables.begin(), var_it);
                sparse_matrix.insert(row_index, col_index, coeff);
            } else {
                fmt::print("Variable {} not found in the problem's variables list!\n", var_name);
            }
        }

        // Update row start for CRS
        sparse_matrix.buildRowStart();
        fmt::print("Constraint '{}' added with index {} at row {}\n", name, constraint_index, row_index);

        // Return reference to the constraint in the list
        return constraints.back();
    }

    // Delete a constraint (row) from the problem
    void delete_constraint(int constraint_index) {
        if (constraint_index >= 0 && constraint_index < constraints.size()) {
            // Get the row index of the constraint
            int row_to_delete = constraint_row_indices[constraint_index];

            // Delete the row from the sparse matrix
            sparse_matrix.delete_row(row_to_delete);

            // Remove the constraint from the constraints list and the row index from the tracking vector
            constraints.erase(constraints.begin() + constraint_index);
            constraint_row_indices.erase(constraint_row_indices.begin() + constraint_index);

            // Update row indices of the remaining constraints
            for (int i = constraint_index; i < constraint_row_indices.size(); ++i) {
                constraint_row_indices[i]--; // Decrease each subsequent row index
            }

            // Rebuild the row structure
            sparse_matrix.buildRowStart();
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

            // Get the corresponding row index in the sparse matrix
            int row_to_delete = constraint_row_indices[constraintIndex];

            // Delete the row from the sparse matrix
            sparse_matrix.delete_row(row_to_delete);

            // Remove the constraint from the constraints list and the row index from the tracking vector
            constraints.erase(it);
            constraint_row_indices.erase(constraint_row_indices.begin() + constraintIndex);

            // Update the row indices for remaining constraints
            for (int i = constraintIndex; i < constraint_row_indices.size(); ++i) {
                constraint_row_indices[i]--; // Decrease each subsequent row index
            }

            // Rebuild the row structure
            sparse_matrix.buildRowStart();
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

#ifdef HIGHS
#include "Highs.h"

    // Function to populate a HiGHS model from this MIPProblem instance
    HighsModel toHighsModel() {
        // Create a new HiGHS model
        Highs      highs;
        HighsModel highsModel;

        // Reserve space for variables and constraints
        int numVars             = variables.size();
        int numConstrs          = constraints.size();
        highsModel.lp_.num_col_ = numVars;
        highsModel.lp_.num_row_ = numConstrs;

        // Resize vectors for variable bounds and objective coefficients
        highsModel.lp_.col_lower_.resize(numVars);
        highsModel.lp_.col_upper_.resize(numVars);
        highsModel.lp_.col_cost_.resize(numVars);

        // Resize vectors for constraint bounds
        highsModel.lp_.row_lower_.resize(numConstrs);
        highsModel.lp_.row_upper_.resize(numConstrs);

        // Sparse matrix format for constraints
        std::vector<int>    startIndices; // For CSR format, holds the starting index of each constraint row
        std::vector<int>    indices;      // Variable indices for each non-zero entry
        std::vector<double> values;       // Coefficients for each non-zero entry
        int                 nzCount = 0;  // Non-zero element counter

        startIndices.push_back(nzCount); // First position should be zero

        // Map to store variable index
        ankerl::unordered_dense::map<std::string, int> highsVars;
        int                                            var_index = 0;

        // Step 1: Add variables to the HiGHS model
        for (const auto &var : variables) {
            highsModel.lp_.col_lower_[var_index] = var.get_lb();
            highsModel.lp_.col_upper_[var_index] = var.get_ub();
            highsModel.lp_.col_cost_[var_index]  = var.get_objective_coefficient();
            highsVars[var.get_name()]            = var_index;
            var_index++;
        }

        // Step 2: Add constraints to the HiGHS model
        int row_index = 0;
        for (const auto &constraint : constraints) {
            double lower_bound, upper_bound;
            char   relation = constraint.get_relation();
            double rhs      = constraint.get_rhs();

            // Set the lower and upper bounds for each constraint based on its type
            if (relation == '<') {
                lower_bound = -kHighsInf; // Lower bound is negative infinity
                upper_bound = rhs;        // Upper bound is the RHS of the constraint
            } else if (relation == '>') {
                lower_bound = rhs;       // Lower bound is the RHS of the constraint
                upper_bound = kHighsInf; // Upper bound is positive infinity
            } else if (relation == '=') {
                lower_bound = upper_bound = rhs; // Both bounds are equal for equality constraints
            }

            // Assign the bounds to the HiGHS model
            highsModel.lp_.row_lower_[row_index] = lower_bound;
            highsModel.lp_.row_upper_[row_index] = upper_bound;

            // Add the terms (sparse matrix entries) for this constraint
            for (const auto &term : constraint.get_terms()) {
                int    varIndex = highsVars.at(term.first);
                double coeff    = term.second;
                indices.push_back(varIndex); // Store the variable index
                values.push_back(coeff);     // Store the coefficient
                nzCount++;
            }

            startIndices.push_back(nzCount); // Mark the start of the next row
            row_index++;
        }

        // Step 3: Assign the sparse matrix values to the HiGHS model
        highsModel.lp_.a_matrix_.start_  = startIndices;
        highsModel.lp_.a_matrix_.index_  = indices;
        highsModel.lp_.a_matrix_.value_  = values;
        highsModel.lp_.a_matrix_.format_ = MatrixFormat::kRowwise;

        // Step 4: Set the objective direction (assuming minimization problem)
        highsModel.lp_.sense_ = ObjSense::kMinimize;

        // Add the model to HiGHS
        highs.passModel(highsModel);

        return highsModel;
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
    std::vector<int> constraint_row_indices; // Track the row index of each constraint

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
