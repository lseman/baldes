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
#include "Highs.h"
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
        lb_vec.push_back(lb);
        ub_vec.push_back(ub);
        c_vec.push_back(obj_coeff);
        var_name_to_index[var_name] = variables.size() - 1;
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

    Constraint *add_constraint(const LinearExpression &expression, double rhs, char relation) {
        int constraint_index = constraints.size(); // Get the current index

        // Add the constraint to the list of constraints and set its index
        auto new_constraint = new Constraint(expression, rhs, relation);
        b.push_back(rhs);
        new_constraint->set_index(constraint_index); // Set the constraint's index

        constraints.push_back(new_constraint); // Add to constraints list

        // Convert the LinearExpression to sparse row format and add to the sparse matrix
        int row_index = sparse_matrix.num_rows;

        // Assuming you have many variables, use a more efficient lookup like a map
        for (const auto &[var_name, coeff] : expression.get_terms()) {
            auto it = var_name_to_index.find(var_name);
            if (it != var_name_to_index.end()) {
                int col_index = it->second;
                sparse_matrix.insert(row_index, col_index, coeff);
            } else {
                fmt::print("Variable {} not found in the problem's variables list!\n", var_name);
            }
        }

        // Update row start for CRS
        // sparse_matrix.buildRowStart();
        return new_constraint;
    }

    Constraint *add_constraint(Constraint *constraint, const std::string &name) {
        int constraint_index = constraints.size(); // Get the current index

        // Set the index and name of the constraint
        constraint->set_index(constraint_index);
        constraint->set_name(name);

        // Add the constraint to the list
        constraints.push_back(constraint);
        b.push_back(constraint->get_rhs());
        // Get the expression and relation (assuming it needs to be used later)
        const LinearExpression &expression = constraint->get_expression();

        // Add terms of the expression into the sparse matrix
        int row_index = sparse_matrix.num_rows;

        for (const auto &[var_name, coeff] : expression.get_terms()) {

            int col_index = var_name_to_index[var_name];
            sparse_matrix.insert(row_index, col_index, coeff);
        }

        // Delay the CRS structure rebuild if batching is possible
        // sparse_matrix.buildRowStart();

        // Return reference to the added constraint
        return constraints.back();
    }

    // Delete a constraint (row) from the problem
    void delete_constraint(int constraint_index) {
        if (constraint_index < 0 || constraint_index >= constraints.size()) {
            throw std::out_of_range("Invalid constraint index");
        }

        // Delete the row from the sparse matrix (assumed efficient row deletion)
        sparse_matrix.delete_row(constraint_index);

        // Erase the constraint from the list
        constraints.erase(constraints.begin() + constraint_index);
        b.erase(b.begin() + constraint_index);

        // Update the indices of the remaining constraints (this step can be costly if many constraints exist)
        for (int i = constraint_index; i < constraints.size(); ++i) { constraints[i]->set_index(i); }

        // Rebuild the row structure only if needed (if sparse_matrix requires full row structure rebuild)
        // sparse_matrix.buildRowStart();
    }

    void delete_constraint(Constraint *constraint) { delete_constraint(constraint->index()); }

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

    void chgCoeff(int constraintIndex, const std::vector<double> &values) {
        if (constraintIndex < 0 || constraintIndex >= constraints.size()) {
            throw std::out_of_range("Invalid constraint index");
        }

        // Get the constraint that is being modified
        Constraint       *constraint = constraints[constraintIndex];
        LinearExpression &expression = constraint->get_expression();

        // Iterate over the values and only update changed entries
        for (int i = 0; i < values.size(); ++i) {
            double new_value = values[i];

            // Update the sparse matrix only if the value has changed
            sparse_matrix.modify_or_delete(constraintIndex, i, new_value);

            const std::string &var_name = variables[i].get_name();

            // Update or remove terms in the LinearExpression
            if (new_value != 0.0) {
                expression.add_or_update_term(var_name, new_value); // Optimized to add/update the term
            } else {
                expression.remove_term(var_name); // Remove the term if the value is 0
            }
        }

        // Rebuild the CRS structure once after all updates
        // sparse_matrix.buildRowStart();
    }

    void chgCoeff(int constraintIndex, int variableIndex, double value) {
        if (constraintIndex < 0 || constraintIndex >= constraints.size() || variableIndex < 0 ||
            variableIndex >= variables.size()) {
            throw std::out_of_range("Invalid constraint or variable index");
        }

        // Update the sparse matrix only if the value has changed
        sparse_matrix.modify_or_delete(constraintIndex, variableIndex, value);

        // Update the LinearExpression in the corresponding Constraint
        Constraint       *constraint = constraints[constraintIndex];
        LinearExpression &expression = constraint->get_expression();

        const std::string &var_name = variables[variableIndex].get_name();

        // Update or remove terms in the LinearExpression
        if (value != 0.0) {
            expression.add_or_update_term(var_name, value); // Add or update the term (variable, coefficient)
        } else {
            expression.remove_term(var_name); // Remove the term if the value is 0
        }

        // Delay calling buildRowStart() until after all updates, if possible
        // sparse_matrix.buildRowStart();
    }

    void addVars(const double *lb, const double *ub, const double *obj, const VarType *vtypes, const std::string *names,
                 const MIPColumn *cols, size_t count);

    void addVars(const double *lb, const double *ub, const double *obj, const VarType *vtypes, const std::string *names,
                 size_t count);
    // Get all variables
    std::vector<Variable> &getVars() { return variables; }
    // Get all constraints
    std::vector<Constraint *> &getConstraints() { return constraints; }

    // Method to get the b vector (RHS values of all constraints)
    std::vector<double> get_b_vector() const {
        std::vector<double> b;
        for (const auto &constraint : constraints) { b.push_back(constraint->get_rhs()); }
        return b;
    }

    void chgCoeff(Constraint *constraint, const std::vector<double> &new_coeffs) {
        auto constraintIndex = constraint->index();
        // Change the coefficients for the constraint
        chgCoeff(constraintIndex, new_coeffs);
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
            if (constraint->get_relation() == '<') {
                gurobiModel.addConstr(gurobiExpr <= constraint->get_rhs(), constraint->get_name());
            } else if (constraint->get_relation() == '>') {
                gurobiModel.addConstr(gurobiExpr >= constraint->get_rhs(), constraint->get_name());
            } else if (constraint->get_relation() == '=') {
                gurobiModel.addConstr(gurobiExpr == constraint->get_rhs(), constraint->get_name());
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
        for (const auto &term : constraint->get_terms()) {
            const std::string &varName = term.first;
            double             coeff   = term.second;
            expr += gurobiVars.at(varName) * coeff;
        }
        return expr;
    }
#endif

#ifdef HIGHS

    // Function to populate a HiGHS model from this MIPProblem instance
    // Function to populate a HiGHS model from this MIPProblem instance
    HighsModel toHighsModel() {
        sparse_matrix.buildRowStart(); // Ensure CRS format is up-to-date

        // Create a new HiGHS model
        Highs      highs;
        HighsModel highsModel;

        int numVars    = variables.size();
        int numConstrs = constraints.size();

        // Set up the number of variables and constraints
        highsModel.lp_.num_col_ = numVars;
        highsModel.lp_.num_row_ = numConstrs;

        // Resize vectors for variable bounds and objective coefficients
        highsModel.lp_.col_lower_.resize(numVars);
        highsModel.lp_.col_upper_.resize(numVars);
        highsModel.lp_.col_cost_.resize(numVars);

        // Set variable bounds and cost
        int var_index = 0;
        for (const auto &var : variables) {
            highsModel.lp_.col_lower_[var_index] = var.get_lb();
            highsModel.lp_.col_upper_[var_index] = var.get_ub();
            highsModel.lp_.col_cost_[var_index]  = var.get_objective_coefficient();
            var_index++;
        }

        // Resize vectors for constraint bounds
        highsModel.lp_.row_lower_.resize(numConstrs);
        highsModel.lp_.row_upper_.resize(numConstrs);

        // Set constraint bounds
        int row_index = 0;
        for (const auto &constraint : constraints) {
            double lower_bound, upper_bound;
            char   relation = constraint->get_relation();
            double rhs      = constraint->get_rhs();

            // Set lower and upper bounds for each constraint based on its type
            if (relation == '<') {
                lower_bound = -kHighsInf;
                upper_bound = rhs;
            } else if (relation == '>') {
                lower_bound = rhs;
                upper_bound = kHighsInf;
            } else if (relation == '=') {
                lower_bound = upper_bound = rhs;
            }

            highsModel.lp_.row_lower_[row_index] = lower_bound;
            highsModel.lp_.row_upper_[row_index] = upper_bound;
            row_index++;
        }

        // Assign sparse matrix data directly from `sparse_matrix`
        highsModel.lp_.a_matrix_.start_  = sparse_matrix.getRowStart();
        highsModel.lp_.a_matrix_.index_  = sparse_matrix.getIndices();
        highsModel.lp_.a_matrix_.value_  = sparse_matrix.getValues();
        highsModel.lp_.a_matrix_.format_ = MatrixFormat::kRowwise;

        // Set the objective direction (assuming minimization)
        highsModel.lp_.sense_ = ObjSense::kMinimize;

        // Add the model to HiGHS
        highs.passModel(highsModel);

        return highsModel;
    }

#endif

    double getSlack(int row, const std::vector<double> &solution) {

        if (sparse_matrix.is_dirty) { sparse_matrix.buildRowStart(); }
        auto rhs = constraints[row]->get_rhs();

        // Compute the dot product of the solution vector and the specified row
        double                    row_value = 0.0;
        SparseMatrix::RowIterator it        = sparse_matrix.rowIterator(row);

        // Pre-fetch the values and reduce function calls
        while (it.valid()) {
            int    col_index = it.col();
            double value     = it.value();
            row_value += value * solution[col_index];
            it.next();
        }

        // Compute the slack as: slack = rhs - row_value
        double slack = rhs - row_value;
        return slack;
    }

    std::vector<double> get_lbs() const {
        std::vector<double> lbs;
        lbs.reserve(variables.size()); // Reserve space upfront
        std::transform(variables.begin(), variables.end(), std::back_inserter(lbs),
                       [](const Variable &var) { return var.get_lb(); });
        return lbs;
    }

    std::vector<double> get_ubs() const {
        std::vector<double> ubs;
        ubs.reserve(variables.size()); // Reserve space upfront
        std::transform(variables.begin(), variables.end(), std::back_inserter(ubs),
                       [](const Variable &var) { return var.get_ub(); });
        return ubs;
    }

    std::vector<double> get_c() const {
        std::vector<double> c;
        c.reserve(variables.size()); // Reserve space upfront
        std::transform(variables.begin(), variables.end(), std::back_inserter(c),
                       [](const Variable &var) { return var.get_objective_coefficient(); });
        return c;
    }

    std::vector<char> get_senses() const {
        std::vector<char> senses;
        senses.reserve(constraints.size()); // Reserve space upfront
        std::transform(constraints.begin(), constraints.end(), std::back_inserter(senses),
                       [](const auto &constraint) { return constraint->get_relation(); });
        return senses;
    }

    std::vector<char> get_vtypes() const {
        std::vector<char> vtypes;
        vtypes.reserve(variables.size()); // Reserve space upfront
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
        sparse_matrix.buildRowStart(); // Build the row start structure for CRS format
        ModelData data;
        data.A_sparse = sparse_matrix;
        data.b        = b;
        data.c        = c_vec;
        data.lb       = lb_vec;
        data.ub       = ub_vec;
        data.vtype    = get_vtypes();
        data.sense    = get_senses();

        return data;
    }

private:
    std::string                                    name;
    std::vector<Variable>                          variables;
    std::vector<Constraint *>                      constraints;    // Store the constraints
    LinearExpression                               objective;      // Store the objective function
    ObjectiveType                                  objective_type; // Minimize or Maximize
    SparseMatrix                                   sparse_matrix;  // Use SparseMatrix for coefficient storage
    ankerl::unordered_dense::map<std::string, int> var_name_to_index;
    std::vector<double>                            b;
    std::vector<double>                            c_vec;
    std::vector<double>                            lb_vec;
    std::vector<double>                            ub_vec;
};

class MIPColumn {
public:
    // Add a term to the column (row index and coefficient)
    void addTerm(int row_index, double value) {
        if (value != 0.0) { terms.push_back({row_index, value}); }
    }

    // Clear the column for reuse
    void                                clear() { terms.clear(); }
    std::vector<std::pair<int, double>> getTerms() const { return terms; }

private:
    std::vector<std::pair<int, double>> terms; // Pairs of row index and value
};
