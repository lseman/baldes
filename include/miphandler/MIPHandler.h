/*
 * @file MIPHandler.h
 * @brief MIP Problem class implementation
 *
 * This file contains the implementation of the MIPProblem class.
 * The MIPProblem class represents a Mixed-Integer Programming (MIP) problem
 * and provides methods to add variables, constraints, and objective functions.
 *
 */

#pragma once
#include "Definitions.h"
#include "SparseMatrix.h" // Include your SparseMatrix class
#include <memory>

#ifdef GUROBI
#include "gurobi_c++.h"
#endif

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "Constraint.h"
#include "LinExp.h"
#include "Variable.h"

#include <fmt/core.h>
#include <sstream>

using baldesCtrPtr = std::shared_ptr<baldesCtr>;
using baldesVarPtr = std::shared_ptr<baldesVar>;
using LinearExpPtr = std::shared_ptr<LinearExpression>;

#ifdef HIGHS
#include "Highs.h"
#endif

// Multiplication of a variable by a coefficient (int or double)
template <typename T, typename = std::enable_if_t<std::is_arithmetic<T>::value>>
std::pair<baldesVarPtr, double> operator*(const baldesVarPtr &var, T coeff) {
    return {var, static_cast<double>(coeff)};
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic<T>::value>>
std::pair<baldesVarPtr, double> operator*(T coeff, const baldesVarPtr &var) {
    return {var, static_cast<double>(coeff)};
}

// Addition of a vector of terms with a single term
template <typename T, typename = std::enable_if_t<std::is_arithmetic<T>::value>>
std::vector<std::pair<baldesVarPtr, double>> operator+(const std::vector<std::pair<baldesVarPtr, double>> &terms,
                                                       const std::pair<baldesVarPtr, T>                   &term) {
    auto result = terms;
    if (term.second != 0.0) { result.push_back({term.first, static_cast<double>(term.second)}); }
    return result;
}

// Subtraction of a vector of terms with a single term
template <typename T, typename = std::enable_if_t<std::is_arithmetic<T>::value>>
std::vector<std::pair<baldesVarPtr, double>> operator-(const std::vector<std::pair<baldesVarPtr, double>> &terms,
                                                       const std::pair<baldesVarPtr, T>                   &term) {
    auto result = terms;
    if (term.second != 0.0) { result.push_back({term.first, static_cast<double>(-term.second)}); }
    return result;
}

// Addition of a single term with a vector of terms
template <typename T, typename = std::enable_if_t<std::is_arithmetic<T>::value>>
inline std::vector<std::pair<baldesVarPtr, double>>
operator+(const std::pair<baldesVarPtr, T> &term, const std::vector<std::pair<baldesVarPtr, double>> &terms) {
    auto result = terms;
    if (term.second != 0.0) { result.insert(result.begin(), {term.first, static_cast<double>(term.second)}); }
    return result;
}

// Subtraction of a single term with a vector of terms
template <typename T, typename = std::enable_if_t<std::is_arithmetic<T>::value>>
inline std::vector<std::pair<baldesVarPtr, double>>
operator-(const std::pair<baldesVarPtr, T> &term, const std::vector<std::pair<baldesVarPtr, double>> &terms) {
    auto result = terms;
    if (term.second != 0.0) { result.insert(result.begin(), {term.first, static_cast<double>(-term.second)}); }
    return result;
}

std::vector<std::pair<baldesVarPtr, double>> operator+(const baldesVarPtr &var1, const baldesVarPtr &var2);
std::vector<std::pair<baldesVarPtr, double>> operator-(const baldesVarPtr &var1, const baldesVarPtr &var2);
LinearExpression                             operator+(const baldesVarPtr &var, const LinearExpression &expr);
LinearExpression                             operator-(const baldesVarPtr &var, const LinearExpression &expr);
LinearExpression                             operator+(const LinearExpression &expr, const baldesVarPtr &var);
LinearExpression                             operator-(const LinearExpression &expr, const baldesVarPtr &var);
LinearExpression                             operator+(const LinearExpression &expr1, const LinearExpression &expr2);
LinearExpression                             operator-(const LinearExpression &expr1, const LinearExpression &expr2);

inline baldesCtrPtr operator<=(const baldesVarPtr &var, double rhs) {
    LinearExpression expr;
    expr.add_term(var->get_name(), 1.0);
    return std::make_shared<baldesCtr>(expr, rhs, '<');
}
inline baldesCtrPtr operator>=(const baldesVarPtr &var, double rhs) {
    LinearExpression expr;
    expr.add_term(var->get_name(), 1.0);
    return std::make_shared<baldesCtr>(expr, rhs, '>');
}

inline baldesCtrPtr operator==(const baldesVarPtr &var, double rhs) {
    LinearExpression expr;
    expr.add_term(var->get_name(), 1.0);
    return std::make_shared<baldesCtr>(expr, rhs, '=');
}

// operator for std::pair<baldesVarPtr, double> and LinearExpression
inline LinearExpression operator+(const std::pair<baldesVarPtr, double> &term, const LinearExpression &expr) {
    LinearExpression result = expr;
    result.add_term(term.first->get_name(), term.second);
    return result;
}

// operator std::vector<std::pair<baldesVarPtr, double>> and LinearExpression
inline LinearExpression operator+(const std::vector<std::pair<baldesVarPtr, double>> &terms,
                                  const LinearExpression                             &expr) {
    LinearExpression result = expr;
    for (const auto &term : terms) { result.add_term(term.first->get_name(), term.second); }
    return result;
}

// - operator for std::pair<baldesVarPtr, double> and LinearExpression
inline LinearExpression operator-(const std::pair<baldesVarPtr, double> &term, const LinearExpression &expr) {
    LinearExpression result = expr;
    // Add the term with a positive coefficient
    for (const auto &[var, coeff] : expr.get_terms()) { result.add_term(var, -coeff); }
    return result;
}

// - operator for std::vector<std::pair<baldesVarPtr, double>> and LinearExpression
inline LinearExpression operator-(const std::vector<std::pair<baldesVarPtr, double>> &terms,
                                  const LinearExpression                             &expr) {
    LinearExpression result;
    // Add terms with positive coefficients
    for (const auto &term : terms) { result.add_term(term.first->get_name(), term.second); }
    // Subtract the expression
    for (const auto &[var, coeff] : expr.get_terms()) { result.add_term(var, -coeff); }
    return result;
}

// int times LinearExpression
inline LinearExpression operator*(int coeff, const LinearExpression &expr) {
    LinearExpression result = expr;
    result.multiply_by_constant(coeff);
    return result;
}

enum class ObjectiveType { Minimize, Maximize };

class MIPColumn;
// Class representing the MIP Problem
class MIPProblem {

public:
    MIPProblem(const std::string &name, int num_rows, int num_cols) : name(name), sparse_matrix(num_rows, num_cols) {}

    MIPProblem *clone() const {
        // Create a new MIPProblem instance with the same name and matrix dimensions
        auto *clone = new MIPProblem(name, sparse_matrix.num_rows, sparse_matrix.num_cols);

        // Clone variables
        for (const auto var : variables) {
            baldesVarPtr newVar = std::make_shared<baldesVar>(var->get_name(), var->get_type(), var->get_lb(),
                                                              var->get_ub(), var->get_objective_coefficient());
            newVar->set_index(var->index());
            clone->variables.push_back(newVar);
            clone->var_name_to_index[var->get_name()] = newVar->index();
        }

        for (const auto &constraint : constraints) {
            auto clonedCtr = constraint->clone();
            clone->constraints.push_back(clonedCtr);
            clone->b_vec.push_back(clonedCtr->get_rhs());
        }

        // Clone the objective
        clone->objective      = objective;
        clone->objective_type = objective_type;

        // Deep copy the sparse matrix
        clone->sparse_matrix = sparse_matrix;

        return clone;
    }

    std::vector<int> reduceByRC(const std::vector<double> &dual_solution, const std::vector<int> &basic,
                                double keep_percentage = 0.7) {
        if (keep_percentage <= 0.0 || keep_percentage > 1.0) {
            throw std::invalid_argument("keep_percentage must be between 0 and 1");
        }

        // Get all RCs and their corresponding variable indices
        std::vector<std::pair<double, int>> rc_pairs;
        for (int i = 0; i < getVars().size(); i++) {
            double rc = getRC(i, dual_solution);
            rc_pairs.push_back({rc, i});
        }

        // Sort by absolute RC value in descending order
        std::sort(rc_pairs.begin(), rc_pairs.end(), [](const auto &a, const auto &b) { return a.first < b.first; });

        // Calculate how many variables to keep
        size_t keep_count = static_cast<size_t>(std::ceil(rc_pairs.size() * keep_percentage));

        // Create a set of indices to remove (the bottom 30%)
        ankerl::unordered_dense::set<int> indices_to_remove;
        for (size_t i = keep_count; i < rc_pairs.size(); i++) {
            int var_index = rc_pairs[i].second;
            // Skip variables that are in the basic set
            if (std::find(basic.begin(), basic.end(), var_index) != basic.end()) { continue; }
            indices_to_remove.insert(var_index);
        }

        // Remove variables from highest index to lowest to maintain validity of remaining indices
        std::vector<int> sorted_indices(indices_to_remove.begin(), indices_to_remove.end());
        std::sort(sorted_indices.begin(), sorted_indices.end(), std::greater<int>());

        // Delete variables
        for (int idx : sorted_indices) { delete_variable(idx); }

        // Rebuild var_name_to_index
        for (int i = 0; i < variables.size(); ++i) { var_name_to_index[variables[i]->get_name()] = i; }

        return sorted_indices;
    }

    // Add a variable to the problem
    baldesVarPtr add_variable(const std::string &var_name, VarType type, double lb = 0.0, double ub = 1.0,
                                          double obj_coeff = 0.0) {
        size_t index  = variables.size();
        auto   newVar = std::make_shared<baldesVar>(var_name, type, lb, ub, obj_coeff);
        newVar->set_index(index);
        variables.emplace_back(newVar);
        var_name_to_index.emplace(var_name, index); // Using emplace instead of [] operator
        return newVar;
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

            // delete the variable from the constraint linear expressions
            for (auto &constraint : constraints) {
                constraint->get_expression().remove_term(variables[var_index]->get_name());
            }

            // Remove the variable from the variables list
            variables.erase(variables.begin() + var_index);

            // reduce index of variables after the deleted variable
            for (int i = var_index; i < variables.size(); ++i) {
                variables[i]->set_index(i);
                var_name_to_index[variables[i]->get_name()] = i;
            }

        } else {
            throw std::out_of_range("Invalid variable index");
        }
    }

    bool constraint_exists(const std::string &name) {
        for (const auto &constraint : constraints) {
            if (constraint->get_name() == name) { return true; }
        }
        return false;
    }

    baldesCtrPtr get_constraint(const std::string &name) {
        for (const auto &constraint : constraints) {
            if (constraint->get_name() == name) { return constraint; }
        }
        return nullptr;
    }

    baldesCtrPtr add_constraint(const LinearExpression &expression, double rhs, char relation) {
        int constraint_index = constraints.size(); // Get the current index
        // Add the constraint to the list of constraints and set its index
        auto new_constraint = std::make_shared<baldesCtr>(expression, rhs, relation);
        b_vec.push_back(rhs);
        new_constraint->set_index(constraint_index); // Set the constraint's index

        constraints.push_back(new_constraint); // Add to constraints list

        // Convert the LinearExpression to sparse row format and add to the sparse matrix
        int row_index = constraint_index;
        // sparse_matrix.num_rows++;

        // Assuming you have many variables, use a more efficient lookup like a map
        // Original individual insertions
        std::vector<int>    batch_rows;
        std::vector<int>    batch_cols;
        std::vector<double> batch_values;

        // Pre-allocate vectors with expected size
        batch_rows.reserve(expression.get_terms().size());
        batch_cols.reserve(expression.get_terms().size());
        batch_values.reserve(expression.get_terms().size());

        // Collect all terms for batch insertion
        for (const auto &[var_name, coeff] : expression.get_terms()) {
            auto it = var_name_to_index.find(var_name);
            if (it != var_name_to_index.end()) {
                batch_rows.push_back(row_index);
                batch_cols.push_back(it->second);
                batch_values.push_back(coeff);
            } else {
                fmt::print("baldesVar {} not found in the problem's variables list!\n", var_name);
            }
        }

        // Perform batch insertion
        sparse_matrix.insert_batch(batch_rows, batch_cols, batch_values);

        // Update row start for CRS
        // sparse_matrix.buildRowStart();
        return new_constraint;
    }

    baldesCtrPtr add_constraint(baldesCtrPtr constraint, const std::string &name) {
        int constraint_index = constraints.size(); // Get the current index
        constraint->set_index(constraint_index);
        constraint->set_name(name);

        // Add the constraint to the list
        constraints.push_back(constraint);
        b_vec.push_back(constraint->get_rhs());
        // Get the expression and relation (assuming it needs to be used later)
        const LinearExpression &expression = constraint->get_expression();

        // Add terms of the expression into the sparse matrix
        int row_index = constraint_index;

        // Prepare batch vectors
        std::vector<int>    batch_rows;
        std::vector<int>    batch_cols;
        std::vector<double> batch_values;

        // Pre-allocate space
        batch_rows.reserve(expression.get_terms().size());
        batch_cols.reserve(expression.get_terms().size());
        batch_values.reserve(expression.get_terms().size());

        // Collect terms for batch insertion
        for (const auto &[var_name, coeff] : expression.get_terms()) {
            int col_index = var_name_to_index[var_name];
            batch_rows.push_back(row_index);
            batch_cols.push_back(col_index);
            batch_values.push_back(coeff);
        }

        // Perform batch insertion
        sparse_matrix.insert_batch(batch_rows, batch_cols, batch_values);

        // Return reference to the added constraint
        return constraints.back();
    }

    void printBranchingbaldesCtr() {
        for (const auto &constraint : constraints) {
            std::string name = constraint->get_name();
            if (name.find("branching") != std::string::npos) {
                std::stringstream        ss(name);
                std::string              segment;
                std::vector<std::string> parts;

                // Split by underscore
                while (std::getline(ss, segment, '_')) { parts.push_back(segment); }

                if (parts.size() >= 4 && parts[1] == "node") {
                    int node  = std::stoi(parts[2]);
                    int bound = std::stoi(parts[3]);

                    print_branching("Branching on node {} is '{}=' bound {}\n", node, constraint->get_relation(),
                                    bound);
                }
                if (parts.size() >= 4 && parts[1] == "edge") {
                    int source = std::stoi(parts[2]);
                    int target = std::stoi(parts[3]);
                    int bound  = std::stoi(parts[4]);

                    print_branching("Branching on edge ({},{}) with bound {}\n", source, target, bound);
                }
                if (parts.size() >= 3 && parts[1] == "vehicle") {
                    int bound = std::stoi(parts[2]);
                    print_branching("Branching on vehicle with bound {}\n", bound);
                }
                if (parts.size() >= 4 && parts[1] == "cluster") {
                    int cluster = std::stoi(parts[2]);
                    int bound   = std::stoi(parts[3]);
                    print_branching("Branching on cluster {} with bound {}\n", cluster, bound);
                }
            }
        }
    }

    int get_current_index(int unique_id) {
        // print the unique_id
        for (int i = 0; i < constraints.size(); i++) {
            if (constraints[i]->get_unique_id() == unique_id) { return i; }
        }
        return -1;
    }

    // Main implementation that does the actual deletion work
    void delete_constraint(int constraint_index) {
        // Delete the row from the sparse matrix
        sparse_matrix.delete_row(constraint_index);

        // Erase the constraint from the constraints vector and b_vec atomically
        constraints.erase(constraints.begin() + constraint_index);
        b_vec.erase(b_vec.begin() + constraint_index);

        // Update the indices of the remaining constraints
        // Consider using parallel processing if constraints.size() is very large
        for (int i = constraint_index; i < static_cast<int>(constraints.size()); ++i) {
            auto &constraint = constraints[i];
            if (constraint) { // null check
                int old_index = constraint->index();
                constraint->set_index(i);

            } else {
                std::cerr << "Null constraint found at position " << i << std::endl;
            }
        }
    }

    void delete_constraint(baldesCtrPtr constraint) { delete_constraint(constraint->index()); }

    void delete_variable(const baldesVarPtr variable) {
        // Find the index of the given variable in the variables vector
        auto var_index = variable->index();
        // Delete the column from the sparse matrix
        sparse_matrix.delete_column(var_index);
        // Remove the variable from the variables list
        for (auto &constraint : constraints) {
            constraint->get_expression().remove_term(variables[var_index]->get_name());
        }

        variables.erase(variables.begin() + var_index);

        for (int i = var_index; i < variables.size(); ++i) {
            variables[i]->set_index(i);
            var_name_to_index[variables[i]->get_name()] = i;
        }
    }

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
    baldesVarPtr getVar(size_t index) {
        if (index >= variables.size()) { throw std::out_of_range("baldesVar index out of range"); }
        return variables[index];
    }

    std::vector<baldesCtrPtr> getSRCconstraints() {
        std::vector<baldesCtrPtr> SRCconstraints;
        for (const auto &constraint : constraints) {
            if (constraint->get_name().find("SRC") != std::string::npos) { SRCconstraints.push_back(constraint); }
        }
        return SRCconstraints;
    }

    std::vector<baldesCtrPtr> getRCCconstraints() {
        std::vector<baldesCtrPtr> RCCconstraints;
        for (const auto &constraint : constraints) {
            if (constraint->get_name().find("RCC") != std::string::npos) { RCCconstraints.push_back(constraint); }
        }
        return RCCconstraints;
    }

    void chgCoeff(int constraintIndex, const std::vector<double> &values) {
        if (constraintIndex < 0 || constraintIndex >= constraints.size()) {
            fmt::print("Invalid constraint index: {}\n", constraintIndex);
            throw std::out_of_range("Invalid constraint index");
        }

        // Get the constraint that is being modified
        baldesCtrPtr      constraint = constraints[constraintIndex];
        LinearExpression &expression = constraint->get_expression();

        // Iterate over the values and only update changed entries
        for (int i = 0; i < values.size(); ++i) {
            double new_value = values[i];

            // Update the sparse matrix only if the value has changed
            sparse_matrix.modify_or_delete(constraintIndex, i, new_value);

            const std::string &var_name = variables[i]->get_name();

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

        // Update the LinearExpression in the corresponding baldesCtr
        baldesCtrPtr      constraint = constraints[constraintIndex];
        LinearExpression &expression = constraint->get_expression();

        const std::string &var_name = variables[variableIndex]->get_name();

        // Update or remove terms in the LinearExpression
        if (value != 0.0) {
            expression.add_or_update_term(var_name, value); // Add or update the term (variable, coefficient)
        } else {
            expression.remove_term(var_name); // Remove the term if the value is 0
        }
    }

    void addVars(const double *lb, const double *ub, const double *obj, const VarType *vtypes, const std::string *names,
                 const MIPColumn *cols, size_t count);

    void addVars(const double *lb, const double *ub, const double *obj, const VarType *vtypes, const std::string *names,
                 size_t count);
    // Get all variables
    std::vector<baldesVarPtr> &getVars() { return variables; }
    // Get all constraints
    std::vector<baldesCtrPtr> &getbaldesCtrs() { return constraints; }

    // Method to get the b vector (RHS values of all constraints)
    std::vector<double> get_b_vector() const {
        std::vector<double> b;
        for (const auto &constraint : constraints) { b.push_back(constraint->get_rhs()); }
        return b;
    }

    void chgCoeff(baldesCtrPtr constraint, const std::vector<double> &new_coeffs) {
        // int current_index = get_current_index(constraint->get_unique_id());
        int current_index = constraint->index();
        // Change the coefficients for the constraint
        chgCoeff(current_index, new_coeffs);
    }

#ifdef GUROBI
    // Function to populate a Gurobi model from this MIPProblem instance
    std::unique_ptr<GRBModel> toGurobiModel(GRBEnv &env) {
        try {
            auto gurobiModel = std::make_unique<GRBModel>(env);

            // Pre-size containers
            ankerl::unordered_dense::map<std::string, GRBVar> gurobiVars;
            gurobiVars.reserve(variables.size());

            // Step 1: Add variables efficiently
            for (const auto &var : variables) {
                gurobiVars.emplace(var->get_name(),
                                   gurobiModel->addVar(var->get_lb(), var->get_ub(), var->get_objective_coefficient(),
                                                       toGRBVarType(var->get_type()), var->get_name()));
            }

            // Step 2: Add constraints
            for (const auto &constraint : constraints) {
                GRBLinExpr expr = 0.0;

                // Build the expression efficiently
                for (const auto &[varName, coeff] : constraint->get_terms()) {
                    try {
                        expr += gurobiVars.at(varName) * coeff;
                    } catch (const std::out_of_range &e) { throw std::runtime_error("Variable not found: " + varName); }
                }

                const double rhs = constraint->get_rhs();
                switch (constraint->get_relation()) {
                case '<': gurobiModel->addConstr(expr <= rhs, constraint->get_name()); break;
                case '>': gurobiModel->addConstr(expr >= rhs, constraint->get_name()); break;
                case '=': gurobiModel->addConstr(expr == rhs, constraint->get_name()); break;
                }
            }

            // Step 3: Set objective
            GRBLinExpr objective = 0.0;
            for (const auto &var : variables) {
                objective += gurobiVars.at(var->get_name()) * var->get_objective_coefficient();
            }

            gurobiModel->setObjective(objective, GRB_MINIMIZE);
            gurobiModel->update();
            return gurobiModel;

        } catch (const GRBException &e) { throw; } catch (const std::exception &e) {
            throw;
        }
    }

    // Helper function to convert MIP variable type to Gurobi variable type
    constexpr char toGRBVarType(VarType varType) {
        switch (varType) {
        case VarType::Continuous: return GRB_CONTINUOUS;
        case VarType::Integer: return GRB_INTEGER;
        case VarType::Binary: return GRB_BINARY;
        default: throw std::invalid_argument("Invalid VarType: " + std::to_string(static_cast<int>(varType)));
        }
    }

    // Helper function to convert MIP constraints into a Gurobi linear expression
    GRBLinExpr convertToGurobiExpr(const baldesCtrPtr                                      &constraint,
                                   const ankerl::unordered_dense::map<std::string, GRBVar> &gurobiVars) {
        // Pre-size the expression if possible (if Gurobi provides such API)
        GRBLinExpr expr;

        // Reserve space for terms if the constraint provides a size hint
        const auto &terms = constraint->get_terms();

        try {
            // Use structured bindings for cleaner code
            for (const auto &[varName, coeff] : terms) {
                // Use .at() for bounds checking
                expr.addTerms(&coeff, &gurobiVars.at(varName), 1);
                // This is more efficient than expr += as it avoids creating temporary objects
            }
        } catch (const std::out_of_range &e) {
            throw std::runtime_error("Variable not found in Gurobi model: " + std::string(e.what()));
        }

        return expr; // Return value optimization will handle this efficiently
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
            highsModel.lp_.col_lower_[var_index] = var->get_lb();
            highsModel.lp_.col_upper_[var_index] = var->get_ub();
            highsModel.lp_.col_cost_[var_index]  = var->get_objective_coefficient();
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

    void update() { sparse_matrix.buildRowStart(); }

    double getSlack(int row, const std::vector<double> &solution) {
        // Ensure the sparse matrix row structure is built (we stay in COO mode)
        auto rhs = constraints[row]->get_rhs(); // Get the right-hand side value for this row

        // Compute the dot product of the solution vector and the specified row
        double row_value = 0.0;

        // Iterate through the non-zero elements in COO format
        for (size_t i = 0; i < sparse_matrix.rows.size(); ++i) {
            if (sparse_matrix.rows[i] == row) {
                int    col_index = sparse_matrix.cols[i];   // Get the column index for the current element
                double value     = sparse_matrix.values[i]; // Get the value of the current element

                // Perform the dot product with the corresponding element in the solution vector
                row_value += value * solution[col_index];
            }
        }

        // Return the slack: rhs - (dot product of the row and solution)
        return rhs - row_value;
    }

    double getRC(int col, const std::vector<double> &dual_solution) {
        // Get the objective coefficient for this variable
        double obj_coeff = variables[col]->get_objective_coefficient();

        // Calculate the sum of (dual_value * coefficient) for all constraints
        double dual_sum = 0.0;

        // Iterate through the non-zero elements in COO format
        for (size_t i = 0; i < sparse_matrix.rows.size(); ++i) {
            if (sparse_matrix.cols[i] == col) {
                int    row_index = sparse_matrix.rows[i];   // Get the row index
                double value     = sparse_matrix.values[i]; // Get the matrix coefficient

                // Multiply the dual variable value by the coefficient and add to sum
                dual_sum += dual_solution[row_index] * value;
            }
        }

        // For minimization problems:
        // Reduced Cost = objective_coefficient - sum(dual_values * coefficients)
        return obj_coeff - dual_sum;
    }

    std::vector<double> getAllReducedCosts(const std::vector<double> &dual_solution) {
        // Initialize reduced costs with objective coefficients
        std::vector<double> reduced_costs(variables.size());
        for (size_t i = 0; i < variables.size(); ++i) { reduced_costs[i] = variables[i]->get_objective_coefficient(); }

        // Subtract the dual contributions in one pass through the sparse matrix
        for (size_t i = 0; i < sparse_matrix.rows.size(); ++i) {
            int    col   = sparse_matrix.cols[i];
            int    row   = sparse_matrix.rows[i];
            double value = sparse_matrix.values[i];

            // Subtract dual_value * coefficient from the corresponding reduced cost
            reduced_costs[col] -= dual_solution[row] * value;
        }

        return reduced_costs;
    }

    ReducedCostResult getMostViolatingReducedCost(const std::vector<double> &dual_solution,
                                                  bool                       is_minimization = true) {
        std::vector<double> reduced_costs = getAllReducedCosts(dual_solution);

        ReducedCostResult result{0.0, -1};

        // For initialization
        if (!reduced_costs.empty()) {
            result.value        = reduced_costs[0];
            result.column_index = 0;
        }

        // Find the most violating reduced cost based on optimization direction
        for (size_t i = 1; i < reduced_costs.size(); ++i) {
            if (is_minimization) {
                // For minimization, we want the most negative reduced cost
                if (reduced_costs[i] < result.value) {
                    result.value        = reduced_costs[i];
                    result.column_index = i;
                }
            } else {
                // For maximization, we want the most positive reduced cost
                if (reduced_costs[i] > result.value) {
                    result.value        = reduced_costs[i];
                    result.column_index = i;
                }
            }
        }

        std::vector<int> col;
        col.assign(b_vec.size(), 0);
        // iterate over sparse matrix at column_index col and populate the col vector
        for (size_t i = 0; i < sparse_matrix.rows.size(); ++i) {
            if (sparse_matrix.cols[i] == result.column_index) { col[sparse_matrix.rows[i]] = sparse_matrix.values[i]; }
        }
        result.col = col;

        return result;
    }

    std::vector<double> get_c() const {
        std::vector<double> c;
        c.reserve(variables.size()); // Reserve space upfront
        std::transform(variables.begin(), variables.end(), std::back_inserter(c),
                       [](const baldesVarPtr var) { return var->get_objective_coefficient(); });
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
            if (var->get_type() == VarType::Continuous) {
                vtypes.push_back('C');
            } else if (var->get_type() == VarType::Integer) {
                vtypes.push_back('I');
            } else if (var->get_type() == VarType::Binary) {
                vtypes.push_back('B');
            }
        }
        return vtypes;
    }

    std::vector<double> get_lb() const {
        std::vector<double> lb;
        lb.reserve(variables.size()); // Reserve space upfront
        std::transform(variables.begin(), variables.end(), std::back_inserter(lb),
                       [](const baldesVarPtr var) { return var->get_lb(); });
        return lb;
    }

    std::vector<double> get_ub() const {
        std::vector<double> ub;
        ub.reserve(variables.size()); // Reserve space upfront
        std::transform(variables.begin(), variables.end(), std::back_inserter(ub),
                       [](const baldesVarPtr var) { return var->get_ub(); });
        return ub;
    }

    ModelData extractModelDataSparse() {
        // sparse_matrix.buildRowStart(); // Build the row start structure for CRS format
        ModelData data;
        data.A_sparse = sparse_matrix;
        data.b        = b_vec;
        data.c        = get_c();
        data.lb       = get_lb();
        data.ub       = get_ub();
        data.vtype    = get_vtypes();
        data.sense    = get_senses();

        // print sparse matrix sparsity
        // fmt::print("Sparsity of the sparse matrix: {:.2f}%\n", sparse_matrix.sparsity() * 100);
        return data;
    }

private:
    std::string                                    name;
    std::vector<baldesVarPtr>                      variables;
    std::vector<baldesCtrPtr>                      constraints;    // Store the constraints
    LinearExpression                               objective;      // Store the objective function
    ObjectiveType                                  objective_type; // Minimize or Maximize
    SparseMatrix                                   sparse_matrix;  // Use SparseMatrix for coefficient storage
    ankerl::unordered_dense::map<std::string, int> var_name_to_index;
    std::vector<double>                            b_vec;
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
