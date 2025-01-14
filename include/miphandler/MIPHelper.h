#pragma once
#include <memory>

#include "Definitions.h"
#include "SparseMatrix.h"  // Include your SparseMatrix class

#ifdef GUROBI
#include "gurobi_c++.h"
#include "solvers/Gurobi.h"

#endif

#include <fmt/core.h>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "Constraint.h"
#include "LinExp.h"
#include "Variable.h"

using baldesCtrPtr = std::shared_ptr<baldesCtr>;
using baldesVarPtr = std::shared_ptr<baldesVar>;
using LinearExpPtr = std::shared_ptr<LinearExpression>;
class MIPColumn {
   public:
    // Add a term to the column (row index and coefficient)
    void addTerm(int row_index, double value) {
        if (value != 0.0) {
            terms.push_back({row_index, value});
        }
    }

    // Clear the column for reuse
    void clear() { terms.clear(); }
    std::vector<std::pair<int, double>> getTerms() const { return terms; }

    std::pair<int, double> get_term(size_t index) const {
        if (index >= terms.size()) {
            throw std::out_of_range("Index out of range");
        }
        return terms[index];
    }

    // create size method
    size_t size() const { return terms.size(); }

   private:
    std::vector<std::pair<int, double>> terms;  // Pairs of row index and value
};

#ifdef GUROBI
class GurobiModelCache {
   private:
    enum class ChangeType {
        AddVariable,
        DeleteVariable,
        AddConstraint,
        DeleteConstraint,
        ModifyCoefficient,
    };

    struct ModelChange {
        ChangeType type;
        int row{-1};
        int col{-1};
        double value{0.0};
        baldesVarPtr var;
        baldesCtrPtr ctr;
        MIPColumn column;
    };

    std::vector<ModelChange> pendingChanges;
    ankerl::unordered_dense::map<std::string, GRBVar> gurobiVars;
    std::unique_ptr<GRBModel> model;
    bool needsRebuild{false};

    static char toGRBVarType(VarType varType) {
        switch (varType) {
            case VarType::Continuous:
                return GRB_CONTINUOUS;
            case VarType::Integer:
                return GRB_INTEGER;
            case VarType::Binary:
                return GRB_BINARY;
            default:
                throw std::invalid_argument(
                    "Invalid VarType: " +
                    std::to_string(static_cast<int>(varType)));
        }
    }

   public:
    explicit GurobiModelCache(GRBEnv &env) {
        model = std::make_unique<GRBModel>(env);
    }

    // Copy constructor
    GurobiModelCache(const GurobiModelCache &other)
        : pendingChanges(other.pendingChanges),
          gurobiVars(other.gurobiVars),
          model(new GRBModel(*other.model)),
          needsRebuild(other.needsRebuild) {}

    // Move constructor
    GurobiModelCache(GurobiModelCache &&other) noexcept = default;

    // Copy assignment
    GurobiModelCache &operator=(const GurobiModelCache &other) {
        if (this != &other) {
            pendingChanges = other.pendingChanges;
            gurobiVars = other.gurobiVars;
            model.reset(new GRBModel(*other.model));
            needsRebuild = other.needsRebuild;
        }
        return *this;
    }

    // Move assignment
    GurobiModelCache &operator=(GurobiModelCache &&other) noexcept = default;

    void initialize(const std::vector<baldesVarPtr> &variables,
                    const std::vector<baldesCtrPtr> &constraints,
                    bool isMinimize = true) {
        try {
            model->reset();
            gurobiVars.clear();
            gurobiVars.reserve(variables.size());

            // Add variables
            for (const auto &var : variables) {
                gurobiVars.emplace(
                    var->get_name(),
                    model->addVar(var->get_lb(), var->get_ub(),
                                  var->get_objective_coefficient(),
                                  toGRBVarType(var->get_type()),
                                  var->get_name()));
            }

            // Add constraints
            for (const auto &constraint : constraints) {
                GRBLinExpr expr = 0.0;
                for (const auto &[varName, coeff] : constraint->get_terms()) {
                    try {
                        expr += gurobiVars.at(varName) * coeff;
                    } catch (const std::out_of_range &e) {
                        throw std::runtime_error("Variable not found: " +
                                                 varName);
                    }
                }

                switch (constraint->get_relation()) {
                    case '<':
                        model->addConstr(expr <= constraint->get_rhs(),
                                         constraint->get_name());
                        break;
                    case '>':
                        model->addConstr(expr >= constraint->get_rhs(),
                                         constraint->get_name());
                        break;
                    case '=':
                        model->addConstr(expr == constraint->get_rhs(),
                                         constraint->get_name());
                        break;
                }
            }

            // Set objective
            GRBLinExpr objective = 0.0;
            for (const auto &var : variables) {
                objective += gurobiVars.at(var->get_name()) *
                             var->get_objective_coefficient();
            }
            model->setObjective(objective,
                                isMinimize ? GRB_MINIMIZE : GRB_MAXIMIZE);
            model->update();
            pendingChanges.clear();
            needsRebuild = false;

        } catch (const GRBException &e) {
            throw;
        } catch (const std::exception &e) {
            throw;
        }
    }

    void printCacheContents() const {
        fmt::print("GurobiModelCache Contents:\n");
        fmt::print("-------------------------\n");

        // Print pending changes
        fmt::print("Pending Changes ({} total):\n", pendingChanges.size());
        for (size_t i = 0; i < pendingChanges.size(); ++i) {
            const auto &change = pendingChanges[i];
            fmt::print("  {}. ", i + 1);

            switch (change.type) {
                case ChangeType::AddVariable:
                    fmt::print(
                        "Add Variable: {} (lb={}, ub={}, obj_coeff={})\n",
                        change.var->get_name(), change.var->get_lb(),
                        change.var->get_ub(),
                        change.var->get_objective_coefficient());
                    break;

                case ChangeType::DeleteVariable:
                    fmt::print("Delete Variable at index {}\n", change.col);
                    break;

                case ChangeType::AddConstraint:
                    fmt::print(
                        "Add Constraint: {} ({} {})\n", change.ctr->get_name(),
                        change.ctr->get_relation(), change.ctr->get_rhs());
                    change.ctr->get_expression().print_expression();
                    break;

                case ChangeType::DeleteConstraint:
                    fmt::print("Delete Constraint at index {}\n", change.row);
                    break;

                case ChangeType::ModifyCoefficient:
                    fmt::print(
                        "Modify Coefficient: row={}, col={}, new_value={}\n",
                        change.row, change.col, change.value);
                    break;
            }
        }

        // Print cached variables
        fmt::print("\nCached Variables ({} total):\n", gurobiVars.size());
        for (const auto &[name, var] : gurobiVars) {
            try {
                fmt::print(
                    "  {}: type={}, lb={}, ub={}, obj_coeff={}\n", name,
                    var.get(GRB_CharAttr_VType), var.get(GRB_DoubleAttr_LB),
                    var.get(GRB_DoubleAttr_UB), var.get(GRB_DoubleAttr_Obj));
            } catch (const GRBException &e) {
                fmt::print("  {}: <error reading variable properties>\n", name);
            }
        }

        // Print model stats
        try {
            fmt::print("\nModel Stats:\n");
            fmt::print("  NumVars: {}\n", model->get(GRB_IntAttr_NumVars));
            fmt::print("  NumConstrs: {}\n",
                       model->get(GRB_IntAttr_NumConstrs));
            fmt::print("  NumSOS: {}\n", model->get(GRB_IntAttr_NumSOS));
            fmt::print("  NumQConstrs: {}\n",
                       model->get(GRB_IntAttr_NumQConstrs));
            fmt::print("  NumGenConstrs: {}\n",
                       model->get(GRB_IntAttr_NumGenConstrs));
            fmt::print("  ModelSense: {}\n",
                       model->get(GRB_IntAttr_ModelSense) == GRB_MINIMIZE
                           ? "Minimize"
                           : "Maximize");
        } catch (const GRBException &e) {
            fmt::print("\nError reading model stats: {}\n", e.getMessage());
        }

        fmt::print("  NeedsRebuild: {}\n", needsRebuild ? "Yes" : "No");
        fmt::print("-------------------------\n");
    }

    void addVariable(const baldesVarPtr &var) {
        ModelChange change{.type = ChangeType::AddVariable, .var = var};
        pendingChanges.push_back(std::move(change));
    }

    void addColumn(const MIPColumn &column) {
        // get the latest addVariable change and modify column field
        pendingChanges.back().column = column;
    }

    void deleteVariable(int varIndex) {
        ModelChange change{.type = ChangeType::DeleteVariable, .col = varIndex};
        pendingChanges.push_back(std::move(change));
        needsRebuild = true;
    }

    void addConstraint(const baldesCtrPtr &ctr) {
        ModelChange change{.type = ChangeType::AddConstraint, .ctr = ctr};
        pendingChanges.push_back(std::move(change));
    }

    void deleteConstraint(int ctrIndex) {
        ModelChange change{.type = ChangeType::DeleteConstraint,
                           .row = ctrIndex};
        pendingChanges.push_back(std::move(change));
        needsRebuild = true;
    }

    void modifyCoefficient(int row, int col, double newValue) {
        ModelChange change{.type = ChangeType::ModifyCoefficient,
                           .row = row,
                           .col = col,
                           .value = newValue};
        pendingChanges.push_back(std::move(change));
    }

    void modifyConstraint(int row, const std::vector<double> &values) {
        for (size_t i = 0; i < values.size(); ++i) {
            modifyCoefficient(row, i, values[i]);
        }
    }

    GRBModel *getModel() {
        // print pendingChanges status
        if (!pendingChanges.empty()) {
            applyChanges();
        }
        return model.get();
    }

   private:
    void applyChanges() {
        try {
            for (const auto &change : pendingChanges) {
                switch (change.type) {
                    case ChangeType::AddVariable: {
                        // Get MIPColumn from the same ChangeType::AddColumn
                        // position
                        MIPColumn column = change.column;
                        GRBColumn gurobiColumn;
                        for (size_t i = 0; i < column.size(); ++i) {
                            // Get constraint with the index
                            auto term = column.get_term(i);
                            auto index = term.first;
                            auto value = term.second;
                            GRBConstr ctr = model->getConstr(index);
                            gurobiColumn.addTerm(value, ctr);
                        }
                        gurobiVars.emplace(
                            change.var->get_name(),  // Variable name
                            model->addVar(
                                change.var->get_lb(),  // Lower bound
                                change.var->get_ub(),  // Upper bound
                                change.var
                                    ->get_objective_coefficient(),  // Objective
                                                                    // coefficient
                                toGRBVarType(
                                    change.var
                                        ->get_type()),  // Variable type
                                                        // (binary, integer,
                                                        // continuous)
                                gurobiColumn,  // The column (constraint
                                               // coefficients)
                                change.var->get_name()  // Name of the variable
                                ));
                        break;
                    }

                    case ChangeType::AddConstraint: {
                        GRBLinExpr expr = 0.0;
                        for (const auto &[varName, coeff] :
                             change.ctr->get_terms()) {
                            expr += gurobiVars.at(varName) * coeff;
                        }

                        switch (change.ctr->get_relation()) {
                            case '<':
                                model->addConstr(expr <= change.ctr->get_rhs(),
                                                 change.ctr->get_name());
                                break;
                            case '>':
                                model->addConstr(expr >= change.ctr->get_rhs(),
                                                 change.ctr->get_name());
                                break;
                            case '=':
                                model->addConstr(expr == change.ctr->get_rhs(),
                                                 change.ctr->get_name());
                                break;
                        }
                        break;
                    }

                    case ChangeType::DeleteVariable:
                        model->remove(model->getVar(change.col));
                        break;

                    case ChangeType::DeleteConstraint:
                        model->remove(model->getConstr(change.row));
                        break;

                    case ChangeType::ModifyCoefficient:
                        model->chgCoeff(model->getConstr(change.row),
                                        model->getVar(change.col),
                                        change.value);
                        break;
                }
            }

            model->update();
            pendingChanges.clear();
        } catch (const GRBException &e) {
            throw;
        } catch (const std::exception &e) {
            throw;
        }
    }
};
#endif
