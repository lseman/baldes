/*
 * @file Variable.h
 * @brief Variable class implementation
 *
 * This file contains the implementation of the Variable class.
 * The Variable class represents a decision variable in a mathematical optimization problem.
 * It encapsulates the name, type, bounds, and objective coefficient of a decision variable.
 *
 */
#pragma once
#include <string>

enum class VarType { Continuous, Integer, Binary };

/**
 * @class Variable
 * @brief Represents a decision variable in a mathematical optimization problem.
 *
 * The Variable class encapsulates the name, type, bounds, and objective coefficient of a decision variable.
 */
class Variable {
public:
    Variable(const std::string &name, VarType type, double lb = 0.0, double ub = 1.0, double obj_coeff = 0.0)
        : name(name), type(type), lb(lb), ub(ub), objective_coefficient(obj_coeff) {}

    std::string get_name() const { return name; }
    VarType     get_type() const { return type; }
    double      get_lb() const { return lb; }
    double      get_ub() const { return ub; }

    void set_bounds(double lower, double upper) {
        lb = lower;
        ub = upper;
    }

    void set_type(VarType var_type) { type = var_type; }

    void setLB(double lower) { lb = lower; }
    void setUB(double upper) { ub = upper; }
    void setOBJ(double obj) { objective_coefficient = obj; }

    // Overload operator * to create a (var, coeff) pair for LinearExpression
    std::pair<Variable *, double> operator*(double coeff) { return {this, coeff}; }
    std::pair<Variable *, double> operator*(int coeff) { return {this, static_cast<double>(coeff)}; }

    double get_objective_coefficient() const { return objective_coefficient; }
    int    index() const { return index_val; }
    void   set_index(int index) { index_val = index; }

private:
    std::string name;
    VarType     type;
    double      lb, ub;
    double      objective_coefficient = 0.0;
    int         index_val             = -1;
};
