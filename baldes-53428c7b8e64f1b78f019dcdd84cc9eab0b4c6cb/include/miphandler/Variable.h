/*
 * @file Variable.h
 * @brief baldesVar class implementation
 *
 * This file contains the implementation of the baldesVar class.
 * The baldesVar class represents a decision variable in a mathematical optimization problem.
 * It encapsulates the name, type, bounds, and objective coefficient of a decision variable.
 *
 */
#pragma once
#include <string>
#include <memory>

enum class VarType { Continuous, Integer, Binary };

/**
 * @class baldesVar
 * @brief Represents a decision variable in a mathematical optimization problem.
 *
 * The baldesVar class encapsulates the name, type, bounds, and objective coefficient of a decision variable.
 */
class baldesVar : public std::enable_shared_from_this<baldesVar> {
public:
    baldesVar(const std::string &name, VarType type, double lb = 0.0, double ub = 1.0, double obj_coeff = 0.0)
        : name(name), type(type), lb(lb), ub(ub), objective_coefficient(obj_coeff) {}
    using baldesVarPtr = std::shared_ptr<baldesVar>;

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
    std::pair<baldesVarPtr, double> operator*(double coeff) { return {shared_from_this(), coeff}; }

    std::pair<baldesVarPtr, double> operator*(int coeff) { return {shared_from_this(), static_cast<double>(coeff)}; }

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
