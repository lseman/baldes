/*
 * @file Constraint.h
 * @brief Constraint class implementation
 *
 * This file contains the implementation of the Constraint class.
 * The Constraint class represents a linear constraint in a mathematical optimization problem.
 * It encapsulates a linear expression, a right-hand side value, and a relational operator to define a constraint.
 *
 */
#pragma once
#include "LinExp.h"
#include "ankerl/unordered_dense.h"

/**
 * @class Constraint
 * @brief Represents a linear constraint in a mathematical optimization problem.
 *
 * The Constraint class encapsulates a linear expression, a right-hand side value,
 * and a relational operator to define a constraint in a mathematical optimization problem.
 */
class Constraint {
public:
    Constraint(const LinearExpression &expr, double rhs, char relation)
        : expression(expr), rhs(rhs), relation(relation) {}

    // default constructor
    Constraint() : expression(), rhs(0.0), relation('=') {}

    LinearExpression &get_expression() { return expression; }
    double            get_rhs() const { return rhs; }
    char              get_relation() const { return relation; }

    int index() const { return indice; }

    void set_index(int i) { indice = i; }

    void set_name(const std::string &n) { name = n; }

    const ankerl::unordered_dense::map<std::string, double> &get_terms() const { return expression.get_terms(); }

    void addTerm(const Variable *var, double coeff) { expression.addTerm(var, coeff); }

    // define get_name
    std::string get_name() const { return name; }

    void print() const {
        fmt::print("Constraint: ");
        for (const auto &[var_name, coeff] : expression.get_terms()) {
            fmt::print("{:.2f} * {} + ", coeff, var_name);
        }
        fmt::print(" {} {} {}\n", relation, rhs, name);
    }

private:
    LinearExpression expression; // The linear expression of the constraint
    double           rhs;        // The right-hand side of the constraint
    char             relation;   // The relation: '<', '>', '='
    int              indice;
    std::string      name;
};