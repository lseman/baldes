#pragma once
#include "Variable.h"
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ankerl/unordered_dense.h"

class Constraint;

/**
 * @class LinearExpression
 * @brief Represents a linear expression in a mathematical optimization problem.
 *
 * The LinearExpression class encapsulates a collection of terms, each consisting of a variable and a coefficient.
 * It provides methods to add terms to the expression and create constraints using the expression.
 */
class LinearExpression {
public:
    LinearExpression() = default;

    // Add a term to the expression
    LinearExpression &operator+=(const std::pair<Variable, double> &term) {
        if (term.second != 0.0) {                        // Skip zero coefficients
            terms[term.first.get_name()] += term.second; // Accumulate coefficients for the same variable
        }
        return *this;
    }

    // Add a term to the expression
    LinearExpression &operator+=(const std::pair<Variable, int> &term) {
        if (term.second != 0.0) {                        // Skip zero coefficients
            terms[term.first.get_name()] += term.second; // Accumulate coefficients for the same variable
        }
        return *this;
    }

    // Add a Variable with an implicit coefficient of 1.0
    LinearExpression &operator+=(const Variable &var) {
        terms[var.get_name()] += 1.0; // Add the variable with coefficient 1.0
        return *this;
    }

    const ankerl::unordered_dense::map<std::string, double> &get_terms() const { return terms; }

    void addTerm(const Variable &var, double coeff) {
        auto var_name = var.get_name();
        terms[var_name] += coeff;
    }

    // Print the expression (for debugging)
    void print_expression() const {
        for (const auto &[var_name, coeff] : terms) { std::cout << coeff << "*" << var_name << " "; }
        std::cout << std::endl;
    }

    // Overload for <= operator
    Constraint operator<=(double rhs) const;

    // Overload for >= operator
    Constraint operator>=(double rhs) const;

    // Overload for == operator
    Constraint operator==(double rhs) const;

private:
    ankerl::unordered_dense::map<std::string, double> terms; // Map of variable name to coefficient
};
