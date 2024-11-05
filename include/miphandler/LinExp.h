/*
 * @file LinExp.h
 * @brief Linear expression class implementation
 *
 * This file contains the implementation of the LinearExpression class.
 * The LinearExpression class represents a linear expression in a mathematical optimization problem.
 * It encapsulates a collection of terms, each consisting of a variable and a coefficient.
 * The class provides methods to add terms to the expression and create constraints using the expression.
 *
 */
#pragma once
#include "Variable.h"
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ankerl/unordered_dense.h"

class baldesCtr;
using baldesCtrPtr = std::shared_ptr<baldesCtr>;
using baldesVarPtr   = std::shared_ptr<baldesVar>;

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
    LinearExpression &operator+=(const std::pair<baldesVarPtr, double> &term) {
        if (term.second != 0.0) {                         // Skip zero coefficients
            terms[term.first->get_name()] += term.second; // Accumulate coefficients for the same variable
        }
        return *this;
    }

    // Add a term to the expression
    LinearExpression &operator+=(const std::pair<baldesVarPtr, int> &term) {
        if (term.second != 0.0) {                         // Skip zero coefficients
            terms[term.first->get_name()] += term.second; // Accumulate coefficients for the same variable
        }
        return *this;
    }

    // Add a baldesVar with an implicit coefficient of 1.0
    LinearExpression &operator+=(const baldesVarPtr var) {
        terms[var->get_name()] += 1.0; // Add the variable with coefficient 1.0
        return *this;
    }

    template <typename T>
    LinearExpression &operator+=(const std::pair<baldesVarPtr, T> &term) {
        if (term.second != 0.0) { // Skip zero coefficients
            terms[term.first->get_name()] += static_cast<double>(term.second);
        }
        return *this;
    }

    const ankerl::unordered_dense::map<std::string, double> &get_terms() const { return terms; }

    void addTerm(const baldesVarPtr var, double coeff) {
        if (coeff == 0.0) return; // Skip zero coefficients
        auto &existing_coeff = terms[var->get_name()];
        existing_coeff += coeff;
        if (existing_coeff == 0.0) {
            terms.erase(var->get_name()); // Remove terms that become zero
        }
    }

    void clear_terms() { terms.clear(); }
    void add_term(const std::string &var_name, double coeff) {
        terms[var_name] += coeff;
        if (terms[var_name] == 0.0) { terms.erase(var_name); }
    }
    void remove_term(const std::string &var_name) { terms.erase(var_name); }

    // Print the expression (for debugging)
    void print_expression() const {
        for (const auto &[var_name, coeff] : terms) { std::cout << coeff << "*" << var_name << " "; }
        std::cout << std::endl;
    }

    void add_or_update_term(const std::string &var_name, double coeff) { terms[var_name] = coeff; }

   baldesCtrPtr operator>=(double rhs) const {
        return std::make_shared<baldesCtr>(*this, rhs, '>');
    }

    baldesCtrPtr operator<=(double rhs) const {
        return std::make_shared<baldesCtr>(*this, rhs, '<');
    }

    baldesCtrPtr operator==(double rhs) const {
        return std::make_shared<baldesCtr>(*this, rhs, '=');
    }

private:
    ankerl::unordered_dense::map<std::string, double> terms; // Map of variable name to coefficient
};
