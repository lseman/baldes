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
using baldesVarPtr = std::shared_ptr<baldesVar>;

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

        // Add a single variable with an implicit coefficient of 1.0
    LinearExpression& operator+=(const baldesVarPtr& var) {
        addTerm(var, 1.0);
        return *this;
    }

    // Subtract a single variable with an implicit coefficient of -1.0
    LinearExpression& operator-=(const baldesVarPtr& var) {
        addTerm(var, -1.0);
        return *this;
    }

    // Add a term with a double coefficient
    LinearExpression& operator+=(const std::pair<baldesVarPtr, double>& term) {
        addTerm(term.first, term.second);
        return *this;
    }

    // Subtract a term with a double coefficient
    LinearExpression& operator-=(const std::pair<baldesVarPtr, double>& term) {
        addTerm(term.first, -term.second);
        return *this;
    }

    // Template to handle numeric types (int, double, etc.)
    template <typename T, typename = std::enable_if_t<std::is_arithmetic<T>::value>>
    LinearExpression& operator+=(const std::pair<baldesVarPtr, T>& term) {
        addTerm(term.first, static_cast<double>(term.second));
        return *this;
    }

    // Add multiple terms to the expression (vector of pairs)
    LinearExpression& operator+=(const std::vector<std::pair<baldesVarPtr, double>>& termsVec) {
        for (const auto& term : termsVec) {
            addTerm(term.first, term.second);
        }
        return *this;
    }

    // Subtract multiple terms from the expression (vector of pairs)
    LinearExpression& operator-=(const std::vector<std::pair<baldesVarPtr, double>>& termsVec) {
        for (const auto& term : termsVec) {
            addTerm(term.first, -term.second);
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

    void multiply_by_constant(double constant) {
        for (auto &[var_name, coeff] : terms) { coeff *= constant; }
    }
    void multiply_by_constant(int constant) {
        for (auto &[var_name, coeff] : terms) { coeff *= constant; }
    }

    void add_or_update_term(const std::string &var_name, double coeff) { terms[var_name] = coeff; }

    baldesCtrPtr operator>=(double rhs) const { return std::make_shared<baldesCtr>(*this, rhs, '>'); }

    baldesCtrPtr operator<=(double rhs) const { return std::make_shared<baldesCtr>(*this, rhs, '<'); }

    baldesCtrPtr operator==(double rhs) const { return std::make_shared<baldesCtr>(*this, rhs, '='); }

private:
    ankerl::unordered_dense::map<std::string, double> terms; // Map of variable name to coefficient
};
