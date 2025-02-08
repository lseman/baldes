/*
 * @file Constraint.h
 * @brief baldesCtr class implementation
 *
 * This file contains the implementation of the baldesCtr class.
 * The baldesCtr class represents a linear constraint in a mathematical optimization problem.
 * It encapsulates a linear expression, a right-hand side value, and a relational operator to define a constraint.
 *
 */

#pragma once
#include "LinExp.h"
#include "ankerl/unordered_dense.h"
#include <atomic>

/**
 * @class baldesCtr
 * @brief Represents a linear constraint in a mathematical optimization problem.
 *
 * The baldesCtr class encapsulates a linear expression, a right-hand side value,
 * and a relational operator to define a constraint in a mathematical optimization problem.
 */
class baldesCtr {
public:
    // Static method to generate unique IDs
    static size_t generate_unique_id() {
        static std::atomic<size_t> counter{0};
        return counter++;
    }

    // Modified constructors to initialize unique_id
    baldesCtr(const LinearExpression &expr, double rhs, char relation)
        : expression(expr)
        , rhs(rhs)
        , relation(relation)
        , unique_id(generate_unique_id())
        , indice(-1)  // Initialize index to invalid value
    {}

    // Default constructor
    baldesCtr() 
        : expression()
        , rhs(0.0)
        , relation('=')
        , unique_id(generate_unique_id())
        , indice(-1)  // Initialize index to invalid value
    {}

    // Copy constructor needs to generate a new unique_id
    baldesCtr(const baldesCtr& other)
        : expression(other.expression)
        , rhs(other.rhs)
        , relation(other.relation)
        , unique_id(other.unique_id)
        , indice(other.indice)
        , name(other.name)
    {}

    baldesCtr(const LinearExpression& expr, double rhs, char relation, size_t existing_id, int indice, const std::string& name = "")
        : expression(expr)
        , rhs(rhs)
        , relation(relation)
        , unique_id(existing_id)  // Use existing ID
        , indice(indice)
        , name(name)
    {}

    std::shared_ptr<baldesCtr> clone() const {
        // Use the private constructor to create a clone with same unique_id
        return std::shared_ptr<baldesCtr>(
            new baldesCtr(expression, rhs, relation, unique_id, indice, name)
        );
    }

    // Assignment operator needs to preserve the unique_id of the target
    baldesCtr& operator=(const baldesCtr& other) {
        if (this != &other) {
            expression = other.expression;
            rhs = other.rhs;
            relation = other.relation;
            indice = other.indice;
            name = other.name;
            // Do NOT copy unique_id
        }
        return *this;
    }

    // Getter for unique ID
    size_t get_unique_id() const { return unique_id; }

    // Existing methods
    LinearExpression &get_expression() { return expression; }
    double get_rhs() const { return rhs; }
    char get_relation() const { return relation; }
    int index() const { return indice; }
    void set_index(int i) { indice = i; }
    void set_name(const std::string &n) { name = n; }
    const ankerl::unordered_dense::map<std::string, double> &get_terms() const { 
        return expression.get_terms(); 
    }
    void addTerm(const baldesVarPtr var, double coeff) { 
        expression.addTerm(var, coeff); 
    }
    std::string get_name() const { return name; }

    void print() const {
        fmt::print("baldesCtr (ID: {}, Index: {}): ", unique_id, indice);
        for (const auto &[var_name, coeff] : expression.get_terms()) {
            fmt::print("{:.2f} * {} + ", coeff, var_name);
        }
        fmt::print(" {} {} {} \n", relation, rhs, name);
    }

private:
    LinearExpression expression;  // The linear expression of the constraint
    double rhs;                  // The right-hand side of the constraint
    char relation;               // The relation: '<', '>', '='
    const size_t unique_id;      // Unique identifier for the constraint
    int indice;                  // Position index in the constraint system
    std::string name;
};