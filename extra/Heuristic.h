/**
 * @file Heuristic.h
 * @brief Defines classes and methods for solving vehicle routing problems using various heuristics.
 *
 * This file provides a set of classes for modeling and solving vehicle routing problems (VRP) and similar
 * optimization problems using heuristic techniques. The main components include:
 *
 * - `Customer`: Represents a customer with specific attributes, such as location, demand, and time windows.
 * - `HProblem`: Encapsulates the details of a heuristic vehicle routing problem, including customers and vehicles.
 * - `Route`: Represents a sequence of customers to be visited on a route.
 * - `DummyHeuristic`: A simple heuristic for generating a solution to the problem.
 * - `LocalSearch`: Provides local search optimization for an existing solution.
 * - `IteratedLocalSearch`: Implements the iterated local search algorithm to explore the solution space further.
 * - `SolomonFormatParser`: Parses problem files in Solomon format and constructs the corresponding problem instance.
 * - `SavingsHeuristic`: Implements the classical savings heuristic algorithm for solving vehicle routing problems.
 *
 * These classes allow users to solve VRP instances, parse problem definitions from files, and apply different
 * heuristic algorithms to generate and improve solutions.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
// Forward declaration of HProblem and Route classes
class HProblem;
class Route;

/**
 * @class Customer
 * @brief Represents a customer with specific attributes and methods.
 *
 * This class encapsulates the properties and behaviors of a customer, including
 * their location, demand, time windows, and service status.
 *
 */
class Customer {
public:
    int    number;
    double x, y;
    int    demand, ready_time, due_date, service_time;
    bool   is_serviced = false;

    Customer(int number, double x, double y, int demand, int ready_time, int due_date, int service_time)
        : number(number), x(x), y(y), demand(demand), ready_time(ready_time), due_date(due_date),
          service_time(service_time) {}

    /**
     * @brief Calculates the Euclidean distance to a target customer.
     *
     * This function computes the Euclidean distance between the current customer
     * and a target customer using their x and y coordinates. The distance is scaled
     * by a factor of 10 and then converted to a double.
     *
     */
    double distance(const Customer &target) const {
        auto aux      = (int)(10 * std::sqrt(std::pow(x - target.x, 2) + std::pow(y - target.y, 2)));
        auto distance = 1.0 * aux;

        return distance;
    }

    /**
     * @brief Overloaded stream insertion operator for the Customer class.
     *
     * This friend function allows a Customer object to be output to an
     * output stream (such as std::cout) in a specific format. The format
     * used is "C_" followed by the customer's number.
     *
     */
    friend std::ostream &operator<<(std::ostream &os, const Customer &customer) {
        return os << "C_" << customer.number;
    }
};

/**
 * @class HProblem
 * @brief Represents a heuristic problem for vehicle routing.
 *
 * This class encapsulates the details of a heuristic problem, including the problem name,
 * a list of customers, the number of vehicles, vehicle capacity, and the depot.
 *
 */
class HProblem {
public:
    std::string           name;
    std::vector<Customer> customers;
    int                   vehicle_number;
    int                   vehicle_capacity;
    Customer              depot;

    HProblem(const std::string &name, std::vector<Customer> &customers, int vehicle_number, int vehicle_capacity)
        : name(name), customers(customers), vehicle_number(vehicle_number), vehicle_capacity(vehicle_capacity),
          depot(*std::find_if(customers.begin(), customers.end(), [](const Customer &c) { return c.number == 0; })) {
        depot.is_serviced = true;
    }

    double      obj_func(const std::vector<Route> &routes) const;
    std::string print_canonical(const std::vector<Route> &routes) const;
};

/**
 * @class Route
 * @brief Represents a route in the heuristic problem.
 *
 * The Route class encapsulates a sequence of customers to be visited, starting and ending at the depot.
 *
 */
class Route {
public:
    const HProblem       &problem;
    std::vector<Customer> customers;
    // create customer_ which is equal customers less first element

    std::vector<Customer> customers_() const {
        std::vector<Customer> result(customers.begin() + 1, customers.end() - 1);
        return result;
    }
    Route(const HProblem &problem, std::vector<Customer> customers) : problem(problem), customers(customers) {
        this->customers.insert(this->customers.begin(), problem.depot);
        this->customers.push_back(problem.depot);
    }

    Route(const Route &other) : problem(other.problem), customers(other.customers) {}

    /**
     * @brief Assignment operator for the Route class.
     *
     * This operator assigns the contents of one Route object to another.
     * It performs a deep copy of the customers vector from the other Route object.
     * Note that reference members cannot be reassigned, so the problem reference is skipped.
     *
     * @param other The Route object to be copied.
     * @return A reference to the assigned Route object.
     */
    Route &operator=(const Route &other) {
        if (this != &other) {
            // Reference members cannot be reassigned, so we skip problem
            customers = other.customers;
        }
        return *this;
    }

    double      total_distance() const;
    std::string canonical_view() const;
    bool        is_feasible() const;

    // define contains method
    /**
     * @brief Checks if a customer with the given number exists in the list of customers.
     *
     * This function searches through the list of customers to determine if there is a customer
     * whose number matches the provided integer `i`.
     *
     * @param i The customer number to search for.
     * @return 1 if a customer with the given number exists, otherwise 0.
     */
    int contains(int i) const {
        return std::find_if(customers.begin(), customers.end(), [i](const Customer &c) { return c.number == i; }) !=
                       customers.end()
                   ? 1
                   : 0;
    }
    /**
     * @brief Checks if a customer with a given number exists in the list of customers.
     *
     * This function searches through the list of customers to determine if there is a customer
     * with the specified number. It returns 1 if the customer is found, and 0 otherwise.
     *
     * @param i The number of the customer to search for.
     * @return int 1 if the customer is found, 0 otherwise.
     */
    int contains_customer(int i) const {
        return std::find_if(customers.begin(), customers.end(), [i](const Customer &c) { return c.number == i; }) !=
                       customers.end()
                   ? 1
                   : 0;
    }

    // define clients() that returns std::vector<int> of customer numbers
    /**
     * @brief Retrieves a list of client numbers.
     *
     * This function transforms the list of customers into a list of client numbers.
     * It iterates over each customer, extracts the client number, and appends it to the result vector.
     *
     * @return A vector containing the client numbers of all customers.
     */
    std::vector<int> clients() const {
        std::vector<int> result;
        std::transform(customers.begin(), customers.end(), std::back_inserter(result),
                       [](const Customer &c) { return c.number; });
        return result;
    }

    // Clear method to reset the route
    /**
     * @brief Clears the list of customers and resets it to contain only the depot.
     *
     * This function clears the current list of customers and then adds the depot
     * as the starting and ending point of the route.
     */
    void clear_customers() {
        customers.clear();
        customers.push_back(problem.depot); // Add depot back as the only customer
        customers.push_back(problem.depot); // End at the depot
    }
};

/**
 * @class DummyHeuristic
 * @brief A heuristic class for solving a problem.
 *
 * This class provides a heuristic approach to solve a given problem.
 */
class DummyHeuristic {
public:
    DummyHeuristic(HProblem &problem) : problem(problem) {}

    std::vector<Route> get_solution();

private:
    HProblem &problem;
};

/**
 * @class LocalSearch
 * @brief A class that performs local search optimization on a given problem.
 *
 * The LocalSearch class is designed to optimize a given solution for a problem
 * using local search techniques. It holds a reference to an HProblem instance
 * and provides a method to optimize a given solution.
 *
 */
class LocalSearch {
public:
    LocalSearch(const HProblem &problem) : problem(problem) {}
    const HProblem &problem;

    std::vector<Route> optimize(const std::vector<Route> &solution) const;

private:
};

/**
 * @class IteratedLocalSearch
 * @brief A class that implements the Iterated Local Search algorithm for solving optimization problems.
 *
 * This class extends the LocalSearch class and provides methods for performing iterated local search
 * on a given problem. It uses a heuristic to generate an initial solution and applies perturbations
 * to explore the solution space.
 *
 */
class IteratedLocalSearch : public LocalSearch {
public:
    IteratedLocalSearch(HProblem &problem, std::function<double(const std::vector<Route> &)> obj_func = nullptr)
        : LocalSearch(problem),
          obj_func(obj_func ? obj_func : std::bind(&HProblem::obj_func, &problem, std::placeholders::_1)),
          initial_solution(DummyHeuristic(problem).get_solution()) {}

    std::vector<Route> perturbation(const std::vector<Route> &routes);

    std::vector<Route> execute();

private:
    std::function<double(const std::vector<Route> &)> obj_func;
    std::vector<Route>                                initial_solution;
};

/**
 * @class SolomonFormatParser
 * @brief A parser for reading Solomon format problem files.
 *
 * This class provides functionality to parse problem files formatted in the Solomon format.
 * The parsed data includes problem name, vehicle information, and customer details.
 *
 */
class SolomonFormatParser {
public:
    /**
     * @brief Reads a problem instance from a file and constructs an HProblem object.
     *
     * This function reads the problem data from the specified file, which includes
     * the problem name, vehicle information, and customer data.
     *
     */
    HProblem get_problem(const std::string &filename) const {
        std::ifstream file(filename);
        if (!file.is_open()) { throw std::runtime_error("Could not open file " + filename); }

        std::string name;
        std::getline(file, name);
        file.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Skip empty line

        std::string vehicle_line;
        std::getline(file, vehicle_line); // Skip "VEHICLE"
        std::getline(file, vehicle_line); // Skip "NUMBER CAPACITY"

        int vehicle_number, vehicle_capacity;
        file >> vehicle_number >> vehicle_capacity;
        file.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Skip rest of the line
        file.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Skip empty line

        std::string customer_line;
        std::getline(file, customer_line);                              // Skip "CUSTOMER"
        std::getline(file, customer_line);                              // Skip header line
        file.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Skip empty line

        std::vector<Customer> customers;
        int                   number;
        double                x, y, demand, ready_time, due_date, service_time;

        while (file >> number >> x >> y >> demand >> ready_time >> due_date >> service_time) {
            customers.emplace_back(number, x, y, demand, 10 * ready_time, 10 * due_date, 10 * service_time);
        }

        return HProblem(name, customers, vehicle_number, vehicle_capacity);
    }
};

using Saving = std::pair<double, std::pair<int, int>>; // (savings value, (customer i, customer j))

/**
 * @class SavingsHeuristic
 * @brief Implements the Savings Heuristic algorithm for solving vehicle routing problems.
 *
 * The Savings Heuristic is a classical algorithm used to generate solutions for vehicle routing problems.
 * It works by initializing routes for each customer, computing savings for merging routes, and then merging
 * routes based on the computed savings.
 *
 */
class SavingsHeuristic {
public:
    SavingsHeuristic(const HProblem &problem) : problem(problem) {}

    /**
     * @brief Generates a solution by initializing routes, computing savings, and merging routes.
     *
     * This function performs the following steps:
     * 1. Initializes routes.
     * 2. Computes savings.
     * 3. Merges routes based on computed savings.
     * 4. Removes any empty routes from the solution.
     *
     */
    std::vector<Route> get_solution() {
        std::vector<Route> solution = initialize_routes();
        compute_savings();      // Step 2: Compute savings
        merge_routes(solution); // Step 4: Merge routes
        // Remove empty routes
        solution.erase(
            std::remove_if(solution.begin(), solution.end(), [](const Route &r) { return r.customers_().empty(); }),
            solution.end());

        return solution;
    }

private:
    const HProblem &problem;

    // Define the type for savings as a pair of (savings_value, (customer_i, customer_j))
    using Saving = std::pair<double, std::pair<int, int>>;
    std::vector<Saving> savings_list;

    /**
     * @brief Initializes routes for each customer in the problem, excluding the depot.
     *
     * This function creates a vector of Route objects, where each Route contains a single customer
     * from the problem's customer list, excluding the depot (customer with number 0).
     *
     */
    std::vector<Route> initialize_routes() {
        std::vector<Route> solution;
        for (const Customer &customer : problem.customers) {
            if (customer.number != 0) { // Exclude the depot (customer 0)
                solution.emplace_back(problem, std::vector<Customer>{customer});
            }
        }
        return solution;
    }

    /**
     * @brief Computes the savings for each pair of customers and stores them in a list.
     *
     * This function calculates the savings for each pair of customers (excluding the depot)
     * based on the distance between them and the depot. The savings are computed as:
     *
     * savings = distance(depot, customer_i) + distance(depot, customer_j) - distance(customer_i, customer_j)
     *
     * The computed savings are stored in a list along with the corresponding customer pair indices.
     * The list is then sorted in descending order of savings.
     */
    void compute_savings() {
        savings_list.clear();
        for (size_t i = 1; i < problem.customers.size(); ++i) { // Exclude depot
            for (size_t j = i + 1; j < problem.customers.size(); ++j) {
                double savings = problem.customers[0].distance(problem.customers[i]) +
                                 problem.customers[0].distance(problem.customers[j]) -
                                 problem.customers[i].distance(problem.customers[j]);
                savings_list.emplace_back(savings, std::make_pair(i, j));
            }
        }
        // Sort savings in descending order
        std::sort(savings_list.begin(), savings_list.end(), std::greater<>());
    }
    /**
     * @brief Merges routes in the given solution based on savings list.
     *
     * This function iterates through a list of savings and attempts to merge routes
     * in the solution if the merge is feasible. It tracks merged customers to avoid
     * redundant operations.
     *
     */
    void merge_routes(std::vector<Route> &solution) {
        std::vector<bool> merged(problem.customers.size(), false); // Track merged customers
        for (const auto &[savings, pair] : savings_list) {
            int i = pair.first;
            int j = pair.second;

            Route *route_i = nullptr;
            Route *route_j = nullptr;

            // Find routes containing customers i and j
            for (auto &route : solution) {
                if (!route_i && route.contains_customer(problem.customers[i].number)) { route_i = &route; }
                if (!route_j && route.contains_customer(problem.customers[j].number)) { route_j = &route; }
                if (route_i && route_j) break;
            }

            // Merge routes if feasible
            if (route_i && route_j && route_i != route_j) {
                // Check if the customers are valid
                auto route_i_customers = route_i->customers_();
                auto route_j_customers = route_j->customers_();

                if (route_i_customers.empty() || route_j_customers.empty()) {
                    continue; // Skip merging if any of the routes is invalid
                }

                // Merge customer lists
                std::vector<Customer> merged_customers = route_i_customers;
                merged_customers.insert(merged_customers.end(), route_j_customers.begin(), route_j_customers.end());

                // Create the new merged route
                Route merged_route(problem, merged_customers);
                if (merged_route.is_feasible()) {
                    *route_i = merged_route;
                    route_j->clear_customers();
                    merged[i] = merged[j] = true;
                }
            }
        }
    }
};
