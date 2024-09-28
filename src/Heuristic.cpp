/**
 * @file Heuristics.cpp
 * @brief Implements heuristic algorithms for solving vehicle routing problems (VRP).
 *
 * This file provides the implementation of various heuristic techniques, including:
 * - `DummyHeuristic`: A basic heuristic for generating initial solutions.
 * - `LocalSearch`: A local search algorithm for optimizing routes.
 * - `IteratedLocalSearch`: An iterated local search algorithm that applies perturbations to improve solutions.
 * - Route manipulation methods such as `two_opt`, `cross`, `insertion`, and `swap` for improving route efficiency.
 * - Calculation of route feasibility, total distance, and canonical representations.
 *
 * These algorithms aim to generate and optimize solutions for VRPs, which involve assigning customers to vehicles
 * while minimizing travel distance and adhering to constraints such as time windows and vehicle capacities.
 */

#include "Definitions.h"

#include "extra/Heuristic.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

/**
 * @brief Generates a solution using a dummy heuristic.
 *
 * This function generates a solution for the problem by iteratively selecting
 * available customers and forming feasible routes until all customers are serviced.
 *
 */
std::vector<Route> DummyHeuristic::get_solution() {
    // std::cout << "Dummy heuristic\n";

    auto get_available_customers = [this]() -> std::vector<std::reference_wrapper<Customer>> {
        std::vector<std::reference_wrapper<Customer>> available_customers;
        for (Customer &customer : problem.customers) { // Make sure customer is not const
            if (!customer.is_serviced && customer.number != 0) { available_customers.push_back(std::ref(customer)); }
        }
        std::sort(available_customers.begin(), available_customers.end(),
                  [](Customer &a, Customer &b) { return a.due_date < b.due_date; });
        return available_customers;
    };

    std::vector<Route> solution;
    while (true) {
        auto customers = get_available_customers();
        if (customers.empty()) { break; }
        std::vector<Customer *> route_customers;
        for (auto &customer_ref : customers) {
            Customer               &customer             = customer_ref.get();
            std::vector<Customer *> temp_route_customers = route_customers;
            temp_route_customers.push_back(&customer);
            std::vector<Customer> temp_customers;
            for (auto *c : temp_route_customers) { temp_customers.push_back(*c); }
            if (Route(problem, temp_customers).is_feasible()) {
                customer.is_serviced = true;
                route_customers.push_back(&customer);
            }
        }
        std::vector<Customer> route_customers_copy;
        for (auto *c : route_customers) { route_customers_copy.push_back(*c); }
        solution.emplace_back(problem, route_customers_copy);
    }

    return solution;
}

/**
 * @brief Performs the 2-opt heuristic on a given route.
 *
 * This function takes a vector of Customer objects representing a route and
 * two indices, i and j. It reverses the segment of the route between these
 * two indices (inclusive) to potentially improve the route.
 *
 */
std::vector<Customer> two_opt(std::vector<Customer> a, int i, int j) {
    std::reverse(a.begin() + i, a.begin() + j + 1);
    return a;
}

/**
 * @brief Performs a crossover operation between two vectors of Customer objects.
 *
 * This function takes two vectors of Customer objects and two indices, and
 * creates two new vectors by combining parts of the input vectors. The first
 * new vector is formed by taking the first 'i' elements from vector 'a' and
 * appending the elements from vector 'b' starting from index 'j' to the end.
 * The second new vector is formed by taking the first 'j' elements from vector
 * 'b' and appending the elements from vector 'a' starting from index 'i' to the end.
 *
 */
std::pair<std::vector<Customer>, std::vector<Customer>> cross(const std::vector<Customer> &a,
                                                              const std::vector<Customer> &b, int i, int j) {
    std::vector<Customer> new_a(a.begin(), a.begin() + i);
    new_a.insert(new_a.end(), b.begin() + j, b.end());
    std::vector<Customer> new_b(b.begin(), b.begin() + j);
    new_b.insert(new_b.end(), a.begin() + i, a.end());
    return {new_a, new_b};
}

/**
 * @brief Inserts a customer from vector `a` into vector `b` at specified positions.
 *
 * This function creates copies of the input vectors `a` and `b`, then inserts the
 * customer at index `i` from vector `a` into vector `b` at index `j`. The customer
 * is then removed from the copied vector `a`. If the vector `a` is empty or the
 * index `i` is out of bounds, the function returns the original vectors.
 *
 */
std::pair<std::vector<Customer>, std::vector<Customer>> insertion(const std::vector<Customer> &a,
                                                                  const std::vector<Customer> &b, int i, int j) {
    std::vector<Customer> new_a(a);
    std::vector<Customer> new_b(b);
    if (!a.empty() && i < a.size()) {
        new_b.insert(new_b.begin() + j, a[i]);
        new_a.erase(new_a.begin() + i);
    }
    return {new_a, new_b};
}

/**
 * @brief Swaps elements between two vectors of Customers at specified indices.
 *
 * This function takes two vectors of Customers and swaps the elements at the
 * specified indices if the indices are within the bounds of their respective vectors.
 *
 */
std::pair<std::vector<Customer>, std::vector<Customer>> swap(std::vector<Customer> a, std::vector<Customer> b, int i,
                                                             int j) {
    if (i < a.size() && j < b.size()) { std::swap(a[i], b[j]); }
    return {a, b};
}

/**
 * @brief Optimizes a given solution using a local search heuristic.
 *
 * This function takes a vector of routes and attempts to optimize each route
 * using a 2-opt local search algorithm. The optimization process continues
 * until no further improvements can be made to the route.
 *
 */
std::vector<Route> LocalSearch::optimize(const std::vector<Route> &solution) const {
    std::vector<Route> new_solution = solution;
    for (auto &route : new_solution) {
        bool is_stucked = false;
        while (!is_stucked) {
            is_stucked = true;
            for (size_t k = 0; k < route.customers_().size() - 1; ++k) {
                for (size_t j = k + 1; j < route.customers_().size(); ++j) {
                    std::vector<Customer> new_customers = two_opt(route.customers_(), k, j);
                    Route                 new_route(problem, new_customers);
                    if (new_route.is_feasible() && new_route.total_distance() < route.total_distance()) {
                        route      = new_route;
                        is_stucked = false;
                    }
                }
            }
        }
    }
    return new_solution;
}

/**
 * @brief Applies perturbation to a set of routes using various heuristics.
 *
 * This function iteratively applies perturbation to the given routes to explore
 * different route configurations. It uses a combination of cross, insertion, and
 * swap operations to generate new routes and selects the best feasible routes
 * based on total distance.
 *
 */
std::vector<Route> IteratedLocalSearch::perturbation(const std::vector<Route> &routes) {
    std::vector<Route> best       = routes;
    bool               is_stucked = false;

    while (!is_stucked) {
        is_stucked = true;

        for (size_t i = 0; i < best.size() - 1; ++i) {
            for (size_t j = i + 1; j < best.size(); ++j) {
                for (size_t k = 0; k <= best[i].customers_().size(); ++k) {
                    for (size_t l = 0; l <= best[j].customers_().size(); ++l) {
                        std::vector<std::function<std::pair<std::vector<Customer>, std::vector<Customer>>(
                            const std::vector<Customer> &, const std::vector<Customer> &, int, int)>>
                            funcs = {cross, insertion, swap};

                        for (const auto &func : funcs) {
                            auto [c1, c2] = func(best[i].customers_(), best[j].customers_(), k, l);
                            Route r1(problem, c1);
                            Route r2(problem, c2);

                            if (r1.is_feasible() && r2.is_feasible() &&
                                (r1.total_distance() + r2.total_distance() <
                                 best[i].total_distance() + best[j].total_distance())) {
                                best[i]    = r1;
                                best[j]    = r2;
                                is_stucked = false;
                            }
                        }
                    }
                }
            }
        }

        best.erase(std::remove_if(best.begin(), best.end(), [](const Route &r) { return r.customers_().empty(); }),
                   best.end());
    }

    return best;
}

/**
 * @brief Executes the Iterated Local Search (ILS) algorithm to optimize a solution.
 *
 * This function starts by optimizing an initial solution using a local search method.
 * It then iteratively applies perturbations to the best solution found so far, followed
 * by further local optimization. If a perturbed and optimized solution is better than
 * the current best solution, it replaces the best solution. This process continues until
 * no better solution is found.
 *
 */
std::vector<Route> IteratedLocalSearch::execute() {
    std::vector<Route> best = optimize(initial_solution);
    print_heur("Local search solution\n");
    // std::cout << problem.print_canonical(best);
    print_heur("Total distance: {}\n", obj_func(best));

    bool is_stucked = false;
    while (!is_stucked) {
        is_stucked                      = true;
        std::vector<Route> new_solution = perturbation(best);
        // print_info("Perturbation step done\n");
        new_solution = optimize(new_solution);
        // print_info("Local search solution\n");
        if (obj_func(new_solution) < obj_func(best)) {
            is_stucked = false;
            best       = std::vector<Route>(new_solution.begin(), new_solution.end());
            print_heur("ILS step\n");
            // std::cout << problem.print_canonical(best);
            print_heur("Total distance: {}\n", obj_func(best));
            // std::cout << "Total distance: " << obj_func(best) << "\n";
        }
    }

    return best;
}

/**
 * @brief Calculates the total distance of the route.
 *
 * This function iterates through the list of customers in the route and sums up the distances
 * between consecutive customers to compute the total distance of the route.
 *
 */
double Route::total_distance() const {
    double distance = 0.0;
    for (size_t i = 0; i < customers.size() - 1; ++i) {
        const auto &source = customers[i];
        const auto &target = customers[i + 1];
        distance += source.distance(target);
    }
    return distance;
}

/**
 * @brief Generates a canonical view of the route.
 *
 * This function constructs a string representation of the route, including
 * the sequence of customer numbers and the cumulative distance traveled to
 * reach each customer. The format of the returned string is:
 * "0 0.0 <customer_number> <cumulative_distance> ...".
 *
 */
std::string Route::canonical_view() const {
    double             time = 0.0;
    std::ostringstream result;
    result << "0 0.0 ";
    double distance = 0.0;
    for (size_t i = 0; i < customers.size() - 1; ++i) {
        const auto &source     = customers[i];
        const auto &target     = customers[i + 1];
        double      start_time = std::max(static_cast<double>(target.ready_time), time + source.distance(target));
        time                   = start_time + target.service_time;
        distance += source.distance(target);
        // result << target.number << " " << start_time << " ";
        result << target.number << " " << distance << " ";
    }
    return result.str();
}

/**
 * @brief Checks if the route is feasible based on time and capacity constraints.
 *
 * This function iterates through the list of customers in the route and checks
 * if the route can be completed within the given time windows and vehicle capacity.
 * It calculates the start time for each customer based on the ready time and the
 * travel time from the previous customer. If the start time exceeds the due date
 * for any customer, the route is deemed infeasible. Additionally, it checks if
 * the total demand of the customers exceeds the vehicle capacity.
 *
 */
bool Route::is_feasible() const {
    double time        = 0.0;
    int    capacity    = problem.vehicle_capacity;
    bool   is_feasible = true;
    for (size_t i = 0; i < customers.size() - 1; ++i) {
        const auto &source     = customers[i];
        const auto &target     = customers[i + 1];
        double      start_time = std::max(static_cast<double>(target.ready_time), time + source.distance(target));
        if (start_time >= target.due_date) {
            is_feasible = false;
            return is_feasible;
        }
        time = start_time + target.service_time;
        capacity -= target.demand;
    }
    if (time >= problem.depot.due_date || capacity < 0) {
        is_feasible = false;
        return is_feasible;
    }
    return is_feasible;
}

/**
 * @brief Calculates the objective function value for a given set of routes.
 *
 * This function computes the total distance of all routes by summing up the
 * total distance of each individual route in the provided vector of routes.
 *
 */
double HProblem::obj_func(const std::vector<Route> &routes) const {
    return std::accumulate(routes.begin(), routes.end(), 0.0,
                           [](double sum, const Route &r) { return sum + r.total_distance(); });
}

/**
 * @brief Generates a canonical string representation of a collection of routes.
 *
 * This function iterates over a vector of Route objects and appends their
 * canonical views to an output string stream, each followed by a newline character.
 *
 */
std::string HProblem::print_canonical(const std::vector<Route> &routes) const {
    std::ostringstream oss;
    for (const auto &route : routes) { oss << route.canonical_view() << "\n"; }
    return oss.str();
}
