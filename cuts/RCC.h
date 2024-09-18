/**
 * @file RCC.h
 * @brief Definitions for generating and separating Rounded Capacity Cuts (RCC) in the context of vehicle routing problems.
 *
 * This header file contains the structure and function definitions required for separating Rounded Capacity Cuts (RCCs) 
 * using Gurobi's optimization model. RCC separation is an important aspect of optimization algorithms, particularly 
 * in vehicle routing problems where capacity constraints must be enforced.
 *
 * Key components of the file include:
 * - `separate_Rounded_Capacity_cuts`: A function that identifies and separates RCCs by solving the relaxed Gurobi model 
 *   and searching for violated constraints in the solution space. The function generates multiple RCC solutions.
 *
 * The file leverages Gurobi for optimization and constraint management.
 *
 * @note Several parts of the function rely on setting up and solving a Gurobi optimization model to identify capacity 
 * violations and generate RCCs.
 */

#pragma once

#include "gurobi_c++.h"
#include "gurobi_c.h"

#include "../include/Hashes.h"

#include <cmath>
#include <iostream>
#include <set>
#include <unordered_map>
#include <vector>

using namespace std;

// Isolated RC cut generation process that returns multiple solutions
/**
 * @brief Separates Rounded Capacity Cuts (RCC) for a given Gurobi model.
 *
 * This function identifies and separates RCCs from a given Gurobi model. It uses the solution from the LP or relaxed problem
 * and searches for multiple solutions that violate the RCC constraints. The function returns multiple sets of nodes corresponding
 * to the solutions found.
 *
 * @param gurobi_model Pointer to the Gurobi model.
 * @param Q Integer parameter representing the capacity.
 * @param demand Vector of integers representing the demand.
 * @param opt_obj Double representing the optimal objective value.
 * @param verbose Boolean flag to enable verbose output.
 * @param allPaths Vector of Path objects representing all paths.
 * @return std::vector<std::set<int>> A vector of sets, where each set contains nodes corresponding to a solution.
 */
std::vector<set<int>> separate_Rounded_Capacity_cuts(GRBModel *gurobi_model, int Q, const std::vector<int> &demand,
                                                     double opt_obj, bool verbose, const std::vector<Path> &allPaths) {

    double epsilon_1 = 0.5;
    double epsilon_2 = 1e-2;
    double epsilon_3 = 1e-3;

    int                        iteration_counter = 0;
    unordered_map<int, GRBVar> delta;
    bool                       need_to_run_RCC_separation = true;

    // Solution from the LP or relaxed problem
    std::vector<double> sol;
    int                 varNumber = gurobi_model->get(GRB_IntAttr_NumVars);
    for (int i = 0; i < varNumber; i++) { sol.push_back(gurobi_model->getVar(i).get(GRB_DoubleAttr_X)); }

    vector<double> x_values     = sol;
    GRBModel       m_separation = GRBModel(gurobi_model->getEnv());
    m_separation.set(GRB_IntParam_OutputFlag, 0);

    // Enable Gurobi to search for multiple solutions
    m_separation.set(GRB_IntParam_PoolSearchMode, 2);

    // Set the number of solutions to be found
    m_separation.set(GRB_IntParam_PoolSolutions, 5);

    GRBVar alpha = m_separation.addVar(0, GRB_INFINITY, 0, GRB_INTEGER);

    set<pair<int, int>>                             relevant_edges;
    unordered_map<pair<int, int>, double, pair_hash> edge_capacities;

    unordered_map<pair<int, int>, GRBVar, pair_hash> gamma;

    std::vector<std::vector<double>> aijs(N_SIZE + 2, std::vector<double>(N_SIZE + 2, 0.0));

    for (int counter = 0; counter < sol.size(); ++counter) {
        auto &nodes = allPaths[counter].route;
        for (int k = 1; k < nodes.size(); ++k) {

            int source = nodes[k - 1];
            int target = (nodes[k] == N_SIZE - 1) ? 0 : nodes[k];
            aijs[source][target] += sol[counter];
        }
    }

    for (int i = 1; i < N_SIZE - 1; ++i) {
        delta[i] = m_separation.addVar(0, 1, 0, GRB_BINARY);
        for (int j = 1; j < N_SIZE - 1; ++j) {
            double edge_capacity = aijs[i][j];
            if (edge_capacity >= epsilon_2) {
                relevant_edges.insert({i, j});
                edge_capacities[{i, j}] = edge_capacity;
                gamma[{i, j}]           = m_separation.addVar(0, GRB_INFINITY, 0, GRB_CONTINUOUS);
            }
        }
    }
    m_separation.update();

    // compute gcd of demands
    auto gcd = 1;
    for (int i = 1; i < N_SIZE - 1; ++i) { gcd = std::gcd(gcd, demand[i]); }
    GRBLinExpr lhs = GRBLinExpr(0.3 + alpha * Q);
    GRBLinExpr rhs = 0.0;
    for (int i = 1; i < N_SIZE - 1; ++i) { rhs += delta[i] * demand[i]; }
    m_separation.addConstr(lhs <= rhs);

    GRBLinExpr obj_separation = 2 * alpha + 2;

    for (int i = 1; i < N_SIZE - 1; ++i) {
        double edge_capacity = aijs[i][0] + aijs[0][i];
        if (edge_capacity >= epsilon_2) { obj_separation += -delta[i] * edge_capacity; }
    }

    for (const auto &[i, j] : relevant_edges) {
        m_separation.addConstr(gamma[{i, j}] <= delta[i]);
        m_separation.addConstr(gamma[{i, j}] <= delta[j]);
    }

    m_separation.addConstr(
        std::accumulate(delta.begin(), delta.end(), GRBLinExpr(0),
                        [](GRBLinExpr sum, const std::pair<const int, GRBVar> &p) { return sum + p.second; }) >= 2);

    m_separation.setObjective(obj_separation - accumulate(relevant_edges.begin(), relevant_edges.end(), GRBLinExpr(0),
                                                          [&](GRBLinExpr sum, const auto &edge) {
                                                              int i = edge.first, j = edge.second;
                                                              return sum + (delta[i] + delta[j] - 2 * gamma[{i, j}]) *
                                                                               edge_capacities[{i, j}];
                                                          }),
                              GRB_MAXIMIZE);

    m_separation.optimize();

    std::vector<std::set<int>> multiple_solutions; // Store multiple sets of nodes

    int    solCount          = m_separation.get(GRB_IntAttr_SolCount); // Number of solutions found
    double eps_for_violation = 1e-3;
    // Retrieve multiple solutions
    for (int solIndex = 0; solIndex < solCount; ++solIndex) {
        m_separation.set(GRB_IntParam_SolutionNumber, solIndex); // Set the solution number
        auto solution = m_separation.get(GRB_DoubleAttr_ObjVal); // Get the objective value for this solution
        if (m_separation.get(GRB_DoubleAttr_PoolObjVal) >= eps_for_violation) {
            fmt::print("RCC Violation {}; Objective value - {}\n", solIndex + 1,
                       m_separation.get(GRB_DoubleAttr_ObjVal));

            std::set<int> S; // Store nodes for this solution

            // Extract solution for this specific delta configuration
            for (int i = 1; i < N_SIZE - 1; ++i) {
                if (delta[i].get(GRB_DoubleAttr_Xn) > 0.5) { // Get solution value for this node in the current solution
                    S.insert(i);
                }
            }

            // Store the set of nodes for this solution
            multiple_solutions.push_back(S);
        }
    }

    return multiple_solutions; // Return all sets of nodes corresponding to the solutions
}
