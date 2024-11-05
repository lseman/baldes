#pragma once

#include "HGS.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <set>
#include <vector>

// Assuming Solution and InstanceData structures are defined in HGS.h

struct Solution {
    std::vector<std::vector<int>> routes;
    double                        totalCost = 0.0;
};

class POPMUSIC {
public:
    POPMUSIC(const InstanceData &instance, HGS &hgs, int alpha, int delta, int timeLimit)
        : instance(instance), hgs(hgs), alpha(alpha), delta(delta), timeLimit(timeLimit) {}

    Solution run() {
        Solution globalSolution;
        hgs.run(instance); // Initial solution using HGS
        globalSolution.routes    = hgs.getBestRoutes();
        globalSolution.totalCost = calculateTotalCost(globalSolution.routes);
        fmt::print("Initial solution cost: {}\n", globalSolution.totalCost);

        std::set<std::vector<int>> exploredSubproblems; // Track solved subproblems
        fmt::print("Running POPMUSIC algorithm\n");
        auto startTime = std::chrono::steady_clock::now();
        int  dimsp     = alpha;

        while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - startTime).count() <
               timeLimit) {
            std::vector<int> customers = getCustomers(globalSolution);
            std::shuffle(customers.begin(), customers.end(), std::mt19937{std::random_device{}()});

            for (int seed : customers) {
                std::vector<int> Vsp = constructSubproblem(globalSolution, seed, dimsp);
                if (exploredSubproblems.count(Vsp) == 0) {
                    exploredSubproblems.insert(Vsp);

                    InstanceData                  subInstance     = createSubInstance(Vsp);
                    std::vector<std::vector<int>> subSolution     = hgs.run(subInstance);
                    double                        subSolutionCost = calculateTotalCost(subSolution);
                    if (subSolutionCost < calculateSubCost(globalSolution, Vsp)) {
                        updateSolution(globalSolution, Vsp, subSolution);
                    }
                }
            }
            dimsp += delta; // Increase subproblem size for next iteration
        }
        return globalSolution;
    }

private:
    const InstanceData &instance;
    HGS                &hgs;
    int                 alpha;
    int                 delta;
    int                 timeLimit;

    // Collect all customers from the current solution's routes
    std::vector<int> getCustomers(const Solution &solution) const {
        std::vector<int> customers;
        for (const auto &route : solution.routes) { customers.insert(customers.end(), route.begin(), route.end()); }
        return customers;
    }

    // Construct subproblem Vsp based on seed and dimsp constraint
    std::vector<int> constructSubproblem(const Solution &solution, int seed, int dimsp) const {
        std::vector<int> Vsp;
        for (const auto &route : solution.routes) {
            if (std::find(route.begin(), route.end(), seed) != route.end() && Vsp.size() + route.size() <= dimsp) {
                Vsp.insert(Vsp.end(), route.begin(), route.end());
            }
        }
        return Vsp;
    }

    // Calculate total cost for the entire solution
    double calculateTotalCost(const std::vector<std::vector<int>> &routes) const {
        double totalCost = 0.0;
        for (const auto &route : routes) { totalCost += calculateRouteCost(route); }
        return totalCost;
    }

    // Calculate the cost of the current subproblem Vsp in the global solution
    double calculateSubCost(const Solution &solution, const std::vector<int> &Vsp) const {
        double subCost = 0.0;
        for (const auto &route : solution.routes) {
            for (int customer : route) {
                if (std::find(Vsp.begin(), Vsp.end(), customer) != Vsp.end()) {
                    subCost += calculateRouteCost(route);
                    break;
                }
            }
        }
        return subCost;
    }

    // Update the global solution by replacing routes associated with Vsp with improved subSolution
    void updateSolution(Solution &solution, const std::vector<int> &Vsp,
                        const std::vector<std::vector<int>> &subSolution) const {
        auto it = std::remove_if(solution.routes.begin(), solution.routes.end(), [&Vsp](const std::vector<int> &route) {
            return std::any_of(route.begin(), route.end(), [&Vsp](int customer) {
                return std::find(Vsp.begin(), Vsp.end(), customer) != Vsp.end();
            });
        });
        solution.routes.erase(it, solution.routes.end());
        solution.routes.insert(solution.routes.end(), subSolution.begin(), subSolution.end());
        solution.totalCost = calculateTotalCost(solution.routes);
    }

    // Generate a new sub-instance for a given subproblem Vsp
    InstanceData createSubInstance(std::vector<int> &Vsp) const {
        InstanceData subInstance;
        // Include depot as the first node in subInstance
        // remove duplicated members in Vsp
        std::sort(Vsp.begin(), Vsp.end());  // Sort the vector
        Vsp.erase(std::unique(Vsp.begin(), Vsp.end()), Vsp.end());
        // print Vsp.size
        subInstance.nN = Vsp.size() + 1; // +1 for depot
        subInstance.nV = 1;
        subInstance.q  = instance.q;

        subInstance.x_coord.resize(subInstance.nN);
        subInstance.y_coord.resize(subInstance.nN);
        subInstance.demand.resize(subInstance.nN);
        subInstance.distance.resize(subInstance.nN, std::vector<double>(subInstance.nN));
        subInstance.n_tw.resize(subInstance.nN);
        subInstance.window_open.resize(subInstance.nN);
        subInstance.window_close.resize(subInstance.nN);
        subInstance.service_time.resize(subInstance.nN);

        // Set depot data
        subInstance.x_coord[0]      = instance.x_coord[0];
        subInstance.y_coord[0]      = instance.y_coord[0];
        subInstance.demand[0]       = instance.demand[0];
        subInstance.n_tw[0]         = instance.n_tw[0];
        subInstance.window_open[0]  = instance.window_open[0];
        subInstance.window_close[0] = instance.window_close[0];
        subInstance.service_time[0] = instance.service_time[0];

        // Fill in data for each node in Vsp, starting from index 1 in subInstance
        int idx = 1;
        for (int customer : Vsp) {
            fmt::print("Customer: {}\n", customer);
            subInstance.x_coord[idx]      = instance.x_coord[customer];
            subInstance.y_coord[idx]      = instance.y_coord[customer];
            subInstance.demand[idx]       = instance.demand[customer];
            subInstance.n_tw[idx]         = instance.n_tw[customer];
            subInstance.window_open[idx]  = instance.window_open[customer];
            subInstance.window_close[idx] = instance.window_close[customer];
            subInstance.service_time[idx] = instance.service_time[customer];
            ++idx;
        }

        // Construct the reduced distance matrix for subInstance, including depot
        for (int i = 0; i < subInstance.nN; ++i) {
            for (int j = 0; j < subInstance.nN; ++j) {
                if (i == j) {
                    subInstance.distance[i][j] = 0.0;
                } else {
                    double dx                  = subInstance.x_coord[i] - subInstance.x_coord[j];
                    double dy                  = subInstance.y_coord[i] - subInstance.y_coord[j];
                    auto   aux                 = (int)(10 * sqrt(dx * dx + dy * dy));
                    subInstance.distance[i][j] = 1.0 * aux;
                }
            }
        }

        return subInstance;
    }

    // Define route cost calculation function based on your cost structure
    double calculateRouteCost(const std::vector<int> &route) const {
        double cost = 0.0;
        for (size_t i = 0; i < route.size() - 1; ++i) { cost += instance.getcij(route[i], route[i + 1]); }
        return cost;
    }
};