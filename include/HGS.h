/**
 * @file HGS.h
 * @brief Declares the Hybrid Genetic Search (HGS) solver class.
 *
 */
#pragma once

#include <time.h>

#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <vector>

#include "../third_party/hgs_vrptw/Genetic.h"
#include "../third_party/hgs_vrptw/Individual.h"
#include "../third_party/hgs_vrptw/LocalSearch.h"
#include "../third_party/hgs_vrptw/Params.h"
#include "../third_party/hgs_vrptw/Population.h"
#include "../third_party/hgs_vrptw/Split.h"
#include "Definitions.h"
#include "Reader.h"

/**
 * @struct HGSArcStats
 * @brief Tracks arc usage frequency across HGS iterations for guided
 * variable reduction.
 */
struct HGSArcStats {
    int    usage_count     = 0; // How often arc (i,j) appears in HGS routes
    int    avoidance_count = 0; // How often arc (i,j) was skipped by HGS
    double total_instances = 0; // Total route-count seen (for averaging)
};

/**
 * @class HGS
 * @brief A class that implements a Hybrid Genetic Search (HGS) algorithm.
 *
 * The HGS class is responsible for running a genetic algorithm to solve a given
 * problem instance. It initializes the necessary data structures, builds an
 * initial population, and executes the genetic algorithm.
 */
class HGS {
public:
    // default constructor
    HGS() = default;

    std::vector<std::vector<int>> bestRoutes;
    double                        best_cost = std::numeric_limits<double>::infinity();

    // Arc frequency tracking across iterative calls
    std::map<long long, HGSArcStats> arc_stats;

    /**
     * @brief Compute a deterministic arc hash key from (i, j).
     */
    static long long arcKey(int i, int j) noexcept {
        return (static_cast<long long>(i) << 32) | static_cast<unsigned int>(j);
    }

    /**
     * @brief Record that arc (i,j) was used in a route.
     */
    void recordArcUsage(int i, int j) noexcept {
        auto key = arcKey(i, j);
        arc_stats[key].usage_count++;
        arc_stats[key].total_instances++;
    }

    /**
     * @brief Record that arc (i,j) was avoided (not in the route) in an
     * iteration.
     */
    void recordArcAvoidance(int i, int j) noexcept {
        auto key = arcKey(i, j);
        arc_stats[key].avoidance_count++;
        arc_stats[key].total_instances++;
    }

    /**
     * @brief Check if an edge should be fixed (avoided) based on consistent
     * avoidance.
     * @param edgeCount  total iterations seen for this arc
     * @param avoidCount how many times it was avoided
     * @param threshold  fraction threshold (e.g. 0.8 = 80% avoidance)
     */
    static bool shouldFixArc(int edgeCount, int avoidCount, double threshold = 0.8) noexcept {
        return edgeCount >= 3 && (static_cast<double>(avoidCount) / edgeCount) >= threshold;
    }

    /**
     * @brief Run HGS once with optional dual bias. This is meant to be called
     * iteratively from within the CG loop.
     */
#ifdef ITERATIVE_HGS
    std::vector<std::vector<int>> runIterative(const InstanceData &instance, const std::vector<double> &duals = {},
                                               const std::vector<std::pair<int, int>> &avoidEdges = {},
                                               int iterations = 500, double timeLimitSec = 2.0) {
        Params params(instance);

        // If duals provided, bias the split algorithm to prefer arcs connecting
        // high-dual nodes.
        if (!duals.empty()) { params.duals_ptr = &duals; }

        // Build forbidden-edge set for this iteration (O(1) lookup)
        if (!avoidEdges.empty()) {
            params.forbidden_edges_ptr = &avoidEdges;
            static std::set<long long> forbidden_set;
            forbidden_set.clear();
            for (const auto &edge : avoidEdges) {
                forbidden_set.insert((static_cast<long long>(edge.first) << 32) |
                                     static_cast<unsigned int>(edge.second));
            }
            params.forbidden_edges_set = &forbidden_set;
        }

        Split          split(&params);
        HGSLocalSearch localSearch(&params);

        Population population(&params, &split, &localSearch);

        Genetic solver(&params, &split, &population, &localSearch);
        solver.run(iterations, timeLimitSec);

        auto sol   = population.extractFeasibleRoutes();
        bestRoutes = population.extractBestFeasibleRoutes();
        best_cost  = population.getBestFeasible()->myCostSol.penalizedCost;
        return sol;
    }
#else
    std::vector<std::vector<int>> runIterative(const InstanceData &instance, const std::vector<double> &duals = {},
                                               const std::vector<std::pair<int, int>> &avoidEdges = {},
                                               int iterations = 500, double timeLimitSec = 2.0) {
        // Stubs when ITERATIVE_HGS not enabled
        return {};
    }
#endif

    std::vector<std::vector<int>> run(const InstanceData &instance) {
        // InstanceData instance;

        // Reading the data file and initializing some data structures
        Params params(instance);
        auto   config = params.config;

        // Creating the Split and Local Search structures
        Split          split(&params);
        HGSLocalSearch localSearch(&params);

        // Initial population
        print_heur("Creating initial population\n");
        Population population(&params, &split, &localSearch);

        // Genetic algorithm
        print_heur("Running genetic algorithm\n");
        Genetic solver(&params, &split, &population, &localSearch);
        solver.run(config.nbIter, config.timeLimit);
        // std::cout << "----- GENETIC ALGORITHM FINISHED, TIME SPENT: " <<
        // params.getTimeElapsedSeconds() << std::endl;
        print_heur("Genetic algorithm finished in {:.2f} seconds\n", params.getTimeElapsedSeconds());

        auto sol = population.extractFeasibleRoutes();
        // auto sol = population.extractTopBestFeasibleRoutes(50);
        bestRoutes = population.extractBestFeasibleRoutes();
        // Return 0 if the program execution was successfull
        //
        // get best cost
        best_cost = population.getBestFeasible()->myCostSol.penalizedCost;
        return sol;
    }

    std::vector<std::vector<int>> getBestRoutes() { return bestRoutes; }
    double                        getBestCost() { return best_cost; }
};
