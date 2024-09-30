/**
 * @file HGS.h
 * @brief This file contains the definition of the HGS class.
 */
#pragma once

#include <iostream>
#include <time.h>
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
 * @class HGS
 * @brief A class that implements a Hybrid Genetic Search (HGS) algorithm.
 *
 * The HGS class is responsible for running a genetic algorithm to solve a given problem instance.
 * It initializes the necessary data structures, builds an initial population, and executes the genetic algorithm.
 */
class HGS {
public:
    // default constructor
    HGS() = default;

    std::vector<std::vector<int>> run(const InstanceData &instance) {

        // InstanceData instance;

        // Reading the data file and initializing some data structures
        Params params(instance);
        auto   config = params.config;

        // Creating the Split and Local Search structures
        Split          split(&params);
        HGSLocalSearch localSearch(&params);

        // Initial population
        print_info("Creating initial population\n");
        Population population(&params, &split, &localSearch);

        /*
        // Genetic algorithm
        print_info("Running genetic algorithm\n");
        Genetic solver(&params, &split, &population, &localSearch);
        solver.run(config.nbIter, config.timeLimit);
        // std::cout << "----- GENETIC ALGORITHM FINISHED, TIME SPENT: " << params.getTimeElapsedSeconds() << std::endl;
        print_info("Genetic algorithm finished in {:.2f} seconds\n", params.getTimeElapsedSeconds());
        */
        auto sol = population.extractFeasibleRoutes();

        // Return 0 if the program execution was successfull
        return sol;
    }
};
