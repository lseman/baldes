#pragma once

#include <iostream>
#include <time.h>
#include <vector>

#include "../external/hgs_vrptw/Genetic.h"
#include "../external/hgs_vrptw/Individual.h"
#include "../external/hgs_vrptw/LocalSearch.h"
#include "../external/hgs_vrptw/Params.h"
#include "../external/hgs_vrptw/Population.h"
#include "../external/hgs_vrptw/Split.h"

#include "Reader.h"

// Main class of the algorithm. Used to read from the parameters from the command line,
// create the structures and initial population, and run the hybrid genetic search
class HGS {
public:
    // default constructor
    HGS() = default;

    std::vector<std::vector<int>> run(const std::string &instance_name) {

        InstanceData instance;

        // Reading the data file and initializing some data structures
        Params params(instance_name);
        auto   config = params.config;

        // Creating the Split and Local Search structures
        Split          split(&params);
        HGSLocalSearch localSearch(&params);

        // Initial population
        std::cout << "----- INSTANCE LOADED WITH " << params.nbClients << " CLIENTS AND " << params.nbVehicles
                  << " VEHICLES" << std::endl;
        std::cout << "----- BUILDING INITIAL POPULATION" << std::endl;
        Population population(&params, &split, &localSearch);
        /*
                // Genetic algorithm
                std::cout << "----- STARTING GENETIC ALGORITHM" << std::endl;
                Genetic solver(&params, &split, &population, &localSearch);
                solver.run(config.nbIter, config.timeLimit);
                std::cout << "----- GENETIC ALGORITHM FINISHED, TIME SPENT: " << params.getTimeElapsedSeconds() <<
           std::endl;
                */
        auto sol = population.extractFeasibleRoutes();

        // Return 0 if the program execution was successfull
        return sol;
    }
};
