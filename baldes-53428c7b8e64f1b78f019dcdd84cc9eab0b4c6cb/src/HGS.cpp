/**
 * @file VRPTW.cpp
 * @brief Main implementation for solving the Vehicle Routing Problem with Time Windows (VRPTW).
 *
 * This file contains the implementation of the main function and supporting functions for solving the
 * Vehicle Routing Problem with Time Windows (VRPTW). The VRPTW class is used to manage instance data,
 * initialize the Restricted Master Problem (RMP), perform heuristic-based route generation, and apply column
 * generation to iteratively solve the problem.
 *
 * The following steps are carried out:
 * 1. Read the VRPTW instance and parse the problem data.
 * 2. Generate initial solutions using heuristic methods.
 * 3. Convert initial routes into the column generation format.
 * 4. Initialize the Gurobi model and the Restricted Master Problem (RMP).
 * 5. Apply column generation with stabilization to solve the problem iteratively.
 *
 * The implementation relies on the Gurobi optimizer for solving linear programming relaxations,
 * and uses various heuristics such as Iterated Local Search and Savings Heuristic to generate initial solutions.
 *
 * @param argc The number of command line arguments.
 * @param argv The array of command line arguments.
 * @return int Returns 0 on successful execution.
 */

// #include "../third_party/lkh/include/lkh_tsp.hpp"

#include <string>
#include <vector>

#include "Definitions.h"
#include "HGS.h"
#include "Reader.h"

using HGSptr = std::shared_ptr<HGS>;
int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <problem_kind> <instance_name>" << std::endl;
        return 1;
    }

    std::string problem_kind  = argv[1];
    std::string instance_name = argv[2];

    // Initialize the instance data and print the problem kind
    InstanceData instance;
    if (problem_kind == "vrptw") {
        if (!VRPTW_read_instance(instance_name, instance)) {
            std::cerr << "Error reading VRPTW instance\n";
            return 1;
        }
        print_info("VRPTW instance read successfully.\n");
    } else if (problem_kind == "cvrp") {
        if (!CVRP_read_instance(instance_name, instance)) {
            std::cerr << "Error reading CVRP instance\n";
            return 1;
        }
        print_info("CVRP instance read successfully.\n");
    } else {
        std::cerr << "Unsupported problem kind: " << problem_kind << "\n";
        return 1;
    } // py::scoped_interpreter guard{};

    printBaldes();

    print_heur("Initializing heuristic solver for initial solution\n");

    HGSptr hgs              = std::make_shared<HGS>();
    auto   initialRoutesHGS = hgs->run(instance);
    return 0;
}
