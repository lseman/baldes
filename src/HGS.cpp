/**
 * @file HGS.cpp
 * @brief Main entry point for the Hybrid Genetic Search (HGS) solver.
 *
 */

// #include "../third_party/lkh/include/lkh_tsp.hpp"

#include <string>
#include <vector>

#include "core/Definitions.h"#include "core/HGS.h"
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
