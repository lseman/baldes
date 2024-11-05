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

#include "bnb/BNB.h"

#include "Definitions.h"
#include "HGS.h"
#include "Reader.h"
#include "bnb/BCP.h"
#include "bnb/Node.h"
#include "miphandler/MIPHandler.h"

#include "Logger.h"

#include "POPMUSIC.h"

#ifdef GUROBI
#include "gurobi_c++.h"
#include "solvers/Gurobi.h"
#endif

/**
 * Initializes the Restricted Master Problem (RMP) for the Vehicle Routing Problem with Time Windows (VRPTW).
 *
 * @param model Pointer to the Gurobi model.
 * @param problem Pointer to the VRProblem instance containing problem data.
 * @param heuristicRoutes Vector of heuristic-generated routes.
 * @return ModelData structure containing the initialized model data.
 *
 * This function performs the following steps:
 * 1. Sets Gurobi model parameters for output flag and optimization method.
 * 2. Prints the number of heuristic routes.
 * 3. Defines a lambda function to compute the cost of a route based on distance.
 * 4. Creates decision variables for each heuristic-generated route.
 * 5. Constructs the constraint matrix in sparse format.
 * 6. Adds constraints to ensure each customer is visited at least once.
 * 7. Adds a constraint to limit the number of vehicles used.
 * 8. Sets the objective to minimize the total cost.
 * 9. Solves the model.
 * 10. Stores and returns the model data including constraint matrix, bounds, variable types, and names.
 */
void initRMP(MIPProblem *model, VRProblem *problem, std::vector<std::vector<int>> &heuristicRoutes) {

    auto instance = problem->instance;
    int  Um       = problem->instance.nV; // Max number of vehicles

    std::vector<MIPColumn>   lambdaCols;
    std::vector<double>      lb, ub, obj;
    std::vector<std::string> names;
    std::vector<VarType>     vtypes;

    // Lambda function to compute the cost of a route
    auto costCalc = [&](const std::vector<int> &route) {
        double distance = 0.0;
        for (size_t i = 0; i < route.size() - 1; ++i) {
            const auto &source = route[i];
            const auto &target = route[i + 1];
            distance += instance.getcij(source, target);
        }
        distance += instance.getcij(route.back(), 0); // Return to the depot
        return distance;
    };

    // Create a decision variable for each heuristic-generated route
    int id = 0;
    for (const auto &column : heuristicRoutes) {
        int         routeID = id;
        std::string name    = "x[" + std::to_string(routeID) + "]";
        double      cost    = costCalc(column);

        // Store the column and variable data
        lb.push_back(0.0);
        ub.push_back(1.0);
        obj.push_back(cost);
        names.push_back(name);
        vtypes.push_back(VarType::Continuous); // Assuming VarType::Continuous is used

        id++;
    }

    // Add variables to the model
    model->addVars(lb.data(), ub.data(), obj.data(), vtypes.data(), names.data(), lb.size());
    // First set of constraints: Each vertex should be visited at least once
    for (int i = 1; i < instance.getNbVertices(); ++i) {
        LinearExpression lhs;
        for (size_t j = 0; j < heuristicRoutes.size(); ++j) {
            if (std::find(heuristicRoutes[j].begin(), heuristicRoutes[j].end(), i) != heuristicRoutes[j].end()) {
                lhs.addTerm(model->getVar(j), 1.0);
            }
        }
        model->add_constraint(lhs, 1.0, '>'); // baldesCtr: lhs >= 1 (visit the node)
    }

    // Second part: Ensure the number of vehicles does not exceed the maximum
    LinearExpression vehicle_constraint_lhs;
    for (size_t j = 0; j < heuristicRoutes.size(); ++j) { vehicle_constraint_lhs.addTerm(model->getVar(j), 1.0); }
    model->add_constraint(vehicle_constraint_lhs, static_cast<double>(Um), '<'); // baldesCtr: sum(lambda) <= Um

    // Set the objective to minimize
    model->setObjectiveSense(ObjectiveType::Minimize);
}

/**
 * @brief Prints the distance matrix to the standard output.
 *
 * This function takes a 2D vector representing a distance matrix and prints
 * it in a formatted manner. Each element is printed with a width of 10 characters.
 *
 */
void printDistanceMatrix(const std::vector<std::vector<double>> &distance) {
    print_info("Printing distance matrix");
    for (int i = 0; i < distance.size(); i++) {
        for (int j = 0; j < distance[i].size(); j++) { std::cout << std::setw(10) << distance[i][j] << " "; }
        std::cout << std::endl;
    }
}

/**
 * @file vrptw.cpp
 * @brief This file contains the main function for solving the Vehicle Routing Problem with Time Windows
 * (VRPTW).
 *
 * The main function performs the following steps:
 * 1. Reads the instance name from the command line arguments.
 * 2. Reads the VRPTW instance data from the specified file.
 * 3. Initializes heuristic solvers for generating initial solutions.
 * 4. Combines solutions from different heuristics.
 * 5. Converts the instance data into a format suitable for optimization.
 * 6. Initializes the Gurobi environment and model.
 * 7. Converts initial routes into labels and paths.
 * 8. Initializes the Restricted Master Problem (RMP) matrix.
 * 9. Solves the Column Generation (CG) problem using the Gurobi model.
 *
 * @param argc The number of command line arguments.
 * @param argv The array of command line arguments.
 * @return int Returns 0 on successful execution.
 */

using HGSptr = std::shared_ptr<HGS>;
int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <problem_kind> <instance_name>" << std::endl;
        return 1;
    }

    printBaldes();

    fmt::print("\n");
    fmt::print("\033[34m_PREPARING THE ROOM \033[0m");
    fmt::print("\n");

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

    Logger::init("logfile.txt");
    Logger::log("Starting the BCP algorithm for the {} problem \n", instance_name);

    print_heur("Initializing heuristic solver for initial solution\n");

    HGSptr hgs = std::make_shared<HGS>();

    /*
    HGS myhgs;
    int alpha = 50, delta = 40, timeLimit = 3600;  // Example parameters
    fmt::print("Running POPMUSIC algorithm\n");
    POPMUSIC popmusic(instance, myhgs, alpha, delta, timeLimit);
    Solution optimizedSolution = popmusic.run();
    fmt::print("Optimized solution cost: {}\n", optimizedSolution.totalCost);
    */

    auto initialRoutesHGS = hgs->run(instance);

    auto topRoutes = hgs->getBestRoutes();

    std::vector<VRPNode> nodes;
    nodes.clear();
    for (int k = 0; k < instance.nN; ++k) {
        int    start_time = instance.window_open[k];
        int    end_time   = instance.window_close[k];
        double duration   = instance.service_time[k];
        double demand     = instance.demand[k];
        double cost       = 0; // Assuming cost can be derived or set as needed
        nodes.emplace_back(k, start_time, end_time, duration, cost, demand);
        nodes[k].set_location(instance.x_coord[k], instance.y_coord[k]);
        if (instance.problem_type == ProblemType::vrptw) {
            nodes[k].lb.push_back(start_time);
            nodes[k].ub.push_back(end_time);
            nodes[k].consumption.push_back(duration);
        }
        // print start time and end time
        if (instance.problem_type == ProblemType::cvrp) {
            nodes[k].lb.push_back(0);
            nodes[k].ub.push_back(instance.q);
            nodes[k].consumption.push_back(demand);
        }
    }

    MIPProblem mip = MIPProblem(problem_kind, 0, 0);

    VRProblem *problem = new VRProblem();

    problem->instance = instance;
    problem->nodes    = nodes;

    std::vector<Path>    paths;
    std::vector<Label *> labels;

    // convert initial routes to labels
    int  labelID        = 0;
    int  labels_counter = 0;
    auto process_route  = [&](const std::vector<int> &route) {
        auto label           = new Label();
        label->nodes_covered = route;
        // calculate total distance
        for (int i = 0; i < route.size() - 1; i++) { label->cost += instance.getcij(route[i], route[i + 1]); }
        // label->cost         = route.total_distance();
        labelID++;
        labels.push_back(label);
        labels_counter++;

        Path path;
        path.route = route;
        // change last element of the route
        path.route[path.route.size() - 1] = N_SIZE - 1;
        path.cost                         = label->cost;
        paths.push_back(path);
    };
    std::for_each(initialRoutesHGS.begin(), initialRoutesHGS.end(), process_route);

    // print size of initialRoutesHGS
    initRMP(&mip, problem, initialRoutesHGS);
#ifdef GUROBI
    // auto gurobi_model = mip.toGurobiModel(GurobiEnvSingleton::getInstance());
#endif

    if (problem_kind == "vrptw") {
        problem->problemType = ProblemType::vrptw;
    } else if (problem_kind == "cvrp") {
        problem->problemType = ProblemType::cvrp;
    }

    BNBNode *node    = new BNBNode(mip);
    node->paths      = paths;
    node->problem    = problem;
    node->mip        = mip;
    node->instance   = instance;
    node->bestRoutes = topRoutes;

    BranchAndBound solver(std::move(problem), BNBNodeSelectionStrategy::DFS); // Choose
    solver.setRootNode(node);
    solver.solve();

    // problem->CG(&model);

    return 0;
}
