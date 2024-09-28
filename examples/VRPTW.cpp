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

// #include "../external/lkh/include/lkh_tsp.hpp"

#include "VRPTW.h"

#include <string>
#include <vector>

#include "../extra/Heuristic.h"
#include "../include/Definitions.h"
#include "../include/HGS.h"
#include "../include/Reader.h"

using VRProblemPtr = std::shared_ptr<VRProblem>;

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
void initRMP(GRBModel *model, VRProblemPtr problem, std::vector<std::vector<int>> &heuristicRoutes) {
    // GRBModel model = node->getModel();
    model->set(GRB_IntParam_OutputFlag, 0);

    // WARNING: DO NOT REMOVE THIS LINE
    model->set(GRB_IntParam_Method, 2);

    auto instance = problem->instance;

    int Um = problem->instance.nV;

    std::vector<GRBVar> lambda;

    // lambda function to compute the cost of a route
    auto costCalc = [&](const std::vector<int> &route) {
        double distance = 0.0;
        for (size_t i = 0; i < route.size() - 1; ++i) {
            const auto &source = route[i];
            const auto &target = route[i + 1];
            distance += instance.getcij(source, target);
        }
        distance += instance.getcij(route.back(), 0);
        return distance;
    };

    // Create a decision variable for each heuristic-generated route
    int id = 0;
    for (const auto &column : heuristicRoutes) {
        int         routeID = id;
        std::string name    = "x[" + std::to_string(routeID) + "]";
        double      cost    = costCalc(column);
        lambda.push_back(model->addVar(0.0, 1.0, cost, GRB_CONTINUOUS, name));
        id++;
    }
    model->update();

    GRBLinExpr lhs;

    // First part: Collect entries for each column based on heuristic routes
    for (int i = 1; i < instance.getNbVertices(); ++i) {
        lhs = 0;
        for (size_t j = 0; j < heuristicRoutes.size(); ++j) {
            // if (heuristicRoutes[j].contains(i)) { lhs += lambda[j]; }
            if (std::find(heuristicRoutes[j].begin(), heuristicRoutes[j].end(), i) != heuristicRoutes[j].end()) {
                { lhs += lambda[j]; }
            }
        }
        model->addConstr(lhs >= 1, "visit(m" + std::to_string(i - 1) + ")");
    }

    // Second part: Collect additional entries for the current column
    lhs = 0;
    std::vector<double> c;
    for (size_t j = 0; j < heuristicRoutes.size(); ++j) {
        int routeID = j;
        lhs += lambda[routeID];
    }

    model->addConstr(lhs <= Um, "limit(l1)");

    model->update();
    model->set(GRB_IntAttr_ModelSense, GRB_MINIMIZE);

    model->optimize();
    return;
}

/**
 * @brief Prints the distance matrix to the standard output.
 *
 * This function takes a 2D vector representing a distance matrix and prints
 * it in a formatted manner. Each element is printed with a width of 10 characters.
 *
 * @param distance A 2D vector of doubles representing the distance matrix.
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
int main(int argc, char *argv[]) {

    printBaldes();

    // get instance name as the first arg
    std::string instance_name = argv[1];

    print_heur("Initializing heuristic solver for initial solution\n");

    InstanceData instance;
    if (VRPTW_read_instance(instance_name, instance)) {
        print_info("Instance read successfully.\n");
    } else {
        std::cerr << "Error reading instance\n";
    }

    HGS  hgs;
    auto initialRoutesHGS = hgs.run(instance);

    /*
    SolomonFormatParser parser;
    HProblem            hproblem = parser.get_problem(instance_name);

    IteratedLocalSearch ils(hproblem);
    auto                initialRoutes = ils.execute();

    SavingsHeuristic heuristic(hproblem);
    auto             initialRoutesSavings = heuristic.get_solution();

    print_heur("Routes from saving heuristics: {}\n", initialRoutesSavings.size());

    // merge initialRoutes and initialRoutesSaving amd put in initialRoutes
    initialRoutes.insert(initialRoutes.end(), initialRoutesSavings.begin(), initialRoutesSavings.end());
    */

    std::vector<VRPJob> jobs;
    jobs.clear();
    for (int k = 0; k < instance.nN; ++k) {
        int    start_time = static_cast<int>(instance.window_open[k]);
        int    end_time   = static_cast<int>(instance.window_close[k]);
        double duration   = instance.service_time[k];
        double demand     = instance.demand[k];
        double cost       = 0; // Assuming cost can be derived or set as needed
        jobs.emplace_back(k, start_time, end_time, duration, cost, demand);
        jobs[k].set_location(instance.x_coord[k], instance.y_coord[k]);
        jobs[k].lb.push_back(start_time);
        jobs[k].ub.push_back(end_time);
        // jobs[k].lb.push_back(0);
        // jobs[k].ub.push_back(instance.q);
        jobs[k].consumption.push_back(duration);
        jobs[k].consumption.push_back(demand);
    }
    GRBEnv env = GRBEnv();
    env.start();
    GRBModel model = GRBModel(env);

    VRProblemPtr problem = std::make_shared<VRProblem>();
    problem->instance    = instance;
    problem->jobs        = jobs;

    std::vector<Path>    paths;
    std::vector<Label *> labels;

    // convert initial routes to labels
    int  labelID        = 0;
    int  labels_counter = 0;
    auto process_route  = [&](const std::vector<int> &route) {
        auto label          = new Label();
        label->jobs_covered = route;
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

    problem->allPaths       = paths;
    problem->labels_counter = labels_counter;

    initRMP(&model, problem, initialRoutesHGS);

    problem->CG(&model);

    return 0;
}
