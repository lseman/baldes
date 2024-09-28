#include "presolve/Activities.h"
#include "presolve/Clique.h"
#include "Definitions.h"
#include "Reader.h"
#include "gurobi_c++.h"

GRBModel solve_vrp(const InstanceData &instance) {
    GRBEnv env = GRBEnv(true);
    env.set(GRB_IntParam_OutputFlag, 1);
    env.start();

    GRBModel model = GRBModel(env);

    int num_customers = instance.nC;
    int num_vehicles  = instance.nV;
    int depot         = 0; // Assume depot is node 0

    // Decision variables
    // x[i][j][v]: Binary variable that represents if vehicle v travels from node i to node j
    std::vector<std::vector<std::vector<GRBVar>>> x(
        num_customers + 1, std::vector<std::vector<GRBVar>>(num_customers + 1, std::vector<GRBVar>(num_vehicles)));

    // t[i]: Continuous variable that represents the arrival time at node i
    std::vector<GRBVar> t(num_customers + 1);

    // Create the decision variables
    for (int i = 0; i <= num_customers; ++i) {
        t[i] = model.addVar(instance.window_open[i], instance.window_close[i], 0.0, GRB_CONTINUOUS,
                            "t_" + std::to_string(i));
        for (int j = 0; j <= num_customers; ++j) {
            if (i != j) {
                for (int v = 0; v < num_vehicles; ++v) {
                    x[i][j][v] =
                        model.addVar(0.0, 1.0, instance.travel_cost[i][j], GRB_BINARY,
                                     "x_" + std::to_string(i) + "_" + std::to_string(j) + "_v" + std::to_string(v));
                }
            }
        }
    }

    model.update(); // Integrate the variables into the model

    // Objective: Minimize total travel cost
    GRBLinExpr obj = 0;
    for (int i = 0; i <= num_customers; ++i) {
        for (int j = 0; j <= num_customers; ++j) {
            if (i != j) {
                for (int v = 0; v < num_vehicles; ++v) { obj += instance.travel_cost[i][j] * x[i][j][v]; }
            }
        }
    }
    model.setObjective(obj, GRB_MINIMIZE);

    // Constraint 1: Flow conservation at each customer
    for (int i = 1; i <= num_customers; ++i) {
        GRBLinExpr flow_in  = 0;
        GRBLinExpr flow_out = 0;
        for (int v = 0; v < num_vehicles; ++v) {
            for (int j = 0; j <= num_customers; ++j) {
                if (i != j) {
                    flow_in += x[j][i][v];
                    flow_out += x[i][j][v];
                }
            }
        }
        model.addConstr(flow_in == 1, "flow_in_" + std::to_string(i));   // Exactly one vehicle enters each customer
        model.addConstr(flow_out == 1, "flow_out_" + std::to_string(i)); // Exactly one vehicle leaves each customer
    }

    // Constraint 2: Each vehicle leaves and returns to the depot
    for (int v = 0; v < num_vehicles; ++v) {
        GRBLinExpr vehicle_start = 0;
        GRBLinExpr vehicle_end   = 0;
        for (int i = 1; i <= num_customers; ++i) {
            vehicle_start += x[depot][i][v];
            vehicle_end += x[i][depot][v];
        }
        model.addConstr(vehicle_start == 1, "vehicle_start_" + std::to_string(v)); // Each vehicle leaves depot once
        model.addConstr(vehicle_end == 1, "vehicle_end_" + std::to_string(v));     // Each vehicle returns to depot once
    }

    // Constraint 3: Time window constraints
    for (int i = 1; i <= num_customers; ++i) {
        for (int j = 1; j <= num_customers; ++j) {
            if (i != j) {
                for (int v = 0; v < num_vehicles; ++v) {
                    model.addConstr(t[i] + instance.service_time[i] + instance.travel_cost[i][j] - t[j] <=
                                        (1 - x[i][j][v]) * instance.T_max,
                                    "time_" + std::to_string(i) + "_" + std::to_string(j) + "_v" + std::to_string(v));
                }
            }
        }
    }

    // Constraint 4: Vehicle capacity constraint
    for (int v = 0; v < num_vehicles; ++v) {
        GRBLinExpr load = 0;
        for (int i = 1; i <= num_customers; ++i) {
            for (int j = 1; j <= num_customers; ++j) {
                if (i != j) { load += instance.getDemand(i) * x[i][j][v]; }
            }
        }
        model.addConstr(load <= instance.getCapacity(), "capacity_v" + std::to_string(v));
    }

    // Optimize the model
    model.update();
    return model;
}

int main(int argc, char *argv[]) {

    // get instance name as the first arg
    std::string  instance_name = argv[1];
    InstanceData instance;
    if (VRPTW_read_instance(instance_name, instance)) {
        print_info("Instance read successfully.\n");
    } else {
        std::cerr << "Error reading instance\n";
    }

    auto model    = solve_vrp(instance);
    int  num_vars = model.get(GRB_IntAttr_NumVars);
    std::cout << "Number of variables: " << num_vars << std::endl;

    // Get the number of constraints
    int num_constrs = model.get(GRB_IntAttr_NumConstrs);
    std::cout << "Number of constraints: " << num_constrs << std::endl;

    print_info("Extracting model data\n");
    auto modelData = extractModelDataSparse(&model);

    Preprocessor preprocessor(modelData);
    print_info("Processing rows\n");

    preprocessor.processRowInformation();
    print_info("Finding knapsack rows\n");
    preprocessor.findKnapSackRows();
    print_info("Converting to LE\n");
    preprocessor.convert2LE();
    print_info("Converting to knapsack\n");
    // iterate over the number of rows
    for (size_t i = 0; i < modelData.A.size(); ++i) {
        print_info("Constraint: {}\n", i);
        preprocessor.convert2Knapsack(i);
    }
    auto          convertedModel = preprocessor.getModelData();
    CliqueManager cm(convertedModel);
    cm.findCliques();
    cm.printCliques();
};