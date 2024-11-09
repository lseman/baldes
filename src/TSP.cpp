#include "BucketGraph.h"
#include "BucketUtils.h"

#include "Common.h"

#include "Definitions.h"
#include "fmt/core.h"

#include "TSP.h"

#include "miphandler/LinExp.h"
#include "miphandler/MIPHandler.h"
#include "miphandler/Variable.h"

#include "solvers/SolverInterface.h"

#ifdef GUROBI
#include "gurobi_c++.h"
#include "gurobi_c.h"
#include "solvers/Gurobi.h"
#endif

std::vector<std::vector<double>> generateTSPInstance(int numNodes) {
    // Initialize the cost matrix
    std::vector<std::vector<double>> costMatrix(numNodes, std::vector<double>(numNodes, 0));

    // Set up random number generation
    std::random_device              rd;
    std::mt19937                    gen(rd());
    std::uniform_int_distribution<> dist(1, 100);

    // Fill the upper triangular matrix with random values
    for (int i = 0; i < numNodes; ++i) {
        for (int j = i + 1; j < numNodes; ++j) {
            int cost         = dist(gen);
            costMatrix[i][j] = cost;
            costMatrix[j][i] = cost; // Make the matrix symmetric
        }
    }

    // make the main diagonal 0
    for (int i = 0; i < numNodes; ++i) {
        costMatrix[i][i] = 0;
    }

    // Add an extra row and column for the end depot
    costMatrix.push_back(costMatrix[0]); // Copy first row to the last row
    for (auto &row : costMatrix) {
        row.push_back(row[0]); // Copy first column to the last column
    }
    //print cost matrix
    for (int i = 0; i < costMatrix.size(); ++i) {
        for (int j = 0; j < costMatrix[i].size(); ++j) {
            fmt::print("{} ", costMatrix[i][j]);
        }
        fmt::print("\n");
    }
    return costMatrix;
}

void solve_atsp(const std::vector<std::vector<double>> &cost_matrix, int num_nodes) {

    MIPProblem model = MIPProblem("tsp", 0, 0);

    int n = num_nodes - 1;
    fmt::print("n = {}\n", n);
    std::vector<std::vector<double>> dist = cost_matrix;

    // Create variables
    std::vector<std::vector<baldesVarPtr>> x(n, std::vector<baldesVarPtr>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            auto name = "x_" + std::to_string(i) + "_" + std::to_string(j);
            if (i != j) { x[i][j] = model.add_variable(name, VarType::Binary, 0.0, 1.0, dist[i][j]); }
        }
    }

    // Create additional u[i] continuous decision variables
    std::vector<baldesVarPtr> u(n);
    for (int i = 0; i < n; ++i) {
        auto name = "u_" + std::to_string(i);
        u[i]      = model.add_variable(name, VarType::Continuous, 0.0, n-1, 0.0);
    }

    // Set objective
    model.setObjectiveSense(ObjectiveType::Minimize);

    // Add constraints
    // Degree constraints
    for (int i = 0; i < n; ++i) {
        LinearExpression lhs_out;
        LinearExpression lhs_in;
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                lhs_out += x[i][j];
                lhs_in += x[j][i];
            }
        }
        model.add_constraint(lhs_out == 1, "degree_out_" + std::to_string(i));
        model.add_constraint(lhs_in == 1, "degree_in_" + std::to_string(i));
    }

    // Add MTZ subtour elimination constraints
    for (int i = 0; i < n; ++i) {
        for (int j = 1; j < n; ++j) {
            if (i != j) {
                auto             name = "mtz_" + std::to_string(i) + "_" + std::to_string(j);
                LinearExpression lhs;
                lhs += u[i] - u[j] + n * x[i][j];
                model.add_constraint(lhs <= n - 1, name);
            }
        }
    }

    // Fix u[0] to 0 (zero)
    //LinearExpression lhs;

    model.add_constraint(u[0] == 0, "u_0");

    GRBEnv &env      = GurobiEnvSingleton::getInstance();
    // Optimize model
    auto    modelGRB = new GRBModel(model.toGurobiModel(env));
    // set model verbose
    modelGRB->set(GRB_IntParam_OutputFlag, 0);
    // print model status
    fmt::print("Model status: {}\n", modelGRB->get(GRB_IntAttr_Status));
    // fmt::print("Model created\n");
    modelGRB->optimize();
    // print model objective val
    fmt::print("Original model objective value: {}\n", modelGRB->get(GRB_DoubleAttr_ObjVal));

    return;
}

// Function to generate initial p-step paths using a randomized greedy heuristic
void generateInitialPaths(int n, int p1, int p2, std::vector<std::vector<int>> &paths, std::vector<double> &path_costs,
                          std::vector<int> &firsts, std::vector<int> &lasts,
                          const std::vector<std::vector<double>> &cost_matrix, int num_paths) {
    std::random_device              rd;
    std::mt19937                    gen(rd());
    std::uniform_int_distribution<> dis(1, n - 2); // Random nodes excluding n-1

    for (int i = 0; i < num_paths; ++i) {
        int p     = (i % 2 == 0) ? p1 : p2;             // Alternate between p1 and p2
        int start = (i < num_paths / 2) ? 0 : dis(gen); // Half paths start at 0, others at random nodes

        std::vector<int> path;
        path.push_back(start);
        int    current    = start;
        double total_cost = 0.0;

        for (int step = 1; step < p && step < n; ++step) {
            std::vector<std::pair<int, double>> candidates;

            // Determine candidates based on the starting node
            if (start == 0) {
                // Paths starting at 0 should exclude n-1
                for (int node = 0; node < n - 1; ++node) {
                    if (std::find(path.begin(), path.end(), node) == path.end()) {
                        candidates.emplace_back(node, cost_matrix[current][node]);
                    }
                }
            } else {
                // Paths not starting at 0 should include n-1 as the last node, exclude it here initially
                for (int node = 0; node < n - 1; ++node) {
                    if (std::find(path.begin(), path.end(), node) == path.end()) {
                        candidates.emplace_back(node, cost_matrix[current][node]);
                    }
                }
            }

            if (!candidates.empty()) {
                // Sort candidates by cost
                std::sort(candidates.begin(), candidates.end(),
                          [](const auto &a, const auto &b) { return a.second < b.second; });

                // Select from the top 3 candidates to introduce randomness
                int                             k = std::min(3, (int)candidates.size());
                std::uniform_int_distribution<> select(0, k - 1);
                int                             chosen_idx = select(gen);

                int next_node = candidates[chosen_idx].first;

                // Retry if next_node is 0 (avoid cycles)
                while (next_node == 0 && k > 1) {
                    chosen_idx = select(gen);
                    next_node  = candidates[chosen_idx].first;
                }

                double min_cost = candidates[chosen_idx].second;
                path.push_back(next_node);
                current = next_node;
                total_cost += min_cost;
            } else {
                break; // No more nodes to visit
            }
        }

        // If path does not start at 0, ensure it ends with n-1
        if (start != 0) {
            // Adjust total cost for the final step to n-1
            if (!path.empty()) {
                total_cost -= cost_matrix[path[path.size() - 2]][path.back()];
                path.pop_back();
            }
            path.push_back(n - 1);
            total_cost += cost_matrix[path[path.size() - 2]][n - 1];
        }

        // Store the generated path and its associated information
        paths.push_back(path);
        path_costs.push_back(total_cost);
        firsts.push_back(start);
        lasts.push_back(path.back());
    }
}



void initializeNodes(std::vector<VRPNode> &nodes, int num_nodes) {
    std::random_device              rd;
    std::mt19937                    gen(rd());
    std::uniform_int_distribution<> random_bounds(0, 1000);
    std::uniform_int_distribution<> random_location(0, 100);

    // Initialize nodes with random properties
    for (int id = 0; id < num_nodes; ++id) {
        VRPNode node;
        node.id = id;

        // Set random bounds
        node.lb = {0};
        node.ub = {1000};

        // Set other properties
        node.duration    = 0;
        node.cost        = 0;
        node.demand      = 0;
        node.consumption = {0};

        // Set random location
        node.set_location(random_location(gen), random_location(gen));

        // Add node to the list
        nodes.push_back(node);
    }
}

int main() {
    // ATSPInstance instance("br17.atsp");
    // int          nC          = instance.dimension;
    // auto         cost_matrix = instance.distance_matrix;
    int                              nC          = 10;
    auto                             cost_matrix = generateTSPInstance(9);
    std::vector<VRPNode> vrpnodes;

    // Initialize nodes
    initializeNodes(vrpnodes, nC);

    solve_atsp(cost_matrix, nC);

    auto bg = BucketGraph(vrpnodes, 10000, 1);
    bg.set_distance_matrix(cost_matrix);
    BucketOptions options;
    options.depot         = 0;
    options.end_depot     = nC;
    options.max_path_size = 4;
    options.pstep = true;
    fmt::print("Solving PSTEP by MTZ\n");
    bg.solvePSTEP_by_MTZ();
    fmt::print("Solved PSTEP by MTZ\n");

    return 0;
}
