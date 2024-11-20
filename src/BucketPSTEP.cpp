#include "Common.h"
#include "Definitions.h"

#include "bucket/BucketGraph.h"
#include "bucket/BucketSolve.h"
#include "bucket/BucketUtils.h"

#include "../third_party/pdqsort.h"

#include "MST.h"
#include <omp.h>

#ifdef GUROBI
#include "gurobi_c++.h"
#include "gurobi_c.h"
#include "solvers/Gurobi.h"
#endif

#ifdef IPM
#include "ipm/IPSolver.h"
#include "solvers/IPM.h"
#endif

std::vector<Label *> BucketGraph::solvePSTEP(PSTEPDuals &inner_pstep_duals) {

    const int CHUNK_SIZE = 10;

    std::vector<Label *> all_paths;
    std::vector<double>  all_costs;
    std::vector<int>     all_firsts, all_lasts;

    // Mutex for shared resources
    std::mutex cuts_mutex;

    const int                JOBS = nodes.size();
    exec::static_thread_pool pool(std::thread::hardware_concurrency());
    auto                     sched = pool.get_scheduler();

    std::vector<std::pair<int, int>> tasks;
    for (int i = 0; i < JOBS - 1; ++i) {
        for (int j = 1; j < JOBS; ++j) {
            if (i == j) continue;
            tasks.emplace_back(i, j);
        }
    }
    auto bulk_sender =
        stdexec::bulk(stdexec::just(), (tasks.size() + CHUNK_SIZE - 1) / CHUNK_SIZE, // Calculate total chunks
                      [&](std::size_t chunk_idx) {
                          // Calculate indices i, j from chunk_idx
                          size_t start_idx = chunk_idx * CHUNK_SIZE;
                          size_t end_idx   = std::min(start_idx + CHUNK_SIZE, tasks.size());
                          for (size_t task_idx = start_idx; task_idx < end_idx; ++task_idx) {
                              auto [i, j] = tasks[task_idx];
                              BucketOptions sub_options;
                              auto          sub_bg = BucketGraph(nodes, R_max[0], 20);
                              sub_bg.set_distance_matrix(distance_matrix);
                              sub_options.depot     = i;
                              sub_options.end_depot = j;
                              // print i and j
                              sub_options.pstep         = true; // print depot and end_depot
                              sub_options.max_path_size = options.max_path_size;
                              sub_options.min_path_size = options.min_path_size;
                              sub_bg.setOptions(sub_options);
                              sub_bg.setPSTEPduals(inner_pstep_duals);
                              sub_bg.setup();
                              sub_bg.reset_pool();
                              sub_bg.mono_initialization();
                              // Solve and get the paths
                              std::vector<double> forward_cbar(sub_bg.fw_buckets.size());
                              forward_cbar = sub_bg.labeling_algorithm<Direction::Forward, Stage::Four, Full::PSTEP>();
                              std::vector<Label *> paths;
                              // print paths.size
                              for (auto bucket : std::ranges::iota_view(0, sub_bg.fw_buckets_size)) {
                                  auto bucket_labels = sub_bg.fw_buckets[bucket].get_labels();
                                  // print bucket_labels size
                                  for (auto label : bucket_labels) {
                                      if (!label) continue;
                                      auto new_label = compute_mono_label(label);
                                      if ((new_label->nodes_covered.size() > 1) &&
                                          (new_label->nodes_covered.back() == sub_options.end_depot) &&
                                          (new_label->nodes_covered.size() >= options.min_path_size &&
                                           new_label->nodes_covered.size() <= options.max_path_size)) {
                                          paths.push_back(new_label);
                                      }
                                  }
                              }

                              // Check if no paths are found
                              if (paths.empty()) return;

                              // Lock to protect shared resources
                              {
                                  std::lock_guard<std::mutex> lock(cuts_mutex);
                                  // Process the found paths
                                  for (const auto &path : paths) { all_paths.push_back(path); }
                              }
                              //   }
                          }
                      });

    // Submit work to the thread pool and wait for completion
    auto work = stdexec::starts_on(sched, bulk_sender);
    stdexec::sync_wait(std::move(work));

    return all_paths;
}
void generateInitialPathsTSPTW(int n, std::vector<std::vector<int>> &paths, std::vector<double> &path_costs,
                               std::vector<int> &firsts, std::vector<int> &lasts,
                               const std::vector<std::vector<double>> &cost_matrix,
                               const std::vector<double> &start_times, const std::vector<double> &end_times,
                               const std::vector<double> &service_times) {
    std::vector<bool> visited(n + 1, false); // Increased size to include node 101
    std::vector<int>  initial_path;
    int               current_node = 0;
    double            current_time = 0.0;
    double            cost         = 0.0;

    // Debug prints
    fmt::print("Start times[0]: {}, End times[0]: {}\n", start_times[0], end_times[0]);
    fmt::print("Start times[101]: {}, End times[101]: {}\n", start_times[101], end_times[101]);

    visited[current_node] = true;
    initial_path.push_back(current_node);

    for (int step = 1; step < n; ++step) {
        int    nearest_node = -1;
        double min_cost     = std::numeric_limits<double>::max();

        for (int j = 1; j < n; ++j) {
            if (!visited[j]) {
                double travel_cost  = cost_matrix[current_node][j];
                double arrival_time = current_time + travel_cost;

                fmt::print("Considering node {}: arrival={}, window=[{},{}]\n", j, arrival_time, start_times[j],
                           end_times[j]);

                if (arrival_time <= end_times[j]) {
                    double wait_time      = std::max(0.0, start_times[j] - arrival_time);
                    double total_cost     = travel_cost + wait_time;
                    double departure_time = std::max(arrival_time, start_times[j]) + service_times[j];

                    if (total_cost < min_cost && departure_time <= end_times[j]) {
                        nearest_node = j;
                        min_cost     = total_cost;
                    }
                }
            }
        }

        if (nearest_node != -1) {
            fmt::print("Selected node {}\n", nearest_node);
            visited[nearest_node] = true;
            current_node          = nearest_node;
            initial_path.push_back(current_node);

            double travel_cost = cost_matrix[initial_path[step - 1]][current_node];
            current_time += travel_cost;
            current_time = std::max(current_time, start_times[current_node]);
            current_time += service_times[current_node];
            cost += travel_cost;

            fmt::print("Current time after service: {}\n", current_time);
        }
    }

    double return_cost   = cost_matrix[current_node][101];
    double final_arrival = current_time + return_cost;
    fmt::print("Final arrival at 101: {}, Window: [{},{}]\n", final_arrival, start_times[101], end_times[101]);

    if (final_arrival <= end_times[101]) {
        cost += return_cost;
        initial_path.push_back(101);
        paths.push_back(initial_path);
        firsts.push_back(initial_path.front());
        lasts.push_back(initial_path.back());
        path_costs.push_back(cost);
    } else {
        fmt::print("Path not feasible - violates end time at node 101\n");
    }

    for (int i = 0; i < paths.size(); ++i) {
        fmt::print("Path {}: ", i);
        for (auto node : paths[i]) { fmt::print("{} ", node); }
        fmt::print("\n");
    }
}
inline void generateInitialPaths(int n, std::vector<std::vector<int>> &paths, std::vector<double> &path_costs,
                                 std::vector<int> &firsts, std::vector<int> &lasts,
                                 const std::vector<std::vector<double>> &cost_matrix) {
    std::vector<bool> visited(n, false);
    std::vector<int>  initial_path;
    int               current_node = 0;
    visited[current_node]          = true;
    initial_path.push_back(current_node);

    auto cost = 0.0;

    // Generate the path using the nearest neighbor heuristic
    for (int step = 1; step < n; ++step) {
        int    nearest_node = -1;
        double min_cost     = std::numeric_limits<double>::max();

        for (int j = 0; j < n; ++j) {
            if (!visited[j] && cost_matrix[current_node][j] < min_cost) {
                nearest_node = j;
                min_cost     = cost_matrix[current_node][j];
            }
        }

        if (nearest_node != -1) {
            initial_path.push_back(nearest_node);
            visited[nearest_node] = true;
            current_node          = nearest_node;
        }
        cost += min_cost;
    }
    // Return to the starting node to complete the cycle
    cost += cost_matrix[current_node][0];
    initial_path.push_back(n);
    paths.push_back(initial_path);
    firsts.push_back(initial_path.front());
    lasts.push_back(initial_path.back());
    path_costs.push_back(cost);

    fmt::print("Initial heuristic cost = {}\n", cost);
}

auto BucketGraph::solveTSP(std::vector<std::vector<int>> &paths, std::vector<double> &path_costs,
                           std::vector<int> &firsts, std::vector<int> &lasts,
                           std::vector<std::vector<double>> &cost_matrix, bool first_time) {
    auto       n     = cost_matrix.size();
    MIPProblem model = MIPProblem("tsp", 0, 0);
    if (first_time) { generateInitialPaths(n - 1, paths, path_costs, firsts, lasts, cost_matrix); }
    // Define variables
    int                       R = paths.size(); // Number of p-steps
    std::vector<baldesVarPtr> x(R);             // Binary variables for each p-step
    std::vector<baldesVarPtr> u(n);             // Continuous variables for each node

    // Create binary variables x_r for each p-step r
    for (int i = 0; i < R; ++i) {
        // fmt::print("path_costs[{}] = {}\n", i, path_costs[i]);
        std::string varname = "x_" + std::to_string(i);
        double      custo   = path_costs[i];
        x[i]                = model.add_variable(varname, VarType::Continuous, 0.0, 1.0, custo);
    }

    // Create continuous variables u_i for each node i
    for (int i = 0; i < n; ++i) {
        auto name = "u_" + std::to_string(i);
        u[i]      = model.add_variable(name, VarType::Continuous, 0.0, n - 1, 0.0);
    }
    model.setObjectiveSense(ObjectiveType::Minimize);

    // Constraints (3.3) and (3.4):
    std::vector<baldesCtrPtr> three_two(n);
    std::vector<baldesCtrPtr> three_three(n);
    for (int i = 1; i < n - 1; ++i) {
        LinearExpression in_constraint;
        LinearExpression out_constraint;
        for (int r = 0; r < R; ++r) {
            if (firsts[r] == i) { in_constraint += x[r]; }
            if (lasts[r] == i) { in_constraint -= x[r]; }
            if (std::find(paths[r].begin(), paths[r].end(), i) != paths[r].end() && i != lasts[r]) {
                out_constraint += x[r];
            }
        }
        three_two[i]   = model.add_constraint(in_constraint == 0, "33_" + std::to_string(i));
        three_three[i] = model.add_constraint(out_constraint == 1, "34_" + std::to_string(i));
    }

    // Constraints (3.5): Avoid sub-tours using MTZ constraints
    std::vector<baldesCtrPtr>              three_five_constraints;
    std::vector<std::vector<baldesCtrPtr>> three_five_constraints_matrix(n, std::vector<baldesCtrPtr>(n));
    for (int i = 0; i < n - 1; ++i) {
        for (int j = 1; j < n; ++j) {
            if (i != j) {
                LinearExpression r_sum;
                for (int r = 0; r < R; ++r) {
                    for (int k = 0; k < paths[r].size() - 1; ++k) {
                        if (paths[r][k] == i && paths[r][k + 1] == j) { r_sum += x[r]; }
                    }
                }
                auto ctr = u[i] - u[j] + (n - 1) * r_sum <= (n - 2);
                three_five_constraints_matrix[i][j] =
                    model.add_constraint(ctr, "mtz_" + std::to_string(i) + "_" + std::to_string(j));
            }
        }
    }

    for (int i = 1; i < n; i++) {
        model.add_constraint(u[i] >= 0, "u_lb_" + std::to_string(i));
        model.add_constraint(u[i] <= n - 1, "u_ub_" + std::to_string(i));
    }

#ifdef IPM
    IPSolver *ipSolver = new IPSolver();
    auto      matrix   = model.extractModelDataSparse();
    ipSolver->run_optimization(matrix, 1e-2);

    auto duals = ipSolver->getDuals();
#else
    auto duals = std::vector<double>(N_SIZE, 0.0);
#endif
    std::vector<double>              three_two_duals(n, 0.0);
    std::vector<double>              three_three_duals(n, 0.0);
    std::vector<std::vector<double>> three_five_duals(n, std::vector<double>(n + 1, 0.0));
    for (int i = 1; i < n - 1; ++i) {
        three_two_duals[i]   = options.three_two_sign * duals[three_two[i]->index()];
        three_three_duals[i] = options.three_three_sign * duals[three_three[i]->index()];
    }

    // Extract duals for three_five constraints
    for (int i = 0; i < n - 1; ++i) {
        for (int j = 1; j < n; ++j) {
            if (i != j) {
                three_five_duals[i][j] =
                    options.three_five_sign * (n - 1) * duals[three_five_constraints_matrix[i][j]->index()];
            }
        }
    }
#ifdef IPM
    auto lp_obj = ipSolver->getObjective();
#else
    auto lp_obj = 0.0;
#endif
    fmt::print("LP Objective: {}\n", lp_obj);
    GRBEnv &env      = GurobiEnvSingleton::getInstance();
    auto    modelGRB = new GRBModel(model.toGurobiModel(env));
    // set model verbose
    modelGRB->set(GRB_IntParam_OutputFlag, 0);
    // set model type as 2
    modelGRB->set(GRB_IntParam_Method, 2);

    // convert x vars to int and solve again
    for (int i = 0; i < R; ++i) {
        modelGRB->getVarByName("x_" + std::to_string(i)).set(GRB_CharAttr_VType, GRB_BINARY);
    }
    modelGRB->update();
    modelGRB->optimize();

    if (modelGRB->get(GRB_IntAttr_Status) == GRB_OPTIMAL) {
        fmt::print("Integer solution found with value: {}\n", modelGRB->get(GRB_DoubleAttr_ObjVal));
        for (int i = 0; i < R; ++i) {
            if (modelGRB->getVarByName("x_" + std::to_string(i)).get(GRB_DoubleAttr_X) > 0.5) {
                fmt::print("x_{} = {}\n", i, modelGRB->getVarByName("x_" + std::to_string(i)).get(GRB_DoubleAttr_X));
                // print path nodes
                for (auto node : paths[i]) { fmt::print("{} ", node); }
                fmt::print("\n");
            }
        }
    }

    return std::make_tuple(three_two_duals, three_three_duals, three_five_duals);
}

std::vector<Label *> BucketGraph::solvePSTEP_by_MTZ() {
    int JOBS = nodes.size();

    std::vector<std::vector<int>> all_paths;
    std::vector<double>           all_costs;
    std::vector<int>              all_firsts;
    std::vector<int>              all_lasts;

    auto duals                = solveTSP(all_paths, all_costs, all_firsts, all_lasts, distance_matrix, true);
    auto max_pstep_iterations = 10;
    for (auto z = 0; z < max_pstep_iterations; ++z) {
        fmt::print("--------------------------------------\n");
        fmt::print("Iteration {}\n", z);
        auto       three_two_duals   = std::get<0>(duals);
        auto       three_three_duals = std::get<1>(duals);
        auto       three_five_duals  = std::get<2>(duals);
        PSTEPDuals inner_pstep_duals;

        // Convert three_two to vector of pairs
        std::vector<std::pair<int, double>> three_two_tuples;
        three_two_tuples.reserve(three_two_duals.size());
        std::transform(three_two_duals.begin(), three_two_duals.end(), std::back_inserter(three_two_tuples),
                       [index = 0](double value) mutable { return std::make_pair(index++, value); });
        inner_pstep_duals.setThreeTwoDualValues(std::move(three_two_tuples));

        // Convert three_three to vector of pairs
        std::vector<std::pair<int, double>> three_three_tuples;
        three_three_tuples.reserve(three_three_duals.size());
        std::transform(three_three_duals.begin(), three_three_duals.end(), std::back_inserter(three_three_tuples),
                       [index = 0](double value) mutable { return std::make_pair(index++, value); });
        inner_pstep_duals.setThreeThreeDualValues(std::move(three_three_tuples));

        // Convert three_five to vector of pairs of pairs
        std::vector<std::pair<std::pair<int, int>, double>> arc_duals_tuples;
        size_t                                              total_size = 0;
        for (const auto &row : three_five_duals) total_size += row.size();
        arc_duals_tuples.reserve(total_size);

        for (size_t i = 0; i < three_five_duals.size(); ++i) {
            for (size_t j = 0; j < three_five_duals[i].size(); ++j) {
                if (i != j) { // Skip diagonal or add specific condition if needed
                    arc_duals_tuples.emplace_back(std::pair{i, j}, three_five_duals[i][j]);
                }
            }
        }
        inner_pstep_duals.setArcDualValues(std::move(arc_duals_tuples));

        auto sub_paths = solvePSTEP(inner_pstep_duals);
        // print sub_paths.size
        auto paths_added = 0;
        for (auto path : sub_paths) {
            if (path->cost >= 0) { continue; }
            paths_added++;
            all_paths.push_back(path->getRoute());
            all_costs.push_back(path->real_cost);
            all_firsts.push_back(path->nodes_covered.front());
            all_lasts.push_back(path->nodes_covered.back());
        }
        if (paths_added == 0) { break; }

        duals = solveTSP(all_paths, all_costs, all_firsts, all_lasts, distance_matrix);
    }
    return {};
}

auto BucketGraph::solveTSPTW(std::vector<std::vector<int>> &paths, std::vector<double> &path_costs,
                             std::vector<int> &firsts, std::vector<int> &lasts,
                             std::vector<std::vector<double>> &cost_matrix, std::vector<double> &service_times,
                             std::vector<double> &time_windows_start, std::vector<double> &time_windows_end,
                             bool first_time) {
    auto       n     = cost_matrix.size();
    MIPProblem model = MIPProblem("tsptw", 0, 0);

    if (first_time) {
        generateInitialPathsTSPTW(n - 1, paths, path_costs, firsts, lasts, cost_matrix, time_windows_start,
                                  time_windows_end, service_times);
    }

    // Variables
    int                       R = paths.size();
    std::vector<baldesVarPtr> x(R);     // Binary variables for p-steps
    std::vector<baldesVarPtr> omega(n); // Continuous variables for accumulated time

    // Create binary variables x_r for each p-step r
    for (int i = 0; i < R; ++i) {
        x[i] = model.add_variable("x_" + std::to_string(i), VarType::Binary, 0.0, 1.0, path_costs[i]);
    }

    // Create continuous variables ω_i for time accumulation
    for (int i = 0; i < n; ++i) {
        omega[i] = model.add_variable("omega_" + std::to_string(i), VarType::Continuous, 0.0, 1e8, 0.0);
    }
    model.setObjectiveSense(ObjectiveType::Minimize);

    std::vector<baldesCtrPtr> three_two(n);
    std::vector<baldesCtrPtr> three_three(n);
    // Constraints (4.11): Flow balance for p-steps
    for (int i = 1; i < n - 1; ++i) {
        LinearExpression flow_balance;
        for (int r = 0; r < R; ++r) {
            if (firsts[r] == i) flow_balance += x[r];
            if (lasts[r] == i) flow_balance -= x[r];
        }
        three_two[i] = model.add_constraint(flow_balance == 0, "flow_" + std::to_string(i));
    }

    // Constraints (4.12): Visit each node at most once
    for (int i = 1; i < n - 1; ++i) {
        LinearExpression visit_once;
        for (int r = 0; r < R; ++r) {
            if (std::find(paths[r].begin(), paths[r].end(), i) != paths[r].end()) { visit_once += x[r]; }
        }
        three_three[i] = model.add_constraint(visit_once <= 1, "visit_" + std::to_string(i));
    }

    // Constraint (4.13): Visit source exactly once
    LinearExpression source_visit;
    for (int r = 0; r < R; ++r) {
        if (firsts[r] == 0) { source_visit += x[r]; }
    }
    model.add_constraint(source_visit == 1, "source");

    // Constraints (4.14): Time accumulation with big-M
    std::vector<baldesCtrPtr>              three_five_constraints;
    std::vector<std::vector<baldesCtrPtr>> three_five_constraints_matrix(n, std::vector<baldesCtrPtr>(n));
    for (int i = 0; i < n - 1; ++i) {
        for (int j = 1; j < n; ++j) {
            if (i != j) {
                double           M_ij = 1e8;
                LinearExpression time_acc;
                for (int r = 0; r < R; ++r) {
                    for (size_t k = 0; k < paths[r].size() - 1; ++k) {
                        if (paths[r][k] == i && paths[r][k + 1] == j) { time_acc += x[r]; }
                    }
                }
                three_five_constraints_matrix[i][j] = model.add_constraint(
                    omega[j] - omega[i] - (cost_matrix[i][j] + service_times[i]) * time_acc - M_ij * time_acc >= -M_ij,
                    "time_" + std::to_string(i) + "_" + std::to_string(j));
            }
        }
    }

    // define omega[0] = 0

    // Constraints (4.15): Time window constraints
    for (int i = 1; i < n - 1; ++i) {
        LinearExpression node_used;
        for (int r = 0; r < R; ++r) {
            if (std::find(paths[r].begin(), paths[r].end(), i) != paths[r].end()) { node_used += x[r]; }
        }
        model.add_constraint(omega[i] - time_windows_start[i] * node_used >= 0, "tw_lb_" + std::to_string(i));
        model.add_constraint(omega[i] - time_windows_end[i] * node_used <= 0, "tw_ub_" + std::to_string(i));
    }

// Solve the model
#ifdef IPM
    IPSolver *ipSolver = new IPSolver();
    auto      matrix   = model.extractModelDataSparse();
    ipSolver->run_optimization(matrix, 1e-2);

    // Extract solution and duals
    auto duals = ipSolver->getDuals();
#else
    auto duals = std::vector<double>(N_SIZE, 0.0);
#endif
    std::vector<double>              three_two_duals(n, 0.0);
    std::vector<double>              three_three_duals(n, 0.0);
    std::vector<std::vector<double>> three_five_duals(n, std::vector<double>(n + 1, 0.0));
    for (int i = 1; i < n - 1; ++i) {
        three_two_duals[i]   = options.three_two_sign * duals[three_two[i]->index()];
        three_three_duals[i] = options.three_three_sign * duals[three_three[i]->index()];
        fmt::print("Duals for node {}: 3.2 = {}, 3.3 = {}\n", i, three_two_duals[i], three_three_duals[i]);
    }

    // Extract duals for three_five constraints
    for (int i = 0; i < n - 1; ++i) {
        for (int j = 1; j < n; ++j) {
            if (i != j) {
                three_five_duals[i][j] =
                    options.three_five_sign * (n - 1) * duals[three_five_constraints_matrix[i][j]->index()];
            }
        }
    }

    // Solution output and return
    //fmt::print("LP Objective: {}\n", ipSolver->getObjective());

    GRBEnv &env = GurobiEnvSingleton::getInstance();

    auto modelGRB = new GRBModel(model.toGurobiModel(env));
    // set model verbose
    modelGRB->set(GRB_IntParam_OutputFlag, 1);
    // set model type as 2
    modelGRB->set(GRB_IntParam_Method, 0);

    modelGRB->update();
    modelGRB->optimize();

    return std::make_tuple(three_two_duals, three_three_duals, three_five_duals);
}

std::vector<Label *> BucketGraph::solveTSPTW_by_MTZ() {
    int JOBS = nodes.size();

    // Initialize containers for storing paths, costs, firsts, and lasts
    std::vector<std::vector<int>> all_paths;
    std::vector<double>           all_costs;
    std::vector<int>              all_firsts;
    std::vector<int>              all_lasts;

    // extract service time from nodes
    std::vector<double> service_times;
    std::vector<double> time_windows_start(JOBS, 0.0);
    std::vector<double> time_windows_end(JOBS, 10000.0);
    for (int i = 0; i < JOBS; ++i) {
        time_windows_start[i] = nodes[i].lb[0];
        time_windows_end[i]   = nodes[i].ub[0];
        service_times.push_back(nodes[i].duration);
    }
    // Step 1: Solve the initial TSPTW using the previously defined method
    auto duals = solveTSPTW(all_paths, all_costs, all_firsts, all_lasts, distance_matrix, service_times,
                            time_windows_start, time_windows_end, true);

    // Step 2: Set up iteration for generating new paths using dual values
    int max_pstep_iterations = 10;
    for (int z = 0; z < max_pstep_iterations; ++z) {
        fmt::print("--------------------------------------\n");
        fmt::print("Iteration {}\n", z);

        // Extract dual values for constraints
        auto three_two_duals   = std::get<0>(duals);
        auto three_three_duals = std::get<1>(duals);
        auto three_five_duals  = std::get<2>(duals);
        // auto arrival_time_duals = std::get<3>(duals);

        PSTEPDuals inner_pstep_duals;

        // Convert dual values for constraint (3.32)
        std::vector<std::pair<int, double>> three_two_tuples;
        three_two_tuples.reserve(three_two_duals.size());
        std::transform(three_two_duals.begin(), three_two_duals.end(), std::back_inserter(three_two_tuples),
                       [index = 0](double value) mutable { return std::make_pair(index++, value); });
        inner_pstep_duals.setThreeTwoDualValues(std::move(three_two_tuples));

        // Convert dual values for constraint (3.33)
        std::vector<std::pair<int, double>> three_three_tuples;
        three_three_tuples.reserve(three_three_duals.size());
        std::transform(three_three_duals.begin(), three_three_duals.end(), std::back_inserter(three_three_tuples),
                       [index = 0](double value) mutable { return std::make_pair(index++, value); });
        inner_pstep_duals.setThreeThreeDualValues(std::move(three_three_tuples));

        // Convert dual values for MTZ constraints (3.34) and time windows (3.35)
        std::vector<std::pair<std::pair<int, int>, double>> arc_duals_tuples;
        for (size_t i = 0; i < three_five_duals.size(); ++i) {
            for (size_t j = 0; j < three_five_duals[i].size(); ++j) {
                if (i != j) { arc_duals_tuples.emplace_back(std::pair{i, j}, three_five_duals[i][j]); }
            }
        }
        inner_pstep_duals.setArcDualValues(std::move(arc_duals_tuples));

        for (size_t i = 0; i < three_five_duals.size(); ++i) {
            for (size_t j = 0; j < three_five_duals[i].size(); ++j) {
                if (i != j) { // Skip diagonal or add specific condition if needed
                    arc_duals_tuples.emplace_back(std::pair{i, j}, three_five_duals[i][j]);
                }
            }
        }
        inner_pstep_duals.setArcDualValues(std::move(arc_duals_tuples));

        // Step 3: Solve the subproblem to generate new p-steps
        fmt::print("Solving PSTEP subproblem\n");
        auto sub_paths = solvePSTEP(inner_pstep_duals);
        fmt::print("Subproblem solved\n");

        // Add the new paths to the existing list if valid
        int paths_added = 0;
        for (auto path : sub_paths) {
            if (path->cost >= 0) { continue; }
            paths_added++;
            all_paths.push_back(path->getRoute());
            all_costs.push_back(path->real_cost);
            all_firsts.push_back(path->nodes_covered.front());
            all_lasts.push_back(path->nodes_covered.back());
        }

        // If no new paths were added, stop the iteration
        if (paths_added == 0) { break; }

        // Step 4: Re-solve the TSPTW with the updated set of p-steps
        duals = solveTSPTW(all_paths, all_costs, all_firsts, all_lasts, distance_matrix, service_times,
                           time_windows_start, time_windows_end);
    }

    return {};
}
