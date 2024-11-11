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

    const int CHUNK_SIZE = 1;

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

                              auto sub_bg = BucketGraph(nodes, 10000, 1);
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
                              forward_cbar =
                                  sub_bg.labeling_algorithm<Direction::Forward, Stage::Enumerate, Full::PSTEP>();
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

    IPSolver *ipSolver = new IPSolver();
    auto      matrix   = model.extractModelDataSparse();
    ipSolver->run_optimization(matrix, 1e-2);

    auto                             duals = ipSolver->getDuals();
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
    auto lp_obj = ipSolver->getObjective();
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

inline auto solveTSP2(std::vector<std::vector<int>> &paths, std::vector<double> &path_costs, std::vector<int> &firsts,
                      std::vector<int> &lasts, std::vector<std::vector<double>> &cost_matrix, bool first_time = false) {

    auto       n     = 10; // Number of nodes
    MIPProblem model = MIPProblem("tsp", 0, 0);
    int        p1 = 4, p2 = 4, num_paths = 100;
    // if (first_time) { generateInitialPaths(n, p1, p2, paths, path_costs, firsts, lasts, cost_matrix, num_paths); }

    // Define variables
    int                                    R = paths.size(); // Number of p-steps
    std::vector<baldesVarPtr>              x(R);             // Continuous variables for each p-step
    std::vector<std::vector<baldesVarPtr>> theta(n, std::vector<baldesVarPtr>(n)); // Binary variables for arcs

    // Create continuous variables x_r for each p-step r
    for (int r = 0; r < R; ++r) {
        std::string varname = "x_" + std::to_string(r);
        x[r]                = model.add_variable(varname, VarType::Continuous, 0.0, 1.0, path_costs[r]);
    }

    // Create binary variables theta_ij for each arc (i, j)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                std::string varname = "theta_" + std::to_string(i) + "_" + std::to_string(j);
                theta[i][j]         = model.add_variable(varname, VarType::Binary, 0.0, 1.0, 0.0);
            }
        }
    }

    // Set the objective function: Minimize sum of path costs
    model.setObjectiveSense(ObjectiveType::Minimize);

    // Constraints (3.14): Ensuring correct entry and exit from nodes
    for (int i = 1; i < n - 1; ++i) {
        LinearExpression in_constraint, out_constraint;
        for (int r = 0; r < R; ++r) {
            if (firsts[r] == i) in_constraint += x[r];
            if (lasts[r] == i) out_constraint += x[r];
        }
        model.add_constraint(in_constraint == 0, "concat_" + std::to_string(i));
        model.add_constraint(out_constraint == 2, "visit_once_" + std::to_string(i));
    }

    // Constraints (3.16): Sub-tour elimination using visit counts
    for (int i = 1; i < n - 1; ++i) {
        LinearExpression visit_sum;
        for (int r = 0; r < R; ++r) { visit_sum += x[r] * (firsts[r] == i ? 1 : lasts[r] == i ? -1 : 0); }
        model.add_constraint(visit_sum >= 0, "subtour_" + std::to_string(i));
    }

    // Constraints (3.23): Linking x_r with theta_ij
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                LinearExpression arc_sum;
                for (int r = 0; r < R; ++r) {
                    // Check if the arc (i, j) belongs to path r
                    for (int k = 0; k < paths[r].size() - 1; ++k) {
                        if (paths[r][k] == i && paths[r][k + 1] == j) {
                            arc_sum += x[r];
                            break;
                        }
                    }
                }
                model.add_constraint(arc_sum - theta[i][j] == 0,
                                     "arc_link_" + std::to_string(i) + "_" + std::to_string(j));
            }
        }
    }

    // Solve the linear relaxation first
    GRBEnv &env      = GurobiEnvSingleton::getInstance();
    auto    modelGRB = new GRBModel(model.toGurobiModel(env));
    modelGRB->set(GRB_IntParam_OutputFlag, 0);
    modelGRB->optimize();

    if (modelGRB->get(GRB_IntAttr_Status) == GRB_OPTIMAL) {
        fmt::print("Linear relaxation solved with value: {}\n", modelGRB->get(GRB_DoubleAttr_ObjVal));
    } else {
        fmt::print("Linear relaxation not optimal\n");
    }

    // Convert theta variables to binary and re-solve
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                modelGRB->getVarByName("theta_" + std::to_string(i) + "_" + std::to_string(j))
                    .set(GRB_CharAttr_VType, GRB_BINARY);
            }
        }
    }
    modelGRB->update();
    modelGRB->optimize();

    if (modelGRB->get(GRB_IntAttr_Status) == GRB_OPTIMAL) {
        fmt::print("Integer solution found with value: {}\n", modelGRB->get(GRB_DoubleAttr_ObjVal));
    } else {
        fmt::print("Integer solution not optimal\n");
    }

    return modelGRB->get(GRB_DoubleAttr_ObjVal);
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
            all_paths.push_back(path->nodes_covered);
            all_costs.push_back(path->real_cost);
            all_firsts.push_back(path->nodes_covered.front());
            all_lasts.push_back(path->nodes_covered.back());
        }
        if (paths_added == 0) { break; }

        duals = solveTSP(all_paths, all_costs, all_firsts, all_lasts, distance_matrix);
    }
    return {};
}