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
                              sub_options.pstep = true; // print depot and end_depot
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
                                      auto new_label = compute_mono_label(label);
                                      if ((new_label->nodes_covered.size() > 1) &&
                                          (new_label->nodes_covered.back() == sub_options.end_depot)) {
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

inline void generateInitialPaths(int n, int p1, int p2, std::vector<std::vector<int>> &paths,
                                 std::vector<double> &path_costs, std::vector<int> &firsts, std::vector<int> &lasts,
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
        firsts.push_back(path.front());
        lasts.push_back(path.back());
    }
}

inline auto solveTSP(std::vector<std::vector<int>> &paths, std::vector<double> &path_costs, std::vector<int> &firsts,
                     std::vector<int> &lasts, std::vector<std::vector<double>> &cost_matrix, bool first_time = false) {
    auto       n         = 10; // paths[0].size(); // Number of nodes
    MIPProblem model     = MIPProblem("tsp", 0, 0);
    int        p1        = 4;
    int        p2        = 4;
    int        num_paths = 100; // Number of initial paths to generate
    fmt::print("n = {}\n", n);
    if (first_time) { generateInitialPaths(n, p1, p2, paths, path_costs, firsts, lasts, cost_matrix, num_paths); }
    // Define variables
    int                       R = paths.size(); // Number of p-steps
    std::vector<baldesVarPtr> x(R);             // Binary variables for each p-step
    std::vector<baldesVarPtr> u(n);             // Continuous variables for each node

    // Create binary variables x_r for each p-step r
    fmt::print("R = {}\n", R);
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
        fmt::print("i = {}\n", i);
        LinearExpression in_constraint;
        LinearExpression out_constraint;
        for (int r = 0; r < R; ++r) {
            if (firsts[r] == i) { in_constraint += x[r]; }
            if (lasts[r] == i) { in_constraint -= x[r]; }
            if (std::find(paths[r].begin(), paths[r].end(), i) != paths[r].end() && i != lasts[r]) {
                out_constraint += x[r];
            }
        }
        three_two[i] = model.add_constraint(in_constraint == 0, "33_" + std::to_string(i));
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

    GRBEnv &env      = GurobiEnvSingleton::getInstance();
    auto    modelGRB = new GRBModel(model.toGurobiModel(env));
    // set model verbose
    modelGRB->set(GRB_IntParam_OutputFlag, 0);
    // set model type as 2
    // modelGRB->set(GRB_IntParam_Method, 2);
    // fmt::print("Model created\n");
    modelGRB->update();
    modelGRB->optimize();
    modelGRB->write("tsp.lp");

    std::vector<double>              three_two_duals(n, 0.0);
    std::vector<double>              three_three_duals(n, 0.0);
    std::vector<std::vector<double>> three_five_duals(n, std::vector<double>(n + 1, 0.0));

    if (modelGRB->get(GRB_IntAttr_Status) == GRB_OPTIMAL) {
        // Extract dual variables for the constraints
        fmt::print("Model optimal with value: {}\n", modelGRB->get(GRB_DoubleAttr_ObjVal));
        for (int i = 1; i < n - 1; ++i) {
            three_two_duals[i]   = modelGRB->getConstrByName("33_" + std::to_string(i)).get(GRB_DoubleAttr_Pi);
            three_three_duals[i] = -modelGRB->getConstrByName("34_" + std::to_string(i)).get(GRB_DoubleAttr_Pi);
        }

        // Extract duals for three_five constraints
        for (int i = 0; i < n - 1; ++i) {
            for (int j = 1; j < n; ++j) {
                if (i != j) {
                    three_five_duals[i][j] =
                        -(n - 1) * modelGRB->getConstrByName("mtz_" + std::to_string(i) + "_" + std::to_string(j))
                                       .get(GRB_DoubleAttr_Pi);
                }
            }
        }
    } else {
        fmt::print("Model not optimal\n");
    }
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

    auto duals = solveTSP(all_paths, all_costs, all_firsts, all_lasts, distance_matrix, true);

    for (auto z = 0; z < 3; z++) {
        auto three_two_duals = std::get<0>(duals);
        // print three two duals
        for (int i = 0; i < three_two_duals.size(); ++i) { fmt::print("{} ", three_two_duals[i]); }
        fmt::print("\n");
        auto three_three_duals = std::get<1>(duals);
        for (int i = 0; i < three_three_duals.size(); ++i) { fmt::print("{} ", three_three_duals[i]); }
        fmt::print("\n");
        auto       three_five_duals = std::get<2>(duals);
        PSTEPDuals inner_pstep_duals;

        // Convert three_two to vector of pairs
        std::vector<std::pair<int, double>> three_two_tuples;
        for (size_t i = 0; i < three_two_duals.size(); ++i) { three_two_tuples.emplace_back(i, three_two_duals[i]); }
        inner_pstep_duals.setThreeTwoDualValues(three_two_tuples);

        // Convert three_three to vector of pairs
        std::vector<std::pair<int, double>> three_three_tuples;
        for (size_t i = 0; i < three_three_duals.size(); ++i) {
            three_three_tuples.emplace_back(i, three_three_duals[i]);
        }
        inner_pstep_duals.setThreeThreeDualValues(three_three_tuples);

        // Convert three_five to vector of pairs of pairs
        std::vector<std::pair<std::pair<int, int>, double>> arc_duals_tuples;
        for (size_t i = 0; i < three_five_duals.size(); ++i) {
            for (size_t j = 0; j < three_five_duals[i].size(); ++j) {
                if (i != j) { // Skip diagonal or add specific condition if needed
                    arc_duals_tuples.emplace_back(std::make_pair(std::make_pair(i, j), three_five_duals[i][j]));
                }
            }
        }
        inner_pstep_duals.setArcDualValues(arc_duals_tuples);
        auto sub_paths = solvePSTEP(inner_pstep_duals);
        // print sub_paths.size

        for (auto path : sub_paths) {
            all_paths.push_back(path->nodes_covered);
            // print path->nodes_covered

            all_costs.push_back(path->real_cost);
            all_firsts.push_back(path->nodes_covered.front());
            all_lasts.push_back(path->nodes_covered.back());
        }

        duals = solveTSP(all_paths, all_costs, all_firsts, all_lasts, distance_matrix);
    }
    return {};
}