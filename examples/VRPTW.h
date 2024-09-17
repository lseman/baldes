/**
 * @file VRPTW.h
 * @brief Defines the Vehicle Routing Problem with Time Windows (VRPTW) class and its associated methods.
 *
 * This file contains the implementation of the `VRProblem` class, which solves the Vehicle Routing Problem with
 * Time Windows (VRPTW) using column generation and various optimization techniques. The VRPTW class handles:
 * - Column generation and constraint handling using the Gurobi optimizer.
 * - Addition and removal of variables and paths from the model.
 * - Handling of dual values and reduced costs for optimization.
 * - Stabilization mechanisms using Limited Memory Rank 1 Cuts and RCC separation.
 * - Integration with external algorithms such as bucket graphs and RCC managers for advanced routing optimization.
 *
 * The VRPTW class uses Gurobi for solving linear programming relaxations and applies iterative methods to refine
 * the solution with constraints and separation cuts.
 */

#pragma once

#include "../../bucket/include/Definitions.h"

#include "../../bucket/include/BucketGraph.h"
#include "../../bucket/include/BucketUtils.h"

#include "../extra/Stabilization.h"

#include "../../bucket/include/Reader.h"

#ifdef RCC
#include "../cvrpsep/basegrph.h"
#include "../cvrpsep/capsep.h"
#include "../cvrpsep/cnstrmgr.h"
#endif

#include "../cuts/RCC.h"
#include "../include/Hashes.h"

#include <tbb/concurrent_unordered_map.h>

class VRProblem {
public:
    InstanceData instance;

    std::vector<VRPJob> jobs;
    std::vector<Path>   allPaths;

    int                           labels_counter = 0;
    std::vector<std::vector<int>> labels;
#ifdef RCC
    CnstrMgrPointer oldCutsCMP = nullptr;
#endif
    RCCmanager rccManager;

    tbb::concurrent_unordered_map<std::pair<int, int>, std::vector<GRBLinExpr>, pair_hash, pair_equal> arcCache;

    /*
     * Adds a column to the GRBModel.
     *
     * @param node A pointer to the GRBModel object.
     * @param columns The columns to be added.
     * @param enumerate Flag indicating whether to enumerate the columns.
     * @return The number of columns added.
     */
    int addColumn(GRBModel *node, const auto &columns, CutStorage &r1c, bool enumerate = false) {

        int                 numConstrs = node->get(GRB_IntAttr_NumConstrs);
        const GRBConstr    *constrs    = node->getConstrs();
        std::vector<double> coluna(numConstrs, 0.0); // Declare outside the loop

        // iterate over columns
        auto counter = 0;
        for (auto &label : columns) {
            if (label->jobs_covered.size() == 0) { continue; }
            if (!enumerate && label->cost > 0) continue;
            counter += 1;
            labels_counter += 1;
            std::fill(coluna.begin(), coluna.end(), 0.0); // Reset for each iteration

            double      travel_cost = label->real_cost;
            std::string name        = "x[" + std::to_string(allPaths.size() - 1) + "]";
            GRBColumn   col;
            for (auto &job : label->jobs_covered) {
                if (job > 0 && job != N_SIZE - 1) {
                    int constr_index = job - 1;
                    coluna[constr_index] += 1;
                }
            }

            for (int i = 0; i < N_SIZE - 1; i++) {
                if (coluna[i] == 0.0) { continue; }
                col.addTerms(&coluna[i], &constrs[i], 1);
            }
#ifdef SRC
            auto vec = r1c.computeLimitedMemoryCoefficients(label->jobs_covered);
            if (vec.size() > 0) {
                for (int i = 0; i < vec.size(); i++) {
                    if (vec[i] != 0) {
                        auto ctr = r1c.getCtr(i);
                        col.addTerms(&vec[i], &constrs[N_SIZE - 1 + i], 1);
                    }
                }
            }
#endif
            node->addVar(0.0, 1.0, travel_cost, GRB_CONTINUOUS, col, name);
            Path path(label->jobs_covered, label->real_cost);
            allPaths.push_back(path);
        }
        node->update();
        return counter;
    }

    /**
     * @brief Changes the coefficients of a constraint in the given GRBModel.
     *
     * This function modifies the coefficients of a constraint in the specified GRBModel
     * by iterating over the variables and updating their coefficients using the provided values.
     *
     * @param model A pointer to the GRBModel object.
     * @param constrName The name of the constraint to modify.
     * @param value A vector containing the new coefficients for each variable.
     */
    void chgCoeff(GRBModel *model, const GRBConstr &constrName, std::vector<double> value) {
        auto varNames = model->getVars();
        int  numVars  = model->get(GRB_IntAttr_NumVars);
        for (int i = 0; i < numVars; i++) { model->chgCoeff(constrName, varNames[i], value[i]); }
    }

    /**
     * Binarizes the variables in the given GRBModel.
     *
     * This function sets the variable type of each variable in the model to binary.
     * It iterates over all variables in the model and sets their variable type to GRB_BINARY.
     * After updating the model, all variables will be binary variables.
     *
     * @param model A pointer to the GRBModel to be modified.
     */
    void binarizeNode(GRBModel *model) {
        auto varNumber = model->get(GRB_IntAttr_NumVars);
        for (int i = 0; i < varNumber; i++) {
            GRBVar var = model->getVar(i);
            var.set(GRB_CharAttr_VType, GRB_BINARY);
        }
        model->update();
    }
    /**
     * Removes variables with negative reduced costs, up to a maximum of 30% of the total variables,
     * and also removes corresponding elements from the allPaths vector.
     *
     * @param model A pointer to the GRBModel object.
     * @param allPaths A reference to the vector of paths associated with the variables.
     */
    void removeNegativeReducedCostVarsAndPaths(GRBModel *model) {
        int                 varNumber = model->get(GRB_IntAttr_NumVars);
        std::vector<GRBVar> vars(varNumber);

        // Collect all variables
        for (int i = 0; i < varNumber; ++i) { vars[i] = model->getVar(i); }

        // Vector to store reduced costs
        std::vector<double> reducedCosts(varNumber);

        // Retrieve the reduced cost for each variable individually
        for (int i = 0; i < varNumber; ++i) { reducedCosts[i] = vars[i].get(GRB_DoubleAttr_RC); }

        // Vector to store indices of variables with negative reduced costs
        std::vector<int> indicesToRemove;

        // Identify variables with negative reduced costs
        for (int i = 0; i < varNumber; ++i) {
            if (reducedCosts[i] < 0) { indicesToRemove.push_back(i); }
        }

        // Limit the number of variables to remove to 30% of the total
        size_t maxRemoval = static_cast<size_t>(0.3 * varNumber);
        if (indicesToRemove.size() > maxRemoval) {
            indicesToRemove.resize(maxRemoval); // Only keep the first 30%
        }

        // Remove the selected variables from the model and corresponding paths from allPaths
        for (int i = indicesToRemove.size() - 1; i >= 0; --i) {
            int index = indicesToRemove[i];
            model->remove(vars[index]);               // Remove the variable from the model
            allPaths.erase(allPaths.begin() + index); // Remove the corresponding path from allPaths
        }

        model->update(); // Apply changes to the model
    }

    /**
     * Retrieves the dual values of the constraints in the given GRBModel.
     *
     * @param model A pointer to the GRBModel object.
     * @return A vector of double values representing the dual values of the constraints.
     */
    std::vector<double> getDuals(GRBModel *model) {
        int                    numConstrs = model->get(GRB_IntAttr_NumConstrs);
        std::vector<GRBConstr> constraints;
        constraints.reserve(numConstrs);

        // Collect all constraints
        for (int i = 0; i < numConstrs; ++i) { constraints.push_back(model->getConstr(i)); }

        // Prepare the duals vector
        std::vector<double> duals(numConstrs);

        // Retrieve all dual values in one call
        auto dualArray = model->get(GRB_DoubleAttr_Pi, constraints.data(), constraints.size());

        duals.assign(dualArray, dualArray + numConstrs);
        return duals;
    }

    /**
     * Extracts the solution from a given GRBModel object.
     *
     * @param model A pointer to the GRBModel object.
     * @return A vector of doubles representing the solution.
     */
    std::vector<double> extractSolution(GRBModel *model) {
        std::vector<double> sol;
        auto                varNumber = model->get(GRB_IntAttr_NumVars);
        for (int i = 0; i < varNumber; i++) { sol.push_back(model->getVar(i).get(GRB_DoubleAttr_X)); }

        return sol;
    }

    /**
     * Handles the cuts for the VRPTW problem.
     *
     * @param r1c The LimitedMemoryRank1Cuts object.
     * @param node The GRBModel pointer.
     * @param constraints The vector of GRBConstr objects.
     * @return A boolean indicating whether any changes were made.
     */
    bool cutHandler(LimitedMemoryRank1Cuts &r1c, GRBModel *node, std::vector<GRBConstr> &constraints) {
        auto &cuts    = r1c.cutStorage;
        bool  changed = false;
        if (cuts.size() > 0) {
            for (Cut &cut : cuts) {
                bool added   = cut.added;
                bool updated = cut.updated;
                if (added && !updated) { continue; }

                changed                     = true;
                int         z               = cut.id;
                std::string constraint_name = "cuts(z" + std::to_string(z) + ")";
                auto        coeffs          = cut.coefficients;
                if (added && updated) {
                    cut.updated = false;
                    if (z >= constraints.size()) { continue; }
                    auto constraint = constraints[z];
                    chgCoeff(node, constraint, coeffs);
                } else {
                    GRBLinExpr lhs;
                    for (int i = 0; i < coeffs.size(); i++) {
                        if (coeffs[i] == 0) { continue; }
                        auto v = node->getVar(i);
                        lhs += v * coeffs[i];
                    }
                    auto ctr = node->addConstr(lhs <= cut.rhs, constraint_name);
                    constraints.push_back(ctr);
                }
                cut.added   = true;
                cut.updated = false;
            }

            r1c.cutStorage = cuts;
        }
        return changed;
    }

    /**
     * @brief Separates Rounded Capacity Cuts (RCC) for the given model and solution.
     *
     * This function identifies and separates RCCs for the provided model and solution.
     * It uses parallel execution to handle multiple tasks concurrently, generating
     * cut expressions and right-hand side values for each identified cut.
     *
     * @param model Pointer to the Gurobi model.
     * @param solution Vector of doubles representing the solution.
     * @param constraints 3D vector of Gurobi constraints.
     * @return True if any RCCs were added, false otherwise.
     */
    bool exactRCCsep(GRBModel *model, const std::vector<double> &solution,
                     std::vector<std::vector<std::vector<GRBConstr>>> &constraints) {
        auto Ss = separate_Rounded_Capacity_cuts(model, instance.q, instance.demand, model->get(GRB_DoubleAttr_ObjVal),
                                                 true, allPaths);

        if (Ss.size() == 0) return false;

        // Define the tasks that need to be executed in parallel
        std::vector<int> tasks(Ss.size());
        std::iota(tasks.begin(), tasks.end(), 0); // Filling tasks with [0, 1, ..., cutsCMP->Size-1]

        std::vector<GRBLinExpr>          cutExpressions(Ss.size());
        std::vector<int>                 rhsValues(Ss.size());
        std::vector<std::vector<RCCarc>> arcGroups(Ss.size());
        std::mutex                       cuts_mutex;
        std::mutex                       arcCache_mutex; // Mutex to protect arcCache

        const int                JOBS = 10;
        exec::static_thread_pool pool(JOBS);
        auto                     sched = pool.get_scheduler();

        //  Create a bulk sender to handle parallel execution of tasks
        auto bulk_sender = stdexec::bulk(
            stdexec::just(), tasks.size(),
            [this, &Ss, &arcCache_mutex, &model, &cuts_mutex, &tasks, &cutExpressions, &rhsValues,
             &arcGroups](std::size_t task_idx) {
                int c = tasks[task_idx];

                auto S = Ss[c];

                GRBLinExpr          cutExpr = 0.0;
                std::vector<RCCarc> arcs;

                std::unordered_map<std::pair<int, int>, GRBLinExpr, pair_hash, pair_equal> localContributions;

                // Generate all possible arc pairs from nodes in S
                for (int i : S) {
                    for (int j : S) {
                        if (i != j) {
                            arcs.emplace_back(i, j);
                            auto cache_key = std::make_pair(i, j);

                            if (localContributions.find(cache_key) == localContributions.end()) {
                                GRBLinExpr contributionSum = 0.0;
                                for (size_t ctr = 0; ctr < allPaths.size(); ++ctr) {
                                    contributionSum += allPaths[ctr].timesArc(i, j) * model->getVar(ctr);
                                }
                                localContributions[cache_key] = contributionSum;
                                cutExpr += contributionSum;
                            } else {
                                cutExpr += localContributions[cache_key];
                            }
                        }
                    }
                }
                auto target_no_vehicles = ceil(
                    std::accumulate(S.begin(), S.end(), 0, [&](int sum, int i) { return sum + instance.demand[i]; }) /
                    instance.q);
                {
                    cutExpressions[c] = cutExpr;
                    rhsValues[c]      = S.size() - target_no_vehicles;

                    arcGroups[c] = std::move(arcs);
                }
            });

        // Define the work and run it synchronously
        auto work = stdexec::on(sched, bulk_sender);
        stdexec::sync_wait(std::move(work));
        std::vector<GRBConstr> newConstraints(cutExpressions.size());

        for (size_t i = 0; i < cutExpressions.size(); ++i) {
            newConstraints[i] = model->addConstr(cutExpressions[i] <= rhsValues[i]);
        }

        rccManager.addCutBulk(arcGroups, rhsValues, newConstraints);
        std::print("Added {} RCC cuts\n", cutExpressions.size());
        model->update();
        model->optimize();

        return true;
    }

#ifdef RCC
    /**
     * @brief Performs RCC (Robust Capacity Cuts) separation for the given model and solution.
     *
     * This function identifies and adds violated RCC cuts to the model to improve the solution.
     * It uses parallel execution to handle the separation and constraint addition efficiently.
     *
     * @param model Pointer to the Gurobi model.
     * @param solution Vector containing the current solution values.
     * @param constraints 3D vector to store the generated constraints.
     * @return True if violated RCC cuts were found and added to the model, false otherwise.
     *
     * The function follows these steps:
     * 1. Initializes the constraint manager and precomputes edge values from the LP solution.
     * 2. Identifies edges with significant values and prepares data for RCC separation.
     * 3. Performs RCC separation to find violated cuts.
     * 4. If no violated cuts are found, returns false indicating an optimal solution.
     * 5. If violated cuts are found, processes them in parallel to generate constraints.
     * 6. Adds the generated constraints to the model and updates the model.
     */

    bool RCCsep(GRBModel *model, const std::vector<double> &solution,
                std::vector<std::vector<std::vector<GRBConstr>>> &constraints) {

        // Constraint manager to store cuts
        CnstrMgrPointer cutsCMP = nullptr;
        CMGR_CreateCMgr(&cutsCMP, 100);

        auto             nVertices = N_SIZE - 1;
        std::vector<int> demands   = instance.demand;
        std::mutex       arcCache_mutex; // Mutex to protect arcCache

        // Precompute edge values from LP solution
        std::vector<std::vector<double>> aijs(N_SIZE + 2, std::vector<double>(N_SIZE + 2, 0.0));

        for (int counter = 0; counter < solution.size(); ++counter) {
            const auto &nodes = allPaths[counter].route;
            for (size_t k = 1; k < nodes.size(); ++k) {
                int source = nodes[k - 1];
                int target = (nodes[k] == N_SIZE - 1) ? 0 : nodes[k];
                aijs[source][target] += solution[counter];
            }
        }

        std::vector<int>    edgex, edgey;
        std::vector<double> edgeval;

        edgex.push_back(0);
        edgey.push_back(0);
        edgeval.push_back(0.0);

        for (int i = 0; i < nVertices; ++i) {
            for (int j = i + 1; j < nVertices; ++j) {
                double xij = aijs[i][j] + aijs[j][i];
                if (xij > 1e-4) {
                    {
                        edgex.push_back((i == 0) ? N_SIZE - 1 : i);
                        edgey.push_back(j);
                        edgeval.push_back(xij);
                    }
                }
            }
        }

        // RCC Separation
        char   intAndFeasible;
        double maxViolation = 0.0;
        CAPSEP_SeparateCapCuts(nVertices - 1, demands.data(), instance.q, edgex.size() - 1, edgex.data(), edgey.data(),
                               edgeval.data(), oldCutsCMP, 20, 1e-4, &intAndFeasible, &maxViolation, cutsCMP);

        if (intAndFeasible) return false; /* Optimal solution found */

        if (cutsCMP->Size == 0) return false;
        // Define the tasks that need to be executed in parallel
        auto tasks = std::vector<int>(cutsCMP->Size);

        std::iota(tasks.begin(), tasks.end(), 0); // Filling tasks with [0, 1, ..., cutsCMP->Size-1]

        print_cut("Found {} violated RCC cuts, max violation: {}\n", cutsCMP->Size, maxViolation);

        std::vector<int>                 rhsValues(cutsCMP->Size);
        std::vector<std::vector<RCCarc>> arcGroups(cutsCMP->Size);
        std::mutex                       cuts_mutex;

        const int                JOBS = 10;
        exec::static_thread_pool pool(JOBS);
        auto                     sched = pool.get_scheduler();

        // Create a bulk sender to handle parallel execution of tasks
        auto bulk_sender = stdexec::bulk(
            stdexec::just(), tasks.size(),
            [this, &cutsCMP, &model, &cuts_mutex, &tasks, &rhsValues, &arcGroups,

             &arcCache_mutex](std::size_t task_idx) {
                // Get the current task (cut index)
                int              cutIdx = tasks[task_idx];
                std::vector<int> S(cutsCMP->CPL[cutIdx]->IntList + 1,
                                   cutsCMP->CPL[cutIdx]->IntList + cutsCMP->CPL[cutIdx]->IntListSize + 1);

                GRBLinExpr                                                                 cutExpr = 0.0;
                std::vector<RCCarc>                                                        arcs;
                std::unordered_map<std::pair<int, int>, GRBLinExpr, pair_hash, pair_equal> localContributions;

                // Generate all possible arc pairs from nodes in S
                for (int i : S) {
                    for (int j : S) {
                        if (i != j) { arcs.emplace_back(i, j); }
                    }
                }

                rhsValues[cutIdx] = cutsCMP->CPL[cutIdx]->RHS;
                arcGroups[cutIdx] = std::move(arcs);
            });

        // Define the work and run it synchronously
        auto work = stdexec::on(sched, bulk_sender);
        stdexec::sync_wait(std::move(work));

        // Create a bulk sender to handle parallel execution of each constraint
        auto bulk_sender_ctr = stdexec::bulk(stdexec::just(), rhsValues.size(), // We want rhsValues.size() tasks
                                             [&](std::size_t i) { // Lambda function for each task (constraint index i)
                                                 GRBLinExpr cutExpr = 0.0;

                                                 // For each arc in arcGroups[i], compute the cut expression
                                                 for (const auto &arc : arcGroups[i]) {
                                                     for (size_t ctr = 0; ctr < allPaths.size(); ++ctr) {
                                                         cutExpr += allPaths[ctr].getArcCount(arc) * model->getVar(ctr);
                                                     }
                                                 }

                                                 // Add the constraint to the model
                                                 auto ctr = model->addConstr(cutExpr <= rhsValues[i]);
                                                 rccManager.addCut(arcGroups[i], rhsValues[i], ctr);

                                             });

        // Attach to the scheduler and run the work synchronously
        auto work_ctr = stdexec::on(sched, bulk_sender_ctr);
        stdexec::sync_wait(std::move(work_ctr)); // Wait for the tasks to complete
        model->update();
        model->optimize();

        return true;
    }
#endif

    /**
     * Column generation algorithm.
     */
    void CG(GRBModel *node) {
        print_info("Column generation started...\n");

        node->optimize();
        int bucket_interval = 20;
        int time_horizon    = instance.T_max;

        int                 numConstrs = node->get(GRB_IntAttr_NumConstrs);
        std::vector<double> duals      = std::vector<double>(numConstrs, 0.0);

        // BucketGraph bucket_graph(jobs, time_horizon, bucket_interval, instance.q, bucket_interval);
        BucketGraph bucket_graph(jobs, time_horizon, bucket_interval);

        bucket_graph.set_distance_matrix(instance.getDistanceMatrix(), 8);

        node->optimize();
        auto integer_solution               = node->get(GRB_DoubleAttr_ObjVal);
        bucket_graph.incumbent              = integer_solution;
        int                    num_veihcles = labels_counter;
        auto                   allJobs      = bucket_graph.getJobs();
        LimitedMemoryRank1Cuts r1c(allJobs, CutType::ThreeRow);
        CutStorage             cuts;
        r1c.cutStorage = cuts;
        std::vector<GRBConstr> constraints;
        std::vector<double>    cutDuals;
        std::vector<double>    jobDuals = getDuals(node);

        double highs_obj_dual = 0.0;
        double highs_obj      = 0.0;

        bucket_graph.setup();

        double gap = 1e-6;

        bool s1    = true;
        bool s2    = false;
        bool s3    = false;
        bool s4    = false;
        bool s5    = false;
        bool ss    = false;
        int  stage = 1;

        double bidi_relation = bucket_graph.bidi_relation;
        double relation      = 0.5;
        double lag_gap       = std::numeric_limits<double>::max();

        auto                 inner_obj = 0.0;
        std::vector<Label *> paths;
        std::vector<double>  solution;
        bool                 can_add = true;

        Stabilization stab(0.9, jobDuals);
        double        split   = time_horizon * 0.5;
        bool          changed = false;
        // set start timer
        auto start_timer = std::chrono::high_resolution_clock::now();

        std::print("\n");
        print_info("Starting column generation..\n\n");
        bool transition = false;

        bucket_graph.rcc_manager = &rccManager;

        std::vector<std::vector<std::vector<GRBConstr>>> cvrsep_ctrs(
            instance.nC + 3, std::vector<std::vector<GRBConstr>>(instance.nC + 3, std::vector<GRBConstr>()));

#ifdef RCC
        CMGR_CreateCMgr(&oldCutsCMP, 100); // For old cuts, if needed
#endif
        bool rcc = false;
        for (int iter = 0; iter < 1000; ++iter) {
            cuts    = r1c.cutStorage;
            changed = cutHandler(r1c, node, constraints);

            if (changed) {
                node->update();
                node->optimize();
            }
            if (cuts.size() > 0) {
                jobDuals          = getDuals(node);
                auto matrixSparse = extractModelDataSparse(node);
                auto numJobs      = bucket_graph.getJobs().size() - 1;
                cutDuals          = std::vector<double>(jobDuals.begin() + numJobs, jobDuals.end());
                cuts.setDuals(cutDuals);
            }

            auto onlyJobDuals = std::vector<double>(jobDuals.begin(), jobDuals.begin() + bucket_graph.getJobs().size());
            bucket_graph.setDuals(onlyJobDuals);

            //////////////////////////////////////////////////////////////////////
            // ADAPTIVE TERMINAL TIME
            //////////////////////////////////////////////////////////////////////
            double n_fw_labels = bucket_graph.n_fw_labels;
            double n_bw_labels = bucket_graph.n_bw_labels;

            if (((n_bw_labels - n_fw_labels) / n_fw_labels) > 0.05) {
                split += 0.05 * time_horizon;
            } else if (((n_fw_labels - n_bw_labels) / n_bw_labels) > 0.05) {
                split -= 0.05 * time_horizon;
            }
            std::vector<double> q_star = {split};

            bucket_graph.cut_storage = &cuts;

            //////////////////////////////////////////////////////////////////////
            // ADAPTIVE STAGE HANDLING
            //////////////////////////////////////////////////////////////////////
            if (s1) {
                stage = 1;
                paths = bucket_graph.bi_labeling_algorithm<Stage::One>(q_star);

                inner_obj = paths[0]->cost;

                if (inner_obj >= -1 || iter >= 10) {
                    s1 = false;
                    s2 = true;
                }
            } else if (s2) {
                s2        = true;
                stage     = 2;
                paths     = bucket_graph.bi_labeling_algorithm<Stage::Two>(q_star);
                inner_obj = paths[0]->cost;
                if (inner_obj >= -100 || iter > 800) {
                    s2 = false;
                    s3 = true;
                }
            } else if (s3) {
                stage     = 3;
                paths     = bucket_graph.bi_labeling_algorithm<Stage::Three>(q_star);
                inner_obj = paths[0]->cost;

                if (inner_obj >= -1e-2) {
                    s4 = true;
                    s3 = false;
                }
            } else if (s4) {
                stage = 4;
                rccManager.computeDualsDeleteAndCache(node);
                paths     = bucket_graph.bi_labeling_algorithm<Stage::Four>(q_star);
                inner_obj = paths[0]->cost;
                if (inner_obj >= -1e-1) {
                    ss = true;
#ifndef SRC
                    break;
#endif
                    print_cut("Going into separation mode..\n");

#ifdef RCC
                    print_cut("Testing RCC feasibility..\n");
                    rcc                      = RCCsep(node, solution, cvrsep_ctrs);
                    bucket_graph.rcc_manager = &rccManager;
                    if (rcc) { duals = getDuals(node); }
#endif
                }
            }

            //////////////////////////////////////////////////////////////////////
            bidi_relation = bucket_graph.bidi_relation;

            auto colAdded = addColumn(node, paths, cuts, false);

            if (colAdded == 0) {
                stab.add_misprice();
            } else {
                stab.reset_misprices();
            }

            double obj;

            node->optimize();
            highs_obj         = node->get(GRB_DoubleAttr_ObjVal);
            auto dual_obj     = node->get(GRB_DoubleAttr_ObjBound);
            jobDuals          = getDuals(node);
            auto matrixSparse = extractModelDataSparse(node);
            jobDuals          = stab.run(matrixSparse, jobDuals);
            solution = extractSolution(node);

#ifdef RCC
            if (rcc) {
                rcc = RCCsep(node, solution, cvrsep_ctrs);
                rccManager.computeDualsDeleteAndCache(node);
            }
#endif
            bool integer = true;
            // Check integrality of the solution
            for (auto &sol : solution) {
                if (sol > 1e-1 && sol < 1 - 1e-1) {
                    integer = false;
                    break;
                }
            }
            if (integer) {
                if (highs_obj < integer_solution) {
                    print_info("Updating integer solution to {}\n", highs_obj);
                    integer_solution       = highs_obj;
                    bucket_graph.incumbent = integer_solution;
                }
            }

            lag_gap          = integer_solution - (highs_obj + std::min(0.0, inner_obj));
            bucket_graph.gap = lag_gap;
            bucket_graph.augment_ng_memories(solution, allPaths, true, 5, 100, 16, N_SIZE);

            bucket_graph.relaxation = highs_obj;

#if defined(SRC3) || defined(SRC)
            if (ss && !rcc) {
                print_info("Removing most negative reduced cost variables\n");
                removeNegativeReducedCostVarsAndPaths(node);
                node->optimize();
                highs_obj    = node->get(GRB_DoubleAttr_ObjVal);
                solution     = extractSolution(node);
                matrixSparse = extractModelDataSparse(node);
                r1c.allPaths = allPaths;
                r1c.separate(matrixSparse.A_sparse, solution, N_SIZE - 2, 1e-3);
                std::print("Separating cuts..\n");
#ifdef SRC
                r1c.the45Heuristic<CutType::FourRow>(matrixSparse.A_sparse, solution, N_SIZE - 2, 1e-3);
                r1c.the45Heuristic<CutType::FiveRow>(matrixSparse.A_sparse, solution, N_SIZE - 2, 1e-3);
#endif
                cuts = r1c.cutStorage;
                if (cuts.size() == 0) {
                    print_info("No violations found, calling it a day\n");
                    break;
                }
                std::print("Found {} violated cuts\n", cuts.size());
                ss = false;
            }
#endif

            if (iter % 10 == 0)
                std::print("| It.: {:4} | Obj.: {:8.2f} | Pricing: {:10.2f} | Cuts: {:4} | Paths: {:4} | "
                           "Stage: {:1} | "
                           "Lag. Gap: {:10.4f} | RCC: {:4} |\n",
                           iter, highs_obj, inner_obj, cuts.size(), paths.size(), stage, lag_gap, rcc);
        }
        auto end_timer        = std::chrono::high_resolution_clock::now();
        auto duration_ms      = std::chrono::duration_cast<std::chrono::milliseconds>(end_timer - start_timer).count();
        auto duration_seconds = duration_ms / 1000;
        auto duration_milliseconds = duration_ms % 1000;

        bucket_graph.print_statistics();

        node->optimize();
        auto relaxed_result = node->get(GRB_DoubleAttr_ObjVal);

        binarizeNode(node);
        node->optimize();
        auto ip_result = node->get(GRB_DoubleAttr_ObjVal);

        // ANSI escape code for blue text
        constexpr auto blue  = "\033[34m";
        constexpr auto reset = "\033[0m";

        std::print("+----------------------+----------------+\n");
        std::print("| {:<14} | {}{:>20}{} |\n", "Bound", blue, relaxed_result, reset);
        std::print("| {:<14} | {}{:>20}{} |\n", "Incumbent", blue, ip_result, reset);
        std::print("| {:<14} | {}{:>16}.{:03}{} |\n", "VRP Duration", blue, duration_seconds, duration_milliseconds,
                   reset);
        std::print("+----------------------+----------------+\n");
    }
};
