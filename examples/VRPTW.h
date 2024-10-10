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

#include "Arc.h"
#include "Cut.h"
#include "Definitions.h"

#include "RCC.h"

#include "bnb/Branching.h"
#include "bnb/Node.h"

#include "bucket/BucketGraph.h"
#include "bucket/BucketSolve.h"
#include "bucket/BucketUtils.h"

#ifdef STAB
#include "extra/Stabilization.h"
#endif

#include "Reader.h"
#include "bnb/Problem.h"

#ifdef RCC
#include "../third_party/cvrpsep/capsep.h"
#include "../third_party/cvrpsep/cnstrmgr.h"
// #include "ModernRCC.h"
#endif

#ifdef EXACT_RCC
#include "cuts/ExactRCC.h"
#endif

#include "Hashes.h"

#ifdef IPM
#include "ipm/IPSolver.h"
#endif

#define NUMERO_CANDIDATOS 10

class VRProblem : public Problem {
public:
    InstanceData instance;

    double ip_result      = std::numeric_limits<double>::max();
    double relaxed_result = std::numeric_limits<double>::max();

    std::vector<VRPNode> nodes;

    std::vector<std::vector<int>> labels;
    int                           numConstrs = 0;

    std::vector<Path> toMerge;

#ifdef EXACT_RCC
    RCCManager rccManager;
#endif

    int addPath(BNBNode *node, const std::vector<Path> paths, bool enumerate = false) {
        auto &r1c  = node->r1c;
        auto &cuts = r1c.cutStorage;

        int              numConstrsLocal = node->get(GRB_IntAttr_NumConstrs);
        const GRBConstr *constrs         = node->getConstrs();

        auto &matrix         = node->matrix;
        auto &allPaths       = node->paths;
        auto &SRCconstraints = node->SRCconstraints;

        std::vector<double> coluna(numConstrsLocal, 0.0); // Declare outside the loop

        // Collect the bounds, costs, names, and columns
        std::vector<double>      lb, ub, obj;
        std::vector<GRBColumn>   cols;
        std::vector<std::string> names;
        std::vector<char>        vtypes;

        auto counter = 0;
        for (auto &label : paths) {
            counter += 1;
            // if (counter > 10) break;

            std::fill(coluna.begin(), coluna.end(), 0.0); // Reset for each iteration

            double      travel_cost = label.cost;
            std::string name        = "x[" + std::to_string(allPaths.size()) + "]";

            GRBColumn col;

            // Fill coluna with coefficients
            // Step 1: Accumulate the coefficients for each node
            for (auto &node : label.route) {
                if (node > 0 && node != N_SIZE - 1) {
                    int constr_index = node - 1;
                    coluna[constr_index] += 1; // Accumulate the coefficient for each node
                }
            }

            // Step 2: Add the non-zero entries to the sparse matrix
            for (int i = 0; i < N_SIZE - 1; ++i) {
                if (coluna[i] > 0) {
                    matrix.A_sparse.elements.push_back({i, matrix.A_sparse.num_cols, static_cast<double>(coluna[i])});
                }
            }

            matrix.lb.push_back(0.0);
            matrix.ub.push_back(1.0);
            matrix.c.push_back(travel_cost);
            // Add terms to GRBColumn
            for (int i = 0; i < N_SIZE - 2; i++) {
                if (coluna[i] == 0.0) continue;
                col.addTerms(&coluna[i], &constrs[i], 1);
            }
            // Add terms for total veicles constraint
            double val = 1;
            col.addTerms(&val, &constrs[N_SIZE - 2], 1);

            // Add terms for the limited memory rank 1 cuts
#if defined(SRC3) || defined(SRC)
            auto vec = cuts.computeLimitedMemoryCoefficients(label.route);
            // print vec size
            if (vec.size() > 0) {
                for (int i = 0; i < vec.size(); i++) {
                    if (vec[i] != 0) { col.addTerms(&vec[i], &SRCconstraints[i], 1); }
                }
            }

#endif

            // Collect bounds, costs, columns, and names
            lb.push_back(0.0);
            ub.push_back(1.0);
            obj.push_back(travel_cost);
            cols.push_back(col);
            names.push_back(name);
            vtypes.push_back(GRB_CONTINUOUS);

            Path path(label.route, label.cost);
            node->addPath(path);
        }
        // Add variables with bounds, objectives, and columns
        if (!lb.empty()) {
            node->addVars(lb.data(), ub.data(), obj.data(), vtypes.data(), names.data(), cols.data(), lb.size());
            node->update();
        }

        return counter;
    }

    /*
     * Adds a column to the GRBModel.
     *
     */
    inline int addColumn(BNBNode *node, const auto &columns, bool enumerate = false) {
        auto &r1c  = node->r1c;
        auto &cuts = r1c.cutStorage;
#ifdef RCC
        auto &rccManager = node->rccManager;
#endif

        int                 numConstrsLocal = node->get(GRB_IntAttr_NumConstrs);
        const GRBConstr    *constrs         = node->getConstrs();
        std::vector<double> coluna(numConstrsLocal, 0.0); // Declare outside the loop

        auto &matrix         = node->matrix;
        auto &allPaths       = node->paths;
        auto &SRCconstraints = node->SRCconstraints;

        // Collect the bounds, costs, names, and columns
        std::vector<double>      lb, ub, obj;
        std::vector<GRBColumn>   cols;
        std::vector<std::string> names;
        std::vector<char>        vtypes;

        auto &pathSet = node->pathSet;

        auto counter = 0;
        for (auto &label : columns) {
            if (label->nodes_covered.empty()) continue;
            if (!enumerate && label->cost > 0) continue;

            Path path(label->nodes_covered, label->real_cost);

            // TODO: check if its better to use a set or simply insert the path in the vector
            if (pathSet.find(path) != pathSet.end()) { continue; }
            pathSet.insert(path);

            counter += 1;
            if (counter >= 10) break;

            std::fill(coluna.begin(), coluna.end(), 0.0); // Reset for each iteration

            double      travel_cost = label->real_cost;
            std::string name        = "x[" + std::to_string(allPaths.size()) + "]";

            GRBColumn col;

            // Fill coluna with coefficients
            // Step 1: Accumulate the coefficients for each node
            for (const auto &node : label->nodes_covered) {
                if (likely(node > 0 && node != N_SIZE - 1)) { // Use likely() if this condition is almost always true
                    coluna[node - 1]++;                       // Simplified increment operation
                }
            }

            //  Add terms to GRBColumn
            for (int i = 0; i < N_SIZE - 2; i++) {
                if (coluna[i] == 0.0) continue;
                col.addTerms(&coluna[i], &constrs[i], 1);
            }
            // Add terms for total veicles constraint
            double val = 1;
            col.addTerms(&val, &constrs[N_SIZE - 2], 1);

            // Add terms for the limited memory rank 1 cuts
#if defined(SRC3) || defined(SRC)
            auto vec = cuts.computeLimitedMemoryCoefficients(label->nodes_covered);
            // print vec size
            if (vec.size() > 0) {
                for (int i = 0; i < vec.size(); i++) {
                    if (abs(vec[i]) > 1e-3) { col.addTerms(&vec[i], &SRCconstraints[i], 1); }
                }
            }
#endif

#if defined(RCC) || defined(EXACT_RCC)
            auto RCCvec         = rccManager.computeRCCCoefficients(label->nodes_covered);
            auto RCCconstraints = rccManager.getConstraints();
            if (RCCvec.size() > 0) {
                for (int i = 0; i < RCCvec.size(); i++) {
                    if (abs(RCCvec[i]) > 1e-3) { col.addTerms(&RCCvec[i], &RCCconstraints[i], 1); }
                }
            }
#endif

            // Collect bounds, costs, columns, and names
            lb.push_back(0.0);
            ub.push_back(1.0);
            obj.push_back(travel_cost);
            cols.push_back(col);
            names.push_back(name);
            vtypes.push_back(GRB_CONTINUOUS);

            node->addPath(path);
        }
        // Add variables with bounds, objectives, and columns
        if (!lb.empty()) {
            node->addVars(lb.data(), ub.data(), obj.data(), vtypes.data(), names.data(), cols.data(), lb.size());
            node->update();
        }

        return counter;
    }

    /**
     * Removes variables with negative reduced costs, up to a maximum of 30% of the total variables,
     * and also removes corresponding elements from the allPaths vector.
     *
     */
    void removeNegativeReducedCostVarsAndPaths(BNBNode *model) {
        auto &allPaths = model->paths;
        model->optimize();
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
     * Handles the cuts for the VRPTW problem.
     *
     */
    bool cutHandler(LimitedMemoryRank1Cuts &r1c, BNBNode *node, std::vector<GRBConstr> &constraints) {
        auto &cuts    = r1c.cutStorage;
        bool  changed = false;

        if (!cuts.empty()) {
            for (auto &cut : cuts) {
                if (cut.added && !cut.updated) { continue; }

                changed            = true;
                const int   z      = cut.id;
                const auto &coeffs = cut.coefficients;

                if (cut.added && cut.updated) {
                    cut.updated = false;
                    if (z >= constraints.size()) { continue; }
                    node->chgCoeff(constraints[z], coeffs);
                } else {
                    GRBLinExpr lhs;
                    for (size_t i = 0; i < coeffs.size(); ++i) {
                        if (coeffs[i] == 0) { continue; }
                        lhs += node->getVar(i) * coeffs[i];
                    }

                    std::string constraint_name = "cuts(z" + std::to_string(z) + ")";
                    // constraint_name << "cuts(z" << z << ")";

                    auto ctr = node->addConstr(lhs <= cut.rhs, constraint_name);
                    constraints.emplace_back(ctr);
                }

                cut.added   = true;
                cut.updated = false;
            }
        }

        node->update();
        return changed;
    }

#ifdef EXACT_RCC
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

        auto Ss = separate_Rounded_Capacity_cuts(model, instance.q, instance.demand, allPaths, solution);

        if (Ss.size() == 0) return false;

        // Define the tasks to be executed sequentially
        std::vector<GRBLinExpr>          cutExpressions(Ss.size());
        std::vector<int>                 rhsValues(Ss.size());
        std::vector<std::vector<RawArc>> arcGroups(Ss.size());

        for (size_t c = 0; c < Ss.size(); ++c) {
            auto S = Ss[c];

            GRBLinExpr          cutExpr = 0.0;
            std::vector<RawArc> arcs;

            // Generate all possible arc pairs from nodes in S
            for (int i : S) {
                for (int j : S) {
                    if (i != j) {

                        GRBLinExpr contributionSum = 0.0;
                        for (size_t ctr = 0; ctr < allPaths.size(); ++ctr) {
                            contributionSum += allPaths[ctr].timesArc(i, j) * model->getVar(ctr);
                        }
                        cutExpr += contributionSum;
                    }
                }
            }

            auto target_no_vehicles =
                ceil(std::accumulate(S.begin(), S.end(), 0, [&](int sum, int i) { return sum + instance.demand[i]; }) /
                     instance.q);

            cutExpressions[c] = cutExpr;
            rhsValues[c]      = S.size() - target_no_vehicles;
            arcGroups[c]      = std::move(arcs);
        }

        // std::vector<GRBConstr> newConstraints(cutExpressions.size());
        for (size_t i = 0; i < cutExpressions.size(); ++i) {
            auto ctr = model->addConstr(cutExpressions[i] <= rhsValues[i]);
            rccManager.addCut(arcGroups[i], rhsValues[i], ctr);
        }

        fmt::print("Added {} RCC cuts\n", cutExpressions.size());
        model->update();
        model->optimize();

        return true;
    }
#endif

#ifdef RCC
    /**
     * @brief Performs RCC (Robust Capacity Cuts) separation for the given model and solution.
     *
     * This function identifies and adds violated RCC cuts to the model to improve the solution.
     * It uses parallel execution to handle the separation and constraint addition efficiently.
     *
     */
    bool RCCsep(BNBNode *model, const std::vector<double> &solution) {

        auto &rccManager = model->rccManager;
        // Constraint manager to store cuts
        CnstrMgrPointer cutsCMP = nullptr;
        CMGR_CreateCMgr(&cutsCMP, 100);

        auto             nVertices = N_SIZE - 1;
        std::vector<int> demands   = instance.demand;

        // Precompute edge values from LP solution
        std::vector<std::vector<double>> aijs(N_SIZE + 1, std::vector<double>(N_SIZE + 1, 0.0));

        auto &allPaths   = model->paths;
        auto &oldCutsCMP = model->oldCutsCMP;

        for (int counter = 0; counter < solution.size(); ++counter) {
            const auto &nodes = allPaths[counter].route;
            for (size_t k = 1; k < nodes.size(); ++k) {
                int source = nodes[k - 1];
                int target = (nodes[k] == nVertices) ? 0 : nodes[k];
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
                    edgex.push_back((i == 0) ? nVertices : i);
                    edgey.push_back(j);
                    edgeval.push_back(xij);
                }
            }
        }

        // RCC Separation
        char   intAndFeasible;
        double maxViolation;
        CAPSEP_SeparateCapCuts(nVertices - 1, demands.data(), instance.q, edgex.size() - 1, edgex.data(), edgey.data(),
                               edgeval.data(), oldCutsCMP, 5, 1e-4, &intAndFeasible, &maxViolation, cutsCMP);

        if (intAndFeasible) return false; /* Optimal solution found */
        if (cutsCMP->Size == 0) return false;

        // print_cut("Found {} violated RCC cuts, max violation: {}\n", cutsCMP->Size, maxViolation);

        std::vector<int>                 rhsValues(cutsCMP->Size);
        std::vector<std::vector<RawArc>> arcGroups(cutsCMP->Size);

        // Instead of parallel execution, use a simple for-loop to process each cut
        for (int cutIdx = 0; cutIdx < cutsCMP->Size; ++cutIdx) {
            std::vector<int> S(cutsCMP->CPL[cutIdx]->IntList + 1,
                               cutsCMP->CPL[cutIdx]->IntList + cutsCMP->CPL[cutIdx]->IntListSize + 1);

            GRBLinExpr          cutExpr = 0.0;
            std::vector<RawArc> arcs;

            // Generate all possible arc pairs from nodes in S
            for (int i : S) {
                for (int j : S) {
                    if (i != j) { arcs.emplace_back(i, j); }
                }
            }

            rhsValues[cutIdx] = cutsCMP->CPL[cutIdx]->RHS;
            arcGroups[cutIdx] = std::move(arcs);
        }

        // Instead of parallel execution, use a simple for-loop to add each constraint
        for (std::size_t i = 0; i < rhsValues.size(); ++i) {
            GRBLinExpr cutExpr = 0.0;

            // For each arc in arcGroups[i], compute the cut expression
            for (const auto &arc : arcGroups[i]) {
                for (size_t ctr = 0; ctr < allPaths.size(); ++ctr) {
                    cutExpr += allPaths[ctr].timesArc(arc.from, arc.to) * model->getVar(ctr);
                }
            }

            // Add the constraint to the model
            auto ctr_name = "RCC_cut_" + std::to_string(rccManager.cut_ctr);
            auto ctr      = model->addConstr(cutExpr <= rhsValues[i], ctr_name);
            rccManager.addCut(arcGroups[i], rhsValues[i], ctr);
        }

        for (auto i = 0; i < cutsCMP->Size; i++) { CMGR_MoveCnstr(cutsCMP, oldCutsCMP, i, 0); }
        model->update();
        return true;
    }

#endif

    /**
     * Column generation algorithm.
     */
    bool CG(BNBNode *node, int max_iter = 2000) {
        print_info("Column generation preparation...\n");

        node->relaxNode();
        node->optimize();

        relaxed_result = std::numeric_limits<double>::max();

        // check if feasible
        if (node->get(GRB_IntAttr_Status) != GRB_OPTIMAL) {
            print_info("Model is infeasible, pruning node.\n");
            node->setPrune(true);
            return false;
        }

        auto &matrix         = node->matrix;
        auto &SRCconstraints = node->SRCconstraints;
        auto &allPaths       = node->paths;
        auto &r1c            = node->r1c;
        auto &branchingDuals = node->branchingDuals;

#ifdef RCC
        auto &rccManager = node->rccManager;
#endif

        int bucket_interval = 20;
        int time_horizon    = instance.T_max;

        numConstrs                = node->get(GRB_IntAttr_NumConstrs);
        node->numConstrs          = numConstrs;
        std::vector<double> duals = std::vector<double>(numConstrs, 0.0);

        // BucketGraph bucket_graph(nodes, time_horizon, bucket_interval, instance.q, bucket_interval);
        BucketGraph bucket_graph(nodes, time_horizon, bucket_interval);

        // print distance matrix size
        bucket_graph.set_distance_matrix(instance.getDistanceMatrix(), 8);
        bucket_graph.branching_duals = &branchingDuals;

        // node->optimize();
        matrix                 = node->extractModelDataSparse();
        auto integer_solution  = node->get(GRB_DoubleAttr_ObjVal);
        bucket_graph.incumbent = integer_solution;

        auto allNodes = bucket_graph.getNodes();

#ifdef SRC
        r1c              = LimitedMemoryRank1Cuts(allNodes);
        CutStorage *cuts = &r1c.cutStorage;
#endif

        std::vector<double> cutDuals;
        std::vector<double> nodeDuals = node->getDuals();
        auto                sizeDuals = nodeDuals.size();

        double lp_obj_dual = 0.0;
        double lp_obj      = node->get(GRB_DoubleAttr_ObjVal);

#ifdef SRC
        bucket_graph.cut_storage = cuts;
#endif

        bucket_graph.setup();

        double gap = 1e-6;

        bool ss    = false;
        int  stage = 1;

        bool TRstop = false;

        bool misprice = true;

        double lag_gap = std::numeric_limits<double>::max();

        auto                 inner_obj = 0.0;
        std::vector<Label *> paths;
        std::vector<double>  solution;
        bool                 can_add = true;

#ifdef STAB
        Stabilization stab(0.9, nodeDuals);
#endif
        bool changed = false;

        print_info("Starting column generation..\n\n");
        bool transition = false;

#ifdef TR
        std::vector<GRBVar> w(numConstrs);
        std::vector<GRBVar> zeta(numConstrs);

        auto epsilon1 = 10;
        auto epsilon2 = 10;

        // Create w_i and zeta_i variables for each constraint
        for (int i = 0; i < numConstrs; i++) {
            w[i]    = node->addVar(0, epsilon1, -1.0, GRB_CONTINUOUS, "w_" + std::to_string(i + 1));
            zeta[i] = node->addVar(0, epsilon2, 1.0, GRB_CONTINUOUS, "zeta_" + std::to_string(i + 1));
        }

        const GRBConstr *constrs = node->getConstrs();
        for (int i = 0; i < numConstrs; i++) {
            node->chgCoeff(constrs[i], w[i], -1);
            node->chgCoeff(constrs[i], zeta[i], 1);
        }
        double              v                   = 100;
        bool                isInsideTrustRegion = false;
        std::vector<double> delta1;
        std::vector<double> delta2;
        for (auto dual : nodeDuals) {
            delta1.push_back(dual - v);
            delta2.push_back(dual + v);
        }
        for (int i = 0; i < numConstrs; i++) {
            // Update coefficients for w[i] and zeta[i] in the objective function
            w[i].set(GRB_DoubleAttr_Obj, std::max(0.0, nodeDuals[i] - v)); // Set w[i]'s coefficient to -delta1
            zeta[i].set(GRB_DoubleAttr_Obj, nodeDuals[i] + v);             // Set zeta[i]'s coefficient to delta2
            w[i].set(GRB_DoubleAttr_UB, epsilon1);
            zeta[i].set(GRB_DoubleAttr_UB, epsilon2);
        }
        fmt::print("Starting trust region with v = {}\n", v);
#endif

#ifdef IPM
        IPSolver solver;
#endif

        bool   rcc         = false;
        bool   reoptimized = false;
        double obj;
        auto   colAdded = 0;

        for (int iter = 0; iter < max_iter; ++iter) {
            reoptimized = false;

#if defined(RCC) || defined(EXACT_RCC)
            if (rcc) {
                node->optimize();
                solution = node->extractSolution();
#ifdef RCC
                rcc = RCCsep(node, solution);
#endif
#ifdef EXACT_RCC
                rcc = exactRCCsep(node, solution);
#endif
                if (rcc) {
                    matrix = node->extractModelDataSparse();
                    node->optimize();
                }
                reoptimized = true;
            }
#endif

            if (ss && !rcc) {
#if defined(RCC) || defined(EXACT_RCC)
                if (!reoptimized) { node->optimize(); }
                solution = node->extractSolution();
#ifdef RCC
                rcc = RCCsep(node, solution);
#endif
#ifdef EXACT_RCC
                rcc = exactRCCsep(node, solution, cvrsep_ctrs);
#endif

                if (rcc) {
                    matrix = node->extractModelDataSparse();
                    node->optimize();
                    reoptimized = true;
                }

#endif

#if defined(SRC3) || defined(SRC)
                if (!rcc) {

                    // removeNegativeReducedCostVarsAndPaths(node);
                    node->optimize();

                    auto cuts_before = cuts->size();
                    ////////////////////////////////////////////////////
                    // Handle non-violated cuts in a single pass
                    ////////////////////////////////////////////////////
                    bool cleared        = false;
                    auto n_cuts_removed = 0;
                    // Iterate over the constraints in reverse order to remove non-violated cuts
                    for (int i = SRCconstraints.size() - 1; i >= 0; --i) {
                        GRBConstr constr = SRCconstraints[i];

                        // Get the slack value of the constraint
                        double slack = constr.get(GRB_DoubleAttr_Slack);

                        // If the slack is positive, it means the constraint is not violated
                        if (slack > 1e-3) {
                            cleared = true;

                            // Remove the constraint from the model and cut storage
                            node->remove(constr);
                            cuts->removeCut(cuts->getID(i));
                            n_cuts_removed++;

                            // Remove from SRCconstraints
                            SRCconstraints.erase(SRCconstraints.begin() + i);
                        }
                    }

                    if (cleared) {
                        node->update();                          // Update the model to reflect the removals
                        node->optimize();                        // Re-optimize the model
                        matrix = node->extractModelDataSparse(); // Extract model data
                    }

                    solution     = node->extractSolution();
                    r1c.allPaths = allPaths;
                    r1c.separate(matrix.A_sparse, solution);
#ifdef SRC
                    r1c.prepare45Heuristic(matrix.A_sparse, solution);
                    r1c.the45Heuristic<CutType::FourRow>(matrix.A_sparse, solution);
                    r1c.the45Heuristic<CutType::FiveRow>(matrix.A_sparse, solution);
#endif
                    if (cuts_before == cuts->size() + n_cuts_removed) {
                        print_info("No violations found, calling it a day\n");
                        break;
                    }

                    changed = cutHandler(r1c, node, SRCconstraints);
                    if (changed) {
                        matrix = node->extractModelDataSparse();
                        node->optimize();
                    }
                }
#endif
                bucket_graph.ss = false;
            }

#ifdef TR
            if (!TRstop) {
                isInsideTrustRegion = true;
                for (int i = 0; i < numConstrs; i++) {
                    if (nodeDuals[i] < delta1[i] || nodeDuals[i] > delta2[i]) { isInsideTrustRegion = false; }
                }
                // if (isInsideTrustRegion) { fmt::print("Fall inside trust region\n"); }
                if (isInsideTrustRegion) {
                    v -= 1;
                    // print v
                    fmt::print("Reducing v to {}\n", v);
                    epsilon1 += 100;
                    epsilon2 += 100;
                    for (int i = 0; i < numConstrs; i++) {
                        // Update coefficients for w[i] and zeta[i] in the objective function
                        w[i].set(GRB_DoubleAttr_Obj, nodeDuals[i]);        // Set w[i]'s coefficient to -delta1
                        zeta[i].set(GRB_DoubleAttr_Obj, nodeDuals[i] + v); // Set zeta[i]'s coefficient to delta2
                        w[i].set(GRB_DoubleAttr_UB, epsilon1);
                        zeta[i].set(GRB_DoubleAttr_UB, epsilon2);
                    }
                }
                for (int i = 0; i < numConstrs; i++) {
                    delta1[i] = nodeDuals[i] - v;
                    delta2[i] = nodeDuals[i] + v;
                }
                if (v <= 25) {
                    for (int i = 0; i < w.size(); i++) {
                        node->remove(w[i]);
                        node->remove(zeta[i]);
                    }
                    TRstop = true;
                    node->update();
                }
            }

#endif

#ifdef IPM
            auto d            = 0.5;
            auto matrixSparse = node->extractModelDataSparse();
            gap               = std::abs(lp_obj - (lp_obj + std::min(0.0, inner_obj))) / std::abs(lp_obj);
            gap               = gap / d;
            if (std::isnan(gap)) { gap = 1e-4; }
            if (std::signbit(gap)) { gap = 1e-4; }

            auto ip_result   = solver.run_optimization(matrixSparse, gap);
            lp_obj           = std::get<0>(ip_result);
            lp_obj_dual      = std::get<1>(ip_result);
            solution         = std::get<2>(ip_result);
            nodeDuals        = std::get<3>(ip_result);
            auto originDuals = nodeDuals;
            for (auto &dual : nodeDuals) { dual = -dual; }
#endif

#ifdef STAB
            stab.update_stabilization_after_master_optim(nodeDuals);

            misprice = true;
            while (misprice) {
                nodeDuals = stab.getStabDualSolAdvanced(nodeDuals);
                solution  = node->extractSolution();
#endif

                bool integer = true;
                // Check integrality of the solution
                for (auto &sol : solution) {
                    // NOTE: 1e-1 is not enough
                    if (sol > 1e-2 && sol < 1 - 1e-2) {
                        integer = false;
                        break;
                    }
                }
                if (integer) {
                    if (lp_obj < integer_solution) {
                        print_info("Updating integer solution to {}\n", lp_obj);
                        integer_solution       = lp_obj;
                        bucket_graph.incumbent = integer_solution;
                    }
                }

                bucket_graph.relaxation = lp_obj;
                bucket_graph.augment_ng_memories(solution, allPaths, true, 5, 100, 16, N_SIZE);

#if defined(SRC3) || defined(SRC)
                // SRC cuts
                if (!SRCconstraints.empty()) {
                    auto duals = node->get(GRB_DoubleAttr_Pi, SRCconstraints.data(), SRCconstraints.size());
                    cutDuals.assign(duals, duals + SRCconstraints.size());
                    cuts->setDuals(cutDuals);
                }
#endif

#ifdef RCC
                // RCC cuts
                if (rccManager.size() > 0) {
                    auto model     = node->getModel();
                    auto arc_duals = rccManager.computeDuals(model);
                    bucket_graph.setArcDuals(arc_duals);
                }
#endif

                // Branching duals
                if (branchingDuals.size() > 0) { branchingDuals.computeDuals(node->getModel()); }
                bucket_graph.setDuals(nodeDuals);

                //////////////////////////////////////////////////////////////////////
                // CALLING BALDES
                //////////////////////////////////////////////////////////////////////
                paths     = bucket_graph.solve();
                inner_obj = paths[0]->cost;
                stage     = bucket_graph.getStage();
                ss        = bucket_graph.ss;
                //////////////////////////////////////////////////////////////////////

                // Adding cols
                colAdded = addColumn(node, paths, false);

#ifdef SRC
                // Define rollback procedure
                if (bucket_graph.getStatus() == Status::Rollback) {
                    for (int i = SRCconstraints.size() - 1; i >= 0; --i) {
                        GRBConstr constr = SRCconstraints[i];
                        node->remove(constr);
                    }
                    node->SRCconstraints = std::vector<GRBConstr>();
                    node->update();
                    node->optimize();
                    cuts->reset();
                    // r1c.cutStorage = cuts;
                    matrix = node->extractModelDataSparse();
                }
#endif

#ifdef SCHRODINGER
                // Adding schrodinger paths
                auto sch_paths = bucket_graph.getSchrodinger();
                colAdded += addPath(node, sch_paths, true);
#endif

#ifdef RIH
                auto rih_paths = bucket_graph.get_rih_labels();
                colAdded += addColumn(node, rih_paths, true);

#endif

#ifdef STAB
                // TODO: check if we should update this before running the stab update
                node->optimize();
                lp_obj    = node->get(GRB_DoubleAttr_ObjVal);
                nodeDuals = node->getDuals();

                lag_gap          = integer_solution - (lp_obj + std::min(0.0, inner_obj));
                bucket_graph.gap = lag_gap;

                matrix = node->extractModelDataSparse();

                stab.update_stabilization_after_pricing_optim(matrix, nodeDuals, lag_gap, paths);

                if (colAdded == 0) {
                    stab.update_stabilization_after_misprice();
                    if (stab.shouldExit()) { misprice = false; }
                } else {
                    misprice = false;
                }
            }

            if (bucket_graph.getStatus() == Status::Optimal && stab.shouldExit()) {
                print_info("Optimal solution found\n");
                break;
            }

            stab.update_stabilization_after_iter(nodeDuals);
#endif

            auto cur_alpha  = 0.0;
            auto n_cuts     = 0;
            auto n_rcc_cuts = 0;

#ifdef STAB
            cur_alpha = stab.base_alpha;
#endif

#ifdef SRC
            n_cuts = cuts->size();
#endif

#ifdef RCC
            n_rcc_cuts = rccManager.size();
#endif
            if (iter % 50 == 0)
                fmt::print("| It.: {:4} | Obj.: {:8.2f} | Price: {:9.2f} | SRC: {:4} | RCC: {:4} | Paths: {:4} | "
                           "Stage: {:1} | "
                           "Lag.: {:10.4f} | alpha: {:4.2f} | \n",
                           iter, lp_obj, inner_obj, n_cuts, n_rcc_cuts, colAdded, stage, lag_gap, cur_alpha);
        }
        bucket_graph.print_statistics();

        node->optimize();
        relaxed_result = node->get(GRB_DoubleAttr_ObjVal);

        return true;
    }

    double objective(BNBNode *node) { return ip_result; }

    double bound(BNBNode *node) { return relaxed_result; }

    void evaluate(BNBNode *node) {
        auto start_timer = std::chrono::high_resolution_clock::now();
        auto cg          = CG(node);
        if (!cg) {
            relaxed_result = std::numeric_limits<double>::max();
            return;
        }
        auto end_timer        = std::chrono::high_resolution_clock::now();
        auto duration_ms      = std::chrono::duration_cast<std::chrono::milliseconds>(end_timer - start_timer).count();
        auto duration_seconds = duration_ms / 1000;
        auto duration_milliseconds = duration_ms % 1000;

        auto &allPaths = node->paths;

        // get solution in which x > 0.5 and print the corresponding allPaths
        std::vector<int> sol;
        for (int i = 0; i < allPaths.size(); i++) {
            if (node->getVar(i).get(GRB_DoubleAttr_X) > 0.5) { sol.push_back(i); }
        }

        for (auto s : sol) {
            fmt::print("Path: ");
            for (auto j : allPaths[s].route) { fmt::print("{} ", j); }
            fmt::print("\n");
        }
        fmt::print("\n");

        node->binarizeNode();
        node->optimize();

        // check if optimal
        if (node->get(GRB_IntAttr_Status) != GRB_OPTIMAL) {
            ip_result = std::numeric_limits<double>::max();
            // node->setPrune(true);
            print_info("No optimal solution found.\n");
        } else {
            ip_result = node->get(GRB_DoubleAttr_ObjVal);
        }

        // ANSI escape code for blue text
        constexpr auto blue  = "\033[34m";
        constexpr auto reset = "\033[0m";

        fmt::print("+----------------------+----------------+\n");
        fmt::print("| {:<14} | {}{:>20}{} |\n", "Bound", blue, relaxed_result, reset);
        fmt::print("| {:<14} | {}{:>20}{} |\n", "Incumbent", blue, ip_result, reset);
        fmt::print("| {:<14} | {}{:>16}.{:03}{} |\n", "CG Duration", blue, duration_seconds, duration_milliseconds,
                   reset);
        fmt::print("+----------------------+----------------+\n");
    }

    void branch(BNBNode *node) {
        fmt::print("\033[34m_STARTING BRANCH PROCEDURE \033[0m");
        fmt::print("\n");

        node->update();
        node->relaxNode();

        auto candidates       = Branching::VRPTWStandardBranching(node, &instance, this);
        auto candidateCounter = 0;

        print_info("Candidates generated: {}\n", candidates.size());
        for (auto &candidate : candidates) {

            if (node->hasCandidate(candidate)) continue;
            if (node->hasRaisedChild(candidate)) continue;

            auto candidatosNode = node->getCandidatos();
            auto childNode      = node->newChild();
            childNode->addCandidate(candidate);
            node->addRaisedChildren(candidate);

            candidateCounter++;
            if (candidateCounter >= NUMERO_CANDIDATOS) break;
        }

        fmt::print("\033[34m_FINISHED BRANCH PROCEDURE \033[0m");
        fmt::print("\n");
    }

    // implement clone method for virtual std::unique_ptr<Problem> clone() const = 0;
    std::unique_ptr<Problem> clone() const {
        auto newProblem      = std::make_unique<VRProblem>();
        newProblem->instance = instance;
        newProblem->nodes    = nodes;
        return newProblem;
    }

    /**
     * Column generation algorithm.
     */
    bool heuristicCG(BNBNode *node, int max_iter = 2000) {
        node->relaxNode();
        node->optimize();

        relaxed_result = std::numeric_limits<double>::max();

        // check if feasible
        if (node->get(GRB_IntAttr_Status) != GRB_OPTIMAL) {
            node->setPrune(true);
            return false;
        }

        auto &matrix         = node->matrix;
        auto &allPaths       = node->paths;
        auto &branchingDuals = node->branchingDuals;

        int bucket_interval = 20;
        int time_horizon    = instance.T_max;

        numConstrs                = node->get(GRB_IntAttr_NumConstrs);
        node->numConstrs          = numConstrs;
        std::vector<double> duals = std::vector<double>(numConstrs, 0.0);

        // BucketGraph bucket_graph(nodes, time_horizon, bucket_interval, instance.q, bucket_interval);
        BucketGraph bucket_graph(nodes, time_horizon, bucket_interval);

        // print distance matrix size
        bucket_graph.set_distance_matrix(instance.getDistanceMatrix(), 8);
        bucket_graph.branching_duals = &branchingDuals;

        // node->optimize();
        matrix                 = node->extractModelDataSparse();
        auto integer_solution  = node->get(GRB_DoubleAttr_ObjVal);
        bucket_graph.incumbent = integer_solution;

        auto allNodes = bucket_graph.getNodes();

        std::vector<double> cutDuals;
        std::vector<double> nodeDuals = node->getDuals();
        auto                sizeDuals = nodeDuals.size();

        double lp_obj_dual = 0.0;
        double lp_obj      = node->get(GRB_DoubleAttr_ObjVal);

        bucket_graph.setup();

        double gap = 1e-6;

        bool ss    = false;
        int  stage = 1;

        bool TRstop = false;

        bool misprice = true;

        double lag_gap = std::numeric_limits<double>::max();

        auto                 inner_obj = 0.0;
        std::vector<Label *> paths;
        std::vector<double>  solution;
        bool                 can_add = true;

#ifdef STAB
        Stabilization stab(0.9, nodeDuals);
#endif
        bool changed = false;

        bool transition = false;

        bool   rcc         = false;
        bool   reoptimized = false;
        double obj;
        auto   colAdded = 0;

        auto &r1c = node->r1c;
#ifdef RCC
        auto &rccManager = node->rccManager;
#endif

#ifdef SRC
        r1c              = LimitedMemoryRank1Cuts(allNodes);
        CutStorage *cuts = &r1c.cutStorage;
#endif

#ifdef SRC
        bucket_graph.cut_storage = cuts;
#endif

        for (int iter = 0; iter < max_iter; ++iter) {
            reoptimized = false;

#ifdef STAB
            stab.update_stabilization_after_master_optim(nodeDuals);
            misprice = true;
            while (misprice) {
                nodeDuals = stab.getStabDualSolAdvanced(nodeDuals);
                solution  = node->extractSolution();
#endif

                bool integer = true;
                // Check integrality of the solution
                for (auto &sol : solution) {
                    // NOTE: 1e-1 is not enough
                    if (sol > 1e-2 && sol < 1 - 1e-2) {
                        integer = false;
                        break;
                    }
                }
                if (integer) {
                    if (lp_obj < integer_solution) {
                        print_info("Updating integer solution to {}\n", lp_obj);
                        integer_solution       = lp_obj;
                        bucket_graph.incumbent = integer_solution;
                    }
                }

                bucket_graph.relaxation = lp_obj;
                bucket_graph.augment_ng_memories(solution, allPaths, true, 5, 100, 16, N_SIZE);

                // Branching duals
                if (branchingDuals.size() > 0) { branchingDuals.computeDuals(node->getModel()); }
                bucket_graph.setDuals(nodeDuals);

                //////////////////////////////////////////////////////////////////////
                // CALLING BALDES
                //////////////////////////////////////////////////////////////////////
                paths     = bucket_graph.solveHeuristic();
                inner_obj = paths[0]->cost;
                stage     = bucket_graph.getStage();
                ss        = bucket_graph.ss;
                //////////////////////////////////////////////////////////////////////

                // Adding cols
                colAdded = addColumn(node, paths, false);

#ifdef STAB
                // TODO: check if we should update this before running the stab update
                node->optimize();
                lp_obj    = node->get(GRB_DoubleAttr_ObjVal);
                nodeDuals = node->getDuals();

                lag_gap          = integer_solution - (lp_obj + std::min(0.0, inner_obj));
                bucket_graph.gap = lag_gap;

                matrix = node->extractModelDataSparse();

                stab.update_stabilization_after_pricing_optim(matrix, nodeDuals, lag_gap, paths);

                if (colAdded == 0) {
                    stab.update_stabilization_after_misprice();
                    if (stab.shouldExit()) { misprice = false; }
                } else {
                    misprice = false;
                }
            }

            if (bucket_graph.getStatus() == Status::Optimal && stab.shouldExit()) {
                print_info("Optimal solution found\n");
                break;
            }

            stab.update_stabilization_after_iter(nodeDuals);
#endif
        }
        // bucket_graph.print_statistics();

        node->optimize();
        relaxed_result = node->get(GRB_DoubleAttr_ObjVal);

        return true;
    }
};
