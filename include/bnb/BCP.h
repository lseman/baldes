/**
 * @file BCP.h
 * @brief Header file for the BCP class.
 *
 * This file contains the definition of the BCP class, which represents the
 * Branch-Cut-and-Price (BCP) algorithm. The class provides methods to solve the
 * Vehicle Routing Problem (VRP) using a column generation approach with
 * stabilization and rank-1 cuts. The BCP algorithm iteratively solves the
 * problem by generating columns, adding cuts, and branching on the resulting
 * nodes.
 *
 */

#pragma once

#include "Arc.h"
#include "Cut.h"
#include "Definitions.h"
#include "RCC.h"
#include "VRPNode.h"
#include "miphandler/LinExp.h"
#include "miphandler/MIPHandler.h"

// #include "TR.h"
#include "bnb/Node.h"
#include "bucket/BucketGraph.h"
#include "bucket/BucketSolve.h"
#include "bucket/BucketUtils.h"
#include "solvers/Gurobi.h"

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

#if defined(IPM) || defined(IPM_ACEL)
#include "ipm/IPSolver.h"
#endif

#ifdef TR
#include "TR.h"
#endif

#include "Logger.h"

#define NUMERO_CANDIDATOS 10

#include "RIH.h"

class VRProblem {
   public:
    InstanceData instance;
    std::vector<VRPNode> nodes;
    double ip_result = std::numeric_limits<double>::max();
    double relaxed_result = std::numeric_limits<double>::max();

    ProblemType problemType = ProblemType::vrptw;

    int numConstrs = 0;

#ifdef EXACT_RCC
    RCCManager rccManager;
#endif

    int addPath(BNBNode *node, const std::vector<Path> paths,
                bool enumerate = false) {
        SRC_MODE_BLOCK(auto &r1c = node->r1c; auto &cuts = r1c->cutStorage;)
        RCC_MODE_BLOCK(auto &rccManager = node->rccManager;)
        int numConstrsLocal = node->getIntAttr("NumConstrs");
        auto &constrs = node->getConstrs();

        auto &matrix = node->matrix;
        auto &allPaths = node->paths;
        auto &SRCconstraints = node->SRCconstraints;

        std::vector<double> coluna(numConstrsLocal,
                                   0.0);  // Declare outside the loop

        // Collect the bounds, costs, names, and columns
        std::vector<double> lb, ub, obj;
        std::vector<MIPColumn> cols;
        std::vector<std::string> names;
        std::vector<VarType> vtypes;

        auto counter = 0;
        for (auto &label : paths) {
            if (label.route.empty()) continue;
            counter += 1;
            if (counter > 10) break;
            // Insert the path into the set to avoid duplicates
            // pathSet.insert(path);

            std::fill(coluna.begin(), coluna.end(),
                      0.0);  // Reset the coefficients

            double travel_cost = label.cost;
            std::string name = "x[" + std::to_string(allPaths.size()) + "]";

            MIPColumn col;  // Use Column instead of GRBColumn

            // Step 1: Accumulate the coefficients for each node
            for (const auto &node : label.route) {
                if (likely(node > 0 &&
                           node != N_SIZE - 1)) {  // Use likely() if applicable
                    coluna[node - 1]++;  // Simplified increment operation
                }
            }

            // Add terms to the Column
            for (int i = 0; i < N_SIZE - 2; i++) {
                if (coluna[i] == 0.0) continue;
                col.addTerm(i, coluna[i]);  // Add term to the Column (row index
                                            // and coefficient)
            }

            // Add the term for total vehicles constraint
            col.addTerm(N_SIZE - 2,
                        1.0);  // Add term for the vehicle constraint

            // Add terms for the limited memory rank 1 cuts
            SRC_MODE_BLOCK(
                auto vec = cuts.computeLimitedMemoryCoefficients(label.route);
                if (vec.size() > 0) {
                    for (int i = 0; i < vec.size(); i++) {
                        if (abs(vec[i]) > 1e-3) {
                            col.addTerm(SRCconstraints[i]->index(),
                                        vec[i]);  // Add to the appropriate
                                                  // constraint in Column
                        }
                    }
                })

#if defined(RCC)
            auto RCCvec = rccManager->computeRCCCoefficients(label.route);
            auto RCCconstraints = rccManager->getbaldesCtrs();
            if (RCCvec.size() > 0) {
                for (int i = 0; i < RCCvec.size(); i++) {
                    if (abs(RCCvec[i]) > 1e-3) {
                        col.addTerm(RCCconstraints[i]->index(),
                                    RCCvec[i]);  // Add RCC terms to the Column
                    }
                }
            }
#endif

            // Collect bounds, costs, columns, and names
            lb.push_back(0.0);
            ub.push_back(1.0);
            obj.push_back(travel_cost);
            cols.push_back(col);  // Add the populated Column
            names.push_back(name);
            vtypes.push_back(
                VarType::Continuous);  // Assuming VarType::Continuous is the
                                       // equivalent of GRB_CONTINUOUS

            node->addPath(label);  // Store the path
        }

        // Add variables with bounds, objectives, and columns
        if (!lb.empty()) {
            // Pass the data to the MIP problem
            node->addVars(lb.data(), ub.data(), obj.data(), vtypes.data(),
                          names.data(), cols.data(), lb.size());
        }

        return counter;
    }

    /*
     * Adds a column to the GRBModel.
     *
     */
    inline int addColumn(BNBNode *node, const auto &columns, double &inner_obj,
                         bool enumerate = false) {
        SRC_MODE_BLOCK(auto &r1c = node->r1c; auto &cuts = r1c->cutStorage;)
        RCC_MODE_BLOCK(auto &rccManager = node->rccManager;)
        int numConstrsLocal = node->getIntAttr("NumConstrs");
        auto &constrs = node->getConstrs();
        std::vector<double> coluna(numConstrsLocal,
                                   0.0);  // Declare outside the loop

        auto &allPaths = node->paths;
        auto &SRCconstraints = node->SRCconstraints;

        // Collect the bounds, costs, names, and columns
        std::vector<double> lb, ub, obj;
        std::vector<MIPColumn> cols;
        std::vector<std::string> names;
        std::vector<VarType> vtypes;

        auto &pathSet = node->pathSet;
        inner_obj = 0.0;

        auto counter = 0;
        for (auto &label : columns) {
            if (label->nodes_covered.empty()) continue;
            if (!enumerate && label->cost > 0) continue;

            Path path(label->getRoute(), label->real_cost);

            // check if path is already in pathSet
            if (pathSet.find(path) != pathSet.end()) continue;
            // Insert the path into the set to avoid duplicates
            pathSet.insert(path);

            if (label->cost < inner_obj) {
                inner_obj = label->cost;
            }

            counter += 1;
            if (counter > N_ADD - 1) break;

            std::fill(coluna.begin(), coluna.end(),
                      0.0);  // Reset the coefficients

            double travel_cost = label->real_cost;
            std::string name = "x[" + std::to_string(allPaths.size()) + "]";

            MIPColumn col;  // Use Column instead of GRBColumn

            // Step 1: Accumulate the coefficients for each node
            for (const auto &node : label->nodes_covered) {
                if (likely(node > 0 &&
                           node != N_SIZE - 1)) {  // Use likely() if applicable
                    coluna[node - 1]++;  // Simplified increment operation
                }
            }

            // Add terms to the Column
            for (int i = 0; i < N_SIZE - 2; i++) {
                if (coluna[i] == 0.0) continue;
                col.addTerm(i, coluna[i]);  // Add term to the Column (row index
                                            // and coefficient)
            }

            // Add the term for total vehicles constraint
            col.addTerm(N_SIZE - 2,
                        1.0);  // Add term for the vehicle constraint

            // Add terms for the limited memory rank 1 cuts
            SRC_MODE_BLOCK(
                auto vec = cuts.computeLimitedMemoryCoefficients(path.route);
                if (vec.size() > 0) {
                    for (int i = 0; i < vec.size(); i++) {
                        if (abs(vec[i]) > 1e-3) {
                            col.addTerm(SRCconstraints[i]->index(), vec[i]);
                        }
                    }
                })

            RCC_MODE_BLOCK(
                auto RCCvec = rccManager->computeRCCCoefficients(path.route);
                if (RCCvec.size() > 0) {
                    auto RCCconstraints = rccManager->getbaldesCtrs();

                    for (int i = 0; i < RCCvec.size(); i++) {
                        if (abs(RCCvec[i]) > 1e-3) {
                            col.addTerm(
                                RCCconstraints[i]->index(),
                                RCCvec[i]);  // Add RCC terms to the Column
                        }
                    }
                })

            auto &branching = node->branchingDuals;
            auto branchingVec = branching->computeCoefficients(path.route);
            if (branchingVec.size() > 0) {
                auto branchingbaldesCtrs = branching->getBranchingbaldesCtrs();
                for (int i = 0; i < branchingVec.size(); i++) {
                    if (abs(branchingVec[i]) > 1e-3) {
                        col.addTerm(branchingbaldesCtrs[i]->index(),
                                    branchingVec[i]);
                    }
                }
            }

            // Collect bounds, costs, columns, and names
            lb.push_back(0.0);
            ub.push_back(1.0);
            obj.push_back(travel_cost);
            cols.push_back(col);  // Add the populated Column
            names.push_back(name);
            vtypes.push_back(
                VarType::Continuous);  // Assuming VarType::Continuous is the
                                       // equivalent of GRB_CONTINUOUS

            node->addPath(path);  // Store the path
        }

        // Add variables with bounds, objectives, and columns
        if (!lb.empty()) {
            // Pass the data to the MIP problem
            node->addVars(lb.data(), ub.data(), obj.data(), vtypes.data(),
                          names.data(), cols.data(), lb.size());
            node->update();
        }

        return counter;
    }

    /**
     * Handles the cuts for the VRPTW problem.
     *
     */
    bool cutHandler(std::shared_ptr<LimitedMemoryRank1Cuts> &r1c, BNBNode *node,
                    std::vector<baldesCtrPtr> &constraints) {
        auto &cuts = r1c->cutStorage;
        bool changed = false;

        if (!cuts.empty()) {
            for (auto &cut : cuts) {
                if (cut.added && !cut.updated) {
                    continue;
                }

                changed = true;
                const int z = cut.id;
                const auto &coeffs = cut.coefficients;

                if (cut.added && cut.updated) {
                    cut.updated = false;
                    // if (z >= constraints.size()) { continue; }
                    node->chgCoeff(constraints[z], coeffs);
                } else {
                    LinearExpression lhs;
                    for (size_t i = 0; i < coeffs.size(); ++i) {
                        if (coeffs[i] == 0) {
                            continue;
                        }
                        lhs += node->getVar(i) * coeffs[i];
                    }

                    std::string constraint_name = "SRC_" + std::to_string(z);
                    // constraint_name << "cuts(z" << z << ")";
                    // auto ctr = ;
                    // print cut.rhs
                    auto ctr = node->addConstr(lhs <= cut.rhs, constraint_name);
                    constraints.emplace_back(ctr);
                }

                cut.added = true;
                cut.updated = false;
            }
        }
        // print constraints size
        return changed;
    }

#ifdef RCC
    /**
     * @brief Performs RCC (Robust Capacity Cuts) separation for the given model
     * and solution.
     *
     * This function identifies and adds violated RCC cuts to the model to
     * improve the solution. It uses parallel execution to handle the separation
     * and constraint addition efficiently.
     *
     */
    bool RCCsep(BNBNode *model, const std::vector<double> &solution) {
        auto &rccManager = model->rccManager;
        // if (rccManager->cut_ctr >= 50) return false;
        //  baldesCtr manager to store cuts
        CnstrMgrPointer cutsCMP = nullptr;
        CMGR_CreateCMgr(&cutsCMP, 100);

        auto nVertices = N_SIZE - 1;
        std::vector<int> demands = instance.demand;

        // Precompute edge values from LP solution
        std::vector<std::vector<double>> aijs(
            N_SIZE + 1, std::vector<double>(N_SIZE + 1, 0.0));

        auto &allPaths = model->paths;
        auto &oldCutsCMP = model->oldCutsCMP;

        for (int counter = 0; counter < allPaths.size(); ++counter) {
            const auto &nodes = allPaths[counter].route;
            for (size_t k = 1; k < nodes.size(); ++k) {
                int source = nodes[k - 1];
                int target = (nodes[k] == nVertices) ? 0 : nodes[k];
                aijs[source][target] += solution[counter];
            }
        }

        std::vector<int> edgex, edgey;
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
        char intAndFeasible;
        double maxViolation;
        int maxCuts = 5;
        if (problemType == ProblemType::cvrp) {
            maxCuts = 10;
        }

        CAPSEP_SeparateCapCuts(nVertices - 1, demands.data(), instance.q,
                               edgex.size() - 1, edgex.data(), edgey.data(),
                               edgeval.data(), oldCutsCMP, maxCuts, 1e-4,
                               &intAndFeasible, &maxViolation, cutsCMP);

        // print_cut("Found {} violated RCC cuts, max violation: {}\n",
        // cutsCMP->Size, maxViolation);

        if (intAndFeasible) return false; /* Optimal solution found */
        if (cutsCMP->Size == 0) return false;

        std::vector<int> rhsValues(cutsCMP->Size);
        std::vector<std::vector<RawArc>> arcGroups(cutsCMP->Size);

        // Instead of parallel execution, use a simple for-loop to process each
        // cut
        for (int cutIdx = 0; cutIdx < cutsCMP->Size; ++cutIdx) {
            std::vector<int> S(cutsCMP->CPL[cutIdx]->IntList + 1,
                               cutsCMP->CPL[cutIdx]->IntList +
                                   cutsCMP->CPL[cutIdx]->IntListSize + 1);

            std::vector<RawArc> arcs;

            // Generate all possible arc pairs from nodes in S
            for (int i : S) {
                for (int j : S) {
                    if (i != j) {
                        arcs.emplace_back(i, j);
                    }
                }
            }

            rhsValues[cutIdx] = cutsCMP->CPL[cutIdx]->RHS;
            arcGroups[cutIdx] = std::move(arcs);
        }

        // Instead of parallel execution, use a simple for-loop to add each
        // constraint
        for (std::size_t i = 0; i < rhsValues.size(); ++i) {
            LinearExpression cutExpr;

            // For each arc in arcGroups[i], compute the cut expression
            for (const auto &arc : arcGroups[i]) {
                for (size_t ctr = 0; ctr < allPaths.size(); ++ctr) {
                    cutExpr += model->getVar(ctr) *
                               allPaths[ctr].timesArc(arc.from, arc.to);
                }
            }

            // Add the constraint to the model
            auto ctr_name = "RCC_" + std::to_string(rccManager->cut_ctr);
            auto ctr = cutExpr <= rhsValues[i];
            ctr = model->addConstr(ctr, ctr_name);
            rccManager->addCut(arcGroups[i], rhsValues[i], ctr);
        }

        for (auto i = 0; i < cutsCMP->Size; i++) {
            CMGR_MoveCnstr(cutsCMP, oldCutsCMP, i, 0);
        }
        return true;
    }

#endif

    /**
     * Column generation algorithm.
     */
    bool CG(BNBNode *node, int max_iter = 5000) {
        print_info("Column generation preparation...\n");

        node->relaxNode();
        node->optimize();
        relaxed_result = std::numeric_limits<double>::max();

        // check if feasible
        if (node->getStatus() != 2) {
            print_info("Model is infeasible, pruning node.\n");
            node->setPrune(true);
            return false;
        }

        auto &matrix = node->matrix;
        auto &SRCconstraints = node->SRCconstraints;
        auto &allPaths = node->paths;

        SRC_MODE_BLOCK(auto &r1c = node->r1c;)
        auto &branchingDuals = node->branchingDuals;
        RCC_MODE_BLOCK(auto &rccManager = node->rccManager;)

        int bucket_interval = 20;
        int time_horizon = instance.T_max;
        // if (problemType == ProblemType::cvrp) { time_horizon = 50000; }
        numConstrs = node->getIntAttr("NumConstrs");
        node->numConstrs = numConstrs;
        std::vector<double> duals = std::vector<double>(numConstrs, 0.0);

        std::unique_ptr<BucketGraph> bucket_graph;

        if (problemType == ProblemType::vrptw) {
            bucket_graph = std::make_unique<BucketGraph>(nodes, time_horizon,
                                                         bucket_interval);
        } else if (problemType == ProblemType::cvrp) {
            bucket_graph = std::make_unique<BucketGraph>(nodes, instance.q,
                                                         bucket_interval);
        }

        if (problemType == ProblemType::cvrp) {
            BucketOptions options;
            options.main_resources = {0};
            options.resources = {"capacity"};
            options.resource_type = {1};
            bucket_graph->options = options;
        }

        bucket_graph->set_distance_matrix(instance.getDistanceMatrix(), 8);
        bucket_graph->branching_duals = branchingDuals;
        bucket_graph->A_MAX = N_SIZE;
        bucket_graph->depth = node->depth;

        bucket_graph->topHeurRoutes = node->bestRoutes;

        matrix = node->extractModelDataSparse();
        auto integer_solution = node->integer_sol;
        bucket_graph->incumbent = integer_solution;

        SRC_MODE_BLOCK(
            r1c->setDistanceMatrix(node->instance.getDistanceMatrix());
            r1c->setNodes(nodes); CutStorage *cuts = &r1c->cutStorage;
            bucket_graph->cut_storage = cuts;)

        std::vector<double> cutDuals;
        std::vector<double> nodeDuals = node->getDuals();
        bucket_graph->setDuals(nodeDuals);
        auto sizeDuals = nodeDuals.size();

        double lp_obj_dual = 0.0;
        double lp_obj = node->getObjVal();
        double lp_obj_old = lp_obj;
        int iter_non_improv = 0;

        bucket_graph->setup();

        double gap = 1e-6;
        bool ss = false;
        int stage = 1;
        bool misprice = true;
        double lag_gap = 0.0;

        auto inner_obj = 0.0;
        std::vector<Label *> paths;
        std::vector<double> solution = node->extractSolution();
        bool can_add = true;

#ifdef STAB
        Stabilization stab(0.5, nodeDuals);
#endif

#ifdef RIH
        IteratedLocalSearch ils(instance);
        bucket_graph->ils = &ils;
        SRC_MODE_BLOCK(ils.cut_storage = cuts;)

#endif

        bool changed = false;
        print_info("Starting column generation..\n\n");
        bool transition = false;

#ifdef TR
        bool TRstop = false;
        TrustRegion tr(numConstrs);
        tr.setup(node, nodeDuals);
        solution = node->extractSolution();
        double v = 0;
#endif

#ifdef IPM_ACEL
        int base_threshold = 20;
        int adaptive_threshold;
        IPSolver ipm_solver;
        bool use_ipm_duals = false;
#endif
        bool rcc = false;
        bool reoptimized = false;
        double obj;
        auto colAdded = 0;
        std::vector<double> originDuals;
        bool force_cuts = false;
        int numK;
        for (int iter = 0; iter < max_iter; ++iter) {
            reoptimized = false;

#if defined(RCC)
            if (rcc) {
                RUN_OPTIMIZATION(node, 1e-6)
                RCC_MODE_BLOCK(rcc = RCCsep(node, solution);)

#ifdef EXACT_RCC
                rcc = exactRCCsep(node, solution);
#endif
            }
#endif

            if ((ss && !rcc) || force_cuts) {
                force_cuts = false;
#if defined(RCC)
                RUN_OPTIMIZATION(node, 1e-8)
                RCC_MODE_BLOCK(rcc = RCCsep(node, solution);)
#endif

                SRC_MODE_BLOCK(if (!rcc) {
                    r1c->allPaths = allPaths;
                    RCC_MODE_BLOCK(
                        // RCC cuts
                        if (rccManager->size() > 0) {
#ifndef IPM
                            auto arc_duals = rccManager->computeDuals(node);
#else
                    auto arc_duals = rccManager->computeDuals(nodeDuals, node);
#endif
                            r1c->setArcDuals(arc_duals);
                        })
                    auto srcResult = r1c->runSeparation(node, SRCconstraints);
                    bool violated = srcResult.first;
                    bool cleared = srcResult.second;
                    if (!violated) {
                        if (bucket_graph->A_MAX == N_SIZE) {
                            if (std::abs(inner_obj) < 1.0) {
                                print_info(
                                    "No violated cuts found, calling it a "
                                    "day\n");
                                break;
                            }
                        } else {
                            auto new_relaxation =
                                std::min(bucket_graph->A_MAX + 10, N_SIZE);
                            print_info("Increasing A_MAX to {}\n",
                                       new_relaxation);
                            bucket_graph->A_MAX = new_relaxation;
                        }

                    } else {
                        changed = cutHandler(r1c, node, SRCconstraints);
                        if (changed) {
#ifndef IPM
                            stab.cut_added = true;
                            node->optimize();
#endif
                        }
                    }
                })
                bucket_graph->ss = false;
            }

#ifdef TR
            if (!TRstop) {
                v = tr.iterate(node, nodeDuals, inner_obj,
                               bucket_graph->getStage());
                TRstop = tr.stop();
            }
#endif

#ifdef IPM
#if defined(STAB) && defined(IPM)
            if (stage >= 4) {
#endif
                double d = 1;
                matrix = node->extractModelDataSparse();
                double obj_change = std::abs(lp_obj - lp_obj_old);
                // fmt::print("Objective change: {}\n", obj_change);
                // double adaptive_factor = std::min(1.0, std::max(1e-4,
                // obj_change / std::abs(lp_obj + 1e-6))); fmt::print("Adaptive
                // factor: {}\n", adaptive_factor);
                numK = std::ceil(
                    std::accumulate(solution.begin(), solution.end(), 0.0));

                // Compute gap based on current objective difference and
                // adaptive factor
                gap = std::abs(lp_obj - (lp_obj + numK * inner_obj)) /
                      std::abs(lp_obj + 1e-6);
                gap = (gap / d);
                // fmt::print("Gap: {}\n", gap);

                // Enforce upper and lower bounds on gap
                if (std::isnan(gap) || std::signbit(gap)) {
                    gap = 1e-1;
                }
                gap = std::clamp(
                    gap, 1e-8,
                    1e-1);  // Clamping gap to be between 1e-6 and 1e-2
                node->ipSolver->run_optimization(matrix, gap);

                lp_obj_old = lp_obj;
                lp_obj = node->ipSolver->getObjective();
                solution = node->ipSolver->getPrimals();
                nodeDuals = node->ipSolver->getDuals();
                originDuals = nodeDuals;
                // print origin duals size
                for (auto &dual : nodeDuals) {
                    dual = -dual;
                }
#endif
#if defined(STAB) && !defined(IPM)
                nodeDuals = node->getDuals();
                auto originDuals = nodeDuals;
#endif

#if defined(STAB) && defined(IPM)
            }
#endif

            auto updateGraphAndSolve = [&](auto &nodeDuals) {
                bucket_graph->relaxation = lp_obj;
                bucket_graph->augment_ng_memories(solution, allPaths, true, 5,
                                                  100, 30, N_SIZE);

                SRC_MODE_BLOCK(  // SRC cuts
                    if (!SRCconstraints.empty()) {
                        // print SRCconstraints size
                        std::vector<double> cutDuals;
                        cutDuals.reserve(SRCconstraints.size());

                        for (int i = 0; i < SRCconstraints.size(); i++) {
                            auto constr = SRCconstraints[i];
                            auto index =
                                constr
                                    ->index();  // node->get_current_index(constr->get_unique_id());
                            // print size of originDuals
                            cutDuals.push_back(originDuals[index]);
                        }

                        cuts->setDuals(cutDuals);
                    })

                RCC_MODE_BLOCK(
                    // RCC cuts
                    if (rccManager->size() > 0) {
#ifndef IPM
                        auto arc_duals = rccManager->computeDuals(node);
#else
                        auto arc_duals =
                            rccManager->computeDuals(nodeDuals, node);
#endif
                        bucket_graph->setArcDuals(arc_duals);
                    })

                // Branching duals
                if (branchingDuals->size() > 0) {
                    branchingDuals->computeDuals(node);
                }
                bucket_graph->setDuals(nodeDuals);

                r1c->setDuals(nodeDuals);

                //////////////////////////////////////////////////////////////////////
                // CALLING BALDES
                //////////////////////////////////////////////////////////////////////
                paths = bucket_graph->solve();
                // inner_obj = bucket_graph->inner_obj;
                stage = bucket_graph->getStage();
                ss = bucket_graph->ss;
                //////////////////////////////////////////////////////////////////////

                // Adding cols
                colAdded = addColumn(node, paths, inner_obj, false);

#ifdef RIH
                // Adding RIH paths
                auto rih_paths = ils.get_labels();
                if (rih_paths.size() > 0) {
                    double inner_obj_rih = 0.0;
                    auto rih_added =
                        addColumn(node, rih_paths, inner_obj_rih, true);
                    // colAdded += rih_added;
                }
#endif

#ifdef SCHRODINGER
                // Adding schrodinger paths
                auto sch_paths = bucket_graph->getSchrodinger();
                colAdded += addPath(node, sch_paths, true);
#endif
            };

            /////////////////////////////////////////////////////////
            // Solve with Stabilization
            /////////////////////////////////////////////////////////

#if defined(STAB) && defined(IPM)
            if (stage >= 4) {
                updateGraphAndSolve();
            }
#endif

#ifdef STAB
#if defined(STAB) && defined(IPM)
            if (stage <= 3) {
#endif

                // define numK as the number of non-zero vars in solution
                numK = std::ceil(
                    std::accumulate(solution.begin(), solution.end(), 0.0));
                double newBound = lp_obj + numK * inner_obj;
                lag_gap = integer_solution - newBound;
                bucket_graph->gap = lag_gap;
                stab.set_pseudo_dual_bound(newBound);
                stab.updateNumK(numK);
                stab.update_stabilization_after_master_optim(nodeDuals);
                stab.setObj(lp_obj);
                nodeDuals = stab.getStabDualSolAdvanced(nodeDuals);

                misprice = true;
                while (misprice) {
                    solution = node->extractSolution();
#endif

                    bool integer = true;
#ifdef TR
                    if (TRstop)
#endif
                    {
                        for (auto sol : solution) {
                            if (sol > 1e-1 && sol < 1 - 1e-1) {
                                integer = false;
                                break;
                            }
                        }
                        // check if the lp_obj itself is within tolerance
                        if (std::abs(std::round(lp_obj) - lp_obj) > 1e-2) {
                            integer = false;
                        }

                        if (integer) {
                            if (std::round(lp_obj) < integer_solution) {
                                // print_info("Updating integer solution to
                                // {}\n", std::round(lp_obj));
                                integer_solution = std::round(lp_obj);
                                bucket_graph->incumbent = integer_solution;
#ifdef STAB
                                stab.clearAlpha();
#endif
                            }
                        }
                    }
#ifdef IPM_ACEL
                    auto updateGapAndRunOptimization =
                        [&](auto node, auto lp_obj, auto inner_obj,
                            auto &ipm_solver, auto &iter_non_improv,
                            auto &use_ipm_duals, auto &nodeDuals) {
                            auto matrix = node->extractModelDataSparse();

                            auto d = 10;
                            // Compute gap based on current objective difference
                            // and adaptive factor

                            double gap =
                                std::abs(lp_obj - (lp_obj + numK * inner_obj)) /
                                std::abs(lp_obj + 1e-6);
                            gap = (gap / d);

                            // Enforce upper and lower bounds on gap
                            if (std::isnan(gap) || std::signbit(gap)) {
                                gap = 1e-1;
                            }
                            gap = std::clamp(gap, 1e-8, 1e-1);

                            // Run optimization and adjust duals
                            ipm_solver.run_optimization(matrix, gap);
                            nodeDuals = ipm_solver.getDuals();
                            for (auto &dual : nodeDuals) {
                                dual = -dual;
                            }
                        };

                    adaptive_threshold = std::max(
                        base_threshold,
                        base_threshold +
                            iter / 100);  // Adapt with total iterations
                    if (std::abs(lp_obj - lp_obj_old) < 1) {
                        iter_non_improv += 1;
                        if (iter_non_improv > adaptive_threshold) {
                            if (stab.alpha > 0) {
                                print_info(
                                    "No improvement in the last iterations, "
                                    "running "
                                    "IPM\n");
                                updateGapAndRunOptimization(
                                    node, lp_obj, inner_obj, ipm_solver,
                                    iter_non_improv, use_ipm_duals, nodeDuals);
                                stab.define_smooth_dual_sol(nodeDuals);
                                iter_non_improv = 0;
                                use_ipm_duals = true;
                            }
                        }
                    } else {
                        iter_non_improv = 0;
                    }
#endif

                    updateGraphAndSolve(nodeDuals);
#ifdef STAB

                    lag_gap = integer_solution - (lp_obj + numK * inner_obj);
                    auto d = 50;
#ifdef HIGHS
                    gap =
                        std::abs(lp_obj - (lp_obj + std::min(0.0, inner_obj))) /
                        std::abs(lp_obj);
                    gap = gap / d;
                    if (std::isnan(gap)) {
                        gap = 1e-2;
                    }
                    if (std::signbit(gap)) {
                        gap = 1e-2;
                    }
                    if (gap > 1e-4) {
                        gap = 1e-4;
                    }
#endif
                    // fmt::print("Gap: {}\n", gap);
                    node->optimize(gap);
                    lp_obj_old = lp_obj;
                    lp_obj = node->getObjVal();
                    nodeDuals = node->getDuals();

                    // auto RC =
                    // node->mip.getMostViolatingReducedCost(nodeDuals);

                    bucket_graph->gap = lp_obj + numK * inner_obj;

                    matrix = node->extractModelDataSparse();
                    stab.lp_obj = lp_obj;
                    // stab.rc     = RC;

                    stab.update_stabilization_after_pricing_optim(
                        matrix, nodeDuals, lag_gap, paths);
                    paths.clear();
                    if (stab.shouldExit()) {
                        misprice = false;
                    }
                    if (colAdded == 0) {
                        stab.update_stabilization_after_misprice();
                        nodeDuals = stab.getStabDualSolAdvanced(nodeDuals);
                    } else {
                        misprice = false;
                    }
                }

                if (bucket_graph->getStatus() == Status::Optimal &&
                    stab.shouldExit()) {
                    print_info("Optimal solution found\n");
                    break;
                }

                if ((colAdded == 0 || inner_obj > -1.0) && stage == 4) {
                    force_cuts = true;
                }

                stab.update_stabilization_after_iter(nodeDuals);
#if defined(STAB) && defined(IPM)
            }
#endif
#endif
            auto cur_alpha = 0.0;
            auto n_cuts = 0;
            auto n_rcc_cuts = 0;
            double tr_val = 0;
            int athreshold = 0;

#ifdef IPM_ACEL
            athreshold = adaptive_threshold;
#endif

#ifdef STAB
            cur_alpha = stab.base_alpha;
#endif

            SRC_MODE_BLOCK(n_cuts = cuts->size();)
            RCC_MODE_BLOCK(n_rcc_cuts = rccManager->size();)

#ifdef GUROBI
            if (allPaths.size() > 5000) {
                auto toRemoveIndices = node->mip.reduceNonBasicVariables(0.66);
                for (auto &index : toRemoveIndices) {
                    allPaths.erase(allPaths.begin() + index);
                }
            }
#endif

#ifdef TR
            tr_val = v;
#endif
            const int threshold = 1000000;
            if (iter % 10 == 0) {
                fmt::print(
                    "| It.: {:4} | Obj.: {:8.2f} | Price: {:9.2f} | SRC: {:3} "
                    "| RCC: {:3} | Paths: {:3} | "
                    "Stage: {:1} | "
                    "Lag.: {:>10} | α: {:4.2f} | tr: {:2.2f} | gap: {:2.4f} "
                    "| Int.: {:>4} "
                    "|\n",
                    iter, lp_obj, inner_obj, n_cuts, n_rcc_cuts, colAdded,
                    stage,
                    (std::isinf(lag_gap)) ? "∞"
                                          : fmt::format("{:10.4f}", lag_gap),
                    cur_alpha, tr_val, gap,
                    (integer_solution > threshold)
                        ? "∞"
                        : fmt::format("{:4}", integer_solution));
                Logger::log(
                    "| It.: {:4} | Obj.: {:8.2f} | Price: {:9.2f} | SRC: {:3} "
                    "| RCC: {:3} | Paths: {:3} | "
                    "Stage: {:1} | "
                    "Lag.: {:10.4f} | α: {:4.2f} | tr: {:2.2f} | gap: {:2.4f} "
                    "|\n",
                    iter, lp_obj, inner_obj, n_cuts, n_rcc_cuts, colAdded,
                    stage, lag_gap, cur_alpha, tr_val, gap);
            }
        }
        bucket_graph->print_statistics();

        node->optimize();
        relaxed_result = node->getObjVal();

        return true;
    }

    std::vector<VRPNode> getNodes() { return nodes; }

    double objective(BNBNode *node) { return ip_result; }

    double bound(BNBNode *node) { return relaxed_result; }

    void evaluate(BNBNode *node) {
        // auto start_timer = std::chrono::high_resolution_clock::now();
        auto cg = CG(node);
        if (!cg) {
            relaxed_result = std::numeric_limits<double>::max();
            return;
        }

        node->binarizeNode();
        node->optimize();

        // check if optimal
        if (node->getStatus() != 2) {
            ip_result = std::numeric_limits<double>::max();
            // node->setPrune(true);
            print_info("No optimal solution found.\n");
        } else {
            ip_result = node->getObjVal();
        }
    }

    static constexpr auto blue = "\033[34m";
    static constexpr auto reset = "\033[0m";

    template <typename T>
    auto blue_text(const T &value) {
        return fmt::format("{}{}{}", blue, value, reset);
    }

    void printSolution(BNBNode *node) {
        auto &allPaths = node->paths;

        // get solution in which x > 0.5 and print the corresponding allPaths
        std::vector<int> sol;
        for (int i = 0; i < allPaths.size(); i++) {
            if (node->getVarValue(i) > 0.5) {
                sol.push_back(i);
            }
        }

        for (auto s : sol) {
            fmt::print("{}", blue_text("_PATH: "));

            Logger::log("PATH: ");
            for (auto j : allPaths[s].route) {
                Logger::log("{} ", j);
                fmt::print("{} ", j);
            }
            fmt::print("\n");
            Logger::log("\n");
        }
        fmt::print("\n");

        fmt::print("+---------------------------------------+\n");
        fmt::print("| {:<14} | {}{:>20}{} |\n", "Bound", blue,
                   relaxed_result / 10, reset);
        fmt::print("| {:<14} | {}{:>20}{} |\n", "Incumbent", blue,
                   ip_result / 10, reset);
        // fmt::print("| {:<14} | {}{:>16}.{:03}{} |\n", "CG Duration", blue,
        // duration_seconds, duration_milliseconds,
        //    reset);
        fmt::print("+---------------------------------------+\n");

        // Logger::log("+---------------------------------------+\n");
        Logger::log("Bound: {} \n", relaxed_result / 10);
        Logger::log("Incumbent: {} \n", ip_result / 10);
        // Logger::log("| {:<14} | {}{:>16}.{:03}{} |\n", "CG Duration", blue,
        // duration_seconds, duration_milliseconds,
        //    reset);
        // Logger::log("+---------------------------------------+\n");
    }

    void branch(BNBNode *node);

    // implement clone method for virtual std::unique_ptr<Problem> clone() const
    // = 0;
    std::unique_ptr<VRProblem> clone() const;

    /**
     * Column generation algorithm.
     */
    bool heuristicCG(BNBNode *node, int max_iter = 100) {
        node->relaxNode();
        node->optimize();
        relaxed_result = std::numeric_limits<double>::max();

        // check if feasible
        if (node->getStatus() != 2) {
            node->setPrune(true);
            return false;
        }
        auto &matrix = node->matrix;
        auto &allPaths = node->paths;
        auto &branchingDuals = node->branchingDuals;

        int bucket_interval = 20;
        int time_horizon = instance.T_max;

        numConstrs = node->getIntAttr("NumConstrs");
        node->numConstrs = numConstrs;
        std::vector<double> duals = std::vector<double>(numConstrs, 0.0);

        std::unique_ptr<BucketGraph> bucket_graph;

        if (problemType == ProblemType::vrptw) {
            bucket_graph = std::make_unique<BucketGraph>(nodes, time_horizon,
                                                         bucket_interval);
        } else if (problemType == ProblemType::cvrp) {
            bucket_graph = std::make_unique<BucketGraph>(
                nodes, time_horizon, bucket_interval, instance.q,
                bucket_interval);
        }

        // print distance matrix size
        bucket_graph->set_distance_matrix(instance.getDistanceMatrix(), 8);
        bucket_graph->branching_duals = branchingDuals;
        bucket_graph->A_MAX = N_SIZE;

        // node->optimize();
        matrix = node->extractModelDataSparse();
        // auto integer_solution = node->getObjVal();
        auto integer_solution = node->integer_sol;
        bucket_graph->incumbent = integer_solution;

        auto allNodes = bucket_graph->getNodes();

        std::vector<double> cutDuals;
        std::vector<double> nodeDuals = node->getDuals();
        auto sizeDuals = nodeDuals.size();

        double lp_obj_dual = 0.0;
        double lp_obj = node->getObjVal();

        bucket_graph->setup();

        double gap = 1e-6;

        bool ss = false;
        int stage = 1;

        bool TRstop = false;

        bool misprice = true;

        double lag_gap = 0.0;

        auto inner_obj = 0.0;
        std::vector<Label *> paths;
        std::vector<double> solution = node->extractSolution();
        bool can_add = true;

#ifdef STAB
        Stabilization stab(0.5, nodeDuals);
#endif
        bool changed = false;

        bool transition = false;

        bool rcc = false;
        bool reoptimized = false;
        double obj;
        auto colAdded = 0;

        RCC_MODE_BLOCK(auto &rccManager = node->rccManager;)

        SRC_MODE_BLOCK(auto &r1c = node->r1c; auto cuts = &r1c->cutStorage;)

        SRC_MODE_BLOCK(bucket_graph->cut_storage = cuts;)

        for (int iter = 0; iter < max_iter; ++iter) {
#ifdef STAB
            stab.update_stabilization_after_master_optim(nodeDuals);
            misprice = true;
            while (misprice) {
                nodeDuals = stab.getStabDualSol(nodeDuals);
                solution = node->extractSolution();
#else
            node->optimize();
            solution = node->extractSolution();
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
                // check if the lp_obj itself is within tolerance
                if (std::abs(std::round(lp_obj) - lp_obj) > 1e-2) {
                    integer = false;
                }

                if (integer) {
                    if (lp_obj < integer_solution) {
                        print_info("Updating integer solution to {}\n", lp_obj);
                        integer_solution = lp_obj;
                        bucket_graph->incumbent = integer_solution;
                    }
                }

                if (!TRstop) {
                    // we do not use integer inside trust region
                    integer = false;
                }

                bucket_graph->relaxation = lp_obj;
                bucket_graph->augment_ng_memories(solution, allPaths, true, 5,
                                                  100, 16, N_SIZE);

                // Branching duals
                if (branchingDuals->size() > 0) {
                    branchingDuals->computeDuals(node);
                }
                bucket_graph->setDuals(nodeDuals);

                //////////////////////////////////////////////////////////////////////
                // CALLING BALDES
                //////////////////////////////////////////////////////////////////////
                paths = bucket_graph->solveHeuristic();
                // inner_obj = paths[0]->cost;
                stage = bucket_graph->getStage();
                ss = bucket_graph->ss;
                //////////////////////////////////////////////////////////////////////

                // Adding cols
                colAdded = addColumn(node, paths, inner_obj, false);

#ifdef STAB
                // TODO: check if we should update this before running the stab
                // update
                node->optimize();
                lp_obj = node->getObjVal();

                nodeDuals = node->getDuals();

                lag_gap =
                    integer_solution - (lp_obj + std::min(0.0, inner_obj));
                bucket_graph->gap = lag_gap;

                matrix = node->extractModelDataSparse();

                stab.update_stabilization_after_pricing_optim(matrix, nodeDuals,
                                                              lag_gap, paths);

                if (colAdded == 0) {
                    stab.update_stabilization_after_misprice();
                    if (stab.shouldExit()) {
                        misprice = false;
                    }
                } else {
                    misprice = false;
                }
            }

            if (bucket_graph->getStatus() == Status::Optimal &&
                stab.shouldExit()) {
                print_info("Optimal solution found\n");
                break;
            }

            stab.update_stabilization_after_iter(nodeDuals);
#endif
        }
        // bucket_graph->print_statistics();

        node->optimize();
        relaxed_result = node->getObjVal();

        return true;
    }
};
