/**
 * @file Node.h
 * @brief Node class implementation
 *
 * This file contains the implementation of the Node class.
 * The Node class is used to represent a node in a tree structure.
 * Each node can have child nodes, and contains state information, objective value, bound value, tree depth, and a
 * unique identifier. It also provides methods for generating a unique identifier and creating new child nodes.
 *
 */
#pragma once

#include "Definitions.h"
#include "bnb/Problem.h"

#include <sstream>

#include "Hashes.h"
#include "Path.h"
#include "SRC.h"

#include "Dual.h"

#include "VRPCandidate.h"

#include <optional>
#include <variant>

// include unordered_dense_map
#include "ankerl/unordered_dense.h"
#include "solvers/IPM.h"
#include "solvers/SolverInterface.h"

#ifdef RCC
#include "../third_party/cvrpsep/capsep.h"
#include "../third_party/cvrpsep/cnstrmgr.h"
// #include "ModernRCC.h"
#endif

#include "MIPHandler/MIPHandler.h"

#ifdef GUROBI
#include "gurobi_c++.h"
#include "gurobi_c.h"
#include "solvers/Gurobi.h"
#endif

#ifdef IPM
#include "ipm/IPSolver.h"
#endif

class Problem;
/**
 * @class BNBNode
 * @brief Represents a node in a tree structure.
 *
 * The Node class is used to represent a node in a tree structure. Each node can
 * have child nodes, and contains state information, objective value, bound
 * value, tree depth, and a unique identifier. It also provides methods for
 * generating a unique identifier and creating new child nodes.
 */
class BNBNode : public std::enable_shared_from_this<BNBNode> {
private:
    mutable std::mutex mtx; // mutable allows for locking in const methods if needed
    // Highs              highs;
    //     create shared mutex
    mutable std::shared_mutex mtx2;
    ModelData                 matrices;

    bool farkas      = false;
    bool ip          = false;
    bool misprice    = false;
    bool pureip      = false;
    bool initialized = false;
    bool prune       = false;
    // define uuid
    std::string uuid;

    int bestLB = 0;

    SolverInterface *solver = nullptr;

public:
    Problem *problem;
    int      numConstrs = 0;

    MIPProblem mip = MIPProblem("node", 0, 0);

// node specific
#ifdef SRC
    std::vector<Constraint> SRCconstraints;
    LimitedMemoryRank1Cuts  r1c;
#endif

    ModelData         matrix;
    std::vector<Path> paths;
    // ankerl::unordered_dense::set<Path, PathHash> pathSet;
    ankerl::unordered_dense::set<Path, PathHash> pathSet;

#ifdef RCC
    CnstrMgrPointer oldCutsCMP = nullptr;
    RCCManager      rccManager;
#endif

    void addPath(Path path) { paths.emplace_back(path); }

    std::vector<VRPCandidate *> candidates;
    std::vector<BNBNode *>      children;
    BNBNode                    *parent = nullptr;
    std::vector<VRPCandidate *> raisedVRPChildren;

#ifdef GUROBI
    explicit BNBNode(GRBModel &eModel) {
        solver = new GurobiSolver(&eModel);

        generateUUID();
        this->initialized = true;
#ifdef RCC
        CMGR_CreateCMgr(&oldCutsCMP, 100); // For old cuts, if needed
#endif
    };
#endif

    explicit BNBNode(const MIPProblem &eModel) {
        mip = eModel;
        auto matrix = extractModelDataSparse();
        // print matrix.A_sparse.num_rows;
        solver = new IPMSolver(matrix);
        generateUUID();
        this->initialized = true;
#ifdef RCC
        CMGR_CreateCMgr(&oldCutsCMP, 100); // For old cuts, if needed
#endif
    };

    void setPaths(std::vector<Path> paths) { this->paths = paths; }

    std::vector<Path> &getPaths() { return paths; }

    void initialize() { this->initialized = true; }

    void start() {}

    bool getPrune() { return prune; }

    void setPrune(bool prune) { this->prune = prune; }

    [[nodiscard]] bool getInitialized() const { return initialized; }

    auto getUUID() const { return uuid; }

    std::vector<BNBNode *> getChildren() { return children; }

    /**
     * Generates a Universally Unique Identifier (UUID) as a string.
     *
     * @return The generated UUID as a string.
     */
    std::string generateUUID() {
        // Create an instance of Xoroshiro128Plus with a seed
        Xoroshiro128Plus rng; // You can use any seed you prefer

        std::stringstream ss;
        ss << std::hex;

        // Generate 8 random hexadecimal characters (0-15)
        for (int i = 0; i < 8; i++) {
            // Generate a random number between 0 and 15 by taking the result of rng() % 16
            int number = rng() % 16;
            ss << number;
        }

        std::string uuid = ss.str();
        return uuid;
    }

    // define hasCandidate to see if the candidate already exists
    bool hasCandidate(VRPCandidate *candidate) {
        for (const auto *c : candidates) {
            if (c == candidate) { return true; }
        }
        return false;
    }

    void addRaisedChildren(VRPCandidate *candidate) {
        // Add the candidate to the list of candidates
        raisedVRPChildren.push_back(candidate);
    }

    void setCandidatos(std::vector<VRPCandidate *> candidatos) { candidates = candidatos; }

    void addCandidate(VRPCandidate *candidate) {
        // Add the candidate to the list of candidates
        candidates.push_back(candidate);
    }

    BNBNode *newChild() {
        // Create a new child node
        auto child = new BNBNode(mip);
        // Set the parent of the child node to this node
        child->parent            = this;
        child->paths             = paths;
        child->historyCandidates = historyCandidates;
        child->candidates        = candidates;
        child->r1c               = r1c;
        child->matrix            = matrix;
        child->rccManager        = rccManager;

        // Add the child node to the list of children
        children.push_back(child);
        // Return the new child node
        return child;
    }

    bool hasRaisedChild(VRPCandidate *strongCandidate) {
        // std::lock_guard<std::mutex> lock(mtx);

        for (const auto *c : raisedVRPChildren) {
            if (c == strongCandidate) { return true; }
        }
        return false;
    }

    auto getCandidatos() { return candidates; }

    //////////////////////////////////////////////////
    // Functions to deal with MIPProblem
    //////////////////////////////////////////////////

    // Change multiple coefficients for a specific constraint
    void chgCoeff(int constraintIndex, const std::vector<double> &values) { mip.chgCoeff(constraintIndex, values); }
    void chgCoeff(Constraint ctr, const std::vector<double> &values) { mip.chgCoeff(ctr, values); }

    // Change a single coefficient for a specific constraint and variable
    void chgCoeff(int constraintIndex, int variableIndex, double value) {
        mip.chgCoeff(constraintIndex, variableIndex, value);
    }

    // Binarize all variables (set to binary type)
    void binarizeNode() {
        auto &vars = mip.getVars();
        for (auto &var : vars) { var.set_type(VarType::Binary); }
    }

    // Relax all variables (set to continuous type)
    void relaxNode() {
        auto &vars = mip.getVars();
        for (auto &var : vars) { var.set_type(VarType::Continuous); }
    }

    void remove(Constraint &ctr) { mip.delete_constraint(ctr); }
    void remove(Variable &var) { mip.delete_variable(var); }

    void addVars(const double *lb, const double *ub, const double *obj, const VarType *vtypes, const std::string *names,
                 const MIPColumn *cols, size_t count) {
        mip.addVars(lb, ub, obj, vtypes, names, cols, count);
    }
    void addVars(const double *lb, const double *ub, const double *obj, const VarType *vtypes, const std::string *names,
                 size_t count) {
        mip.addVars(lb, ub, obj, vtypes, names, count);
    }
    // Set and get attributes (emulating Gurobi's behavior)
    double getIntAttr(const std::string &attr) {
        if (attr == "NumVars") {
            return mip.getVars().size();
        } else if (attr == "NumConstrs") {
            return mip.getConstraints().size();
        }
        throw std::invalid_argument("Unknown attribute");
    }

    void setIntAttr(const std::string &attr, int value) { throw std::invalid_argument("Unknown attribute"); }
    void setDoubleAttr(const std::string &attr, double value) { throw std::invalid_argument("Unknown attribute"); }

    // Add a new constraint using a linear expression and name (placeholder)
    int addConstr(const LinearExpression &expr, const std::string &name) {
        mip.add_constraint(expr, 0.0, '<'); // Placeholder for <= relation
        return mip.getConstraints().size() - 1;
    }

    int addConstr(const Constraint ctr, const std::string &name) {
        mip.add_constraint(ctr, name);
        return mip.getConstraints().size() - 1;
    }

    // Remove a constraint
    void removeConstr(int constraintIndex) { mip.delete_constraint(constraintIndex); }

    // Get a variable by index
    Variable &getVar(int i) { return mip.getVar(i); }

    // Get all constraints
    const std::vector<Constraint> &getConstrs() { return mip.getConstraints(); }

    ModelData extractModelDataSparse() { return mip.extractModelDataSparse(); }

    /////////////////////////////////////////////////////
    // Solver Interface
    /////////////////////////////////////////////////////

    int                 getStatus() { return solver->getStatus(); }
    double              getObjVal() { return solver->getObjVal(); }
    std::vector<double> getDuals() { return solver->getDuals(); }
    std::vector<double> extractSolution() { return solver->extractSolution(); }
    void                optimize() { 
        solver->setModel(mip.extractModelDataSparse());
        solver->optimize(); }
    double              getVarValue(int i) { return solver->getVarValue(i); }
    auto                getModel() { return &mip; }
    auto getDualVal(int i) { return solver->getDualVal(i); }

#ifdef GUROBI
    // Update
    void update() {
        // delete model; // Delete the old model to free memory
        GRBEnv &env = GurobiEnvSingleton::getInstance();
        // Create a new model with the existing environment
        auto model = new GRBModel(mip.toGurobiModel(env)); // Pass the retrieved or new environment

        solver = new GurobiSolver(model);
        solver->optimize(); // Optimize the model
        // set mute
    }
#else
    void update() {
        // delete model; // Delete the old model to free memory
        auto matrix = mip.extractModelDataSparse();
        solver->setModel(matrix);
        //solver = model;
        //solver->optimize(); // Optimize the model
    }
#endif

    std::pair<bool, double> solveRestrictedMasterLP() {
        bool feasible = false;
        relaxNode();
        // model->optimize();
        // if (model->get(GRB_IntAttr_Status) == GRB_OPTIMAL) {
        //    feasible = true;
        //    return std::make_pair(feasible, model->get(GRB_DoubleAttr_ObjVal));
        //} else {
        //    return std::make_pair(feasible, 0.0);
        //}
    }

    ///////////////////////////////////////////////
    // Branching
    ///////////////////////////////////////////////

    std::vector<BranchingQueueItem> historyCandidates;

    Constraint addBranchingConstraint(double rhs, const BranchingDirection &sense, const CandidateType &ctype,
                                      std::optional<std::variant<int, std::pair<int, int>>> payload = std::nullopt) {
        /*
                // Get the decision variables from the model
                auto varsPtr = model->getVars(); // Assumes model is a pointer to GRBModel
                // deference the pointer
                auto vars = std::vector<GRBVar>(varsPtr, varsPtr + model->get(GRB_IntAttr_NumVars));

                GRBLinExpr linExpr; // Initialize linExpr

                if (ctype == CandidateType::Vehicle) {
                    for (auto i = 0; i < vars.size(); i++) { linExpr += vars[i]; }
                } else if (ctype == CandidateType::Node) {
                    for (auto i = 0; i < vars.size(); i++) {
                        // Use the payload as int for Node
                        std::visit(
                            [&](auto &&arg) {
                                using T = std::decay_t<decltype(arg)>;
                                if constexpr (std::is_same_v<T, int>) {
                                    linExpr += vars[i] * paths[i].contains(arg); // Use the int payload
                                } else {
                                    throw std::invalid_argument("Payload for Node must be an int.");
                                }
                            },
                            *payload);
                    }
                } else if (ctype == CandidateType::Edge) {
                    for (auto i = 0; i < vars.size(); i++) {
                        // Use the payload as std::pair<int, int> for Edge
                        std::visit(
                            [&](auto &&arg) {
                                using T = std::decay_t<decltype(arg)>;
                                if constexpr (std::is_same_v<T, std::pair<int, int>>) {
                                    linExpr += vars[i] * paths[i].timesArc(arg.first, arg.second); // Use the std::pair
                                    payload
                                } else {
                                    throw std::invalid_argument("Payload for Edge must be a std::pair<int, int>.");
                                }
                            },
                            *payload);
                    }
                }

                // Add the constraint based on the sense
                GRBConstr constraint;
                if (sense == BranchingDirection::Greater) {
                    constraint = model->addConstr(linExpr, GRB_GREATER_EQUAL, rhs);
                } else if (sense == BranchingDirection::Less) {
                    constraint = model->addConstr(linExpr, GRB_LESS_EQUAL, rhs);
                } else {
                    constraint = model->addConstr(linExpr, GRB_EQUAL, rhs);
                }

                // Update the model to reflect the changes
                model->update();

                return constraint;
                */
        return Constraint();
    }

    BranchingDuals branchingDuals;

    void enforceBranching() {
        // Iterate over the candidates and enforce the branching constraints
        for (const auto &candidate : candidates) {
            Constraint ctr = addBranchingConstraint(candidate->boundValue, candidate->boundType,
                                                    candidate->candidateType, candidate->payload);
            branchingDuals.addCandidate(candidate, ctr);
        }
    }
};
