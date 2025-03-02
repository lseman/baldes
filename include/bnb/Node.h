/**
 * @file Node.h
 * @brief Node class implementation
 *
 * This file contains the implementation of the Node class.
 * The Node class is used to represent a node in a tree structure.
 * Each node can have child nodes, and contains state information, objective
 * value, bound value, tree depth, and a unique identifier. It also provides
 * methods for generating a unique identifier and creating new child nodes.
 *
 */
#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <sstream>
#include <variant>

#include "Definitions.h"
#include "Dual.h"
#include "Hashes.h"
#include "Path.h"
#include "SRC.h"
#include "Serializer.h"
#include "State.h"
#include "VRPCandidate.h"
#include "miphandler/Constraint.h"

// include unordered_dense_map
#include "ankerl/unordered_dense.h"
#include "solvers/SolverInterface.h"

#ifdef RCC
#include "../third_party/cvrpsep/capsep.h"
#include "../third_party/cvrpsep/cnstrmgr.h"
// #include "ModernRCC.h"
#endif

#include "miphandler/MIPHandler.h"

#ifdef GUROBI
#include "gurobi_c++.h"
#include "gurobi_c.h"
#include "solvers/Gurobi.h"
#endif

#ifdef IPM
#include "ipm/IPSolver.h"
#include "solvers/IPM.h"
#endif

#ifdef HIGHS
#include "solvers/HighsSolver.h"
#endif

class VRProblem;

/**
 * @class BNBNode
 * @brief Represents a node in a tree structure./
 *
 * The Node class is used to represent a node in a tree structure. Each node can
 * have child nodes, and contains state information, objective value, bound
 * value, tree depth, and a unique identifier. It also provides methods for
 * generating a unique identifier and creating new child nodes.
 */
class BNBNode : public std::enable_shared_from_this<BNBNode> {
   private:
    mutable std::mutex
        mtx;  // mutable allows for locking in const methods if needed
    // Highs              highs;
    //     create shared mutex
    mutable std::shared_mutex mtx2;
    ModelData matrices;

    bool farkas = false;
    bool ip = false;
    bool misprice = false;
    bool pureip = false;
    bool initialized = false;
    bool prune = false;
    // define uuid
    std::string uuid;

    int bestLB = 0;

    SolverInterface *solver = nullptr;

   public:
#ifdef IPM
    IPSolver *ipSolver = nullptr;
#endif

    double integer_sol = std::numeric_limits<double>::max();
    VRProblem *problem;
    InstanceData instance;

    std::vector<std::vector<int>> bestRoutes;

    int depth = 0;
    int numK = 0;
    int numConstrs = 0;

    MIPProblem mip = MIPProblem("node", 0, 0);

    // node specific
    SRC_MODE_BLOCK(std::vector<baldesCtrPtr> SRCconstraints;
                   using LimitedMemoryRank1CutsPtr =
                       std::shared_ptr<LimitedMemoryRank1Cuts>;
                   LimitedMemoryRank1CutsPtr r1c =
                       std::make_shared<LimitedMemoryRank1Cuts>();)

    ModelData matrix;
    std::vector<Path> paths;
    ankerl::unordered_dense::set<Path, PathHash> pathSet;

    RCC_MODE_BLOCK(CnstrMgrPointer oldCutsCMP = nullptr;
                   using RCCManagerPtr = std::shared_ptr<RCCManager>;
                   RCCManagerPtr rccManager = std::make_shared<RCCManager>();)

    void addPath(Path path) { paths.emplace_back(path); }

    std::vector<VRPCandidate *> candidates;
    std::vector<BNBNode *> children;
    BNBNode *parent = nullptr;
    std::vector<VRPCandidate *> raisedVRPChildren;

    explicit BNBNode(const MIPProblem eModel) {
#ifdef IPM
        ipSolver = new IPSolver();
#endif
        mip = eModel;

#ifdef HIGHS
        auto highsmodel = mip.toHighsModel();
        solver = new HighsSolver(highsmodel);
#endif
#ifdef GUROBI
        auto gurobi_model =
            mip.toGurobiModel(GurobiEnvSingleton::getInstance());
        solver = new GurobiSolver(gurobi_model);
#endif
        uuid = generateUUID();
        this->initialized = true;
        RCC_MODE_BLOCK(
            CMGR_CreateCMgr(&oldCutsCMP, 100);  // For old cuts, if needed
        )
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

    std::vector<std::vector<double>> getDistanceMatrix() {
        return instance.getDistanceMatrix();
    }

    /**
     * Generates a Universally Unique Identifier (UUID) as a string.
     *
     * @return The generated UUID as a string.
     */
    std::string generateUUID() {
        // Create an instance of Xoroshiro128Plus with a seed
        Xoroshiro128Plus rng;  // You can use any seed you prefer

        std::stringstream ss;
        ss << std::hex;

        // Generate 8 random hexadecimal characters (0-15)
        for (int i = 0; i < 8; i++) {
            // Generate a random number between 0 and 15 by taking the result of
            // rng() % 16
            int number = rng() % 16;
            ss << number;
        }

        std::string uuid = ss.str();
        return uuid;
    }

    // define hasCandidate to see if the candidate already exists
    bool hasCandidate(VRPCandidate *candidate) {
        for (const auto *c : candidates) {
            if (c == candidate) {
                return true;
            }
        }
        return false;
    }

    void addRaisedChildren(VRPCandidate *candidate) {
        // Add the candidate to the list of candidates
        raisedVRPChildren.push_back(candidate);
    }

    void setCandidatos(std::vector<VRPCandidate *> candidatos) {
        candidates = candidatos;
    }

    void addCandidate(VRPCandidate *candidate) {
        // Add the candidate to the list of candidates
        candidates.push_back(candidate);
    }

    int getDepth() { return depth; }

    BNBNode *newChild() {
        // TODO: verify if this is the best approach
        auto emipClone = mip.clone();
        // Create a new child node
        auto child = new BNBNode(*emipClone);
        // Set the parent of the child node to this node
        child->parent = this;
        child->paths = paths;
        child->historyCandidates = historyCandidates;
        child->candidates = candidates;
        child->r1c = std::make_shared<LimitedMemoryRank1Cuts>(*r1c);
        child->rccManager = rccManager->clone();
        child->instance = instance;
        child->SRCconstraints.clear();
        child->SRCconstraints.reserve(
            SRCconstraints.size());  // Reserve space if size is known
        child->depth = depth + 1;
        child->prune = false;

        child->SRCconstraints = child->mip.getSRCconstraints();
        auto RCCconstraints = child->mip.getRCCconstraints();

        child->rccManager->setCutCtrs(RCCconstraints);

        // child->clearSRC();
        child->matrix = matrix;
        // Return the new child node
        return child;
    }

    void clearSRC() {
        for (auto c : SRCconstraints) {
            mip.delete_constraint(c);
        }
    }

    void addChildren(BNBNode *child) {
        // std::lock_guard<std::mutex> lock(mtx);
        children.push_back(child);
    }

    bool hasRaisedChild(VRPCandidate *strongCandidate) {
        // std::lock_guard<std::mutex> lock(mtx);

        for (const auto *c : raisedVRPChildren) {
            if (c == strongCandidate) {
                return true;
            }
        }
        return false;
    }

    auto getCandidatos() { return candidates; }

    //////////////////////////////////////////////////
    // Functions to deal with MIPProblem
    //////////////////////////////////////////////////

    // Change multiple coefficients for a specific constraint
    void chgCoeff(int constraintIndex, const std::vector<double> &values) {
        mip.chgCoeff(constraintIndex, values);
    }
    void chgCoeff(baldesCtrPtr ctr, const std::vector<double> &values) {
        mip.chgCoeff(ctr, values);
    }

    // Change a single coefficient for a specific constraint and variable
    void chgCoeff(int constraintIndex, int variableIndex, double value) {
        mip.chgCoeff(constraintIndex, variableIndex, value);
    }

    double getSlack(int constraintIndex, const std::vector<double> &solution) {
        // #if defined(IPM)
        // return mip.getSlack(constraintIndex, solution);
        // #elif defined(GUROBI)
        return solver->getSlack(constraintIndex);
        // #endif
    }

    // Binarize all variables (set to binary type)
    void binarizeNode() {
        auto &vars = mip.getVars();
        for (auto var : vars) {
            var->set_type(VarType::Binary);
        }
    }

    // Relax all variables (set to continuous type)
    void relaxNode() {
        auto &vars = mip.getVars();
        for (auto var : vars) {
            var->set_type(VarType::Continuous);
        }
    }

    void remove(baldesCtrPtr ctr) { mip.delete_constraint(ctr); }
    void remove(baldesVarPtr var) { mip.delete_variable(var); }
    int get_current_index(int unique_id) {
        return mip.get_current_index(unique_id);
    }

    baldesVarPtr addVar(const std::string &name, VarType type, double lb,
                        double ub, double obj) {
        return mip.add_variable(name, type, lb, ub, obj);
    }

    void addVars(const double *lb, const double *ub, const double *obj,
                 const VarType *vtypes, const std::string *names,
                 const MIPColumn *cols, size_t count) {
        mip.addVars(lb, ub, obj, vtypes, names, cols, count);
    }
    void addVars(const double *lb, const double *ub, const double *obj,
                 const VarType *vtypes, const std::string *names,
                 size_t count) {
        mip.addVars(lb, ub, obj, vtypes, names, count);
    }
    // Set and get attributes (emulating Gurobi's behavior)
    double getIntAttr(const std::string &attr) {
        if (attr == "NumVars") {
            return mip.getVars().size();
        } else if (attr == "NumConstrs") {
            return mip.getbaldesCtrs().size();
        }
        throw std::invalid_argument("Unknown attribute");
    }

    void setIntAttr(const std::string &attr, int value) {
        throw std::invalid_argument("Unknown attribute");
    }
    void setDoubleAttr(const std::string &attr, double value) {
        throw std::invalid_argument("Unknown attribute");
    }

    baldesCtrPtr addConstr(baldesCtrPtr ctr, const std::string &name) {
        ctr = mip.add_constraint(ctr, name);
        return ctr;
    }

    // Remove a constraint
    void removeConstr(int constraintIndex) {
        mip.delete_constraint(constraintIndex);
    }

    // Get a variable by index
    baldesVarPtr getVar(int i) { return mip.getVar(i); }

    // Get all constraints
    std::vector<baldesCtrPtr> &getConstrs() { return mip.getbaldesCtrs(); }

    ModelData extractModelDataSparse() { return mip.extractModelDataSparse(); }

    /////////////////////////////////////////////////////
    // Solver Interface
    /////////////////////////////////////////////////////

    int getStatus() { return solver->getStatus(); }
    double getObjVal() { return solver->getObjVal(); }
    std::vector<double> getDuals() { return solver->getDuals(); }
    std::vector<double> extractSolution() { return solver->extractSolution(); }

    void optimize(double tol = 1e-6) {
#ifdef HIGHS
        solver->setModel(mip.toHighsModel());
#endif
#ifdef GUROBI
        GRBEnv &env = GurobiEnvSingleton::getInstance();
        // auto    model = new GRBModel(mip.toGurobiModel(env)); // Pass the
        // retrieved or new environment
        // solver->setModel(mip.toGurobiModel(env));
        auto model = mip.toGurobiModel(env);  // Returns unique_ptr<GRBModel>
        solver->setModel(std::any(GRBModelWrapper(std::move(model))));

#endif
        solver->optimize(tol);
    }
    double getVarValue(int i) { return solver->getVarValue(i); }
    auto getModel() { return &mip; }
    auto getDualVal(int i) { return solver->getDualVal(i); }

    // Update
    void update() { mip.update(); }

    double getDualObjVal() { return solver->getDualObjVal(); }

    std::pair<bool, double> solveRestrictedMasterLP() {
        bool feasible = false;
        relaxNode();
        optimize();
        if (getStatus() == 2) {
            feasible = true;
            return std::make_pair(feasible, getDualObjVal());
        } else {
            return std::make_pair(feasible,
                                  -std::numeric_limits<double>::max());
        }
    }

    ///////////////////////////////////////////////
    // Branching
    ///////////////////////////////////////////////

    std::vector<BranchingQueueItem> historyCandidates;

    baldesCtrPtr addBranchingbaldesCtr(
        double rhs, const BranchingDirection &sense, const CandidateType &ctype,
        std::optional<std::variant<int, std::pair<int, int>, std::vector<int>>>
            payload = std::nullopt) {
        // Get the decision variables from the MIP problem
        auto &variables =
            mip.getVars();         // Assumes mip is a pointer to MIPProblem
        LinearExpression linExpr;  // Initialize linear expression for MIP
        std::string name;

        if (ctype == CandidateType::Vehicle) {
            for (auto var : variables) {
                linExpr.add_or_update_term(
                    var->get_name(),
                    1.0);  // Add each variable with coefficient 1.0
                name = "branching_vehicle_" + std::to_string(int(rhs));
            }
        } else if (ctype == CandidateType::Node) {
            for (size_t i = 0; i < variables.size(); ++i) {
                // Use the payload as int for Node
                std::visit(
                    [&](auto &&arg) {
                        using T = std::decay_t<decltype(arg)>;
                        if constexpr (std::is_same_v<T, int>) {
                            linExpr.add_or_update_term(
                                variables[i]->get_name(),
                                paths[i].contains(arg)
                                    ? 1.0
                                    : 0.0);  // Use int payload
                            name = "branching_node_" + std::to_string(arg) +
                                   "_" + std::to_string(int(rhs));
                        } else {
                            throw std::invalid_argument(
                                "Payload for Node must be an int.");
                        }
                    },
                    *payload);
            }
        } else if (ctype == CandidateType::Edge) {
            for (size_t i = 0; i < variables.size(); ++i) {
                // Use the payload as std::pair<int, int> for Edge
                std::visit(
                    [&](auto &&arg) {
                        using T = std::decay_t<decltype(arg)>;
                        if constexpr (std::is_same_v<T, std::pair<int, int>>) {
                            linExpr.add_or_update_term(
                                variables[i]->get_name(),
                                paths[i].timesArc(arg.first, arg.second)
                                    ? 1.0
                                    : 0.0);  // Use pair payload
                            name = "branching_edge_" +
                                   std::to_string(arg.first) + "_" +
                                   std::to_string(arg.second) + "_" +
                                   std::to_string(int(rhs));
                        } else {
                            throw std::invalid_argument(
                                "Payload for Edge must be a std::pair<int, "
                                "int>.");
                        }
                    },
                    *payload);
            }
        } else if (ctype == CandidateType::Cluster) {
            for (size_t i = 0; i < variables.size(); ++i) {
                // Use the payload as std::vector<int> for Cluster
                std::visit(
                    [&](auto &&arg) {
                        using T = std::decay_t<decltype(arg)>;
                        if constexpr (std::is_same_v<T, std::vector<int>>) {
                            for (int cluster_ele : arg) {
                                // Assuming `contains` checks if the path
                                // contains `cluster_ele`
                                if (paths[i].contains(cluster_ele)) {
                                    linExpr.add_or_update_term(
                                        variables[i]->get_name(),
                                        1.0);  // Add term with coefficient 1.0
                                }
                                name = "branching_cluster_" +
                                       std::to_string(cluster_ele) + "_" +
                                       std::to_string(int(rhs));
                            }
                        } else {
                            throw std::invalid_argument(
                                "Payload for Cluster must be a "
                                "std::vector<int>.");
                        }
                    },
                    *payload);
            }
        }

        // Add the constraint to MIP based on the branching direction
        baldesCtrPtr constraint;
        if (mip.constraint_exists(name)) {
            constraint = mip.get_constraint(name);
            return constraint;
        }
        if (sense == BranchingDirection::Greater) {
            constraint = mip.add_constraint(linExpr, rhs, '>');
            constraint->set_name(name);
        } else if (sense == BranchingDirection::Less) {
            constraint = mip.add_constraint(linExpr, rhs, '<');
            constraint->set_name(name);
        } else {
            constraint = mip.add_constraint(linExpr, rhs, '=');
            constraint->set_name(name);
        }

        // Update the MIP structure if necessary
        // mip->update();
        return constraint;
    }
    using BranchingDualsPtr = std::shared_ptr<BranchingDuals>;
    BranchingDualsPtr branchingDuals = std::make_shared<BranchingDuals>();

    void enforceBranching() {
        // Iterate over the candidates and enforce the branching constraints
        for (const auto &candidate : candidates) {
            auto ctr = addBranchingbaldesCtr(
                candidate->boundValue, candidate->boundType,
                candidate->candidateType, candidate->payload);
            branchingDuals->addCandidate(candidate, ctr);
        }
        mip.printBranchingbaldesCtr();
    }

#ifdef GUROBI
    // Assuming model is your GRBModel and obj is your objective sense
    std::vector<int> getBasicVariableIndices() {
        return solver->getBasicVariableIndices();
    }
#endif

    bool loadedState = false;

    void loadState() {
        Snapshot snapshot;
        serializer::Serializer<Snapshot>::fromFile(snapshot, "snapshot.bin");
        paths = snapshot.paths;
        r1c->cutStorage = snapshot.cutStorage;
        auto cutStorage = r1c->cutStorage;
        // print cutStorage.size()
        fmt::print("CutStorage size: {}\n", cutStorage.size());
        // print paths.size
        fmt::print("Paths size: {}\n", paths.size());
        loadedState = true;
        integer_sol = 10301;
        numK = 9;
    }
    void saveState() {
        auto cutStorage = r1c->cutStorage;
        Snapshot snapshot;
        snapshot.paths = paths;
        snapshot.cutStorage = cutStorage;

        serializer::Serializer<Snapshot>::toFile(snapshot, "snapshot.bin");
    }
};
