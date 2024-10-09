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

#include "VRPCandidate.h"

#include <optional>
#include <variant>

// include unordered_dense_map
#include "ankerl/unordered_dense.h"

#ifdef RCC
#include "../third_party/cvrpsep/capsep.h"
#include "../third_party/cvrpsep/cnstrmgr.h"
// #include "ModernRCC.h"
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

    GRBModel *model;
    int       bestLB = 0;

public:
    Problem *problem;
    int      numConstrs = 0;

// node specific
#ifdef SRC
    std::vector<GRBConstr> SRCconstraints;
    LimitedMemoryRank1Cuts r1c;
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

    [[nodiscard]] explicit BNBNode(const GRBModel &eModel) : model(nullptr) {
        model = new GRBModel(eModel);
        generateUUID();
        this->initialized = true;
#ifdef RCC
        CMGR_CreateCMgr(&oldCutsCMP, 100); // For old cuts, if needed
#endif
    };

    void setPaths(std::vector<Path> paths) { this->paths = paths; }

    std::vector<Path> &getPaths() { return paths; }

    void initialize() { this->initialized = true; }

    void optimize() { model->optimize(); }

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
        std::random_device              rd;
        std::mt19937                    gen(rd());
        std::uniform_int_distribution<> dis(0, 15);
        std::stringstream               ss;
        ss << std::hex;
        for (int i = 0; i < 8; i++) {
            int number = dis(gen);
            ss << number;
        }
        uuid = ss.str();
        return uuid;
    }

    /**
     * @brief Changes the coefficients of a constraint in the given GRBModel.
     *
     * This function modifies the coefficients of a constraint in the specified GRBModel
     * by iterating over the variables and updating their coefficients using the provided values.
     *
     */
    void chgCoeff(const GRBConstr &constrName, std::vector<double> value) {
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
     */
    void binarizeNode() {
        auto varNumber = model->get(GRB_IntAttr_NumVars);
        for (int i = 0; i < varNumber; i++) {
            GRBVar var = model->getVar(i);
            var.set(GRB_CharAttr_VType, GRB_BINARY);
        }
        model->update();
    }

    void relaxNode() {
        auto varNumber = model->get(GRB_IntAttr_NumVars);
        for (int i = 0; i < varNumber; i++) {
            GRBVar var = model->getVar(i);
            var.set(GRB_CharAttr_VType, GRB_CONTINUOUS);
        }
        model->update();
    }

    /**
     * Retrieves the dual values of the constraints in the given GRBModel.
     *
     */
    std::vector<double> getDuals() {
        // int                    numConstrs = model->get(GRB_IntAttr_NumConstrs);
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

    std::vector<double> getAllDuals() {
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
     */
    std::vector<double> extractSolution() {
        std::vector<double> sol;
        auto                varNumber = model->get(GRB_IntAttr_NumVars);
        auto                vals      = model->get(GRB_DoubleAttr_X, model->getVars(), varNumber);
        sol.assign(vals, vals + varNumber);
        return sol;
    }

    // define method get that run model->get
    double get(GRB_IntAttr attr) { return model->get(attr); }

    double get(GRB_DoubleAttr attr) { return model->get(attr); }

    void set(GRB_IntAttr attr, int value) { model->set(attr, value); }
    void set(GRB_DoubleAttr attr, double value) { model->set(attr, value); }
    // set with GRB_IntParam
    void set(GRB_IntParam attr, int value) { model->set(attr, value); }

    // define get for auto duals = node->get(GRB_DoubleAttr_Pi, SRCconstraints.data(), SRCconstraints.size());
    auto get(GRB_DoubleAttr attr, const GRBConstr *constrs, int size) { return model->get(attr, constrs, size); }

    ModelData extractModelDataSparse() {
        model->update();
        return ::extractModelDataSparse(model);
    }

    GRBModel *getModel() { return model; }

    // define addConstr
    GRBConstr addConstr(auto expr, const std::string &name) { return model->addConstr(expr, name); }

    // define getVar
    GRBVar getVar(int i) { return model->getVar(i); }

    // define update
    void update() { model->update(); }

    void remove(GRBConstr constr) { model->remove(constr); }

    // define getConstrs
    const GRBConstr *getConstrs() { return model->getConstrs(); }

    // define addVar method
    [[nodiscard]] GRBVar addVar(auto lb, auto ub, auto obj, auto vtype, auto name) {
        return model->addVar(lb, ub, obj, vtype, name);
    }

    // define addVar with col input also
    void addVar(auto lb, auto ub, double obj, auto vtype, auto col, auto name) {
        model->addVar(lb, ub, obj, vtype, col, name);
    }

    void addVars(auto lb, auto ub, auto obj, auto vtype, auto name, auto col, auto row) {
        model->addVars(lb, ub, obj, vtype, name, col, row);
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
        auto child = new BNBNode(*model);
        // Set the parent of the child node to this node
        child->parent            = this;
        child->paths             = paths;
        child->historyCandidates = historyCandidates;
        child->candidates        = candidates;

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

    std::pair<bool, double> solveRestrictedMasterLP() {
        bool feasible = false;
        relaxNode();
        model->optimize();
        if (model->get(GRB_IntAttr_Status) == GRB_OPTIMAL) {
            feasible = true;
            return std::make_pair(feasible, model->get(GRB_DoubleAttr_ObjVal));
        } else {
            return std::make_pair(feasible, 0.0);
        }
    }

    ///////////////////////////////////////////////
    // Branching
    ///////////////////////////////////////////////

    std::vector<BranchingQueueItem> historyCandidates;

    void addBranchingConstraint(double rhs, const BranchingDirection &sense, const CandidateType &ctype,
                                std::optional<std::variant<int, std::pair<int, int>>> payload = std::nullopt) {

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
                            linExpr += vars[i] * paths[i].timesArc(arg.first, arg.second); // Use the std::pair payload
                        } else {
                            throw std::invalid_argument("Payload for Edge must be a std::pair<int, int>.");
                        }
                    },
                    *payload);
            }
        }

        // Add the constraint based on the sense
        if (sense == BranchingDirection::Greater) {
            model->addConstr(linExpr, GRB_GREATER_EQUAL, rhs);
        } else if (sense == BranchingDirection::Less) {
            model->addConstr(linExpr, GRB_LESS_EQUAL, rhs);
        } else {
            model->addConstr(linExpr, GRB_EQUAL, rhs);
        }

        // Update the model to reflect the changes
        model->update();
    }

    void enforceBranching() {
        // Iterate over the candidates and enforce the branching constraints
        for (const auto &candidate : candidates) {
            addBranchingConstraint(candidate->boundValue, candidate->boundType, candidate->candidateType,
                                   candidate->payload);
        }
    }

    int getNumVariables() { return model->get(GRB_IntAttr_NumVars); }

    // define remove for var
    void remove(GRBVar var) { model->remove(var); }
};
