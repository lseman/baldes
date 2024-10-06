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

#include "Path.h"
#include "SRC.h"

#include "VRPCandidate.h"

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
    std::vector<GRBConstr> SRCconstraints;
    ModelData              matrix;
    std::vector<Path>      paths;
    LimitedMemoryRank1Cuts r1c;

    void addPath(Path path) { paths.emplace_back(path); }

    std::vector<VRPCandidate> candidates;
    std::vector<BNBNode *>    children;
    BNBNode                  *parent = nullptr;
    std::vector<VRPCandidate> raisedVRPChildren;

    [[nodiscard]] explicit BNBNode(const GRBModel &eModel) : model(nullptr) {
        model = new GRBModel(eModel);
        generateUUID();
        this->initialized = true;
    };

    void setPaths(std::vector<Path> paths) { this->paths = paths; }

    std::vector<Path> &getPaths() { return paths; }

    void initialize() { this->initialized = true; }

    void optimize() { model->optimize(); }

    void start() {}

    void enforceBranching() {}

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

    ModelData extractModelDataSparse() { return ::extractModelDataSparse(model); }

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
    bool hasCandidate(const VRPCandidate &candidate) {
        for (const auto &c : candidates) {
            if (c == candidate) { return true; }
        }
        return false;
    }

    void addRaisedChildren(const VRPCandidate &candidate) {
        // Add the candidate to the list of candidates
        candidates.push_back(candidate);
    }

    void setCandidatos(std::vector<VRPCandidate> candidatos) { candidates = candidatos; }

    void addCandidate(const VRPCandidate &candidate) {
        // Add the candidate to the list of candidates
        candidates.push_back(candidate);
    }

    BNBNode *newChild() {
        // Create a new child node
        auto child = new BNBNode(*model);
        // Set the parent of the child node to this node
        child->parent = this;
        child->paths  = paths;

        // Add the child node to the list of children
        children.push_back(child);
        // Return the new child node
        return child;
    }

    bool hasRaisedChild(const VRPCandidate &strongCandidate) {
        // std::lock_guard<std::mutex> lock(mtx);

        for (const auto &c : raisedVRPChildren) {
            if (c == strongCandidate) { return true; }
        }
        return false;
    }

    auto getCandidatos() { return candidates; }

    double solveRestrictedMasterLP() {
        relaxNode();
        model->optimize();
        return model->get(GRB_DoubleAttr_ObjVal);
    }

    void addBranchingConstraint(const std::vector<double> &coeffs, double rhs) {
        // Get the decision variables from the model
        auto vars = model->getVars(); // Assumes model is a pointer to GRBModel

        // Create a linear expression for the constraint
        GRBLinExpr linExpr;
        for (size_t i = 0; i < coeffs.size(); ++i) { linExpr += coeffs[i] * vars[i]; }

        // Add the constraint to the model: linExpr <= rhs
        model->addConstr(linExpr, GRB_LESS_EQUAL, rhs);

        // Update the model to reflect the changes
        model->update();
    }

    int getNumVariables() { return model->get(GRB_IntAttr_NumVars); }
};
