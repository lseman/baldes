#pragma once

#include "Definitions.h"
#include "bnb/Problem.h"
#include "gurobi_c++.h"
#include <sstream>

class Problem;
/**
 * @class Node
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
    // define uuid
    std::string uuid;

    GRBModel *model;
    int       bestLB     = 0;

public:
    Problem *problem;
    int       numConstrs = 0;

    [[nodiscard]] explicit BNBNode(const GRBModel &eModel) : model(nullptr) {
        model = new GRBModel(eModel);
        generateUUID();
        this->initialized = true;
    };

    void initialize() { this->initialized = true; }

    void optimize() { model->optimize(); }

    void start() {}

    void enforceBranching() {}

    bool getPrune() { return false; }

    [[nodiscard]] bool getInitialized() const { return initialized; }

    auto getUUID() const { return uuid; }

    std::vector<BNBNode *> getChildren() {
        std::vector<BNBNode *> children;
        return children;
    }

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

    // define get for auto duals = node->get(GRB_DoubleAttr_Pi, SRCconstraints.data(), SRCconstraints.size());
    auto get(GRB_DoubleAttr attr, const GRBConstr *constrs, int size) { return model->get(attr, constrs, size); }

    ModelData extractModelDataSparse() { return ::extractModelDataSparse(model); }

    GRBModel *getModel() { return model; }

    // define addConstr
    GRBConstr addConstr(auto expr, const std::string &name) {
        // std::lock_guard<std::mutex> lock(mtx);
        return model->addConstr(expr, name);
    }

    // define getVar
    GRBVar getVar(int i) { return model->getVar(i); }

    // define update
    void update() { model->update(); }

    void remove(GRBConstr constr) { model->remove(constr); }

    // define getConstrs
    const GRBConstr *getConstrs() { return model->getConstrs(); }

    // define addVar method
    [[nodiscard]] GRBVar addVar(auto lb, auto ub, auto obj, auto vtype, auto name) {
        // std::lock_guard<std::mutex> lock(mtx);
        return model->addVar(lb, ub, obj, vtype, name);
    }


    // define addVar with col input also
    void addVar(auto lb, auto ub, double obj, auto vtype, auto col, auto name) {
        // std::lock_guard<std::mutex> lock(mtx);
        this->model->addVar(lb, ub, obj, vtype, col, name);
    }

    // define   287 |             addVars(node, lb.data(), ub.data(), obj.data(), vtypes.data(), names.data(),
    // cols.data(), lb.size());
    void addVars(auto lb, auto ub, auto obj, auto vtype, auto name, auto col, auto row) {
        // std::lock_guard<std::mutex> lock(mtx);
        model->addVars(lb, ub, obj, vtype, name, col, row);
    }
};
