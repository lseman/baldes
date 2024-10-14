#pragma once
#include "SolverInterface.h"
#include <gurobi_c++.h>
#include <iostream>

class GurobiEnvSingleton {
private:
    // Private constructor to prevent direct instantiation
    GurobiEnvSingleton() {
        try {
            env.set(GRB_IntParam_OutputFlag, 0); // Set default parameters, if needed
        } catch (GRBException &e) {
            std::cerr << "Error code = " << e.getErrorCode() << std::endl;
            std::cerr << e.getMessage() << std::endl;
        }
    }

    GRBEnv env;

public:
    // Delete copy constructor and assignment operator
    GurobiEnvSingleton(const GurobiEnvSingleton &)            = delete;
    GurobiEnvSingleton &operator=(const GurobiEnvSingleton &) = delete;

    // Provide a global point of access
    static GRBEnv &getInstance() {
        static GurobiEnvSingleton instance; // Guaranteed to be destroyed and initialized on first use
        return instance.env;
    }
};

class GurobiSolver : public SolverInterface {
    GRBModel *model = nullptr;
    int       numConstrs;

public:
    // Constructor
    GurobiSolver(GRBModel *model) : model(model) { numConstrs = model->get(GRB_IntAttr_NumConstrs); }
    GurobiSolver(GRBModel &model) : model(&model) { numConstrs = model.get(GRB_IntAttr_NumConstrs); }

    GurobiSolver(GRBModel *model, bool mute) : model(model) {}
    void setModel(GRBModel *model) { this->model = model; }

    int getStatus() const override { return model->get(GRB_IntAttr_Status); }

    double getObjVal() const override { return model->get(GRB_DoubleAttr_ObjVal); }

    double getVarValue(int i) const override { return model->getVar(i).get(GRB_DoubleAttr_X); }

    double getDualVal(int i) const override { return model->getConstr(i).get(GRB_DoubleAttr_Pi); }

    void                optimize() override { model->optimize(); }
    std::vector<double> getDuals() const override {
        int                    numConstrs = model->get(GRB_IntAttr_NumConstrs);
        std::vector<GRBConstr> constraints;
        constraints.reserve(numConstrs);

        for (int i = 0; i < numConstrs; ++i) { constraints.push_back(model->getConstr(i)); }

        std::vector<double> duals(numConstrs);
        auto                dualArray = model->get(GRB_DoubleAttr_Pi, constraints.data(), constraints.size());
        duals.assign(dualArray, dualArray + numConstrs);
        return duals;
    }

    std::vector<double> extractSolution() const override {
        int                 varNumber = model->get(GRB_IntAttr_NumVars);
        std::vector<double> sol(varNumber);
        auto                vals = model->get(GRB_DoubleAttr_X, model->getVars(), varNumber);
        sol.assign(vals, vals + varNumber);
        return sol;
    }
};