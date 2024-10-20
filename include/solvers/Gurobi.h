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
            // set method
            // env.set(GRB_IntParam_Method, 2);
            // set gurobi multicore
            env.set(GRB_IntParam_Threads, std::thread::hardware_concurrency());

            // reduce gurobi tolerance to 1e-4
            // set ConcurrentMethod
            // env.set(GRB_IntParam_ConcurrentMIP, 3);
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

    void setModel(const std::any &modelData) override { // Type check and assign
        // print the type of modelData
        // fmt::print("ModelData type: {}\n", modelData.type().name());
        if (modelData.type() == typeid(GRBModel *)) {
            // print modelData.type().name();
            auto grbmodel = std::any_cast<GRBModel *>(modelData);
            model         = grbmodel;
        } else {
            throw std::invalid_argument("Invalid model type for Gurobi");
        }
    }

    int getStatus() const override { return model->get(GRB_IntAttr_Status); }

    double getObjVal() const override { return model->get(GRB_DoubleAttr_ObjVal); }

    double getVarValue(int i) const override { return model->getVar(i).get(GRB_DoubleAttr_X); }

    double getDualVal(int i) const override { return model->getConstr(i).get(GRB_DoubleAttr_Pi); }

    double getSlack(int i) const override { return model->getConstr(i).get(GRB_DoubleAttr_Slack); }

    void                optimize(double tol = 1e-6) override { model->optimize(); }
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
