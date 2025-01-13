#pragma once
#include <gurobi_c++.h>

#include <iostream>

#include "SolverInterface.h"

struct GRBModelWrapper {
    std::unique_ptr<GRBModel> model;

    // Move constructor
    GRBModelWrapper(std::unique_ptr<GRBModel> m) : model(std::move(m)) {}

    // Copy constructor
    GRBModelWrapper(const GRBModelWrapper &other)
        : model(other.model ? new GRBModel(*other.model) : nullptr) {}

    // Move assignment
    GRBModelWrapper &operator=(GRBModelWrapper &&other) noexcept {
        model = std::move(other.model);
        return *this;
    }

    // Copy assignment
    GRBModelWrapper &operator=(const GRBModelWrapper &other) {
        if (this != &other) {
            model.reset(other.model ? new GRBModel(*other.model) : nullptr);
        }
        return *this;
    }
};

class GurobiEnvSingleton {
   private:
    // Private constructor to prevent direct instantiation
    GurobiEnvSingleton() {
        try {
            env.set(GRB_IntParam_OutputFlag,
                    0);  // Set default parameters, if needed
            // set method
            env.set(GRB_IntParam_Method, 1);
            // set gurobi multicore
            // env.set(GRB_IntParam_Threads,
            // std::thread::hardware_concurrency());

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
    GurobiEnvSingleton(const GurobiEnvSingleton &) = delete;
    GurobiEnvSingleton &operator=(const GurobiEnvSingleton &) = delete;

    // Provide a global point of access
    static GRBEnv &getInstance() {
        static GurobiEnvSingleton instance;  // Guaranteed to be destroyed and
                                             // initialized on first use
        return instance.env;
    }
};

class GurobiSolver : public SolverInterface {
    GRBModel *model = nullptr;
    bool owns_model = false;  // Add this flag
    int numConstrs;

   public:
    // Constructor
    GurobiSolver(GRBModel *model) : model(model) {
        numConstrs = model->get(GRB_IntAttr_NumConstrs);
    }
    GurobiSolver(GRBModel &model) : model(&model) {
        numConstrs = model.get(GRB_IntAttr_NumConstrs);
    }

    // Constructor for unique_ptr
    GurobiSolver(std::unique_ptr<GRBModel> &model) : model(model.get()) {
        numConstrs = model->get(GRB_IntAttr_NumConstrs);
    }

    GurobiSolver(GRBModel *model, bool mute) : model(model) {}

    void setModel(const std::any &modelData) override {
        GRBModel *oldModel = model;
        bool old_owns = owns_model;
        model = nullptr;
        owns_model = false;

        try {
            if (modelData.type() == typeid(GRBModelWrapper)) {
                const auto &wrapper =
                    std::any_cast<const GRBModelWrapper &>(modelData);
                if (!wrapper.model) {
                    throw std::runtime_error("Null model in wrapper");
                }
                model = new GRBModel(*wrapper.model);
                owns_model = true;  // We own this model
            } else if (modelData.type() == typeid(GRBModel *)) {
                auto grbmodel = std::any_cast<GRBModel *>(modelData);
                if (!grbmodel) {
                    throw std::runtime_error("Null model pointer");
                }
                model = grbmodel;
                owns_model = false;  // We don't own this model
            } else {
                model = oldModel;
                owns_model = old_owns;
                throw std::invalid_argument("Invalid model type for Gurobi");
            }

            if (oldModel != nullptr && oldModel != model && old_owns) {
                delete oldModel;
            }

            if (model) {
                numConstrs = model->get(GRB_IntAttr_NumConstrs);
            }

        } catch (const std::exception &e) {
            if (model == nullptr) {
                model = oldModel;
                owns_model = old_owns;
            }
            throw;
        }
    }

    int getStatus() const override { return model->get(GRB_IntAttr_Status); }

    double getObjVal() const override {
        return model->get(GRB_DoubleAttr_ObjVal);
    }

    double getDualObjVal() const override {
        return model->get(GRB_DoubleAttr_ObjBound);
    }

    double getVarValue(int i) const override {
        return model->getVar(i).get(GRB_DoubleAttr_X);
    }

    double getDualVal(int i) const override {
        return model->getConstr(i).get(GRB_DoubleAttr_Pi);
    }

    double getSlack(int i) const override {
        return model->getConstr(i).get(GRB_DoubleAttr_Slack);
    }

    void optimize(double tol = 1e-6) override { model->optimize(); }
    std::vector<double> getDuals() const override {
        int numConstrs = model->get(GRB_IntAttr_NumConstrs);
        std::vector<GRBConstr> constraints;
        constraints.reserve(numConstrs);

        for (int i = 0; i < numConstrs; ++i) {
            constraints.push_back(model->getConstr(i));
        }

        std::vector<double> duals(numConstrs);
        auto dualArray = model->get(GRB_DoubleAttr_Pi, constraints.data(),
                                    constraints.size());
        duals.assign(dualArray, dualArray + numConstrs);
        return duals;
    }

    std::vector<double> extractSolution() const override {
        int varNumber = model->get(GRB_IntAttr_NumVars);
        std::vector<double> sol(varNumber);
        auto vals = model->get(GRB_DoubleAttr_X, model->getVars(), varNumber);
        sol.assign(vals, vals + varNumber);
        return sol;
    }

    std::vector<int> getBasicVariableIndices() override {
        std::vector<int> basicVariableIndices;
        try {
            // Get variables and numVars
            GRBVar *vars = model->getVars();
            int numVars = model->get(GRB_IntAttr_NumVars);

            // Create vector to store basis status for each variable
            for (int i = 0; i < numVars; i++) {
                int basisStatus = vars[i].get(GRB_IntAttr_VBasis);
                // Check if the variable is basic (status == 0)
                if (basisStatus == 0) {
                    basicVariableIndices.push_back(i);
                }
            }

            delete[] vars;  // Clean up allocated memory
        } catch (GRBException &e) {
            std::cout << "Error code = " << e.getErrorCode() << std::endl;
            std::cout << e.getMessage() << std::endl;
        }

        return basicVariableIndices;
    }

    ~GurobiSolver() {
        if (model != nullptr && owns_model) {
            delete model;
        }
    }
};
