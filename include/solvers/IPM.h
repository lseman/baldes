#pragma once
#include "../include/ipm/IPSolver.h"
#include "Definitions.h"
#include "SolverInterface.h"

class IPMSolver : public SolverInterface {
    int       numConstrs;
    IPSolver  ipm;
    ModelData matrices;

public:
    IPMSolver() : numConstrs(0) {}

    IPMSolver(ModelData &model) : numConstrs(0) { matrices = model; }

    int getStatus() const override { return 2; }

    void setModel(const std::any &modelData) override {
        // Type check and assign
        if (modelData.type() == typeid(ModelData)) {
            matrices = std::any_cast<ModelData>(modelData);
        } else {
            throw std::invalid_argument("Invalid model type for IPMInterface");
        }
    }

    double getObjVal() const override { return ipm.getObjective(); }

    double getVarValue(int i) const override { return ipm.getPrimals()[i]; }

    double getDualVal(int i) const override { return ipm.getDuals()[i]; }

    void optimize(double tol = 1e-6) override { ipm.run_optimization(matrices, tol); }

    std::vector<double> getDuals() const override { return ipm.getDuals(); }

    std::vector<double> extractSolution() const override { return ipm.getPrimals(); }
};