#pragma once
#include "SolverInterface.h"
#include <iostream>

#include "Highs.h"

class HighsSolver : public SolverInterface {
    Highs *model = nullptr;
    int    numConstrs;

public:
    // Constructor
    HighsSolver(HighsModel &highsmodel) {
        model = new Highs();
        model->passModel(highsmodel);
        // model->setOptionValue("solver", "ipm");
        //   model->setOptionValue("primal_feasibility_tolerance", 1e-8);
        //   model->setOptionValue("dual_feasibility_tolerance", 1e-8);
        model->setOptionValue("run_crossover", "on");
        // model->setOptionValue("run_centring", true);

        //  disable output to screen
        model->setOptionValue("output_flag", "off");
    }

    void setModel(const std::any &modelData) override { // Type check and assign
        // print the type of modelData
        // fmt::print("ModelData type: {}\n", modelData.type().name());
        if (modelData.type() == typeid(HighsModel)) {
            // print modelData.type().name();
            auto highsmodel = std::any_cast<HighsModel>(modelData);
            model->passModel(highsmodel);
        } else {
            throw std::invalid_argument("Invalid model type for Highs");
        }
    }

    int getStatus() const override { return 2; }

    double getObjVal() const override { return model->getObjectiveValue(); }

    double getVarValue(int i) const override { return model->getSolution().col_value[i]; }

    double getDualVal(int i) const override { return model->getSolution().row_dual[i]; }

    double getSlack(int i) const override { return model->getSolution().row_value[i]; }

    void optimize(double tol = 1e-8) override {
        model->setOptionValue("ipm_optimality_tolerance", tol);

        model->run();
    }

    std::vector<double> getDuals() const override { return model->getSolution().row_dual; }

    std::vector<double> extractSolution() const override { return model->getSolution().col_value; }
};