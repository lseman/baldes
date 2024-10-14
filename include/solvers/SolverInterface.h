#pragma once
#include <string>
#include <vector>
class SolverInterface {
public:
    virtual ~SolverInterface() = default;

    // Virtual functions for solver-specific implementations
    virtual int    getStatus() const = 0;
    virtual double getObjVal() const = 0;
    // virtual double              getVarValue(int i) const = 0;
    virtual std::vector<double> getDuals() const         = 0;
    virtual std::vector<double> extractSolution() const  = 0;
    virtual void                optimize()               = 0;
    virtual double              getVarValue(int i) const = 0;

    // virtual void                update()                = 0;
};
