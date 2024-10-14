#pragma once

#include "Definitions.h"
#include "fmt/base.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#ifdef GUROBI
#include "gurobi_c++.h"
#include "gurobi_c.h"
#endif

#include <iostream>
#include <new>
#include <stdexcept>
#include <vector>

// #include "CuSolver.h"
#include "LDLT.h"

#ifdef CHOLMOD
#include "cholmod.h"
#endif

/**
 * @struct OptimizationData
 * @brief A structure to hold data for an optimization problem.
 *
 * This structure contains the following members:
 * - As: A sparse matrix representing the constraint coefficients.
 * - bs: A vector representing the right-hand side of the constraints.
 * - cs: A vector representing the coefficients of the objective function.
 * - lo: A vector representing the lower bounds of the variables.
 * - hi: A vector representing the upper bounds of the variables.
 * - sense: A vector representing the sense of the constraints (e.g., equality, inequality).
 */
struct OptimizationData {
    Eigen::SparseMatrix<double> As;
    Eigen::VectorXd             bs;
    Eigen::VectorXd             cs;
    Eigen::VectorXd             lo;
    Eigen::VectorXd             hi;
    Eigen::VectorXd             sense;
};

/**
 * @struct Residuals
 * @brief A structure to hold various residual vectors and their norms.
 *
 * This structure contains the following members:
 * - rp: Residual vector for primal feasibility.
 * - ru: Residual vector for dual feasibility.
 * - rd: Residual vector for duality gap.
 * - rl: Residual vector for lower bounds.
 * - rpn: Norm of the primal feasibility residual.
 * - run: Norm of the dual feasibility residual.
 * - rdn: Norm of the duality gap residual.
 * - rgn: Norm of the gradient residual.
 * - rg: Gradient residual.
 * - rln: Norm of the lower bounds residual.
 */
struct Residuals {
    Eigen::VectorXd rp, ru, rd, rl;
    double          rpn, run, rdn, rgn, rg, rln;
};

#ifdef GUROBI
OptimizationData extractOptimizationComponents(GRBModel &model);
#endif
OptimizationData convertToOptimizationData(const ModelData &modelData);

#ifdef CHOLMOD
/**
 * @class CholmodSolver
 * @brief A solver class that utilizes CHOLMOD for sparse matrix factorization and solving linear systems.
 *
 * This class provides an interface to the CHOLMOD library for performing sparse matrix factorization
 * and solving linear systems using the factorized matrix. It supports both GPU and CPU computations.
 *
 * @note This class requires Eigen library for matrix and vector representations.
 */
class CholmodSolver {
public:
    cholmod_common  c;
    cholmod_factor *L;
    cholmod_dense  *X = NULL, *Y = NULL, *E = NULL;
    bool            firstFactorization = true;

    CholmodSolver() {
        cholmod_start(&c);
        c.useGPU             = 0;
        c.nmethods           = 1;
        c.method[0].ordering = CHOLMOD_METIS;
        c.postorder          = true;

#ifdef AUGMENTED
        c.supernodal = CHOLMOD_SIMPLICIAL;
#else
        c.supernodal = CHOLMOD_SUPERNODAL;
#endif

        L = nullptr;
    }

    ~CholmodSolver() {
        if (L) { cholmod_free_factor(&L, &c); }
        cholmod_finish(&c);
    }

    void factorizeMatrix(const Eigen::SparseMatrix<double, Eigen::RowMajor, int> &matrix) {
        cholmod_sparse *A = viewAsCholmod(matrix);

        if (firstFactorization) {
            if (L) { cholmod_free_factor(&L, &c); }
            L                  = cholmod_analyze(A, &c);
            firstFactorization = false;
        }

        cholmod_factorize(A, L, &c);
    }

    Eigen::VectorXd solve(const Eigen::VectorXd &rhs) {
        cholmod_dense *b = viewAsCholmod(rhs);
        cholmod_solve2(CHOLMOD_A, L, b, NULL, &X, NULL, &Y, &E, &c);
        Eigen::VectorXd result = viewAsEigen(X);
        return result;
    }

private:
    static cholmod_sparse *viewAsCholmod(const Eigen::SparseMatrix<double, Eigen::RowMajor, int> &matrix) {
        cholmod_sparse *result = new cholmod_sparse;
        result->nrow           = matrix.rows();
        result->ncol           = matrix.cols();
        result->p              = const_cast<int *>(matrix.outerIndexPtr());
        result->i              = const_cast<int *>(matrix.innerIndexPtr());
        result->x              = const_cast<double *>(matrix.valuePtr());
        result->z              = nullptr;
        result->stype          = -1;
        result->itype          = CHOLMOD_INT;
        result->xtype          = CHOLMOD_REAL;
        result->dtype          = CHOLMOD_DOUBLE;
        result->sorted         = 1;
        result->packed         = 1;
        return result;
    }

    static cholmod_dense *viewAsCholmod(const Eigen::VectorXd &vector) {
        cholmod_dense *result = new cholmod_dense;
        result->nrow          = vector.size();
        result->ncol          = 1;
        result->nzmax         = vector.size();
        result->d             = vector.size();
        result->x             = const_cast<double *>(vector.data());
        result->z             = nullptr;
        result->xtype         = CHOLMOD_REAL;
        result->dtype         = CHOLMOD_DOUBLE;
        return result;
    }

    static Eigen::VectorXd viewAsEigen(cholmod_dense *vectorPtr) {
        return Eigen::VectorXd::Map(reinterpret_cast<double *>(vectorPtr->x), vectorPtr->nrow);
    }
};
#endif

/**
 * @class SparseSolver
 * @brief A class for solving sparse linear systems using various solver types.
 *
 * The SparseSolver class provides an interface for factorizing and solving sparse linear systems.
 * It supports different solver types, with the default being CHOLMOD.
 *
 */
class SparseSolver {
public:
    int                         n;
    int                         m;
    Eigen::VectorXd             theta;
    Eigen::VectorXd             regP;
    Eigen::VectorXd             regD;
    Eigen::SparseMatrix<double> A;
    Eigen::SparseMatrix<double> S;
    Eigen::SparseMatrix<double> AD;
    Eigen::SparseMatrix<double> D;
    bool                        firstFactorization = true;

    enum SolverType {
#ifdef CHOLMOD
        CH,
#endif
        LDLT,
        //CU
    };

    // SparseSolver() { solver = new SolverWrapper<CholmodSolver>(); }

    SparseSolver(SolverType type = LDLT) {
        switch (type) {
#ifdef CHOLMOD
        case CH: solver = new SolverWrapper<CholmodSolver>(); break;
#endif
        case LDLT:
            solver = new SolverWrapper<LDLTSolver>();
            break;
            // case CU: solver = new SolverWrapper<CuSolver>(); break;
        }
    }

    ~SparseSolver() { delete solver; }

    void factorizeMatrix(const Eigen::SparseMatrix<double, Eigen::ColMajor, int> &matrix) {
        if (firstFactorization) {
            // A = matrix;
            solver->factorizeMatrix(matrix);
            firstFactorization = false;
        } else {
            // A = matrix;
            solver->factorizeMatrix(matrix);
        }
    }

    Eigen::VectorXd solve(const Eigen::VectorXd &rhs) { return solver->solve(rhs); }

private:
    SolverType solverType;

    struct SolverBase {
        virtual void            factorizeMatrix(const Eigen::SparseMatrix<double, Eigen::ColMajor, int> &matrix) = 0;
        virtual Eigen::VectorXd solve(const Eigen::VectorXd &rhs)                                                = 0;
        virtual ~SolverBase() = default;
    };

    template <typename Solver>
    struct SolverWrapper : public SolverBase {
        Solver solver;
        void   factorizeMatrix(const Eigen::SparseMatrix<double, Eigen::ColMajor, int> &matrix) override {
            solver.factorizeMatrix(matrix);
        }
        Eigen::VectorXd solve(const Eigen::VectorXd &rhs) override { return solver.solve(rhs); }
    };

    SolverBase *solver;
};

/**
 * @class IPSolver
 * @brief A class for solving linear programming problems using an interior point method.
 *
 * This class provides methods to convert linear programming problems to standard form,
 * update residuals, solve augmented systems, and run the optimization process.
 *
 */
class IPSolver {

public:
    Residuals    res;
    SparseSolver ls;
    double       tau, kappa, tol;
    int          max_iter;
    double       infty = std::numeric_limits<double>::infinity();

    // Create history of the old values
    Eigen::VectorXd x_old;
    Eigen::VectorXd lambda_old;
    Eigen::VectorXd s_old;
    Eigen::VectorXd v_old;
    Eigen::VectorXd w_old;

    std::vector<double> dual_vals;
    std::vector<double> primal_vals;
    double objVal;

    std::vector<double> getDuals() const { return dual_vals; }
    std::vector<double> getPrimals() const { return primal_vals; }
    double              getObjective() const { return objVal; }

    // create default constructor
    IPSolver() {}
    Eigen::SparseMatrix<double> convertToSparseDiagonal(const Eigen::VectorXd &vec);

    // Method to convert the given linear programming problem to standard form
    void convert_to_standard_form(const Eigen::SparseMatrix<double> &A, const Eigen::VectorXd &b,
                                  const Eigen::VectorXd &c, const Eigen::VectorXd &lb, const Eigen::VectorXd &ub,
                                  const Eigen::VectorXd &sense, Eigen::SparseMatrix<double> &As, Eigen::VectorXd &bs,
                                  Eigen::VectorXd &cs);

    // Method to update residuals of a self-dual interior point method
    void update_residuals(Residuals &res, const Eigen::VectorXd &x, const Eigen::VectorXd &lambda,
                          const Eigen::VectorXd &s, const Eigen::VectorXd &v, const Eigen::VectorXd &w,
                          const Eigen::SparseMatrix<double> &A, const Eigen::VectorXd &b, const Eigen::VectorXd &c,
                          const Eigen::VectorXd &ubv, const Eigen::VectorXi &ubi, const Eigen::VectorXd &vbv,
                          const Eigen::VectorXi &vbi, double tau, double kappa);

    // Method to solve the augmented system of equations to obtain the solution vectors dx and dy
    void solve_augmented_system(Eigen::VectorXd &dx, Eigen::VectorXd &dy, SparseSolver &ls, const Eigen::VectorXd &xi_p,
                                const Eigen::VectorXd &xi_d);

    // Method to solve the augmented system of equations to compute the values of delta_x, delta_y, and delta_z
    void solve_augsys(Eigen::VectorXd &delta_x, Eigen::VectorXd &delta_y, Eigen::VectorXd &delta_z, SparseSolver &ls,
                      const Eigen::VectorXd &theta_vw, const Eigen::VectorXi &ubi, const Eigen::VectorXd &xi_p,
                      const Eigen::VectorXd &xi_d, const Eigen::VectorXd &xi_u);

    // Method to solve the Newton system of equations to update the variables Delta_x, Delta_lambda, Delta_w,
    // Delta_s, Delta_v, Delta_tau, and Delta_kappa
    void solve_newton_system(Eigen::VectorXd &Delta_x, Eigen::VectorXd &Delta_lambda, Eigen::VectorXd &Delta_w,
                             Eigen::VectorXd &Delta_s, Eigen::VectorXd &Delta_v, double &Delta_tau, double &Delta_kappa,
                             SparseSolver &ls, const Eigen::VectorXd &theta_vw, const Eigen::VectorXd &b,
                             const Eigen::VectorXd &c, const Eigen::VectorXi &ubi, const Eigen::VectorXd &ubv,
                             const Eigen::VectorXd &delta_x, const Eigen::VectorXd &delta_y,
                             const Eigen::VectorXd &delta_w, double delta_0, const Eigen::VectorXd &iter_x,
                             const Eigen::VectorXd &iter_lambda, const Eigen::VectorXd &iter_w,
                             const Eigen::VectorXd &iter_s, const Eigen::VectorXd &iter_v, double iter_tau,
                             double iter_kappa, const Eigen::VectorXd &xi_p, const Eigen::VectorXd &xi_u,
                             const Eigen::VectorXd &xi_d, double xi_g, const Eigen::VectorXd &xi_xs,
                             const Eigen::VectorXd &xi_vw, double xi_tau_kappa);

    // Method to calculate the maximum value of alpha based on the given vectors v and dv
    double max_alpha_single(const Eigen::VectorXd &v, const Eigen::VectorXd &dv);

    // Method to calculate the maximum alpha value for a given set of parameters
    double max_alpha(const Eigen::VectorXd &x, const Eigen::VectorXd &dx, const Eigen::VectorXd &v,
                     const Eigen::VectorXd &dv, const Eigen::VectorXd &s, const Eigen::VectorXd &ds,
                     const Eigen::VectorXd &w, const Eigen::VectorXd &dw, double tau, double dtau, double kappa,
                     double dkappa);

    // Method to run the optimization process
    std::tuple<double, double, std::vector<double>, std::vector<double>> run_optimization(ModelData   &model,
                                                                                          const double tol);

#ifdef GUROBI
    // Method to extract optimization components from a Gurobi model
    OptimizationData extractOptimizationComponents(GRBModel &model);
#endif

    // Method to convert model data to optimization data
    OptimizationData convertToOptimizationData(const ModelData &modelData);

    // Method to update the linear solver with new theta and regularization parameters
    void update_linear_solver(SparseSolver &ls, const Eigen::VectorXd &theta, const Eigen::VectorXd &regP,
                              const Eigen::VectorXd &regD);

    // Method to start the linear solver by initializing necessary data structures and performing factorization
    void start_linear_solver(SparseSolver &ls, const Eigen::SparseMatrix<double> A);
};
