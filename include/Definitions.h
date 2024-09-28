/**
 * @file Definitions.h
 * @brief Header file containing definitions and structures for the solver.
 *
 * This file includes various definitions, enumerations, and structures used in the solver.
 * It also provides documentation for each structure and its members, along with methods
 * and operators associated with them.
 *
 */
#pragma once

#include "Common.h"

struct BucketOptions {
    int depot         = 0;
    int end_depot     = N_SIZE - 1;
    int max_path_size = N_SIZE / 2;
};

enum class Direction { Forward, Backward };
enum class Stage { One, Two, Three, Four, Enumerate, Fix };
enum class ArcType { Job, Bucket, Jump };
enum class Mutability { Const, Mut };
enum class Full { Full, Partial, Reverse };
enum class Status { Optimal, Separation, NotOptimal, Error, Rollback };
enum class CutType { ThreeRow, FourRow, FiveRow };

// Comparator function for Stage enum
constexpr bool operator<(Stage lhs, Stage rhs) { return static_cast<int>(lhs) < static_cast<int>(rhs); }
constexpr bool operator>(Stage lhs, Stage rhs) { return rhs < lhs; }
constexpr bool operator<=(Stage lhs, Stage rhs) { return !(lhs > rhs); }
constexpr bool operator>=(Stage lhs, Stage rhs) { return !(lhs < rhs); }

const size_t num_words = (N_SIZE + 63) / 64; // This will be 2 for 100 clients


/**
 * @struct Interval
 * @brief Represents an interval with a duration and a horizon.
 *
 * The Interval struct is used to store information about an interval, which consists of a duration and a
 * horizon. The duration is represented by a double value, while the horizon is represented by an integer value.
 */
struct Interval {
    int interval;
    int horizon;

    Interval(double interval, int horizon) : interval(interval), horizon(horizon) {}
};

// ANSI color code for yellow
constexpr const char *yellow       = "\033[93m";
constexpr const char *vivid_yellow = "\033[38;5;226m"; // Bright yellow
constexpr const char *vivid_red    = "\033[38;5;196m"; // Bright red
constexpr const char *vivid_green  = "\033[38;5;46m";  // Bright green
constexpr const char *vivid_blue   = "\033[38;5;27m";  // Bright blue
constexpr const char *reset_color  = "\033[0m";
constexpr const char *blue         = "\033[34m";
constexpr const char *dark_yellow  = "\033[93m";

/**
 * @brief Prints an informational message with a specific format.
 *
 * This function prints a message prefixed with "[info] " where "info" is colored yellow.
 * The message format and arguments are specified by the caller.
 *
 */
template <typename... Args>
inline void print_info(fmt::format_string<Args...> format, Args &&...args) {
    // Print "[", then yellow "info", then reset color and print "] "
    fmt::print(fg(fmt::color::yellow), "[info] ");
    fmt::print(format, std::forward<Args>(args)...);
}

/**
 * @brief Prints a formatted heuristic message with a specific color scheme.
 *
 * This function prints a message prefixed with "[heuristic] " where "heuristic"
 * is displayed in a vivid blue color. The rest of the message is formatted
 * according to the provided format string and arguments.
 *
 */
template <typename... Args>
inline void print_heur(fmt::format_string<Args...> format, Args &&...args) {
    // Print "[", then yellow "info", then reset color and print "] "
    fmt::print(fg(fmt::color::blue), "[heuristic] ");
    fmt::print(format, std::forward<Args>(args)...);
}

/**
 * @brief Prints a formatted message with a specific prefix.
 *
 * This function prints a message prefixed with "[cut]" in green color.
 * The message is formatted according to the provided format string and arguments.
 *
 */
template <typename... Args>
inline void print_cut(fmt::format_string<Args...> format, Args &&...args) {
    // Print "[", then yellow "info", then reset color and print "] "
    fmt::print(fg(fmt::color::green), "[cut] ");
    fmt::print(format, std::forward<Args>(args)...);
}

/**
 * @brief Prints a formatted message with a blue "info" tag.
 *
 * This function prints a message prefixed with a blue "info" tag enclosed in square brackets.
 * The message is formatted according to the provided format string and arguments.
 *
 */
template <typename... Args>
inline void print_blue(fmt::format_string<Args...> format, Args &&...args) {
    // Print "[", then blue "info", then reset color and print "] "
    fmt::print(fg(fmt::color::blue), "[debug] ");
    fmt::print(format, std::forward<Args>(args)...);
}

/**
 * @struct SparseModel
 * @brief Represents a sparse matrix model.
 *
 * This structure is used to store a sparse matrix in a compressed format.
 * It contains vectors for row indices, column indices, and values, as well
 * as the number of rows and columns in the matrix.
 */
struct SparseModel {
    std::vector<int>    row_indices;
    std::vector<int>    col_indices;
    std::vector<double> values;
    int                 num_rows = 0;
    int                 num_cols = 0;
};

/**
 * @struct ModelData
 * @brief Represents the data structure for a mathematical model.
 *
 * This structure contains all the necessary components to define a mathematical
 * optimization model, including the coefficient matrix, constraints, objective
 * function coefficients, variable bounds, and types.
 *
 */
struct ModelData {
    SparseModel                      A_sparse;
    std::vector<std::vector<double>> A;     // Coefficient matrix for constraints
    std::vector<double>              b;     // Right-hand side coefficients for constraints
    std::vector<char>                sense; // Sense of each constraint ('<', '=', '>')
    std::vector<double>              c;     // Coefficients for the objective function
    std::vector<double>              lb;    // Lower bounds for variables
    std::vector<double>              ub;    // Upper bounds for variables
    std::vector<char>                vtype; // Variable types ('C', 'I', 'B')
    std::vector<std::string>         name;
    std::vector<std::string>         cname;
};

/**
 * @brief Prints the BALDES banner with formatted text.
 *
 * This function prints a banner for the BALDES algorithm, which is a Bucket Graph Labeling Algorithm
 * for Vehicle Routing. The banner includes bold and colored text to highlight the name and description
 * of the algorithm. The text is formatted to fit within a box of fixed width.
 *
 * The BALDES algorithm is a C++ implementation of a Bucket Graph-based labeling algorithm designed
 * to solve the Resource-Constrained Shortest Path Problem (RSCPP). This problem commonly arises as
 * a subproblem in state-of-the-art Branch-Cut-and-Price algorithms for various Vehicle Routing Problems (VRPs).
 */
inline void printBaldes() {
    constexpr auto bold  = "\033[1m";
    constexpr auto blue  = "\033[34m";
    constexpr auto reset = "\033[0m";

    fmt::print("\n");
    fmt::print("+------------------------------------------------------+\n");
    fmt::print("| {}{:<52}{} |\n", bold, "BALDES", reset); // Bold "BALDES"
    fmt::print("| {:<52} |\n", " ");
    fmt::print("| {}{:<52}{} |\n", blue, "BALDES, a Bucket Graph Labeling Algorithm", reset); // Blue text
    fmt::print("| {:<52} |\n", "for Vehicle Routing");
    fmt::print("| {:<52} |\n", " ");
    fmt::print("| {:<52} |\n", "a modern C++ implementation");
    fmt::print("| {:<52} |\n", "of the Bucket Graph-based labeling algorithm");
    fmt::print("| {:<52} |\n", "for the Resource-Constrained Shortest Path Problem");
    fmt::print("| {:<52} |\n", " ");
    fmt::print("| {:<52} |\n", "https://github.com/lseman/baldes");
    fmt::print("| {:<52} |\n", " ");

    fmt::print("+------------------------------------------------------+\n");
    fmt::print("\n");
}

/**
 * @brief Extracts model data from a given Gurobi model in a sparse format.
 *
 * This function retrieves the variables and constraints from the provided Gurobi model
 * and stores them in a ModelData structure. It handles variable bounds, objective coefficients,
 * variable types, and constraint information including the sparse representation of the constraint matrix.
 *
 */
inline ModelData extractModelDataSparse(GRBModel *model) {
    ModelData data;
    try {
        // Variables
        int     numVars = model->get(GRB_IntAttr_NumVars);
        GRBVar *vars    = model->getVars();

        // Reserve memory to avoid frequent reallocations
        data.ub.reserve(numVars);
        data.lb.reserve(numVars);
        data.c.reserve(numVars);
        data.vtype.reserve(numVars);
        data.name.reserve(numVars);

        for (int i = 0; i < numVars; ++i) {
            double ub = vars[i].get(GRB_DoubleAttr_UB);
            data.ub.push_back(ub > 1e10 ? std::numeric_limits<double>::infinity() : ub);

            double lb = vars[i].get(GRB_DoubleAttr_LB);
            data.lb.push_back(lb < -1e10 ? -std::numeric_limits<double>::infinity() : lb);

            data.c.push_back(vars[i].get(GRB_DoubleAttr_Obj));

            char type = vars[i].get(GRB_CharAttr_VType);
            data.vtype.push_back(type);

            data.name.push_back(vars[i].get(GRB_StringAttr_VarName));
        }

        // Constraints
        int         numConstrs = model->get(GRB_IntAttr_NumConstrs);
        SparseModel A_sparse;

        // Reserve memory for constraint matrices
        A_sparse.row_indices.reserve(numConstrs * 10); // Estimate 10 non-zeros per row
        A_sparse.col_indices.reserve(numConstrs * 10);
        A_sparse.values.reserve(numConstrs * 10);
        data.b.reserve(numConstrs);
        data.cname.reserve(numConstrs);
        data.sense.reserve(numConstrs);

        for (int i = 0; i < numConstrs; ++i) {
            GRBConstr  constr = model->getConstr(i);
            GRBLinExpr expr   = model->getRow(constr);

            int exprSize = expr.size();
            for (int j = 0; j < exprSize; ++j) {
                GRBVar var      = expr.getVar(j);
                double coeff    = expr.getCoeff(j);
                int    varIndex = var.index();
                A_sparse.row_indices.push_back(i);
                A_sparse.col_indices.push_back(varIndex);
                A_sparse.values.push_back(coeff);
            }

            data.cname.push_back(constr.get(GRB_StringAttr_ConstrName));
            data.b.push_back(constr.get(GRB_DoubleAttr_RHS));

            char sense = constr.get(GRB_CharAttr_Sense);
            data.sense.push_back(sense == GRB_LESS_EQUAL ? '<' : (sense == GRB_GREATER_EQUAL ? '>' : '='));
        }

        // Store the sparse matrix in data
        A_sparse.num_cols = numVars;
        A_sparse.num_rows = numConstrs;
        data.A_sparse     = A_sparse;

    } catch (GRBException &e) {
        std::cerr << "Error code = " << e.getErrorCode() << std::endl;
        std::cerr << e.getMessage() << std::endl;
    }

    return data;
}

/**
 * @brief Extracts model data from a given Gurobi model.
 *
 */
inline GRBModel createDualModel(GRBEnv *env, const ModelData &primalData) {
    try {
        // Create new model for the dual problem
        GRBModel dualModel = GRBModel(*env);

        // Dual variables: These correspond to the primal constraints
        std::vector<GRBVar> y;
        y.reserve(primalData.b.size());

        // Create dual variables and set their bounds based on primal constraint senses
        for (size_t i = 0; i < primalData.b.size(); ++i) {
            GRBVar dualVar;
            char   sense = primalData.sense[i];
            if (sense == '<') {
                // Dual variable for primal <= constraint, non-negative
                dualVar = dualModel.addVar(0.0, GRB_INFINITY, primalData.b[i], GRB_CONTINUOUS,
                                           "y[" + std::to_string(i) + "]");
            } else if (sense == '>') {
                // Dual variable for primal >= constraint, non-positive
                dualVar = dualModel.addVar(-GRB_INFINITY, 0.0, primalData.b[i], GRB_CONTINUOUS,
                                           "y[" + std::to_string(i) + "]");
            } else if (sense == '=') {
                // Dual variable for primal = constraint, free
                dualVar = dualModel.addVar(-GRB_INFINITY, GRB_INFINITY, primalData.b[i], GRB_CONTINUOUS,
                                           "y[" + std::to_string(i) + "]");
            }
            y.push_back(dualVar);
        }

        dualModel.update();

        // Dual objective: Maximize b^T y
        GRBLinExpr dualObjective = 0;
        for (size_t i = 0; i < primalData.b.size(); ++i) { dualObjective += primalData.b[i] * y[i]; }
        dualModel.setObjective(dualObjective, GRB_MAXIMIZE);

        // Dual constraints: These correspond to primal variables
        for (int j = 0; j < primalData.A_sparse.num_cols; ++j) {
            GRBLinExpr lhs = 0;
            // Iterate over the rows (constraints) that involve variable x_j
            for (size_t i = 0; i < primalData.A_sparse.row_indices.size(); ++i) {
                if (primalData.A_sparse.col_indices[i] == j) {
                    lhs += primalData.A_sparse.values[i] * y[primalData.A_sparse.row_indices[i]];
                }
            }
            // Add dual constraint: lhs >= primalData.c[j] for primal variable x_j
            char vtype = primalData.vtype[j];
            if (vtype == GRB_CONTINUOUS) {
                dualModel.addConstr(lhs == primalData.c[j], "dual_constr[" + std::to_string(j) + "]");
            } else if (vtype == GRB_BINARY || vtype == GRB_INTEGER) {
                // Handle integer/binary variables accordingly (if necessary)
                dualModel.addConstr(lhs >= primalData.c[j], "dual_constr[" + std::to_string(j) + "]");
            }
        }

        dualModel.update();
        return dualModel;

    } catch (GRBException &e) {
        std::cerr << "Error: " << e.getErrorCode() << " - " << e.getMessage() << std::endl;
        throw;
    }
}

using DualSolution = std::vector<double>;

// Paralell sections

// Macro to define parallel sections
#define PARALLEL_SECTIONS(SCHEDULER, ...)                  \
    auto work = parallel_sections(SCHEDULER, __VA_ARGS__); \
    stdexec::sync_wait(std::move(work));

// Macro to define individual sections (tasks)
#define SECTION [this]() -> void

// Template for scheduling parallel sections
template <typename Scheduler, typename... Tasks>
auto parallel_sections(Scheduler &scheduler, Tasks &&...tasks) {
    // Schedule and combine all tasks using stdexec::when_all
    return stdexec::when_all((stdexec::schedule(scheduler) | stdexec::then(std::forward<Tasks>(tasks)))...);
}

inline ModelData extractModelData(GRBModel &model) {
    ModelData data;
    try {
        // Variables
        int     numVars = model.get(GRB_IntAttr_NumVars);
        GRBVar *vars    = model.getVars();
        for (int i = 0; i < numVars; ++i) {

            if (vars[i].get(GRB_DoubleAttr_UB) > 1e10) {
                data.ub.push_back(std::numeric_limits<double>::infinity());
            } else {
                data.ub.push_back(vars[i].get(GRB_DoubleAttr_UB));
            }
            if (vars[i].get(GRB_DoubleAttr_LB) < -1e10) {
                data.lb.push_back(-std::numeric_limits<double>::infinity());
            } else {
                data.lb.push_back(vars[i].get(GRB_DoubleAttr_LB));
            }
            data.c.push_back(vars[i].get(GRB_DoubleAttr_Obj));

            auto type = vars[i].get(GRB_CharAttr_VType);
            if (type == GRB_BINARY) {
                data.vtype.push_back('B');
            } else if (type == GRB_INTEGER) {
                data.vtype.push_back('I');
            } else {
                data.vtype.push_back('C');
            }

            data.name.push_back(vars[i].get(GRB_StringAttr_VarName));
        }

        // Constraints
        int numConstrs = model.get(GRB_IntAttr_NumConstrs);
        for (int i = 0; i < numConstrs; ++i) {
            GRBConstr           constr = model.getConstr(i);
            GRBLinExpr          expr   = model.getRow(constr);
            std::vector<double> aRow(numVars, 0.0);
            for (int j = 0; j < expr.size(); ++j) {
                GRBVar var      = expr.getVar(j);
                double coeff    = expr.getCoeff(j);
                int    varIndex = var.index();
                aRow[varIndex]  = coeff;
            }

            data.cname.push_back(constr.get(GRB_StringAttr_ConstrName));

            data.A.push_back(aRow);
            double rhs = constr.get(GRB_DoubleAttr_RHS);
            data.b.push_back(rhs);
            char sense = constr.get(GRB_CharAttr_Sense);
            data.sense.push_back(sense == GRB_LESS_EQUAL ? '<' : (sense == GRB_GREATER_EQUAL ? '>' : '='));
        }
    } catch (GRBException &e) {
        std::cerr << "Error code = " << e.getErrorCode() << std::endl;
        std::cerr << e.getMessage() << std::endl;
    }

    return data;
}

