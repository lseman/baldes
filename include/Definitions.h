/**
 * @file Definitions.h
 * @brief Header file containing definitions and structures for the solver.
 *
 * This file includes various definitions, enumerations, and structures used in
 * the solver. It also provides documentation for each structure and its
 * members, along with methods and operators associated with them.
 *
 */
#pragma once

#include "Common.h"
#include "SparseMatrix.h"
#include "ankerl/unordered_dense.h"

struct ReducedCostResult {
    double value;
    int column_index;
    std::vector<int> col;
};

struct BucketOptions {
    int depot = 0;
    int end_depot = N_SIZE - 1;
    int pstep = false;
    int max_path_size = N_SIZE / 2;
    int min_path_size = N_SIZE / 2;

    int pstep_depot = 0;
    int pstep_end_depot = N_SIZE - 1;

    bool manual_arcs = false;
    bool bucket_fixing = true;
    int size = N_SIZE;

    int three_two_sign = 1;
    int three_three_sign = 1;
    int three_five_sign = 1;

    bool symmetric = false;

    std::vector<int> main_resources = {0};
    std::vector<std::string> resources = {"time"};
    std::vector<int> resource_type = {1};
    std::vector<int> or_resources = {1};

    int n_warm_start = 100;

    bool warm_start = true;

    // EVRP options
    int battery_capacity = 100;
    int max_recharges = 3;
    int max_recharge_time = 30;

    // Singleton pattern: Get the single instance of BucketOptions
    static BucketOptions &getInstance() {
        static BucketOptions instance;
        return instance;
    }
};

enum class Direction { Forward, Backward };
enum class Stage { One, Two, Three, Four, Enumerate, Fix, Eliminate, Extend };
enum class ArcType { Node, Bucket, Jump };
enum class Mutability { Const, Mut };
enum class Full { Full, Partial, Reverse, PSTEP, TSP };
enum class Status { Optimal, Separation, NotOptimal, Error, Rollback };
enum class CutType { OneRow, ThreeRow, FourRow, FiveRow };
enum class BranchingDirection { Greater, Less, Equal };
enum class CandidateType { Vehicle, Node, Edge, Cluster };
enum class ProblemType { vrptw, cvrp, evrp };
enum class Symmetry { Asymmetric, Symmetric };

using Payload =
    std::optional<std::variant<int, std::pair<int, int>>>;  // Optional variant
                                                            // for payload data
                                                            //
// Step 1: Implement Aggregated baldesVars for Branching baldesCtrs
struct BranchingQueueItem {
    int sourceNode;          // route index (or source node for edges)
    int targetNode;          // source node or customer for edges
    double fractionalValue;  // Fractionality of the route/variable
    double productValue;     // Additional score or weight
    ankerl::unordered_dense::map<int, double> g_m;  // Vehicle type aggregation
    ankerl::unordered_dense::map<int, double>
        g_m_v;  // Customer-service aggregation
    ankerl::unordered_dense::map<std::pair<int, int>, double>
        g_v_vp;  // Edge usage aggregation
    CandidateType
        candidateType;  // Type of the candidate (Vehicle, Node, or Edge)
    std::pair<bool, bool> flags;  // Flags for additional information
    double score = 0.0;           // Pseudo-cost or score for the candidate
    ankerl::unordered_dense::map<int, double> g_c;  // Cluster aggregation
    std::vector<int> clusters;  // Clusters for the candidate
};

// Comparator function for Stage enum
constexpr bool operator<(Stage lhs, Stage rhs) {
    return static_cast<int>(lhs) < static_cast<int>(rhs);
}
constexpr bool operator>(Stage lhs, Stage rhs) { return rhs < lhs; }
constexpr bool operator<=(Stage lhs, Stage rhs) { return !(lhs > rhs); }
constexpr bool operator>=(Stage lhs, Stage rhs) { return !(lhs < rhs); }

const size_t num_words = (N_SIZE + 63) / 64;  // This will be 2 for 100 clients

/**
 * @struct Interval
 * @brief Represents an interval with a duration and a horizon.
 *
 * The Interval struct is used to store information about an interval, which
 * consists of a duration and a horizon. The duration is represented by a double
 * value, while the horizon is represented by an integer value.
 */
struct Interval {
    int interval;
    int horizon;

    Interval(double interval, int horizon)
        : interval(interval), horizon(horizon) {}
};

// ANSI color code for yellow
constexpr const char *yellow = "\033[93m";
constexpr const char *vivid_yellow = "\033[38;5;226m";  // Bright yellow
constexpr const char *vivid_red = "\033[38;5;196m";     // Bright red
constexpr const char *vivid_green = "\033[38;5;46m";    // Bright green
constexpr const char *vivid_blue = "\033[38;5;27m";     // Bright blue
constexpr const char *reset_color = "\033[0m";
constexpr const char *blue = "\033[34m";
constexpr const char *dark_yellow = "\033[93m";
constexpr auto reset = "\033[0m";

/**
 * @brief Prints an informational message with a specific format.
 *
 * This function prints a message prefixed with "[info] " where "info" is
 * colored yellow. The message format and arguments are specified by the caller.
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
    fmt::print("{}{}{}", blue, "[heur] ", reset);
    fmt::print(format, std::forward<Args>(args)...);
}

/**
 * @brief Prints a formatted message with a specific prefix.
 *
 * This function prints a message prefixed with "[cut]" in green color.
 * The message is formatted according to the provided format string and
 * arguments.
 *
 */
template <typename... Args>
inline void print_cut(fmt::format_string<Args...> format, Args &&...args) {
    // Print "[", then yellow "info", then reset color and print "] "
    fmt::print(fg(fmt::color::cyan), "[cut] ");
    fmt::print(format, std::forward<Args>(args)...);
}

template <typename... Args>
inline void print_branching(fmt::format_string<Args...> format,
                            Args &&...args) {
    // Print "[", then yellow "info", then reset color and print "] "
    fmt::print(fg(fmt::color::dark_cyan), "[branching] ");
    fmt::print(format, std::forward<Args>(args)...);
}
/**
 * @brief Prints a formatted message with a blue "info" tag.
 *
 * This function prints a message prefixed with a blue "info" tag enclosed in
 * square brackets. The message is formatted according to the provided format
 * string and arguments.
 *
 */
template <typename... Args>
inline void print_blue(fmt::format_string<Args...> format, Args &&...args) {
    // Print "[", then blue "info", then reset color and print "] "
    fmt::print(fg(fmt::color::blue), "[debug] ");
    fmt::print(format, std::forward<Args>(args)...);
}

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
    SparseMatrix A_sparse;
    std::vector<double> b;    // Right-hand side coefficients for constraints
    std::vector<char> sense;  // Sense of each constraint ('<', '=', '>')
    std::vector<double> c;    // Coefficients for the objective function
    std::vector<double> lb;   // Lower bounds for variables
    std::vector<double> ub;   // Upper bounds for variables
    std::vector<char> vtype;  // baldesVar types ('C', 'I', 'B')
    std::vector<std::string> name;
    std::vector<std::string> cname;
};

/**
 * @brief Prints the BALDES banner with formatted text.
 *
 * This function prints a banner for the BALDES algorithm, which is a Bucket
 * Graph Labeling Algorithm for Vehicle Routing. The banner includes bold and
 * colored text to highlight the name and description of the algorithm. The text
 * is formatted to fit within a box of fixed width.
 *
 * The BALDES algorithm is a C++ implementation of a Bucket Graph-based labeling
 * algorithm designed to solve the Resource-Constrained Shortest Path Problem
 * (RSCPP). This problem commonly arises as a subproblem in state-of-the-art
 * Branch-Cut-and-Price algorithms for various Vehicle Routing Problems (VRPs).
 */
inline void printBaldes() {
    constexpr auto bold = "\033[1m";
    constexpr auto blue = "\033[34m";
    constexpr auto reset = "\033[0m";

    fmt::print("\n");
    fmt::print("+------------------------------------------------------+\n");
    fmt::print("| {}{:<52}{} |\n", bold,
               fmt::format("BALDES (commit {})", GIT_COMMIT_HASH), reset);
    fmt::print("| {:<52} |\n", " ");
    fmt::print("| {}{:<52}{} |\n", blue,
               "BALDES, a Bucket Graph Labeling Algorithm",
               reset);  // Blue text
    fmt::print("| {:<52} |\n", "for Vehicle Routing");
    fmt::print("| {:<52} |\n", " ");
    fmt::print("| {:<52} |\n", "a modern C++ implementation of the");
    fmt::print("| {:<52} |\n", "Bucket Graph-based Labeling Algorithm");
    fmt::print("| {:<52} |\n",
               "for the Resource-Constrained Shortest Path Problem");
    fmt::print("| {:<52} |\n", " ");
    fmt::print("| {:<52} |\n", "https://github.com/lseman/baldes");
    fmt::print("| {:<52} |\n", " ");
    fmt::print("+------------------------------------------------------+\n");
    fmt::print("\n");
}

#ifdef GUROBI
/**
 * @brief Extracts model data from a given Gurobi model in a sparse format.
 *
 * This function retrieves the variables and constraints from the provided
 * Gurobi model and stores them in a ModelData structure. It handles variable
 * bounds, objective coefficients, variable types, and constraint information
 * including the sparse representation of the constraint matrix.
 *
 */
inline ModelData extractModelDataSparse(GRBModel *model) {
    ModelData data;
    try {
        // baldesVars
        int numVars = model->get(GRB_IntAttr_NumVars);
        GRBVar *vars = model->getVars();

        // Reserve memory based on the actual number of variables
        data.ub.reserve(numVars);
        data.lb.reserve(numVars);
        data.c.reserve(numVars);
        data.vtype.reserve(numVars);
        data.name.reserve(numVars);

        std::vector<double> ub(numVars), lb(numVars), obj(numVars);
        std::vector<char> vtype(numVars);

        auto ubs =
            model->get(GRB_DoubleAttr_UB, vars, numVars);  // Fetch upper bounds
        auto lbs =
            model->get(GRB_DoubleAttr_LB, vars, numVars);  // Fetch lower bounds
        auto objs = model->get(GRB_DoubleAttr_Obj, vars,
                               numVars);  // Fetch objective coefficients
        auto vtypes = model->get(GRB_CharAttr_VType, vars,
                                 numVars);  // Fetch variable types

        // Copy the values to the vectors
        data.ub.assign(ubs, ubs + numVars);
        data.lb.assign(lbs, lbs + numVars);
        data.c.assign(objs, objs + numVars);
        data.vtype.assign(vtypes, vtypes + numVars);

        // baldesCtrs
        int numConstrs = model->get(GRB_IntAttr_NumConstrs);
        SparseMatrix A_sparse;

        // Reserve memory for constraint matrices, estimating 10 non-zeros per
        // row
        A_sparse.values.reserve(numConstrs * 10);  // Estimate non-zero elements
        data.b.reserve(numConstrs);
        data.cname.reserve(numConstrs);
        data.sense.reserve(numConstrs);

        for (int i = 0; i < numConstrs; ++i) {
            GRBConstr constr = model->getConstr(i);

            // Get the row corresponding to the constraint
            GRBLinExpr expr = model->getRow(constr);
            int exprSize = expr.size();

            // Reserve space for constraint elements to minimize reallocation
            A_sparse.values.reserve(A_sparse.values.size() + exprSize);

            for (int j = 0; j < exprSize; ++j) {
                GRBVar var = expr.getVar(j);
                double coeff = expr.getCoeff(j);
                int varIndex = var.index();

                // Populate SparseElement for A_sparse
                A_sparse.insert(i, varIndex, coeff);
            }

            // Batch fetch baldesCtr Name, RHS, and Sense (improve by reducing
            // single fetches)
            data.cname.push_back(constr.get(
                GRB_StringAttr_ConstrName));  // Fetch constraint names
            data.b.push_back(
                constr.get(GRB_DoubleAttr_RHS));  // Fetch RHS values
            data.sense.push_back(
                constr.get(GRB_CharAttr_Sense));  // Fetch sense values
        }

        // Set matrix dimensions and build row_start
        A_sparse.num_cols = numVars;
        A_sparse.num_rows = numConstrs;
        A_sparse.buildRowStart();

        // Store the sparse matrix in data
        data.A_sparse = std::move(A_sparse);

    } catch (GRBException &e) {
        std::cerr << "Error code = " << e.getErrorCode() << std::endl;
        std::cerr << e.getMessage() << std::endl;
    }

    return data;
}
#endif

using DualSolution = std::vector<double>;

// Paralell sections

// Macro to define parallel sections
#define PARALLEL_SECTIONS(NAME, SCHEDULER, ...)            \
    auto NAME = parallel_sections(SCHEDULER, __VA_ARGS__); \
    stdexec::sync_wait(std::move(NAME));

// Macro to define individual sections (tasks)
#define SECTION [this]() -> void

// #define SECTION_CUSTOM(capture_list) [capture_list]() -> void
#define SECTION_CUSTOM(capture_list...) [capture_list]() -> void

// Template for scheduling parallel sections
template <typename Scheduler, typename... Tasks>
auto parallel_sections(Scheduler &scheduler, Tasks &&...tasks) {
    // Schedule and combine all tasks using stdexec::when_all
    return stdexec::when_all((stdexec::schedule(scheduler) |
                              stdexec::then(std::forward<Tasks>(tasks)))...);
}

#define CONDITIONAL(D, FW_ACTION, BW_ACTION)         \
    if constexpr (D == Direction::Forward) {         \
        FW_ACTION;                                   \
    } else if constexpr (D == Direction::Backward) { \
        BW_ACTION;                                   \
    }

inline double roundToTwoDecimalPlaces(double value) {
    return std::round(value * 100.0) / 100.0;
}

#ifdef IPM
#define RUN_OPTIMIZATION(node, tol)                  \
    matrix = (node)->extractModelDataSparse();       \
    (node)->ipSolver->run_optimization(matrix, tol); \
    solution = (node)->ipSolver->getPrimals();
#else
#define RUN_OPTIMIZATION(node, tol) \
    (node)->optimize();             \
    solution = (node)->extractSolution();
#endif

#ifdef IPM
#define GET_SOL(node) solution = (node)->ipSolver->getPrimals();
#else
#define GET_SOL(node) solution = (node)->extractSolution();
#endif

#ifdef SRC
#define SRC_MODE_BLOCK(code_block) code_block
#else
#define SRC_MODE_BLOCK(code_block)  // No operation if SRC is not defined
#endif

#ifdef RCC
#define RCC_MODE_BLOCK(code_block) code_block
#else
#define RCC_MODE_BLOCK(code_block)  // No operation if RCC is not defined
#endif

#define PARALLEL_EXECUTE(tasks, lambda_func, ...)                             \
    do {                                                                      \
        const int JOBS = std::thread::hardware_concurrency();                 \
        exec::static_thread_pool pool(JOBS);                                  \
        auto sched = pool.get_scheduler();                                    \
                                                                              \
        const int chunk_size = (tasks.size() + JOBS - 1) / JOBS;              \
        std::mutex tasks_mutex; /* Protect shared resources */                \
                                                                              \
        auto bulk_sender = stdexec::bulk(                                     \
            stdexec::just(), (tasks.size() + chunk_size - 1) / chunk_size,    \
            [&](std::size_t chunk_idx) {                                      \
                size_t start_idx = chunk_idx * chunk_size;                    \
                size_t end_idx =                                              \
                    std::min(start_idx + chunk_size, tasks.size());           \
                                                                              \
                /* Process a chunk of tasks */                                \
                for (size_t task_idx = start_idx; task_idx < end_idx;         \
                     ++task_idx) {                                            \
                    (lambda_func)(tasks[task_idx], tasks_mutex, __VA_ARGS__); \
                }                                                             \
            });                                                               \
                                                                              \
        /* Submit work to the thread pool and wait for completion */          \
        auto work = stdexec::starts_on(sched, bulk_sender);                   \
        stdexec::sync_wait(std::move(work));                                  \
    } while (0)
