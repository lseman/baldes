/**
 * @file Definitions.h
 * @brief Header file containing definitions and structures for the solver.
 *
 * This file includes various definitions, enumerations, and structures used in the solver.
 * It also provides documentation for each structure and its members, along with methods
 * and operators associated with them.
 *
 * @details
 * The file includes the following:
 * - Enumerations for Direction, Stage, ArcType, Mutability, and Full.
 * - Comparator functions for the Stage enumeration.
 * - Structures for Interval, Path, and Label.
 * - Hash functions for RCCarc and std::pair<int, int>.
 * - Methods for managing cuts, computing duals, and handling paths and labels.
 *
 * The structures and methods are documented with detailed descriptions, parameters, and return values.
 */
#pragma once
#include "config.h"

#include "gurobi_c++.h"
#include "gurobi_c.h"

#include "../external/pdqsort.h"

#include "Hashes.h"

#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>

#include <execution>
#ifdef AVX
#include <immintrin.h>
#endif

#include <algorithm>
#include <array>
#include <cstring>
#include <deque>
#include <fstream>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <ranges>
#include <unordered_map>
#include <vector>

#include <fmt/color.h>
#include <fmt/core.h>

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
 * @struct Cut
 * @brief Represents a cut in the optimization problem.
 *
 * The Cut structure holds information about a specific cut, including its base set,
 * neighbors, coefficients, multipliers, and other properties.
 *
 */
struct Cut {
    int                             cutMaster;
    std::array<uint64_t, num_words> baseSet;      // Bit-level baseSet
    std::array<uint64_t, num_words> neighbors;    // Bit-level neighbors
    std::vector<int>                baseSetOrder; // Order for baseSet
    std::vector<double>             coefficients; // Cut coefficients
    std::vector<double>             multipliers = {0.5, 0.5, 0.5};
    double                          rhs         = 1;
    int                             id          = -1;
    bool                            added       = false;
    bool                            updated     = false;
    CutType                         type        = CutType::ThreeRow;
    GRBConstr                       grbConstr;
    size_t                          key;
    // Default constructor
    Cut() = default;

    // constructor to receive array
    Cut(const std::array<uint64_t, num_words> baseSetInput, const std::array<uint64_t, num_words> &neighborsInput,
        const std::vector<double> &coefficients)
        : baseSet(baseSetInput), neighbors(neighborsInput), coefficients(coefficients) {}

    Cut(const std::array<uint64_t, num_words> baseSetInput, const std::array<uint64_t, num_words> &neighborsInput,
        const std::vector<double> &coefficients, const std::vector<double> &multipliers)
        : baseSet(baseSetInput), neighbors(neighborsInput), coefficients(coefficients), multipliers(multipliers) {}

    // Define size of the cut
    size_t size() const { return coefficients.size(); }
};

using Cuts = std::vector<Cut>;

/**
 * @class CutStorage
 * @brief Manages the storage and operations related to cuts in a solver.
 *
 * The CutStorage class provides functionalities to add, manage, and query cuts.
 * It also allows setting dual values and computing coefficients with limited memory.
 *
 */
class CutStorage {
public:
    int latest_column = 0;

    Cut &operator[](std::size_t index) { return cuts[index]; }

    // Add a cut to the storage
    /**
     * @brief Adds a cut to the current collection of cuts.
     *
     * This function takes a reference to a Cut object and adds it to the
     * collection of cuts maintained by the solver. The cut is used to
     * refine the solution space and improve the efficiency of the solver.
     *
     */
    void addCut(Cut &cut);

    void reset() {
        cuts.clear();
        cutMaster_to_cut_map.clear();
        indexCuts.clear();
        SRCDuals = {};
    }

    Cut &getCut(int cutIndex) { return cuts[cutIndex]; }

    int getID(int cutIndex) { return cuts[cutIndex].id; }

    void removeCut(int cutIndex) {
        // Ensure the cutIndex is within bounds
        if (cutIndex < 0 || cutIndex >= cuts.size()) {
            std::cerr << "Cut index " << cutIndex << " is out of bounds." << std::endl;
            return;
        }

        // Erase the cut from the cuts vector
        cuts.erase(cuts.begin() + cutIndex);

        for (auto &entry : cuts) {
            if (entry.id > cutIndex) {
                entry.id--; // Decrement index to account for the removed cut
            }
        }

        // Find the corresponding entry in the map
        auto it = std::find_if(cutMaster_to_cut_map.begin(), cutMaster_to_cut_map.end(),
                               [cutIndex](const auto &pair) { return pair.second == cutIndex; });

        if (it != cutMaster_to_cut_map.end()) {
            // Remove the entry for the removed cut
            cutMaster_to_cut_map.erase(it);
        } else {
            std::cerr << "Cut index " << cutIndex << " not found in cutMaster_to_cut_map." << std::endl;
            return; // No need to continue if the cut wasn't found in the map
        }

        // Reorganize the indices in the map for all subsequent cuts
        for (auto &entry : cutMaster_to_cut_map) {
            if (entry.second > cutIndex) {
                entry.second--; // Decrement index to account for the removed cut
            }
        }
    }

    std::vector<double> SRCDuals = {};

    /**
     * @brief Sets the dual values for the SRC.
     *
     * This function assigns the provided vector of dual values to the SRCDuals member.
     *
     */
    void setDuals(const std::vector<double> &duals) { SRCDuals = duals; }

    // Define size method
    size_t size() noexcept { return cuts.size(); }

    // Define begin and end
    auto begin() const noexcept { return cuts.begin(); }
    auto end() const noexcept { return cuts.end(); }
    auto begin() noexcept { return cuts.begin(); }
    auto end() noexcept { return cuts.end(); }

    // Define empty method
    bool empty() const noexcept { return cuts.empty(); }

    /**
     * @brief Checks if a cut exists for the given cut key and returns its size and coefficients.
     *
     * This function searches for the specified cut key in the cutMaster_to_cut_map. If the cut key
     * is found, it retrieves the size and coefficients of the corresponding cut from the cuts vector.
     * If the cut key is not found, it returns a pair with -1 and an empty vector.
     *
     */
    std::pair<int, std::vector<double>> cutExists(const std::size_t &cut_key) const {
        auto it = cutMaster_to_cut_map.find(cut_key);
        if (it != cutMaster_to_cut_map.end()) {
            auto tam    = cuts[it->second].size();
            auto coeffs = cuts[it->second].coefficients;
            return {tam, coeffs};
        }
        return {-1, {}};
    }

    /**
     * @brief Retrieves the constraint at the specified index.
     *
     * This function returns the Gurobi constraint object associated with the
     * cut at the given index.
     *
     */
    auto getCtr(int i) const { return cuts[i].grbConstr; }

    /**
     * @brief Computes limited memory coefficients for a given set of cuts.
     *
     * This function iterates over a collection of cuts and computes a set of coefficients
     * based on the provided vector P. The computation involves checking membership of nodes
     * in specific sets and updating coefficients accordingly.
     *
     */
    auto computeLimitedMemoryCoefficients(const std::vector<int> &P) {
        // iterate over cuts
        std::vector<double> alphas;
        alphas.reserve(cuts.size());
        for (auto c : cuts) {
            double alpha = 0.0;
            double S     = 0;
            auto   AM    = c.neighbors;
            auto   C     = c.baseSet;
            auto   p     = c.multipliers;
            auto   order = c.baseSetOrder;

            for (size_t j = 1; j < P.size() - 1; ++j) {
                int vj = P[j];

                // Check if the node vj is in AM (bitwise check)
                if (!(AM[vj / 64] & (1ULL << (vj % 64)))) {
                    S = 0; // Reset S if vj is not in AM
                }
                if (C[vj / 64] & (1ULL << (vj % 64))) {
                    // Get the position of vj in C by counting the set bits up to vj
                    int pos = order[vj];
                    S += p[pos];
                    if (S >= 1) {
                        S -= 1;
                        alpha += 1;
                    }
                }
            }

            alphas.push_back(alpha);
        }
        return alphas;
    }

    std::size_t generateCutKey(const int &cutMaster, const std::vector<bool> &baseSetStr) const;

private:
    std::unordered_map<std::size_t, int>              cutMaster_to_cut_map;
    Cuts                                              cuts;
    std::unordered_map<std::size_t, std::vector<int>> indexCuts;
};

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

struct Arc {
    int                 from;
    int                 to;
    std::vector<double> resource_increment;
    double              cost_increment;
    bool                fixed    = false;
    double              priority = 0;

    Arc(int from, int to, const std::vector<double> &res_inc, double cost_inc);

    Arc(int from, int to, const std::vector<double> &res_inc, double cost_inc, bool fixed);

    Arc(int from, int to, const std::vector<double> &res_inc, double cost_inc, double priority);
};

/**
 * @struct Path
 * @brief Represents a path with a route and its associated cost.
 *
 * The Path struct encapsulates a route represented as a vector of integers and a cost associated with the
 * route. It provides various utility methods to interact with the route, such as checking for the presence of
 * elements, counting occurrences, and managing arcs between route points.
 *
 */
struct Path {
    std::vector<int> route;
    double           cost;
    double           red_cost = std::numeric_limits<double>::max();

    // default constructor
    Path() : route({}), cost(0.0) {}
    Path(const std::vector<int> &route, double cost) : route(route), cost(cost) {
        [[maybe_unused]] auto future = std::async(std::launch::async, &Path::precomputeArcs, this); // Ignore the future
    }

    // define begin and end methods linking to route
    auto begin() { return route.begin(); }
    auto end() { return route.end(); }
    // define size
    auto size() { return route.size(); }
    // make the [] operator available
    int operator[](int i) const { return route[i]; }

    /**
     * @brief Checks if the given integer is present in the route.
     *
     * This function searches for the specified integer within the route
     * and returns true if the integer is found, otherwise false.
     *
     */
    bool contains(int i) { return std::find(route.begin(), route.end(), i) != route.end(); }

    /**
     * @brief Counts the occurrences of a given integer in the route.
     *
     * This function iterates through the 'route' container and counts how many times
     * the specified integer 'i' appears in it.
     *
     */
    int countOccurrences(int i) { return std::count(route.begin(), route.end(), i); }

    /**
     * @brief Counts the number of times an arc (i, j) appears in the route.
     *
     * This function iterates through the route and counts how many times the arc
     * from node i to node j appears consecutively.
     *
     */
    int timesArc(int i, int j) const {
        int       times = 0;
        const int size  = route.size();
        for (int n = 1; n < size; ++n) {
            if ((route[n - 1] == i && route[n] == j)) { times++; }
        }

        return times;
    }

    std::unordered_map<std::pair<int, int>, int, pair_hash> arcMap; // Maps arcs to their counts

    /**
     * @brief Adds an arc between two nodes and increments its count in the arc map.
     *
     * This function creates a pair representing an arc between nodes `i` and `j`,
     * and increments the count of this arc in the `arcMap`. If the arc does not
     * already exist in the map, it is added with an initial count of 1.
     *
     */
    void addArc(int i, int j) {
        std::pair<int, int> arc = std::make_pair(i, j);
        arcMap[arc]++; // Increment the count of the arc
    }

    /**
     * @brief Precomputes arcs for the given route.
     *
     * This function iterates through the route and adds arcs between consecutive nodes.
     * It assumes that the route is a valid sequence of nodes and that the addArc function
     * is defined to handle the addition of arcs between nodes.
     */
    void precomputeArcs() {
        for (int n = 0; n < route.size() - 1; ++n) { addArc(route[n], route[n + 1]); }
    }

    /**
     * @brief Retrieves the count of arcs between two nodes.
     *
     * This function takes two integers representing nodes and returns the count
     * of arcs between them. If the arc pair (i, j) exists in the arcMap, the
     * function returns the associated count. Otherwise, it returns 0.
     *
     */
    auto getArcCount(int i, int j) const {
        // Construct the arc pair
        std::pair<int, int> arc = std::make_pair(i, j);
        return (arcMap.find(arc) != arcMap.end()) ? arcMap.at(arc) : 0;
    }

    /**
     * @brief Retrieves the count of a specified arc.
     *
     * This function takes an arc represented by an RCCarc object and constructs
     * a pair from its 'from' and 'to' members. It then checks if this pair exists
     * in the arcMap. If the pair is found, the function returns the count associated
     * with the arc. If the pair is not found, it returns 0.
     *
     */
};

// Bucket structure to hold a collection of labels
// Structure to represent a bucket arc
/**
 * @struct BucketArc
 * @brief Represents an arc between two buckets.
 *
 * This struct contains information about an arc between two buckets in a solver.
 * It stores the index of the source bucket, the index of the target bucket,
 * the resource increments associated with the arc, the cost increment,
 * and a flag indicating whether the arc is fixed.
 *
 */
struct BucketArc {
    int                 from_bucket;
    int                 to_bucket;
    std::vector<double> resource_increment;
    double              cost_increment;
    bool                fixed = false;

    bool operator==(const BucketArc &other) const {
        return from_bucket == other.from_bucket && to_bucket == other.to_bucket;
    }

    BucketArc(int from, int to, const std::vector<double> &res_inc, double cost_inc);

    BucketArc(int from, int to, const std::vector<double> &res_inc, double cost_inc, bool fixed);

    // Overload < operator for map comparison
    bool operator<(const BucketArc &other) const {
        if (from_bucket != other.from_bucket) return from_bucket < other.from_bucket;
        if (to_bucket != other.to_bucket) return to_bucket < other.to_bucket;
        if (cost_increment != other.cost_increment) return cost_increment < other.cost_increment;
        if (resource_increment != other.resource_increment) return resource_increment < other.resource_increment;
        return fixed < other.fixed;
    }
};

// Structure to represent a jump arc
/**
 * @struct JumpArc
 * @brief Represents a jump arc between two buckets.
 *
 * This struct contains information about a jump arc, including the base bucket, jump bucket,
 * resource increment, and cost increment.
 *
 */
struct JumpArc {
    int                 base_bucket;
    int                 jump_bucket;
    std::vector<double> resource_increment;
    double              cost_increment;

    JumpArc(int base, int jump, const std::vector<double> &res_inc, double cost_inc);
};

struct ViewPeriod {
    double rise;
    double set;
    double trx_on;
    double trx_off;
    int    index;
};

// Structure to represent a job
/**
 * @struct VRPJob
 * @brief Represents a job in a Vehicle Routing Problem.
 *
 * This struct contains information about a job, such as its ID, start time, end time, duration, cost, demand,
 * and capacity constraints. It provides constructors to initialize the job with different sets of parameters.
 * The `setDuals` method allows updating the cost of the job.
 */
struct VRPJob {
    double                        x;
    double                        y;
    int                           id;
    int                           start_time;
    int                           end_time;
    int                           duration;
    double                        cost = 0.0;
    double                        demand;
    std::vector<int>              lb;
    std::vector<int>              ub;
    std::vector<int>              mtw_lb;
    std::vector<int>              mtw_ub;
    std::vector<Arc>              fw_arcs;
    std::vector<Arc>              bw_arcs;
    std::vector<std::vector<Arc>> fw_arcs_scc;
    std::vector<std::vector<Arc>> bw_arcs_scc;

    std::string                                              track_id;
    int                                                      subject;
    int                                                      week;
    int                                                      year;
    double                                                   duration_min;
    int                                                      setup_time;
    int                                                      teardown_time;
    long long                                                time_window_start;
    long long                                                time_window_end;
    std::unordered_map<std::string, std::vector<ViewPeriod>> antenna_view_periods;

    std::vector<double> consumption;

    // default constructor
    VRPJob() = default;

    VRPJob(int i, int st, int et, int dur, double c) : id(i), start_time(st), end_time(et), duration(dur), cost(c) {}

    VRPJob(int i, int st, int et, int dur, double c, double d)
        : id(i), start_time(st), end_time(et), duration(dur), cost(c), demand(d) {}

    /**
     * @brief Adds an arc between two buckets with specified resource increments and cost.
     *
     * This function adds a forward or backward arc between the specified buckets.
     * The arc is characterized by resource increments and a cost increment.
     *
     */
    void add_arc(int from_bucket, int to_bucket, std::vector<double> res_inc, double cost_inc, bool fw) {
        if (fw) {
            fw_arcs.push_back({from_bucket, to_bucket, std::move(res_inc), cost_inc});
        } else {
            bw_arcs.push_back({from_bucket, to_bucket, std::move(res_inc), cost_inc});
        }
    }

    /**
     * @brief Adds an arc to the forward or backward arc list.
     *
     * This function adds an arc between two buckets, either to the forward arc list
     * or the backward arc list, based on the direction specified by the `fw` parameter.
     *
     */
    void add_arc(int from_bucket, int to_bucket, std::vector<double> res_inc, double cost_inc, bool fw, bool fixed) {
        if (fw) {
            fw_arcs.push_back({from_bucket, to_bucket, std::move(res_inc), cost_inc, fixed});
        } else {
            bw_arcs.push_back({from_bucket, to_bucket, std::move(res_inc), cost_inc, fixed});
        }
    }

    void add_arc(int from_bucket, int to_bucket, std::vector<double> res_inc, double cost_inc, bool fw,
                 double priority) {
        if (fw) {
            fw_arcs.push_back({from_bucket, to_bucket, std::move(res_inc), cost_inc, priority});
        } else {
            bw_arcs.push_back({from_bucket, to_bucket, std::move(res_inc), cost_inc, priority});
        }
    }

    /**
     * @brief Sorts the forward and backward arcs based on their priority.
     *
     * This function sorts the `fw_arcs` in descending order of priority and
     * the `bw_arcs` in ascending order of priority.
     */
    void sort_arcs() {
        std::sort(fw_arcs.begin(), fw_arcs.end(), [](const Arc &a, const Arc &b) { return a.priority > b.priority; });
        std::sort(bw_arcs.begin(), bw_arcs.end(), [](const Arc &a, const Arc &b) { return a.priority < b.priority; });
    }

    /**
     * @brief Sets the location coordinates.
     *
     * This function sets the x and y coordinates for the location.
     */
    void set_location(double x, double y) {
        this->x = x;
        this->y = y;
    }

    /**
     * @brief Retrieves a constant reference to the vector of arcs based on the specified direction.
     *
     * This function template returns a constant reference to either the forward arcs or backward arcs
     * vector, depending on the direction specified by the template parameter.
     *
     */
    template <Direction dir>
    inline const std::vector<Arc> &get_arcs() const {
        if constexpr (dir == Direction::Forward) {
            return fw_arcs;
        } else {
            return bw_arcs;
        }
    }

    /**
     * @brief Retrieves the arcs associated with a given strongly connected component (SCC) in the specified
     * direction.
     *
     */
    template <Direction dir>
    inline const std::vector<Arc> &get_arcs(int scc) const {
        if constexpr (dir == Direction::Forward) {
            return fw_arcs_scc[scc];
        } else {
            return bw_arcs_scc[scc];
        }
    }

    /**
     * @brief Clears all forward and backward arcs.
     *
     * This function empties the containers holding the forward arcs (fw_arcs)
     * and backward arcs (bw_arcs), effectively removing all stored arcs.
     */
    void clear_arcs() {
        fw_arcs.clear();
        bw_arcs.clear();
    }

    // define setDuals method
    void setDuals(double d) { cost = d; }
};

/**
 * @brief Logs debug information to a file.
 *
 * This function appends the provided debug information to a file named "debug_info.txt".
 * If the file does not exist, it will be created. If the file is already open, the
 * information will be appended to the end of the file.
 *
 */
inline void log_debug_info(const std::string &info) {
    std::ofstream debug_file("debug_info.txt", std::ios_base::app);
    if (debug_file.is_open()) { debug_file << info << std::endl; }
}

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
using ArcVariant   = std::variant<Arc, BucketArc>;

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

/**
 * @class TreeNode
 * @brief Represents a node in a multi-dimensional tree structure.
 *
 */
class TreeNode {
public:
    std::vector<int> low;          // Lower bounds for each dimension
    std::vector<int> high;         // Upper bounds for each dimension
    int              bucket_index; // Bucket index for this node
    TreeNode        *left;
    TreeNode        *right;
    TreeNode        *parent;

    TreeNode(const std::vector<int> &low, const std::vector<int> &high, int bucket_index)
        : low(low), high(high), bucket_index(bucket_index), left(nullptr), right(nullptr), parent(nullptr) {}

    bool contains(const std::vector<int> &point) const {
        for (size_t i = 0; i < low.size(); ++i) {
            if (point[i] < low[i] || point[i] > high[i]) { return false; }
        }
        return true;
    }

    bool is_less_than(const std::vector<int> &point) const {
        for (size_t i = 0; i < low.size(); ++i) {
            if (high[i] < point[i]) {
                return true;
            } else if (low[i] > point[i]) {
                return false;
            }
        }
        return false; // This case shouldn't be reached if comparing proper intervals
    }
};

/**
 * @class SplayTree
 * @brief A class representing a Splay Tree, which is a self-adjusting binary search tree.
 *
 * The Splay Tree supports efficient insertion, deletion, and search operations by performing
 * splay operations that move accessed nodes closer to the root, thereby improving access times
 * for frequently accessed nodes.
 *
 * The tree nodes store intervals [low, high] and a bucket index associated with each interval.
 *
 */
class SplayTree {
    TreeNode *root;

    void zig(TreeNode *x) {
        TreeNode *p = x->parent;
        TreeNode *B = (p->left == x) ? x->right : x->left;

        x->parent = p->parent;
        p->parent = x;

        if (p->left == x) {
            x->right = p;
            p->left  = B;
        } else {
            x->left  = p;
            p->right = B;
        }

        if (B != nullptr) B->parent = p;
    }

    void zig_zig(TreeNode *x) {
        TreeNode *p = x->parent;
        TreeNode *g = p->parent;
        if (p->left == x) {
            TreeNode *B = x->right;
            TreeNode *C = p->right;

            x->parent = g->parent;
            x->right  = p;

            p->parent = x;
            p->left   = B;
            p->right  = g;

            g->parent = p;
            g->left   = C;

            if (x->parent != nullptr) {
                if (x->parent->left == g)
                    x->parent->left = x;
                else
                    x->parent->right = x;
            }

            if (B != nullptr) B->parent = p;
            if (C != nullptr) C->parent = g;
        } else {
            TreeNode *B = p->left;
            TreeNode *C = x->left;

            x->parent = g->parent;
            x->left   = p;

            p->parent = x;
            p->left   = g;
            p->right  = C;

            g->parent = p;
            g->right  = B;

            if (x->parent != nullptr) {
                if (x->parent->left == g)
                    x->parent->left = x;
                else
                    x->parent->right = x;
            }

            if (B != nullptr) B->parent = g;
            if (C != nullptr) C->parent = p;
        }
    }

    void zig_zag(TreeNode *x) {
        TreeNode *p = x->parent;
        TreeNode *g = p->parent;

        if (p->right == x) {
            TreeNode *B = x->left;
            TreeNode *C = x->right;

            x->parent = g->parent;
            x->left   = p;
            x->right  = g;

            p->parent = x;
            p->right  = B;

            g->parent = x;
            g->left   = C;

            if (x->parent != nullptr) {
                if (x->parent->left == g)
                    x->parent->left = x;
                else
                    x->parent->right = x;
            }

            if (B != nullptr) B->parent = p;
            if (C != nullptr) C->parent = g;
        } else {
            TreeNode *B = x->left;
            TreeNode *C = x->right;

            x->parent = g->parent;
            x->left   = g;
            x->right  = p;

            p->parent = x;
            p->left   = C;

            g->parent = x;
            g->right  = B;

            if (x->parent != nullptr) {
                if (x->parent->left == g)
                    x->parent->left = x;
                else
                    x->parent->right = x;
            }

            if (B != nullptr) B->parent = g;
            if (C != nullptr) C->parent = p;
        }
    }

    void rotate(TreeNode *x) {
        TreeNode *p = x->parent;
        TreeNode *g = p->parent;

        if (p->left == x) { // Left child
            p->left = x->right;
            if (x->right) x->right->parent = p;
            x->right = p;
        } else { // Right child
            p->right = x->left;
            if (x->left) x->left->parent = p;
            x->left = p;
        }

        x->parent = g;
        p->parent = x;

        if (g) {
            if (g->left == p)
                g->left = x;
            else
                g->right = x;
        }
    }

    void splay(TreeNode *x) {
        while (x->parent != nullptr) {
            TreeNode *p = x->parent;
            TreeNode *g = p->parent;

            if (g == nullptr) {
                // Zig step (single rotation)
                rotate(x);
            } else if ((g->left == p && p->left == x) || (g->right == p && p->right == x)) {
                // Zig-zig step (double rotation)
                rotate(p); // First rotate parent
                rotate(x); // Then rotate x
            } else {
                // Zig-zag step (rotating x twice in different directions)
                rotate(x); // Rotate x first
                rotate(x); // Then rotate x again
            }
        }
        root = x;
    }

public:
    SplayTree() : root(nullptr) {}

    TreeNode *find(const std::vector<int> &point) {
        TreeNode *curr = root;

        while (curr != nullptr) {
            if (curr->contains(point)) {
                splay(curr); // Splay only if we find the node
                return curr;
            } else if (curr->is_less_than(point)) {
                curr = curr->right;
            } else {
                curr = curr->left;
            }
        }

        // If not found, no splaying needed for the closest node
        return nullptr;
    }

    int query(const std::vector<int> &point) {
        TreeNode *node = find(point);
        if (node != nullptr) return node->bucket_index;
        return -1;
    }

    // Insert a new multidimensional interval
    void insert(const std::vector<int> &low, const std::vector<int> &high, int bucket_index) {

        if (root == nullptr) {
            root = new TreeNode(low, high, bucket_index);
            return;
        }

        TreeNode *curr = root;
        while (curr != nullptr) {
            if (low < curr->low) {
                if (curr->left == nullptr) {
                    TreeNode *newNode = new TreeNode(low, high, bucket_index);
                    curr->left        = newNode;
                    newNode->parent   = curr;
                    splay(newNode);
                    return;
                } else {
                    curr = curr->left;
                }
            } else if (low > curr->low) {
                if (curr->right == nullptr) {
                    TreeNode *newNode = new TreeNode(low, high, bucket_index);
                    curr->right       = newNode;
                    newNode->parent   = curr;
                    splay(newNode);
                    return;
                } else {
                    curr = curr->right;
                }
            } else {
                splay(curr);
                return; // Duplicate interval
            }
        }
    }

    void inOrderPrint(TreeNode *node) {
        if (node == nullptr) return;
        inOrderPrint(node->left);
        std::cout << "Bucket " << node->bucket_index << ": [";
        for (size_t i = 0; i < node->low.size(); ++i) {
            std::cout << "[" << node->low[i] << ", " << node->high[i] << "]";
            if (i != node->low.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
        inOrderPrint(node->right);
    }

    void print() { inOrderPrint(root); }
};

#define CONDITIONAL(D, FW_ACTION, BW_ACTION)         \
    if constexpr (D == Direction::Forward) {         \
        FW_ACTION;                                   \
    } else if constexpr (D == Direction::Backward) { \
        BW_ACTION;                                   \
    }
