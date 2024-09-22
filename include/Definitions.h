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
 * - Structures for Interval, RCCarc, RCCcut, RCCmanager, Path, and Label.
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

enum class Direction { Forward, Backward };
enum class Stage { One, Two, Three, Four, Enumerate, Fix };
enum class ArcType { Job, Bucket, Jump };
enum class Mutability { Const, Mut };
enum class Full { Full, Partial };
enum class Status { Optimal, Separation, NotOptimal, Error };

// Comparator function for Stage enum
constexpr bool operator<(Stage lhs, Stage rhs) { return static_cast<int>(lhs) < static_cast<int>(rhs); }
constexpr bool operator>(Stage lhs, Stage rhs) { return rhs < lhs; }
constexpr bool operator<=(Stage lhs, Stage rhs) { return !(lhs > rhs); }
constexpr bool operator>=(Stage lhs, Stage rhs) { return !(lhs < rhs); }

class CutStorage;

/**
 * @struct Interval
 * @brief Represents an interval with a duration and a horizon.
 *
 * The Interval struct is used to store information about an interval, which consists of a duration and a horizon.
 * The duration is represented by a double value, while the horizon is represented by an integer value.
 */
struct Interval {
    int interval;
    int horizon;

    Interval(double interval, int horizon) : interval(interval), horizon(horizon) {}
};

/**
 * @struct RCCarc
 * @brief Represents an arc with two endpoints.
 *
 * The RCCarc struct is used to represent an arc with two endpoints, `from` and `to`.
 * It provides a constructor for initializing these endpoints and an equality operator
 * to compare two RCCarc objects.
 *
 */
struct RCCarc {
    int from;
    int to;

    RCCarc(int from, int to) : from(from), to(to) {}

    // Define operator== to compare two RCCarc objects
    bool operator==(const RCCarc &other) const {
        return (from == other.from && to == other.to) || (from == other.to && to == other.from);
    }
};

/**
 * @struct RCCcut
 * @brief Represents a cut in the RCC (Resource Constrained Cut) problem.
 *
 * The RCCcut structure holds information about a specific cut, including its arcs,
 * right-hand side value, identifier, associated constraint, and dual value.
 *
 */
struct RCCcut {
    std::vector<RCCarc> arcs;
    double              rhs;
    int                 id;
    GRBConstr           constr;
    double              dual = 0.0;

    RCCcut(const std::vector<RCCarc> &arcs, double rhs) : arcs(arcs), rhs(rhs) {}
    RCCcut(const std::vector<RCCarc> &arcs, double rhs, GRBConstr constr) : arcs(arcs), rhs(rhs), constr(constr) {}
    RCCcut(const std::vector<RCCarc> &arcs, double rhs, int id) : arcs(arcs), rhs(rhs), id(id) {}
    RCCcut(const std::vector<RCCarc> &arcs, double rhs, int id, GRBConstr constr)
        : arcs(arcs), rhs(rhs), id(id), constr(constr) {}
};

/**
 * @struct RCCarcHash
 * @brief A hash function object for RCCarc structures.
 *
 * This struct defines a custom hash function for RCCarc objects, which can be used
 * in hash-based containers like std::unordered_map or std::unordered_set.
 *
 * @note The hash function combines the hash values of the 'from' and 'to' members
 *       of the RCCarc structure using the XOR (^) operator.
 *
 * @param arc The RCCarc object to be hashed.
 * @return A std::size_t value representing the hash of the given RCCarc object.
 */
struct RCCarcHash {
    std::size_t operator()(const RCCarc &arc) const { return std::hash<int>()(arc.from) ^ std::hash<int>()(arc.to); }
};

/**
 * @struct RCCmanager
 * @brief Manages RCC cuts, dual cache, and arc-to-cut mappings with thread safety.
 *
 * This struct provides methods to add cuts, compute duals, and manage a cache of dual sums
 * for arcs. It supports parallel processing for efficiency.
 *
 */
struct RCCmanager {
    std::vector<RCCcut>                                                       cuts;
    int                                                                       cut_counter = 0;
    std::unordered_map<RCCarc, double, RCCarcHash>                            dualCache;
    std::unordered_map<std::pair<int, int>, std::vector<RCCcut *>, pair_hash> arcCutMap;
    std::mutex cache_mutex; // Protect shared resources during concurrent access

    RCCmanager() = default;

    // Add a cut to the list and update the arc-to-cut mapping
    void addCut(const std::vector<RCCarc> &arcs, double rhs, GRBConstr constr = GRBConstr()) {
        cuts.emplace_back(arcs, rhs, cut_counter++, constr);
        for (const auto &cutArc : arcs) { arcCutMap[{cutArc.from, cutArc.to}].push_back(&cuts.back()); }
    }

    // Bulk addition of cuts with parallel processing for efficiency
    void addCutBulk(const std::vector<std::vector<RCCarc>> &arcsVec, const std::vector<int> &rhsVec,
                    const std::vector<GRBConstr> &constrVec) {
        assert(arcsVec.size() == rhsVec.size() && rhsVec.size() == constrVec.size());

        cuts.reserve(cuts.size() + arcsVec.size());

        for (size_t i = 0; i < arcsVec.size(); ++i) {
            cuts.emplace_back(arcsVec[i], rhsVec[i], cut_counter++, constrVec[i]);
            for (const auto &cutArc : arcsVec[i]) {
                auto &cutList = arcCutMap[{cutArc.from, cutArc.to}];
                cutList.emplace_back(&cuts.back());
            }
        }
    }

    // Compute duals for cuts, remove small dual cuts, and update cache in parallel
    void computeDualsDeleteAndCache(GRBModel *model) {
        dualCache.clear();
        std::vector<RCCcut *> toRemoveFromCache;
        std::vector<int>      toRemoveIndices;
        const int             JOBS = 10;

        exec::static_thread_pool pool(JOBS);
        auto                     sched = pool.get_scheduler();

        // Step 1: Parallel gathering of cuts for removal
        auto gather_cuts = stdexec::bulk(stdexec::just(), cuts.size(),
                                         [this, model, &toRemoveFromCache, &toRemoveIndices](std::size_t i) {
                                             RCCcut &cut = cuts[i];
                                             cut.dual    = cut.constr.get(GRB_DoubleAttr_Pi);

                                             if (std::abs(cut.dual) > 1e6 || std::isnan(cut.dual)) { cut.dual = 0.0; }

                                             if (std::abs(cut.dual) < 1e-6) {
                                                 std::lock_guard<std::mutex> lock(cache_mutex);
                                                 toRemoveFromCache.push_back(&cut);
                                                 toRemoveIndices.push_back(i);
                                             }
                                         });

        auto gather_work = stdexec::starts_on(sched, gather_cuts);
        stdexec::sync_wait(std::move(gather_work));

        // Step 2: Parallel removal of cuts from the cache
        auto remove_cuts =
            stdexec::bulk(stdexec::just(), toRemoveFromCache.size(), [this, &toRemoveFromCache](std::size_t i) {
                RCCcut *cut = toRemoveFromCache[i];
                for (const auto &cutArc : cut->arcs) {
                    auto &cutList = arcCutMap[{cutArc.from, cutArc.to}];
                    cutList.erase(std::remove(cutList.begin(), cutList.end(), cut), cutList.end());

                    if (cutList.empty()) {
                        std::lock_guard<std::mutex> lock(cache_mutex);
                        dualCache[{cutArc.from, cutArc.to}] = 0.0;
                        arcCutMap.erase({cutArc.from, cutArc.to});
                    }
                }
            });

        auto remove_work = stdexec::starts_on(sched, remove_cuts);
        stdexec::sync_wait(std::move(remove_work));

        // Step 3: Bulk erasure of cuts from the cut vector
        std::sort(toRemoveIndices.begin(), toRemoveIndices.end(), std::greater<>());
        for (int idx : toRemoveIndices) {
            // fmt::print("Erasing cut {}\n", idx);
            cuts.erase(cuts.begin() + idx);
        }

        // Step 4: Parallel dual sum computation for arcs
        auto compute_duals = stdexec::bulk(stdexec::just(), arcCutMap.size(), [this](std::size_t idx) {
            const auto &[arcKey, cutList] = *std::next(arcCutMap.begin(), idx);
            double dualSum                = 0.0;

            for (const auto *cut : cutList) {
                dualSum += cut->dual;
                if (std::abs(dualSum) > 1e6) {
                    dualSum = 0.0;
                    break;
                }
            }

            std::lock_guard<std::mutex> lock(cache_mutex);
            dualCache[{arcKey.first, arcKey.second}] = dualSum;
        });

        auto compute_duals_work = stdexec::starts_on(sched, compute_duals);
        stdexec::sync_wait(std::move(compute_duals_work));

        // Step 5: Update the model
        model->update();
    }

    // Function to retrieve cached dual sum for a given arc
    double getCachedDualSumForArc(int from, int to) {
        RCCarc arc(from, to);
        auto   it = dualCache.find(arc);
        if (it == dualCache.end()) { return 0.0; }

        double cachedDual = it->second;
        if (std::abs(cachedDual) > 1e6 || std::isnan(cachedDual)) { return 0.0; }
        return cachedDual;
    }

    // Method to retrieve all cuts (if needed)
    std::vector<RCCcut> getAllCuts() const { return cuts; }
};

/**
 * @struct Path
 * @brief Represents a path with a route and its associated cost.
 *
 * The Path struct encapsulates a route represented as a vector of integers and a cost associated with the route.
 * It provides various utility methods to interact with the route, such as checking for the presence of elements,
 * counting occurrences, and managing arcs between route points.
 *
 */
struct Path {
    std::vector<int> route;
    double           cost;

    // default constructor
    Path() : route({}), cost(0.0) {}
    Path(const std::vector<int> &route, double cost) : route(route), cost(cost) {
        (void)std::async(std::launch::async, &Path::precomputeArcs, this); // Ignore the future
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
     * @param i The integer to search for in the route.
     * @return true if the integer is found in the route, false otherwise.
     */
    bool contains(int i) { return std::find(route.begin(), route.end(), i) != route.end(); }

    /**
     * @brief Counts the occurrences of a given integer in the route.
     *
     * This function iterates through the 'route' container and counts how many times
     * the specified integer 'i' appears in it.
     *
     * @param i The integer value to count within the route.
     * @return int The number of times the integer 'i' appears in the route.
     */
    int countOccurrences(int i) { return std::count(route.begin(), route.end(), i); }

    /**
     * @brief Counts the number of times an arc (i, j) appears in the route.
     *
     * This function iterates through the route and counts how many times the arc
     * from node i to node j appears consecutively.
     *
     * @param i The starting node of the arc.
     * @param j The ending node of the arc.
     * @return The number of times the arc (i, j) appears in the route.
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
     * @param i The starting node of the arc.
     * @param j The ending node of the arc.
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
     * @param i The first node of the arc.
     * @param j The second node of the arc.
     * @return The count of arcs between node i and node j. Returns 0 if the arc
     *         does not exist in the arcMap.
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
     * @param arc The RCCarc object representing the arc whose count is to be retrieved.
     * @return The count of the specified arc if it exists in the arcMap, otherwise 0.
     */
    auto getArcCount(RCCarc arc) const {
        // Construct the arc pair
        std::pair<int, int> arcPair = std::make_pair(arc.from, arc.to);
        return (arcMap.find(arcPair) != arcMap.end()) ? arcMap.at(arcPair) : 0;
    }
};
// Label structure to hold the details of each label in the graph
/**
 * @struct Label
 * @brief Represents a label used in a solver.
 *
 * This struct contains various properties and methods related to a label used in a solver.
 * It stores information such as the set of Fj, id, is_extended flag, vertex, cost, real_cost, SRC_cost,
 * resources, predecessor, is_dominated flag, jobs_covered, jobs_ordered, job_id, cut_storage,
 * parent, children, status, visited, and SRCmap.
 *
 * The struct provides constructors to initialize the label with or without a job_id.
 * It also provides methods to set the covered jobs, add a job to the covered jobs, check if a job is
 * already covered, and initialize the label with new values.
 *
 * The struct overloads the equality and greater than operators for comparison.
 */
const size_t num_words = (N_SIZE + 63) / 64; // This will be 2 for 100 clients
struct Label {
    // int                   id;
    bool                       is_extended = false;
    int                        vertex;
    double                     cost         = 0.0;
    double                     real_cost    = 0.0;
    std::array<double, R_SIZE> resources    = {};
    std::vector<int>           jobs_covered = {}; // Add jobs_covered to Label
    int                        job_id       = -1; // Add job_id to Label
    const Label               *parent       = nullptr;
#ifdef SRC3
    std::array<std::uint16_t, MAX_SRC_CUTS> SRCmap = {};
#endif
#ifdef SRC
    std::vector<double> SRCmap;
#endif
    // uint64_t             visited_bitmap; // Bitmap for visited jobs
    std::array<uint64_t, num_words> visited_bitmap = {0};
#ifdef UNREACHABLE_DOMINANCE
    std::array<uint64_t, num_words> unreachable_bitmap = {0};
#endif

    // Constructor with job_id
    Label(int v, double c, const std::vector<double> &res, int pred, int job_id)
        : vertex(v), cost(c), resources({res[0]}), job_id(job_id) {}

    // Constructor without job_id
    Label(int v, double c, const std::vector<double> &res, int pred)
        : vertex(v), cost(c), resources({res[0]}), job_id(-1) {}

    // Default constructor
    Label() : vertex(-1), cost(0), resources({0.0}), job_id(-1) {}

    void set_extended(bool extended) { is_extended = extended; }

    /**
     * @brief Checks if a job has been visited.
     *
     * This function determines whether a job, identified by its job_id, has been visited.
     * It uses a bitmask (visited_bitmap) where each bit represents the visit status of a job.
     *
     * @param job_id The identifier of the job to check.
     * @return true if the job has been visited, false otherwise.
     */
    bool visits(int job_id) const { return visited_bitmap[job_id / 64] & (1ULL << (job_id % 64)); }

    /**
     * @brief Resets the state of the object to its initial values.
     *
     * This function sets the following member variables to their default values:
     * - vertex: -1
     * - cost: 0.0
     * - resources: an empty container
     * - job_id: -1
     * - real_cost: 0.0
     * - parent: nullptr
     * - is_extended: false
     * - status: 0
     * - jobs_covered: cleared
     *
     * Additionally, it zeroes out the following bitmaps if the corresponding macros are defined:
     * - visited_bitmap
     * - unreachable_bitmap (if UNREACHABLE_DOMINANCE is defined)
     * - SRCmap (if SRC3 or SRC is defined)
     *
     * If the SRC macro is defined, it also sets SRCcost to 0.0.
     */
    inline void reset() {
        this->vertex    = -1;
        this->cost      = 0.0;
        this->resources = {};
        // this->job_id      = -1;
        this->real_cost   = 0.0;
        this->parent      = nullptr;
        this->is_extended = false;
        // this->jobs_covered.clear();

        std::memset(visited_bitmap.data(), 0, visited_bitmap.size() * sizeof(uint64_t));
#ifdef UNREACHABLE_DOMINANCE
        std::memset(unreachable_bitmap.data(), 0, unreachable_bitmap.size() * sizeof(uint64_t));
#endif
#ifdef SRC3
        std::memset(SRCmap.data(), 0, SRCmap.size() * sizeof(std::uint16_t));
#endif
#ifdef SRC
        SRCmap.clear();
#endif
    }

    void addJob(int job) { jobs_covered.push_back(job); }

    /**
     * @brief Initializes the object with the given parameters.
     *
     * @param vertex The vertex identifier.
     * @param cost The cost associated with the vertex.
     * @param resources A vector of resource values.
     * @param job_id The job identifier.
     */
    inline void initialize(int vertex, double cost, const std::vector<double> &resources, int job_id) {
        this->vertex = vertex;
        this->cost   = cost;
        for (size_t i = 0; i < resources.size(); ++i) { this->resources[i] = resources[i]; }
        // this->resources = {resources[0], resources[1]};
        this->job_id = job_id;
    }

    bool operator>(const Label &other) const { return cost > other.cost; }

    bool operator<(const Label &other) const { return cost < other.cost; }
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
 * @param from_bucket The index of the source bucket.
 * @param to_bucket The index of the target bucket.
 * @param resource_increment The resource increments associated with the arc.
 * @param cost_increment The cost increment associated with the arc.
 * @param fixed A flag indicating whether the arc is fixed.
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
 * @param base_bucket The index of the base bucket.
 * @param jump_bucket The index of the jump bucket.
 * @param resource_increment The vector of resource increments.
 * @param cost_increment The cost increment.
 */
struct JumpArc {
    int                 base_bucket;
    int                 jump_bucket;
    std::vector<double> resource_increment;
    double              cost_increment;

    JumpArc(int base, int jump, const std::vector<double> &res_inc, double cost_inc);
};

/**
 * @class LabelPool
 * @brief A class that manages a pool of Label objects.
 *
 * The LabelPool class is responsible for managing a pool of Label objects. It provides methods to acquire and
 * release labels from the pool, as well as resetting the pool to its initial state.
 *
 * The pool size is determined during construction and can be optionally limited to a maximum size. Labels can be
 * acquired from the pool using the `acquire()` method, and released back to the pool using the `release()` method.
 * If the pool is full, a new label will be allocated.
 *
 * The `reset()` method can be used to reset the pool to its initial state. This will delete all labels in the pool
 * and reallocate labels to match the initial pool size.
 *
 * @note The LabelPool class is not thread-safe by default. If thread safety is required, appropriate
 * synchronization mechanisms should be used when accessing the pool.
 */
class LabelPool {
public:
    explicit LabelPool(size_t initial_pool_size, size_t max_pool_size = 5000000)
        : pool_size(initial_pool_size), max_pool_size(max_pool_size) {
        allocate_labels(pool_size);
    }
    /**
     * @brief Acquires a Label object for use.
     *
     * This function retrieves a Label object from the pool of available labels if any are present.
     * If the pool is empty, it creates a new Label object. The acquired Label is reset and moved
     * to the in-use list.
     *
     * @return A pointer to the acquired Label object.
     */
    Label *acquire() {
        if (!available_labels.empty()) {
            Label *new_label = available_labels.back();
            available_labels.pop_back();
            in_use_labels.push_back(new_label);
            new_label->reset();
            return new_label;
        }

        auto *new_label = new Label();
        in_use_labels.push_back(new_label);
        return new_label;
    }

    // Reserve space in available_labels to prevent multiple reallocations
    /**
     * @brief Resets the state by moving all labels from in-use to available.
     *
     * This function performs the following operations:
     * 1. Reserves additional space in the `available_labels` vector to prevent multiple reallocations.
     * 2. Moves all labels from the `in_use_labels` vector to the `available_labels` vector.
     * 3. Clears the `in_use_labels` vector after moving the labels.
     *
     * Note: The parallel reset of labels is commented out and not currently in use.
     */
    void reset() {
        // Reserve space in available_labels to prevent multiple reallocations
        available_labels.reserve(available_labels.size() + in_use_labels.size());

        // Parallel reset of labels
        // std::for_each(std::execution::par, in_use_labels.begin(), in_use_labels.end(), [](Label *label) {
        //    label->reset(); // Reset each label in parallel
        //});

        // Move labels from in_use_labels to available_labels in bulk
        available_labels.insert(available_labels.end(), std::make_move_iterator(in_use_labels.begin()),
                                std::make_move_iterator(in_use_labels.end()));

        // Clear in-use labels after moving
        in_use_labels.clear();
    }

private:
    /**
     * @brief Allocates a specified number of labels.
     *
     * This function reserves space for a given number of labels in the
     * available_labels container and initializes each label by creating
     * a new Label object.
     *
     * @param count The number of labels to allocate.
     */
    void allocate_labels(size_t count) {
        available_labels.reserve(count);
        for (size_t i = 0; i < count; ++i) { available_labels.push_back(new Label()); }
    }

    /**
     * @brief Cleans up and deallocates memory for all labels and states.
     *
     * This function iterates through the containers `available_labels`,
     * `in_use_labels`, and `deleted_states`, deleting each label and
     * clearing the containers to free up memory.
     */
    void cleanup() {
        for (auto &label : available_labels) { delete label; }
        available_labels.clear();

        for (auto &label : in_use_labels) { delete label; }
        in_use_labels.clear();

        for (auto &label : deleted_states) { delete label; }
        deleted_states.clear();
    }

    size_t pool_size;
    size_t max_pool_size;

    std::vector<Label *> available_labels; // Labels ready to be acquired
    std::vector<Label *> in_use_labels;    // Labels currently in use
    std::deque<Label *>  deleted_states;   // Labels available for recycling
};

/**
 * @struct Bucket
 * @brief Represents a bucket.
 *
 * A bucket is a data structure that contains labels, job ID, lower bounds, upper bounds, forward arcs, backward
 * arcs, forward jump arcs, and backward jump arcs. It provides methods to add arcs, add jump arcs, get arcs, get
 * jump arcs, add labels, remove labels, get labels, clear labels, reset labels, and clear arcs.
 */
struct Bucket {
    // std::vector<Label *>   labels_vec;
    std::vector<Label *> labels_vec; // Use deque for efficient insertion/removal

    int                    job_id = -1;
    std::vector<int>       lb;
    std::vector<int>       ub;
    std::vector<Arc>       fw_arcs;
    std::vector<Arc>       bw_arcs;
    std::vector<BucketArc> fw_bucket_arcs;
    std::vector<BucketArc> bw_bucket_arcs;
    std::vector<JumpArc>   fw_jump_arcs;
    std::vector<JumpArc>   bw_jump_arcs;

    /**
     * @brief Deletes a bucket arc from the specified direction.
     *
     * This function removes a bucket arc from either the forward or backward bucket arcs list,
     * depending on the value of the `fw` parameter. The arc to be removed is identified by the
     * `from_bucket` and `to_bucket` parameters.
     *
     * @param from_bucket The starting bucket of the arc to be deleted.
     * @param to_bucket The ending bucket of the arc to be deleted.
     * @param fw A boolean indicating the direction of the arc. If true, the arc is removed from
     *           the forward bucket arcs list. If false, the arc is removed from the backward
     *           bucket arcs list.
     */
    void delete_bucket_arc(int from_bucket, int to_bucket, bool fw) {
        if (fw) {
            fw_bucket_arcs.erase(std::remove_if(fw_bucket_arcs.begin(), fw_bucket_arcs.end(),
                                                [from_bucket, to_bucket](const BucketArc &arc) {
                                                    return arc.from_bucket == from_bucket && arc.to_bucket == to_bucket;
                                                }),
                                 fw_bucket_arcs.end());
        } else {
            bw_bucket_arcs.erase(std::remove_if(bw_bucket_arcs.begin(), bw_bucket_arcs.end(),
                                                [from_bucket, to_bucket](const BucketArc &arc) {
                                                    return arc.from_bucket == from_bucket && arc.to_bucket == to_bucket;
                                                }),
                                 bw_bucket_arcs.end());
        }
    }

    /**
     * @brief Adds an arc between two buckets.
     *
     * This function adds an arc from one bucket to another, either in the forward or backward direction.
     * The arc is characterized by resource increments, cost increment, and whether it is fixed or not.
     *
     * @param from_bucket The index of the source bucket.
     * @param to_bucket The index of the destination bucket.
     * @param res_inc A vector of resource increments associated with the arc.
     * @param cost_inc The cost increment associated with the arc.
     * @param fw A boolean indicating the direction of the arc. If true, the arc is forward; otherwise, it is backward.
     * @param fixed A boolean indicating whether the arc is fixed.
     */
    void add_bucket_arc(int from_bucket, int to_bucket, std::vector<double> res_inc, double cost_inc, bool fw,
                        bool fixed) {
        if (fw) {
            fw_bucket_arcs.push_back({from_bucket, to_bucket, std::move(res_inc), cost_inc, fixed});
        } else {
            bw_bucket_arcs.push_back({from_bucket, to_bucket, std::move(res_inc), cost_inc, fixed});
        }
    }

    /**
     * @brief Adds a jump arc between two buckets.
     *
     * This function adds a jump arc from one bucket to another with the specified resource increment and cost
     * increment. The direction of the jump arc is determined by the `fw` parameter.
     *
     * @param from_bucket The index of the source bucket.
     * @param to_bucket The index of the destination bucket.
     * @param res_inc A vector of resource increments associated with the jump arc.
     * @param cost_inc The cost increment associated with the jump arc.
     * @param fw A boolean indicating the direction of the jump arc. If true, the arc is added to the forward jump arcs;
     *           otherwise, it is added to the backward jump arcs.
     */
    void add_jump_arc(int from_bucket, int to_bucket, std::vector<double> res_inc, double cost_inc, bool fw) {
        if (fw) {
            fw_jump_arcs.push_back({from_bucket, to_bucket, std::move(res_inc), cost_inc});
        } else {
            bw_jump_arcs.push_back({from_bucket, to_bucket, std::move(res_inc), cost_inc});
        }
    }

    /**
     * @brief Retrieves a reference to the vector of arcs based on the specified direction.
     *
     * This function template returns a reference to either the forward arcs (fw_arcs)
     * or the backward arcs (bw_arcs) depending on the template parameter `dir`.
     *
     * @tparam dir The direction for which to retrieve the arcs. It can be either
     *             Direction::Forward or Direction::Backward.
     * @return std::vector<Arc>& A reference to the vector of arcs corresponding to the specified direction.
     */
    template <Direction dir>
    std::vector<Arc> &get_arcs() {
        if constexpr (dir == Direction::Forward) {
            return fw_arcs;
        } else {
            return bw_arcs;
        }
    }

    template <Direction dir>
    std::vector<BucketArc> &get_bucket_arcs() {
        if constexpr (dir == Direction::Forward) {
            return fw_bucket_arcs;
        } else {
            return bw_bucket_arcs;
        }
    }

    template <Direction dir>
    std::vector<JumpArc> &get_jump_arcs() {
        if constexpr (dir == Direction::Forward) {
            return fw_jump_arcs;
        } else {
            return bw_jump_arcs;
        }
    }

    Bucket(int job_id, std::vector<int> lb, std::vector<int> ub)
        : job_id(job_id), lb(std::move(lb)), ub(std::move(ub)) {

        labels_vec.reserve(250);
    }

    // create default constructor
    Bucket() { labels_vec.reserve(250); }

    /**
     * @brief Adds a label to the labels vector.
     *
     * This function adds a label to the labels vector. The label is currently added to the end of the vector.
     *
     * @param label Pointer to the Label object to be added.
     */
    void add_label(Label *label) noexcept { labels_vec.push_back(label); }

    /**
     * @brief Adds a label to the labels_vec in sorted order based on the cost.
     *
     * This function inserts the given label into the labels_vec such that the vector
     * remains sorted in ascending order of the label's cost. The insertion is done
     * using binary search to find the appropriate position, ensuring efficient insertion.
     *
     * @param label Pointer to the Label object to be added.
     */
    void add_sorted_label(Label *label) noexcept {
        if (labels_vec.empty() || label->cost >= labels_vec.back()->cost) {
            labels_vec.push_back(label); // Direct insertion at the end
        } else if (label->cost <= labels_vec.front()->cost) {
            labels_vec.insert(labels_vec.begin(), label); // Direct insertion at the beginning
        } else {
            auto it = std::lower_bound(labels_vec.begin(), labels_vec.end(), label,
                                       [](const Label *a, const Label *b) { return a->cost > b->cost; });
            labels_vec.insert(it, label); // Insertion in the middle
        }
    }

    /**
     * @brief Adds a label to the labels vector with a limit on the number of labels.
     *
     * This function attempts to add a given label to the labels vector. If the vector
     * has not yet reached the specified limit, the label is simply added. If the vector
     * has reached the limit, the function will replace the label with the highest cost
     * if the new label has a lower cost.
     *
     * @param label Pointer to the label to be added.
     * @param limit The maximum number of labels allowed in the vector.
     */
    void add_label_lim(Label *label, size_t limit) noexcept {
        if (labels_vec.size() < limit) {
            labels_vec.push_back(label);
        } else {
            auto it = std::max_element(labels_vec.begin(), labels_vec.end(),
                                       [](const Label *a, const Label *b) { return a->cost < b->cost; });
            if (label->cost < (*it)->cost) { *it = label; }
        }
    }

    /**
     * @brief Removes a label from the labels vector.
     *
     * This function searches for the specified label in the labels vector.
     * If found, it replaces the label with the last element in the vector
     * and then removes the last element, effectively removing the specified label.
     *
     * @param label Pointer to the label to be removed.
     */
    void remove_label(Label *label) noexcept {
        auto it = std::find(labels_vec.begin(), labels_vec.end(), label);
        if (it != labels_vec.end()) {
            // Move the last element to the position of the element to remove
            *it = std::move(labels_vec.back());
            labels_vec.pop_back(); // Remove the last element
        }
    }

    // std::vector<Label *> &get_labels() { return labels_vec; }
    inline auto &get_labels() { return labels_vec; }

    inline auto &get_sorted_labels() {
        std::sort(labels_vec.begin(), labels_vec.end(),
                  [](const Label *a, const Label *b) { return a->cost > b->cost; });
        return labels_vec;
    }

    inline const auto get_unextended_labels() {
        return labels_vec | std::views::filter([](Label *label) { return !label->is_extended; });
    }

    void clear() { labels_vec.clear(); }

    void reset() { labels_vec.clear(); }

    /**
     * @brief Clears the arcs in the specified direction.
     *
     * This function clears the arcs in either the forward or backward direction
     * based on the input parameter.
     *
     * @param fw A boolean value indicating the direction of arcs to clear.
     *           - If true, clears the forward bucket arcs.
     *           - If false, clears the backward bucket arcs.
     */
    void clear_arcs(bool fw) {
        if (fw) {
            fw_bucket_arcs.clear();
        } else {
            bw_bucket_arcs.clear();
        }
    }
    /**
     * @brief Retrieves the best label from the labels vector.
     *
     * This function returns the first label in the labels vector if it is not empty.
     * If the vector is empty, it returns a nullptr.
     *
     * @return A pointer to the best label, or nullptr if the labels vector is empty.
     */
    Label *get_best_label() {
        if (labels_vec.empty()) return nullptr;
        return labels_vec.front();
    }

    bool empty() const { return labels_vec.empty(); }
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
 * This struct contains information about a job, such as its ID, start time, end time, duration, cost, demand, and
 * capacity constraints. It provides constructors to initialize the job with different sets of parameters. The
 * `setDuals` method allows updating the cost of the job.
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
     * @param from_bucket The starting bucket of the arc.
     * @param to_bucket The ending bucket of the arc.
     * @param res_inc A vector of resource increments associated with the arc.
     * @param cost_inc The cost increment associated with the arc.
     * @param fw A boolean flag indicating whether the arc is forward (true) or backward (false).
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
     * @param from_bucket The index of the source bucket.
     * @param to_bucket The index of the destination bucket.
     * @param res_inc A vector of resource increments associated with the arc.
     * @param cost_inc The cost increment associated with the arc.
     * @param fw A boolean indicating the direction of the arc. If true, the arc is added
     *           to the forward arc list; otherwise, it is added to the backward arc list.
     * @param fixed A boolean indicating whether the arc is fixed.
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
     *
     * @param x The x-coordinate to set.
     * @param y The y-coordinate to set.
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
     * @tparam dir The direction for which to retrieve the arcs. It can be either Direction::Forward or
     * Direction::Backward.
     * @return const std::vector<Arc>& A constant reference to the vector of arcs corresponding to the specified
     * direction.
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
     * @brief Retrieves the arcs associated with a given strongly connected component (SCC) in the specified direction.
     *
     * @tparam dir The direction of the arcs to retrieve. It can be either Direction::Forward or Direction::Backward.
     * @param scc The index of the strongly connected component.
     * @return const std::vector<Arc>& A reference to the vector of arcs in the specified direction for the given SCC.
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
 * @param info The debug information to be logged.
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
 * @tparam Args Variadic template parameter pack for the format arguments.
 * @param format The format string for the message.
 * @param args The arguments to be formatted and printed according to the format string.
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
 * @tparam Args Variadic template parameter pack for the format arguments.
 * @param format The format string for the message.
 * @param args The arguments to be formatted and printed according to the format string.
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
 * @tparam Args The types of the arguments to be formatted.
 * @param format The format string.
 * @param args The arguments to be formatted according to the format string.
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
 * @tparam Args The types of the arguments to be formatted.
 * @param format The format string.
 * @param args The arguments to be formatted according to the format string.
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
    fmt::print("| {:<52} |\n", "a C++ implementation");
    fmt::print("| {:<52} |\n", "of a Bucket Graph-based labeling algorithm");
    fmt::print("| {:<52} |\n", "designed to solve the Resource-Constrained");
    fmt::print("| {:<52} |\n", "Shortest Path Problem (RSCPP), which commonly");
    fmt::print("| {:<52} |\n", "arises as a subproblem in state-of-the-art");
    fmt::print("| {:<52} |\n", "Branch-Cut-and-Price algorithms for various");
    fmt::print("| {:<52} |\n", "Vehicle Routing Problems (VRPs).");
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
 * @param model Pointer to the Gurobi model (GRBModel) from which data is to be extracted.
 * @return ModelData structure containing the extracted model data.
 *
 * @note The function assumes that the model is already created and populated with variables and constraints.
 *       It also handles cases where variable bounds are set to very large values by treating them as infinity.
 *
 * @throws GRBException if there is an error during the extraction process.
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

using DualSolution = std::vector<double>;
using ArcVariant   = std::variant<Arc, BucketArc>;
