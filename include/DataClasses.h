#pragma once
#include <deque>
#include <iostream>
#include <tuple>
#include <vector>

#include "Definitions.h"


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
    Label                     *parent       = nullptr;
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

/**
 * @struct Bucket
 * @brief Represents a bucket.
 *
 * A bucket is a data structure that contains labels, job ID, lower bounds, upper bounds, forward arcs, backward
 * arcs, forward jump arcs, and backward jump arcs. It provides methods to add arcs, add jump arcs, get arcs,
 * get jump arcs, add labels, remove labels, get labels, clear labels, reset labels, and clear arcs.
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

    Bucket(const Bucket &other) {
        // Perform deep copy of all relevant members
        labels_vec     = other.labels_vec;
        job_id         = other.job_id;
        lb             = other.lb;
        ub             = other.ub;
        fw_arcs        = other.fw_arcs;
        bw_arcs        = other.bw_arcs;
        fw_bucket_arcs = other.fw_bucket_arcs;
        bw_bucket_arcs = other.bw_bucket_arcs;
        fw_jump_arcs   = other.fw_jump_arcs;
        bw_jump_arcs   = other.bw_jump_arcs;
    }

    Bucket &operator=(const Bucket &other) {
        if (this == &other) return *this; // Handle self-assignment

        // Perform deep copy of all relevant members
        labels_vec     = other.labels_vec;
        job_id         = other.job_id;
        lb             = other.lb;
        ub             = other.ub;
        fw_arcs        = other.fw_arcs;
        bw_arcs        = other.bw_arcs;
        fw_bucket_arcs = other.fw_bucket_arcs;
        bw_bucket_arcs = other.bw_bucket_arcs;
        fw_jump_arcs   = other.fw_jump_arcs;
        bw_jump_arcs   = other.bw_jump_arcs;

        return *this;
    }

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
     * @param fw A boolean indicating the direction of the arc. If true, the arc is forward; otherwise, it is
     * backward.
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
     * @param fw A boolean indicating the direction of the jump arc. If true, the arc is added to the forward
     * jump arcs; otherwise, it is added to the backward jump arcs.
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
                                       [](const Label *a, const Label *b) { return a->cost > b->cost; });
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
        pdqsort(labels_vec.begin(), labels_vec.end(), [](const Label *a, const Label *b) { return a->cost < b->cost; });
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


// TODO: Add SRC and RCC penalty to the path cost
class SchrodingerPool {
private:
    std::deque<std::tuple<int, Path>> paths; // Stores tuples of (iteration added, Path)
    int                               current_iteration = 0;
    int                               max_live_time; // Max iterations a Path can stay active
    std::vector<double>               duals;         // Dual variables for each path
    std::vector<VRPJob>              *jobs = nullptr;

public:
    std::vector<std::vector<double>> distance_matrix; // Distance matrix for the graph

    SchrodingerPool(int live_time) : max_live_time(live_time) {}

    void setJobs(std::vector<VRPJob> *jobs) { this->jobs = jobs; }

    int getcij(int i, int j) { return distance_matrix[i][j]; }

    void remove_old_paths() {
        // Remove old paths that have lived beyond their allowed time
        while (!paths.empty() && std::get<0>(paths.front()) + max_live_time <= current_iteration) {
            paths.pop_front(); // Remove the oldest path
        }
    }

    void add_path(const Path &path) {
        // Add new path with the current iteration
        paths.push_back({current_iteration, path});
    }

    void add_paths(const std::vector<Path> &new_paths) {
        remove_old_paths();
        for (const Path &path : new_paths) { add_path(path); }
        computeRC();
    }

    void computeRC() {
        for (auto &path : paths) {
            int iteration_added = std::get<0>(path); // Get the iteration when the path was added

            // Stop processing if the path is older than current_iteration + max_life
            if (iteration_added + max_live_time < current_iteration) { break; }

            Path &p    = std::get<1>(path);
            p.red_cost = p.cost;

            if (p.size() > 1) {
                for (int i = 0; i < p.size() - 1; i++) {
                    auto &job = (*jobs)[p[i]]; // Dereference jobs and access element
                    p.red_cost -= job.cost;
                }
            }
        }
    }

    std::vector<Path> get_paths_with_negative_red_cost() const {
        std::vector<Path> result;

        for (const auto &path_tuple : paths) {
            int iteration_added = std::get<0>(path_tuple); // Get the iteration when the path was added

            // Stop processing if the path is older than current_iteration + max_life
            if (iteration_added + max_live_time < current_iteration) { break; }

            const Path &p = std::get<1>(path_tuple);

            // Add paths with negative red_cost to the result
            if (p.red_cost < 0) { result.push_back(p); }
        }

        // Sort the result based on red_cost
        std::sort(result.begin(), result.end(), [](const Path &a, const Path &b) { return a.red_cost < b.red_cost; });

        return result;
    }

    void iterate() { current_iteration++; }
};

/**
 * @class LabelPool
 * @brief A class that manages a pool of Label objects.
 *
 * The LabelPool class is responsible for managing a pool of Label objects. It provides methods to acquire and
 * release labels from the pool, as well as resetting the pool to its initial state.
 *
 * The pool size is determined during construction and can be optionally limited to a maximum size. Labels can
 * be acquired from the pool using the `acquire()` method, and released back to the pool using the `release()`
 * method. If the pool is full, a new label will be allocated.
 *
 * The `reset()` method can be used to reset the pool to its initial state. This will delete all labels in the
 * pool and reallocate labels to match the initial pool size.
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