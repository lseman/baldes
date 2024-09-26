/**
 * @file DataClasses.h
 * @brief Defines the core structures and classes for managing labels, arcs, and buckets in a graph-based solver.
 *
 * This file includes definitions for handling labels, buckets, and related resources in optimization problems,
 * specifically for applications like Vehicle Routing Problems (VRP) or other resource-constrained path-finding
 * algorithms.
 *
 * The key components defined in this file include:
 *  - `Label`: Represents a node in the solution space with properties like cost, resources, and job coverage.
 *  - `Bucket`: A container managing labels and arcs for efficient graph traversal and label extension.
 *  - `SchrodingerPool`: Manages paths with a limited lifespan, reducing costs and filtering based on dual values.
 *  - `LabelPool`: Manages a pool of reusable labels, facilitating efficient memory management and recycling of labels.
 *  - `RCCmanager`: Handles the management of RCC cuts, dual cache, and arc-to-cut mappings with thread safety.
 *  - `PSTEPDuals`: Manages the dual values for arcs and nodes in a network.
 *
 * Each of these components is designed to operate efficiently in a graph-based solver and support parallel processing
 * where applicable.
 */

#pragma once

#include "Definitions.h"

#include <deque>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

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

        // Assuming `resources` is a vector or array-like structure with the same size as the input
        std::copy(resources.begin(), resources.end(), this->resources.begin());

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
    void add_bucket_arc(int from_bucket, int to_bucket, const std::vector<double> &res_inc, double cost_inc, bool fw,
                        bool fixed) {
        if (fw) {
            fw_bucket_arcs.emplace_back(from_bucket, to_bucket, res_inc, cost_inc, fixed);
        } else {
            bw_bucket_arcs.emplace_back(from_bucket, to_bucket, res_inc, cost_inc, fixed);
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
    void add_jump_arc(int from_bucket, int to_bucket, const std::vector<double> &res_inc, double cost_inc, bool fw) {
        if (fw) {
            fw_jump_arcs.emplace_back(from_bucket, to_bucket, res_inc, cost_inc);
        } else {
            bw_jump_arcs.emplace_back(from_bucket, to_bucket, res_inc, cost_inc);
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
            *it = labels_vec.back();
            labels_vec.pop_back(); // Remove the last element
        }
    }

    // std::vector<Label *> &get_labels() { return labels_vec; }
    inline auto &get_labels() { return labels_vec; }

    inline auto &get_sorted_labels() {
        pdqsort(labels_vec.begin(), labels_vec.end(), [](const Label *a, const Label *b) { return a->cost < b->cost; });
        return labels_vec;
    }

    inline auto get_unextended_labels() {
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

    [[nodiscard]] bool empty() const { return labels_vec.empty(); }
};

/**
 * @class SchrodingerPool
 * @brief Manages a pool of paths with a limited lifespan, computes reduced costs, and filters paths based on their
 * reduced costs.
 *
 * This class is designed to manage a collection of paths, each associated with an iteration when it was added.
 * Paths have a limited lifespan, defined by `max_live_time`, after which they are removed from the pool.
 * The class also provides functionality to compute reduced costs for paths and filter paths based on their reduced
 * costs.
 *
 */
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

            if (p.size() > 3) {
                for (int i = 1; i < p.size() - 1; i++) {
                    auto &job = (*jobs)[p[i]]; // Dereference jobs and access element
                    p.red_cost -= job.cost;
                }
            }
        }
    }

    std::vector<Path> get_paths_with_negative_red_cost() {
        std::vector<Path> result;

        // Remove the paths that are older than current_iteration + max_life or have a negative red_cost
        auto it = paths.begin();
        while (it != paths.end()) {
            int         iteration_added = std::get<0>(*it); // Get the iteration when the path was added
            const Path &p               = std::get<1>(*it);

            // If the path is older than max_live_time, or has a negative red_cost
            if (iteration_added + max_live_time < current_iteration || p.red_cost < 0) {
                if (p.red_cost < 0) {
                    result.push_back(p); // Add paths with negative red_cost to the result
                }
                it = paths.erase(it); // Remove from paths and move iterator to the next element
            } else {
                ++it; // Only move the iterator if not erasing
            }
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

/**
 * @struct PSTEPDuals
 * @brief A structure to manage dual values for arcs and nodes in a network.
 *
 * This structure provides methods to set, get, and clear dual values for arcs and nodes.
 * Dual values are stored in unordered maps for efficient access.
 *
 */
struct PSTEPDuals {
    using Arc = std::pair<int, int>; // Represents an arc as a pair (from, to)

    std::unordered_map<Arc, double, pair_hash> arcDuals;          // Stores dual values for arcs
    std::unordered_map<int, double>            three_two_Duals;   // Stores dual values for nodes
    std::unordered_map<int, double>            three_three_Duals; // Stores dual values for nodes

    // Set dual values for arcs
    void setArcDualValues(const std::vector<std::pair<Arc, double>> &values) {
        for (const auto &[arc, value] : values) { arcDuals[arc] = value; }
    }

    // Set dual values for nodes
    void setThreeTwoDualValues(const std::vector<std::pair<int, double>> &values) {
        for (const auto &[node, value] : values) { three_two_Duals[node] = value; }
    }

    void setThreeThreeDualValues(const std::vector<std::pair<int, double>> &values) {
        for (const auto &[node, value] : values) { three_three_Duals[node] = value; }
    }

    // Clear all dual values (arcs and nodes)
    void clearDualValues() {
        arcDuals.clear();
        three_two_Duals.clear();
        three_three_Duals.clear();
    }

    // Get dual value for arcs (from, to)
    double getArcDualValue(int from, int to) const {
        Arc  arc   = {from, to};
        auto arcIt = arcDuals.find(arc);
        if (arcIt != arcDuals.end()) { return arcIt->second; }
        return 0.0; // Default value if arc is not found
    }

    // Get dual value from three_two_Duals for nodes
    double getThreeTwoDualValue(int node) const {
        auto it = three_two_Duals.find(node);
        if (it != three_two_Duals.end()) { return it->second; }
        return 0.0; // Default value if node is not found
    }

    // Get dual value from three_three_Duals for nodes
    double getThreeThreeDualValue(int node) const {
        auto it = three_three_Duals.find(node);
        if (it != three_three_Duals.end()) { return it->second; }
        return 0.0; // Default value if node is not found
    }
};

/**
 * @class LabelComparator
 * @brief Comparator class for comparing two Label objects based on their cost.
 *
 * This class provides an overloaded operator() that allows for comparison
 * between two Label pointers. The comparison is based on the cost attribute
 * of the Label objects, with the comparison being in descending order.
 */
class LabelComparator {
public:
    bool operator()(Label *a, Label *b) { return a->cost > b->cost; }
};
