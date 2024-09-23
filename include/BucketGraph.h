/**
 * @file BucketGraph.h
 * @brief Defines the BucketGraph class and associated functionalities used in solving vehicle routing problems (VRPs).
 *
 * This file contains the definition of the BucketGraph class, a key component used in bucket-based optimization for
 * resource-constrained shortest path problems (RCSPP) and vehicle routing problems (VRPs). The BucketGraph class
 * provides methods for arc generation, label management, dominance checking, feasibility tests, and various
 * operations related to the buckets in both forward and backward directions. It also includes utilities for managing
 * neighborhood relationships, handling strongly connected components (SCCs), and checking non-dominance of labels.
 *
 * Key Components:
 * - `LabelComparator`: A utility class for comparing Label objects based on cost.
 * - `BucketGraph`: The primary class implementing bucket-based graph management.
 * - Functions for parallel arc generation, feasibility checks, and job visitation management.
 * - Support for multiple stages of optimization and various arc types.
 *
 * Additionally, this file includes specialized bitmap operations for tracking visited and unreachable jobs, and
 * provides multiple templates to handle direction (`Forward`/`Backward`) and stage-specific optimization.
 */

#pragma once

#include "DataClass.h"
#include "Definitions.h"

#include <queue>
#include <set>
#include <string_view>

#include "../include/SCCFinder.h"

#define RCESPP_TOL_ZERO 1.E-6

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

/**
 * @class BucketGraph
 * @brief Represents a graph structure used for bucket-based optimization in a solver.
 *
 * The BucketGraph class provides various functionalities for managing and optimizing
 * a graph structure using buckets. It includes methods for initialization, configuration
 * printing, job visitation checks, statistics printing, bucket assignment, and more.
 *
 * The class also supports parallel arc generation, label management, and feasibility checks.
 * It is designed to work with different directions (Forward and Backward) and stages of
 * the optimization process.
 *
 * @note This class relies on several preprocessor directives (e.g., RIH, RCC, SRC) to
 *       conditionally enable or disable certain features.
 *
 * @tparam Direction The direction of the graph (Forward or Backward).
 * @tparam Stage The stage of the optimization process.
 * @tparam Full Indicates whether the full labeling algorithm should be used.
 * @tparam ArcType The type of arc (Bucket or regular).
 * @tparam Mutability Indicates whether the label is mutable or immutable.
 */
class BucketGraph {
    using NGRouteBitmap = uint64_t;

public:
    SchrodingerPool sPool = SchrodingerPool(200);

    // Note: very tricky way to unroll the loop at compile time and check for disposability
    static constexpr std::string_view resources[] = {RESOURCES}; // RESOURCES will expand to your string list
    static constexpr bool             resource_disposability[] = {
        RESOURCES_DISPOSABLE}; // RESOURCES_DISPOSABLE expands to disposability list

    template <Direction D, size_t I, size_t N, typename Gamma, typename VRPJob>
    constexpr bool process_all_resources(std::vector<double>              &new_resources,
                                         const std::array<double, R_SIZE> &initial_resources, const Gamma &gamma,
                                         const VRPJob &theJob) {
        if constexpr (I < N) {
            // Process the resource at index I
            if (!process_resource<D, I, N>(new_resources[I], initial_resources, gamma, theJob)) {
                return false; // Constraint violated, return false
            }
            // Recur to process the next resource
            return process_all_resources<D, I + 1, N>(new_resources, initial_resources, gamma, theJob);
        }
        return true; // All resources processed successfully
    }

    // Template recursion for compile-time unrolling
    template <Direction D, size_t I, size_t N, typename Gamma, typename VRPJob>
    constexpr bool process_resource(double &new_resource, const std::array<double, R_SIZE> &initial_resources,
                                    const Gamma &gamma, const VRPJob &theJob) {
        if constexpr (resource_disposability[I]) { // Checked at compile-time
            if constexpr (D == Direction::Forward) {
                new_resource =
                    std::max(initial_resources[I] + gamma.resource_increment[I], static_cast<double>(theJob.lb[I]));
                if (new_resource > theJob.ub[I]) {
                    return false; // Exceeds upper bound, return false to stop processing
                }
            } else {
                new_resource =
                    std::min(initial_resources[I] - gamma.resource_increment[I], static_cast<double>(theJob.ub[I]));
                if (new_resource < theJob.lb[I]) {
                    return false; // Below lower bound, return false to stop processing
                }
            }
        } else {
            // TODO: Add specific handling for non-disposable resources
        }

        return true; // Successfully processed all resources
    }
    // Function to automatically get the size
    template <typename T, std::size_t N>
    constexpr std::size_t array_size(const T (&)[N]) {
        return N;
    }

    bool                s1    = true;
    bool                s2    = false;
    bool                s3    = false;
    bool                s4    = false;
    bool                s5    = false;
    bool                ss    = false;
    int                 stage = 1;
    std::vector<double> q_star;
    int                 iter       = 0;
    bool                transition = true;
    Status              status     = Status::NotOptimal;

    void                 setSplit(std::vector<double> q_star) { this->q_star = q_star; }
    int                  getStage() { return stage; }
    Status               getStatus() { return status; }
    std::vector<Label *> solve();

    RCCmanager *rcc_manager = nullptr;

    std::vector<Label *> merged_labels_rih;

    // std::shared_ptr<ThreadPool> fw_thread_pool = std::make_shared<ThreadPool>(1);
    // std::shared_ptr<ThreadPool> bw_thread_pool = std::make_shared<ThreadPool>(1);

    std::thread rih_thread;
    std::mutex  mtx; // For thread-safe access to merged_labels_improved

    std::vector<std::vector<int>> fw_ordered_sccs;
    std::vector<std::vector<int>> bw_ordered_sccs;
    std::vector<int>              fw_topological_order;
    std::vector<int>              bw_topological_order;
    std::vector<std::vector<int>> fw_sccs;
    std::vector<std::vector<int>> bw_sccs;
    std::vector<std::vector<int>> fw_sccs_sorted;
    std::vector<std::vector<int>> bw_sccs_sorted;

    double incumbent  = std::numeric_limits<double>::infinity();
    double relaxation = std::numeric_limits<double>::infinity();
    bool   fixed      = false;

    exec::static_thread_pool            bi_pool  = exec::static_thread_pool(2);
    exec::static_thread_pool::scheduler bi_sched = bi_pool.get_scheduler();

    exec::static_thread_pool            cat_pool        = exec::static_thread_pool(4);
    exec::static_thread_pool::scheduler cat_sched       = bi_pool.get_scheduler();
    int                                 fw_buckets_size = 0;
    int                                 bw_buckets_size = 0;

    double bidi_relation = 1.0;

    std::vector<std::vector<bool>> fixed_arcs;
    std::vector<std::vector<bool>> fw_fixed_buckets;
    std::vector<std::vector<bool>> bw_fixed_buckets;

    double gap = std::numeric_limits<double>::infinity();

    CutStorage          *cut_storage = nullptr;
    static constexpr int max_buckets = 12000; // Define maximum number of buckets beforehand

    std::array<Bucket, max_buckets> fw_buckets;
    std::array<Bucket, max_buckets> bw_buckets;
    LabelPool                       label_pool_fw = LabelPool(1000);
    LabelPool                       label_pool_bw = LabelPool(1000);
    std::vector<BucketArc>          fw_arcs;
    std::vector<BucketArc>          bw_arcs;
    std::vector<Label *>            merged_labels;
    std::vector<std::vector<int>>   neighborhoods;

    std::vector<std::vector<double>> distance_matrix;
    std::vector<std::vector<int>>    Phi_fw;
    std::vector<std::vector<int>>    Phi_bw;

    std::unordered_map<int, std::vector<int>> fw_bucket_graph;
    std::unordered_map<int, std::vector<int>> bw_bucket_graph;

    std::vector<double> fw_c_bar;
    std::vector<double> bw_c_bar;

    int n_fw_labels = 0;
    int n_bw_labels = 0;

#if defined(RCC) || defined(EXACT_RCC)
    std::vector<std::vector<double>> cvrsep_duals;
#endif

    std::vector<std::vector<int>> job_to_bit_map;
    std::vector<int>              num_buckets_fw;
    std::vector<int>              num_buckets_bw;
    std::vector<int>              num_buckets_index_fw;
    std::vector<int>              num_buckets_index_bw;

    // Statistics
    int stat_n_labels_fw = 0;
    int stat_n_labels_bw = 0;
    int stat_n_dom_fw    = 0;
    int stat_n_dom_bw    = 0;

    std::vector<double>                R_max;
    std::vector<double>                R_min;
    std::vector<std::vector<uint64_t>> neighborhoods_bitmap; // Bitmap for neighborhoods of each job
    std::mutex                         label_pool_mutex;

    std::vector<Path> getSchrodinger() {
        std::vector<Path> negative_cost_paths = sPool.get_paths_with_negative_red_cost();
        if (negative_cost_paths.size() > N_ADD) { negative_cost_paths.resize(N_ADD); }
        return negative_cost_paths;
    }

    /**
     * @brief Prints the configuration information of the BucketGraph.
     *
     * This function outputs the configuration details including resource size,
     * number of clients, and maximum SRC cuts. It also conditionally prints
     * whether RIH, RCC, and SRC are enabled or disabled based on the preprocessor
     * directives.
     *
     * The output format is as follows:
     *
     * +----------------------------------+
     * |        CONFIGURATION INFO        |
     * +----------------------------------+
     * Resources: <R_SIZE>
     * Number of Clients: <N_SIZE>
     * Maximum SRC cuts: <MAX_SRC_CUTS>
     * RIH: <enabled/disabled>
     * RCC: <enabled/disabled>
     * SRC: <enabled/disabled>
     * +----------------------------------+
     */
    // TODO: add more configuration details
    void initInfo() {

        // Print header
        fmt::print("\n+----------------------------------+\n");
        fmt::print("|        CONFIGURATION INFO     |\n");
        fmt::print("+----------------------------------+\n");

        // Print Resource size
        fmt::print("Resources: {}\n", R_SIZE);
        fmt::print("Number of Clients: {}\n", N_SIZE);
        fmt::print("Maximum SRC cuts: {}\n", MAX_SRC_CUTS);

        // Conditional configuration (RIH enabled/disabled)
#ifdef RIH
        fmt::print("RIH: enabled\n");
#else
        fmt::print("RIH: disabled\n");
#endif
#ifdef RCC
        fmt::print("RCC: enabled\n");
#else
        fmt::print("RCC: disabled\n");
#endif
#ifdef SRC
        fmt::print("SRC: enabled\n");
#else
        fmt::print("SRC: disabled\n");
#endif
        fmt::print("+----------------------------------+\n");

        fmt::print("\n");
    }

    /**
     * @brief Runs forward and backward labeling algorithms in parallel and synchronizes the results.
     *
     * This function creates tasks for forward and backward labeling algorithms using the provided
     * scheduling mechanism. The tasks are executed in parallel, and the results are synchronized
     * and stored in the provided vectors.
     *
     * @tparam state The stage of the algorithm.
     * @tparam fullness The fullness state of the algorithm.
     * @param forward_cbar A reference to a vector where the results of the forward labeling algorithm will be stored.
     * @param backward_cbar A reference to a vector where the results of the backward labeling algorithm will be stored.
     * @param q_star A constant reference to a vector used as input for the labeling algorithms.
     */
    template <Stage state, Full fullness>
    void run_labeling_algorithms(std::vector<double> &forward_cbar, std::vector<double> &backward_cbar,
                                 const std::vector<double> &q_star) {
        // Create tasks for forward and backward labeling algorithms

        auto forward_task = stdexec::schedule(bi_sched) | stdexec::then([&]() {
                                return labeling_algorithm<Direction::Forward, state, fullness>(q_star);
                            });

        auto backward_task = stdexec::schedule(bi_sched) | stdexec::then([&]() {
                                 return labeling_algorithm<Direction::Backward, state, fullness>(q_star);
                             });

        // Execute the tasks in parallel and synchronize
        auto work = stdexec::when_all(std::move(forward_task), std::move(backward_task)) |
                    stdexec::then([&](auto forward_result, auto backward_result) {
                        forward_cbar  = std::move(forward_result);
                        backward_cbar = std::move(backward_result);
                    });

        stdexec::sync_wait(std::move(work));
    }

    double min_red_cost = std::numeric_limits<double>::infinity();
    bool   first_reset  = true;

    template <Stage S>
    void bucket_fixing(const std::vector<double> &q_star);

    template <Stage S>
    void heuristic_fixing(const std::vector<double> &q_star);

    /**
     * @brief Checks if a job has been visited based on a bitmap.
     *
     * This function determines if a specific job, identified by job_id, has been visited
     * by checking the corresponding bit in a bitmap array. The bitmap is an array of
     * 64-bit unsigned integers, where each bit represents the visited status of a job.
     *
     * @param bitmap A constant reference to an array of 64-bit unsigned integers representing the bitmap.
     * @param job_id An integer representing the ID of the job to check.
     * @return true if the job has been visited (i.e., the corresponding bit is set to 1), false otherwise.
     */
    inline bool is_job_visited(const std::array<uint64_t, num_words> &bitmap, int job_id) {
        int word_index   = job_id / 64; // Determine which 64-bit segment contains the job_id
        int bit_position = job_id % 64; // Determine the bit position within that segment
        return (bitmap[word_index] & (1ULL << bit_position)) != 0;
    }
    /**
     * @brief Marks a job as visited in the bitmap.
     *
     * This function sets the bit corresponding to the given job_id in the provided bitmap,
     * indicating that the job has been visited.
     *
     * @param bitmap A reference to an array of 64-bit unsigned integers representing the bitmap.
     * @param job_id The ID of the job to be marked as visited.
     */
    inline void set_job_visited(std::array<uint64_t, num_words> &bitmap, int job_id) {
        int word_index   = job_id / 64; // Determine which 64-bit segment contains the job_id
        int bit_position = job_id % 64; // Determine the bit position within that segment
        bitmap[word_index] |= (1ULL << bit_position);
    }

    /**
     * @brief Checks if a job is unreachable based on a bitmap.
     *
     * This function determines if a job, identified by its job_id, is unreachable
     * by checking a specific bit in a bitmap. The bitmap is represented as an
     * array of 64-bit unsigned integers.
     *
     * @param bitmap A constant reference to an array of 64-bit unsigned integers
     *               representing the bitmap.
     * @param job_id An integer representing the job identifier.
     * @return true if the job is unreachable (i.e., the corresponding bit in the
     *         bitmap is set), false otherwise.
     */
    inline bool is_job_unreachable(const std::array<uint64_t, num_words> &bitmap, int job_id) {
        int word_index   = job_id / 64; // Determine which 64-bit segment contains the job_id
        int bit_position = job_id % 64; // Determine the bit position within that segment
        return (bitmap[word_index] & (1ULL << bit_position)) != 0;
    }

    /**
     * @brief Marks a job as unreachable in the given bitmap.
     *
     * This function sets the bit corresponding to the specified job_id in the bitmap,
     * indicating that the job is unreachable.
     *
     * @param bitmap A reference to an array of 64-bit unsigned integers representing the bitmap.
     * @param job_id The ID of the job to be marked as unreachable.
     */
    inline void set_job_unreachable(std::array<uint64_t, num_words> &bitmap, int job_id) {
        int word_index   = job_id / 64; // Determine which 64-bit segment contains the job_id
        int bit_position = job_id % 64; // Determine the bit position within that segment
        bitmap[word_index] |= (1ULL << bit_position);
    }

    /**
     * @brief Prints the statistics of the bucket graph.
     *
     * This function outputs a formatted table displaying various metrics related to the bucket graph.
     * The table includes headers and values for forward and backward labels, as well as dominance checks.
     * The output is color-coded for better readability:
     * - Bold blue for metric names
     * - Green for values (backward labels)
     * - Reset color to default after each line
     *
     * The table structure:
     * +----------------------+-----------------+-----------------+
     * | Metric               | Forward         | Backward        |
     * +----------------------+-----------------+-----------------+
     * | Labels               | <stat_n_labels_fw> | <stat_n_labels_bw> |
     * | Dominance Check      | <stat_n_dom_fw> | <stat_n_dom_bw> |
     * +----------------------+-----------------+-----------------+
     *
     * Note: The actual values for the metrics (e.g., stat_n_labels_fw, stat_n_labels_bw, etc.) should be
     *       provided by the corresponding member variables or functions.
     */
    void print_statistics() {
        const char *blue_bold = "\033[1;34m"; // Bold blue for metrics
        const char *green     = "\033[32m";   // Green for values (backward labels)
        const char *reset     = "\033[0m";    // Reset color

        // Print table header with horizontal line and separators
        fmt::print("\n+----------------------+-----------------+-----------------+\n");
        fmt::print("|{}{:<20}{}| {:<15} | {:<15} |\n", blue_bold, " Metric", reset, " Forward", " Backward");
        fmt::print("+----------------------+-----------------+-----------------+\n");

        // Print labels for forward and backward with bold blue metric
        fmt::print("|{}{:<20}{}| {:<15} | {:<15} |\n", blue_bold, " Labels", reset, stat_n_labels_fw, stat_n_labels_bw);

        // Print dominated forward and backward labels with bold blue metric
        fmt::print("|{}{:<20}{}| {:<15} | {:<15} |\n", blue_bold, " Dominance Check", reset, stat_n_dom_fw,
                   stat_n_dom_bw);

        // Print the final horizontal line
        fmt::print("+----------------------+-----------------+-----------------+\n");

        fmt::print("\n");
    }

    /**
     * @brief Assigns the buckets based on the specified direction.
     *
     * This function returns a reference to the buckets based on the specified direction.
     * If the direction is Forward, it returns a reference to the forward buckets.
     * If the direction is Backward, it returns a reference to the backward buckets.
     *
     * @tparam D The direction (Forward or Backward).
     * @param FW The forward buckets.
     * @param BW The backward buckets.
     * @return A reference to the buckets based on the specified direction.
     */
    template <Direction D>
    constexpr auto &assign_buckets(auto &FW, auto &BW) noexcept {
        return (D == Direction::Forward) ? FW : BW;
    }

    // Common Initialization for Stages
    void common_initialization();

    /**
     * @brief Sets up the initial configuration for the BucketGraph.
     *
     * This function performs the following steps:
     * 1. Initializes the sizes of `fixed_arcs` based on the number of jobs.
     * 2. Resizes `fw_fixed_buckets` and `bw_fixed_buckets` to match the size of `fw_buckets`.
     * 3. Sets all elements in `fw_fixed_buckets` and `bw_fixed_buckets` to 0.
     * 4. Defines the initial relationships by calling `set_adjacency_list()` and `generate_arcs()`.
     * 5. Sorts the arcs for each job in the `jobs` list.
     */
    void setup() {
        // Initialize the sizes
        fixed_arcs.resize(getJobs().size());
        for (int i = 0; i < getJobs().size(); ++i) { fixed_arcs[i].resize(getJobs().size()); }

        // Resize and initialize fw_fixed_buckets and bw_fixed_buckets for std::vector<bool>
        fw_fixed_buckets.assign(fw_buckets.size(), std::vector<bool>(fw_buckets.size(), false));
        bw_fixed_buckets.assign(fw_buckets.size(), std::vector<bool>(fw_buckets.size(), false));
        // define initial relationships
        set_adjacency_list();
        generate_arcs();
        for (auto &VRPJob : jobs) { VRPJob.sort_arcs(); }

        sPool.distance_matrix = distance_matrix;
        sPool.setJobs(&jobs);
        fmt::print("Setup done\n");
    }

    /**
     * @brief Redefines the bucket intervals and reinitializes various data structures.
     *
     * This function updates the bucket interval and reinitializes the intervals, buckets,
     * fixed arcs, and fixed buckets. It also generates arcs and sorts them for each job.
     *
     * @param bucket_interval The new interval for the buckets.
     */
    void redefine(int bucket_interval) {
        this->bucket_interval = bucket_interval;
        intervals.clear();
        for (int i = 0; i < R_SIZE; ++i) { intervals.push_back(Interval(bucket_interval, 0)); }

        define_buckets<Direction::Forward>();
        define_buckets<Direction::Backward>();

        fixed_arcs.resize(getJobs().size());
        for (int i = 0; i < getJobs().size(); ++i) { fixed_arcs[i].resize(getJobs().size()); }

        // make every fixed_buckets also have size buckets.size()
        fw_fixed_buckets.resize(fw_buckets.size());
        bw_fixed_buckets.resize(fw_buckets.size());

        for (auto &fb : fw_fixed_buckets) { fb.resize(fw_buckets.size()); }
        for (auto &bb : bw_fixed_buckets) { bb.resize(bw_buckets.size()); }
        // set fixed_buckets to 0
        for (auto &fb : fw_fixed_buckets) {
            for (std::size_t i = 0; i < fb.size(); ++i) { fb[i] = 0; }
        }
        for (auto &bb : bw_fixed_buckets) {
            for (std::size_t i = 0; i < bb.size(); ++i) { bb[i] = 0; }
        }

        generate_arcs();
        for (auto &VRPJob : jobs) { VRPJob.sort_arcs(); }
    }

    int RIH1(std::priority_queue<Label *, std::vector<Label *>, LabelComparator> &best_labels_in,
             std::priority_queue<Label *, std::vector<Label *>, LabelComparator> &best_labels_out, int max_n_labels);

    int RIH2(std::priority_queue<Label *, std::vector<Label *>, LabelComparator> &best_labels_in,
             std::priority_queue<Label *, std::vector<Label *>, LabelComparator> &best_labels_out, int max_n_labels);

    int RIH3(std::priority_queue<Label *, std::vector<Label *>, LabelComparator> &best_labels_in,
             std::priority_queue<Label *, std::vector<Label *>, LabelComparator> &best_labels_out, int max_n_labels);

    int RIH4(std::priority_queue<Label *, std::vector<Label *>, LabelComparator> &best_labels_in,
             std::priority_queue<Label *, std::vector<Label *>, LabelComparator> &best_labels_out, int max_n_labels);

    // define default
    BucketGraph() = default;

    BucketGraph(const std::vector<VRPJob> &jobs, int time_horizon, int bucket_interval, int capacity,
                int capacity_interval);

    BucketGraph(const std::vector<VRPJob> &jobs, int time_horizon, int bucket_interval);

    BucketGraph(const std::vector<VRPJob> &jobs, std::vector<int> &bounds, std::vector<int> &bucket_intervals);

    template <Direction D>
    void add_arc(int from_bucket, int to_bucket, const std::vector<double> &res_inc, double cost_inc);

    template <Direction D>
    bool is_feasible(const std::vector<double> &resources, const VRPJob &VRPJob, Bucket &bucket);

    template <Direction D>
    const BucketArc &find_arc(int from_bucket, int to_bucket) const;

    template <Direction D>
    void generate_arcs();

    template <Direction D>
    void SCC_handler();

    /**
     * @brief Generates arcs in both forward and backward directions in parallel.
     *
     * This function uses OpenMP to parallelize the generation of arcs in both
     * forward and backward directions. It performs the following steps for each
     * direction:
     * - Calls the generate_arcs function template with the appropriate direction.
     * - Clears and resizes the Phi vector for the respective direction.
     * - Computes the Phi values for each bucket and stores them in the Phi vector.
     * - Calls the SCC_handler function template with the appropriate direction.
     *
     * The forward direction operations are performed in one OpenMP section, and
     * the backward direction operations are performed in another OpenMP section.
     */
    void generate_arcs() {

        PARALLEL_SECTIONS(
            bi_sched,
            SECTION {
                // Task for Forward Direction
                generate_arcs<Direction::Forward>();
                Phi_fw.clear();
                Phi_fw.resize(fw_buckets_size);
                for (int i = 0; i < fw_buckets_size; ++i) { Phi_fw[i] = computePhi(i, true); }
                SCC_handler<Direction::Forward>();
            },
            SECTION {
                // Task for Backward Direction
                generate_arcs<Direction::Backward>();
                Phi_bw.clear();
                Phi_bw.resize(bw_buckets_size);
                for (int i = 0; i < bw_buckets_size; ++i) { Phi_bw[i] = computePhi(i, false); }
                SCC_handler<Direction::Backward>();
            });
    }
    template <Direction D>
    Label &get_best_label();

    template <Direction D, Stage S, Full F>
    std::vector<double> labeling_algorithm(std::vector<double> q_point, bool full = false) noexcept;

    template <Direction D>
    int get_bucket_number(int job, const std::vector<double> &values) noexcept;

    template <Direction D>
    Label *get_best_label(const std::vector<int> &topological_order, const std::vector<double> &c_bar,
                          std::vector<std::vector<int>> &sccs);

    void set_adjacency_list();

    template <Direction D>
    void define_buckets();

    template <Direction D, Stage S>
    bool DominatedInCompWiseSmallerBuckets(Label *L, int bucket, std::vector<double> &c_bar,
                                           std::vector<uint64_t>               &Bvisited,
                                           const std::vector<std::vector<int>> &bucket_order) noexcept;

    template <Direction D, Stage S, ArcType A, Mutability M, Full F>
    inline Label *
    Extend(const std::conditional_t<M == Mutability::Mut, Label *, const Label *>          L_prime,
           const std::conditional_t<A == ArcType::Bucket, BucketArc,
                                    std::conditional_t<A == ArcType::Jump, JumpArc, Arc>> &gamma) noexcept;
    template <Direction D, Stage S>
    bool is_dominated(Label *&new_label, Label *&labels) noexcept;

    /**
     * @brief Resets the forward and backward label pools.
     *
     * This function resets both the forward (label_pool_fw) and backward
     * (label_pool_bw) label pools to their initial states. It is typically
     * used to clear any existing labels and prepare the pools for reuse.
     */
    void reset_pool() {
        label_pool_fw.reset();
        label_pool_bw.reset();
    }

    void forbidCycle(const std::vector<int> &cycle, bool aggressive);
    void augment_ng_memories(std::vector<double> &solution, std::vector<Path> &paths, bool aggressive, int eta1,
                             int eta2, int eta_max, int nC);

    template <Stage S>
    std::vector<Label *> bi_labeling_algorithm(std::vector<double> q_start);

    template <Stage S>
    void ConcatenateLabel(const Label *&L, int &b, Label *&pbest, std::vector<uint64_t> &Bvisited,
                          const std::vector<double> &q_star);

    Label               get_best_path();
    inline double       getcij(int i, int j) const { return distance_matrix[i][j]; }
    void                calculate_neighborhoods(size_t num_closest);
    std::vector<VRPJob> getJobs() const { return jobs; }
    std::vector<int>    computePhi(int &bucket, bool fw);

    /**
     * @brief Sets the dual values for the jobs.
     *
     * This function assigns the provided dual values to the jobs. It iterates
     * through the given vector of duals and sets each job's dual value to the
     * corresponding value from the vector.
     *
     * @param duals A vector of double values representing the duals to be set.
     */
    void setDuals(const std::vector<double> &duals) {
        for (size_t i = 1; i < N_SIZE - 1; ++i) { jobs[i].setDuals(duals[i - 1]); }
    }

    /**
     * @brief Sets the distance matrix and calculates neighborhoods.
     *
     * This function assigns the provided distance matrix to the internal
     * distance matrix of the class and then calculates the neighborhoods
     * based on the given number of nearest neighbors.
     *
     * @param distance_matrix A 2D vector representing the distance matrix.
     * @param n_ng The number of nearest neighbors to consider when calculating
     *             neighborhoods. Default value is 8.
     */
    void set_distance_matrix(const std::vector<std::vector<double>> &distance_matrix, int n_ng = 8) {
        this->distance_matrix = distance_matrix;
        calculate_neighborhoods(n_ng);
    }

    template <Direction D>
    void UpdateBucketsSet(const double theta, Label *&label, std::unordered_set<int> &Bbidi, int &current_bucket,
                          std::unordered_set<int> &Bvisited);

    template <Direction D>
    void ObtainJumpBucketArcs();
    bool BucketSetContains(const std::set<int> &bucket_set, const int &bucket);

    template <Direction D>
    void BucketArcElimination(double theta);

    template <Direction D>
    int get_opposite_bucket_number(int current_bucket_index);

    /**
     * @brief Resets all fixed arcs in the graph.
     *
     * This function iterates over each row in the fixed_arcs matrix and sets all elements to 0.
     * It effectively clears any fixed arc constraints that may have been previously set.
     */
    void reset_fixed() {
        for (auto &row : fixed_arcs) { std::fill(row.begin(), row.end(), 0); }
    }

    /**
     * @brief Checks the feasibility of a given forward and backward label.
     *
     * This function determines if the transition from a forward label to a backward label
     * is feasible based on resource constraints and job durations.
     *
     * @param fw_label Pointer to the forward label.
     * @param bw_label Pointer to the backward label.
     * @return true if the transition is feasible, false otherwise.
     *
     * The function performs the following checks:
     * - If either of the labels is null, it returns false.
     * - It retrieves the job associated with the forward label and checks if the sum of the
     *   forward label's resources, the cost between the jobs, and the job's duration exceeds
     *   the backward label's resources.
     * - If the resource size is greater than 1, it iterates through the resources and checks
     *   if the forward label's resources plus the job's demand exceed the backward label's resources.
     */
    inline bool check_feasibility(const Label *fw_label, const Label *bw_label) {
        if (!fw_label || !bw_label) return false;
        // print bw_label->job_id
        const struct VRPJob &VRPJob = jobs[fw_label->job_id];
        if (fw_label->resources[TIME_INDEX] + getcij(fw_label->job_id, bw_label->job_id) + VRPJob.duration >
            bw_label->resources[TIME_INDEX]) {
            return false;
        }
        if constexpr (R_SIZE > 1) {
            for (size_t r = 1; r < fw_label->resources.size(); r++)
                if (fw_label->resources[1] + VRPJob.demand > bw_label->resources[1]) { return false; }
        }
        return true;
    }

    void async_rih_processing(std::vector<Label *> initial_labels, int LABELS_MAX);

    std::vector<Label *> get_rih_labels() { return merged_labels_rih; }

    double knapsackBound(const Label *l);

private:
    std::vector<Interval> intervals;
    std::vector<VRPJob>   jobs;
    int                   time_horizon;
    int                   capacity;
    int                   bucket_interval;

    double best_cost;
    Label *fw_best_label;
    Label *bw_best_label;

    template <Direction D, Stage S>
    bool is_dominated(const Label &new_label, const std::vector<Label> &labels) noexcept;

    Label *compute_label(const Label *L, const Label *L_prime);
};
