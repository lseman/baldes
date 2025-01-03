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
 * - Functions for parallel arc generation, feasibility checks, and node visitation management.
 * - Support for multiple stages of optimization and various arc types.
 *
 * Additionally, this file includes specialized bitmap operations for tracking visited and unreachable nodes, and
 * provides multiple templates to handle direction (`Forward`/`Backward`) and stage-specific optimization.
 */

#pragma once

#include "UnionFind.h"

#include "Definitions.h"

#include "Cut.h"

#include "Pools.h"

#include "RCC.h"
#include "Trees.h"

#include "Bucket.h"
#include "VRPNode.h"

#include "PSTEP.h"

#include "SCCFinder.h"

#include "Dual.h"

#include "RIH.h"

#include "muSPP.h"

#include <condition_variable>

#define RCESPP_TOL_ZERO 1.E-6

/**
 * @class BucketGraph
 * @brief Represents a graph structure used for bucket-based optimization in a solver.
 *
 * The BucketGraph class provides various functionalities for managing and optimizing
 * a graph structure using buckets. It includes methods for initialization, configuration
 * printing, node visitation checks, statistics printing, bucket assignment, and more.
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
    BucketOptions options;
    void          mono_initialization();
    Label        *compute_mono_label(const Label *L);
    bool          just_fixed          = false;
    using BranchingDualsPtr           = std::shared_ptr<BranchingDuals>;
    BranchingDualsPtr branching_duals = std::make_shared<BranchingDuals>();

#if defined(RCC) || defined(EXACT_RCC)
    ArcDuals arc_duals;
    void     setArcDuals(const ArcDuals &arc_duals) { this->arc_duals = arc_duals; }
#endif

    PSTEPDuals pstep_duals;
    void       setPSTEPduals(const PSTEPDuals &arc_duals) { this->pstep_duals = arc_duals; }
    auto       solveTSP(std::vector<std::vector<uint16_t>> &paths, std::vector<double> &path_costs, std::vector<int> &firsts,
                        std::vector<int> &lasts, std::vector<std::vector<double>> &cost_matrix, bool first_time = false);
    auto solveTSPTW(std::vector<std::vector<uint16_t>> &paths, std::vector<double> &path_costs, std::vector<int> &firsts,
                    std::vector<int> &lasts, std::vector<std::vector<double>> &cost_matrix,
                    std::vector<double> &service_times, std::vector<double> &time_windows_start,
                    std::vector<double> &time_windows_end, bool first_time = false);
    /**
     * @brief Solves the PSTEP problem and returns a vector of labels representing paths.
     *
     * This function performs the following steps:
     * 1. Resets the pool.
     * 2. Initializes the mono algorithm.
     * 3. Runs a labeling algorithm in the forward direction.
     * 4. Iterates through the forward buckets and computes new labels.
     * 5. Filters and collects labels that meet the criteria.
     *
     * @return std::vector<Label*> A vector of pointers to Label objects representing the paths.
     */
    std::vector<Label *> solvePSTEP(PSTEPDuals &inner_pstep_duals);

    std::vector<Label *> solvePSTEP_by_MTZ();
    std::vector<Label *> solveTSPTW_by_MTZ();

    void setOptions(const BucketOptions &options) { this->options = options; }

#ifdef SCHRODINGER
    SchrodingerPool sPool = SchrodingerPool(200);
#endif

    // ankerl::unordered_dense::map<int, BucketIntervalTree> fw_interval_trees;
    // ankerl::unordered_dense::map<int, BucketIntervalTree> bw_interval_trees;

    ArcList manual_arcs;
    void    setManualArcs(const ArcList &manual_arcs) { this->manual_arcs = manual_arcs; }

    template <Direction D>
    inline bool is_within_bounds(const BucketRange<D> &new_range, const BucketRange<D> &fixed_range) {
        return (new_range.lower_bound >= fixed_range.lower_bound && new_range.upper_bound <= fixed_range.upper_bound);
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

    template <Symmetry SYM>
    constexpr auto &assign_symmetry(auto &FW, auto &BW) noexcept {
        return (SYM == Symmetry::Symmetric) ? FW : BW;
    }

    /**
     * @brief Processes all resources by iterating through them and applying constraints.
     *
     * This function recursively processes each resource in the `new_resources` vector by calling
     * `process_resource` for each index from `I` to `N-1`. If any resource processing fails (i.e.,
     * `process_resource` returns false), the function returns false immediately. If all resources
     * are processed successfully, the function returns true.
     *
     */
    template <Direction D, typename Gamma, typename VRPNode>
    bool process_all_resources(std::vector<double> &new_resources, const std::array<double, R_SIZE> &initial_resources,
                               const Gamma &gamma, const VRPNode &theNode, size_t N);

    // Template recursion for compile-time unrolling
    /**
     * @brief Processes a resource based on its disposability type and direction.
     *
     * This function updates the `new_resource` value based on the initial resources,
     * the increment provided by `gamma`, and the constraints defined by `theNode`.
     * The behavior varies depending on the disposability type of the resource.
     *
     */
    template <Direction D, typename Gamma, typename VRPNode>
    constexpr bool       process_resource(double &new_resource, const std::array<double, R_SIZE> &initial_resources,
                                          const Gamma &gamma, const VRPNode &theNode, size_t I);
    bool                 s1    = true;
    bool                 s2    = false;
    bool                 s3    = false;
    bool                 s4    = false;
    bool                 s5    = false;
    bool                 ss    = false;
    int                  stage = 1;
    std::vector<double>  q_star;
    int                  iter       = 0;
    bool                 transition = true;
    Status               status     = Status::NotOptimal;
    std::vector<Label *> merged_labels_rih;
    int                  A_MAX = N_SIZE;

#ifdef RIH
    std::thread rih_thread;
#endif
    std::mutex mtx; // For thread-safe access to merged_labels_improved
    //
    int redefine_counter = 0;
    int depth            = 0;

    IteratedLocalSearch *ils = nullptr;

    double inner_obj = -std::numeric_limits<double>::infinity();

    std::vector<std::vector<int>> fw_ordered_sccs;
    std::vector<std::vector<int>> bw_ordered_sccs;
    std::vector<int>              fw_topological_order;
    std::vector<int>              bw_topological_order;
    std::vector<std::vector<int>> fw_sccs;
    std::vector<std::vector<int>> bw_sccs;
    std::vector<std::vector<int>> fw_sccs_sorted;
    std::vector<std::vector<int>> bw_sccs_sorted;

    std::vector<double> fw_base_intervals; // Forward base intervals for each node
    std::vector<double> bw_base_intervals; // Backward base intervals for each node

    double incumbent  = std::numeric_limits<double>::infinity();
    double relaxation = std::numeric_limits<double>::infinity();
    bool   fixed      = false;

    exec::static_thread_pool            bi_pool  = exec::static_thread_pool(2);
    exec::static_thread_pool::scheduler bi_sched = bi_pool.get_scheduler();

    ~BucketGraph() {
        bi_pool.request_stop();
        merged_labels.clear();
    }

    int fw_buckets_size = 0;
    int bw_buckets_size = 0;

    std::vector<std::vector<bool>> fixed_arcs;
    std::vector<std::vector<bool>> fw_fixed_buckets;
    std::vector<std::vector<bool>> bw_fixed_buckets;

    double gap = std::numeric_limits<double>::infinity();

    CutStorage          *cut_storage = new CutStorage();
    static constexpr int max_buckets = 10000; // Define maximum number of buckets beforehand

    std::vector<Bucket> fw_buckets;
    std::vector<Bucket> bw_buckets;

    using LabelPoolPtr = std::shared_ptr<LabelPool>;

    LabelPoolPtr                  label_pool_fw = std::make_shared<LabelPool>(100);
    LabelPoolPtr                  label_pool_bw = std::make_shared<LabelPool>(100);
    std::vector<BucketArc>        fw_arcs;
    std::vector<BucketArc>        bw_arcs;
    std::vector<Label *>          merged_labels;
    std::vector<std::vector<int>> neighborhoods;

    std::vector<std::vector<double>> distance_matrix;
    std::vector<std::vector<int>>    Phi_fw;
    std::vector<std::vector<int>>    Phi_bw;

    ankerl::unordered_dense::map<int, std::vector<int>> fw_bucket_graph;
    ankerl::unordered_dense::map<int, std::vector<int>> bw_bucket_graph;

    std::vector<double> fw_c_bar;
    std::vector<double> bw_c_bar;

    int n_fw_labels = 0;
    int n_bw_labels = 0;

    std::vector<int> num_buckets_fw;
    std::vector<int> num_buckets_bw;
    std::vector<int> num_buckets_index_fw;
    std::vector<int> num_buckets_index_bw;

    // Statistics
    int stat_n_labels_fw = 0;
    int stat_n_labels_bw = 0;
    int stat_n_dom_fw    = 0;
    int stat_n_dom_bw    = 0;

    std::vector<double>                R_max;
    std::vector<double>                R_min;
    std::vector<std::vector<uint64_t>> neighborhoods_bitmap; // Bitmap for neighborhoods of each node
    std::vector<std::vector<uint64_t>> elementarity_bitmap;  // Bitmap for elementarity sets
    std::vector<std::vector<uint64_t>> packing_bitmap;       // Bitmap for packing sets

    void define_elementarity_sets() {
        size_t num_nodes = nodes.size();

        // Initialize elementarity sets
        elementarity_bitmap.resize(num_nodes); // Bitmap for elementarity sets

        for (size_t i = 0; i < num_nodes; ++i) {
            // Each customer should appear in its own elementarity set
            size_t num_segments = (num_nodes + 63) / 64;
            elementarity_bitmap[i].resize(num_segments, 0);

            // Mark the node itself in the elementarity set
            size_t segment_self      = i >> 6;
            size_t bit_position_self = i & 63;
            elementarity_bitmap[i][segment_self] |= (1ULL << bit_position_self);
        }
    }

    void define_packing_sets() {
        size_t num_nodes = nodes.size();

        // Initialize packing sets
        packing_bitmap.resize(num_nodes); // Bitmap for packing sets

        for (size_t i = 0; i < num_nodes; ++i) {
            // Each customer should appear in its own packing set
            size_t num_segments = (num_nodes + 63) / 64;
            packing_bitmap[i].resize(num_segments, 0);

            // Mark the node itself in the packing set
            size_t segment_self      = i >> 6;
            size_t bit_position_self = i & 63;
            packing_bitmap[i][segment_self] |= (1ULL << bit_position_self);
        }
    }

    double min_red_cost = std::numeric_limits<double>::infinity();
    bool   first_reset  = true;

    std::vector<int> dominance_checks_per_bucket;
    int              non_dominated_labels_per_bucket;

    // Interval tree to store bucket intervals
    std::vector<SplayTree> fw_node_interval_trees;
    std::vector<SplayTree> bw_node_interval_trees;

    template <Direction D>
    ankerl::unordered_dense::map<int, BucketIntervalTree<D>> rebuild_buckets() {
        // References to the forward or backward fixed buckets and ranges
        auto &fixed_buckets = assign_buckets<D>(fw_fixed_buckets, bw_fixed_buckets);
        auto &buckets       = assign_buckets<D>(fw_buckets, bw_buckets);
        // auto &interval_tree = assign_buckets<D>(fw_interval_trees, bw_interval_trees);
        ankerl::unordered_dense::map<int, BucketIntervalTree<D>> interval_tree;
        // interval_tree.clear();

        // Iterate over all fixed arcs (fixed_buckets[from][to])
        for (int from_bucket = 0; from_bucket < fw_buckets_size; ++from_bucket) {
            auto from_node = buckets[from_bucket].node_id;
            for (int to_bucket = 0; to_bucket < fw_buckets_size; ++to_bucket) {
                auto to_node = buckets[to_bucket].node_id;
                // Check if there is a fixed arc between from_bucket and to_bucket
                if (fixed_buckets[from_bucket][to_bucket] == 0) { continue; }

                // Get the range for from_bucket
                BucketRange<D> from_range = {buckets[from_bucket].lb, buckets[from_bucket].ub};

                // Get the range for to_bucket
                BucketRange<D> to_range = {buckets[to_bucket].lb, buckets[to_bucket].ub};

                // check if interval_tree[from_node] exists
                if (interval_tree.find(from_node) == interval_tree.end()) {
                    interval_tree[from_node] = BucketIntervalTree<D>();
                }
                interval_tree[from_node].insert(from_range, to_range, to_node);
            }
        }
        // print trees, iterate over interval_tree and plot each
        /*
        for (auto &tree : interval_tree) {
            fmt::print("Tree for node: {}\n", tree.first);
            tree.second.print();
        }
        */
        return interval_tree;
    }

#ifdef SCHRODINGER
    /**
     * @brief Retrieves a list of paths with negative reduced costs.
     *
     * This function fetches paths from the sPool that have negative reduced costs.
     * If the number of such paths exceeds a predefined limit (N_ADD), the list is
     * truncated to contain only the first N_ADD paths.
     *
     */
    std::vector<Path> getSchrodinger() {
        std::vector<Path> negative_cost_paths = sPool.get_paths();
        if (negative_cost_paths.size() > N_ADD) { negative_cost_paths.resize(N_ADD); }
        return negative_cost_paths;
    }
#endif

    // define default
    BucketGraph() = default;
    BucketGraph(const std::vector<VRPNode> &nodes, int horizon, int bucket_interval, int capacity,
                int capacity_interval);
    BucketGraph(const std::vector<VRPNode> &nodes, int horizon, int bucket_interval);
    BucketGraph(const std::vector<VRPNode> &nodes, std::vector<int> &bounds, std::vector<int> &bucket_intervals);

    // Common Tools
    static void initInfo();

    template <Symmetry SYM = Symmetry::Asymmetric>
    std::vector<Label *> solve(bool trigger = false);

    std::vector<Label *> solveHeuristic();
    void                 common_initialization();
    void                 setup();
    void                 print_statistics();
    void                 generate_arcs();

    template <Symmetry SYM = Symmetry::Asymmetric>
    void set_adjacency_list();
    void set_adjacency_list_manual();

    double knapsackBound(const Label *l);

    template <Stage S>
    Label *compute_label(const Label *L, const Label *L_prime);

    bool   BucketSetContains(const std::set<int> &bucket_set, const int &bucket);
    void   setSplit(const std::vector<double> q_star) { this->q_star = q_star; }
    int    getStage() const { return stage; }
    Status getStatus() const { return status; }

    void forbidCycle(const std::vector<int> &cycle, bool aggressive);
    void augment_ng_memories(std::vector<double> &solution, std::vector<Path> &paths, bool aggressive, int eta1,
                             int eta2, int eta_max, int nC);

    inline double getcij(int i, int j) const {
        // if (i == options.end_depot) { i = options.depot; }
        // if (j == options.end_depot) { j = options.depot; }
        return distance_matrix[i][j];
    }
    void                 calculate_neighborhoods(size_t num_closest);
    std::vector<VRPNode> getNodes() const { return nodes; }
    std::vector<int>     computePhi(int &bucket, bool fw);

    template <Stage state, Full fullness>
    void run_labeling_algorithms(std::vector<double> &forward_cbar, std::vector<double> &backward_cbar);

    template <Stage S>
    void bucket_fixing();

    template <Stage S>
    void heuristic_fixing();

    /**
     * @brief Checks if a node has been visited based on a bitmap.
     *
     * This function determines if a specific node, identified by node_id, has been visited
     * by checking the corresponding bit in a bitmap array. The bitmap is an array of
     * 64-bit unsigned integers, where each bit represents the visited status of a node.
     *
     */
    static inline bool is_node_visited(const std::array<uint64_t, num_words> &bitmap, int node_id) {
        int word_index   = node_id >> 6; // Determine which 64-bit segment contains the node_id
        int bit_position = node_id & 63; // Determine the bit position within that segment
        return (bitmap[word_index] & (1ULL << bit_position)) != 0;
    }

    /**
     * @brief Marks a node as visited in the bitmap.
     *
     * This function sets the bit corresponding to the given node_id in the provided bitmap,
     * indicating that the node has been visited.
     *
     */
    static inline void set_node_visited(std::array<uint64_t, num_words> &bitmap, int node_id) {
        int word_index   = node_id >> 6; // Determine which 64-bit segment contains the node_id
        int bit_position = node_id & 63; // Determine the bit position within that segment
        bitmap[word_index] |= (1ULL << bit_position);
    }

    /**
     * @brief Checks if a node is unreachable based on a bitmap.
     *
     * This function determines if a node, identified by its node_id, is unreachable
     * by checking a specific bit in a bitmap. The bitmap is represented as an
     * array of 64-bit unsigned integers.
     *
     */
    static inline bool is_node_unreachable(const std::array<uint64_t, num_words> &bitmap, int node_id) {
        int word_index   = node_id >> 6; // Determine which 64-bit segment contains the node_id
        int bit_position = node_id & 63; // Determine the bit position within that segment
        return (bitmap[word_index] & (1ULL << bit_position)) != 0;
    }

    /**
     * @brief Marks a node as unreachable in the given bitmap.
     *
     * This function sets the bit corresponding to the specified node_id in the bitmap,
     * indicating that the node is unreachable.
     *
     */
    static inline void set_node_unreachable(std::array<uint64_t, num_words> &bitmap, int node_id) {
        int word_index   = node_id >> 6; // Determine which 64-bit segment contains the node_id
        int bit_position = node_id & 63; // Determine the bit position within that segment
        bitmap[word_index] |= (1ULL << bit_position);
    }

    /**
     * @brief Processes the buckets in a specific direction for heritages.
     *
     * This function processes the buckets in the specified direction (Forward or Backward)
     * to identify fixed arcs based on heritages. It iterates through each bucket and checks
     * if the node exists in the save_rebuild map. If the node is found, it searches for arcs
     * and updates the fixed buckets accordingly.
     *
     */
    template <Direction D>
    void process_buckets(auto buckets_size, auto &buckets, auto &save_rebuild, auto &fixed_buckets) {
        std::vector<int> tasks;
        std::atomic<int> n_fixed = 0; // Declare n_fixed as atomic for thread safety

        unsigned int total_threads = std::thread::hardware_concurrency();
        // Divide by 2 to use only half of the available threads
        unsigned int half_threads = 1;

        const int                JOBS = half_threads;
        exec::static_thread_pool pool(JOBS);
        auto                     sched = pool.get_scheduler();

        // Create tasks for each bucket
        for (int bucket = 0; bucket < buckets_size; ++bucket) { tasks.push_back(bucket); }

        // Define chunk size to reduce parallelization overhead
        const int chunk_size = 50; // You can adjust this based on performance experiments

        // Parallel execution in chunks using NVIDIA stdexec
        auto bulk_sender = stdexec::bulk(
            stdexec::just(), (tasks.size() + chunk_size - 1) / chunk_size, [&, chunk_size](std::size_t chunk_idx) {
                size_t start_idx = chunk_idx * chunk_size;
                size_t end_idx   = std::min(start_idx + chunk_size, tasks.size());

                // Process a chunk of tasks (buckets)
                for (size_t task_idx = start_idx; task_idx < end_idx; ++task_idx) {
                    int            bucket      = tasks[task_idx];
                    auto          &bucket_arcs = buckets[bucket].template get_bucket_arcs<D>();
                    auto          &from_bucket = buckets[bucket];
                    BucketRange<D> from_range  = {from_bucket.lb, from_bucket.ub};
                    auto           from_node   = from_bucket.node_id;

                    // Check if the node exists in the save_rebuild map (only once per bucket)
                    auto it = save_rebuild.find(from_node);
                    if (it != save_rebuild.end()) {
                        const auto &save_tree = it->second;

                        for (auto &arc : bucket_arcs) {
                            auto           to_bucket = arc.to_bucket;
                            auto           to_node   = buckets[to_bucket].node_id;
                            BucketRange<D> to_range  = {buckets[to_bucket].lb, buckets[to_bucket].ub};

                            // Perform the search and update fixed buckets
                            auto is_contained = save_tree.search(from_range, to_range, to_node);
                            if (is_contained) {
                                fixed_buckets[bucket][to_bucket] = 1;
                                // Update fixed count in a thread-safe manner
                                std::atomic_fetch_add(&n_fixed, 1);
                            }
                        }
                    }
                }
            });

        auto work = stdexec::starts_on(sched, bulk_sender);
        // Execute the bulk sender
        stdexec::sync_wait(std::move(work));

        if constexpr (D == Direction::Forward) {
            print_info("[Fw] {} arcs fixed from heritages\n", n_fixed.load());
        } else {
            print_info("[Bw] {} arcs fixed from heritages\n", n_fixed.load());
        }

        ObtainJumpBucketArcs<D>();
    }

    /**
     * @brief Redefines the bucket intervals and reinitializes various data structures.
     *
     * This function updates the bucket interval and reinitializes the intervals, buckets,
     * fixed arcs, and fixed buckets. It also generates arcs and sorts them for each node.
     *
     */
    void redefine(int bucketInterval) {
        this->bucket_interval = bucketInterval;
        intervals.clear();
        for (int i = 0; i < options.resources.size(); ++i) { intervals.push_back(Interval(bucketInterval, 0)); }

        // auto fw_save_rebuild = rebuild_buckets<Direction::Forward>();
        // auto bw_save_rebuild = rebuild_buckets<Direction::Backward>();

        reset_fixed();
        reset_fixed_buckets();

        PARALLEL_SECTIONS(
            work, bi_sched,
            SECTION {
                fw_buckets_size = 0;
                // clear the array
                fw_buckets.clear();
                // Section 1: Forward direction
                define_buckets<Direction::Forward>();
                // split_buckets<Direction::Forward>();
                fw_fixed_buckets.clear();
                fw_fixed_buckets.resize(fw_buckets_size);
                for (auto &fb : fw_fixed_buckets) { fb.assign(fw_buckets_size, 0); }
            },
            SECTION {
                // Section 2: Backward direction
                bw_buckets_size = 0;
                bw_buckets.clear();
                define_buckets<Direction::Backward>();
                // split_buckets<Direction::Backward>();
                bw_fixed_buckets.clear();
                bw_fixed_buckets.resize(bw_buckets_size);
                for (auto &bb : bw_fixed_buckets) { bb.assign(bw_buckets_size, 0); }
            });

        generate_arcs();
        /*
        PARALLEL_SECTIONS(
            workB, bi_sched,
            [&, this, fw_save_rebuild]() -> void {
                // Forward direction processing
                process_buckets<Direction::Forward>(fw_buckets_size, fw_buckets, fw_save_rebuild, fw_fixed_buckets);
            },
            [&, this, bw_save_rebuild]() -> void {
                // Backward direction processing
                process_buckets<Direction::Backward>(bw_buckets_size, bw_buckets, bw_save_rebuild, bw_fixed_buckets);
            });
            */
    }

    template <Direction D>
    void split_buckets() {
        auto &buckets           = assign_buckets<D>(fw_buckets, bw_buckets);
        auto &buckets_size      = assign_buckets<D>(fw_buckets_size, bw_buckets_size);
        auto &num_buckets       = assign_buckets<D>(num_buckets_fw, num_buckets_bw);
        auto &num_buckets_index = assign_buckets<D>(num_buckets_index_fw, num_buckets_index_bw);

        int                 num_intervals = options.main_resources.size();
        std::vector<double> total_ranges(num_intervals);
        std::vector<double> base_intervals(num_intervals);

        // Determine the base interval and other relevant values for each resource
        for (int r = 0; r < num_intervals; ++r) {
            total_ranges[r]   = R_max[r] - R_min[r];
            base_intervals[r] = total_ranges[r] / intervals[r].interval;
        }

        int original_num_buckets = buckets_size;

        // To track splits per node_id
        ankerl::unordered_dense::map<int, int> splits_per_node;

        // Loop until the second-to-last bucket to avoid out-of-bounds access
        for (int i = 0; i < original_num_buckets - 1; ++i) {
            const auto &bucket = buckets[i];

            // Calculate mid-point for each dimension
            std::vector<double> mid_point(bucket.lb.size());
            if constexpr (D == Direction::Forward) {
                for (int r = 0; r < bucket.lb.size(); ++r) { mid_point[r] = (bucket.lb[r] + base_intervals[r]) / 2.0; }
            } else {
                for (int r = 0; r < bucket.lb.size(); ++r) { mid_point[r] = (bucket.ub[r] - base_intervals[r]) / 2.0; }
            }

            // Create two new buckets by splitting at the mid-point
            Bucket bucket1, bucket2;
            if constexpr (D == Direction::Forward) {
                bucket1 = Bucket(bucket.node_id, bucket.lb, bucket.ub);
                bucket2 = Bucket(bucket.node_id, mid_point, bucket.ub);
            } else {
                bucket1 = Bucket(bucket.node_id, bucket.lb, bucket.ub);
                bucket2 = Bucket(bucket.node_id, bucket.lb, mid_point);
            }

            // Manually shift elements to the right to make space for the new bucket
            for (int j = original_num_buckets; j > i + 1; --j) { buckets[j] = buckets[j - 1]; }

            // Insert the new bucket at position i+1
            buckets[i + 1] = bucket2;

            // Replace the current bucket with the first half (bucket1)
            buckets[i] = bucket1;

            // Track the number of splits for this node_id
            splits_per_node[bucket.node_id]++;

            // Since we added a new bucket at i+1, increment i to skip over the new bucket
            ++i;
            ++original_num_buckets; // Adjust the count to account for the new bucket
        }

        // Update the global bucket size
        buckets_size = original_num_buckets;

        // Now update the num_buckets_index for each node_id based on the final positions
        int current_index = 0;
        for (int i = 0; i < original_num_buckets; ++i) {
            const auto &bucket = buckets[i];
            if (num_buckets_index[bucket.node_id] == -1) { num_buckets_index[bucket.node_id] = current_index; }
            current_index++;
        }
    }

    /**
     * @brief Resets the forward and backward label pools.
     *
     * This function resets both the forward (label_pool_fw) and backward
     * (label_pool_bw) label pools to their initial states. It is typically
     * used to clear any existing labels and prepare the pools for reuse.
     */
    void reset_pool() {
        label_pool_fw->reset();
        label_pool_bw->reset();
    }

    /**
     * @brief Sets the dual values for the nodes.
     *
     * This function assigns the provided dual values to the nodes. It iterates
     * through the given vector of duals and sets each node's dual value to the
     * corresponding value from the vector.
     *
     */
    void setDuals(const std::vector<double> &duals) {
        // print nodes.size
        for (size_t i = 1; i < options.size - 1; ++i) { nodes[i].setDuals(duals[i - 1]); }
    }

    /**
     * @brief Sets the distance matrix and calculates neighborhoods.
     *
     * This function assigns the provided distance matrix to the internal
     * distance matrix of the class and then calculates the neighborhoods
     * based on the given number of nearest neighbors.
     *
     */
    void set_distance_matrix(const std::vector<std::vector<double>> &distanceMatrix, int n_ng = 8) {
        this->distance_matrix = distanceMatrix;
        calculate_neighborhoods(n_ng);
    }

    /**
     * @brief Resets all fixed arcs in the graph.
     *
     * This function iterates over each row in the fixed_arcs matrix and sets all elements to 0.
     * It effectively clears any fixed arc constraints that may have been previously set.
     */
    void reset_fixed() {
        for (auto &row : fixed_arcs) { std::fill(row.begin(), row.end(), 0); }
    }

    void reset_fixed_buckets() {
        for (auto &fb : fw_fixed_buckets) { std::fill(fb.begin(), fb.end(), 0); }
        for (auto &bb : bw_fixed_buckets) { std::fill(bb.begin(), bb.end(), 0); }
    }

    /**
     * @brief Checks the feasibility of a given forward and backward label.
     *
     * This function determines if the transition from a forward label to a backward label
     * is feasible based on resource constraints and node durations.
     *
     */
    inline bool check_feasibility(const Label *fw_label, const Label *bw_label) {
        if (!fw_label || !bw_label) return false;

        // Cache resources and node data
        const auto           &fw_resources = fw_label->resources;
        const auto           &bw_resources = bw_label->resources;
        const struct VRPNode &VRPNode      = nodes[fw_label->node_id];

        // Time feasibility check
        for (auto r = 0; r < options.resources.size(); ++r) {
            if (options.resources[r] != "time") { continue; }
            const auto time_fw     = fw_resources[r];
            const auto time_bw     = bw_resources[r];
            const auto travel_time = getcij(fw_label->node_id, bw_label->node_id);
            if (time_fw + travel_time + VRPNode.duration > time_bw) { return false; }
        }

        // Resource feasibility check (if applicable)
        if (options.resources.size() > 1) {
            for (size_t i = 1; i < options.resources.size(); ++i) {
                const auto resource_fw = fw_resources[i];
                const auto resource_bw = bw_resources[i];
                const auto demand      = VRPNode.demand;

                if (resource_fw + demand > resource_bw) { return false; }
            }
        }

        return true;
    }

    void updateSplit() {
        //////////////////////////////////////////////////////////////////////
        // ADAPTIVE TERMINAL TIME
        //////////////////////////////////////////////////////////////////////
        // Adjust the terminal time dynamically based on the difference between the number of forward and
        // backward labels
        for (auto &split : q_star) {
            // print n_fw_labels and n_bw_labels
            // If there are more backward labels than forward labels, increase the terminal time slightly
            if (((static_cast<double>(n_bw_labels) - static_cast<double>(n_fw_labels)) /
                 static_cast<double>(n_fw_labels)) > 0.2) {
                split += 0.05 * R_max[options.main_resources[0]];
            }
            // If there are more forward labels than backward labels, decrease the terminal time slightly
            else if (((static_cast<double>(n_fw_labels) - static_cast<double>(n_bw_labels)) /
                      static_cast<double>(n_bw_labels)) > 0.2) {
                split -= 0.05 * R_max[options.main_resources[0]];
            }
        }
    }

    std::vector<std::vector<int>> topHeurRoutes;

    bool updateStepSize() {
        bool updated = false;
        if (bucket_interval >= 100) { return false; }
        // compute the mean of dominance_checks_per_bucket
        double mean_dominance_checks = 0.0;
        for (size_t i = 0; i < dominance_checks_per_bucket.size(); ++i) {
            mean_dominance_checks += dominance_checks_per_bucket[i];
        }
        auto step_calc = mean_dominance_checks / non_dominated_labels_per_bucket;
        if (step_calc > 100) {

            // redefine_counter = 0;
            print_info("Increasing bucket interval to {}\n", bucket_interval + 25);
            bucket_interval = bucket_interval + 25;
            redefine(bucket_interval);
            updated = true;
            fixed   = false;
        }
        return updated;
    }
    template <Direction D>
    void add_arc(int from_bucket, int to_bucket, const std::vector<double> &res_inc, double cost_inc);

    template <Direction D>
    void generate_arcs();

    template <Direction D>
    void SCC_handler();

    template <Direction D, Stage S, Full F>
    std::vector<double> labeling_algorithm();

    UnionFind fw_union_find;
    UnionFind bw_union_find;
    template <Direction D>
    int get_bucket_number(int node, std::vector<double> &values) noexcept;

    template <Direction D>
    Label *get_best_label(const std::vector<int> &topological_order, const std::vector<double> &c_bar,
                          const std::vector<std::vector<int>> &sccs);

    template <Direction D>
    void define_buckets();

    template <Direction D, Stage S>
    bool DominatedInCompWiseSmallerBuckets(const Label *L, int bucket, const std::vector<double> &c_bar,
                                           std::vector<uint64_t>               &Bvisited,
                                           const std::vector<std::vector<int>> &bucket_order) noexcept;

    template <Direction D, Stage S, ArcType A, Mutability M, Full F>
    inline std::vector<Label *>
    Extend(std::conditional_t<M == Mutability::Mut, Label *, const Label *>                L_prime,
           const std::conditional_t<A == ArcType::Bucket, BucketArc,
                                    std::conditional_t<A == ArcType::Jump, JumpArc, Arc>> &gamma,
           int                                                                             depth = 0) noexcept;

    template <Direction D, Stage S>
    bool is_dominated(const Label *new_label, const Label *labels) noexcept;

    template <Stage S, Symmetry SYM = Symmetry::Asymmetric>
    std::vector<Label *> bi_labeling_algorithm();

    template <Stage S, Symmetry SYM = Symmetry::Asymmetric>
    void ConcatenateLabel(const Label *L, int &b, double &best_cost, std::vector<uint64_t> &Bvisited);

    template <Direction D>
    void UpdateBucketsSet(double theta, const Label *label, ankerl::unordered_dense::set<int> &Bbidi,
                          int &current_bucket, std::vector<uint64_t> &Bvisited);

    template <Direction D>
    void ObtainJumpBucketArcs();

    template <Direction D>
    void BucketArcElimination(double theta);

    template <Direction D>
    int get_opposite_bucket_number(int current_bucket_index, std::vector<double> &inc);

    void update_neighborhoods(const std::vector<std::pair<size_t, size_t>> &conflicts) {
        size_t num_nodes = nodes.size();

        // For each conflict (u,v), add u to v's neighborhood and vice versa
        for (const auto &conflict : conflicts) {
            size_t u = conflict.first;
            size_t v = conflict.second;

            print_info("Adding conflict between {} and {}\n", u, v);
            // Add v to u's neighborhood
            size_t segment_v      = v >> 6;
            size_t bit_position_v = v & 63;
            neighborhoods_bitmap[u][segment_v] |= (1ULL << bit_position_v);

            // Add u to v's neighborhood
            size_t segment_u      = u >> 6;
            size_t bit_position_u = u & 63;
            neighborhoods_bitmap[v][segment_u] |= (1ULL << bit_position_u);
        }
    }
    // Helper method to check if node j is in node i's neighborhood
    bool is_in_neighborhood(size_t i, size_t j) const {
        if (i >= neighborhoods_bitmap.size()) return false;

        const size_t segment      = j >> 6;
        const size_t bit_position = j & 63;

        if (segment >= neighborhoods_bitmap[i].size()) return false;

        return (neighborhoods_bitmap[i][segment] & (1ULL << bit_position)) != 0;
    }

    // Method to get current neighborhood size for node i
    size_t get_neighborhood_size(size_t i) const {
        if (i >= neighborhoods_bitmap.size()) return 0;

        size_t count = 0;
        for (const auto &segment : neighborhoods_bitmap[i]) {
            count += __builtin_popcountll(segment); // Count set bits
        }
        return count;
    }

    // Method to get all neighbors of node i
    std::vector<size_t> get_neighbors(size_t i) const {
        std::vector<size_t> neighbors;
        if (i >= neighborhoods_bitmap.size()) return neighbors;

        size_t num_nodes = nodes.size();
        for (size_t j = 0; j < num_nodes; ++j) {
            if (is_in_neighborhood(i, j)) { neighbors.push_back(j); }
        }
        return neighbors;
    }

    void set_deleted_arcs(const std::vector<std::pair<int, int>> &arcs) {
        size_t num_nodes = nodes.size();

        // Mark each arc in the input list as fixed/deleted
        for (const auto &[from, to] : arcs) {
            if (from < num_nodes && to < num_nodes) { fixed_arcs[from][to] = 1; }
        }
    }

    std::vector<Label *> extend_path(const std::vector<int> &path, std::vector<double> &resources);

    Label *compute_red_cost(Label *L, bool fw) {
        double real_cost = 0.0;
        double red_cost  = 0.0;

        // Initialize an empty SRCmap for the current label
        std::vector<uint16_t> updated_SRCmap(cut_storage->size(), 0.0);

        int last_node = -1;
        if (L->nodes_covered.size() <= 3) { return nullptr; }
        auto new_label = new Label();

        // Traverse through the label nodes from current to root
        for (auto node_id : L->nodes_covered) {

            /////////////////////////////////////////////
            // NG-route feasibility check
            /////////////////////////////////////////////
            ///
            if (is_node_visited(new_label->visited_bitmap, node_id)) { return nullptr; }

            size_t limit = new_label->visited_bitmap.size();
            for (size_t i = 0; i < limit; ++i) {
                uint64_t current_visited = new_label->visited_bitmap[i]; // Get visited nodes in the current segment

                if (!current_visited) continue; // Skip if no nodes were visited in this segment

                uint64_t neighborhood_mask =
                    neighborhoods_bitmap[node_id][i]; // Get neighborhood mask for the current node
                uint64_t bits_to_clear = current_visited & neighborhood_mask; // Determine which bits to clear

                new_label->visited_bitmap[i] = bits_to_clear; // Clear irrelevant visited nodes
            }
            set_node_visited(new_label->visited_bitmap, node_id); // Mark the new node as visited

            if (last_node != -1) {
                double cij_cost = getcij(last_node, node_id);
                real_cost += cij_cost;
                red_cost += cij_cost;
            }

            red_cost -= nodes[node_id].cost;

            /////////////////////////////////////////////
            // SRC logic
            /////////////////////////////////////////////
            ///
            size_t segment      = node_id >> 6; // Determine the segment in the bitmap
            size_t bit_position = node_id & 63; // Determine the bit position in the segment

            // Update the SRCmap for each cut
            auto          &cutter   = cut_storage;
            auto          &SRCDuals = cutter->SRCDuals;
            const uint64_t bit_mask = 1ULL << bit_position;

            for (std::size_t idx = 0; idx < cutter->size(); ++idx) {
                const auto &cut          = cutter->getCut(idx);
                const auto &baseSet      = cut.baseSet;
                const auto &baseSetorder = cut.baseSetOrder;
                const auto &neighbors    = cut.neighbors;
                const auto &multipliers  = cut.p;

#if defined(SRC)
                // Apply SRC logic to update the SRCmap
                const bool bitIsSet = neighbors[segment] & bit_mask;

                auto &src_map_value = updated_SRCmap[idx];
                if (!bitIsSet) {
                    src_map_value = 0.0; // Reset if bit is not set in neighbors
                    continue;
                }
                const bool bitIsSet2 = baseSet[segment] & bit_mask;

                if (bitIsSet2) {
                    auto &den = multipliers.den;
                    src_map_value += multipliers.num[baseSetorder[node_id]];
                    if (src_map_value >= den) {
                        red_cost -= SRCDuals[idx]; // Apply the SRC dual value if threshold is exceeded
                        src_map_value -= den;      // Reset the SRC map value
                    }
                }
#endif
            }

            // Adjust for arc duals if in Stage::Four
#if defined(RCC) || defined(EXACT_RCC)
            if (last_node) {
                auto arc_dual = arc_duals.getDual(last_node, node_id);
                red_cost -= arc_dual;
            }
#endif

            // Adjust for branching duals if they exist
            if (branching_duals->size() > 0 && last_node) { red_cost -= branching_duals->getDual(last_node, node_id); }

            // Move to the parent node and update last_node
            last_node = node_id;
        }

        new_label->cost          = red_cost;
        new_label->real_cost     = real_cost;
        new_label->parent        = nullptr;
        new_label->node_id       = L->node_id;
        new_label->nodes_covered = L->nodes_covered;
        new_label->is_extended   = false;
        new_label->resources     = L->resources;
        new_label->SRCmap        = updated_SRCmap;

        // Bucket number for the new label
        std::vector<double> new_resources(options.resources.size());
        for (size_t i = 0; i < options.resources.size(); ++i) { new_resources[i] = new_label->resources[i]; }
        int bucket        = fw ? get_bucket_number<Direction::Forward>(new_label->node_id, new_resources)
                               : get_bucket_number<Direction::Backward>(new_label->node_id, new_resources);
        new_label->vertex = bucket;

        new_label->fresh = false;

        return new_label;
    };

    // Define types for better readability
    using AdjList = std::unordered_map<int, std::vector<std::tuple<int, double, int>>>;

    template <Symmetry SYM>
    AdjList get_adjacency_list() {
        AdjList adjacency_list;

        // Iterate through all nodes in the graph
        for (const auto &node : nodes) {
            std::vector<std::tuple<int, double, int>> arcs;

            // Retrieve all arcs associated with the current node
            for (const auto &arc : node.get_arcs<Direction::Forward>()) {
                int    to_node  = arc.to;
                double cost     = getcij(node.id, to_node) - nodes[to_node].cost;
                int    capacity = 1;

                // Only add arcs with non-zero capacity
                if (capacity > 0) { arcs.emplace_back(to_node, cost, capacity); }
            }

            // Add the node's arcs to the adjacency list if it has any outgoing edges
            if (!arcs.empty()) { adjacency_list[node.id] = arcs; }
        }

        return adjacency_list;
    }
    template <Symmetry SYM>
    std::vector<std::vector<int>> solve_min_cost_flow(int source, int sink) {
        fmt::print("Generating adjacency list...\n");
        auto adj_list = get_adjacency_list<SYM>();

        auto  labels       = bi_labeling_algorithm<Stage::Eliminate, Symmetry::Asymmetric>();
        auto  initial_path = labels[0]->getRoute();
        MUSSP solver;

        // addNode for each node
        std::vector<MUSSP::Node *> SPPnodes = {};
        fmt::print("Adding nodes...\n");
        for (auto i = 0; i < nodes.size(); ++i) { SPPnodes.push_back(solver.addNode()); }
        fmt::print("Added {} nodes\n", SPPnodes.size());

        solver.setSource(SPPnodes[source]);
        solver.setSink(SPPnodes[sink]);

        // addEdge
        for (const auto &[from_node, arcs] : adj_list) {
            for (const auto &[to_node, cost, capacity] : arcs) {
                // fmt::print("Adding edge from {} to {} with cost {} and capacity {}\n", from_node, to_node, cost,
                // capacity);
                solver.addEdge(SPPnodes[from_node], SPPnodes[to_node], cost, capacity);
            }
        }

        fmt::print("Solving minimum cost flow...\n");
        solver.solve();
        // auto path_set = solver.getPaths();

        // for (auto label : labels) { path_set.push_back(label->getRoute()); }
        //  path_set.push_back(initial_path);
        std::vector<std::vector<int>> path_set;
        return path_set; // Return all computed paths
    }

private:
    std::vector<Interval> intervals;
    std::vector<VRPNode>  nodes;
    int                   horizon{};
    int                   capacity{};
    int                   bucket_interval{};

    double best_cost{};
    Label *fw_best_label{};
    Label *bw_best_label{};
};
