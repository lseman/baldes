/**
 * @file BucketGraph.h
 * @brief Defines the BucketGraph class and associated functionalities used in
 * solving vehicle routing problems (VRPs).
 *
 * This file contains the definition of the BucketGraph class, a key component
 * used in bucket-based optimization for resource-constrained shortest path
 * problems (RCSPP) and vehicle routing problems (VRPs). The BucketGraph class
 * provides methods for arc generation, label management, dominance checking,
 * feasibility tests, and various operations related to the buckets in both
 * forward and backward directions. It also includes utilities for managing
 * neighborhood relationships, handling strongly connected components (SCCs),
 * and checking non-dominance of labels.
 *
 * Key Components:
 * - `LabelComparator`: A utility class for comparing Label objects based on
 * cost.
 * - `BucketGraph`: The primary class implementing bucket-based graph
 * management.
 * - Functions for parallel arc generation, feasibility checks, and node
 * visitation management.
 * - Support for multiple stages of optimization and various arc types.
 *
 * Additionally, this file includes specialized bitmap operations for tracking
 * visited and unreachable nodes, and provides multiple templates to handle
 * direction (`Forward`/`Backward`) and stage-specific optimization.
 */

#pragma once

#include <condition_variable>

#include "Bucket.h"
#include "CostFunction.h"
#include "Cut.h"
#include "Definitions.h"
#include "Dual.h"
#include "PSTEP.h"
#include "Pools.h"
#include "RCC.h"
#include "RIH.h"
#include "SCCFinder.h"
#include "Trees.h"
#include "UnionFind.h"
#include "VRPNode.h"

#define RCESPP_TOL_ZERO 1.E-6

class Stats {
    std::deque<double> obj_history;

   public:
    void addIteration(double obj_value) {
        obj_history.push_back(obj_value);
        if (obj_history.size() > 100) {  // Keep last 100 iterations
            obj_history.pop_front();
        }
    }

    bool hasHistory() const { return obj_history.size() >= 2; }

    double getRecentImprovement(size_t window) const {
        if (obj_history.size() < window) {
            return 1.0;  // Default if not enough history
        }
        auto start = obj_history.end() - window;
        auto end = obj_history.end();
        double avg_change =
            std::accumulate(start + 1, end, 0.0,
                            [prev = *start](double sum, double curr) mutable {
                                double change = (prev - curr) /
                                                std::max(1e-10, std::abs(prev));
                                prev = curr;
                                return sum + change;
                            }) /
            (window - 1);
        return avg_change;
    }

    double getRecentVariance(size_t window) const {
        if (obj_history.size() < window) {
            return 0.0;  // Default if not enough history
        }
        auto start = obj_history.end() - window;
        auto end = obj_history.end();
        double mean = std::accumulate(start, end, 0.0) / window;
        double variance =
            std::accumulate(start, end, 0.0,
                            [mean](double sum, double curr) {
                                return sum + std::pow(curr - mean, 2);
                            }) /
            window;
        return variance;
    }
};
/**
 * @class BucketGraph
 * @brief Represents a graph structure used for bucket-based
 * optimization in a solver.
 *
 * The BucketGraph class provides various functionalities for
 * managing and optimizing a graph structure using buckets. It
 * includes methods for initialization, configuration printing, node
 * visitation checks, statistics printing, bucket assignment, and
 * more.
 *
 * The class also supports parallel arc generation, label
 * management, and feasibility checks. It is designed to work with
 * different directions (Forward and Backward) and stages of the
 * optimization process.
 *
 * @note This class relies on several preprocessor directives (e.g.,
 * RIH, RCC, SRC) to conditionally enable or disable certain
 * features.
 *
 * @tparam Direction The direction of the graph (Forward or
 * Backward).
 * @tparam Stage The stage of the optimization process.
 * @tparam Full Indicates whether the full labeling algorithm should
 * be used.
 * @tparam ArcType The type of arc (Bucket or regular).
 * @tparam Mutability Indicates whether the label is mutable or
 * immutable.
 */
class BucketGraph {
    using NGRouteBitmap = uint64_t;
    Stats stats;

   public:
    static  // Precompute bitmask lookup table at compile time
        constexpr std::array<uint64_t, 64>
            bit_mask_lookup = []() {
                std::array<uint64_t, 64> masks{};
                for (size_t i = 0; i < 64; ++i) {
                    masks[i] = 1ULL << i;
                }
                return masks;
            }();

    ankerl::unordered_dense::map<std::pair<int, int>, bool> fw_union_cache;
    ankerl::unordered_dense::map<std::pair<int, int>, bool> bw_union_cache;

    CostFunction cost_calculator;
    std::vector<ankerl::unordered_dense::map<Arc, int, arc_hash>>
        fw_arc_scores;  // Track arc performance
    std::vector<ankerl::unordered_dense::map<Arc, int, arc_hash>>
        bw_arc_scores;  // Track arc performance

    BucketOptions options;
    void mono_initialization();
    Label *compute_mono_label(const Label *L);
    bool just_fixed = false;
    using BranchingDualsPtr = std::shared_ptr<BranchingDuals>;
    BranchingDualsPtr branching_duals = std::make_shared<BranchingDuals>();
#if defined(RCC) || defined(EXACT_RCC)
    ArcDuals arc_duals;
    void setArcDuals(const ArcDuals &arc_duals) { this->arc_duals = arc_duals; }
#endif

    PSTEPDuals pstep_duals;
    void setPSTEPduals(const PSTEPDuals &arc_duals) {
        this->pstep_duals = arc_duals;
    }
    auto solveTSP(std::vector<std::vector<uint16_t>> &paths,
                  std::vector<double> &path_costs, std::vector<int> &firsts,
                  std::vector<int> &lasts,
                  std::vector<std::vector<double>> &cost_matrix,
                  bool first_time = false);
    auto solveTSPTW(std::vector<std::vector<uint16_t>> &paths,
                    std::vector<double> &path_costs, std::vector<int> &firsts,
                    std::vector<int> &lasts,
                    std::vector<std::vector<double>> &cost_matrix,
                    std::vector<double> &service_times,
                    std::vector<double> &time_windows_start,
                    std::vector<double> &time_windows_end,
                    bool first_time = false);
    /**
     * @brief Solves the PSTEP problem and returns a vector of
     * labels representing paths.
     *
     * This function performs the following steps:
     * 1. Resets the pool.
     * 2. Initializes the mono algorithm.
     * 3. Runs a labeling algorithm in the forward direction.
     * 4. Iterates through the forward buckets and computes new
     * labels.
     * 5. Filters and collects labels that meet the criteria.
     *
     * @return std::vector<Label*> A vector of pointers to Label
     * objects representing the paths.
     */
    std::vector<Label *> solvePSTEP(PSTEPDuals &inner_pstep_duals);

    std::vector<Label *> solvePSTEP_by_MTZ();
    std::vector<Label *> solveTSPTW_by_MTZ();

    void setOptions(const BucketOptions &options) { this->options = options; }

#ifdef SCHRODINGER
    SchrodingerPool sPool = SchrodingerPool(200);
#endif

    // ankerl::unordered_dense::map<int, BucketIntervalTree>
    // fw_interval_trees; ankerl::unordered_dense::map<int,
    // BucketIntervalTree> bw_interval_trees;

    ArcList manual_arcs;
    void setManualArcs(const ArcList &manual_arcs) {
        this->manual_arcs = manual_arcs;
    }

    /**
     * @brief Assigns the buckets based on the specified direction.
     *
     * This function returns a reference to the buckets based on the
     * specified direction. If the direction is Forward, it returns
     * a reference to the forward buckets. If the direction is
     * Backward, it returns a reference to the backward buckets.
     *
     * @tparam D The direction (Forward or Backward).
     * @param FW The forward buckets.
     * @param BW The backward buckets.
     * @return A reference to the buckets based on the specified
     * direction.
     */
    template <Direction D>
    constexpr auto &assign_buckets(auto &FW, auto &BW) noexcept {
        return (D == Direction::Forward) ? FW : BW;
    }

    template <Direction D>
    constexpr const auto &assign_buckets(const auto &FW,
                                         const auto &BW) const noexcept {
        return (D == Direction::Forward) ? FW : BW;
    }

    template <Symmetry SYM>
    constexpr auto &assign_symmetry(auto &FW, auto &BW) noexcept {
        return (SYM == Symmetry::Symmetric) ? FW : BW;
    }

    template <Symmetry SYM>
    constexpr const auto &assign_symmetry(const auto &FW,
                                          const auto &BW) const noexcept {
        return (SYM == Symmetry::Symmetric) ? FW : BW;
    }
    /**
     * @brief Processes all resources by iterating through them and
     * applying constraints.
     *
     * This function recursively processes each resource in the
     * `new_resources` vector by calling `process_resource` for each
     * index from `I` to `N-1`. If any resource processing fails
     * (i.e., `process_resource` returns false), the function
     * returns false immediately. If all resources are processed
     * successfully, the function returns true.
     *
     */
    template <Direction D, typename Gamma, typename VRPNode>
    bool process_all_resources(
        std::vector<double> &new_resources,
        const std::array<double, R_SIZE> &initial_resources, const Gamma &gamma,
        const VRPNode &theNode, size_t N);

    // Template recursion for compile-time unrolling
    /**
     * @brief Processes a resource based on its disposability type
     * and direction.
     *
     * This function updates the `new_resource` value based on the
     * initial resources, the increment provided by `gamma`, and the
     * constraints defined by `theNode`. The behavior varies
     * depending on the disposability type of the resource.
     *
     */
    template <Direction D, typename Gamma, typename VRPNode>
    constexpr bool process_resource(
        double &new_resource,
        const std::array<double, R_SIZE> &initial_resources, const Gamma &gamma,
        const VRPNode &theNode, size_t I);
    bool s1 = true;
    bool s2 = false;
    bool s3 = false;
    bool s4 = false;
    bool s5 = false;
    bool ss = false;
    int stage = 1;
    std::vector<double> q_star;
    int iter = 0;
    bool transition = false;
    Status status = Status::NotOptimal;
    std::vector<Label *> merged_labels_rih;
    int A_MAX = N_SIZE;

#ifdef RIH
    std::thread rih_thread;
#endif
    std::mutex mtx;  // For thread-safe access to merged_labels_improved
    //
    int redefine_counter = 0;
    int depth = 0;

    IteratedLocalSearch *ils = nullptr;

    double inner_obj = -std::numeric_limits<double>::infinity();

    std::vector<std::vector<int>> fw_ordered_sccs;
    std::vector<std::vector<int>> bw_ordered_sccs;
    std::vector<int> fw_topological_order;
    std::vector<int> bw_topological_order;
    std::vector<std::vector<int>> fw_sccs;
    std::vector<std::vector<int>> bw_sccs;
    std::vector<std::vector<int>> fw_sccs_sorted;
    std::vector<std::vector<int>> bw_sccs_sorted;

    std::vector<double>
        fw_base_intervals;  // Forward base intervals for each node
    std::vector<double>
        bw_base_intervals;  // Backward base intervals for each node

    double incumbent = std::numeric_limits<double>::infinity();
    double relaxation = std::numeric_limits<double>::infinity();
    bool fixed = false;

    exec::static_thread_pool bi_pool = exec::static_thread_pool(2);
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
    std::vector<uint64_t>
        fw_fixed_buckets_bitmap;  // Bitmap for fixed bucket arcs
    std::vector<uint64_t>
        bw_fixed_buckets_bitmap;  // Bitmap for fixed bucket arcs

    template <Direction D>
    inline bool is_bucket_not_fixed(int from, int to) const noexcept {
        auto &fixed_buckets_bitmap =
            assign_buckets<D>(fw_fixed_buckets_bitmap, bw_fixed_buckets_bitmap);
        auto n_buckets = assign_buckets<D>(fw_buckets_size, bw_buckets_size);
        size_t bit_pos = from * n_buckets + to;
        return (fixed_buckets_bitmap[bit_pos / 64] &
                (1ULL << (bit_pos % 64))) == 0;
    }

    inline bool is_bucket_not_fixed_forward(int from, int to) const noexcept {
        return is_bucket_not_fixed<Direction::Forward>(from, to);
    }

    inline bool is_bucket_not_fixed_backward(int from, int to) const noexcept {
        return is_bucket_not_fixed<Direction::Backward>(from, to);
    }

    template <Direction D>
    inline bool is_bucket_fixed(int from, int to) const noexcept {
        auto &fixed_buckets_bitmap =
            assign_buckets<D>(fw_fixed_buckets_bitmap, bw_fixed_buckets_bitmap);
        auto n_buckets = assign_buckets<D>(fw_buckets_size, bw_buckets_size);
        size_t bit_pos = from * n_buckets + to;
        return (fixed_buckets_bitmap[bit_pos / 64] &
                (1ULL << (bit_pos % 64))) != 0;
    }
    double gap = std::numeric_limits<double>::infinity();

    CutStorage *cut_storage = new CutStorage();
    static constexpr int max_buckets =
        10000;  // Define maximum number of buckets beforehand

    std::vector<Bucket> fw_buckets;
    std::vector<Bucket> bw_buckets;

    using LabelPoolPtr = std::shared_ptr<LabelPool>;

    LabelPoolPtr label_pool_fw = std::make_shared<LabelPool>(100);
    LabelPoolPtr label_pool_bw = std::make_shared<LabelPool>(100);
    std::vector<BucketArc> fw_arcs;
    std::vector<BucketArc> bw_arcs;
    std::vector<Label *> merged_labels;
    std::vector<std::vector<int>> neighborhoods;

    std::vector<std::vector<double>> distance_matrix;
    std::vector<std::vector<int>> Phi_fw;
    std::vector<std::vector<int>> Phi_bw;

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
    int stat_n_dom_fw = 0;
    int stat_n_dom_bw = 0;

    std::vector<double> R_max;
    std::vector<double> R_min;
    std::vector<std::vector<uint64_t>>
        neighborhoods_bitmap;  // Bitmap for neighborhoods of each
                               // node
    std::vector<std::vector<uint64_t>>
        elementarity_bitmap;  // Bitmap for elementarity sets
    std::vector<std::vector<uint64_t>>
        packing_bitmap;  // Bitmap for packing sets

    void define_elementarity_sets() {
        size_t num_nodes = nodes.size();

        // Initialize elementarity sets
        elementarity_bitmap.resize(num_nodes);  // Bitmap for elementarity sets

        for (size_t i = 0; i < num_nodes; ++i) {
            // Each customer should appear in its own elementarity
            // set
            size_t num_segments = (num_nodes + 63) / 64;
            elementarity_bitmap[i].resize(num_segments, 0);

            // Mark the node itself in the elementarity set
            size_t segment_self = i >> 6;
            size_t bit_position_self = i & 63;
            elementarity_bitmap[i][segment_self] |= (1ULL << bit_position_self);
        }
    }

    void define_packing_sets() {
        size_t num_nodes = nodes.size();

        // Initialize packing sets
        packing_bitmap.resize(num_nodes);  // Bitmap for packing sets

        for (size_t i = 0; i < num_nodes; ++i) {
            // Each customer should appear in its own packing set
            size_t num_segments = (num_nodes + 63) / 64;
            packing_bitmap[i].resize(num_segments, 0);

            // Mark the node itself in the packing set
            size_t segment_self = i >> 6;
            size_t bit_position_self = i & 63;
            packing_bitmap[i][segment_self] |= (1ULL << bit_position_self);
        }
    }

    double min_red_cost = std::numeric_limits<double>::infinity();
    bool first_reset = true;

    std::vector<int> dominance_checks_per_bucket;
    int non_dominated_labels_per_bucket;

    // Interval tree to store bucket intervals
    std::vector<SplayTree> fw_node_interval_trees;
    std::vector<SplayTree> bw_node_interval_trees;

#ifdef SCHRODINGER
    /**
     * @brief Retrieves a list of paths with negative reduced costs.
     *
     * This function fetches paths from the sPool that have negative
     * reduced costs. If the number of such paths exceeds a
     * predefined limit (N_ADD), the list is truncated to contain
     * only the first N_ADD paths.
     *
     */
    std::vector<Path> getSchrodinger() {
        std::vector<Path> negative_cost_paths = sPool.get_paths();
        if (negative_cost_paths.size() > N_ADD) {
            negative_cost_paths.resize(N_ADD);
        }
        return negative_cost_paths;
    }
#endif

    // define default
    BucketGraph() = default;
    BucketGraph(const std::vector<VRPNode> &nodes, int horizon,
                int bucket_interval, int capacity, int capacity_interval);
    BucketGraph(const std::vector<VRPNode> &nodes, int horizon,
                int bucket_interval);
    BucketGraph(const std::vector<VRPNode> &nodes, std::vector<int> &bounds,
                std::vector<int> &bucket_intervals);

    // Common Tools
    static void initInfo();

    template <Symmetry SYM = Symmetry::Asymmetric>
    std::vector<Label *> solve(bool trigger = false);

    std::vector<Label *> solveHeuristic();

    template <Direction D>
    void common_initialization();

    void common_initialization() {
        PARALLEL_SECTIONS(
            work, bi_sched,
            SECTION { common_initialization<Direction::Forward>(); },
            SECTION { common_initialization<Direction::Backward>(); });
    }
    std::vector<Label *> fw_warm_labels;
    std::vector<Label *> bw_warm_labels;

    void setup();
    void print_statistics();
    void generate_arcs();

    ankerl::unordered_dense::map<int, int> fw_bucket_splits;
    ankerl::unordered_dense::map<int, int> bw_bucket_splits;

    template <Symmetry SYM = Symmetry::Asymmetric>
    void set_adjacency_list();
    void set_adjacency_list_manual();

    double knapsackBound(const Label *l);

    template <Stage S>
    Label *compute_label(const Label *L, const Label *L_prime,
                         double cost = 0.0);

    bool BucketSetContains(const std::set<int> &bucket_set, const int &bucket);
    void setSplit(const std::vector<double> q_star) { this->q_star = q_star; }
    int getStage() const { return stage; }
    Status getStatus() const { return status; }

    void forbidCycle(const std::vector<int> &cycle, bool aggressive);
    void augment_ng_memories(std::vector<double> &solution,
                             std::vector<Path> &paths, bool aggressive,
                             int eta1, int eta2, int eta_max, int nC);

    inline double getcij(int i, int j) const {
        // if (i == options.end_depot) { i = options.depot; }
        // if (j == options.end_depot) { j = options.depot; }
        return distance_matrix[i][j];
    }
    void calculate_neighborhoods(size_t num_closest);
    std::vector<VRPNode> getNodes() const { return nodes; }
    std::vector<int> computePhi(int &bucket, bool fw);

    template <Stage state, Full fullness>
    void run_labeling_algorithms(std::vector<double> &forward_cbar,
                                 std::vector<double> &backward_cbar);

    template <Stage S>
    void bucket_fixing();

    template <Stage S>
    void heuristic_fixing();

    /**
     * @brief Checks if a node has been visited based on a bitmap.
     *
     * This function determines if a specific node, identified by
     * node_id, has been visited by checking the corresponding bit
     * in a bitmap array. The bitmap is an array of 64-bit unsigned
     * integers, where each bit represents the visited status of a
     * node.
     *
     */
    static inline bool is_node_visited(
        const std::array<uint64_t, num_words> &bitmap, int node_id) {
        int word_index = node_id >> 6;  // Determine which 64-bit segment
                                        // contains the node_id
        int bit_position =
            node_id & 63;  // Determine the bit position within that segment
        return (bitmap[word_index] & (1ULL << bit_position)) != 0;
    }

    /**
     * @brief Marks a node as visited in the bitmap.
     *
     * This function sets the bit corresponding to the given node_id
     * in the provided bitmap, indicating that the node has been
     * visited.
     *
     */
    static inline void set_node_visited(std::array<uint64_t, num_words> &bitmap,
                                        int node_id) {
        int word_index = node_id >> 6;  // Determine which 64-bit segment
                                        // contains the node_id
        int bit_position =
            node_id & 63;  // Determine the bit position within that segment
        bitmap[word_index] |= (1ULL << bit_position);
    }

    /**
     * @brief Checks if a node is unreachable based on a bitmap.
     *
     * This function determines if a node, identified by its
     * node_id, is unreachable by checking a specific bit in a
     * bitmap. The bitmap is represented as an array of 64-bit
     * unsigned integers.
     *
     */
    static inline bool is_node_unreachable(
        const std::array<uint64_t, num_words> &bitmap, int node_id) {
        int word_index = node_id >> 6;  // Determine which 64-bit segment
                                        // contains the node_id
        int bit_position =
            node_id & 63;  // Determine the bit position within that segment
        return (bitmap[word_index] & (1ULL << bit_position)) != 0;
    }

    /**
     * @brief Marks a node as unreachable in the given bitmap.
     *
     * This function sets the bit corresponding to the specified
     * node_id in the bitmap, indicating that the node is
     * unreachable.
     *
     */
    static inline void set_node_unreachable(
        std::array<uint64_t, num_words> &bitmap, int node_id) {
        int word_index = node_id >> 6;  // Determine which 64-bit segment
                                        // contains the node_id
        int bit_position =
            node_id & 63;  // Determine the bit position within that segment
        bitmap[word_index] |= (1ULL << bit_position);
    }

    /**
     * @brief Redefines the bucket intervals and reinitializes
     * various data structures.
     *
     * This function updates the bucket interval and reinitializes
     * the intervals, buckets, fixed arcs, and fixed buckets. It
     * also generates arcs and sorts them for each node.
     *
     */
    void redefine(int bucketInterval) {
        this->bucket_interval = bucketInterval;
        intervals.clear();
        for (int i = 0; i < options.resources.size(); ++i) {
            intervals.push_back(Interval(bucketInterval, 0));
        }

        // auto fw_save_rebuild =
        // rebuild_buckets<Direction::Forward>(); auto
        // bw_save_rebuild = rebuild_buckets<Direction::Backward>();

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
                // fw_fixed_buckets.clear();
                // fw_fixed_buckets.resize(fw_buckets_size);
                // for (auto &fb : fw_fixed_buckets) {
                // fb.assign(fw_buckets_size, 0); }
            },
            SECTION {
                // Section 2: Backward direction
                bw_buckets_size = 0;
                bw_buckets.clear();
                define_buckets<Direction::Backward>();
                // split_buckets<Direction::Backward>();
                // bw_fixed_buckets.clear();
                // bw_fixed_buckets.resize(bw_buckets_size);
                // for (auto &bb : bw_fixed_buckets) {
                // bb.assign(bw_buckets_size, 0); }
            });

        generate_arcs();
        /*
        PARALLEL_SECTIONS(
            workB, bi_sched,
            [&, this, fw_save_rebuild]() -> void {
                // Forward direction processing
                process_buckets<Direction::Forward>(fw_buckets_size,
        fw_buckets, fw_save_rebuild, fw_fixed_buckets);
            },
            [&, this, bw_save_rebuild]() -> void {
                // Backward direction processing
                process_buckets<Direction::Backward>(bw_buckets_size,
        bw_buckets, bw_save_rebuild, bw_fixed_buckets);
            });
            */
    }

    template <Direction D>
    void split_buckets() {
        auto &buckets = assign_buckets<D>(fw_buckets, bw_buckets);
        auto &buckets_size =
            assign_buckets<D>(fw_buckets_size, bw_buckets_size);
        auto &num_buckets = assign_buckets<D>(num_buckets_fw, num_buckets_bw);
        auto &num_buckets_index =
            assign_buckets<D>(num_buckets_index_fw, num_buckets_index_bw);

        int num_intervals = options.main_resources.size();
        std::vector<double> total_ranges(num_intervals);
        std::vector<double> base_intervals(num_intervals);

        // Determine the base interval and other relevant values for
        // each resource
        for (int r = 0; r < num_intervals; ++r) {
            total_ranges[r] = R_max[r] - R_min[r];
            base_intervals[r] = total_ranges[r] / intervals[r].interval;
        }

        int original_num_buckets = buckets_size;

        // To track splits per node_id
        ankerl::unordered_dense::map<int, int> splits_per_node;

        // Loop until the second-to-last bucket to avoid
        // out-of-bounds access
        for (int i = 0; i < original_num_buckets - 1; ++i) {
            const auto &bucket = buckets[i];

            // Calculate mid-point for each dimension
            std::vector<double> mid_point(bucket.lb.size());
            if constexpr (D == Direction::Forward) {
                for (int r = 0; r < bucket.lb.size(); ++r) {
                    mid_point[r] = (bucket.lb[r] + base_intervals[r]) / 2.0;
                }
            } else {
                for (int r = 0; r < bucket.lb.size(); ++r) {
                    mid_point[r] = (bucket.ub[r] - base_intervals[r]) / 2.0;
                }
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

            // Manually shift elements to the right to make space
            // for the new bucket
            for (int j = original_num_buckets; j > i + 1; --j) {
                buckets[j] = buckets[j - 1];
            }

            // Insert the new bucket at position i+1
            buckets[i + 1] = bucket2;

            // Replace the current bucket with the first half
            // (bucket1)
            buckets[i] = bucket1;

            // Track the number of splits for this node_id
            splits_per_node[bucket.node_id]++;

            // Since we added a new bucket at i+1, increment i to
            // skip over the new bucket
            ++i;
            ++original_num_buckets;  // Adjust the count to account
                                     // for the new bucket
        }

        // Update the global bucket size
        buckets_size = original_num_buckets;

        // Now update the num_buckets_index for each node_id based
        // on the final positions
        int current_index = 0;
        for (int i = 0; i < original_num_buckets; ++i) {
            const auto &bucket = buckets[i];
            if (num_buckets_index[bucket.node_id] == -1) {
                num_buckets_index[bucket.node_id] = current_index;
            }
            current_index++;
        }
    }

    /**
     * @brief Resets the forward and backward label pools.
     *
     * This function resets both the forward (label_pool_fw) and
     * backward (label_pool_bw) label pools to their initial states.
     * It is typically used to clear any existing labels and prepare
     * the pools for reuse.
     */
    void reset_pool() {
        label_pool_fw->reset();
        label_pool_bw->reset();
    }

    /**
     * @brief Sets the dual values for the nodes.
     *
     * This function assigns the provided dual values to the nodes.
     * It iterates through the given vector of duals and sets each
     * node's dual value to the corresponding value from the vector.
     *
     */
    void setDuals(const std::vector<double> &duals) {
        // print nodes.size
        for (size_t i = 1; i < options.size - 1; ++i) {
            nodes[i].setDuals(duals[i - 1]);
        }
    }

    /**
     * @brief Sets the distance matrix and calculates neighborhoods.
     *
     * This function assigns the provided distance matrix to the
     * internal distance matrix of the class and then calculates the
     * neighborhoods based on the given number of nearest neighbors.
     *
     */
    void set_distance_matrix(
        const std::vector<std::vector<double>> &distanceMatrix, int n_ng = 8) {
        this->distance_matrix = distanceMatrix;
        calculate_neighborhoods(n_ng);
    }

    /**
     * @brief Resets all fixed arcs in the graph.
     *
     * This function iterates over each row in the fixed_arcs matrix
     * and sets all elements to 0. It effectively clears any fixed
     * arc constraints that may have been previously set.
     */
    void reset_fixed() {
        for (auto &row : fixed_arcs) {
            std::fill(row.begin(), row.end(), 0);
        }
    }

    void reset_fixed_buckets() {
        size_t fw_bitmap_size = (fw_buckets.size() * fw_buckets.size() + 63) /
                                64;  // Round up division by 64
        size_t bw_bitmap_size = (bw_buckets.size() * bw_buckets.size() + 63) /
                                64;  // Round up division by 64
        fw_fixed_buckets_bitmap.assign(fw_bitmap_size, 0);
        bw_fixed_buckets_bitmap.assign(bw_bitmap_size, 0);
    }

    /**
     * @brief Checks the feasibility of a given forward and backward
     * label.
     *
     * This function determines if the transition from a forward
     * label to a backward label is feasible based on resource
     * constraints and node durations.
     *
     */
    inline bool check_feasibility(const Label *fw_label,
                                  const Label *bw_label) {
        if (!fw_label || !bw_label) return false;

        // Cache resources and node data
        const auto &fw_resources = fw_label->resources;
        const auto &bw_resources = bw_label->resources;
        const struct VRPNode &VRPNode = nodes[fw_label->node_id];

        // Time feasibility check
        for (auto r = 0; r < options.resources.size(); ++r) {
            if (options.resources[r] != "time") {
                continue;
            }
            const auto time_fw = fw_resources[r];
            const auto time_bw = bw_resources[r];
            const auto travel_time =
                getcij(fw_label->node_id, bw_label->node_id);
            if (time_fw + travel_time + VRPNode.duration > time_bw) {
                return false;
            }
        }

        // Resource feasibility check (if applicable)
        if (options.resources.size() > 1) {
            for (size_t i = 1; i < options.resources.size(); ++i) {
                const auto resource_fw = fw_resources[i];
                const auto resource_bw = bw_resources[i];
                const auto demand = VRPNode.demand;

                if (resource_fw + demand > resource_bw) {
                    return false;
                }
            }
        }

        return true;
    }

    void updateSplit() {
        const double BASE_IMBALANCE_THRESHOLD =
            0.15;  // Slightly lower base threshold
        const double MIN_ADJUSTMENT_FACTOR = 0.02;
        const double MAX_ADJUSTMENT_FACTOR = 0.10;
        const double LEARNING_RATE_DECAY =
            0.99;                      // Decay factor for learning rate
        const double EMA_ALPHA = 0.1;  // Smoothing factor for EMA

        static double learning_rate =
            MAX_ADJUSTMENT_FACTOR;  // Initialize learning rate
        static double ema_imbalance_ratio =
            0.0;  // Initialize EMA of imbalance ratio

        // Calculate label counts and ratios
        double fw_labels = static_cast<double>(n_fw_labels);
        double bw_labels = static_cast<double>(n_bw_labels);
        double total_labels = fw_labels + bw_labels;

        // Skip if we don't have enough labels for meaningful
        // adjustment
        if (total_labels < 10) return;

        // Calculate imbalance ratio
        double fw_ratio = fw_labels / total_labels;
        double bw_ratio = bw_labels / total_labels;
        double imbalance_ratio = std::abs(fw_ratio - bw_ratio);

        // Update EMA of imbalance ratio
        ema_imbalance_ratio =
            EMA_ALPHA * imbalance_ratio + (1 - EMA_ALPHA) * ema_imbalance_ratio;

        // Dynamic threshold based on total number of labels
        // As we get more labels, we can be more sensitive to
        // imbalance
        double dynamic_threshold =
            BASE_IMBALANCE_THRESHOLD * std::exp(-total_labels / 1000.0);

        if (ema_imbalance_ratio > dynamic_threshold) {
            // Calculate adaptive adjustment factor
            // More severe imbalance = larger adjustment
            double severity = (ema_imbalance_ratio - dynamic_threshold) /
                              (1.0 - dynamic_threshold);
            double adjustment_factor =
                MIN_ADJUSTMENT_FACTOR +
                (MAX_ADJUSTMENT_FACTOR - MIN_ADJUSTMENT_FACTOR) * severity;

            // Adjust learning rate based on the severity of
            // imbalance
            learning_rate *= LEARNING_RATE_DECAY;

            // Adjust based on resource range
            double resource_range = R_max[options.main_resources[0]] -
                                    R_min[options.main_resources[0]];
            double base_adjustment =
                adjustment_factor * resource_range * learning_rate;

            // Scale adjustment based on current split position
            for (size_t i = 0; i < q_star.size(); ++i) {
                double relative_pos =
                    (q_star[i] - R_min[options.main_resources[0]]) /
                    resource_range;

                // Smaller adjustments near boundaries
                double position_factor =
                    4.0 * relative_pos * (1.0 - relative_pos);

                // Determine direction and apply adjustment
                double direction = (bw_labels > fw_labels) ? 1.0 : -1.0;
                double final_adjustment =
                    base_adjustment * position_factor * direction;

                // Apply adjustment with boundary protection
                q_star[i] = std::clamp(q_star[i] + final_adjustment,
                                       R_min[options.main_resources[0]] +
                                           resource_range * 0.1,  // Lower bound
                                       R_max[options.main_resources[0]] -
                                           resource_range * 0.1  // Upper bound
                );
            }

            if (std::abs(base_adjustment) > resource_range * 0.05) {
                print_info(
                    "Split adjustment: imbalance={:.3f}, "
                    "adjustment={:.3f}, "
                    "learning_rate={:.3f}\n",
                    ema_imbalance_ratio, base_adjustment, learning_rate);
            }
        }
    }

    std::vector<std::vector<int>> topHeurRoutes;

    bool updateStepSize() {
        bool updated = false;
        if (bucket_interval >= 100) {
            return false;
        }
        // compute the mean of dominance_checks_per_bucket
        double mean_dominance_checks = 0.0;
        for (size_t i = 0; i < dominance_checks_per_bucket.size(); ++i) {
            mean_dominance_checks += dominance_checks_per_bucket[i];
        }
        auto step_calc =
            mean_dominance_checks / non_dominated_labels_per_bucket;
        if (step_calc > 100) {
            // redefine_counter = 0;
            print_info("Increasing bucket interval to {}\n",
                       bucket_interval + 25);
            bucket_interval = bucket_interval + 25;
            redefine(bucket_interval);
            updated = true;
            fixed = false;
        }
        return updated;
    }
    template <Direction D>
    void add_arc(int from_bucket, int to_bucket,
                 const std::vector<double> &res_inc, double cost_inc);

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
    Label *get_best_label(const std::vector<int> &topological_order,
                          const std::vector<double> &c_bar,
                          const std::vector<std::vector<int>> &sccs);

    template <Direction D>
    void define_buckets();

    template <Direction D, Stage S>
    bool DominatedInCompWiseSmallerBuckets(const Label *L, int bucket,
                                           const std::vector<double> &c_bar,
                                           std::vector<uint64_t> &Bvisited,
                                           int &stat_n_dom) noexcept;

    template <Direction D, Stage S, ArcType A, Mutability M, Full F>
    inline std::vector<Label *> Extend(
        std::conditional_t<M == Mutability::Mut, Label *, const Label *>
            L_prime,
        const std::conditional_t<
            A == ArcType::Bucket, BucketArc,
            std::conditional_t<A == ArcType::Jump, JumpArc, Arc>> &gamma,
        int depth = 0) noexcept;

    template <Direction D, Stage S>
    bool is_dominated(const Label *new_label, const Label *labels) noexcept;

    template <Stage S, Symmetry SYM = Symmetry::Asymmetric>
    std::vector<Label *> bi_labeling_algorithm();

    template <Stage S, Symmetry SYM = Symmetry::Asymmetric>
    void ConcatenateLabel(const Label *L, int &b, double &best_cost,
                          std::vector<uint64_t> &Bvisited);

    template <Direction D>
    void UpdateBucketsSet(double theta, const Label *label,
                          ankerl::unordered_dense::set<int> &Bbidi,
                          int &current_bucket, std::vector<uint64_t> &Bvisited);

    template <Direction D>
    void ObtainJumpBucketArcs();

    template <Direction D>
    void BucketArcElimination(double theta);

    template <Direction D>
    int get_opposite_bucket_number(int current_bucket_index,
                                   std::vector<double> &inc);

    void update_neighborhoods(
        const std::vector<std::pair<size_t, size_t>> &conflicts) {
        size_t num_nodes = nodes.size();

        // For each conflict (u,v), add u to v's neighborhood and
        // vice versa
        for (const auto &conflict : conflicts) {
            size_t u = conflict.first;
            size_t v = conflict.second;

            print_info("Adding conflict between {} and {}\n", u, v);
            // Add v to u's neighborhood
            size_t segment_v = v >> 6;
            size_t bit_position_v = v & 63;
            neighborhoods_bitmap[u][segment_v] |= (1ULL << bit_position_v);

            // Add u to v's neighborhood
            size_t segment_u = u >> 6;
            size_t bit_position_u = u & 63;
            neighborhoods_bitmap[v][segment_u] |= (1ULL << bit_position_u);
        }
    }
    // Helper method to check if node j is in node i's neighborhood
    bool is_in_neighborhood(size_t i, size_t j) const {
        if (i >= neighborhoods_bitmap.size()) return false;

        const size_t segment = j >> 6;
        const size_t bit_position = j & 63;

        if (segment >= neighborhoods_bitmap[i].size()) return false;

        return (neighborhoods_bitmap[i][segment] & (1ULL << bit_position)) != 0;
    }

    // Method to get current neighborhood size for node i
    size_t get_neighborhood_size(size_t i) const {
        if (i >= neighborhoods_bitmap.size()) return 0;

        size_t count = 0;
        for (const auto &segment : neighborhoods_bitmap[i]) {
            count += __builtin_popcountll(segment);  // Count set bits
        }
        return count;
    }

    // Method to get all neighbors of node i
    std::vector<size_t> get_neighbors(size_t i) const {
        std::vector<size_t> neighbors;
        if (i >= neighborhoods_bitmap.size()) return neighbors;

        size_t num_nodes = nodes.size();
        for (size_t j = 0; j < num_nodes; ++j) {
            if (is_in_neighborhood(i, j)) {
                neighbors.push_back(j);
            }
        }
        return neighbors;
    }

    void set_deleted_arcs(const std::vector<std::pair<int, int>> &arcs) {
        size_t num_nodes = nodes.size();

        // Mark each arc in the input list as fixed/deleted
        for (const auto &[from, to] : arcs) {
            if (from < num_nodes && to < num_nodes) {
                fixed_arcs[from][to] = 1;
            }
        }
    }

    std::vector<Label *> extend_path(const std::vector<int> &path,
                                     std::vector<double> &resources);

    Label *compute_red_cost(Label *L, bool fw) {
        double real_cost = 0.0;
        double red_cost = 0.0;

        const double main_resource = L->resources[options.main_resources[0]];
        const double q_star_value = q_star[options.main_resources[0]];

        if (fw) {
            if (main_resource > q_star_value) {
                return nullptr;
            }
        } else {
            if (main_resource <= q_star_value) {
                return nullptr;
            }
        }

        // Initialize an empty SRCmap for the current label
        std::vector<uint16_t> updated_SRCmap(cut_storage->size(), 0.0);

        int last_node = -1;
        if (L->nodes_covered.size() <= 3) {
            return nullptr;
        }
        auto new_label = new Label();

        // Traverse through the label nodes from current to root
        for (auto node_id : L->nodes_covered) {
            /////////////////////////////////////////////
            // NG-route feasibility check
            /////////////////////////////////////////////
            ///
            if (is_node_visited(new_label->visited_bitmap, node_id)) {
                return nullptr;
            }

            size_t limit = new_label->visited_bitmap.size();
            for (size_t i = 0; i < limit; ++i) {
                uint64_t current_visited =
                    new_label->visited_bitmap[i];  // Get visited nodes
                                                   // in the current
                                                   // segment

                if (!current_visited)
                    continue;  // Skip if no nodes were visited in
                               // this segment

                uint64_t neighborhood_mask =
                    neighborhoods_bitmap[node_id][i];  // Get neighborhood mask
                                                       // for the current node
                uint64_t bits_to_clear =
                    current_visited & neighborhood_mask;  // Determine which
                                                          // bits to clear

                new_label->visited_bitmap[i] =
                    bits_to_clear;  // Clear irrelevant visited
                                    // nodes
            }
            set_node_visited(new_label->visited_bitmap,
                             node_id);  // Mark the new node as visited

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
            size_t segment =
                node_id >> 6;  // Determine the segment in the bitmap
            size_t bit_position =
                node_id & 63;  // Determine the bit position in the segment

            // Update the SRCmap for each cut
            auto &cutter = cut_storage;
            auto &SRCDuals = cutter->SRCDuals;
            const uint64_t bit_mask = 1ULL << bit_position;

            for (std::size_t idx = 0; idx < cutter->size(); ++idx) {
                const auto &cut = cutter->getCut(idx);
                const auto &baseSet = cut.baseSet;
                const auto &baseSetorder = cut.baseSetOrder;
                const auto &neighbors = cut.neighbors;
                const auto &multipliers = cut.p;

#if defined(SRC)
                // Apply SRC logic to update the SRCmap
                const bool bitIsSet = neighbors[segment] & bit_mask;

                auto &src_map_value = updated_SRCmap[idx];
                if (!bitIsSet) {
                    src_map_value = 0.0;  // Reset if bit is not set
                                          // in neighbors
                    continue;
                }
                const bool bitIsSet2 = baseSet[segment] & bit_mask;

                if (bitIsSet2) {
                    auto &den = multipliers.den;
                    src_map_value += multipliers.num[baseSetorder[node_id]];
                    if (src_map_value >= den) {
                        red_cost -= SRCDuals[idx];  // Apply the SRC dual
                                                    // value if threshold is
                                                    // exceeded
                        src_map_value -= den;       // Reset the SRC map value
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
            if (branching_duals->size() > 0 && last_node) {
                red_cost -= branching_duals->getDual(last_node, node_id);
            }

            // Move to the parent node and update last_node
            last_node = node_id;
        }

        new_label->cost = red_cost;
        new_label->real_cost = real_cost;
        new_label->parent = nullptr;
        new_label->node_id = L->node_id;
        new_label->nodes_covered = L->nodes_covered;
        new_label->is_extended = false;
        new_label->resources = L->resources;
        new_label->SRCmap = updated_SRCmap;

        // Bucket number for the new label
        std::vector<double> new_resources(options.resources.size());
        for (size_t i = 0; i < options.resources.size(); ++i) {
            new_resources[i] = new_label->resources[i];
        }
        int bucket = fw ? get_bucket_number<Direction::Forward>(
                              new_label->node_id, new_resources)
                        : get_bucket_number<Direction::Backward>(
                              new_label->node_id, new_resources);
        new_label->vertex = bucket;
        new_label->fresh = false;

        return new_label;
    };

    // Define types for better readability
    using AdjList =
        std::unordered_map<int, std::vector<std::tuple<int, double, int>>>;

    template <Symmetry SYM>
    AdjList get_adjacency_list() {
        AdjList adjacency_list;

        // Iterate through all nodes in the graph
        for (const auto &node : nodes) {
            std::vector<std::tuple<int, double, int>> arcs;

            // Retrieve all arcs associated with the current node
            for (const auto &arc : node.get_arcs<Direction::Forward>()) {
                int to_node = arc.to;
                double cost = getcij(node.id, to_node) - nodes[to_node].cost;
                int capacity = 1;

                // Only add arcs with non-zero capacity
                if (capacity > 0) {
                    arcs.emplace_back(to_node, cost, capacity);
                }
            }

            // Add the node's arcs to the adjacency list if it has
            // any outgoing edges
            if (!arcs.empty()) {
                adjacency_list[node.id] = arcs;
            }
        }

        return adjacency_list;
    }

    bool shallUpdateStep() {
        int update_ctr = 0;

        ankerl::unordered_dense::set<int> updated_buckets;
        for (size_t b = 0; b < fw_buckets.size(); b++) {
            if (fw_buckets[b].shall_split || bw_buckets[b].shall_split) {
                auto inner_id = fw_buckets[b].node_id;
                if (inner_id != options.depot &&
                    inner_id != options.end_depot) {
                    if (updated_buckets.contains(inner_id)) {
                        continue;
                    }

                    if (fw_bucket_splits[inner_id] == 100) {
                        continue;
                    }
                    auto update_target =
                        std::min(fw_bucket_splits[inner_id] * 2, 100);
                    fw_bucket_splits[inner_id] = update_target;
                    bw_bucket_splits[inner_id] = update_target;
                    updated_buckets.insert(inner_id);
                    update_ctr++;
                }
            }
        }

        const int CHANGE_THRESHOLD = nodes.size() / 6;
        if (update_ctr >= CHANGE_THRESHOLD && cut_storage->size() < 40) {
            fw_union_cache.clear();
            bw_union_cache.clear();
            print_info("Updating bucket splits with {} changes\n", update_ctr);
            // reset_fixed();
            // reset_fixed_buckets();

            PARALLEL_SECTIONS(
                work, bi_sched,
                SECTION {
                    // Section 1: Forward direction
                    define_buckets<Direction::Forward>();
                },
                SECTION {
                    // Section 2: Backward direction
                    define_buckets<Direction::Backward>();
                });

            generate_arcs();

            PARALLEL_SECTIONS(
                workB, bi_sched,
                SECTION {
                    // Forward direction processing
                    ObtainJumpBucketArcs<Direction::Forward>();
                },
                SECTION {
                    // Backward direction processing
                    ObtainJumpBucketArcs<Direction::Backward>();
                });
            // fmt::print("Generated arcs\n");

            // fixed = false;
            return true;
        }
        return false;
    }

   private:
    std::vector<Interval> intervals;
    std::vector<VRPNode> nodes;
    int horizon{};
    int capacity{};
    int bucket_interval{};

    double best_cost{};
    Label *fw_best_label{};
    Label *bw_best_label{};
};
