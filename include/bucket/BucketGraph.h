/**
 * @file BucketGraph.h
 * @brief Defines the BucketGraph class and its bucket-based VRP labeling functionality.
 *
 */

#pragma once

#include <atomic>
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
#include "Stats.h"
#include "Trees.h"
#include "UnionFind.h"
#include "VRPNode.h"

#define RCESPP_TOL_ZERO 1.E-6

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
    struct WarmLabelState {
        double                     cost      = 0.0;
        int                        node_id   = -1;
        std::array<double, R_SIZE> resources = {};
        std::vector<uint16_t>      nodes_covered;
        int                        path_len = 0;
    };

    double threshold = -1.5;

    int        n_segments = 0;
    std::mutex merge_mutex;

    static // Precompute bitmask lookup table at compile time
        constexpr std::array<uint64_t, 64>
            bit_mask_lookup = []() {
                std::array<uint64_t, 64> masks{};
                for (size_t i = 0; i < 64; ++i) { masks[i] = 1ULL << i; }
                return masks;
            }();

    CostFunction                                                  cost_calculator;
    std::vector<ankerl::unordered_dense::map<Arc, int, arc_hash>> fw_arc_scores; // Track arc performance
    std::vector<ankerl::unordered_dense::map<Arc, int, arc_hash>> bw_arc_scores; // Track arc performance

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
    auto solveTSP(std::vector<std::vector<uint16_t>> &paths, std::vector<double> &path_costs, std::vector<int> &firsts,
                  std::vector<int> &lasts, std::vector<std::vector<double>> &cost_matrix, bool first_time = false);
    auto solveTSPTW(std::vector<std::vector<uint16_t>> &paths, std::vector<double> &path_costs,
                    std::vector<int> &firsts, std::vector<int> &lasts, std::vector<std::vector<double>> &cost_matrix,
                    std::vector<double> &service_times, std::vector<double> &time_windows_start,
                    std::vector<double> &time_windows_end, bool first_time = false);
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
    void    setManualArcs(const ArcList &manual_arcs) { this->manual_arcs = manual_arcs; }

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
    constexpr const auto &assign_buckets(const auto &FW, const auto &BW) const noexcept {
        return (D == Direction::Forward) ? FW : BW;
    }

    template <Symmetry SYM>
    constexpr auto &assign_symmetry(auto &FW, auto &BW) noexcept {
        return (SYM == Symmetry::Symmetric) ? FW : BW;
    }

    template <Symmetry SYM>
    constexpr const auto &assign_symmetry(const auto &FW, const auto &BW) const noexcept {
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
    bool process_all_resources(std::vector<double> &new_resources, const std::array<double, R_SIZE> &initial_resources,
                               const Gamma &gamma, const VRPNode &theNode, size_t N);

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
    bool                 transition = false;
    Status               status     = Status::NotOptimal;
    std::vector<Label *> merged_labels_rih;
    int                  A_MAX = N_SIZE;

    struct EnumerationPolicy {
        bool   enabled                     = true;
        bool   require_nonnegative_pricing = false;
        double max_relative_gap            = 0.02;
        double force_relative_gap          = 0.005;
        int    min_exact_iterations        = 2;
        int    min_stable_cut_passes       = 1;
        int    max_bucket_arcs_per_node    = 10000;
        int    max_total_bucket_arcs       = 1000000;
        int    max_labels_per_direction    = 2000000;
        int    max_routes                  = 1000;
    };

    EnumerationPolicy enumeration_policy{};
    std::atomic_bool  enumeration_failed{false};

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
    std::vector<int>              fw_bucket_scc_rank;
    std::vector<int>              bw_bucket_scc_rank;

    std::vector<double> fw_base_intervals; // Forward base intervals for each node
    std::vector<double> bw_base_intervals; // Backward base intervals for each node

    double incumbent  = std::numeric_limits<double>::infinity();
    double relaxation = std::numeric_limits<double>::infinity();
    bool   fixed      = false;

    exec::static_thread_pool            bi_pool  = exec::static_thread_pool(2);
    exec::static_thread_pool::scheduler bi_sched = bi_pool.get_scheduler();

    const int                           MERGE_SCHED_CONCURRENCY = std::thread::hardware_concurrency();
    exec::static_thread_pool            merge_pool              = exec::static_thread_pool(MERGE_SCHED_CONCURRENCY);
    exec::static_thread_pool::scheduler merge_sched             = merge_pool.get_scheduler();

    ~BucketGraph() {
        bi_pool.request_stop();
        merged_labels.clear();
    }

    int fw_buckets_size = 0;
    int bw_buckets_size = 0;

    std::vector<uint64_t> fixed_arcs_bitmap;
    std::vector<uint64_t> fw_fixed_buckets_bitmap; // Bitmap for fixed bucket arcs
    std::vector<uint64_t> bw_fixed_buckets_bitmap; // Bitmap for fixed bucket arcs

    inline bool is_arc_fixed(int from, int to) const noexcept {
        size_t bit_pos = from * nodes.size() + to;
        return (fixed_arcs_bitmap[bit_pos / 64] & (1ULL << (bit_pos % 64))) != 0;
    }

    inline bool is_arc_not_fixed(int from, int to) const noexcept {
        size_t bit_pos = from * nodes.size() + to;
        return (fixed_arcs_bitmap[bit_pos / 64] & (1ULL << (bit_pos % 64))) == 0;
    }

    inline bool fix_arc(int from, int to) noexcept {
        size_t bit_pos = from * nodes.size() + to;
        fixed_arcs_bitmap[bit_pos / 64] |= (1ULL << (bit_pos % 64));
        return true;
    }

    template <Direction D>
    inline bool is_bucket_not_fixed(int from, int to) const noexcept {
        auto  &fixed_buckets_bitmap = assign_buckets<D>(fw_fixed_buckets_bitmap, bw_fixed_buckets_bitmap);
        auto   n_buckets            = assign_buckets<D>(fw_buckets_size, bw_buckets_size);
        size_t bit_pos              = from * n_buckets + to;
        return (fixed_buckets_bitmap[bit_pos / 64] & (1ULL << (bit_pos % 64))) == 0;
    }

    inline bool is_bucket_not_fixed_forward(int from, int to) const noexcept {
        return is_bucket_not_fixed<Direction::Forward>(from, to);
    }

    inline bool is_bucket_not_fixed_backward(int from, int to) const noexcept {
        return is_bucket_not_fixed<Direction::Backward>(from, to);
    }

    template <Direction D>
    inline bool is_bucket_fixed(int from, int to) const noexcept {
        auto  &fixed_buckets_bitmap = assign_buckets<D>(fw_fixed_buckets_bitmap, bw_fixed_buckets_bitmap);
        auto   n_buckets            = assign_buckets<D>(fw_buckets_size, bw_buckets_size);
        size_t bit_pos              = from * n_buckets + to;
        return (fixed_buckets_bitmap[bit_pos / 64] & (1ULL << (bit_pos % 64))) != 0;
    }
    double gap = std::numeric_limits<double>::infinity();

    CutStorage          *cut_storage = new CutStorage();
    static constexpr int max_buckets = 10000; // Define maximum number of buckets beforehand

    std::vector<Bucket> fw_buckets;
    std::vector<Bucket> bw_buckets;

    using LabelPoolPtr = std::shared_ptr<LabelPool>;

    LabelPoolPtr                  label_pool_fw = std::make_shared<LabelPool>(100, 5000000);
    LabelPoolPtr                  label_pool_bw = std::make_shared<LabelPool>(100, 5000000);
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

    std::vector<std::vector<double>> fw_rc2_bin;
    std::vector<std::vector<double>> bw_rc2_bin;
    std::vector<std::vector<double>> fw_rc2_till_this_bin;
    std::vector<std::vector<double>> bw_rc2_till_this_bin;

    int n_fw_labels = 0;
    int n_bw_labels = 0;

    std::vector<int> num_buckets_fw;
    std::vector<int> num_buckets_bw;
    std::vector<int> num_buckets_index_fw;
    std::vector<int> num_buckets_index_bw;

    // Statistics
    int  stat_n_labels_fw = 0;
    int  stat_n_labels_bw = 0;
    uint stat_n_dom_fw    = 0;
    uint stat_n_dom_bw    = 0;

    static constexpr size_t                   PROFILE_STAGE_COUNT = static_cast<size_t>(Stage::Enumerate) + 1;
    std::array<uint64_t, PROFILE_STAGE_COUNT> profile_dominance_checks_fw{};
    std::array<uint64_t, PROFILE_STAGE_COUNT> profile_dominance_checks_bw{};
    std::array<uint64_t, PROFILE_STAGE_COUNT> profile_inner_bin_scan_labels_fw{};
    std::array<uint64_t, PROFILE_STAGE_COUNT> profile_inner_bin_scan_labels_bw{};
    std::array<uint64_t, PROFILE_STAGE_COUNT> profile_inner_bin_scan_events_fw{};
    std::array<uint64_t, PROFILE_STAGE_COUNT> profile_inner_bin_scan_events_bw{};
    std::array<uint64_t, PROFILE_STAGE_COUNT> profile_labels_tested_fw{};
    std::array<uint64_t, PROFILE_STAGE_COUNT> profile_labels_tested_bw{};
    std::array<uint64_t, PROFILE_STAGE_COUNT> profile_labels_survived_fw{};
    std::array<uint64_t, PROFILE_STAGE_COUNT> profile_labels_survived_bw{};

    std::vector<double>                R_max;
    std::vector<double>                R_min;
    std::vector<std::vector<uint64_t>> neighborhoods_bitmap; // Bitmap for neighborhoods of each
                                                             // node
    std::vector<std::vector<uint64_t>> elementarity_bitmap;  // Bitmap for elementarity sets
    std::vector<std::vector<uint64_t>> packing_bitmap;       // Bitmap for packing sets

    void define_elementarity_sets() {
        size_t num_nodes = nodes.size();

        // Initialize elementarity sets
        elementarity_bitmap.resize(num_nodes); // Bitmap for elementarity sets

        for (size_t i = 0; i < num_nodes; ++i) {
            // Each customer should appear in its own elementarity
            // set
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
    int              non_dominated_labels_per_bucket_bw = 0;

    // Running average of dominance-checks-per-non-dominant-label across
    // labeling passes. Drives considerRegenerate(): when the average
    // exceeds REGEN_RATIO_THRESHOLD we halve bucket_interval, mirroring
    // RouteOpt's regenerateGraphBucket trigger (BucketResizeFactor...).
    double regen_ratio_sum   = 0.0;
    int    regen_ratio_count = 0;

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

    template <Direction D>
    void common_initialization();

    void common_initialization() {
        merged_labels.clear();
        merged_labels.reserve(50);
        if (options.profile_labeling) { profile_reset_labeling_metrics(); }
        PARALLEL_SECTIONS(
            work, bi_sched, SECTION { common_initialization<Direction::Forward>(); },
            SECTION { common_initialization<Direction::Backward>(); });
    }
    std::vector<WarmLabelState> fw_warm_labels;
    std::vector<WarmLabelState> bw_warm_labels;

    void setup();
    void print_statistics();
    void print_labeling_profile();
    void profile_reset_labeling_metrics() noexcept;
    void profile_record_dominance_check(Direction D, Stage S) noexcept;
    void profile_record_inner_bin_scan(Direction D, Stage S, uint64_t scanned_labels) noexcept;
    void profile_record_new_label(Direction D, Stage S) noexcept;
    void profile_record_non_dominated_label(Direction D, Stage S) noexcept;
    void generate_arcs();

    ankerl::unordered_dense::map<int, std::vector<int>> fw_bucket_splits;
    ankerl::unordered_dense::map<int, std::vector<int>> bw_bucket_splits;

    template <Symmetry SYM = Symmetry::Asymmetric>
    void set_adjacency_list();
    void set_adjacency_list_manual();

    double knapsackBound(const Label *l);

    template <Stage S>
    Label *compute_label(const Label *L, const Label *L_prime, double cost = 0.0);

    struct SpliceState {
        std::array<double, R_SIZE> resources = {};
        double                     cost_delta = 0.0;
        SRC_MODE_BLOCK(SRCMap SRCmap; bool has_src = false;)
    };

    bool   BucketSetContains(const std::set<int> &bucket_set, const int &bucket);
    void   setSplit(const std::vector<double> q_star) { this->q_star = q_star; }
    int    getStage() const { return stage; }
    Status getStatus() const { return status; }
    bool   enumerationFailed() const noexcept { return enumeration_failed.load(std::memory_order_relaxed); }

    size_t current_bucket_arc_count() const noexcept {
        size_t count = 0;
        for (const auto &bucket : fw_buckets) { count += bucket.fw_bucket_arcs.size(); }
        for (const auto &bucket : bw_buckets) { count += bucket.bw_bucket_arcs.size(); }
        return count;
    }

    double relative_gap_to_incumbent(double lower_bound) const noexcept {
        if (!std::isfinite(incumbent) || !std::isfinite(lower_bound)) return std::numeric_limits<double>::infinity();
        const double denom = std::max(1.0, std::abs(incumbent));
        return std::max(0.0, incumbent - lower_bound) / denom;
    }

    bool shouldAttemptEnumeration(double lower_bound, bool cuts_stable, int exact_iterations,
                                  bool no_negative_pricing) const noexcept {
        if (!enumeration_policy.enabled || depth != 0) return false;
        if (s5 || stage >= 5) return false;
        if (!cuts_stable) return false;

        const double rel_gap = relative_gap_to_incumbent(lower_bound);
        if (!std::isfinite(rel_gap) || rel_gap > enumeration_policy.max_relative_gap) return false;

        const bool force_gap = rel_gap <= enumeration_policy.force_relative_gap;
        if (!force_gap && exact_iterations < enumeration_policy.min_exact_iterations) return false;
        if (enumeration_policy.require_nonnegative_pricing && !no_negative_pricing && !force_gap) return false;

        const size_t arc_count = current_bucket_arc_count();
        if (arc_count > static_cast<size_t>(enumeration_policy.max_total_bucket_arcs)) return false;
        const size_t denom = std::max<size_t>(1, nodes.size());
        if (arc_count / denom > static_cast<size_t>(enumeration_policy.max_bucket_arcs_per_node)) return false;

        return true;
    }

    void enableEnumeration(double lower_bound) noexcept {
        s1 = s2 = s3 = s4 = false;
        s5                 = true;
        ss                 = false;
        transition         = false;
        enumeration_failed.store(false, std::memory_order_relaxed);
        gap = std::max(0.0, incumbent - lower_bound);
    }

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

    template <Direction D>
    void capture_warm_start_labels() {
        auto &warm_labels  = assign_buckets<D>(fw_warm_labels, bw_warm_labels);
        auto &buckets      = assign_buckets<D>(fw_buckets, bw_buckets);
        auto &buckets_size = assign_buckets<D>(fw_buckets_size, bw_buckets_size);

        warm_labels.clear();
        if (!options.warm_start || just_fixed) return;

        warm_labels.reserve(buckets_size);
        for (int bucket = 0; bucket < buckets_size; ++bucket) {
            const Label *label = buckets[bucket].get_best_label();
            if (!label) continue;

            WarmLabelState snapshot;
            snapshot.cost          = label->cost;
            snapshot.node_id       = label->node_id;
            snapshot.resources     = label->resources;
            snapshot.nodes_covered = label->nodes_covered;
            snapshot.path_len      = label->path_len;
            warm_labels.push_back(std::move(snapshot));
        }
    }

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
    static inline bool is_node_visited(const std::array<uint64_t, num_words> &bitmap, int node_id) {
        int word_index = node_id >> 6;   // Determine which 64-bit segment
                                         // contains the node_id
        int bit_position = node_id & 63; // Determine the bit position within that segment
        return (bitmap[word_index] & (1ULL << bit_position)) != 0;
    }

    static inline bool visited_overlap(const std::array<uint64_t, num_words> &lhs,
                                       const std::array<uint64_t, num_words> &rhs) {
        for (size_t i = 0; i < lhs.size(); ++i) {
            if ((lhs[i] & rhs[i]) != 0) return true;
        }
        return false;
    }

    /**
     * @brief Marks a node as visited in the bitmap.
     *
     * This function sets the bit corresponding to the given node_id
     * in the provided bitmap, indicating that the node has been
     * visited.
     *
     */
    static inline void set_node_visited(std::array<uint64_t, num_words> &bitmap, int node_id) {
        int word_index = node_id >> 6;   // Determine which 64-bit segment
                                         // contains the node_id
        int bit_position = node_id & 63; // Determine the bit position within that segment
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
    static inline bool is_node_unreachable(const std::array<uint64_t, num_words> &bitmap, int node_id) {
        int word_index = node_id >> 6;   // Determine which 64-bit segment
                                         // contains the node_id
        int bit_position = node_id & 63; // Determine the bit position within that segment
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
    static inline void set_node_unreachable(std::array<uint64_t, num_words> &bitmap, int node_id) {
        int word_index = node_id >> 6;   // Determine which 64-bit segment
                                         // contains the node_id
        int bit_position = node_id & 63; // Determine the bit position within that segment
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
        for (int i = 0; i < options.resources.size(); ++i) { intervals.push_back(Interval(bucketInterval, 0)); }

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
        auto &buckets           = assign_buckets<D>(fw_buckets, bw_buckets);
        auto &buckets_size      = assign_buckets<D>(fw_buckets_size, bw_buckets_size);
        auto &num_buckets       = assign_buckets<D>(num_buckets_fw, num_buckets_bw);
        auto &num_buckets_index = assign_buckets<D>(num_buckets_index_fw, num_buckets_index_bw);

        int                 num_intervals = options.main_resources.size();
        std::vector<double> total_ranges(num_intervals);
        std::vector<double> base_intervals(num_intervals);

        // Determine the base interval and other relevant values for
        // each resource
        for (int r = 0; r < num_intervals; ++r) {
            total_ranges[r]   = R_max[r] - R_min[r];
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

            // Manually shift elements to the right to make space
            // for the new bucket
            for (int j = original_num_buckets; j > i + 1; --j) { buckets[j] = buckets[j - 1]; }

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
            ++original_num_buckets; // Adjust the count to account
                                    // for the new bucket
        }

        // Update the global bucket size
        buckets_size = original_num_buckets;

        // Now update the num_buckets_index for each node_id based
        // on the final positions
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
     * This function resets both the forward (label_pool_fw) and
     * backward (label_pool_bw) label pools to their initial states.
     * It is typically used to clear any existing labels and prepare
     * the pools for reuse.
     */
    void reset_pool() {
        label_pool_fw->fast_reset();
        label_pool_bw->fast_reset();
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
        for (size_t i = 1; i < options.size - 1; ++i) { nodes[i].setDuals(duals[i - 1]); }
    }

    /**
     * @brief Sets the distance matrix and calculates neighborhoods.
     *
     * This function assigns the provided distance matrix to the
     * internal distance matrix of the class and then calculates the
     * neighborhoods based on the given number of nearest neighbors.
     *
     */
    void set_distance_matrix(const std::vector<std::vector<double>> &distanceMatrix, int n_ng = 8) {
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
        // for (auto &row : fixed_arcs) {
        // std::fill(row.begin(), row.end(), 0);
        // }
        size_t bitmap_size = (nodes.size() * nodes.size() + 63) / 64;
        fixed_arcs_bitmap.assign(bitmap_size, 0);
    }

    void reset_fixed_buckets() {
        size_t fw_bitmap_size = (fw_buckets.size() * fw_buckets.size() + 63) / 64; // Round up division by 64
        size_t bw_bitmap_size = (bw_buckets.size() * bw_buckets.size() + 63) / 64; // Round up division by 64
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
    inline bool check_feasibility(const Label *fw_label, const Label *bw_label,
                                  const SpliceState *splice_state = nullptr) {
        if (!fw_label || !bw_label) return false;

        // Cache resources and node data.
        const auto &fw_resources = splice_state ? splice_state->resources : fw_label->resources;
        const auto &bw_resources = bw_label->resources;
        const auto &node         = nodes[fw_label->node_id];

        const size_t n_resources = options.resources.size();

        // Compute travel time only once.
        const auto travel_time = getcij(fw_label->node_id, bw_label->node_id);

        // Assuming the "time" resource is at a known index (e.g., index 0).
        // If not, you can either iterate to find it once or precompute a
        // boolean array.
        if (n_resources > 0 && options.resources[0] == "time") {
            // Check time feasibility.
            const double fw_time = splice_state ? fw_resources[0] : fw_resources[0] + travel_time + node.duration;
            if (numericutils::gt(fw_time, bw_resources[0])) return false;
        } else {
            // Fallback: iterate to check any resource named "time".
            for (size_t r = 0; r < n_resources; ++r) {
                if (options.resources[r] == "time") {
                    const double fw_time = splice_state ? fw_resources[r] : fw_resources[r] + travel_time + node.duration;
                    if (numericutils::gt(fw_time, bw_resources[r])) return false;
                }
            }
        }

        // Check additional resource feasibility (assuming resources[0] is
        // "time") and additional resources (from index 1 onward) represent
        // capacity constraints.
        for (size_t i = 1; i < n_resources; ++i) {
            const double fw_resource = splice_state ? fw_resources[i] : fw_resources[i] + node.demand;
            if (fw_resource > bw_resources[i]) return false;
        }

        return true;
    }

    void updateSplit() {
        // Base thresholds and parameters for adjustments.
        constexpr double BASE_IMBALANCE_THRESHOLD = 0.10; // Lower threshold —
                                                          // work-weighted signal
                                                          // is more sensitive
                                                          // than raw counts.
        constexpr double MIN_ADJUSTMENT_FACTOR = 0.02;
        constexpr double MAX_ADJUSTMENT_FACTOR = 0.10;
        constexpr double EMA_ALPHA             = 0.1; // Smoothing factor for EMA.

        // Static variables to persist state across calls.
        static double ema_imbalance_ratio = 0.0;

        // Prefer a work-weighted signal: total labels scanned during dominance
        // walks captures both check count and per-bin depth, which is what the
        // labeler actually pays for. Fall back to label counts when the
        // profiler hasn't recorded data yet (first iteration / profiling off).
        // Sum across stages because Stage 1-3 may be active early and Stage 4
        // dominates later iterations.
        uint64_t fw_work = 0, bw_work = 0;
        for (size_t i = 0; i < PROFILE_STAGE_COUNT; ++i) {
            fw_work += profile_inner_bin_scan_labels_fw[i];
            bw_work += profile_inner_bin_scan_labels_bw[i];
        }
        const bool   have_work_signal = (fw_work + bw_work) > 0;
        const double fw_labels    = have_work_signal ? static_cast<double>(fw_work) : static_cast<double>(n_fw_labels);
        const double bw_labels    = have_work_signal ? static_cast<double>(bw_work) : static_cast<double>(n_bw_labels);
        const double total_labels = fw_labels + bw_labels;

        // Skip adjustment if too few labels exist.
        if (total_labels < 10) return;

        // Compute imbalance ratios for forward and backward labels.
        const double fw_ratio        = fw_labels / total_labels;
        const double bw_ratio        = bw_labels / total_labels;
        const double imbalance_ratio = std::abs(fw_ratio - bw_ratio);

        // Update exponential moving average (EMA) of the imbalance ratio.
        ema_imbalance_ratio = EMA_ALPHA * imbalance_ratio + (1.0 - EMA_ALPHA) * ema_imbalance_ratio;

        // Dynamic threshold decreases as total_labels increase.
        const double dynamic_threshold = BASE_IMBALANCE_THRESHOLD * std::exp(-total_labels / 1000.0);

        // Only adjust split if the imbalance exceeds the dynamic threshold.
        if (ema_imbalance_ratio > dynamic_threshold) {
            // Severity scales between 0 and 1.
            const double severity = (ema_imbalance_ratio - dynamic_threshold) / (1.0 - dynamic_threshold);

            // Adaptive adjustment factor increases with severity.
            const double adjustment_factor =
                MIN_ADJUSTMENT_FACTOR + (MAX_ADJUSTMENT_FACTOR - MIN_ADJUSTMENT_FACTOR) * severity;

            // Compute resource range based on the main resource.
            const double resource_range = R_max[options.main_resources[0]] - R_min[options.main_resources[0]];

            // Base adjustment scales with the resource range; no decay so
            // adjustments stay aggressive when imbalance persists late.
            const double base_adjustment = adjustment_factor * resource_range;

            // Adjust each split position in q_star.
            for (size_t i = 0; i < q_star.size(); ++i) {
                // Compute relative position (0: min, 1: max) within the
                // resource range.
                const double relative_pos = (q_star[i] - R_min[options.main_resources[0]]) / resource_range;

                // Position factor gives smaller adjustments near boundaries.
                const double position_factor = 4.0 * relative_pos * (1.0 - relative_pos);

                // Determine direction: shift toward balancing label counts.
                const double direction = (bw_labels > fw_labels) ? 1.0 : -1.0;

                // Final adjustment for this split.
                const double final_adjustment = base_adjustment * position_factor * direction;

                // Apply the adjustment with boundary protection.
                q_star[i] = std::clamp(q_star[i] + final_adjustment,
                                       R_min[options.main_resources[0]] + resource_range * 0.1, // Lower bound.
                                       R_max[options.main_resources[0]] - resource_range * 0.1  // Upper bound.
                );
            }

            // Log adjustments if the base adjustment is significant.
            if (std::abs(base_adjustment) > resource_range * 0.05) {
                print_info("Split adjustment: imbalance={:.3f}, adjustment={:.3f}\n", ema_imbalance_ratio,
                           base_adjustment);
            }
        }
    }

    std::vector<std::vector<int>> topHeurRoutes;

    // Mirrors RouteOpt's considerRegenerateBucketGraph. After each labeling
    // pass we sample dominance_checks / non_dominant_labels and fold it into a
    // running average. When the average exceeds REGEN_RATIO_THRESHOLD we halve
    // bucket_interval (== RouteOpt's step_size /= 2), which doubles the bucket
    // count and gives finer per-bin partitioning to relieve crowding.
    //
    // Guards mirror RouteOpt:
    //   - need bucket_interval / 2 >= 1 (can still refine)
    //   - need bucket_interval even (clean halving)
    //   - need total bucket count below REGEN_BUCKETS_CAP (don't explode)
    bool considerRegenerate() {
        constexpr double REGEN_RATIO_THRESHOLD = 800.0;  // RouteOpt's value.
        constexpr int    REGEN_BUCKETS_CAP     = 200000; // ~10k arcs/vertex
                                                         // analog for our scale.

        // Sample the most recent labeling pass: total dominance scans across
        // all forward buckets, divided by non-dominant label count summed over
        // both directions. Avoids div-by-zero when the pass produced nothing.
        double total_checks = 0.0;
        for (int v : dominance_checks_per_bucket) total_checks += v;
        const int non_dom_total = non_dominated_labels_per_bucket + non_dominated_labels_per_bucket_bw;
        if (non_dom_total <= 0) return false;
        const double pass_ratio = total_checks / static_cast<double>(non_dom_total);

        // Update running average.
        regen_ratio_sum += pass_ratio;
        ++regen_ratio_count;
        const double avg_ratio = regen_ratio_sum / regen_ratio_count;

        if (avg_ratio <= REGEN_RATIO_THRESHOLD) return false;

        // Guards.
        if (bucket_interval <= 1 || (bucket_interval & 1) != 0) return false;
        const int total_buckets = static_cast<int>(fw_buckets.size() + bw_buckets.size());
        if (total_buckets >= REGEN_BUCKETS_CAP) return false;

        const int new_interval = bucket_interval / 2;
        print_info("Halving bucket_interval {} -> {} (avg dom-check ratio {:.1f})\n", bucket_interval, new_interval,
                   avg_ratio);
        bucket_interval = new_interval;
        redefine(bucket_interval);

        // Reset the running average so we judge the new partition on its own.
        regen_ratio_sum   = 0.0;
        regen_ratio_count = 0;
        fixed             = false;
        return true;
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
    inline int get_static_bucket_number(int node, std::vector<double> &resource_values_vec) noexcept {
        const size_t num_resources = options.main_resources.size();

        // Optionally adjust resource values (epsilon adjustments can be enabled
        // here)
        for (size_t r = 0; r < num_resources; ++r) {
            if constexpr (D == Direction::Forward) {
                // resource_values_vec[r] += numericutils::eps; // Uncomment if
                // needed.
            } else {
                // resource_values_vec[r] -= numericutils::eps; // Uncomment if
                // needed.
            }
        }

        // Query the appropriate node interval tree based on direction.
        if constexpr (D == Direction::Forward) {
            return fw_node_interval_trees[node].queryStatic(resource_values_vec);
        } else { // Direction::Backward
            return bw_node_interval_trees[node].queryStatic(resource_values_vec);
        }
    }

    template <Direction D>
    Label *get_best_label(const std::vector<int> &topological_order, const std::vector<double> &c_bar,
                          const std::vector<std::vector<int>> &sccs);

    template <Direction D>
    void define_buckets();

    template <Direction D, Stage S>
    bool DominatedInCompWiseSmallerBuckets(const Label *L, int bucket, const std::vector<double> &c_bar,
                                           std::vector<uint64_t> &Bvisited, std::vector<uint32_t> &touched_segments,
                                           uint &stat_n_dom) noexcept;

    template <Direction D, Stage S, ArcType A, Mutability M, Full F>
    inline auto Extend(const std::conditional_t<M == Mutability::Mut, Label *, const Label *>          L_prime,
                       const std::conditional_t<A == ArcType::Bucket, BucketArc,
                                                std::conditional_t<A == ArcType::Jump, JumpArc, Arc>> &gamma,
                       int depth = 0) noexcept;
    template <Direction D, Stage S>
    bool is_dominated(const Label *new_label, const Label *labels) noexcept;

    template <Stage S, Symmetry SYM = Symmetry::Asymmetric>
    std::vector<Label *> bi_labeling_algorithm();

    template <Stage S, Symmetry SYM = Symmetry::Asymmetric>
    void ConcatenateLabel(const Label *L, int &b, std::atomic<double> &best_cost,
                          const SpliceState *splice_state = nullptr);

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

        // For each conflict (u,v), add u to v's neighborhood and
        // vice versa
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
            if (from < num_nodes && to < num_nodes) {
                // fixed_arcs[from][to] = 1;
                fix_arc(from, to);
            }
        }
    }

    std::vector<Label *> extend_path(const std::vector<int> &path, std::vector<double> &resources);

    Label *compute_red_cost(const WarmLabelState &L, bool fw) {
        // Calculate main and reduced costs.
        double real_cost = 0.0;
        double red_cost  = 0.0;

        // Check main resource feasibility.
        const double main_resource = L.resources[options.main_resources[0]];
        const double q_star_value  = q_star[options.main_resources[0]];
        if (fw) {
            if (main_resource > q_star_value) return nullptr;
        } else {
            if (main_resource <= q_star_value) return nullptr;
        }

        // Ensure sufficient nodes in the route.
        if (L.path_len <= 3) return nullptr;

        // Initialize an empty SRCmap vector with one slot per pricing-active
        // cut. Stored cuts with tiny duals are ignored by pricing.
        std::vector<uint16_t> updated_SRCmap(cut_storage->activeSize(), 0);

        // Prepare new label.
        auto  pool      = fw ? label_pool_fw : label_pool_bw;
        auto  new_label = pool->acquire();
        auto &route     = new_label->nodes_covered;
        route.clear();
        route.reserve(L.nodes_covered.size());
        route.insert(route.end(), L.nodes_covered.begin(), L.nodes_covered.end());
        new_label->path_len = L.path_len;

        // Variables to keep track of costs.
        int last_node = -1;

        // Traverse the nodes in the route (from current label to root).
        for (auto node_id : L.nodes_covered) {
            // --- NG-route Feasibility Check ---
            // Clear visited bitmap in the neighborhood of node_id.
            for (size_t seg = 0, limit = new_label->visited_bitmap.size(); seg < limit; ++seg) {
                uint64_t current_visited = new_label->visited_bitmap[seg];
                if (!current_visited) continue;
                uint64_t neighborhood_mask     = neighborhoods_bitmap[node_id][seg];
                new_label->visited_bitmap[seg] = current_visited & neighborhood_mask;
            }
            set_node_visited(new_label->visited_bitmap, node_id);

            // --- Cost Accumulation ---
            if (last_node != -1) {
                double cij_cost = getcij(last_node, node_id);
                real_cost += cij_cost;
                red_cost += cij_cost;
            }
            red_cost -= nodes[node_id].cost;

            // --- SRC Logic ---
            size_t         segment     = node_id >> 6;
            const uint64_t bit_mask    = bit_mask_lookup[node_id & 63];
            auto          &cutter      = cut_storage;
            const auto     active_cuts = cutter->getActiveCuts();

#if defined(SRC)
#if !defined(SRC_MEMORY_MODE_ARC)
            for (const auto &update : cutter->getSRCNodeUpdates(node_id)) {
                auto &src_val = updated_SRCmap[update.active_idx];
                src_val += update.add;
                if (src_val >= update.den) {
                    red_cost -= update.dual;
                    src_val -= update.den;
                }
            }
            for (const auto active_idx : cutter->getSRCNodeClears(node_id)) { updated_SRCmap[active_idx] = 0; }
#else
            for (const auto &active_cut : active_cuts) {
                const size_t idx = active_cut.index;
                const auto  &cut = *active_cut.cut_ptr;

                auto &src_val = updated_SRCmap[idx];
                if (!cut.isSRCMemoryArc(last_node, node_id)) { src_val = 0; }
                if (cut.isSRCset(node_id)) {
                    src_val += cut.srcMultiplier(node_id);
                    if (src_val >= cut.p.den) {
                        red_cost -= active_cut.dual_value;
                        src_val -= cut.p.den;
                    }
                }
            }
#endif
#endif

            // --- Arc and Branching Dual Adjustments ---
#if defined(RCC) || defined(EXACT_RCC)
            if (last_node != -1) { red_cost -= arc_duals.getDual(last_node, node_id); }
#endif
            if (branching_duals->size() > 0 && last_node != -1) {
                red_cost -= branching_duals->getDual(last_node, node_id);
            }

            last_node = node_id; // Update last_node.
        }

        // Set final label properties.
        new_label->cost        = red_cost;
        new_label->real_cost   = real_cost;
        new_label->parent      = nullptr;
        new_label->node_id     = L.node_id;
        new_label->is_extended = false;
        new_label->resources   = L.resources;
        new_label->SRCmap      = updated_SRCmap;
        new_label->fresh       = false;

        // Determine bucket number for the new label.
        std::vector<double> new_resources(options.resources.size());
        for (size_t i = 0; i < options.resources.size(); ++i) { new_resources[i] = new_label->resources[i]; }
        int bucket        = fw ? get_bucket_number<Direction::Forward>(new_label->node_id, new_resources)
                               : get_bucket_number<Direction::Backward>(new_label->node_id, new_resources);
        new_label->vertex = bucket;

        return new_label;
    }

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

            // Add the node's arcs to the adjacency list if it has
            // any outgoing edges
            if (!arcs.empty()) { adjacency_list[node.id] = arcs; }
        }

        return adjacency_list;
    }

    bool shallUpdateStep() {
        int                               update_ctr = 0;
        ankerl::unordered_dense::set<int> updated_buckets;

        // Iterate over all buckets and check if either forward or backward
        // bucket indicates a split is needed.
        for (size_t b = 0; b < fw_buckets.size(); ++b) {
            if (fw_buckets[b].shall_split || bw_buckets[b].shall_split) {
                const int inner_id = fw_buckets[b].node_id;
                // Skip depot buckets.
                if (inner_id == options.depot || inner_id == options.end_depot) { continue; }
                // If this node was already updated, skip it.
                if (updated_buckets.contains(inner_id)) { continue; }
                for (auto r = 0; r < options.resources.size(); ++r) {
                    // If the bucket split value is already at maximum (100),
                    // skip.(127)
                    //
                    if (fw_bucket_splits[inner_id][r] == 100) { continue; }
                    // Update the bucket splits by doubling the current value,
                    // capped at 100.
                    const int update_target       = std::min(fw_bucket_splits[inner_id][r] * 2, 100);
                    fw_bucket_splits[inner_id][r] = update_target;
                    bw_bucket_splits[inner_id][r] = update_target;
                    updated_buckets.insert(inner_id);
                }
                update_ctr++;
            }
        }

        // If we have reached a significant number of updates and we haven't
        // accumulated too many cuts, then update bucket splits.
        const int CHANGE_THRESHOLD = nodes.size() / 6;
        if (update_ctr >= CHANGE_THRESHOLD && cut_storage->size() < 40) {
            print_info("Updating bucket splits with {} changes\n", update_ctr);

            // Redefine buckets in parallel for both forward and backward
            // directions.
            PARALLEL_SECTIONS(
                work, bi_sched,
                SECTION {
                    // Forward bucket redefinition.
                    define_buckets<Direction::Forward>();
                },
                SECTION {
                    // Backward bucket redefinition.
                    define_buckets<Direction::Backward>();
                });

            // Regenerate arcs after updating bucket splits.
            generate_arcs();

            // Process jump bucket arcs in parallel.
            PARALLEL_SECTIONS(
                workB, bi_sched, SECTION { ObtainJumpBucketArcs<Direction::Forward>(); },
                SECTION { ObtainJumpBucketArcs<Direction::Backward>(); });

            return true;
        }
        return false;
    }

    template <Direction D>
    void define_buckets(const int fixed_num_buckets) {
        // Get references to forward/backward buckets.
        auto &buckets = assign_buckets<D>(fw_buckets, bw_buckets);
        buckets.clear();

        // Also clear the fixed bitmap (if used elsewhere).
        auto &fixed_buckets_bitmap = assign_buckets<D>(fw_fixed_buckets_bitmap, bw_fixed_buckets_bitmap);
        fixed_buckets_bitmap.clear();

        // Get other direction-specific containers.
        const int num_intervals       = options.main_resources.size(); // number of resource dimensions
        auto     &num_buckets         = assign_buckets<D>(num_buckets_fw, num_buckets_bw);
        auto     &num_buckets_index   = assign_buckets<D>(num_buckets_index_fw, num_buckets_index_bw);
        auto     &node_interval_trees = assign_buckets<D>(fw_node_interval_trees, bw_node_interval_trees);
        auto     &buckets_size        = assign_buckets<D>(fw_buckets_size, bw_buckets_size);
        auto     &bucket_splits       = assign_buckets<D>(fw_bucket_splits, bw_bucket_splits);

        // Pre-allocate per-node containers.
        const size_t num_nodes = nodes.size();
        num_buckets.resize(num_nodes);
        num_buckets_index.resize(num_nodes);
        node_interval_trees.assign(num_nodes, SplayTree());

        // Lambda to calculate an interval for a given resource dimension.
        // Rounding is applied after computing boundaries.
        auto calculate_interval = [&](double lb, double ub, double base_interval, int pos, int max_interval,
                                      bool is_forward) -> std::pair<double, double> {
            double start, end;
            if (is_forward) {
                start = (pos == 0) ? lb : lb + pos * base_interval;
                end   = (pos == max_interval - 1) ? ub : lb + (pos + 1) * base_interval;
            } else {
                start = (pos == max_interval - 1) ? lb : ub - (pos + 1) * base_interval;
                end   = (pos == 0) ? ub : ub - pos * base_interval;
            }
            return {roundToTwoDecimalPlaces(start), roundToTwoDecimalPlaces(end)};
        };

        int                 bucket_index = 0;
        int                 cum_sum      = 0;
        std::vector<double> interval_start(num_intervals);
        std::vector<double> interval_end(num_intervals);
        // "pos" holds the current combination indices when
        // multi-dimensional.
        std::vector<int> pos(num_intervals, 0);

        // Process each node (job) to define its buckets.
        for (const auto &VRPNode : nodes) {
            // Compute the number of splits for each resource dimension.
            // For a single dimension, we use the fixed number directly.
            // For multiple dimensions we choose an equal split in each
            // dimension.
            std::vector<int> node_split_counts(num_intervals, 1);
            if (num_intervals == 1) {
                node_split_counts[0] = fixed_num_buckets;
            } else {
                // Compute an equal split count per dimension (rounded).
                int equal_split =
                    std::max(1, static_cast<int>(std::round(std::pow(fixed_num_buckets, 1.0 / num_intervals))));
                for (int r = 0; r < num_intervals; ++r) { node_split_counts[r] = equal_split; }
                // Adjust the first dimension if the product is less than
                // fixed_num_buckets.
                int product = 1;
                for (int r = 0; r < num_intervals; ++r) { product *= node_split_counts[r]; }
                if (product < fixed_num_buckets) {
                    int factor = fixed_num_buckets / product;
                    node_split_counts[0] *= factor;
                }
            }

            // Compute the width (base interval) for each resource
            // dimension.
            std::vector<double> node_base_interval(num_intervals);
            for (int r = 0; r < num_intervals; ++r) {
                node_base_interval[r] = (VRPNode.ub[r] - VRPNode.lb[r]) / static_cast<double>(node_split_counts[r]);
            }

            // Optionally store the per-node split information.
            // (Here we store just the first dimension's split count; you
            // could store the full vector.)
            bucket_splits[VRPNode.id] = (num_intervals == 1) ? node_split_counts[0] : node_split_counts[0];

            SplayTree node_tree;
            int       n_buckets = 0;

            if (num_intervals == 1) {
                // Single-dimensional splitting.
                for (int j = 0; j < node_split_counts[0]; ++j) {
                    auto [start, end] = calculate_interval(VRPNode.lb[0], VRPNode.ub[0], node_base_interval[0], j,
                                                           node_split_counts[0], D == Direction::Forward);
                    // Clamp to the node's bounds.
                    if constexpr (D == Direction::Backward) {
                        start = std::max(start, VRPNode.lb[0]);
                    } else {
                        end = std::min(end, VRPNode.ub[0]);
                    }
                    buckets.push_back(Bucket(VRPNode.id, std::vector<double>{start}, std::vector<double>{end}));
                    node_tree.insert(std::vector<double>{start}, std::vector<double>{end}, bucket_index++);
                    n_buckets++;
                    cum_sum++;
                }
            } else {
                // Multi-dimensional splitting.
                std::fill(pos.begin(), pos.end(), 0);
                while (true) {
                    for (int r = 0; r < num_intervals; ++r) {
                        auto [start, end] = calculate_interval(VRPNode.lb[r], VRPNode.ub[r], node_base_interval[r],
                                                               pos[r], node_split_counts[r], D == Direction::Forward);
                        interval_start[r] = start;
                        interval_end[r]   = end;
                        // Optionally clamp to global resource bounds.
                        if constexpr (D == Direction::Backward) {
                            interval_start[r] = std::max(interval_start[r], R_min[r]);
                        } else {
                            interval_end[r] = std::min(interval_end[r], R_max[r]);
                        }
                    }
                    buckets.push_back(Bucket(VRPNode.id, interval_start, interval_end));
                    node_tree.insert(interval_start, interval_end, bucket_index++);
                    n_buckets++;
                    cum_sum++;

                    // Generate next combination of splits.
                    int i = 0;
                    while (i < num_intervals) {
                        ++pos[i];
                        if (pos[i] < node_split_counts[i]) {
                            break;
                        } else {
                            pos[i] = 0;
                            ++i;
                        }
                    }
                    if (i == num_intervals) break;
                }
            }

            // Update per-node bucket bookkeeping.
            num_buckets[VRPNode.id]         = n_buckets;
            num_buckets_index[VRPNode.id]   = cum_sum - n_buckets;
            node_interval_trees[VRPNode.id] = node_tree;
        }

        // Update the overall bucket count.
        buckets_size = buckets.size();
    }

    void                   prune_ng_cycles(int max_age, int min_usage, int current_iteration);
    std::vector<CycleData> ng_cycles;
    int                    ng_iteration_counter = 0;

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
