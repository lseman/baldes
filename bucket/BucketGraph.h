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

#include "Definitions.h"

#include "DataClasses.h"

#include <queue>
#include <set>
#include <string_view>

#include "../include/SCCFinder.h"

#define RCESPP_TOL_ZERO 1.E-6

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
    BucketOptions options;
    void          mono_initialization();
    Label        *compute_mono_label(const Label *L);

#ifdef PSTEP
    PSTEPDuals pstep_duals;
    void       setArcDuals(const PSTEPDuals &arc_duals) { this->pstep_duals = arc_duals; }
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
    std::vector<Label *> solvePSTEP() {
        std::vector<Label *> paths;
        double               inner_obj;

        reset_pool();
        mono_initialization();

        std::vector<double> forward_cbar(fw_buckets.size());

        forward_cbar = labeling_algorithm<Direction::Forward, Stage::Four, Full::Full>(q_star);

        for (auto bucket : std::ranges::iota_view(0, fw_buckets_size)) {
            auto bucket_labels = fw_buckets[bucket].get_labels();
            for (auto label : bucket_labels) {
                auto new_label = compute_mono_label(label);
                if (new_label->jobs_covered.size() < options.max_path_size) { continue; }
                paths.push_back(new_label);
            }
        }

        return paths;
    }

#endif
    void setOptions(const BucketOptions &options) { this->options = options; }

#ifdef SCHRODINGER
    SchrodingerPool sPool = SchrodingerPool(200);
#endif
    // Note: very tricky way to unroll the loop at compile time and check for disposability
    inline static constexpr std::string_view resources[] = {RESOURCES}; // RESOURCES will expand to your string list
    inline static constexpr int              resource_disposability[] = {RESOURCES_DISPOSABLE};

    template <Direction D, size_t I, size_t N, typename Gamma, typename VRPJob>
    inline constexpr bool process_all_resources(std::vector<double>              &new_resources,
                                                const std::array<double, R_SIZE> &initial_resources, const Gamma &gamma,
                                                const VRPJob &theJob);

    template <Direction D, size_t I, size_t N, typename Gamma, typename VRPJob>
    inline constexpr bool process_resource(double &new_resource, const std::array<double, R_SIZE> &initial_resources,
                                           const Gamma &gamma, const VRPJob &theJob);

    bool                s1    = true;
    bool                s2    = false;
    bool                s3    = false;
    bool                s4    = false;
    bool                s5    = false;
    bool                ss    = false;
    int                 stage = 1;
    std::vector<double> q_star;
    int                 iter        = 0;
    bool                transition  = true;
    Status              status      = Status::NotOptimal;
    RCCmanager         *rcc_manager = nullptr;

    std::vector<Label *> merged_labels_rih;

    std::thread rih_thread;

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

    int fw_buckets_size = 0;
    int bw_buckets_size = 0;

    std::vector<std::vector<bool>> fixed_arcs;
    std::vector<std::vector<bool>> fw_fixed_buckets;
    std::vector<std::vector<bool>> bw_fixed_buckets;

    double gap = std::numeric_limits<double>::infinity();

    CutStorage                 *cut_storage = nullptr;
    inline static constexpr int max_buckets = 12000; // Define maximum number of buckets beforehand

    std::array<Bucket, max_buckets> fw_buckets;
    std::array<Bucket, max_buckets> bw_buckets;
    LabelPool                       label_pool_fw = LabelPool(100);
    LabelPool                       label_pool_bw = LabelPool(100);
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

    double min_red_cost = std::numeric_limits<double>::infinity();
    bool   first_reset  = true;

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
        std::vector<Path> negative_cost_paths = sPool.get_paths_with_negative_red_cost();
        if (negative_cost_paths.size() > N_ADD) { negative_cost_paths.resize(N_ADD); }
        return negative_cost_paths;
    }
#endif

#ifdef RIH
    void async_rih_processing(std::vector<Label *> initial_labels, int LABELS_MAX);

    int RIH1(std::priority_queue<Label *, std::vector<Label *>, LabelComparator> &best_labels_in,
             std::priority_queue<Label *, std::vector<Label *>, LabelComparator> &best_labels_out, int max_n_labels);

    int RIH2(std::priority_queue<Label *, std::vector<Label *>, LabelComparator> &best_labels_in,
             std::priority_queue<Label *, std::vector<Label *>, LabelComparator> &best_labels_out, int max_n_labels);

    int RIH3(std::priority_queue<Label *, std::vector<Label *>, LabelComparator> &best_labels_in,
             std::priority_queue<Label *, std::vector<Label *>, LabelComparator> &best_labels_out, int max_n_labels);

    int RIH4(std::priority_queue<Label *, std::vector<Label *>, LabelComparator> &best_labels_in,
             std::priority_queue<Label *, std::vector<Label *>, LabelComparator> &best_labels_out, int max_n_labels);

    int RIH5(std::priority_queue<Label *, std::vector<Label *>, LabelComparator> &best_labels_in,
             std::priority_queue<Label *, std::vector<Label *>, LabelComparator> &best_labels_out, int max_n_labels);

    inline std::vector<size_t> findBestInsertionPositions(const std::vector<int> &route, int &customer);

    double calculateInsertionCost(const std::vector<int> &route, int &customer, size_t pos);
    void   performSwap(std::vector<int> &new_route, const std::vector<int> &current_route, size_t pos_i, size_t pos_j,
                       size_t best_pos_v, size_t best_pos_v_prime);
#endif

    static void          initInfo();
    void                 setSplit(std::vector<double> q_star) { this->q_star = q_star; }
    int                  getStage() const { return stage; }
    Status               getStatus() const { return status; }
    std::vector<Label *> solve();

    template <Stage state, Full fullness>
    void run_labeling_algorithms(std::vector<double> &forward_cbar, std::vector<double> &backward_cbar,
                                 const std::vector<double> &q_star);

    template <Stage S>
    void bucket_fixing(const std::vector<double> &q_star);

    template <Stage S>
    void heuristic_fixing(const std::vector<double> &q_star);

    static bool is_job_visited(const std::array<uint64_t, num_words> &bitmap, int job_id);
    static void set_job_visited(std::array<uint64_t, num_words> &bitmap, int job_id);
    static bool is_job_unreachable(const std::array<uint64_t, num_words> &bitmap, int job_id);
    static void set_job_unreachable(std::array<uint64_t, num_words> &bitmap, int job_id);

    void print_statistics();

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
    inline constexpr auto &assign_buckets(auto &FW, auto &BW) noexcept {
        return (D == Direction::Forward) ? FW : BW;
    }

    // Common Initialization for Stages
    void common_initialization();

    void setup();

    [[maybe_unused]] void redefine(int bucketInterval);

    BucketGraph() = default;
    BucketGraph(const std::vector<VRPJob> &jobs, int time_horizon, int bucket_interval, int capacity,
                int capacity_interval);
    BucketGraph(const std::vector<VRPJob> &jobs, int time_horizon, int bucket_interval);
    BucketGraph(const std::vector<VRPJob> &jobs, std::vector<int> &bounds, std::vector<int> &bucket_intervals);

    template <Direction D>
    void add_arc(int from_bucket, int to_bucket, const std::vector<double> &res_inc, double cost_inc);

    template <Direction D>
    void generate_arcs();

    template <Direction D>
    void SCC_handler();

    template <Direction D, Stage S, Full F>
    std::vector<double> labeling_algorithm(std::vector<double> q_point, bool full = false) noexcept;

    template <Direction D>
    int get_bucket_number(int job, const std::vector<double> &values) noexcept;

    template <Direction D>
    Label *get_best_label(const std::vector<int> &topological_order, const std::vector<double> &c_bar,
                          std::vector<std::vector<int>> &sccs);

    template <Direction D>
    void define_buckets();

    template <Direction D, Stage S>
    inline bool DominatedInCompWiseSmallerBuckets(const Label *L, int bucket, const std::vector<double> &c_bar,
                                                  std::vector<uint64_t>               &Bvisited,
                                                  const std::vector<std::vector<int>> &bucket_order) noexcept;

    template <Stage S>
    std::vector<Label *> bi_labeling_algorithm(std::vector<double> q_start);

    template <Stage S>
    void ConcatenateLabel(const Label *L, int &b, Label *&pbest, std::vector<uint64_t> &Bvisited);

    template <Direction D, Stage S, ArcType A, Mutability M, Full F>
    inline Label *
    Extend(std::conditional_t<M == Mutability::Mut, Label *, const Label *>                L_prime,
           const std::conditional_t<A == ArcType::Bucket, BucketArc,
                                    std::conditional_t<A == ArcType::Jump, JumpArc, Arc>> &gamma) noexcept;
    template <Direction D, Stage S>
    bool is_dominated(const Label *new_label, const Label *labels) noexcept;

    template <Direction D>
    void UpdateBucketsSet(double theta, const Label *label, std::unordered_set<int> &Bbidi, int &current_bucket,
                          std::unordered_set<int> &Bvisited);

    template <Direction D>
    void ObtainJumpBucketArcs();

    template <Direction D>
    void BucketArcElimination(double theta);

    template <Direction D>
    int get_opposite_bucket_number(int current_bucket_index);

    void reset_pool();
    void set_adjacency_list();
    void generate_arcs();
    void forbidCycle(const std::vector<int> &cycle, bool aggressive);
    void augment_ng_memories(std::vector<double> &solution, std::vector<Path> &paths, bool aggressive, int eta1,
                             int eta2, int eta_max, int nC);
    inline double        getcij(int i, int j) const { return distance_matrix[i][j]; }
    void                 calculate_neighborhoods(size_t num_closest);
    std::vector<VRPJob>  getJobs() const { return jobs; }
    std::vector<int>     computePhi(int &bucket, bool fw);
    void                 setDuals(const std::vector<double> &duals);
    void                 set_distance_matrix(const std::vector<std::vector<double>> &distanceMatrix, int n_ng = 8);
    bool                 BucketSetContains(const std::set<int> &bucket_set, const int &bucket);
    void                 reset_fixed();
    bool                 check_feasibility(const Label *fw_label, const Label *bw_label);
    std::vector<Label *> get_rih_labels() const { return merged_labels_rih; }
    double               knapsackBound(const Label *l);
    Label               *compute_label(const Label *L, const Label *L_prime);

private:
    std::vector<Interval> intervals;
    std::vector<VRPJob>   jobs;
    int                   time_horizon{};
    int                   capacity{};
    int                   bucket_interval{};

    double best_cost{};
    Label *fw_best_label{};
    Label *bw_best_label{};
};
