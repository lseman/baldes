/**
 * @file BucketGraph.cpp
 * @brief Implementation of the BucketGraph class and related structures for solving vehicle routing problems (VRP).
 *
 * This file contains the implementation of the BucketGraph class, which is used to manage and solve vehicle routing
 * problems (VRP) using bucket-based graph structures. The file includes the implementation of various constructors,
 * methods, and helper functions for managing arcs, labels, and buckets within the graph.
 *
 * The main components of this file include:
 * - Arc: A structure representing an arc in the graph, with constructors for different configurations.
 * - BucketArc: A structure representing an arc between buckets in the graph.
 * - JumpArc: A structure representing a jump arc between buckets in the graph.
 * - BucketGraph: The main class representing the bucket-based graph for solving VRP, with methods for initialization,
 *   label computation, adjacency list setup, and neighborhood calculations.
 *
 * The BucketGraph class provides methods for:
 * - Initializing the graph with jobs, time horizon, and bucket intervals.
 * - Computing new labels based on existing labels.
 * - Computing phi values for buckets in forward and backward directions.
 * - Calculating neighborhoods for jobs based on the number of closest jobs.
 * - Augmenting memories in the graph by identifying and forbidding cycles.
 * - Setting the adjacency list for the graph based on travel costs and resource consumption.
 * - Common initialization tasks for setting up forward and backward buckets.
 *
 * The file also includes detailed documentation comments for each method, explaining their purpose, parameters, and
 * return values.
 */

#include "../include/BucketGraph.h"
#include "../include/BucketUtils.h"

// Implementation of Arc constructors

Arc::Arc(int from, int to, const std::vector<double> &res_inc, double cost_inc)
    : from(from), to(to), resource_increment(res_inc), cost_increment(cost_inc) {}

Arc::Arc(int from, int to, const std::vector<double> &res_inc, double cost_inc, bool fixed)
    : from(from), to(to), resource_increment(res_inc), cost_increment(cost_inc), fixed(fixed) {}

Arc::Arc(int from, int to, const std::vector<double> &res_inc, double cost_inc, double priority)
    : from(from), to(to), resource_increment(res_inc), cost_increment(cost_inc), priority(priority) {}

BucketArc::BucketArc(int from, int to, const std::vector<double> &res_inc, double cost_inc)
    : from_bucket(from), to_bucket(to), resource_increment(res_inc), cost_increment(cost_inc) {}

BucketArc::BucketArc(int from, int to, const std::vector<double> &res_inc, double cost_inc, bool fixed)
    : from_bucket(from), to_bucket(to), resource_increment(res_inc), cost_increment(cost_inc), fixed(fixed) {}

JumpArc::JumpArc(int base, int jump, const std::vector<double> &res_inc, double cost_inc)
    : base_bucket(base), jump_bucket(jump), resource_increment(res_inc), cost_increment(cost_inc) {}

/**
 * @brief Constructs a BucketGraph object.
 *
 * This constructor initializes a BucketGraph with the given jobs, time horizon, and bucket interval.
 * It sets up the forward and backward buckets, initializes the dual values for the CVRP separation,
 * and defines the intervals and resource limits.
 *
 * @param jobs A vector of VRPJob objects representing the jobs to be scheduled.
 * @param time_horizon An integer representing the total time horizon for the scheduling.
 * @param bucket_interval An integer representing the interval for the buckets.
 * @param capacity An integer representing the total capacity for the scheduling.
 * @param capacity_interval An integer representing the interval for the capacity buckets.
 */
BucketGraph::BucketGraph(const std::vector<VRPJob> &jobs, int time_horizon, int bucket_interval, int capacity,
                         int capacity_interval)
    : fw_buckets(), bw_buckets(), jobs(jobs), time_horizon(time_horizon), capacity(capacity),
      bucket_interval(bucket_interval), best_cost(std::numeric_limits<double>::infinity()), fw_best_label() {

    initInfo();
    Interval intervalTime(bucket_interval, time_horizon);
    Interval intervalCap(capacity_interval, capacity);

    intervals = {intervalTime, intervalCap};
    R_min     = {0, 0};
    R_max     = {static_cast<double>(time_horizon), static_cast<double>(capacity)};

    define_buckets<Direction::Forward>();
    define_buckets<Direction::Backward>();
}

/**
 * @brief Constructs a BucketGraph object.
 *
 * This constructor initializes a BucketGraph with the given jobs, time horizon, and bucket interval.
 * It sets up the forward and backward buckets, initializes the dual values for the CVRP separation,
 * and defines the intervals and resource limits.
 *
 * @param jobs A vector of VRPJob objects representing the jobs to be scheduled.
 * @param time_horizon An integer representing the total time horizon for the scheduling.
 * @param bucket_interval An integer representing the interval for the buckets.
 */
BucketGraph::BucketGraph(const std::vector<VRPJob> &jobs, int time_horizon, int bucket_interval)
    : fw_buckets(), bw_buckets(), jobs(jobs), time_horizon(time_horizon), bucket_interval(bucket_interval),
      best_cost(std::numeric_limits<double>::infinity()), fw_best_label() {

    cvrsep_duals.assign(jobs.size() + 2, std::vector<double>(jobs.size() + 2, 0.0));

    initInfo();
    Interval intervalTime(bucket_interval, time_horizon);

    intervals = {intervalTime};
    R_min     = {0};
    R_max     = {static_cast<double>(time_horizon)};

    define_buckets<Direction::Forward>();
    define_buckets<Direction::Backward>();
}

BucketGraph::BucketGraph(const std::vector<VRPJob> &jobs, std::vector<int> &bounds, std::vector<int> &bucket_intervals)
    : fw_buckets(), bw_buckets(), jobs(jobs), time_horizon(bounds[0]), bucket_interval(bucket_intervals[0]),
      best_cost(std::numeric_limits<double>::infinity()), fw_best_label() {

    cvrsep_duals.assign(jobs.size() + 2, std::vector<double>(jobs.size() + 2, 0.0));

    initInfo();
    for (int i = 0; i < bounds.size(); ++i) {
        Interval interval(bucket_intervals[i], bounds[i]);
        intervals.push_back(interval);
    }

    for (int i = 1; i < bounds.size(); ++i) {
        R_min.push_back(0);
        R_max.push_back(static_cast<double>(bounds[i]));
    }

    define_buckets<Direction::Forward>();
    define_buckets<Direction::Backward>();
}

/**
 * Computes a new label based on the given labels L and L_prime.
 *
 * @param L - The first label.
 * @param L_prime - The second label.
 * @return A new label computed from L and L_prime.
 */
Label *BucketGraph::compute_label(const Label *L, const Label *L_prime) {
    double cij_cost = getcij(L->job_id, L_prime->job_id);
    double new_cost = L->cost + L_prime->cost + cij_cost;

    double real_cost = L->real_cost + L_prime->real_cost + cij_cost;
#ifdef RCC
    // new_cost -= rcc_manager->getCachedDualSumForArc(L->job_id, L_prime->job_id);
#endif
    // Directly acquire new_label and set the cost
    auto new_label       = label_pool_fw.acquire();
    new_label->cost      = new_cost;
    new_label->real_cost = real_cost;

#ifdef SRC
    //  Check SRCDuals condition for specific stages
    auto        sumSRC   = 0.0;
    const auto &SRCDuals = cut_storage->SRCDuals;
    if (!SRCDuals.empty()) {
        for (size_t i = 0; i < SRCDuals.size(); ++i) {
            if (L->SRCmap[i] + L_prime->SRCmap[i] >= 1) { sumSRC += SRCDuals[i]; }
        }
    }
    new_label->cost -= sumSRC;
#endif
#ifdef SRC3
    //  Check SRCDuals condition for specific stages
    auto        sumSRC   = 0.0;
    const auto &SRCDuals = cut_storage->SRCDuals;
    if (!SRCDuals.empty()) {
        double sumSRC = 0;
        for (size_t i = 0; i < SRCDuals.size(); ++i) {
            if (SRCDuals[i] != 0 && (L->SRCmap[i] % 2 + L_prime->SRCmap[i] % 2 >= 1)) { sumSRC += SRCDuals[i]; }
        }
    }
    new_label->cost += sumSRC;
#endif

    // Reserve space for the combined jobs_covered vector in advance
    size_t total_size = 0;

    // Traverse forward list to calculate size
    for (auto L_fw = L; L_fw != nullptr; L_fw = L_fw->parent) { total_size++; }

    // Traverse backward list to calculate size
    for (auto L_bw = L_prime; L_bw != nullptr; L_bw = L_bw->parent) { total_size++; }

    // Reserve space in new_label->jobs_covered
    new_label->jobs_covered.clear();
    new_label->jobs_covered.reserve(total_size);

    // Traverse forward list and insert elements directly in reverse order
    auto L_fw = L;
    while (L_fw != nullptr) {
        new_label->jobs_covered.push_back(L_fw->job_id); // Add job_id to jobs_covered
        if (L_fw->parent == nullptr) { break; }
        L_fw = L_fw->parent; // Move to the parent
    }
    std::reverse(new_label->jobs_covered.begin(), new_label->jobs_covered.end());

    // Traverse backward list and insert elements normally
    for (auto L_bw = L_prime; L_bw != nullptr; L_bw = L_bw->parent) { new_label->jobs_covered.push_back(L_bw->job_id); }

    return new_label;
}

/**
 * Computes the phi values for a given bucket ID and direction.
 *
 * @param bucket_id The ID of the bucket for which to compute the phi values.
 * @param fw        A boolean indicating the direction of the computation. If true, computes phi values in the
 * forward direction; otherwise, computes phi values in the backward direction.
 *
 * @return A vector of integers representing the computed phi values.
 */
std::vector<int> BucketGraph::computePhi(int &bucket_id, bool fw) {
    std::vector<int> phi;

    // Ensure bucket_id is within valid bounds
    auto &buckets = fw ? fw_buckets : bw_buckets;

    if constexpr (R_SIZE > 1) {
        if (bucket_id >= buckets.size() || bucket_id < 0) return phi;

        // Determine the base intervals for all resource dimensions
        std::vector<int> total_ranges(intervals.size());
        std::vector<int> base_intervals(intervals.size());
        std::vector<int> remainders(intervals.size());

        for (int r = 0; r < intervals.size(); ++r) {
            total_ranges[r]   = static_cast<int>(R_max[r] - R_min[r] + 1); // Ensure integer type for total range
            base_intervals[r] = total_ranges[r] / static_cast<int>(intervals[r].interval);
            remainders[r]     = total_ranges[r] % static_cast<int>(intervals[r].interval);
        }

        auto &num_buckets       = fw ? num_buckets_fw : num_buckets_bw;
        auto &num_buckets_index = fw ? num_buckets_index_fw : num_buckets_index_bw;

        // Get the job ID and current bucket
        int   job_id         = buckets[bucket_id].job_id;
        auto &current_bucket = buckets[bucket_id];

        // Forward search: find buckets with -1 interval for each dimension
        if (fw) {
            for (int i = num_buckets_index[job_id]; i < num_buckets_index[job_id] + num_buckets[job_id]; ++i) {
                for (int r = 0; r < intervals.size(); ++r) {
                    if (r == 0) { // Special case for time dimension
                        if (buckets[i].job_id == job_id &&
                            buckets[i].lb[r] == current_bucket.lb[r] - base_intervals[r]) {
                            bool same_for_other_dims = true;
                            for (int other_r = 1; other_r < intervals.size(); ++other_r) {
                                if (buckets[i].lb[other_r] != current_bucket.lb[other_r]) {
                                    same_for_other_dims = false;
                                    break;
                                }
                            }
                            if (same_for_other_dims) { phi.push_back(i); }
                        }
                    } else { // Handle generic resource dimensions
                        if (buckets[i].job_id == job_id &&
                            buckets[i].lb[r] == current_bucket.lb[r] - base_intervals[r]) {
                            bool same_for_other_dims = true;
                            for (int other_r = 0; other_r < intervals.size(); ++other_r) {
                                if (other_r != r && buckets[i].lb[other_r] != current_bucket.lb[other_r]) {
                                    same_for_other_dims = false;
                                    break;
                                }
                            }
                            if (same_for_other_dims) { phi.push_back(i); }
                        }
                    }
                }
            }
        }
        // Backward search: find buckets with +1 interval for each dimension
        else {
            for (int i = num_buckets_index[job_id]; i < num_buckets_index[job_id] + num_buckets[job_id]; ++i) {
                for (int r = 0; r < intervals.size(); ++r) {
                    if (r == 0) { // Special case for time dimension
                        if (buckets[i].job_id == job_id &&
                            buckets[i].ub[r] == current_bucket.ub[r] + base_intervals[r]) {
                            bool same_for_other_dims = true;
                            for (int other_r = 1; other_r < intervals.size(); ++other_r) {
                                if (buckets[i].ub[other_r] != current_bucket.ub[other_r]) {
                                    same_for_other_dims = false;
                                    break;
                                }
                            }
                            if (same_for_other_dims) { phi.push_back(i); }
                        }
                    } else { // Handle generic resource dimensions
                        if (buckets[i].job_id == job_id &&
                            buckets[i].ub[r] == current_bucket.ub[r] + base_intervals[r]) {
                            bool same_for_other_dims = true;
                            for (int other_r = 0; other_r < intervals.size(); ++other_r) {
                                if (other_r != r && buckets[i].ub[other_r] != current_bucket.ub[other_r]) {
                                    same_for_other_dims = false;
                                    break;
                                }
                            }
                            if (same_for_other_dims) { phi.push_back(i); }
                        }
                    }
                }
            }
        }
    } else {
        auto smaller = 0;
        smaller      = fw ? bucket_id - 1 : bucket_id - 1;

        if (smaller >= buckets.size() || smaller < 0) return phi;

        //  check if smalller has the same job_id as vertex
        if (buckets[smaller].job_id == buckets[bucket_id].job_id) { phi.push_back(smaller); }
        // phi.push_back(bucket_id);
        return phi;
    }
    return phi;
}

/**
 * Calculates the neighborhoods for each job for the ng-routes.
 *
 * @param num_closest The number of closest jobs to consider for each job.
 */
void BucketGraph::calculate_neighborhoods(size_t num_closest) {
    size_t num_jobs = jobs.size();

    // Initialize the neighborhood bitmaps as vectors of uint64_t for forward and backward neighborhoods
    neighborhoods_bitmap.resize(num_jobs);                           // Forward neighborhood
    job_to_bit_map.resize(num_jobs, std::vector<int>(num_jobs, -1)); // Map job IDs to bit positions

    for (size_t i = 0; i < num_jobs; ++i) {
        std::vector<std::pair<double, size_t>> forward_distances; // Distances for forward neighbors

        for (size_t j = 0; j < num_jobs; ++j) {
            if (i != j) {
                // Forward distance (i -> j)
                double forward_distance = getcij(i, j);
                forward_distances.push_back({forward_distance, j});
            }
        }

        // Sort distances to find the closest jobs
        std::sort(forward_distances.begin(), forward_distances.end());

        // Initialize the neighborhood bitmap vector for job i (forward and backward)
        size_t num_segments = (num_jobs + 63) / 64;
        neighborhoods_bitmap[i].resize(num_segments, 0); // Resizing for forward bitmap

        // Include the job itself in both forward and backward neighborhoods
        size_t segment_self      = i / 64;
        size_t bit_position_self = i % 64;
        neighborhoods_bitmap[i][segment_self] |= (1ULL << bit_position_self); // Forward

        // Map the top 'num_closest' closest jobs for forward and set them in the backward neighborhoods
        for (size_t k = 0; k < num_closest && k < forward_distances.size(); ++k) {
            size_t job_index = forward_distances[k].second;

            // Determine the segment and the bit within the segment for the job_index (forward)
            size_t segment      = job_index / 64;
            size_t bit_position = job_index % 64;
            neighborhoods_bitmap[i][segment] |= (1ULL << bit_position); // Forward neighbor
        }
    }
}

/**
 * Augments the memories in the BucketGraph.
 *
 * This function takes a solution vector, a SparseModel, and several parameters to augment the memories in the
 * BucketGraph. It identifies cycles in the SparseModel that meet certain conditions and forbids them in the
 * BucketGraph. The function prioritizes smaller cycles and limits the number of forbidden cycles based on the
 * given parameters.
 *
 * @param solution The solution vector.
 * @param A The SparseModel.
 * @param aggressive Flag indicating whether to use aggressive augmentation.
 * @param eta1 The threshold for cycle size to be forbidden.
 * @param eta2 The maximum number of cycles to be forbidden.
 * @param eta_max The maximum size of neighborhoods involved in the cycle.
 */
void BucketGraph::augment_ng_memories(std::vector<double> &solution, std::vector<Path> &paths, bool aggressive,
                                      int eta1, int eta2, int eta_max, int nC) {
    std::set<std::pair<int, int>> forbidden_augmentations;
    std::vector<std::vector<int>> cycles;

    for (int col = 0; col < paths.size(); ++col) {

        if (solution[col] > 1e-2 && solution[col] < 1 - 1e-2) {
            std::unordered_map<int, int> visited_clients;
            std::vector<int>             cycle;
            bool                         has_cycle = false;

            for (int i = 0; i < paths[col].size(); ++i) {
                int client = paths[col][i];
                if (client == 0 || client == N_SIZE - 1) {
                    continue; // Ignore 0 in cycle detection
                }
                if (visited_clients.find(client) != visited_clients.end()) {
                    has_cycle = true;
                    // Start from the first occurrence of the repeated client to form the cycle
                    for (int j = visited_clients[client]; j <= i; ++j) { cycle.push_back(paths[col][j]); }
                    break; // Stop once the cycle is stored
                }
                visited_clients[client] = i;
            }

            if (has_cycle) { cycles.push_back(cycle); }
        }
    }

    // Sort cycles by size to prioritize smaller cycles
    std::sort(cycles.begin(), cycles.end(),
              [](const std::vector<int> &a, const std::vector<int> &b) { return a.size() < b.size(); });

    int forbidden_count = 0;

    for (const auto &cycle : cycles) {
        // Check the current sizes of neighborhoods involved in the cycle
        bool can_forbid = true;
        for (const auto &node : cycle) {
            // Count the number of 1s in neighborhoods_bitmap[node]
            int count = 0;
            for (const auto &segment : neighborhoods_bitmap[node]) {
                count += __builtin_popcountll(segment); // Counts the number of set bits (1s)
                if (count >= eta_max) {
                    can_forbid = false;
                    break;
                }
            }
            if (!can_forbid) { break; }
        }

        if (can_forbid && (cycle.size() <= eta1 || (forbidden_count < eta2 && !cycle.empty()))) {
            // Forbid the cycle
            forbidCycle(cycle, aggressive);
            forbidden_count++;
        }

        if (forbidden_count >= eta2) { break; }
    }
}

/**
 * Forbids a cycle in the bucket graph.
 *
 * This function takes a vector representing a cycle in the graph and forbids the edges
 * corresponding to the cycle. If the 'aggressive' flag is set to true, it also forbids
 * additional edges between the vertices of the cycle.
 *
 * @param cycle The vector representing the cycle in the graph.
 * @param aggressive Flag indicating whether to forbid additional edges between the vertices of the cycle.
 */
void BucketGraph::forbidCycle(const std::vector<int> &cycle, bool aggressive) {
    for (size_t i = 0; i < cycle.size() - 1; ++i) {
        int v1 = cycle[i];
        int v2 = cycle[i + 1];

        // Update the bitmap to forbid v2 in the neighborhood of v1
        size_t segment      = v2 / 64;
        size_t bit_position = v2 % 64;
        neighborhoods_bitmap[v1][segment] |= (1ULL << bit_position);

        if (aggressive) {
            segment      = v1 / 64;
            bit_position = v1 % 64;
            neighborhoods_bitmap[v2][segment] |= (1ULL << bit_position);
        }
    }
}

/**
 * @brief Sets the adjacency list for the BucketGraph.
 *
 * This function initializes the adjacency list for each job in the graph by clearing existing arcs
 * and then adding new arcs based on the travel cost and resource consumption between jobs.
 *
 * The function iterates over all jobs and for each job, it calculates the travel cost and resource
 * increments to other jobs. It then adds arcs to the adjacency list of the current job and the
 * reverse arcs to the adjacency list of the next job.
 *
 * The arcs are stored in flat containers to optimize for performance by reducing frequent reallocations.
 *
 * @note The function skips jobs with id 0 and jobs that are the same as the current job.
 *
 * @note The function also skips adding arcs if the resource increment exceeds the upper bound of the next job.
 *
 * @note The function uses a lambda function to encapsulate the logic for adding arcs for a job.
 */
void BucketGraph::set_adjacency_list() {
    for (auto &job : jobs) { job.clear_arcs(); }

    auto add_arcs_for_job = [&](const VRPJob &job, int from_bucket, std::vector<double> &res_inc) {
        using Arc = std::tuple<double, int, std::vector<double>, double>;
        std::vector<Arc> best_arcs;         // Use a flat container to store the best arcs
        best_arcs.reserve(jobs.size());     // Reserve space to avoid frequent reallocations
        std::vector<Arc> best_arcs_rev;     // Use a flat container to store the best arcs
        best_arcs_rev.reserve(jobs.size()); // Reserve space to avoid frequent reallocations

        for (const auto &next_job : jobs) {
            if (next_job.id == 0) continue;
            if (job.id == next_job.id) continue;

            auto   travel_cost = getcij(job.id, next_job.id);
            double cost_inc    = travel_cost - next_job.cost;

            for (int r = 0; r < R_SIZE; ++r) { res_inc[r] = job.consumption[r]; }
            res_inc[TIME_INDEX] += travel_cost;

            int to_bucket = next_job.id;
            if (from_bucket == to_bucket) continue;

            bool feasible = true;
            for (int r = 0; r < R_SIZE; ++r) {
                if (job.lb[r] + res_inc[r] > next_job.ub[r]) {
                    feasible = false;
                    break;
                }
            }
            if (!feasible) continue;

            // if (job.lb[TIME_INDEX] + res_inc[TIME_INDEX] > next_job.ub[TIME_INDEX]) continue;

            double aux_double = next_job.cost + 1.E-5 * next_job.start_time;
            best_arcs.emplace_back(aux_double, next_job.id, res_inc, cost_inc);
            double aux_double_rev = 1.E-5 * job.end_time;
            best_arcs_rev.emplace_back(aux_double_rev, next_job.id, res_inc, cost_inc);
        }

        for (const auto &arc : best_arcs) {
            auto [priority_value, to_bucket, res_inc_local, cost_inc] = arc;

            auto next_job = to_bucket;
            jobs[job.id].add_arc(job.id, next_job, res_inc_local, cost_inc, true, priority_value);
        }
        for (const auto &arc : best_arcs_rev) {
            auto [priority_value, to_bucket, res_inc_local, cost_inc] = arc;
            auto next_job                                             = to_bucket;

            jobs[next_job].add_arc(next_job, job.id, res_inc_local, cost_inc, false, priority_value);
        }
    };

    for (const auto &VRPJob : jobs) {
        if (VRPJob.id == N_SIZE - 1) continue;
        std::vector<double> res_inc(intervals.size());
        add_arcs_for_job(VRPJob, VRPJob.id, res_inc);
    }
}

/**
 * @brief Initializes the BucketGraph by clearing previous data and setting up forward and backward buckets.
 *
 * This function performs the following steps:
 * - Clears previous data from merged_labels, fw_c_bar, and bw_c_bar.
 * - Resizes the cost vectors fw_c_bar and bw_c_bar to match the number of forward and backward buckets,
 * respectively.
 * - Assigns the number of buckets and bucket indices for the forward direction.
 * - Calculates base intervals and total ranges for each resource dimension.
 * - Clears the forward and backward buckets.
 * - Initializes forward buckets by iterating over all intervals and setting up labels for each bucket.
 * - Initializes backward buckets by iterating over all intervals and setting up labels for each bucket.
 *
 * The function uses helper lambdas to update the current position of the intervals and to calculate the index
 * for each bucket.
 */
void BucketGraph::common_initialization() {
    // Clear previous data
    merged_labels.clear();
    merged_labels.reserve(100);
    fw_c_bar.clear();
    bw_c_bar.clear();

    // Resize cost vectors to match the number of buckets
    fw_c_bar.resize(fw_buckets.size(), std::numeric_limits<double>::infinity());
    bw_c_bar.resize(bw_buckets.size(), std::numeric_limits<double>::infinity());

    auto &num_buckets      = assign_buckets<Direction::Forward>(num_buckets_fw, num_buckets_bw);
    auto &num_bucket_index = assign_buckets<Direction::Forward>(num_buckets_index_fw, num_buckets_index_bw);

    int              num_intervals = intervals.size(); // Determine how many resources we have (number of intervals)
    std::vector<int> total_ranges(num_intervals);
    std::vector<int> base_intervals(num_intervals);
    std::vector<int> remainders(num_intervals);

    // Calculate base intervals and total ranges for each resource dimension
    for (int r = 0; r < intervals.size(); ++r) {
        total_ranges[r]   = R_max[r] - R_min[r] + 1;
        base_intervals[r] = total_ranges[r] / intervals[r].interval;
        remainders[r]     = total_ranges[r] % intervals[r].interval; // Use std::fmod for floating-point modulo
    }

    // Clear forward and backward buckets
    for (auto b = 0; b < fw_buckets_size; b++) {
        fw_buckets[b].clear();
        bw_buckets[b].clear();
    }

    auto &VRPJob = jobs[0]; // Example for the first job

    std::vector<int> job_total_ranges(num_intervals);
    for (int r = 0; r < num_intervals; ++r) { job_total_ranges[r] = VRPJob.ub[r] - VRPJob.lb[r]; }

    // Helper lambda to update current position of the intervals
    auto update_position = [&](std::vector<int> &current_pos) -> bool {
        bool done = true;
        for (int r = num_intervals - 1; r >= 0; --r) {
            current_pos[r]++;
            if (current_pos[r] * base_intervals[r] < job_total_ranges[r]) {
                done = false;
                break;
            } else {
                current_pos[r] = 0;
            }
        }
        return done;
    };

    auto calculate_index = [&](const std::vector<int> &current_pos, int &total_buckets) -> int {
        int index = 0;

        // Loop through each interval (dimension) and compute the index
        for (int r = 0; r < current_pos.size(); ++r) {
            index += current_pos[r]; // Accumulate the positional index across intervals
        }

        return index;
    };

    // Initialize forward buckets (generic for multiple dimensions)
    std::vector<int> current_pos(num_intervals, 0);

    // Iterate over all intervals for the forward direction
    while (true) {
        auto depot = label_pool_fw.acquire();
        // print num_intervals
        std::vector<double> interval_starts(num_intervals);
        for (int r = 0; r < num_intervals; ++r) {
            interval_starts[r] =
                std::min(static_cast<int>(R_max[r]), VRPJob.lb[r] + current_pos[r] * base_intervals[r]);
        }

        // Adjust to calculate index using `num_buckets[0]`, which is likely multi-dimensional for the depot
        int calculated_index =
            calculate_index(current_pos, num_buckets[0]) + num_bucket_index[0]; // Calculate index once
        depot->initialize(calculated_index, 0.0, interval_starts, 0);
        depot->is_extended = false;
        set_job_visited(depot->visited_bitmap, 0);
#ifdef SRC
        depot->SRCmap.assign(cut_storage->SRCDuals.size(), 0);
#endif
        fw_buckets[calculated_index].add_label(depot);
        fw_buckets[calculated_index].job_id = 0;

        if (update_position(current_pos)) break; // Update position and break if done
    }

    // Initialize backward buckets (generic for multiple dimensions)
    current_pos.assign(num_intervals, 0);

    // Iterate over all intervals for the backward direction
    while (true) {
        auto end_depot = label_pool_bw.acquire();

        std::vector<double> interval_ends(num_intervals);
        for (int r = 0; r < num_intervals; ++r) {
            interval_ends[r] = std::max(static_cast<int>(R_min[r]), VRPJob.ub[r] - current_pos[r] * base_intervals[r]);
        }

        // Calculate index for backward direction
        int calculated_index = calculate_index(current_pos, num_buckets[N_SIZE - 1]) +
                               num_bucket_index[N_SIZE - 1]; // Use num_buckets[0] for consistency
        // print interval_ends size
        end_depot->initialize(calculated_index, 0.0, interval_ends, N_SIZE - 1);
        end_depot->is_extended = false;
        set_job_visited(end_depot->visited_bitmap, N_SIZE - 1);
#ifdef SRC
        end_depot->SRCmap.assign(cut_storage->SRCDuals.size(), 0);
#endif
        bw_buckets[calculated_index].add_label(end_depot);
        bw_buckets[calculated_index].job_id = N_SIZE - 1;

        if (update_position(current_pos)) break; // Update position and break if done
    }
}

#include "Knapsack.h"

/**
 * Computes the knapsack bound for a given label.
 *
 * This function calculates the upper bound of the knapsack problem for a given label `l`.
 * It initializes a knapsack with the remaining capacity and iterates through the jobs to add
 * items that have not been visited and fit within the remaining capacity.
 * The function returns the difference between the label's cost and the solution to the knapsack problem.
 *
 * @param l A pointer to the Label object for which the knapsack bound is being calculated.
 * @return The computed knapsack bound as a double.
 */
double BucketGraph::knapsackBound(const Label *l) {
    Knapsack kp;
    int      rload = R_max[DEMAND_INDEX] - l->resources[DEMAND_INDEX];
    kp.setCapacity(rload);

    for (int i = 1; i < jobs.size(); ++i) {
        if (!l->visits(i) && jobs[i].consumption[DEMAND_INDEX] <= rload) {
            kp.addItem(jobs[i].cost, jobs[i].consumption[DEMAND_INDEX]);
        }
    }

    return l->cost - kp.solve();
}

/**
 * Checks if a given bucket is present in the bucket set.
 *
 * @param bucket_set The set of buckets to search in.
 * @param bucket The bucket to check for.
 * @return True if the bucket is found in the set, false otherwise.
 */
bool BucketGraph::BucketSetContains(const std::set<int> &bucket_set, const int &bucket) {
    return bucket_set.find(bucket) != bucket_set.end();
}

void BucketGraph::async_rih_processing(std::vector<Label *> initial_labels, int LABELS_MAX) {
    merged_labels_rih.clear();
    const int                                                           LABELS_MAX_RIH = 5;
    std::priority_queue<Label *, std::vector<Label *>, LabelComparator> best_labels_in;
    std::priority_queue<Label *, std::vector<Label *>, LabelComparator> best_labels_out;

    for (auto &label : initial_labels) {
        best_labels_in.push(label);
        if (best_labels_in.size() >= LABELS_MAX) break;
    }

    // RIH2, RIH3 etc. processing here...
    RIH2(best_labels_in, best_labels_out, LABELS_MAX);

    while (!best_labels_out.empty()) {
        best_labels_in.push(best_labels_out.top());
        best_labels_out.pop();
    }

    RIH4(best_labels_in, best_labels_out, LABELS_MAX);

    // RIH3(best_labels_in, best_labels_out, LABELS_MAX);

    // After processing, populate the merged_labels_improved vector
    while (!best_labels_out.empty()) {
        merged_labels_rih.push_back(best_labels_out.top());
        best_labels_out.pop();
    }

    if (merged_labels_rih.size() > LABELS_MAX_RIH) { merged_labels_rih.resize(LABELS_MAX_RIH); }

    // Sort or further process if needed
    std::sort(merged_labels_rih.begin(), merged_labels_rih.end(),
              [](const Label *a, const Label *b) { return a->cost < b->cost; });
}
