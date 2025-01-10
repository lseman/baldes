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
 * - Initializing the graph with nodes, time horizon, and bucket intervals.
 * - Computing new labels based on existing labels.
 * - Computing phi values for buckets in forward and backward directions.
 * - Calculating neighborhoods for nodes based on the number of closest nodes.
 * - Augmenting memories in the graph by identifying and forbidding cycles.
 * - Setting the adjacency list for the graph based on travel costs and resource consumption.
 * - Common initialization tasks for setting up forward and backward buckets.
 *
 * The file also includes detailed documentation comments for each method, explaining their purpose, parameters, and
 * return values.
 */

#include "Common.h"
#include "Definitions.h"

#include "bucket/BucketGraph.h"
#include "bucket/BucketSolve.h"
#include "bucket/BucketUtils.h"

#include "../third_party/pdqsort.h"

#include "MST.h"

#ifdef GUROBI
#include "gurobi_c++.h"
#include "gurobi_c.h"
#include "solvers/Gurobi.h"
#endif

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

JumpArc::JumpArc(int base, int jump, const std::vector<double> &res_inc, double cost_inc, int to_job)
    : base_bucket(base), jump_bucket(jump), resource_increment(res_inc), cost_increment(cost_inc), to_job(to_job) {}
/**
 * @brief Constructs a BucketGraph object.
 *
 * This constructor initializes a BucketGraph with the given nodes, time horizon, and bucket interval.
 * It sets up the forward and backward buckets, initializes the dual values for the CVRP separation,
 * and defines the intervals and resource limits.
 *
 */
BucketGraph::BucketGraph(const std::vector<VRPNode> &nodes, int horizon, int bucket_interval, int capacity,
                         int capacity_interval)
    : fw_buckets(), bw_buckets(), nodes(nodes), horizon(horizon), capacity(capacity), bucket_interval(bucket_interval),
      best_cost(std::numeric_limits<double>::infinity()), fw_best_label() {

    // initInfo();
    Interval intervalTime(bucket_interval, horizon);
    Interval intervalCap(capacity_interval, capacity);

    intervals = {intervalTime, intervalCap};
    R_min     = {0, 0};
    R_max     = {static_cast<double>(horizon), static_cast<double>(capacity)};
}

/**
 * @brief Constructs a BucketGraph object.
 *
 * This constructor initializes a BucketGraph with the given nodes, time horizon, and bucket interval.
 * It sets up the forward and backward buckets, initializes the dual values for the CVRP separation,
 * and defines the intervals and resource limits.
 *
 */
BucketGraph::BucketGraph(const std::vector<VRPNode> &nodes, int horizon, int bucket_interval)
    : fw_buckets(), bw_buckets(), nodes(nodes), horizon(horizon), bucket_interval(bucket_interval),
      best_cost(std::numeric_limits<double>::infinity()), fw_best_label() {

#if defined(RCC) || defined(EXACT_RCC)
    // cvrsep_duals.assign(nodes.size() + 2, std::vector<double>(nodes.size() + 2, 0.0));
#endif
    // initInfo();
    Interval interval(bucket_interval, horizon);

    intervals = {interval};
    R_min     = {0};
    R_max     = {static_cast<double>(horizon)};
}

BucketGraph::BucketGraph(const std::vector<VRPNode> &nodes, std::vector<int> &bounds,
                         std::vector<int> &bucket_intervals)
    : fw_buckets(), bw_buckets(), nodes(nodes), horizon(bounds[0]), bucket_interval(bucket_intervals[0]),
      best_cost(std::numeric_limits<double>::infinity()), fw_best_label() {

#if defined(RCC) || defined(EXACT_RCC)
    // cvrsep_duals.assign(nodes.size() + 2, std::vector<double>(nodes.size() + 2, 0.0));
#endif

    // initInfo();
    for (int i = 0; i < bounds.size(); ++i) {
        Interval interval(bucket_intervals[i], bounds[i]);
        intervals.push_back(interval);
    }

    for (int i = 1; i < bounds.size(); ++i) {
        R_min.push_back(0);
        R_max.push_back(static_cast<double>(bounds[i]));
    }

    PARALLEL_SECTIONS(
        work, bi_sched, SECTION { define_buckets<Direction::Forward>(); },
        SECTION { define_buckets<Direction::Backward>(); });
}

/**
 * Computes the phi values for a given bucket ID and direction.
 *
 */
std::vector<int> BucketGraph::computePhi(int &bucket_id, bool fw) {
    std::vector<int> phi;

    // Ensure bucket_id is within valid bounds
    auto &buckets             = fw ? fw_buckets : bw_buckets;
    auto &fixed_buckets       = fw ? fw_fixed_buckets : bw_fixed_buckets;
    auto &node_interval_trees = fw ? fw_node_interval_trees : bw_node_interval_trees;

    if (options.resources.size() > 1) {
        if (bucket_id >= buckets.size() || bucket_id < 0) return phi;

        std::vector<double> total_ranges(intervals.size());
        std::vector<double> base_intervals(intervals.size());

        for (int r = 0; r < intervals.size(); ++r) {
            total_ranges[r]   = R_max[r] - R_min[r]; // Ensure integer type for total range
            base_intervals[r] = total_ranges[r] / intervals[r].interval;
        }

        // Get the node ID and current bucket
        int   node_id        = buckets[bucket_id].node_id;
        auto &current_bucket = buckets[bucket_id];

        // Retrieve the pre-built Splay Tree for this node
        auto &node_tree = node_interval_trees[node_id];

        // Search for matching intervals using the existing Splay Tree
        if (fw) {
            // Forward search: find the interval just below the current bucket
            std::vector<double> target_low = current_bucket.lb;
            for (int r = 0; r < intervals.size(); ++r) {
                target_low[r] -= base_intervals[r]; // Adjust for the base intervals
            }

            TreeNode *found_node = node_tree.find(target_low);
            if (found_node != nullptr && buckets[found_node->bucket_index].node_id == node_id) {
                // Check if the found bucket is fixed
#ifdef FIX_BUCKETS
                if (is_bucket_not_fixed<Direction::Forward>(found_node->bucket_index, bucket_id))
#endif
                {
                    phi.push_back(found_node->bucket_index);
                }
            }
        } else {
            // Backward search: find the interval just above the current bucket
            std::vector<double> target_high = current_bucket.ub;
            for (int r = 0; r < intervals.size(); ++r) {
                target_high[r] += base_intervals[r]; // Adjust for the base intervals
            }

            TreeNode *found_node = node_tree.find(target_high);
            if (found_node != nullptr && buckets[found_node->bucket_index].node_id == node_id) {
                // Check if the found bucket is fixed
#ifdef FIX_BUCKETS
                if (is_bucket_not_fixed<Direction::Forward>(found_node->bucket_index, bucket_id))
#endif
                {
                    phi.push_back(found_node->bucket_index);
                }
            }
        }

    } else {
        // Handle the case where R_SIZE == 1 with a simpler approach
        int smaller = bucket_id - 1;

        if (smaller >= 0 && buckets[smaller].node_id == buckets[bucket_id].node_id) {
#ifdef FIX_BUCKETS
            if (fw ? is_bucket_not_fixed_forward(smaller, bucket_id) : is_bucket_not_fixed_backward(smaller, bucket_id))
#endif
            {
                phi.push_back(smaller);
            }
        }
    }
    return phi;
}

/**
 * Calculates the neighborhoods for each node for the ng-routes.
 *
 */
void BucketGraph::calculate_neighborhoods(size_t num_closest) {
    size_t num_nodes = nodes.size();

    // Initialize the neighborhood bitmaps as vectors of uint64_t for forward and backward neighborhoods
    neighborhoods_bitmap.resize(num_nodes); // Forward neighborhood

    for (size_t i = 0; i < num_nodes; ++i) {
        std::vector<std::pair<double, size_t>> forward_distances; // Distances for forward neighbors

        for (size_t j = 0; j < num_nodes; ++j) {
            if (i != j) {
                // Forward distance (i -> j)
                double forward_distance = getcij(i, j);
                forward_distances.push_back({forward_distance, j});
            }
        }

        // Sort distances to find the closest nodes
        pdqsort(forward_distances.begin(), forward_distances.end());

        // Initialize the neighborhood bitmap vector for node i (forward and backward)
        size_t num_segments = (num_nodes + 63) / 64;
        neighborhoods_bitmap[i].resize(num_segments, 0); // Resizing for forward bitmap

        // Include the node itself in both forward and backward neighborhoods
        size_t segment_self      = i >> 6;
        size_t bit_position_self = i & 63;
        neighborhoods_bitmap[i][segment_self] |= (1ULL << bit_position_self); // Forward

        // Map the top 'num_closest' closest nodes for forward and set them in the backward neighborhoods
        for (size_t k = 0; k < num_closest && k < forward_distances.size(); ++k) {
            size_t node_index = forward_distances[k].second;

            // Determine the segment and the bit within the segment for the node_index (forward)
            size_t segment      = node_index >> 6;
            size_t bit_position = node_index & 63;
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
 */
void BucketGraph::augment_ng_memories(std::vector<double> &solution, std::vector<Path> &paths, bool aggressive,
                                      int eta1, int eta2, int eta_max, int nC) {
    std::set<std::pair<int, int>> forbidden_augmentations;
    std::vector<std::vector<int>> cycles;

    for (int col = 0; col < paths.size(); ++col) {

        if (solution[col] > 1e-2 && solution[col] < 1 - 1e-2) {
            ankerl::unordered_dense::map<int, int> visited_clients;
            std::vector<int>                       cycle;
            bool                                   has_cycle = false;

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
    pdqsort(cycles.begin(), cycles.end(),
            [](const std::vector<int> &a, const std::vector<int> &b) { return a.size() < b.size(); });
    int forbidden_count = 0;

    for (const auto &cycle : cycles) {
        // Check the current sizes of neighborhoods involved in the cycle
        bool can_forbid = true;
        for (const auto &node : cycle) {
            // Count the number of 1s in neighborhoods_bitmap[node]
            int count = 0;
            // print size of neighborhoods_bitmap
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
 */
void BucketGraph::forbidCycle(const std::vector<int> &cycle, bool aggressive) {
    for (size_t i = 0; i < cycle.size() - 1; ++i) {
        int v1 = cycle[i];
        int v2 = cycle[i + 1];

        // Update the bitmap to forbid v2 in the neighborhood of v1
        size_t segment      = v2 >> 6;
        size_t bit_position = v2 & 63;
        neighborhoods_bitmap[v1][segment] |= (1ULL << bit_position);

        if (aggressive) {
            segment      = v1 >> 6;
            bit_position = v1 & 63;
            neighborhoods_bitmap[v2][segment] |= (1ULL << bit_position);
        }
    }
}

void BucketGraph::set_adjacency_list_manual() {
    // Clear existing arcs for each node
    for (auto &node : nodes) {
        node.clear_arcs(); // Remove any existing arcs associated with the node
    }

    // Step 1: Compute the clusters using MST-based clustering
    MST    mst_solver(nodes, [&](int from, int to) { return this->getcij(from, to); });
    double theta    = 1.0; // Experiment with different values of θ
    auto   clusters = mst_solver.cluster(theta);

    // Create a job-to-cluster mapping (cluster ID for each job/node)
    std::vector<int> job_to_cluster(nodes.size(), -1); // Mapping from job (node) to cluster ID
    for (int cluster_id = 0; cluster_id < clusters.size(); ++cluster_id) {
        for (int job : clusters[cluster_id]) { job_to_cluster[job] = cluster_id; }
    }

    // Step 2: Modify add_arcs_for_node to give priority based on cluster membership
    auto add_arcs_for_node = [&](const VRPNode &node, int from_bucket, std::vector<double> &res_inc) {
        using Arc = std::tuple<double, int, std::vector<double>,
                               double>; // Arc: priority, to_node, resource increments, cost increment

        std::vector<Arc> best_arcs;
        best_arcs.reserve(nodes.size()); // Reserve space for forward arcs

        std::vector<Arc> best_arcs_rev;
        best_arcs_rev.reserve(nodes.size()); // Reserve space for reverse arcs

        for (const auto &next_node : nodes) {
            if (!manual_arcs.has_arc(node.id, next_node.id)) continue; // Skip arcs not in the manual list

            if (next_node.id == options.depot || node.id == next_node.id) continue; // Skip depot and same node

            auto   travel_cost = getcij(node.id, next_node.id); // Calculate travel cost
            double cost_inc    = travel_cost - next_node.cost;  // Adjust cost increment by subtracting next node's cost

            for (int r = 0; r < options.resources.size(); ++r) {
                if (options.resources[r] == "time") {
                    res_inc[r] = travel_cost + node.duration; // Update resource increment based on node duration
                } else {
                    res_inc[r] = node.consumption[r];
                }
            }
            // res_inc[TIME_INDEX] += travel_cost; // Add travel time to resource increment

            int to_bucket = next_node.id;
            if (from_bucket == to_bucket) continue; // Skip arcs that loop back to the same bucket

            bool feasible = true; // Check feasibility based on resource constraints
            for (int r = 0; r < options.resources.size(); ++r) {
                if (node.lb[r] + res_inc[r] > next_node.ub[r]) {
                    feasible = false;
                    break;
                }
            }
            if (!feasible) continue; // Skip infeasible arcs

            // Step 3: Calculate priority based on cluster membership
            double priority_value;
            double reverse_priority_value;
            if (job_to_cluster[node.id] == job_to_cluster[next_node.id]) {
                // Higher priority if both nodes are in the same cluster
                priority_value         = 5.0 + 1.E-5 * next_node.start_time; // Adjust weight for same-cluster priority
                reverse_priority_value = 1.0 + 1.E-5 * node.start_time;      // Adjust weight for same-cluster priority
            } else {
                // Lower priority for cross-cluster arcs
                priority_value         = 1.0 + 1.E-5 * next_node.start_time; // Higher base value for cross-cluster arcs
                reverse_priority_value = 5.0 + 1.E-5 * node.start_time;      // Higher base value for cross-cluster arcs
            }

            best_arcs.emplace_back(priority_value, next_node.id, res_inc, cost_inc); // Store the forward arc
            best_arcs_rev.emplace_back(reverse_priority_value, next_node.id, res_inc,
                                       cost_inc); // Store the reverse arc
        }

        // Add forward arcs from the current node to its neighbors
        for (const auto &arc : best_arcs) {
            auto [priority_value, to_bucket, res_inc_local, cost_inc] = arc;
            nodes[node.id].template add_arc<Direction::Forward>(node.id, to_bucket, res_inc_local, cost_inc,
                                                                priority_value); // Add forward arc
        }

        // Add reverse arcs from neighboring nodes to the current node
        for (const auto &arc : best_arcs_rev) {
            auto [priority_value, to_bucket, res_inc_local, cost_inc] = arc;
            nodes[to_bucket].template add_arc<Direction::Backward>(to_bucket, node.id, res_inc_local, cost_inc,
                                                                   priority_value); // Add reverse arc
        }
    };

    // Step 4: Iterate over all nodes to set the adjacency list
    for (const auto &VRPNode : nodes) {
        if (VRPNode.id == options.end_depot) continue; // Skip the last node (depot)

        std::vector<double> res_inc(intervals.size());   // Resource increment vector
        add_arcs_for_node(VRPNode, VRPNode.id, res_inc); // Add arcs for the current node
    }
}

/**
 * @brief Initializes the BucketGraph by clearing previous data and setting up forward and backward buckets.
 *
 */
void BucketGraph::common_initialization() {
    // Clear previous data

    std::vector<Label *> fw_warm_labels;
    std::vector<Label *> bw_warm_labels;

    if (merged_labels.size() > 0 && options.warm_start && !just_fixed) {
        PARALLEL_SECTIONS(
            work, bi_sched,
            SECTION_CUSTOM(this, &fw_warm_labels) {
                fw_warm_labels.clear();
                fw_warm_labels.reserve(fw_buckets_size);
                for (auto bucket = 0; bucket < fw_buckets_size; ++bucket) {
                    auto &current_bucket = fw_buckets[bucket];
                    if (auto *label = current_bucket.get_best_label()) { fw_warm_labels.push_back(std::move(label)); }
                }
            },
            SECTION_CUSTOM(this, &bw_warm_labels) {
                bw_warm_labels.clear();
                bw_warm_labels.reserve(bw_buckets_size);
                for (auto bucket = 0; bucket < bw_buckets_size; ++bucket) {
                    auto &current_bucket = bw_buckets[bucket];
                    if (auto *label = current_bucket.get_best_label()) { bw_warm_labels.push_back(std::move(label)); }
                }
            });
    }

    merged_labels.clear();
    merged_labels.reserve(50);
    fw_c_bar.clear();
    bw_c_bar.clear();

    dominance_checks_per_bucket.assign(fw_buckets_size + 1, 0);
    non_dominated_labels_per_bucket = 0;

    // Resize cost vectors to match the number of buckets
    fw_c_bar.resize(fw_buckets_size, std::numeric_limits<double>::infinity());
    bw_c_bar.resize(bw_buckets_size, std::numeric_limits<double>::infinity());

    auto &num_buckets      = assign_buckets<Direction::Forward>(num_buckets_fw, num_buckets_bw);
    auto &num_bucket_index = assign_buckets<Direction::Forward>(num_buckets_index_fw, num_buckets_index_bw);

    int                 num_intervals = options.main_resources.size();
    std::vector<double> total_ranges(num_intervals);
    std::vector<double> base_intervals(num_intervals);

    auto &VRPNode = nodes[0]; // Example for the first node

    // Calculate base intervals and total ranges for each resource dimension
    for (int r = 0; r < intervals.size(); ++r) {
        base_intervals[r] = (VRPNode.ub[r] - VRPNode.lb[r]) / intervals[r].interval;
    }

    // Clear forward and backward buckets
    for (auto b = 0; b < fw_buckets.size(); b++) {
        fw_buckets[b].clear();
        bw_buckets[b].clear();
    }

    if (options.warm_start && !just_fixed) {
        // sort the warm labels for the ones with the smallest cost
        PARALLEL_SECTIONS(
            work, bi_sched,
            SECTION_CUSTOM(this, &fw_warm_labels) {
                // Sort forward labels
                pdqsort(fw_warm_labels.begin(), fw_warm_labels.end(),
                        [](const Label *a, const Label *b) { return a->cost < b->cost; });

                // Create a new vector for processed labels
                std::vector<Label *> processed_labels;
                const size_t         process_size =
                    fw_warm_labels.size() > options.n_warm_start ? options.n_warm_start : fw_warm_labels.size();
                processed_labels.reserve(process_size);

                // Process labels
                for (size_t i = 0; i < process_size; ++i) {
                    auto label = fw_warm_labels[i];
                    if (!label->fresh) { continue; }

                    // compute_red_cost returns a new label from the pool
                    auto new_label = compute_red_cost(label, true);
                    if (new_label != nullptr) {
                        // Add to bucket first since we have a valid label
                        fw_buckets[new_label->vertex].add_label(new_label);
                        processed_labels.push_back(new_label);
                    }
                }

                // Replace original vector contents
                fw_warm_labels.clear(); // Clear without deleting original labels
                fw_warm_labels = std::move(processed_labels);
            },
            SECTION_CUSTOM(this, &bw_warm_labels) {
                // Sort backward labels
                pdqsort(bw_warm_labels.begin(), bw_warm_labels.end(),
                        [](const Label *a, const Label *b) { return a->cost < b->cost; });

                // Create a new vector for processed labels
                std::vector<Label *> processed_labels;
                const size_t         process_size =
                    bw_warm_labels.size() > options.n_warm_start ? options.n_warm_start : bw_warm_labels.size();
                processed_labels.reserve(process_size);

                // Process labels
                for (size_t i = 0; i < process_size; ++i) {
                    auto label = bw_warm_labels[i];
                    if (!label->fresh) { continue; }

                    // compute_red_cost returns a new label from the pool
                    auto new_label = compute_red_cost(label, false);
                    if (new_label != nullptr) {
                        // Add to bucket first since we have a valid label
                        bw_buckets[new_label->vertex].add_label(new_label);
                        processed_labels.push_back(new_label);
                    }
                }

                // Replace original vector contents
                bw_warm_labels.clear(); // Clear without deleting original labels
                bw_warm_labels = std::move(processed_labels);
            });
    }
    // Initialize forward buckets (generic for multiple dimensions)
    std::vector<int> current_pos(num_intervals, 0);

    std::vector<double> interval_starts(num_intervals);
    std::vector<double> interval_ends(num_intervals);

    // Initialize interval_starts to lower bounds and interval_ends to upper bounds
    for (int r = 0; r < num_intervals; ++r) {
        interval_starts[r] = VRPNode.lb[r];
        interval_ends[r]   = VRPNode.ub[r];
    }
    int offset_fw = 0;
    int offset_bw = 0;

    // Lambda to handle forward and backward direction initialization across combinations
    auto initialize_intervals_combinations = [&](bool is_forward) {
        int        &offset                = is_forward ? offset_fw : offset_bw;
        auto       &label_pool            = is_forward ? label_pool_fw : label_pool_bw;
        auto       &buckets               = is_forward ? fw_buckets : bw_buckets;
        const auto &depot_id              = is_forward ? options.depot : options.end_depot;
        int         calculated_index_base = is_forward ? options.depot : num_buckets_index_bw[options.end_depot];

        std::vector<int>    current_pos(num_intervals, 0); // Tracks current position in each interval dimension
        std::vector<double> interval_bounds(num_intervals);

        // Recursive lambda to generate all combinations of intervals
        std::function<void(int)> generate_combinations = [&](int depth) {
            if (depth == num_intervals) {
                // All dimensions processed, now initialize for the current combination
                auto depot            = label_pool->acquire();
                int  calculated_index = calculated_index_base + offset;

                // Set interval_bounds to current combination of interval starts or ends
                for (int r = 0; r < num_intervals; ++r) {
                    interval_bounds[r] =
                        is_forward ? interval_starts[r] + current_pos[r] * roundToTwoDecimalPlaces(base_intervals[r])
                                   : interval_ends[r] - current_pos[r] * roundToTwoDecimalPlaces(base_intervals[r]);
                }

                // print interval
                // Initialize depot with the current interval boundaries
                depot->initialize(calculated_index, 0.0, interval_bounds, depot_id);
                depot->is_extended = false;
                // depot->nodes_covered.push_back(depot_id);
                set_node_visited(depot->visited_bitmap, depot_id);
                SRC_MODE_BLOCK(depot->SRCmap.assign(cut_storage->SRCDuals.size(), 0);)
                buckets[calculated_index].add_label(depot);
                buckets[calculated_index].node_id = depot_id;

                offset++; // Increment offset for each combination
                return;
            }

            // Loop through each interval in the current dimension
            for (int k = 0; k < intervals[depth].interval; ++k) {
                current_pos[depth] = k;
                generate_combinations(depth + 1);
            }
        };

        // Start generating combinations from depth 0
        generate_combinations(0);
    };

    initialize_intervals_combinations(true);  // Forward direction
    initialize_intervals_combinations(false); // Backward direction
}

/**
 * @brief Initializes the BucketGraph for the mono-directional case.
 */
void BucketGraph::mono_initialization() {
    // Clear previous data
    merged_labels.clear();
    merged_labels.reserve(100);
    fw_c_bar.clear();
    bw_c_bar.clear();

    dominance_checks_per_bucket.assign(fw_buckets_size + 1, 0);
    non_dominated_labels_per_bucket = 0;

    // Resize cost vectors to match the number of buckets
    fw_c_bar.resize(fw_buckets_size, std::numeric_limits<double>::infinity());
    bw_c_bar.resize(bw_buckets_size, std::numeric_limits<double>::infinity());

    auto &num_buckets      = assign_buckets<Direction::Forward>(num_buckets_fw, num_buckets_bw);
    auto &num_bucket_index = assign_buckets<Direction::Forward>(num_buckets_index_fw, num_buckets_index_bw);

    int                 num_intervals = options.main_resources.size();
    std::vector<double> total_ranges(num_intervals);
    std::vector<double> base_intervals(num_intervals);

    auto &VRPNode = nodes[0]; // Example for the first node

    // Calculate base intervals and total ranges for each resource dimension
    for (int r = 0; r < intervals.size(); ++r) {
        base_intervals[r] = (VRPNode.ub[r] - VRPNode.lb[r]) / intervals[r].interval;
    }

    // Clear forward and backward buckets
    for (auto b = 0; b < fw_buckets.size(); b++) {
        fw_buckets[b].clear();
        bw_buckets[b].clear();
    }

    // Initialize forward buckets (generic for multiple dimensions)
    std::vector<int> current_pos(num_intervals, 0);

    std::vector<double> interval_starts(num_intervals);
    std::vector<double> interval_ends(num_intervals);

    // Initialize interval_starts to lower bounds and interval_ends to upper bounds
    for (int r = 0; r < num_intervals; ++r) {
        interval_starts[r] = VRPNode.lb[r];
        interval_ends[r]   = VRPNode.ub[r];
    }
    int offset_fw = 0;
    int offset_bw = 0;

    // Lambda to handle forward and backward direction initialization across combinations
    auto initialize_intervals_combinations = [&](bool is_forward) {
        int        &offset     = is_forward ? offset_fw : offset_bw;
        auto       &label_pool = is_forward ? label_pool_fw : label_pool_bw;
        auto       &buckets    = is_forward ? fw_buckets : bw_buckets;
        const auto &depot_id   = is_forward ? options.depot : options.end_depot;
        // print depot_id
        int calculated_index_base =
            is_forward ? num_buckets_index_fw[options.depot] : num_buckets_index_bw[options.end_depot];

        std::vector<int>    current_pos(num_intervals, 0); // Tracks current position in each interval dimension
        std::vector<double> interval_bounds(num_intervals);

        // Recursive lambda to generate all combinations of intervals
        std::function<void(int)> generate_combinations = [&](int depth) {
            if (depth == num_intervals) {
                // All dimensions processed, now initialize for the current combination
                auto depot            = label_pool->acquire();
                int  calculated_index = calculated_index_base + offset;

                // Set interval_bounds to current combination of interval starts or ends
                for (int r = 0; r < num_intervals; ++r) {
                    interval_bounds[r] =
                        is_forward ? interval_starts[r] + current_pos[r] * roundToTwoDecimalPlaces(base_intervals[r])
                                   : interval_ends[r] - current_pos[r] * roundToTwoDecimalPlaces(base_intervals[r]);
                }

                // print interval
                // Initialize depot with the current interval boundaries
                depot->initialize(calculated_index, 0.0, interval_bounds, depot_id);
                depot->is_extended = false;
                depot->cost += pstep_duals.getThreeTwoDualValue(depot_id);
                depot->cost += pstep_duals.getThreeThreeDualValue(depot_id);
                set_node_visited(depot->visited_bitmap, depot_id);
                SRC_MODE_BLOCK(depot->SRCmap.assign(cut_storage->SRCDuals.size(), 0);)
                buckets[calculated_index].add_label(depot);
                buckets[calculated_index].node_id = depot_id;

                offset++; // Increment offset for each combination
                return;
            }

            // Loop through each interval in the current dimension
            for (int k = 0; k < intervals[depth].interval; ++k) {
                current_pos[depth] = k;
                generate_combinations(depth + 1);
            }
        };

        // Start generating combinations from depth 0
        generate_combinations(0);
    };

    // Call the lambda for both forward and backward directions, ensuring all combinations are processed
    initialize_intervals_combinations(true); // Forward direction
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
void BucketGraph::print_statistics() {
    const char *blue_bold = "\033[1;34m"; // Bold blue for metrics
    const char *green     = "\033[32m";   // Green for values (backward labels)
    const char *reset     = "\033[0m";    // Reset color

    // Print table header with horizontal line and separators
    fmt::print("\n+--------------------------------------------------------+\n");
    fmt::print("|{}{:<20}{}| {:<15} | {:<15} |\n", blue_bold, " Metric", reset, " Forward", " Backward");
    fmt::print("+--------------------------------------------------------+\n");

    // Print labels for forward and backward with bold blue metric
    fmt::print("|{}{:<20}{}| {:<15} | {:<15} |\n", blue_bold, " Labels", reset, stat_n_labels_fw, stat_n_labels_bw);

    // Print dominated forward and backward labels with bold blue metric
    fmt::print("|{}{:<20}{}| {:<15} | {:<15} |\n", blue_bold, " Dominance Check", reset, stat_n_dom_fw, stat_n_dom_bw);

    // Print the final horizontal line
    fmt::print("+--------------------------------------------------------+\n");

    fmt::print("\n");
}

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
void BucketGraph::generate_arcs() {

    PARALLEL_SECTIONS(
        work, bi_sched,
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

/**
 * @brief Sets up the initial configuration for the BucketGraph.
 *
 */
void BucketGraph::setup() {

    PARALLEL_SECTIONS(
        work, bi_sched, SECTION { define_buckets<Direction::Forward>(); },
        SECTION { define_buckets<Direction::Backward>(); });
    // Initialize the sizes
    fixed_arcs.resize(getNodes().size());
    for (int i = 0; i < getNodes().size(); ++i) { fixed_arcs[i].resize(getNodes().size()); }

    // Resize and initialize fw_fixed_buckets and bw_fixed_buckets for std::vector<bool>
    // fw_fixed_buckets.assign(fw_buckets.size(), std::vector<bool>(fw_buckets.size(), false));
    // bw_fixed_buckets.assign(fw_buckets.size(), std::vector<bool>(fw_buckets.size(), false));

    size_t fw_bitmap_size = (fw_buckets.size() * fw_buckets.size() + 63) / 64; // Round up division by 64
    size_t bw_bitmap_size = (bw_buckets.size() * bw_buckets.size() + 63) / 64; // Round up division by 64
    fw_fixed_buckets_bitmap.assign(fw_bitmap_size, 0);
    bw_fixed_buckets_bitmap.assign(bw_bitmap_size, 0);

    // define initial relationships
    nodes.resize(options.size);
    if (!options.manual_arcs) {
        if (options.symmetric) {
            set_adjacency_list<Symmetry::Symmetric>();
        } else {
            set_adjacency_list<Symmetry::Asymmetric>();
        }
    } else {
        set_adjacency_list_manual();
    }
    generate_arcs();
    for (auto &VRPNode : nodes) { VRPNode.sort_arcs(); }

#ifdef SCHRODINGER
    sPool.distance_matrix = distance_matrix;
    sPool.setNodes(&nodes);
    // sPool.setCutStorage(cut_storage);
#endif

    // Initialize the split
    for (int i = 0; i < options.main_resources.size(); ++i) { q_star.push_back((R_max[i] - R_min[i] + 1) / 2); }
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
void BucketGraph::initInfo() {

    // Print header
    fmt::print("\n+----------------------------------+\n");
    fmt::print("|        CONFIGURATION INFO     |\n");
    fmt::print("+----------------------------------+\n");

    // Print Resource size
    fmt::print("Resources: {}\n", R_SIZE);
    fmt::print("Number of Clients: {}\n", N_SIZE - 2);

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
 * @brief Computes the mono label for the BucketGraph.
 *
 * This function computes the mono label for the BucketGraph by acquiring a new label from the label pool
 * and setting the cost and real cost values from the given label `L`. It then calculates the number of nodes
 * covered by the label and its ancestors, reserves space for the nodes covered, and inserts the nodes from the
 *
 */
Label *BucketGraph::compute_mono_label(const Label *L) {
    // Directly acquire new_label and set the cost
    auto new_label           = new Label();
    new_label->cost          = L->cost;      // Use the cost from L
    new_label->real_cost     = L->real_cost; // Use the real cost from L
    new_label->nodes_covered = L->nodes_covered;

    // Calculate the number of nodes covered by the label (its ancestors)
    // size_t label_size = 0;
    // for (auto current_label = L; current_label != nullptr; current_label = current_label->parent) { label_size++; }

    // Reserve space in one go
    // new_label->nodes_covered.reserve(label_size);

    // Insert the nodes from the label and its ancestors
    // for (auto current_label = L; current_label != nullptr; current_label = current_label->parent) {
    // new_label->nodes_covered.push_back(current_label->node_id);
    // }

    // std::reverse(new_label->nodes_covered.begin(), new_label->nodes_covered.end());

    return new_label;
}

std::vector<Label *> BucketGraph::extend_path(const std::vector<int> &path, std::vector<double> &resources) {

    // Add the new nodes to the path
    auto label     = label_pool_fw->acquire();
    label->node_id = path.back();
    label->cost    = 0.0;
    for (size_t i = 0; i < resources.size(); ++i) { label->resources[i] = resources[i]; }
    label->addRoute(path);
    for (auto node : path) { set_node_visited(label->visited_bitmap, node); }

    label->cost = 0.0;
    std::vector<Label *> new_labels;
    const auto          &arcs = nodes[label->node_id].get_arcs<Direction::Forward>();
    for (const auto &arc : arcs) {
        auto labels =
            Extend<Direction::Forward, Stage::Extend, ArcType::Node, Mutability::Mut, Full::Partial>(label, arc);
        for (auto new_label : labels) { new_labels.push_back(new_label); }
    }

    // std::vector<Label *> new_labels_to_return;

    // for (auto new_label : new_labels) {
    //     //new_label = compute_mono_label(new_label);
    //     // add label->nodes_covered to the front of new_labels->nodes_covered
    //     //new_label->nodes_covered.insert(new_label->nodes_covered.begin(), label->nodes_covered.begin(),
    //                                     // label->nodes_covered.end() - 1);
    //     // print new_labels->nodes_covered
    //     new_labels_to_return.push_back(new_label);
    // }
    return new_labels;
}
