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

#include "bucket/BucketGraph.h"
#include "bucket/BucketUtils.h"

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
 * This constructor initializes a BucketGraph with the given nodes, time horizon, and bucket interval.
 * It sets up the forward and backward buckets, initializes the dual values for the CVRP separation,
 * and defines the intervals and resource limits.
 *
 */
BucketGraph::BucketGraph(const std::vector<VRPNode> &nodes, int time_horizon, int bucket_interval, int capacity,
                         int capacity_interval)
    : fw_buckets(), bw_buckets(), nodes(nodes), time_horizon(time_horizon), capacity(capacity),
      bucket_interval(bucket_interval), best_cost(std::numeric_limits<double>::infinity()), fw_best_label() {

    // initInfo();
    Interval intervalTime(bucket_interval, time_horizon);
    Interval intervalCap(capacity_interval, capacity);

    intervals = {intervalTime, intervalCap};
    R_min     = {0, 0};
    R_max     = {static_cast<double>(time_horizon), static_cast<double>(capacity)};

    PARALLEL_SECTIONS(
        work, bi_sched, SECTION { define_buckets<Direction::Forward>(); },
        SECTION { define_buckets<Direction::Backward>(); });
}

/**
 * @brief Constructs a BucketGraph object.
 *
 * This constructor initializes a BucketGraph with the given nodes, time horizon, and bucket interval.
 * It sets up the forward and backward buckets, initializes the dual values for the CVRP separation,
 * and defines the intervals and resource limits.
 *
 */
BucketGraph::BucketGraph(const std::vector<VRPNode> &nodes, int time_horizon, int bucket_interval)
    : fw_buckets(), bw_buckets(), nodes(nodes), time_horizon(time_horizon), bucket_interval(bucket_interval),
      best_cost(std::numeric_limits<double>::infinity()), fw_best_label() {

#if defined(RCC) || defined(EXACT_RCC)
    // cvrsep_duals.assign(nodes.size() + 2, std::vector<double>(nodes.size() + 2, 0.0));
#endif
    // initInfo();
    Interval intervalTime(bucket_interval, time_horizon);

    intervals = {intervalTime};
    R_min     = {0};
    R_max     = {static_cast<double>(time_horizon)};

    PARALLEL_SECTIONS(
        work, bi_sched, SECTION { define_buckets<Direction::Forward>(); },
        SECTION { define_buckets<Direction::Backward>(); });
}

BucketGraph::BucketGraph(const std::vector<VRPNode> &nodes, std::vector<int> &bounds,
                         std::vector<int> &bucket_intervals)
    : fw_buckets(), bw_buckets(), nodes(nodes), time_horizon(bounds[0]), bucket_interval(bucket_intervals[0]),
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
 * Computes a new label based on the given labels L and L_prime.
 *
 */
Label *BucketGraph::compute_label(const Label *L, const Label *L_prime) {
    double cij_cost = getcij(L->node_id, L_prime->node_id);
    double new_cost = L->cost + L_prime->cost + cij_cost;

    double real_cost = L->real_cost + L_prime->real_cost + cij_cost;

#if defined(RCC) || defined(EXACT_RCC)
    auto arc_dual = arc_duals.getDual(L->node_id, L_prime->node_id);
    new_cost -= arc_dual;
#endif

    // Branching duals
    if (branching_duals->size() > 0) { new_cost -= branching_duals->getDual(L->node_id, L_prime->node_id); }

    // Directly acquire new_label and set the cost
    auto new_label       = label_pool_fw.acquire();
    new_label->cost      = new_cost;
    new_label->real_cost = real_cost;

#ifdef SRC
    //  Check SRCDuals condition for specific stages
    auto        sumSRC   = 0.0;
    const auto &SRCDuals = cut_storage->SRCDuals;
    if (!SRCDuals.empty()) {
        size_t idx = 0;
        auto   sumSRC =
            std::transform_reduce(SRCDuals.begin(), SRCDuals.end(), 0.0, std::plus<>(), [&](const auto &dual) {
                size_t curr_idx = idx++;
                return (L->SRCmap[curr_idx] + L_prime->SRCmap[curr_idx] >= 1) ? dual : 0.0;
            });

        new_label->cost -= sumSRC;
    }

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
    new_label->cost -= sumSRC;
#endif

    new_label->nodes_covered.clear();

    // Start by inserting backward list elements
    size_t forward_size = 0;
    auto   L_bw         = L_prime;
    for (; L_bw != nullptr; L_bw = L_bw->parent) {
        new_label->nodes_covered.push_back(L_bw->node_id); // Insert backward elements directly
    }

    // Now insert forward list elements in reverse order without using std::reverse
    auto L_fw = L;
    for (; L_fw != nullptr; L_fw = L_fw->parent) {
        new_label->nodes_covered.insert(new_label->nodes_covered.begin(),
                                        L_fw->node_id); // Insert forward elements at the front
        forward_size++;
    }

    return new_label;
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

    if constexpr (R_SIZE > 1) {
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
                if (fixed_buckets[found_node->bucket_index][bucket_id] == 0)
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
                if (fixed_buckets[found_node->bucket_index][bucket_id] == 0)
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
            if (fixed_buckets[smaller][bucket_id] == 0)
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
        std::sort(forward_distances.begin(), forward_distances.end());

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

/**
 * @brief Sets the adjacency list for the BucketGraph.
 *
 * This function initializes the adjacency list for each node in the graph by clearing existing arcs
 * and then adding new arcs based on the travel cost and resource consumption between nodes.
 *
 */
void BucketGraph::set_adjacency_list() {
    // Clear existing arcs for each node
    for (auto &node : nodes) {
        node.clear_arcs(); // Remove any existing arcs associated with the node
    }

    // Lambda function to add arcs for a specific node and bucket
    auto add_arcs_for_node = [&](const VRPNode &node, int from_bucket, std::vector<double> &res_inc) {
        using Arc =
            std::tuple<double, int, std::vector<double>, double>; // Define an Arc as a tuple with priority value,
                                                                  // node id, resource increments, and cost increment

        // Containers to store the best arcs for forward and reverse directions
        std::vector<Arc> best_arcs;
        best_arcs.reserve(nodes.size()); // Reserve space for best arcs to avoid frequent memory reallocations

        std::vector<Arc> best_arcs_rev;
        best_arcs_rev.reserve(nodes.size()); // Reserve space for reverse arcs

        // Iterate over all nodes to determine potential arcs
        for (const auto &next_node : nodes) {
            if (next_node.id == options.depot) continue; // Skip the depot
            if (node.id == next_node.id) continue;       // Skip arcs to the same node

            // Calculate the travel cost between the current node and the next node
            auto   travel_cost = getcij(node.id, next_node.id);
            double cost_inc =
                travel_cost - next_node.cost; // Adjust the cost increment by subtracting the next node's cost

            // Initialize the resource increments based on the current node's consumption
            for (int r = 0; r < R_SIZE; ++r) { res_inc[r] = node.consumption[r]; }
            res_inc[TIME_INDEX] += travel_cost; // Add travel time to the resource increment

            int to_bucket = next_node.id;
            if (from_bucket == to_bucket) continue; // Skip arcs that loop back to the same bucket

            // Check feasibility of the arc based on resource constraints
            bool feasible = true;
            for (int r = 0; r < R_SIZE; ++r) {
                if (node.lb[r] + res_inc[r] >
                    next_node.ub[r]) { // If resource exceeds the upper bound of the next node, the arc is infeasible
                    feasible = false;
                    break;
                }
            }
            if (!feasible) continue; // Skip if the arc is not feasible

            // Calculate priority values for forward and reverse arcs
            double aux_double = 1.E-5 * next_node.start_time; // Small weight for start time
            best_arcs.emplace_back(aux_double, next_node.id, res_inc,
                                   cost_inc); // Store the arc for forward direction

            double aux_double_rev = 1.E-5 * node.end_time; // Small weight for end time
            best_arcs_rev.emplace_back(aux_double_rev, next_node.id, res_inc,
                                       cost_inc); // Store the arc for reverse direction
        }

        // Add forward arcs from the current node to its neighbors
        for (const auto &arc : best_arcs) {
            auto [priority_value, to_bucket, res_inc_local, cost_inc] = arc;

            auto next_node = to_bucket;
            nodes[node.id].add_arc(node.id, next_node, res_inc_local, cost_inc, true,
                                   priority_value); // Add forward arc to the adjacency list
        }

        // Add reverse arcs from neighboring nodes to the current node
        for (const auto &arc : best_arcs_rev) {
            auto [priority_value, to_bucket, res_inc_local, cost_inc] = arc;
            auto next_node                                            = to_bucket;

            nodes[next_node].add_arc(next_node, node.id, res_inc_local, cost_inc, false,
                                     priority_value); // Add reverse arc to the adjacency list
        }
    };

    // Iterate over all nodes to set the adjacency list
    for (const auto &VRPNode : nodes) {
        if (VRPNode.id == options.end_depot) continue; // Skip the last node (depot)

        // Initialize the resource increment vector based on the number of intervals
        std::vector<double> res_inc(intervals.size());

        // Add arcs for the current node
        add_arcs_for_node(VRPNode, VRPNode.id, res_inc);
    }
}

/**
 * @brief Initializes the BucketGraph by clearing previous data and setting up forward and backward buckets.
 *
 */
void BucketGraph::common_initialization() {
    // Clear previous data
    merged_labels.clear();
    merged_labels.reserve(100);
    fw_c_bar.clear();
    bw_c_bar.clear();

    // print fw_buckets size
    // fmt::print("fw_buckets size: {}\n", fw_buckets.size());
    // fmt::print("fw_buckets size: {}\n", bw_buckets_size);
    dominance_checks_per_bucket.assign(fw_buckets_size + 1, 0);
    non_dominated_labels_per_bucket = 0;

    // Resize cost vectors to match the number of buckets
    fw_c_bar.resize(fw_buckets_size, std::numeric_limits<double>::infinity());
    bw_c_bar.resize(bw_buckets_size, std::numeric_limits<double>::infinity());

    auto &num_buckets      = assign_buckets<Direction::Forward>(num_buckets_fw, num_buckets_bw);
    auto &num_bucket_index = assign_buckets<Direction::Forward>(num_buckets_index_fw, num_buckets_index_bw);

    int                 num_intervals = intervals.size(); // Determine how many resources we have (number of intervals)
    std::vector<double> total_ranges(num_intervals);
    std::vector<double> base_intervals(num_intervals);

    // Calculate base intervals and total ranges for each resource dimension
    for (int r = 0; r < intervals.size(); ++r) {
        total_ranges[r]   = R_max[r] - R_min[r];
        base_intervals[r] = total_ranges[r] / intervals[r].interval;
    }

    // Clear forward and backward buckets
    for (auto b = 0; b < fw_buckets.size(); b++) {
        fw_buckets[b].clear();
        bw_buckets[b].clear();
    }

    auto &VRPNode = nodes[0]; // Example for the first node

    std::vector<int> node_total_ranges(num_intervals);
    for (int r = 0; r < num_intervals; ++r) { node_total_ranges[r] = VRPNode.ub[r] - VRPNode.lb[r]; }

    // Helper lambda to update current position of the intervals
    auto update_position = [&](std::vector<int> &current_pos) -> bool {
        bool done = true;
        for (int r = num_intervals - 1; r >= 0; --r) {
            current_pos[r]++;
            if (current_pos[r] * base_intervals[r] < node_total_ranges[r]) {
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
            interval_starts[r] = std::min(R_max[r], VRPNode.lb[r] + current_pos[r] * base_intervals[r]);
        }

        // Adjust to calculate index using `num_buckets[0]`, which is likely multi-dimensional for the depot
        int calculated_index = calculate_index(current_pos, num_buckets[options.depot]) +
                               num_bucket_index[options.depot]; // Calculate index once
        depot->initialize(calculated_index, 0.0, interval_starts, options.depot);
        depot->is_extended = false;
        set_node_visited(depot->visited_bitmap, options.depot);
#ifdef SRC
        depot->SRCmap.assign(cut_storage->SRCDuals.size(), 0);
#endif
        fw_buckets[calculated_index].add_label(depot);
        fw_buckets[calculated_index].node_id = options.depot;

        if (update_position(current_pos)) break; // Update position and break if done
    }

    // Initialize backward buckets (generic for multiple dimensions)
    current_pos.assign(num_intervals, 0);

    // Iterate over all intervals for the backward direction
    while (true) {
        auto end_depot = label_pool_bw.acquire();

        std::vector<double> interval_ends(num_intervals);
        for (int r = 0; r < num_intervals; ++r) {
            interval_ends[r] = std::max(R_min[r], VRPNode.ub[r] - current_pos[r] * base_intervals[r]);
        }

        // Calculate index for backward direction
        int calculated_index = calculate_index(current_pos, num_buckets[options.end_depot]) +
                               num_bucket_index[options.end_depot]; // Use num_buckets[0] for consistency
        // print interval_ends size
        end_depot->initialize(calculated_index, 0.0, interval_ends, options.end_depot);
        end_depot->is_extended = false;
        set_node_visited(end_depot->visited_bitmap, options.end_depot);
#ifdef SRC
        end_depot->SRCmap.assign(cut_storage->SRCDuals.size(), 0);
#endif
        bw_buckets[calculated_index].add_label(end_depot);
        bw_buckets[calculated_index].node_id = options.end_depot;

        if (update_position(current_pos)) break; // Update position and break if done
    }
}

/**
 * @brief Initializes the BucketGraph for the mono-directional case.
 */
void BucketGraph::mono_initialization() {
    // Clear previous data
    merged_labels.clear();
    merged_labels.reserve(100);
    fw_c_bar.clear();

    // Resize cost vectors to match the number of buckets
    fw_c_bar.resize(fw_buckets.size(), std::numeric_limits<double>::infinity());

    auto &num_buckets      = assign_buckets<Direction::Forward>(num_buckets_fw, num_buckets_bw);
    auto &num_bucket_index = assign_buckets<Direction::Forward>(num_buckets_index_fw, num_buckets_index_bw);

    int                 num_intervals = intervals.size(); // Determine how many resources we have (number of intervals)
    std::vector<double> total_ranges(num_intervals);
    std::vector<double> base_intervals(num_intervals);

    // Calculate base intervals and total ranges for each resource dimension
    for (int r = 0; r < intervals.size(); ++r) {
        total_ranges[r]   = R_max[r] - R_min[r];
        base_intervals[r] = total_ranges[r] / intervals[r].interval;
    }

    // Clear forward and backward buckets
    for (auto b = 0; b < fw_buckets_size; b++) { fw_buckets[b].clear(); }

    auto &VRPNode = nodes[0]; // Example for the first node

    std::vector<int> node_total_ranges(num_intervals);
    for (int r = 0; r < num_intervals; ++r) { node_total_ranges[r] = VRPNode.ub[r] - VRPNode.lb[r]; }

    // Helper lambda to update current position of the intervals
    auto update_position = [&](std::vector<int> &current_pos) -> bool {
        bool done = true;
        for (int r = num_intervals - 1; r >= 0; --r) {
            current_pos[r]++;
            if (current_pos[r] * base_intervals[r] < node_total_ranges[r]) {
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
            interval_starts[r] = std::min(R_max[r], VRPNode.lb[r] + current_pos[r] * base_intervals[r]);
        }

        // Adjust to calculate index using `num_buckets[0]`, which is likely multi-dimensional for the depot
        int calculated_index = calculate_index(current_pos, num_buckets[options.depot]) +
                               num_bucket_index[options.depot]; // Calculate index once
        depot->initialize(calculated_index, 0.0, interval_starts, options.depot);
        depot->is_extended = false;
        set_node_visited(depot->visited_bitmap, options.depot);
#ifdef SRC
        depot->SRCmap.assign(cut_storage->SRCDuals.size(), 0);
#endif
        fw_buckets[calculated_index].add_label(depot);
        fw_buckets[calculated_index].node_id = options.depot;

        if (update_position(current_pos)) break; // Update position and break if done
    }
}

#include "Knapsack.h"

/**
 * Computes the knapsack bound for a given label.
 *
 * This function calculates the upper bound of the knapsack problem for a given label `l`.
 * It initializes a knapsack with the remaining capacity and iterates through the nodes to add
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

    for (int i = 1; i < nodes.size(); ++i) {
        if (!l->visits(i) && nodes[i].consumption[DEMAND_INDEX] <= rload) {
            kp.addItem(nodes[i].cost, nodes[i].consumption[DEMAND_INDEX]);
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

#ifdef RIH
/** Async RIH Processing
 * @brief Performs the RIH processing asynchronously.
 * @param initial_labels The initial labels to start the processing with.
 * @param LABELS_MAX The maximum number of labels to keep.
 */
void BucketGraph::async_rih_processing(std::vector<Label *> initial_labels, int LABELS_MAX) {
    merged_labels_rih.clear();
    const int                                                           LABELS_MAX_RIH = 10;
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

    RIH1(best_labels_in, best_labels_out, LABELS_MAX);

    while (!best_labels_out.empty()) {
        best_labels_in.push(best_labels_out.top());
        best_labels_out.pop();
    }

    RIH4(best_labels_in, best_labels_out, LABELS_MAX);

    while (!best_labels_out.empty()) {
        best_labels_in.push(best_labels_out.top());
        best_labels_out.pop();
    }

    RIH3(best_labels_in, best_labels_out, LABELS_MAX);
    /*
    while (!best_labels_out.empty()) {
        best_labels_in.push(best_labels_out.top());
        best_labels_out.pop();
    }
    */

    // RIH5(best_labels_in, best_labels_out, LABELS_MAX);

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
#endif

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
    fmt::print("\n+----------------------+-----------------+-----------------+\n");
    fmt::print("|{}{:<20}{}| {:<15} | {:<15} |\n", blue_bold, " Metric", reset, " Forward", " Backward");
    fmt::print("+----------------------+-----------------+-----------------+\n");

    // Print labels for forward and backward with bold blue metric
    fmt::print("|{}{:<20}{}| {:<15} | {:<15} |\n", blue_bold, " Labels", reset, stat_n_labels_fw, stat_n_labels_bw);

    // Print dominated forward and backward labels with bold blue metric
    fmt::print("|{}{:<20}{}| {:<15} | {:<15} |\n", blue_bold, " Dominance Check", reset, stat_n_dom_fw, stat_n_dom_bw);

    // Print the final horizontal line
    fmt::print("+----------------------+-----------------+-----------------+\n");

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
    // Initialize the sizes
    fixed_arcs.resize(getNodes().size());
    for (int i = 0; i < getNodes().size(); ++i) { fixed_arcs[i].resize(getNodes().size()); }

    // Resize and initialize fw_fixed_buckets and bw_fixed_buckets for std::vector<bool>
    fw_fixed_buckets.assign(fw_buckets.size(), std::vector<bool>(fw_buckets.size(), false));
    bw_fixed_buckets.assign(fw_buckets.size(), std::vector<bool>(fw_buckets.size(), false));
    // define initial relationships
    set_adjacency_list();
    generate_arcs();
    for (auto &VRPNode : nodes) { VRPNode.sort_arcs(); }

#ifdef SCHRODINGER
    sPool.distance_matrix = distance_matrix;
    sPool.setNodes(&nodes);
    // sPool.setCutStorage(cut_storage);
#endif

    // Initialize the split
    for (int i = 0; i < MAIN_RESOURCES; ++i) { q_star.push_back((R_max[i] - R_min[i] + 1) / 2); }
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
    auto new_label       = label_pool_fw.acquire();
    new_label->cost      = L->cost;      // Use the cost from L
    new_label->real_cost = L->real_cost; // Use the real cost from L

    // Calculate the number of nodes covered by the label (its ancestors)
    size_t label_size = 0;
    for (auto current_label = L; current_label != nullptr; current_label = current_label->parent) { label_size++; }

    // Reserve space in one go
    new_label->nodes_covered.clear();
    new_label->nodes_covered.reserve(label_size);

    // Insert the nodes from the label and its ancestors
    for (auto current_label = L; current_label != nullptr; current_label = current_label->parent) {
        new_label->nodes_covered.push_back(current_label->node_id);
    }

    std::reverse(new_label->nodes_covered.begin(), new_label->nodes_covered.end());

    return new_label;
}
