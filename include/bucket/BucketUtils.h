/**
 * @file BucketUtils.h
 * @brief Header file for utilities related to the Bucket Graph in the Vehicle Routing Problem (VRP).
 *
 * This file contains various template functions and algorithms for managing buckets in the Bucket Graph. The
 * Bucket Graph is a representation of the VRP problem, where nodes are assigned to "buckets" based on resource
 * intervals, and arcs represent feasible transitions between buckets. The utilities provided include adding arcs,
 * defining buckets, generating arcs, extending labels, and managing strongly connected components (SCCs).
 *
 * Key components:
 * - `add_arc`: Adds directed arcs between buckets based on the direction and resource increments.
 * - `get_bucket_number`: Computes the bucket number for a given node and resource values.
 * - `define_buckets`: Defines the structure and intervals for the buckets based on resource bounds.
 * - `generate_arcs`: Generates arcs between buckets based on resource constraints and feasibility.
 * - `SCC_handler`: Identifies and processes SCCs in the bucket graph.
 * - `Extend`: Extends a label with a given arc, checking for feasibility based on resources.
 *
 * The utilities use template parameters for direction (Forward or Backward), stages, and other configurations,
 * allowing flexible handling of the bucket graph in both directions.
 */

#pragma once

#include "BucketJump.h"
#include "Definitions.h"
#include "Trees.h"
#include "cuts/SRC.h"
#include <cstring>

#include "Bucket.h"
#include "utils/NumericUtils.h"

/**
 * @brief Represents a bucket in the Bucket Graph.
 * Adds an arc to the bucket graph.
 *
 */
template <Direction D>
void BucketGraph::add_arc(int from_bucket, int to_bucket, const std::vector<double> &res_inc, double cost_inc) {

    if constexpr (D == Direction::Forward) {
        fw_arcs.emplace_back(from_bucket, to_bucket, res_inc, cost_inc);
        fw_bucket_graph[from_bucket].push_back(to_bucket);

    } else if constexpr (D == Direction::Backward) {

        bw_arcs.emplace_back(from_bucket, to_bucket, res_inc, cost_inc);
        bw_bucket_graph[from_bucket].push_back(to_bucket);
    }
}

/**
 * @brief Get the bucket number for a given node and value.
 *
 * This function returns the bucket number for a given node and value in the bucket graph.
 * The bucket graph is represented by the `buckets` vector, which contains intervals of buckets.
 * The direction of the bucket graph is specified by the template parameter `D`.
 */
/*
template <Direction D>
inline int BucketGraph::get_bucket_number(int node, std::vector<double> &resource_values_vec) noexcept {
    const int num_intervals = MAIN_RESOURCES; // Number of resources
    auto     &theNode       = nodes[node];    // Get the node

    auto &base_intervals    = assign_buckets<D>(fw_base_intervals, bw_base_intervals);
    auto &num_buckets_index = assign_buckets<D>(num_buckets_index_fw, num_buckets_index_bw);
    auto &num_buckets       = assign_buckets<D>(num_buckets_fw, num_buckets_bw);

    int bucket_offset = num_buckets_index[node];
    int cumulative_base = 1;

    // Precompute base intervals only once
    std::array<double, MAIN_RESOURCES> node_base_interval;
    for (int r = 0; r < num_intervals; ++r) {
        node_base_interval[r] = (theNode.ub[r] - theNode.lb[r]) / intervals[r].interval;
    }

    for (int r = 0; r < num_intervals; ++r) {
        double value = roundToTwoDecimalPlaces(resource_values_vec[r]);
        int i = 0;

        if constexpr (D == Direction::Forward) {
            // Forward: map value to bucket based on lb
            i = static_cast<int>((value - theNode.lb[r]) / node_base_interval[r]);
        } else if constexpr (D == Direction::Backward) {
            // Backward: map value to bucket based on ub
            i = static_cast<int>((theNode.ub[r] - value) / node_base_interval[r]);
        }

        // Bound the index to valid intervals
        i = std::clamp(i, 0, intervals[r].interval - 1);

        // Update bucket offset and cumulative base
        bucket_offset += cumulative_base * i;
        cumulative_base *= intervals[r].interval; // Update base for the next dimension
    }

    return bucket_offset;
}

*/

template <Direction D>
inline int BucketGraph::get_bucket_number(int node, std::vector<double> &resource_values_vec) noexcept {

    for (int r = 0; r < MAIN_RESOURCES; ++r) {
        resource_values_vec[r] = roundToTwoDecimalPlaces(resource_values_vec[r]);
    }
    if constexpr (D == Direction::Forward) {
        auto val = fw_node_interval_trees[node].query(resource_values_vec);
        return val;
    } else if constexpr (D == Direction::Backward) {
        auto val = bw_node_interval_trees[node].query(resource_values_vec);
        return val;
    }
    // std::throw_with_nested(std::runtime_error("BucketGraph::get_bucket_number: Invalid direction"));
    return -100; // If no bucket is found
}

/**
 * @brief Defines the buckets for the BucketGraph.
 *
 * This function determines the number of buckets based on the time intervals and assigns buckets to the graph.
 * It computes resource bounds for each vertex and defines the bounds of each bucket.
 *
 */
template <Direction D>
void BucketGraph::define_buckets() {
    int                 num_intervals = MAIN_RESOURCES;
    std::vector<double> total_ranges(num_intervals);
    std::vector<double> base_intervals(num_intervals);

    // Ensure the base_intervals storage is resized for all nodes
    if constexpr (D == Direction::Forward) {
        fw_base_intervals.resize(num_intervals);
    } else {
        bw_base_intervals.resize(num_intervals);
    }

    // Determine the base interval and other relevant values for each resource
    for (int r = 0; r < num_intervals; ++r) {
        total_ranges[r]   = R_max[r] - R_min[r];
        base_intervals[r] = total_ranges[r] / intervals[r].interval;
    }

    if constexpr (D == Direction::Forward) {
        fw_base_intervals = base_intervals;
    } else {
        bw_base_intervals = base_intervals;
    }

    auto &buckets             = assign_buckets<D>(fw_buckets, bw_buckets);
    auto &num_buckets         = assign_buckets<D>(num_buckets_fw, num_buckets_bw);
    auto &num_buckets_index   = assign_buckets<D>(num_buckets_index_fw, num_buckets_index_bw);
    auto &node_interval_trees = assign_buckets<D>(fw_node_interval_trees, bw_node_interval_trees);
    auto &buckets_size        = assign_buckets<D>(fw_buckets_size, bw_buckets_size);
    num_buckets.resize(nodes.size());
    num_buckets_index.resize(nodes.size());

    int cum_sum      = 0; // Tracks global bucket index
    int bucket_index = 0;

    node_interval_trees.assign(nodes.size(), SplayTree());

    // Loop through each node to define its specific buckets
    for (const auto &VRPNode : nodes) {
        std::vector<double> node_base_interval(num_intervals);
        for (int r = 0; r < num_intervals; ++r) {
            node_base_interval[r] = (VRPNode.ub[r] - VRPNode.lb[r]) / intervals[r].interval;
        }
        SplayTree node_tree;

        // Multiple buckets case
        int                 n_buckets = 0;
        std::vector<int>    current_pos(num_intervals, 0);
        std::vector<double> interval_start(num_intervals), interval_end(num_intervals);

        // Calculate the start and end for each interval dimension
        for (int r = 0; r < num_intervals; ++r) {
            for (auto j = 0; j < intervals[r].interval; ++j) {

                if constexpr (D == Direction::Forward) {
                    interval_start[r] = VRPNode.lb[r] + current_pos[r] * node_base_interval[r];

                    if (j == intervals[r].interval - 1) {
                        interval_end[r] = VRPNode.ub[r];
                    } else {
                        interval_end[r] = VRPNode.lb[r] + (current_pos[r] + 1) * node_base_interval[r];
                    }
                } else {
                    if (j == intervals[r].interval - 1) {
                        interval_start[r] = VRPNode.lb[r];
                    } else {
                        interval_start[r] = VRPNode.ub[r] - (current_pos[r] + 1) * node_base_interval[r];
                    }
                    // interval_start[r] = VRPNode.ub[r] - (current_pos[r] + 1) * node_base_interval[r];
                    interval_end[r] = VRPNode.ub[r] - current_pos[r] * node_base_interval[r];
                }

                // Apply rounding to two decimal places before using values
                interval_start[r] = roundToTwoDecimalPlaces(interval_start[r]);
                interval_end[r]   = roundToTwoDecimalPlaces(interval_end[r]);

                if constexpr (D == Direction::Backward) {
                    interval_start[r] = std::max(interval_start[r], R_min[r]);
                } else {
                    interval_end[r] = std::min(interval_end[r], R_max[r]);
                }

                buckets.push_back(Bucket(VRPNode.id, interval_start, interval_end));

                node_tree.insert(interval_start, interval_end, bucket_index);

                bucket_index++;
                n_buckets++;
                cum_sum++;

                current_pos[r]++;
            }
        }

        // Update node-specific bucket data
        num_buckets[VRPNode.id]       = n_buckets;
        num_buckets_index[VRPNode.id] = cum_sum - n_buckets;

        node_interval_trees[VRPNode.id] = node_tree;
    }

    // Update global bucket sizes based on direction
    buckets_size = cum_sum;
}

/**
 * @brief Generates arcs in the bucket graph based on the specified direction.
 *
 * Generates arcs in the bucket graph based on the specified direction.
 *
 */
template <Direction D>
void BucketGraph::generate_arcs() {
    // Mutex to synchronize access to shared resources
    auto buckets_mutex = std::mutex();

    // Clear the appropriate bucket graph (forward or backward)
    if constexpr (D == Direction::Forward) {
        fw_bucket_graph.clear(); // Clear the forward bucket graph
    } else {
        bw_bucket_graph.clear(); // Clear the backward bucket graph
    }

    // Assign the forward or backward fixed buckets and other bucket-related structures
    auto &fixed_buckets     = assign_buckets<D>(fw_fixed_buckets, bw_fixed_buckets);
    auto &buckets           = assign_buckets<D>(fw_buckets, bw_buckets);
    auto &num_buckets       = assign_buckets<D>(num_buckets_fw, num_buckets_bw);
    auto &num_buckets_index = assign_buckets<D>(num_buckets_index_fw, num_buckets_index_bw);

    // Clear all buckets in parallel, removing any existing arcs
    std::for_each(buckets.begin(), buckets.end(), [&](auto &bucket) {
        bucket.clear();                             // Clear bucket data
        bucket.clear_arcs(D == Direction::Forward); // Clear arcs in the bucket
    });

    auto add_arcs_for_node = [&](const VRPNode &node, int from_bucket, std::vector<double> &res_inc,
                                 std::vector<std::pair<int, int>> &local_arcs) {
        // Retrieve the arcs for the node in the given direction (Forward/Backward)
        auto arcs = node.get_arcs<D>();
        // Compute base intervals for each resource dimension based on R_max and R_min
        std::vector<double> node_intervals(intervals.size());
        for (int r = 0; r < intervals.size(); ++r) {
            node_intervals[r] = (node.ub[r] - node.lb[r]) / intervals[r].interval;
        }

        // Iterate over all arcs of the node
        for (const auto &arc : arcs) {
            const auto &next_node = nodes[arc.to]; // Get the destination node of the arc

            std::vector<double> next_node_intervals(intervals.size());
            for (int r = 0; r < intervals.size(); ++r) {
                next_node_intervals[r] = (next_node.ub[r] - next_node.lb[r]) / intervals[r].interval;
            }
            // Skip self-loops (no arc from a node to itself)
            if (node.id == next_node.id) continue;

            // Calculate travel cost and cost increment based on node's properties
            const auto travel_cost = getcij(node.id, next_node.id);
            double     cost_inc    = travel_cost - next_node.cost;
            res_inc[0]             = travel_cost + node.duration; // Update resource increment based on node duration
            // Iterate over all possible destination buckets for the next node

            for (int j = 0; j < num_buckets[next_node.id]; ++j) {
                int to_bucket = j + num_buckets_index[next_node.id];
                if (from_bucket == to_bucket) continue; // Skip arcs that loop back to the same bucket

                if (fixed_buckets[from_bucket][to_bucket] == 1) continue; // Skip fixed arcs

                bool valid_arc = true;
                for (int r = 0; r < res_inc.size(); ++r) {
                    // Forward direction: Check that resource increment doesn't exceed upper bounds
                    if constexpr (D == Direction::Forward) {
                        if (buckets[from_bucket].lb[r] + res_inc[r] > next_node.ub[r]) {
                            valid_arc = false;
                            break;
                        }
                    }
                    // Backward direction: Check that resource decrement doesn't drop below lower bounds
                    else if constexpr (D == Direction::Backward) {
                        if (buckets[from_bucket].ub[r] - res_inc[r] < next_node.lb[r]) {
                            valid_arc = false;
                            break;
                        }
                    }
                }
                if (!valid_arc) continue; // Skip invalid arcs

                // Further refine arc validity based on the base intervals and node bounds
                if constexpr (D == Direction::Forward) {
                    for (int r = 0; r < res_inc.size(); ++r) {
                        double max_calc = std::max(buckets[from_bucket].lb[r] + res_inc[r], next_node.lb[r]);
                        if (max_calc < buckets[to_bucket].lb[r] ||
                            max_calc >= buckets[to_bucket].lb[r] + next_node_intervals[r] + numericutils::eps) {
                            valid_arc = false;
                            break;
                        }
                    }
                } else if constexpr (D == Direction::Backward) {
                    for (int r = 0; r < res_inc.size(); ++r) {
                        double min_calc = std::min(buckets[from_bucket].ub[r] - res_inc[r], next_node.ub[r]);
                        if (min_calc > buckets[to_bucket].ub[r] ||
                            min_calc <= buckets[to_bucket].ub[r] - next_node_intervals[r] - numericutils::eps) {
                            valid_arc = false;
                            break;
                        }
                    }
                }
                if (!valid_arc) continue; // Skip invalid arcs

                // Store the arc data locally before committing it to the global structure
                local_arcs.emplace_back(from_bucket, to_bucket);
                double              local_cost_inc = cost_inc;
                std::vector<double> local_res_inc  = res_inc;

                // Add the arc to the global structure and the bucket
                {
                    std::lock_guard<std::mutex> lock(buckets_mutex); // Lock the mutex to ensure thread safety
                    add_arc<D>(from_bucket, to_bucket, local_res_inc, local_cost_inc); // Add the arc globally
                    buckets[from_bucket].add_bucket_arc(from_bucket, to_bucket, local_res_inc, local_cost_inc,
                                                        D == Direction::Forward,
                                                        false); // Add the arc to the bucket
                }
            }
        }
    };

    unsigned int total_threads = std::thread::hardware_concurrency();
    // Divide by 2 to use only half of the available threads
    unsigned int half_threads = total_threads / 2;

    const int                JOBS = half_threads;
    exec::static_thread_pool pool(JOBS);
    auto                     sched = pool.get_scheduler();

    // Iterate over all nodes in parallel, generating arcs for each
    std::vector<int> tasks; // Tasks will store node ids
    for (int node_id = 0; node_id < nodes.size(); ++node_id) {
        tasks.push_back(node_id); // Store the node id to be processed
    }

    // Define chunk size to reduce parallelization overhead
    const int chunk_size = 10; // Adjust based on your performance needs

    // Parallel execution in chunks
    auto bulk_sender = stdexec::bulk(
        stdexec::just(), (tasks.size() + chunk_size - 1) / chunk_size,
        [this, &tasks, &num_buckets, &num_buckets_index, &add_arcs_for_node, chunk_size](std::size_t chunk_idx) {
            size_t start_idx = chunk_idx * chunk_size;
            size_t end_idx   = std::min(start_idx + chunk_size, tasks.size());

            // Process a chunk of tasks (i.e., a group of nodes)
            for (size_t task_idx = start_idx; task_idx < end_idx; ++task_idx) {
                int            node_id = tasks[task_idx]; // Get the node id
                const VRPNode &VRPNode = nodes[node_id];

                std::vector<double> res_inc = {static_cast<double>(VRPNode.duration)}; // Resource increment vector
                std::vector<std::pair<int, int>> local_arcs;                           // Local storage for arcs

                // Generate arcs for all buckets associated with the current node
                for (int i = 0; i < num_buckets[VRPNode.id]; ++i) {
                    int from_bucket = i + num_buckets_index[VRPNode.id]; // Determine the source bucket
                    add_arcs_for_node(VRPNode, from_bucket, res_inc,
                                      local_arcs); // Add arcs for this node and bucket
                }
            }
        });

    // Submit work to the thread pool
    auto work = stdexec::starts_on(sched, bulk_sender);
    stdexec::sync_wait(std::move(work));
}

/**
 * @brief Retrieves the best label from the bucket graph based on the given topological order, c_bar values, and
 * strongly connected components.
 *
 * This function iterates through the given topological order and for each component, it retrieves the labels
 * from the corresponding buckets in the bucket graph. It then compares the cost of each label and keeps track
 * of the label with the lowest cost. The best label, along with its associated bucket, is returned.
 *
 */
template <Direction D>
Label *BucketGraph::get_best_label(const std::vector<int> &topological_order, const std::vector<double> &c_bar,
                                   const std::vector<std::vector<int>> &sccs) {
    double best_cost  = std::numeric_limits<double>::infinity();
    Label *best_label = nullptr; // Ensure this is initialized
    auto  &buckets    = assign_buckets<D>(fw_buckets, bw_buckets);

    for (const int component_index : topological_order) {
        const auto &component_buckets = sccs[component_index];

        for (const int bucket : component_buckets) {
            const auto &label = buckets[bucket].get_best_label();
            if (!label) continue;

            if (label->cost < best_cost) {
                best_cost  = label->cost;
                best_label = label;
            }
        }
    }

    return best_label;
}

/**
 * Concatenates the label L with the bucket b and updates the best label pbest.
 *
 */
template <Stage S>
void BucketGraph::ConcatenateLabel(const Label *L, int &b, Label *&pbest, std::vector<uint64_t> &Bvisited) {
    // Use a vector for iterative processing as a stack
    std::vector<int> bucket_stack;
    bucket_stack.reserve(50);
    bucket_stack.push_back(b);

    const auto &L_node_id   = L->node_id;
    const auto &L_resources = L->resources;
    const auto &L_last_node = nodes[L_node_id];

    while (!bucket_stack.empty()) {
        // Pop the next bucket from the stack (vector back)
        int current_bucket = bucket_stack.back();
        bucket_stack.pop_back();

        // Mark the bucket as visited
        const size_t segment      = current_bucket >> 6; // Equivalent to current_bucket / 64
        const size_t bit_position = current_bucket & 63; // Equivalent to current_bucket % 64

        Bvisited[segment] |= (1ULL << bit_position);

        const auto &bucketLprimenode = bw_buckets[current_bucket].node_id;
        double      cost             = getcij(L_node_id, bucketLprimenode);

#if defined(RCC) || defined(EXACT_RCC)
        if constexpr (S == Stage::Four) { cost -= arc_duals.getDual(L_node_id, bucketLprimenode); }
#endif

        // Branching duals
        if (branching_duals->size() > 0) { cost -= branching_duals->getDual(L_node_id, bucketLprimenode); }

#ifdef SRC
        decltype(cut_storage)            cutter   = nullptr;
        decltype(cut_storage->SRCDuals) *SRCDuals = nullptr;

        if constexpr (S > Stage::Three) {
            cutter   = cut_storage;       // Initialize cutter
            SRCDuals = &cutter->SRCDuals; // Initialize SRCDuals
        }
#endif

        double L_cost_plus_cost = L->cost + cost;

        // Early exit based on cost comparison
        if ((S != Stage::Enumerate && L_cost_plus_cost + bw_c_bar[current_bucket] >= pbest->cost) ||
            (S == Stage::Enumerate && L_cost_plus_cost + bw_c_bar[current_bucket] >= gap)) {
            continue;
        }

        // Get the labels for the current bucket
        auto       &bucket = bw_buckets[current_bucket];
        const auto &labels = bucket.get_labels();

        for (const auto &L_bw : labels) {
            if (L_bw->node_id == L_node_id || !check_feasibility(L, L_bw)) continue;
            double candidate_cost = L_cost_plus_cost + L_bw->cost;

#ifdef SRC
            if constexpr (S > Stage::Three) {
                for (auto it = cutter->begin(); it < cutter->end(); ++it) {
                    if ((*SRCDuals)[it->id] == 0) continue;
                    auto den = it->p.den;
                    auto sum = (L->SRCmap[it->id] + L_bw->SRCmap[it->id]);
                    if (sum >= den) { candidate_cost -= (*SRCDuals)[it->id]; }
                }
            }
#endif

            // Check for visited overlap and skip if true

            if constexpr (S >= Stage::Three) {
                bool visited_overlap = false;
                for (size_t i = 0; i < L->visited_bitmap.size(); ++i) {
                    if (L->visited_bitmap[i] & L_bw->visited_bitmap[i]) {
                        visited_overlap = true;
                        break;
                    }
                }
                if (visited_overlap) continue;
            }

            // Early exit based on candidate cost
            if ((S != Stage::Enumerate && candidate_cost >= pbest->cost) ||
                (S == Stage::Enumerate && candidate_cost >= gap)) {
                continue;
            }

            // Compute and store the new label
            pbest = compute_label(L, L_bw);
            merged_labels.push_back(pbest);
        }

        // Add unvisited neighboring buckets to the stack (vector back)
        for (int b_prime : Phi_bw[current_bucket]) {
            const size_t segment_prime      = b_prime >> 6;
            const size_t bit_position_prime = b_prime & 63;
            if (!(Bvisited[segment_prime] & (1ULL << bit_position_prime))) {
                bucket_stack.push_back(b_prime); // Add to the vector stack
            }
        }
    }
}

/**
 * @brief Handles the computation of Strongly Connected Components (SCCs) for the BucketGraph.
 *
 * This function processes the bucket graph to identify SCCs using Tarjan's algorithm. It extends the bucket
 * graph with arcs defined by the Phi sets, computes the SCCs, and orders them topologically. It also sorts the
 * buckets within each SCC based on their lower or upper bounds, depending on the direction. Finally, it splits
 * the arcs for each SCC and removes duplicates.
 *
 */
template <Direction D>
void BucketGraph::SCC_handler() {
    auto &Phi          = assign_buckets<D>(Phi_fw, Phi_bw);
    auto &buckets      = assign_buckets<D>(fw_buckets, bw_buckets);
    auto &bucket_graph = assign_buckets<D>(fw_bucket_graph, bw_bucket_graph);
    ankerl::unordered_dense::map<int, std::vector<int>> extended_bucket_graph = bucket_graph;

    // Extend the bucket graph with arcs defined by the Phi sets
    for (auto i = 0; i < extended_bucket_graph.size(); ++i) {
        auto phi_set = Phi[i];
        if (phi_set.empty()) continue;
        for (auto &phi_bucket : phi_set) { extended_bucket_graph[phi_bucket].push_back(i); }
    }

    SCC scc_finder;
    scc_finder.convertFromUnorderedMap(extended_bucket_graph); // print extended bucket graph

    auto sccs              = scc_finder.tarjanSCC();
    auto topological_order = scc_finder.topologicalOrderOfSCCs(sccs);

#ifdef VERBOSE
    // print SCCs and buckets in it
    constexpr auto blue  = "\033[34m";
    constexpr auto reset = "\033[0m";
    if constexpr (D == Direction::Forward) {
        fmt::print("FW SCCs: \n");
    } else {
        fmt::print("BW SCCs: \n");
    }
    for (auto scc : topological_order) {
        if constexpr (D == Direction::Forward) {
            fmt::print("{}({}) -> {}", blue, scc, reset);
        } else {
            fmt::print("{}({}) -> {}", blue, scc, reset);
        }
        for (auto &bucket : sccs[scc]) { fmt::print("{} ", bucket); }
    }
    fmt::print("\n");
#endif

    std::vector<std::vector<int>> ordered_sccs;
    ordered_sccs.reserve(sccs.size()); // Reserve space for all SCCs
    for (int i : topological_order) { ordered_sccs.push_back(sccs[i]); }

    auto sorted_sccs = sccs;
    for (auto &scc : sorted_sccs) {
        if constexpr (D == Direction::Forward) {
            std::sort(scc.begin(), scc.end(), [&buckets](int a, int b) { return buckets[a].lb[0] < buckets[b].lb[0]; });
        } else {
            std::sort(scc.begin(), scc.end(), [&buckets](int a, int b) { return buckets[a].ub[0] > buckets[b].ub[0]; });
        }
    }

    // iterate over nodes
    for (auto &node : nodes) {
        if constexpr (D == Direction::Forward) {
            node.fw_arcs_scc.resize(sccs.size());
        } else {
            node.bw_arcs_scc.resize(sccs.size());
        }
    }

    // Split arcs for each SCC
    auto scc_ctr = 0;
    for (const auto &scc : sccs) {
        // Iterate over each bucket in the SCC
        for (int bucket : scc) {
            int      from_node_id = buckets[bucket].node_id; // Get the source node ID
            VRPNode &node         = nodes[from_node_id];     // Access the corresponding node
            //  Define filtered arcs depending on the direction
            if constexpr (D == Direction::Forward) {
                std::vector<Arc> &filtered_fw_arcs = nodes[from_node_id].fw_arcs_scc[scc_ctr]; // For forward direction

                // Iterate over the arcs from the current bucket
                const auto &bucket_arcs = buckets[bucket].template get_bucket_arcs<D>();
                for (const auto &arc : bucket_arcs) {
                    int to_node_id = buckets[arc.to_bucket].node_id; // Get the destination node ID

                    // Search for the arc from `from_node_id` to `to_node_id` in the node's arcs
                    auto it = std::find_if(node.fw_arcs.begin(), node.fw_arcs.end(),
                                           [&to_node_id](const Arc &a) { return a.to == to_node_id; });

                    // If both nodes are within the current SCC, retain the arc
                    if (it != node.fw_arcs.end()) {
                        // Add the arc to the filtered arcs
                        filtered_fw_arcs.push_back(*it); // Forward arcs
                    }
                }
            } else {
                std::vector<Arc> &filtered_bw_arcs = nodes[from_node_id].bw_arcs_scc[scc_ctr]; // For forward direction

                // Iterate over the arcs from the current bucket
                const auto &bucket_arcs = buckets[bucket].template get_bucket_arcs<D>();
                for (const auto &arc : bucket_arcs) {
                    int to_node_id = buckets[arc.to_bucket].node_id; // Get the destination node ID

                    // Search for the arc from `from_node_id` to `to_node_id` in the node's arcs
                    auto it = std::find_if(node.bw_arcs.begin(), node.bw_arcs.end(),
                                           [&to_node_id](const Arc &a) { return a.to == to_node_id; });

                    // If both nodes are within the current SCC, retain the arc
                    if (it != node.bw_arcs.end()) {
                        // Add the arc to the filtered arcs
                        filtered_bw_arcs.push_back(*it); // Forward arcs
                    }
                }
            }
        }

        // Increment SCC counter
        ++scc_ctr;
    }

    for (auto &node : nodes) {
        if constexpr (D == Direction::Forward) {
            // Iterate over all SCCs for each node
            for (auto &fw_arcs_scc : node.fw_arcs_scc) {
                // Sort arcs by from_bucket and to_bucket
                std::sort(fw_arcs_scc.begin(), fw_arcs_scc.end(),
                          [](const Arc &a, const Arc &b) { return std::tie(a.from, a.to) < std::tie(b.from, b.to); });

                // Remove consecutive duplicates
                auto last = std::unique(fw_arcs_scc.begin(), fw_arcs_scc.end(),
                                        [](const Arc &a, const Arc &b) { return a.from == b.from && a.to == b.to; });

                // Erase the duplicates from the vector
                fw_arcs_scc.erase(last, fw_arcs_scc.end());
            }
        } else {
            // Iterate over all SCCs for each node
            for (auto &bw_arcs_scc : node.bw_arcs_scc) {
                // Sort arcs by from_bucket and to_bucket
                std::sort(bw_arcs_scc.begin(), bw_arcs_scc.end(),
                          [](const Arc &a, const Arc &b) { return std::tie(a.from, a.to) < std::tie(b.from, b.to); });

                // Remove consecutive duplicates
                auto last = std::unique(bw_arcs_scc.begin(), bw_arcs_scc.end(),
                                        [](const Arc &a, const Arc &b) { return a.from == b.from && a.to == b.to; });

                // Erase the duplicates from the vector
                bw_arcs_scc.erase(last, bw_arcs_scc.end());
            }
        }
    }

    UnionFind uf(ordered_sccs);
    if constexpr (D == Direction::Forward) {
        fw_ordered_sccs      = ordered_sccs;
        fw_topological_order = topological_order;
        fw_sccs              = sccs;
        fw_sccs_sorted       = sorted_sccs;
        fw_union_find        = uf;
    } else {
        bw_ordered_sccs      = ordered_sccs;
        bw_topological_order = topological_order;
        bw_sccs              = sccs;
        bw_sccs_sorted       = sorted_sccs;
        bw_union_find        = uf;
    }
}

/**
 * @brief Get the opposite bucket number for a given bucket index.
 *
 * This function retrieves the opposite bucket number based on the current bucket index
 * and the specified direction. It determines the node and bounds of the current bucket,
 * then calculates the opposite bucket index using the appropriate direction.
 *
 */
template <Direction D>
int BucketGraph::get_opposite_bucket_number(int current_bucket_index, std::vector<double> &inc) {

    // TODO: adjust to multi-resource case
    auto &current_bucket =
        (D == Direction::Forward) ? fw_buckets[current_bucket_index] : bw_buckets[current_bucket_index];
    int  &node    = current_bucket.node_id;
    auto &theNode = nodes[node];

    // Find the opposite bucket using the appropriate direction
    int                 opposite_bucket_index = -1;
    std::vector<double> reference_point(MAIN_RESOURCES);
    for (int r = 0; r < MAIN_RESOURCES; ++r) {
        if constexpr (D == Direction::Forward) {
            reference_point[r] = std::max(inc[r], theNode.lb[r]);
        } else {
            reference_point[r] = std::min(inc[r], theNode.ub[r]);
        }
    }
    if constexpr (D == Direction::Forward) {
        opposite_bucket_index = get_bucket_number<Direction::Backward>(node, reference_point);
    } else {
        opposite_bucket_index = get_bucket_number<Direction::Forward>(node, reference_point);
    }

    return opposite_bucket_index;
}

/**
 * @brief Fixes the bucket arcs for the specified stage.
 *
 * This function performs the bucket arc fixing for the given stage. It initializes
 * necessary variables and runs labeling algorithms to compute forward and backward
 * reduced costs. Based on the computed gap, it performs arc elimination in both
 * forward and backward directions and generates the necessary arcs.
 *
 */
template <Stage S>
void BucketGraph::bucket_fixing() {
    // Stage 4 bucket arc fixing
    if (!fixed) {
        fmt::print("\033[34m_STARTING BUCKET FIXING PROCEDURE \033[0m");
        fmt::print("\n");
        fixed = true;
        common_initialization();

        std::vector<double> forward_cbar(fw_buckets.size(), std::numeric_limits<double>::infinity());
        std::vector<double> backward_cbar(bw_buckets.size(), std::numeric_limits<double>::infinity());

        run_labeling_algorithms<Stage::Four, Full::Full>(forward_cbar, backward_cbar);

        gap = std::ceil(incumbent - (relaxation + std::min(0.0, min_red_cost)));

        // check if gap is -inf and early exit, due to IPM
        if (gap < 0) {
            fmt::print("\033[34m_BUCKET FIXING PROCEDURE CAN'T BE EXECUTED DUE TO GAP\033[0m");
            fmt::print("\n");
            // print the gap
            fmt::print("gap: {}\n", gap);
            return;
        }
        fw_c_bar = forward_cbar;
        bw_c_bar = backward_cbar;

        print_info("performing bucket arc elimination with theta = {}\n", gap);

        PARALLEL_SECTIONS(
            work, bi_sched,
            SECTION {
                // Section 1: Forward direction
                BucketArcElimination<Direction::Forward>(gap);
                ObtainJumpBucketArcs<Direction::Forward>();
            },
            SECTION {
                // Section 2: Backward direction
                BucketArcElimination<Direction::Backward>(gap);
                ObtainJumpBucketArcs<Direction::Backward>();
            });

        generate_arcs();
        fmt::print("\033[34m_BUCKET FIXING PROCEDURE FINISHED\033[0m");
        fmt::print("\n");
    }
}

/**
 * @brief Applies heuristic fixing to the current solution.
 *
 * This function modifies the current solution based on the heuristic
 * fixing strategy using the provided vector of values.
 *
 */
template <Stage S>
void BucketGraph::heuristic_fixing() {
    // Stage 3 fixing heuristic
    reset_pool();
    reset_fixed();
    common_initialization();

    std::vector<double> forward_cbar(fw_buckets.size(), std::numeric_limits<double>::infinity());
    std::vector<double> backward_cbar(bw_buckets.size(), std::numeric_limits<double>::infinity());

    run_labeling_algorithms<Stage::Two, Full::Partial>(forward_cbar, backward_cbar);

    std::vector<std::vector<Label *>> fw_labels_map(nodes.size());
    std::vector<std::vector<Label *>> bw_labels_map(nodes.size());

    auto group_labels = [&](auto &buckets, auto &labels_map) {
        for (auto &bucket : buckets) {
            for (auto &label : bucket.get_labels()) {
                labels_map[label->node_id].push_back(label); // Directly index using node_id
            }
        }
    };

    // Create tasks for forward and backward labels grouping
    auto forward_task = stdexec::schedule(bi_sched) | stdexec::then([&]() { group_labels(fw_buckets, fw_labels_map); });
    auto backward_task =
        stdexec::schedule(bi_sched) | stdexec::then([&]() { group_labels(bw_buckets, bw_labels_map); });

    // Execute the tasks in parallel
    auto work = stdexec::when_all(std::move(forward_task), std::move(backward_task));

    stdexec::sync_wait(std::move(work));

    auto num_fixes = 0;
    //  Function to find the minimum cost label in a vector of labels
    auto find_min_cost_label = [](const std::vector<Label *> &labels) -> const Label * {
        return *std::min_element(labels.begin(), labels.end(),
                                 [](const Label *a, const Label *b) { return a->cost < b->cost; });
    };
    for (const auto &node_I : nodes) {
        const auto &fw_labels = fw_labels_map[node_I.id];
        if (fw_labels.empty()) continue; // Skip if no labels for this node_id

        for (const auto &node_J : nodes) {
            if (node_I.id == node_J.id) continue; // Compare based on id (or other key field)
            const auto &bw_labels = bw_labels_map[node_J.id];
            if (bw_labels.empty()) continue; // Skip if no labels for this node_id

            const Label *min_fw_label = find_min_cost_label(fw_labels);
            const Label *min_bw_label = find_min_cost_label(bw_labels);

            if (!min_fw_label || !min_bw_label) continue;

            const VRPNode &L_last_node = nodes[min_fw_label->node_id];
            auto           cost        = getcij(min_fw_label->node_id, min_bw_label->node_id);

            // Check for infeasibility
            if (min_fw_label->resources[TIME_INDEX] + cost + L_last_node.consumption[TIME_INDEX] >
                min_bw_label->resources[TIME_INDEX]) {
                continue;
            }

            if (min_fw_label->cost + cost + min_bw_label->cost > gap) {
                fixed_arcs[node_I.id][node_J.id] = 1; // Index with node ids
                num_fixes++;
            }
        }
    }
}
