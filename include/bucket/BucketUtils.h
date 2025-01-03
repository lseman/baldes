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

#include "MST.h"

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

template <Direction D>
inline int BucketGraph::get_bucket_number(int node, std::vector<double> &resource_values_vec) noexcept {

    for (int r = 0; r < options.main_resources.size(); ++r) {
        if constexpr (D == Direction::Forward) {
            resource_values_vec[r] = (resource_values_vec[r]); // + numericutils::eps;
        } else {
            resource_values_vec[r] = (resource_values_vec[r]); // - numericutils::eps;
        }
    }
    auto val = -1;
    if constexpr (D == Direction::Forward) {
        val = fw_node_interval_trees[node].query(resource_values_vec);
    } else if constexpr (D == Direction::Backward) {
        val = bw_node_interval_trees[node].query(resource_values_vec);
    }

    return val;
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
    int                 num_intervals = options.main_resources.size();
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
        // Check if there is only one interval and proceed with the single interval logic
        if (num_intervals == 1) {
            for (auto j = 0; j < intervals[0].interval; ++j) {
                // Perform the same interval logic for a single interval
                if constexpr (D == Direction::Forward) {
                    interval_start[0] = VRPNode.lb[0] + current_pos[0] * node_base_interval[0];

                    if (j == intervals[0].interval - 1) {
                        interval_end[0] = VRPNode.ub[0];
                    } else {
                        interval_end[0] = VRPNode.lb[0] + (current_pos[0] + 1) * node_base_interval[0];
                    }
                } else {
                    if (j == intervals[0].interval - 1) {
                        interval_start[0] = VRPNode.lb[0];
                    } else {
                        interval_start[0] = VRPNode.ub[0] - (current_pos[0] + 1) * node_base_interval[0];
                    }
                    interval_end[0] = VRPNode.ub[0] - current_pos[0] * node_base_interval[0];
                }

                interval_start[0] = roundToTwoDecimalPlaces(interval_start[0]);
                interval_end[0]   = roundToTwoDecimalPlaces(interval_end[0]);

                if constexpr (D == Direction::Backward) {
                    interval_start[0] = std::max(interval_start[0], VRPNode.lb[0]);
                } else {
                    interval_end[0] = std::min(interval_end[0], VRPNode.ub[0]);
                }
                buckets.push_back(Bucket(VRPNode.id, interval_start, interval_end));
                node_tree.insert(interval_start, interval_end, bucket_index);

                bucket_index++;
                n_buckets++;
                cum_sum++;
                current_pos[0]++;
            }
        } else {
            // Multiple intervals case, nested loop logic to generate all combinations
            std::vector<int> pos(num_intervals, 0);

            while (true) {
                for (int r = 0; r < num_intervals; ++r) {
                    if constexpr (D == Direction::Forward) {
                        interval_start[r] = VRPNode.lb[r] + pos[r] * node_base_interval[r];

                        if (pos[r] == intervals[r].interval - 1) {
                            interval_end[r] = VRPNode.ub[r];
                        } else {
                            interval_end[r] = VRPNode.lb[r] + (pos[r] + 1) * node_base_interval[r];
                        }
                    } else {
                        if (pos[r] == intervals[r].interval - 1) {
                            interval_start[r] = VRPNode.lb[r];
                        } else {
                            interval_start[r] = VRPNode.ub[r] - (pos[r] + 1) * node_base_interval[r];
                        }
                        interval_end[r] = VRPNode.ub[r] - pos[r] * node_base_interval[r];
                    }

                    interval_start[r] = roundToTwoDecimalPlaces(interval_start[r]);
                    interval_end[r]   = roundToTwoDecimalPlaces(interval_end[r]);

                    if constexpr (D == Direction::Backward) {
                        interval_start[r] = std::max(interval_start[r], R_min[r]);
                    } else {
                        interval_end[r] = std::min(interval_end[r], R_max[r]);
                    }
                }

                /*
                                // Print and store
                                fmt::print("Creating bucket with interval: [");
                                for (int i = 0; i < num_intervals; ++i) {
                                    fmt::print("({}, {})", interval_start[i], interval_end[i]);
                                    if (i < num_intervals - 1) fmt::print(", ");
                                }
                                fmt::print("]\n");
                */

                buckets.push_back(Bucket(VRPNode.id, interval_start, interval_end));
                node_tree.insert(interval_start, interval_end, bucket_index);

                bucket_index++;
                n_buckets++;
                cum_sum++;

                // Increment the positions for combinations
                int i = 0;
                while (i < num_intervals && ++pos[i] >= intervals[i].interval) {
                    pos[i] = 0;
                    i++;
                }
                if (i == num_intervals) break; // All combinations generated
            }
        }

        // Update node-specific bucket data
        num_buckets[VRPNode.id]         = n_buckets;
        num_buckets_index[VRPNode.id]   = cum_sum - n_buckets;
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
        std::vector<double> node_intervals(options.resources.size());
        for (int r = 0; r < options.resources.size(); ++r) {
            node_intervals[r] = (node.ub[r] - node.lb[r]) / intervals[r].interval;
        }

        // Iterate over all arcs of the node
        for (const auto &arc : arcs) {
            const auto &next_node = nodes[arc.to]; // Get the destination node of the arc

            std::vector<double> next_node_intervals(options.resources.size());
            for (int r = 0; r < options.resources.size(); ++r) {
                next_node_intervals[r] = (next_node.ub[r] - next_node.lb[r]) / intervals[r].interval;
            }
            // Skip self-loops (no arc from a node to itself)
            if (node.id == next_node.id) continue;

            // Calculate travel cost and cost increment based on node's properties
            const auto travel_cost = getcij(node.id, next_node.id);
            double     cost_inc    = travel_cost - next_node.cost;
            for (auto r = 0; r < options.resources.size(); r++) {
                res_inc[r] = node.consumption[r];
                if (options.resources[r] == "time") { res_inc[r] += travel_cost; }
            }

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
                    buckets[from_bucket].template add_bucket_arc<D>(from_bucket, to_bucket, local_res_inc,
                                                                    local_cost_inc, false); // Add the arc to the bucket
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

                std::vector<double>              res_inc(options.resources.size()); // Resource increment vector
                std::vector<std::pair<int, int>> local_arcs;                        // Local storage for arcs

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
            // print label->cost
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
template <Stage S, Symmetry SYM>
void BucketGraph::ConcatenateLabel(const Label *L, int &b, double &best_cost, std::vector<uint64_t> &Bvisited) {
    static thread_local std::vector<int> bucket_stack;
    bucket_stack.clear();
    bucket_stack.reserve(50);
    bucket_stack.push_back(b);

    auto &other_buckets = assign_symmetry<SYM>(fw_buckets, bw_buckets);
    auto &other_c_bar   = assign_symmetry<SYM>(fw_c_bar, bw_c_bar);

    // Cache frequently accessed values
    const int    L_node_id   = L->node_id;
    const auto  &L_resources = L->resources;
    const auto  &L_last_node = nodes[L_node_id];
    const double L_cost      = L->cost;
    const size_t bitmap_size = L->visited_bitmap.size();

    // Pre-compute constants for bit operations
    constexpr uint64_t one           = 1ULL;
    const bool         has_branching = !branching_duals->empty();

    // SRC mode setup
#if defined(SRC)
    decltype(cut_storage)            cutter   = nullptr;
    decltype(cut_storage->SRCDuals) *SRCDuals = nullptr;
    if constexpr (S > Stage::Three) {
        cutter   = cut_storage;
        SRCDuals = &cutter->SRCDuals;
    }
#endif

    while (!bucket_stack.empty()) {
        const int current_bucket = bucket_stack.back();
        bucket_stack.pop_back();

        // Optimize bit operations for visited tracking
        const size_t   segment  = current_bucket >> 6;
        const uint64_t bit_mask = one << (current_bucket & 63);
        Bvisited[segment] |= bit_mask;

        const int bucketLprimenode = other_buckets[current_bucket].node_id;
        double    travel_cost      = getcij(L_node_id, bucketLprimenode);

        // Apply arc duals if needed
#if defined(RCC) || defined(EXACT_RCC)
        if constexpr (S == Stage::Four) { travel_cost -= arc_duals.getDual(L_node_id, bucketLprimenode); }
#endif

        if (has_branching) { travel_cost -= branching_duals->getDual(L_node_id, bucketLprimenode); }

        const double path_cost = L_cost + travel_cost;
        const double bound     = other_c_bar[current_bucket];

        // Early bound check
        if ((S != Stage::Enumerate && path_cost + bound >= best_cost) ||
            (S == Stage::Enumerate && path_cost + bound >= gap)) {
            continue;
        }

        // Process labels in current bucket
        const auto &bucket = other_buckets[current_bucket];
        const auto &labels = bucket.get_labels();

        for (const Label *L_bw : labels) {
            // Early rejection tests
            if (L_bw->node_id == L_node_id || !check_feasibility(L, L_bw)) { continue; }

            // Visited nodes overlap check
            if constexpr (S >= Stage::Three) {
                bool has_overlap = false;
                for (size_t i = 0; i < bitmap_size; ++i) {
                    if (L->visited_bitmap[i] & L_bw->visited_bitmap[i]) {
                        has_overlap = true;
                        break;
                    }
                }
                if (has_overlap) continue;
            }

            double total_cost = path_cost + L_bw->cost;

            // SRC cost adjustments
#if defined(SRC)
            if constexpr (S == Stage::Four) {
                for (auto it = cutter->begin(); it < cutter->end(); ++it) {
                    const auto &dual = (*SRCDuals)[it->id];
                    if (dual == 0) continue;

                    if (L->SRCmap[it->id] + L_bw->SRCmap[it->id] >= it->p.den) { total_cost -= dual; }
                }
            }
#endif

            // Cost-based filtering
            if ((S != Stage::Enumerate && total_cost >= best_cost) || (S == Stage::Enumerate && total_cost >= gap)) {
                continue;
            }

            // Create new merged label
            auto pbest = compute_label<S>(L, L_bw);
            best_cost = pbest->cost;
            // fmt::print("Best cost: {}\n", best_cost);
            merged_labels.push_back(pbest);
        }

        // Process neighbor buckets
        for (int b_prime : Phi_bw[current_bucket]) {
            const size_t   seg_prime  = b_prime >> 6;
            const uint64_t mask_prime = one << (b_prime & 63);
            if (!(Bvisited[seg_prime] & mask_prime)) { bucket_stack.push_back(b_prime); }
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
    std::vector<double> reference_point(options.main_resources.size());
    for (int r = 0; r < options.main_resources.size(); ++r) {
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
        just_fixed = true;
        return;
    }
    just_fixed = false;
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
            for (auto label : bucket.get_labels()) {
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

            bool violated = false;
            for (auto r = 0; r < options.resources.size(); ++r) {
                if (options.resources[r] == "time") {
                    if (min_fw_label->resources[TIME_INDEX] + cost + L_last_node.duration >
                        min_bw_label->resources[TIME_INDEX]) {
                        violated = true;
                        break;
                    }
                } else {
                    if (min_fw_label->resources[r] + L_last_node.consumption[r] > min_bw_label->resources[r]) {
                        violated = true;
                        break;
                    }
                }
            }

            if (violated) continue;

            if (min_fw_label->cost + cost + min_bw_label->cost > gap) {
                fixed_arcs[node_I.id][node_J.id] = 1; // Index with node ids
                num_fixes++;
            }
        }
    }
}

template <Symmetry SYM>
void BucketGraph::set_adjacency_list() {
    // Clear existing arcs for each node
    for (auto &node : nodes) {
        node.clear_arcs(); // Remove any existing arcs associated with the node
    }

    /*
        RawArcList heur_arcs;
        for (const auto &path : topHeurRoutes) {
            for (size_t i = 0; i < path.size() - 1; ++i) {
                int    from = path[i];
                int    to   = path[i + 1];
                RawArc arc(from, to);
                heur_arcs.add_arc(arc);
            }
        }
    */
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
            if (next_node.id == options.depot || node.id == next_node.id) continue; // Skip depot and same node

            if (options.pstep == true) {
                if (next_node.id == options.pstep_depot || next_node.id == options.pstep_end_depot)
                    continue; // Skip depot and end depot
            }

            auto   travel_cost = getcij(node.id, next_node.id); // Calculate travel cost
            double cost_inc    = travel_cost - next_node.cost;  // Adjust cost increment by subtracting next node's cost

            for (int r = 0; r < options.resources.size(); ++r) {
                if (options.resources[r] == "time") {
                    if constexpr (SYM == Symmetry::Asymmetric) {
                        res_inc[r] = travel_cost + node.duration; // Update resource increment based on node duration
                    } else {
                        res_inc[r] = node.duration / 2 + travel_cost +
                                     next_node.duration / 2; // Update resource increment based on node duration
                    }
                } else {
                    if constexpr (SYM == Symmetry::Asymmetric) {
                        res_inc[r] = node.consumption[r];
                    } else {
                        res_inc[r] = node.consumption[r] / 2 + next_node.consumption[r] / 2;
                    }
                }
            }

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

            // bool is_heuristic_arc = heur_arcs.has_arc(node.id, next_node.id);

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
            // fmt::print("Node ID: {}, To Bucket: {}, Cost Inc: {}\n", node.id, to_bucket, cost_inc);
        }

        // Add reverse arcs from neighboring nodes to the current node
        for (const auto &arc : best_arcs_rev) {
            auto [priority_value, to_bucket, res_inc_local, cost_inc] = arc;
            nodes[to_bucket].template add_arc<Direction::Backward>(to_bucket, node.id, res_inc_local, cost_inc,
                                                                   priority_value); // Add reverse arc
        }
    };

    // Step 4: Iterate over all nodes to set the adjacency list
    // print depot and end depot
    for (const auto &VRPNode : nodes) {
        // fmt::print("Node ID: {}\n", VRPNode.id);

        if (VRPNode.id == options.end_depot) {
            continue; // Skip the last node (depot)
        }

        std::vector<double> res_inc(intervals.size());   // Resource increment vector
        add_arcs_for_node(VRPNode, VRPNode.id, res_inc); // Add arcs for the current node
    }
}
