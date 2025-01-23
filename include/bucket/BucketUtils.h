/**
 * @file BucketUtils.h
 * @brief Header file for utilities related to the Bucket Graph in the Vehicle
 * Routing Problem (VRP).
 *
 * This file contains various template functions and algorithms for managing
 * buckets in the Bucket Graph. The Bucket Graph is a representation of the VRP
 * problem, where nodes are assigned to "buckets" based on resource intervals,
 * and arcs represent feasible transitions between buckets. The utilities
 * provided include adding arcs, defining buckets, generating arcs, extending
 * labels, and managing strongly connected components (SCCs).
 *
 * Key components:
 * - `add_arc`: Adds directed arcs between buckets based on the direction and
 * resource increments.
 * - `get_bucket_number`: Computes the bucket number for a given node and
 * resource values.
 * - `define_buckets`: Defines the structure and intervals for the buckets based
 * on resource bounds.
 * - `generate_arcs`: Generates arcs between buckets based on resource
 * constraints and feasibility.
 * - `SCC_handler`: Identifies and processes SCCs in the bucket graph.
 * - `Extend`: Extends a label with a given arc, checking for feasibility based
 * on resources.
 *
 * The utilities use template parameters for direction (Forward or Backward),
 * stages, and other configurations, allowing flexible handling of the bucket
 * graph in both directions.
 */

#pragma once

#include <cstring>

#include "Bucket.h"
#include "BucketJump.h"
#include "Definitions.h"
#include "MST.h"
#include "Trees.h"
#include "cuts/SRC.h"
#include "utils/NumericUtils.h"

template <typename T>
class ThreadLocalPool {
    static thread_local std::vector<std::vector<T>> pool;
    static thread_local size_t current_index;

   public:
    std::vector<T> &acquire() {
        if (current_index >= pool.size()) {
            pool.emplace_back();
        }
        return pool[current_index++];
    }
    void reset() { current_index = 0; }
};

// Define static members
template <typename T>
thread_local std::vector<std::vector<T>> ThreadLocalPool<T>::pool;

template <typename T>
thread_local size_t ThreadLocalPool<T>::current_index = 0;

// Explicit instantiation for the types we use
template class ThreadLocalPool<double>;
/**
 * @brief Represents a bucket in the Bucket Graph.
 * Adds an arc to the bucket graph.
 *
 */
template <Direction D>
void BucketGraph::add_arc(int from_bucket, int to_bucket,
                          const std::vector<double> &res_inc, double cost_inc) {
    if constexpr (D == Direction::Forward) {
        fw_arcs.emplace_back(from_bucket, to_bucket, res_inc, cost_inc);
        fw_bucket_graph[from_bucket].push_back(to_bucket);

    } else if constexpr (D == Direction::Backward) {
        bw_arcs.emplace_back(from_bucket, to_bucket, res_inc, cost_inc);
        bw_bucket_graph[from_bucket].push_back(to_bucket);
    }
}

template <Direction D>
inline int BucketGraph::get_bucket_number(
    int node, std::vector<double> &resource_values_vec) noexcept {
    for (int r = 0; r < options.main_resources.size(); ++r) {
        if constexpr (D == Direction::Forward) {
            resource_values_vec[r] =
                (resource_values_vec[r]);  // + numericutils::eps;
        } else {
            resource_values_vec[r] =
                (resource_values_vec[r]);  // - numericutils::eps;
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
 * This function determines the number of buckets based on the time intervals
 * and assigns buckets to the graph. It computes resource bounds for each vertex
 * and defines the bounds of each bucket.
 *
 */
template <Direction D>
void BucketGraph::define_buckets() {
    std::vector<Bucket> old_buckets;
    std::vector<bool> was_fixed;
    auto &buckets = assign_buckets<D>(fw_buckets, bw_buckets);
    auto old_size = buckets.size();

    // Store the range of fixed buckets for each node pair
    std::unordered_map<
        int, std::unordered_map<
                 int, std::pair<std::vector<double>, std::vector<double>>>>
        node_pair_fixed_ranges;

    if (old_size > 0) {
        old_buckets = buckets;
        was_fixed.resize(old_size * old_size, false);
        auto &old_bitmap =
            assign_buckets<D>(fw_fixed_buckets_bitmap, bw_fixed_buckets_bitmap);

        // First pass: mark which bucket pairs were fixed
        for (size_t i = 0; i < old_size; ++i) {
            for (size_t j = 0; j < old_size; ++j) {
                size_t old_bit_pos = i * old_size + j;
                if (old_bit_pos / 64 < old_bitmap.size()) {
                    was_fixed[old_bit_pos] =
                        (old_bitmap[old_bit_pos / 64] &
                         (1ULL << (old_bit_pos % 64))) != 0;
                }
            }
        }

        const int num_intervals = options.main_resources.size();
        // For each fixed bucket pair, track the range by node pair
        for (size_t i = 0; i < old_size; ++i) {
            for (size_t j = 0; j < old_size; ++j) {
                if (!was_fixed[i * old_size + j]) continue;

                const auto &bucket_from = old_buckets[i];
                const auto &bucket_to = old_buckets[j];

                int from_node = bucket_from.node_id;
                int to_node = bucket_to.node_id;

                // Initialize ranges for this node pair if not already done
                if (node_pair_fixed_ranges[from_node].find(to_node) ==
                    node_pair_fixed_ranges[from_node].end()) {
                    node_pair_fixed_ranges[from_node][to_node] = {
                        std::vector<double>(num_intervals,
                                            std::numeric_limits<double>::max()),
                        std::vector<double>(
                            num_intervals,
                            std::numeric_limits<double>::lowest())};
                }

                // Update the ranges for this node pair
                auto &[min_lbs, max_ubs] =
                    node_pair_fixed_ranges[from_node][to_node];
                for (int r = 0; r < num_intervals; ++r) {
                    min_lbs[r] = std::min(min_lbs[r], bucket_from.lb[r]);
                    max_ubs[r] = std::max(max_ubs[r], bucket_from.ub[r]);
                }
            }
        }

        // Debug output
        // for (const auto &[from_node, to_nodes] : node_pair_fixed_ranges) {
        //     for (const auto &[to_node, range] : to_nodes) {
        //         fmt::print("Arc ({}, {}) has fixed range: ", from_node,
        //                    to_node);
        //         for (int r = 0; r < num_intervals; ++r) {
        //             fmt::print("({}, {}), ", range.first[r],
        //             range.second[r]);
        //         }
        //         fmt::print("\n");
        //     }
        // }
    }

    // Clear existing buckets and bitmap
    buckets.clear();
    auto &fixed_buckets_bitmap =
        assign_buckets<D>(fw_fixed_buckets_bitmap, bw_fixed_buckets_bitmap);
    fixed_buckets_bitmap.clear();

    const int num_intervals = options.main_resources.size();
    std::vector<double> total_ranges(num_intervals);
    std::vector<double> base_intervals(num_intervals);

    // Get references to direction-specific containers
    auto &num_buckets = assign_buckets<D>(num_buckets_fw, num_buckets_bw);
    auto &num_buckets_index =
        assign_buckets<D>(num_buckets_index_fw, num_buckets_index_bw);
    auto &node_interval_trees =
        assign_buckets<D>(fw_node_interval_trees, bw_node_interval_trees);
    auto &buckets_size = assign_buckets<D>(fw_buckets_size, bw_buckets_size);
    auto &bucket_splits = assign_buckets<D>(fw_bucket_splits, bw_bucket_splits);

    // Pre-allocate containers
    const size_t num_nodes = nodes.size();
    num_buckets.resize(num_nodes);
    num_buckets_index.resize(num_nodes);
    node_interval_trees.assign(num_nodes, SplayTree());

    // Lambda for interval calculations
    auto calculate_interval = [](double lb, double ub, double base_interval,
                                 int pos, int max_interval,
                                 bool is_forward) -> std::pair<double, double> {
        double start, end;
        if (is_forward) {
            start = lb + pos * base_interval;
            end =
                (pos == max_interval - 1) ? ub : lb + (pos + 1) * base_interval;
        } else {
            start =
                (pos == max_interval - 1) ? lb : ub - (pos + 1) * base_interval;
            end = ub - pos * base_interval;
        }
        return {roundToTwoDecimalPlaces(start), roundToTwoDecimalPlaces(end)};
    };

    int cum_sum = 0;
    int bucket_index = 0;

    // Temporary vectors for interval calculations
    std::vector<double> interval_start(num_intervals);
    std::vector<double> interval_end(num_intervals);
    std::vector<int> pos(num_intervals, 0);

    // Process each node
    for (const auto &VRPNode : nodes) {
        std::vector<double> node_base_interval(num_intervals);

        // check if fw_bucket_splits with node_id exists
        if (bucket_splits.find(VRPNode.id) == bucket_splits.end()) {
            for (int r = 0; r < num_intervals; ++r) {
                node_base_interval[r] =
                    (VRPNode.ub[r] - VRPNode.lb[r]) / intervals[r].interval;
            }
            bucket_splits[VRPNode.id] = intervals[0].interval;
        } else {
            for (int r = 0; r < num_intervals; ++r) {
                node_base_interval[r] =
                    (VRPNode.ub[r] - VRPNode.lb[r]) / bucket_splits[VRPNode.id];
            }
        }

        SplayTree node_tree;
        int n_buckets = 0;

        if (num_intervals == 1) {
            // Single interval case
            for (int j = 0; j < bucket_splits[VRPNode.id]; ++j) {
                auto [start, end] = calculate_interval(
                    VRPNode.lb[0], VRPNode.ub[0], node_base_interval[0], j,
                    bucket_splits[VRPNode.id], D == Direction::Forward);

                interval_start[0] = start;
                interval_end[0] = end;

                if constexpr (D == Direction::Backward) {
                    interval_start[0] =
                        std::max(interval_start[0], VRPNode.lb[0]);
                } else {
                    interval_end[0] = std::min(interval_end[0], VRPNode.ub[0]);
                }

                buckets.push_back(
                    Bucket(VRPNode.id, interval_start, interval_end));
                node_tree.insert(interval_start, interval_end, bucket_index++);
                n_buckets++;
                cum_sum++;
            }
        } else {
            fmt::print("Multiple intervals\n");
            // Multiple intervals case
            std::fill(pos.begin(), pos.end(), 0);
            do {
                for (int r = 0; r < num_intervals; ++r) {
                    auto [start, end] = calculate_interval(
                        VRPNode.lb[r], VRPNode.ub[r], node_base_interval[r],
                        pos[r], intervals[r].interval, D == Direction::Forward);

                    interval_start[r] = start;
                    interval_end[r] = end;

                    if constexpr (D == Direction::Backward) {
                        interval_start[r] =
                            std::max(interval_start[r], R_min[r]);
                    } else {
                        interval_end[r] = std::min(interval_end[r], R_max[r]);
                    }
                }

                buckets.push_back(
                    Bucket(VRPNode.id, interval_start, interval_end));
                node_tree.insert(interval_start, interval_end, bucket_index++);
                n_buckets++;
                cum_sum++;

                // Generate next combination
                int i = 0;
                while (i < num_intervals && ++pos[i] >= intervals[i].interval) {
                    pos[i] = 0;
                    i++;
                }
                if (i == num_intervals) break;
            } while (true);
        }

        // Update node-specific data
        num_buckets[VRPNode.id] = n_buckets;
        num_buckets_index[VRPNode.id] = cum_sum - n_buckets;
        node_interval_trees[VRPNode.id] = node_tree;
    }

    if (!node_pair_fixed_ranges.empty()) {
        size_t new_size = buckets.size();
        size_t required_bitmap_words =
            std::max(size_t(1), ((new_size * new_size) + 63) / 64);

        fixed_buckets_bitmap.clear();
        fixed_buckets_bitmap.resize(required_bitmap_words, 0);

        for (size_t from = 0; from < new_size; ++from) {
            const auto &bucket_from = buckets[from];
            int from_node = bucket_from.node_id;

            // Skip if no outgoing fixed arcs from this node
            if (node_pair_fixed_ranges.find(from_node) ==
                node_pair_fixed_ranges.end())
                continue;

            for (size_t to = 0; to < new_size; ++to) {
                const auto &bucket_to = buckets[to];
                int to_node = bucket_to.node_id;

                // Skip if this arc wasn't fixed before
                if (node_pair_fixed_ranges[from_node].find(to_node) ==
                    node_pair_fixed_ranges[from_node].end())
                    continue;

                const auto &range = node_pair_fixed_ranges[from_node][to_node];

                // Check if both buckets are within their fixed ranges
                bool is_in_range = true;
                for (int r = 0; r < options.main_resources.size(); ++r) {
                    if (bucket_from.lb[r] < range.first[r] ||
                        bucket_from.ub[r] > range.second[r]) {
                        is_in_range = false;
                        break;
                    }
                }

                if (is_in_range) {
                    size_t bit_pos = from * new_size + to;
                    size_t word_idx = bit_pos / 64;
                    if (word_idx < fixed_buckets_bitmap.size()) {
                        fixed_buckets_bitmap[word_idx] |=
                            (1ULL << (bit_pos % 64));
                    }
                }
            }
        }
    }

    buckets_size = buckets.size();
}

/**
 * @brief Generates arcs in the bucket graph based on the specified direction.
 *
 * Generates arcs in the bucket graph based on the specified direction.
 *
 */
template <Direction D>
void BucketGraph::generate_arcs() {
    auto buckets_mutex = std::mutex();

    if constexpr (D == Direction::Forward) {
        fw_bucket_graph.clear();
    } else {
        bw_bucket_graph.clear();
    }

    auto &buckets = assign_buckets<D>(fw_buckets, bw_buckets);
    auto &num_buckets = assign_buckets<D>(num_buckets_fw, num_buckets_bw);
    auto &num_buckets_index =
        assign_buckets<D>(num_buckets_index_fw, num_buckets_index_bw);
    auto &bucket_splits = assign_buckets<D>(fw_bucket_splits, bw_bucket_splits);
    // Pre-compute intervals for all nodes
    std::vector<std::vector<double>> node_intervals(nodes.size());
    for (size_t node_id = 0; node_id < nodes.size(); ++node_id) {
        const auto &node = nodes[node_id];
        node_intervals[node_id].resize(options.resources.size());
        for (int r = 0; r < options.resources.size(); ++r) {
            node_intervals[node_id][r] =
                (node.ub[r] - node.lb[r]) / bucket_splits[node_id];
        }
    }

    // Clear buckets
    for (auto &bucket : buckets) {
        bucket.clear();
        bucket.clear_arcs(D == Direction::Forward);
    }

    auto add_arcs_for_node = [&](const VRPNode &node, int from_bucket,
                                 std::vector<double> &res_inc,
                                 std::vector<std::pair<int, int>> &local_arcs) {
        const auto arcs = node.get_arcs<D>();
        const auto &node_interval = node_intervals[node.id];

        for (const auto &arc : arcs) {
            // print node_id
            const auto &next_node = nodes[arc.to];
            if (node.id == next_node.id) continue;

            const auto &next_node_interval = node_intervals[arc.to];
            const auto travel_cost = getcij(node.id, next_node.id);
            const double cost_inc = travel_cost - next_node.cost;

            // Pre-calculate resource increments
            for (int r = 0; r < options.resources.size(); r++) {
                res_inc[r] = node.consumption[r];
                if (options.resources[r] == "time") {
                    res_inc[r] += travel_cost;
                }
            }

            for (int j = 0; j < num_buckets[next_node.id]; ++j) {
                const int to_bucket = j + num_buckets_index[next_node.id];
                if (from_bucket == to_bucket ||
                    is_bucket_fixed<D>(from_bucket, to_bucket)) {
                    continue;
                }

                bool valid_arc = true;

                // Resource bounds check
                if constexpr (D == Direction::Forward) {
                    for (int r = 0; r < res_inc.size() && valid_arc; ++r) {
                        if (buckets[from_bucket].lb[r] + res_inc[r] >
                            next_node.ub[r]) {
                            valid_arc = false;
                        } else {
                            double max_calc = std::max(
                                buckets[from_bucket].lb[r] + res_inc[r],
                                next_node.lb[r]);
                            if (max_calc < buckets[to_bucket].lb[r] ||
                                max_calc >= buckets[to_bucket].lb[r] +
                                                next_node_interval[r] +
                                                numericutils::eps) {
                                valid_arc = false;
                            }
                        }
                    }
                } else {
                    for (int r = 0; r < res_inc.size() && valid_arc; ++r) {
                        if (buckets[from_bucket].ub[r] - res_inc[r] <
                            next_node.lb[r]) {
                            valid_arc = false;
                        } else {
                            double min_calc = std::min(
                                buckets[from_bucket].ub[r] - res_inc[r],
                                next_node.ub[r]);
                            if (min_calc > buckets[to_bucket].ub[r] ||
                                min_calc <= buckets[to_bucket].ub[r] -
                                                next_node_interval[r] -
                                                numericutils::eps) {
                                valid_arc = false;
                            }
                        }
                    }
                }

                if (valid_arc) {
                    local_arcs.emplace_back(from_bucket, to_bucket);
                    std::lock_guard<std::mutex> lock(buckets_mutex);
                    add_arc<D>(from_bucket, to_bucket, res_inc, cost_inc);
                    buckets[from_bucket].template add_bucket_arc<D>(
                        from_bucket, to_bucket, res_inc, cost_inc, false);
                }
            }
        }
    };

    const unsigned int thread_count = std::thread::hardware_concurrency() / 2;
    exec::static_thread_pool pool(thread_count);
    auto sched = pool.get_scheduler();

    std::vector<int> tasks(nodes.size());
    std::iota(tasks.begin(), tasks.end(), 0);

    const int chunk_size = 10;
    auto bulk_sender = stdexec::bulk(
        stdexec::just(), (tasks.size() + chunk_size - 1) / chunk_size,
        [this, &tasks, &num_buckets, &num_buckets_index, &add_arcs_for_node,
         chunk_size](std::size_t chunk_idx) {
            const size_t start_idx = chunk_idx * chunk_size;
            const size_t end_idx =
                std::min(start_idx + chunk_size, tasks.size());

            std::vector<double> res_inc(options.resources.size());
            std::vector<std::pair<int, int>> local_arcs;
            local_arcs.reserve(chunk_size *
                               100);  // Estimate average arcs per chunk

            for (size_t task_idx = start_idx; task_idx < end_idx; ++task_idx) {
                const int node_id = tasks[task_idx];
                const auto &node = nodes[node_id];

                for (int i = 0; i < num_buckets[node.id]; ++i) {
                    const int from_bucket = i + num_buckets_index[node.id];
                    add_arcs_for_node(node, from_bucket, res_inc, local_arcs);
                }
            }
        });

    auto work = stdexec::starts_on(sched, bulk_sender);
    stdexec::sync_wait(std::move(work));
}

/**
 * @brief Retrieves the best label from the bucket graph based on the given
 * topological order, c_bar values, and strongly connected components.
 *
 * This function iterates through the given topological order and for each
 * component, it retrieves the labels from the corresponding buckets in the
 * bucket graph. It then compares the cost of each label and keeps track of the
 * label with the lowest cost. The best label, along with its associated bucket,
 * is returned.
 *
 */
template <Direction D>
Label *BucketGraph::get_best_label(const std::vector<int> &topological_order,
                                   const std::vector<double> &c_bar,
                                   const std::vector<std::vector<int>> &sccs) {
    double best_cost = std::numeric_limits<double>::infinity();
    Label *best_label = nullptr;  // Ensure this is initialized
    auto &buckets = assign_buckets<D>(fw_buckets, bw_buckets);

    for (const int component_index : topological_order) {
        const auto &component_buckets = sccs[component_index];

        for (const int bucket : component_buckets) {
            const auto &label = buckets[bucket].get_best_label();
            if (!label) continue;
            // print label->cost
            if (label->cost < best_cost) {
                best_cost = label->cost;
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
void BucketGraph::ConcatenateLabel(const Label *L, int &b, double &best_cost,
                                   std::vector<uint64_t> &Bvisited) {
    // Reuse thread-local bucket_stack to avoid repeated allocations
    static thread_local std::vector<int> bucket_stack;
    bucket_stack.clear();
    bucket_stack.reserve(50);  // Adjust based on expected size
    bucket_stack.push_back(b);

    auto &other_buckets = assign_symmetry<SYM>(fw_buckets, bw_buckets);
    auto &other_c_bar = assign_symmetry<SYM>(fw_c_bar, bw_c_bar);

    // Cache frequently accessed values
    const int L_node_id = L->node_id;
    const auto &L_resources = L->resources;
    const auto &L_last_node = nodes[L_node_id];
    const double L_cost = L->cost;
    const size_t bitmap_size = L->visited_bitmap.size();

    // Pre-compute constants for bit operations
    constexpr uint64_t one = 1ULL;
    const bool has_branching = !branching_duals->empty();

    // SRC mode setup
#if defined(SRC)
    decltype(cut_storage) cutter = nullptr;
    if constexpr (S > Stage::Three) {
        cutter = cut_storage;
    }
    const auto active_cuts = cutter->getActiveCuts();
#endif

    while (!bucket_stack.empty()) {
        const int current_bucket = bucket_stack.back();
        bucket_stack.pop_back();

        // Optimize bit operations for visited tracking
        const size_t segment = current_bucket >> 6;
        const uint64_t bit_mask = one << (current_bucket & 63);
        Bvisited[segment] |= bit_mask;

        const int bucketLprimenode = other_buckets[current_bucket].node_id;
        double travel_cost = getcij(L_node_id, bucketLprimenode);

        // Apply arc duals if needed
#if defined(RCC) || defined(EXACT_RCC)
        if constexpr (S == Stage::Four) {
            travel_cost -= arc_duals.getDual(L_node_id, bucketLprimenode);
        }
#endif

        if (has_branching) {
            travel_cost -=
                branching_duals->getDual(L_node_id, bucketLprimenode);
        }

        const double path_cost = L_cost + travel_cost;
        const double bound = other_c_bar[current_bucket];

        // Early bound check
        if ((S != Stage::Enumerate && path_cost + bound >= best_cost) ||
            (S == Stage::Enumerate && path_cost + bound >= gap)) {
            continue;
        }

        // Process labels in current bucket
        const auto &bucket = other_buckets[current_bucket];
        const auto &labels = bucket.get_labels();

// Parallelize the inner loop if possible
#pragma omp parallel for schedule(dynamic) reduction(min : best_cost)
        for (size_t i = 0; i < labels.size(); ++i) {
            const Label *L_bw = labels[i];

            // Early rejection tests
            if (L_bw->node_id == L_node_id || !check_feasibility(L, L_bw)) {
                continue;
            }

            // Visited nodes overlap check
            if constexpr (S >= Stage::Three) {
                bool has_overlap = false;
                for (size_t j = 0; j < bitmap_size; ++j) {
                    if (L->visited_bitmap[j] & L_bw->visited_bitmap[j]) {
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
                for (const auto &active_cut : active_cuts) {
                    const auto &cut = *active_cut.cut_ptr;
                    const size_t idx = active_cut.index;
                    const double dual = active_cut.dual_value;

                    if (L->SRCmap[idx] + L_bw->SRCmap[idx] >= cut.p.den) {
                        total_cost -= dual;
                    }
                }
            }
#endif

            // Cost-based filtering
            if ((S != Stage::Enumerate && total_cost >= best_cost) ||
                (S == Stage::Enumerate && total_cost >= gap)) {
                continue;
            }

            // Create new merged label
            auto pbest = compute_label<S>(L, L_bw);
#pragma omp critical
            {
                if (pbest->cost < best_cost) {
                    best_cost = pbest->cost;
                }
                merged_labels.push_back(pbest);
            }
        }

        // Process neighbor buckets
        for (int b_prime : Phi_bw[current_bucket]) {
            const size_t seg_prime = b_prime >> 6;
            const uint64_t mask_prime = one << (b_prime & 63);
            if (!(Bvisited[seg_prime] & mask_prime)) {
                bucket_stack.push_back(b_prime);
            }
        }
    }
}

/**
 * @brief Handles the computation of Strongly Connected Components (SCCs) for
 * the BucketGraph.
 *
 * This function processes the bucket graph to identify SCCs using Tarjan's
 * algorithm. It extends the bucket graph with arcs defined by the Phi sets,
 * computes the SCCs, and orders them topologically. It also sorts the buckets
 * within each SCC based on their lower or upper bounds, depending on the
 * direction. Finally, it splits the arcs for each SCC and removes duplicates.
 *
 */
template <Direction D>
void BucketGraph::SCC_handler() {
    auto &Phi = assign_buckets<D>(Phi_fw, Phi_bw);
    auto &buckets = assign_buckets<D>(fw_buckets, bw_buckets);
    auto &bucket_graph = assign_buckets<D>(fw_bucket_graph, bw_bucket_graph);
    ankerl::unordered_dense::map<int, std::vector<int>> extended_bucket_graph =
        bucket_graph;

    // Extend the bucket graph with arcs defined by the Phi sets
    for (auto i = 0; i < extended_bucket_graph.size(); ++i) {
        auto phi_set = Phi[i];
        if (phi_set.empty()) continue;
        for (auto &phi_bucket : phi_set) {
            extended_bucket_graph[phi_bucket].push_back(i);
        }
    }

    SCC scc_finder;
    scc_finder.convertFromUnorderedMap(
        extended_bucket_graph);  // print extended bucket graph

    auto sccs = scc_finder.tarjanSCC();
    auto topological_order = scc_finder.topologicalOrderOfSCCs(sccs);

#ifdef VERBOSE
    // print SCCs and buckets in it
    constexpr auto blue = "\033[34m";
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
        for (auto &bucket : sccs[scc]) {
            fmt::print("{} ", bucket);
        }
    }
    fmt::print("\n");
#endif

    std::vector<std::vector<int>> ordered_sccs;
    ordered_sccs.reserve(sccs.size());  // Reserve space for all SCCs
    for (int i : topological_order) {
        ordered_sccs.push_back(sccs[i]);
    }

    auto sorted_sccs = sccs;
    for (auto &scc : sorted_sccs) {
        if constexpr (D == Direction::Forward) {
            std::sort(scc.begin(), scc.end(), [&buckets](int a, int b) {
                return buckets[a].lb[0] < buckets[b].lb[0];
            });
        } else {
            std::sort(scc.begin(), scc.end(), [&buckets](int a, int b) {
                return buckets[a].ub[0] > buckets[b].ub[0];
            });
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
            const int from_node_id =
                buckets[bucket].node_id;          // Cache source node ID
            VRPNode &node = nodes[from_node_id];  // Cache node reference

            // Define filtered arcs depending on the direction
            auto &filtered_arcs = (D == Direction::Forward)
                                      ? node.fw_arcs_scc[scc_ctr]
                                      : node.bw_arcs_scc[scc_ctr];

            // Reserve space for filtered arcs to avoid reallocations
            filtered_arcs.reserve(
                buckets[bucket].template get_bucket_arcs<D>().size());

            // Create a map for faster arc lookups
            // std::unordered_map<int, Arc> arc_map;
            ankerl::unordered_dense::map<int, Arc> arc_map;
            if constexpr (D == Direction::Forward) {
                for (const auto &arc : node.fw_arcs) {
                    arc_map[arc.to] = arc;
                }
            } else {
                for (const auto &arc : node.bw_arcs) {
                    arc_map[arc.to] = arc;
                }
            }

            // Iterate over the arcs from the current bucket
            for (const auto &arc :
                 buckets[bucket].template get_bucket_arcs<D>()) {
                const int to_node_id =
                    buckets[arc.to_bucket]
                        .node_id;  // Cache destination node ID

                // Check if the arc exists in the map
                auto it = arc_map.find(to_node_id);
                if (it != arc_map.end()) {
                    // Add the arc to the filtered arcs
                    filtered_arcs.push_back(it->second);
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
                // Skip if the vector is empty
                if (fw_arcs_scc.empty()) continue;

                // Sort arcs by from_bucket and to_bucket
                pdqsort(fw_arcs_scc.begin(), fw_arcs_scc.end(),
                        [](const Arc &a, const Arc &b) {
                            return std::tie(a.from, a.to) <
                                   std::tie(b.from, b.to);
                        });

                // Remove consecutive duplicates
                auto last =
                    std::unique(fw_arcs_scc.begin(), fw_arcs_scc.end(),
                                [](const Arc &a, const Arc &b) {
                                    return a.from == b.from && a.to == b.to;
                                });

                // Erase the duplicates from the vector
                fw_arcs_scc.erase(last, fw_arcs_scc.end());
            }
        } else {
            // Iterate over all SCCs for each node
            for (auto &bw_arcs_scc : node.bw_arcs_scc) {
                // Skip if the vector is empty
                if (bw_arcs_scc.empty()) continue;

                // Sort arcs by from_bucket and to_bucket
                pdqsort(bw_arcs_scc.begin(), bw_arcs_scc.end(),
                        [](const Arc &a, const Arc &b) {
                            return std::tie(a.from, a.to) <
                                   std::tie(b.from, b.to);
                        });

                // Remove consecutive duplicates
                auto last =
                    std::unique(bw_arcs_scc.begin(), bw_arcs_scc.end(),
                                [](const Arc &a, const Arc &b) {
                                    return a.from == b.from && a.to == b.to;
                                });

                // Erase the duplicates from the vector
                bw_arcs_scc.erase(last, bw_arcs_scc.end());
            }
        }
    }

    UnionFind uf(ordered_sccs);
    if constexpr (D == Direction::Forward) {
        fw_ordered_sccs = ordered_sccs;
        fw_topological_order = topological_order;
        fw_sccs = sccs;
        fw_sccs_sorted = sorted_sccs;
        fw_union_find = uf;
    } else {
        bw_ordered_sccs = ordered_sccs;
        bw_topological_order = topological_order;
        bw_sccs = sccs;
        bw_sccs_sorted = sorted_sccs;
        bw_union_find = uf;
    }
}

/**
 * @brief Get the opposite bucket number for a given bucket index.
 *
 * This function retrieves the opposite bucket number based on the current
 * bucket index and the specified direction. It determines the node and bounds
 * of the current bucket, then calculates the opposite bucket index using the
 * appropriate direction.
 *
 */
template <Direction D>
int BucketGraph::get_opposite_bucket_number(int current_bucket_index,
                                            std::vector<double> &inc) {
    // TODO: adjust to multi-resource case
    auto &current_bucket = (D == Direction::Forward)
                               ? fw_buckets[current_bucket_index]
                               : bw_buckets[current_bucket_index];
    int &node = current_bucket.node_id;
    auto &theNode = nodes[node];

    // Find the opposite bucket using the appropriate direction
    int opposite_bucket_index = -1;
    std::vector<double> reference_point(options.main_resources.size());
    for (int r = 0; r < options.main_resources.size(); ++r) {
        if constexpr (D == Direction::Forward) {
            reference_point[r] = std::max(inc[r], theNode.lb[r]);
        } else {
            reference_point[r] = std::min(inc[r], theNode.ub[r]);
        }
    }
    if constexpr (D == Direction::Forward) {
        opposite_bucket_index =
            get_bucket_number<Direction::Backward>(node, reference_point);
    } else {
        opposite_bucket_index =
            get_bucket_number<Direction::Forward>(node, reference_point);
    }

    return opposite_bucket_index;
}

/**
 * @brief Fixes the bucket arcs for the specified stage.
 *
 * This function performs the bucket arc fixing for the given stage. It
 * initializes necessary variables and runs labeling algorithms to compute
 * forward and backward reduced costs. Based on the computed gap, it performs
 * arc elimination in both forward and backward directions and generates the
 * necessary arcs.
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

        std::vector<double> forward_cbar(
            fw_buckets.size(), std::numeric_limits<double>::infinity());
        std::vector<double> backward_cbar(
            bw_buckets.size(), std::numeric_limits<double>::infinity());

        run_labeling_algorithms<Stage::Four, Full::Full>(forward_cbar,
                                                         backward_cbar);

        gap = std::ceil(incumbent - (relaxation + std::min(0.0, min_red_cost)));

        // check if gap is -inf and early exit, due to IPM
        if (gap < 0) {
            fmt::print(
                "\033[34m_BUCKET FIXING PROCEDURE CAN'T BE EXECUTED DUE TO "
                "GAP\033[0m");
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

    std::vector<double> forward_cbar(fw_buckets.size(),
                                     std::numeric_limits<double>::infinity());
    std::vector<double> backward_cbar(bw_buckets.size(),
                                      std::numeric_limits<double>::infinity());

    run_labeling_algorithms<Stage::Two, Full::Partial>(forward_cbar,
                                                       backward_cbar);

    std::vector<std::vector<Label *>> fw_labels_map(nodes.size());
    std::vector<std::vector<Label *>> bw_labels_map(nodes.size());

    auto group_labels = [&](auto &buckets, auto &labels_map) {
        for (auto &bucket : buckets) {
            for (auto label : bucket.get_labels()) {
                labels_map[label->node_id].push_back(
                    label);  // Directly index using node_id
            }
        }
    };

    // Create tasks for forward and backward labels grouping
    auto forward_task = stdexec::schedule(bi_sched) | stdexec::then([&]() {
                            group_labels(fw_buckets, fw_labels_map);
                        });
    auto backward_task = stdexec::schedule(bi_sched) | stdexec::then([&]() {
                             group_labels(bw_buckets, bw_labels_map);
                         });

    // Execute the tasks in parallel
    auto work =
        stdexec::when_all(std::move(forward_task), std::move(backward_task));

    stdexec::sync_wait(std::move(work));

    auto num_fixes = 0;
    //  Function to find the minimum cost label in a vector of labels
    auto find_min_cost_label =
        [](const std::vector<Label *> &labels) -> const Label * {
        return *std::min_element(
            labels.begin(), labels.end(),
            [](const Label *a, const Label *b) { return a->cost < b->cost; });
    };
    for (const auto &node_I : nodes) {
        const auto &fw_labels = fw_labels_map[node_I.id];
        if (fw_labels.empty()) continue;  // Skip if no labels for this node_id

        for (const auto &node_J : nodes) {
            if (node_I.id == node_J.id)
                continue;  // Compare based on id (or other key field)
            const auto &bw_labels = bw_labels_map[node_J.id];
            if (bw_labels.empty())
                continue;  // Skip if no labels for this node_id

            const Label *min_fw_label = find_min_cost_label(fw_labels);
            const Label *min_bw_label = find_min_cost_label(bw_labels);

            if (!min_fw_label || !min_bw_label) continue;

            const VRPNode &L_last_node = nodes[min_fw_label->node_id];
            auto cost = getcij(min_fw_label->node_id, min_bw_label->node_id);

            bool violated = false;
            for (auto r = 0; r < options.resources.size(); ++r) {
                if (options.resources[r] == "time") {
                    if (min_fw_label->resources[TIME_INDEX] + cost +
                            L_last_node.duration >
                        min_bw_label->resources[TIME_INDEX]) {
                        violated = true;
                        break;
                    }
                } else {
                    if (min_fw_label->resources[r] +
                            L_last_node.consumption[r] >
                        min_bw_label->resources[r]) {
                        violated = true;
                        break;
                    }
                }
            }

            if (violated) continue;

            if (min_fw_label->cost + cost + min_bw_label->cost > gap) {
                fixed_arcs[node_I.id][node_J.id] = 1;  // Index with node ids
                num_fixes++;
            }
        }
    }
}

template <Symmetry SYM>
void BucketGraph::set_adjacency_list() {
    // Clear existing arcs
    for (auto &node : nodes) {
        node.clear_arcs();
    }

    // Compute clusters using MST-based clustering
    MST mst_solver(nodes,
                   [this](int from, int to) { return this->getcij(from, to); });

    // Compute theta dynamically based on the 90th percentile of edge weights
    auto mst = mst_solver.compute_mst();
    std::vector<double> edge_weights;
    edge_weights.reserve(mst.size());
    for (const auto &[weight, from, to] : mst) {
        edge_weights.push_back(weight);
    }

    // Sort edge weights to compute the 90th percentile
    pdqsort(edge_weights.begin(), edge_weights.end());
    double theta = edge_weights[static_cast<size_t>(
        0.9 * edge_weights.size())];  // 75th percentile
    theta = theta / 100;

    print_info("Computed theta: {}\n", theta);

    auto clusters = mst_solver.cluster(theta);
    // print number of clusters
    print_info("Number of clusters: {}\n", clusters.size());

    // Create job-to-cluster mapping
    std::vector<int> job_to_cluster(nodes.size(), -1);
    for (int cluster_id = 0; cluster_id < clusters.size(); ++cluster_id) {
        for (int job : clusters[cluster_id]) {
            job_to_cluster[job] = cluster_id;
        }
    }

    // Pre-calculate time resource index
    const int time_resource_idx =
        std::find(options.resources.begin(), options.resources.end(), "time") -
        options.resources.begin();

    // Lambda for processing node arcs
    auto add_arcs_for_node = [&](const VRPNode &node, int from_bucket,
                                 std::vector<double> &res_inc) {
        using Arc = std::tuple<double, int, std::vector<double>, double>;
        std::vector<Arc> forward_arcs;
        std::vector<Arc> reverse_arcs;
        forward_arcs.reserve(nodes.size());
        reverse_arcs.reserve(nodes.size());

        for (const auto &next_node : nodes) {
            if (next_node.id == options.depot || node.id == next_node.id ||
                (options.pstep && (next_node.id == options.pstep_depot ||
                                   next_node.id == options.pstep_end_depot))) {
                continue;
            }

            const auto travel_cost = getcij(node.id, next_node.id);
            const double cost_inc = travel_cost - next_node.cost;
            const int to_bucket = next_node.id;

            if (from_bucket == to_bucket) continue;

            // Calculate resource increments
            if constexpr (SYM == Symmetry::Asymmetric) {
                for (int r = 0; r < options.resources.size(); ++r) {
                    res_inc[r] = r == time_resource_idx
                                     ? travel_cost + node.duration
                                     : node.consumption[r];
                }
            } else {
                for (int r = 0; r < options.resources.size(); ++r) {
                    res_inc[r] =
                        r == time_resource_idx
                            ? node.duration / 2 + travel_cost +
                                  next_node.duration / 2
                            : (node.consumption[r] + next_node.consumption[r]) /
                                  2;
                }
            }

            // Check resource feasibility
            bool feasible = true;
            for (int r = 0; r < options.resources.size(); ++r) {
                if (node.lb[r] + res_inc[r] > next_node.ub[r]) {
                    feasible = false;
                    break;
                }
            }
            if (!feasible) continue;

            // Calculate priorities
            const bool same_cluster =
                job_to_cluster[node.id] == job_to_cluster[next_node.id];
            const double base_priority = same_cluster ? 5.0 : 1.0;

            // Incorporate dual values (if available)
            double dual_value = -nodes[node.id].cost;

            // Refine priority calculation
            const double priority =
                base_priority + 1.E-5 * next_node.start_time + dual_value;
            const double rev_priority = (same_cluster ? 1.0 : 5.0) +
                                        1.E-5 * node.start_time + dual_value;

            forward_arcs.emplace_back(priority, next_node.id, res_inc,
                                      cost_inc);
            reverse_arcs.emplace_back(rev_priority, next_node.id, res_inc,
                                      cost_inc);
        }

        // Sort arcs by priority (descending order) using pdqsort
        pdqsort(forward_arcs.begin(), forward_arcs.end(),
                [](const Arc &a, const Arc &b) {
                    return std::get<0>(a) > std::get<0>(b);
                });
        pdqsort(reverse_arcs.begin(), reverse_arcs.end(),
                [](const Arc &a, const Arc &b) {
                    return std::get<0>(a) > std::get<0>(b);
                });

        // Add forward arcs
        for (const auto &[priority, to_bucket, res_inc_local, cost_inc] :
             forward_arcs) {
            nodes[node.id].template add_arc<Direction::Forward>(
                node.id, to_bucket, res_inc_local, cost_inc, priority);
        }

        // Add reverse arcs
        for (const auto &[priority, to_bucket, res_inc_local, cost_inc] :
             reverse_arcs) {
            nodes[to_bucket].template add_arc<Direction::Backward>(
                to_bucket, node.id, res_inc_local, cost_inc, priority);
        }
    };

    // Process all nodes
    std::vector<double> res_inc(options.resources.size());
    for (const auto &node : nodes) {
        if (node.id != options.end_depot) {
            add_arcs_for_node(node, node.id, res_inc);
        }
    }
}

/**
 * @brief Initializes the BucketGraph by clearing previous data and setting up
 * forward and backward buckets.
 *
 */
template <Direction D>
void BucketGraph::common_initialization() {
    // Pre-allocate vectors with exact sizes
    merged_labels.clear();
    merged_labels.reserve(50);

    const size_t num_intervals = options.main_resources.size();
    std::vector<double> base_intervals(num_intervals);
    std::vector<double> interval_starts(num_intervals);
    std::vector<double> interval_ends(num_intervals);

    auto &warm_labels = assign_buckets<D>(fw_warm_labels, bw_warm_labels);
    auto &buckets_size = assign_buckets<D>(fw_buckets_size, bw_buckets_size);
    auto &buckets = assign_buckets<D>(fw_buckets, bw_buckets);
    auto &c_bar = assign_buckets<D>(fw_c_bar, bw_c_bar);
    auto &bucket_splits = assign_buckets<D>(fw_bucket_splits, bw_bucket_splits);

    if (merged_labels.size() > 0 && options.warm_start && !just_fixed) {
        warm_labels.reserve(buckets_size);
        warm_labels.clear();
        warm_labels.reserve(buckets_size);
        for (auto bucket = 0; bucket < buckets_size; ++bucket) {
            if (auto *label = buckets[bucket].get_best_label()) {
                warm_labels.push_back(label);
            }
        }
    }

    // Initialize vectors with exact sizes once
    c_bar.resize(buckets_size, std::numeric_limits<double>::infinity());
    // print size of c_bar

    if constexpr (Direction::Forward == D) {
        dominance_checks_per_bucket.assign(buckets_size + 1, 0);
        non_dominated_labels_per_bucket = 0;
    }

    const auto &VRPNode = nodes[0];

    for (size_t b = 0; b < buckets_size; b++) {
        buckets[b].clear();
    }

    // Calculate intervals once
    for (size_t r = 0; r < intervals.size(); ++r) {
        base_intervals[r] =
            (VRPNode.ub[r] - VRPNode.lb[r]) / bucket_splits[VRPNode.id];
        interval_starts[r] = VRPNode.lb[r];
        interval_ends[r] = VRPNode.ub[r];
    }

    if (options.warm_start && !just_fixed) {
        pdqsort(
            warm_labels.begin(), warm_labels.end(),
            [](const Label *a, const Label *b) { return a->cost < b->cost; });

        std::vector<Label *> processed_labels;
        const size_t process_size = std::min(
            warm_labels.size(), static_cast<size_t>(options.n_warm_start));
        processed_labels.reserve(process_size);

        for (size_t i = 0; i < process_size; ++i) {
            auto label = warm_labels[i];
            if (!label->fresh) continue;

            auto new_label = compute_red_cost(label, true);
            if (new_label != nullptr) {
                fw_buckets[new_label->vertex].add_label(new_label);
                processed_labels.push_back(new_label);
            }
        }
        warm_labels = std::move(processed_labels);
    }

    // Initialize intervals
    std::vector<int> current_pos(num_intervals, 0);
    int offset = 0;

    // Lambda for interval combination generation
    auto generate_combinations = [&](bool is_forward, int &offset) {
        auto &label_pool = is_forward ? label_pool_fw : label_pool_bw;
        auto &buckets = is_forward ? fw_buckets : bw_buckets;
        const auto depot_id = is_forward ? options.depot : options.end_depot;
        const int calculated_index_base =
            is_forward ? num_buckets_index_fw[options.depot]
                       : num_buckets_index_bw[options.end_depot];

        std::vector<double> interval_bounds(num_intervals);
        std::function<void(int)> process_intervals = [&](int depth) {
            if (depth == num_intervals) {
                auto depot = label_pool->acquire();
                int calculated_index = calculated_index_base + offset;

                for (int r = 0; r < num_intervals; ++r) {
                    interval_bounds[r] =
                        is_forward
                            ? interval_starts[r] +
                                  current_pos[r] *
                                      roundToTwoDecimalPlaces(base_intervals[r])
                            : interval_ends[r] -
                                  current_pos[r] * roundToTwoDecimalPlaces(
                                                       base_intervals[r]);
                }

                depot->initialize(calculated_index, 0.0, interval_bounds,
                                  depot_id);
                depot->is_extended = false;
                depot->nodes_covered.push_back(depot_id);
                set_node_visited(depot->visited_bitmap, depot_id);
                SRC_MODE_BLOCK(
                    depot->SRCmap.assign(cut_storage->SRCDuals.size(), 0);)
                buckets[calculated_index].add_label(depot);
                buckets[calculated_index].node_id = depot_id;

                offset++;
                return;
            }

            for (int k = 0; k < bucket_splits[depot_id]; ++k) {
                current_pos[depth] = k;
                process_intervals(depth + 1);
            }
        };

        process_intervals(0);
    };

    // Process forward and backward directions
    if constexpr (Direction::Forward == D) {
        generate_combinations(true, offset);
    } else {
        generate_combinations(false, offset);
    }
}
