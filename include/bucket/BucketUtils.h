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
    const size_t num_resources = options.main_resources.size();

    const auto &buckets = assign_buckets<D>(fw_buckets, bw_buckets);

    // Optionally adjust resource values (uncomment if you need epsilon
    // adjustments)
    for (size_t r = 0; r < num_resources; ++r) {
        if constexpr (D == Direction::Forward) {
            // resource_values_vec[r] += numericutils::eps;
        } else {  // Direction::Backward
            // resource_values_vec[r] -= numericutils::eps;
        }
    }
    auto &num_buckets_index =
        assign_buckets<D>(num_buckets_index_fw, num_buckets_index_bw);
    // Start with the offset for this node’s buckets.
    int bucket_number = num_buckets_index[node];
    int multiplier = 1;

    for (size_t r = 0; r < num_resources; ++r) {
        // Node-specific resource bounds.
        double node_lb = nodes[node].lb[r];
        double node_ub = nodes[node].ub[r];

        // Global bounds (if needed for computing splits).
        double global_lb = R_min[r];
        double global_ub = R_max[r];
        double full_range = global_ub - global_lb;
        double node_range = node_ub - node_lb;

        // Determine the number of splits (i.e. intervals) for resource r.
        int splits =
            std::max(1, static_cast<int>(std::round((node_range / full_range) *
                                                    intervals[r].interval)));
        double interval_width = node_range / splits;

        int pos = 0;
        if constexpr (D == Direction::Forward) {
            // Compute the bucket "digit" from the lower bound.
            pos = std::min(
                splits - 1,
                static_cast<int>(std::floor((resource_values_vec[r] - node_lb) /
                                            interval_width)));
        } else {  // Direction::Backward
            // For backward, we consider the distance from the upper bound.
            pos = std::min(
                splits - 1,
                static_cast<int>(std::floor((node_ub - resource_values_vec[r]) /
                                            interval_width)));
        }

        bucket_number += pos * multiplier;
        multiplier *= splits;
    }

    // check if buckets[bucket_number] contains the resource_vec
    if (unlikely(!buckets[bucket_number].contains(resource_values_vec))) {
        // print resource_values_vec
        for (auto val : resource_values_vec) {
            fmt::print("{}\n", val);
        }
        // print bucket bounds
        fmt::print("{}\n", buckets[bucket_number].lb[0]);
        fmt::print("{}\n", buckets[bucket_number].ub[0]);

        std::throw_with_nested(
            std::runtime_error("Resource values not contained in bucket"));
        return -1;
    }

    return bucket_number;
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
// A constant representing the number of splits for a node that spans the full
// range. For example, if a node covers the entire full range, it will get
// FULL_RANGE_SPLITS buckets.
void BucketGraph::define_buckets() {
    // 1. Save the old buckets and fixed status.
    std::vector<Bucket> old_buckets;
    std::vector<bool> was_fixed;
    auto &buckets = assign_buckets<D>(fw_buckets, bw_buckets);
    size_t old_size = buckets.size();

    // Map to store the fixed range (min_lb and max_ub) for each node pair.
    ankerl::unordered_dense::map<
        int, ankerl::unordered_dense::map<
                 int, std::pair<std::vector<double>, std::vector<double>>>>
        node_pair_fixed_ranges;

    if (old_size > 0) {
        old_buckets = buckets;
        was_fixed.resize(old_size * old_size, false);
        auto &old_bitmap =
            assign_buckets<D>(fw_fixed_buckets_bitmap, bw_fixed_buckets_bitmap);

        // Mark fixed bucket pairs.
        for (size_t i = 0; i < old_size; ++i) {
            for (size_t j = 0; j < old_size; ++j) {
                size_t pos = i * old_size + j;
                if (pos / 64 < old_bitmap.size()) {
                    was_fixed[pos] =
                        ((old_bitmap[pos / 64] & (1ULL << (pos % 64))) != 0);
                }
            }
        }

        const int num_intervals = options.main_resources.size();
        // For each fixed bucket pair, update the fixed range for the
        // corresponding node pair.
        for (size_t i = 0; i < old_size; ++i) {
            for (size_t j = 0; j < old_size; ++j) {
                if (!was_fixed[i * old_size + j]) continue;
                const auto &bucket_from = old_buckets[i];
                const auto &bucket_to = old_buckets[j];
                int from_node = bucket_from.node_id;
                int to_node = bucket_to.node_id;
                // Initialize if necessary.
                if (node_pair_fixed_ranges[from_node].find(to_node) ==
                    node_pair_fixed_ranges[from_node].end()) {
                    node_pair_fixed_ranges[from_node][to_node] = {
                        std::vector<double>(num_intervals,
                                            std::numeric_limits<double>::max()),
                        std::vector<double>(
                            num_intervals,
                            std::numeric_limits<double>::lowest())};
                }
                auto &[min_lbs, max_ubs] =
                    node_pair_fixed_ranges[from_node][to_node];
                for (int r = 0; r < num_intervals; ++r) {
                    min_lbs[r] = std::min(min_lbs[r], bucket_from.lb[r]);
                    max_ubs[r] = std::max(max_ubs[r], bucket_from.ub[r]);
                }
            }
        }
    }  // End if old buckets exist.

    // 2. Clear the current buckets and fixed bitmap.
    buckets.clear();
    auto &fixed_buckets_bitmap =
        assign_buckets<D>(fw_fixed_buckets_bitmap, bw_fixed_buckets_bitmap);
    fixed_buckets_bitmap.clear();

    // 3. Prepare variables and containers for new bucket definition.
    const int num_intervals = options.main_resources.size();
    std::vector<double> total_ranges(num_intervals);  // unused in current code
    std::vector<double> base_intervals(
        num_intervals);  // unused in current code

    auto &num_buckets = assign_buckets<D>(num_buckets_fw, num_buckets_bw);
    auto &num_buckets_index =
        assign_buckets<D>(num_buckets_index_fw, num_buckets_index_bw);
    auto &node_interval_trees =
        assign_buckets<D>(fw_node_interval_trees, bw_node_interval_trees);
    auto &buckets_size = assign_buckets<D>(fw_buckets_size, bw_buckets_size);
    auto &bucket_splits = assign_buckets<D>(fw_bucket_splits, bw_bucket_splits);

    const size_t num_nodes = nodes.size();
    num_buckets.resize(num_nodes);
    num_buckets_index.resize(num_nodes);
    node_interval_trees.assign(num_nodes, SplayTree());

    // Lambda to calculate an interval for one dimension.
    auto calculate_interval =
        [&](double lb, double ub, double base_interval, int pos,
            int max_intervals, bool is_forward) -> std::pair<double, double> {
        double start, end;
        if (is_forward) {
            // For the first interval, start exactly at lb.
            start = (pos == 0) ? lb : lb + pos * base_interval;
            // For the last interval, end exactly at ub.
            end = (pos == max_intervals - 1) ? ub
                                             : lb + (pos + 1) * base_interval;
        } else {
            // For Backward: first interval gets ub exactly, last gets lb
            // exactly.
            start = (pos == max_intervals - 1) ? lb
                                               : ub - (pos + 1) * base_interval;
            end = (pos == 0) ? ub : ub - pos * base_interval;
        }
        start = roundToTwoDecimalPlaces(start);
        end = roundToTwoDecimalPlaces(end);
        return {start, end};
    };

    int cum_sum = 0;
    int bucket_index = 0;
    std::vector<double> interval_start(num_intervals);
    std::vector<double> interval_end(num_intervals);
    std::vector<int> pos(num_intervals,
                         0);  // combination indices for multi-dimensions

    // 4. Process each node to define new buckets.
    for (const auto &vrp_node : nodes) {
        std::vector<int> node_split_counts(num_intervals);
        std::vector<double> node_base_interval(num_intervals);
        // Determine the number of splits and base interval for each
        // resource dimension.
        for (int r = 0; r < num_intervals; ++r) {
            double full_range = R_max[r] - R_min[r];
            double node_range = vrp_node.ub[r] - vrp_node.lb[r];
            int splits = 1;
            if (std::fabs(full_range) >
                std::numeric_limits<double>::epsilon()) {
                double splits_d =
                    (node_range * intervals[r].interval) / full_range;
                splits = std::max(1, static_cast<int>(std::round(splits_d)));
            }
            node_split_counts[r] = splits;
            node_base_interval[r] =
                (vrp_node.ub[r] - vrp_node.lb[r]) / static_cast<double>(splits);
        }
        // Optionally store node-specific split counts.
        bucket_splits[vrp_node.id] = node_split_counts;

        int n_buckets = 0;
        if (num_intervals == 1) {
            // Single-dimensional case.
            for (int j = 0; j < node_split_counts[0]; ++j) {
                auto [int_start, int_end] = calculate_interval(
                    vrp_node.lb[0], vrp_node.ub[0], node_base_interval[0], j,
                    node_split_counts[0], D == Direction::Forward);
                if constexpr (D == Direction::Backward)
                    int_start = std::max(int_start, vrp_node.lb[0]);
                else
                    int_end = std::min(int_end, vrp_node.ub[0]);
                buckets.push_back(Bucket(vrp_node.id,
                                         std::vector<double>{int_start},
                                         std::vector<double>{int_end}));
                n_buckets++;
                cum_sum++;
            }
        } else {
            // Multi-dimensional: iterate over all combinations.
            fmt::print("Multiple intervals\n");
            std::fill(pos.begin(), pos.end(), 0);
            while (true) {
                for (int r = 0; r < num_intervals; ++r) {
                    auto [int_start, int_end] = calculate_interval(
                        vrp_node.lb[r], vrp_node.ub[r], node_base_interval[r],
                        pos[r], node_split_counts[r], D == Direction::Forward);
                    interval_start[r] = int_start;
                    interval_end[r] = int_end;
                    if constexpr (D == Direction::Backward)
                        interval_start[r] =
                            std::max(interval_start[r], R_min[r]);
                    else
                        interval_end[r] = std::min(interval_end[r], R_max[r]);
                }
                buckets.push_back(
                    Bucket(vrp_node.id, interval_start, interval_end));
                n_buckets++;
                cum_sum++;

                // Generate next combination.
                int i = 0;
                while (i < num_intervals) {
                    ++pos[i];
                    if (pos[i] < node_split_counts[i])
                        break;
                    else {
                        pos[i] = 0;
                        ++i;
                    }
                }
                if (i == num_intervals) break;
            }
        }
        num_buckets[vrp_node.id] = n_buckets;
        num_buckets_index[vrp_node.id] = cum_sum - n_buckets;
        // Optionally assign node_interval_trees[vrp_node.id] = node_tree;
        //
        if constexpr (D == Direction::Forward) {
            n_segments = (n_buckets + 63) >> 6;
        }
    }

    // 5. Process fixed bucket ranges (if available).
    if (!node_pair_fixed_ranges.empty()) {
        size_t new_size = buckets.size();
        size_t required_bitmap_words =
            std::max(size_t(1), ((new_size * new_size) + 63) / 64);
        fixed_buckets_bitmap.clear();
        fixed_buckets_bitmap.resize(required_bitmap_words, 0);
        for (size_t from = 0; from < new_size; ++from) {
            const auto &bucket_from = buckets[from];
            int from_node = bucket_from.node_id;
            if (node_pair_fixed_ranges.find(from_node) ==
                node_pair_fixed_ranges.end())
                continue;
            for (size_t to = 0; to < new_size; ++to) {
                const auto &bucket_to = buckets[to];
                int to_node = bucket_to.node_id;
                if (node_pair_fixed_ranges[from_node].find(to_node) ==
                    node_pair_fixed_ranges[from_node].end())
                    continue;
                const auto &range = node_pair_fixed_ranges[from_node][to_node];
                bool in_range = true;
                for (int r = 0; r < options.main_resources.size(); ++r) {
                    if (bucket_from.lb[r] < range.first[r] ||
                        bucket_from.ub[r] > range.second[r]) {
                        in_range = false;
                        break;
                    }
                }
                if (in_range) {
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

    // 6. Finalize: record the total bucket count.
    buckets_size = buckets.size();
}

/**
 * @brief Generates arcs in the bucket graph based on the specified
 * direction.
 *
 * Generates arcs in the bucket graph based on the specified direction.
 *
 */
template <Direction D>
void BucketGraph::generate_arcs() {
    // Mutex for protecting global bucket updates.
    std::mutex buckets_mutex;

    // Clear the appropriate graph.
    if constexpr (D == Direction::Forward) {
        fw_bucket_graph.clear();
    } else {
        bw_bucket_graph.clear();
    }

    // Resolve references based on the direction.
    auto &buckets = assign_buckets<D>(fw_buckets, bw_buckets);
    auto &num_buckets = assign_buckets<D>(num_buckets_fw, num_buckets_bw);
    auto &num_buckets_idx =
        assign_buckets<D>(num_buckets_index_fw, num_buckets_index_bw);
    // Note: we no longer rely solely on bucket_splits here.

    // Pre-compute intervals (base widths) for each node and each resource,
    // using the same logic as in define_buckets.
    const int num_resources = options.resources.size();
    std::vector<std::vector<double>> node_base_intervals(nodes.size());
    // Also store the computed splits per resource for each node.
    std::vector<std::vector<int>> node_split_counts(nodes.size());

    for (size_t node_id = 0; node_id < nodes.size(); ++node_id) {
        const auto &node = nodes[node_id];
        node_base_intervals[node_id].resize(num_resources);
        node_split_counts[node_id].resize(num_resources);
        for (int r = 0; r < num_resources; ++r) {
            double full_range = R_max[r] - R_min[r];
            double node_range = node.ub[r] - node.lb[r];
            int splits = 1;
            if (std::fabs(full_range) >
                std::numeric_limits<double>::epsilon()) {
                double splits_d =
                    (node_range * intervals[r].interval) / full_range;
                splits = std::max(1, static_cast<int>(std::round(splits_d)));
            }
            node_split_counts[node_id][r] = splits;
            node_base_intervals[node_id][r] =
                (node.ub[r] - node.lb[r]) / static_cast<double>(splits);
        }
    }

    // Clear all buckets and their arcs.
    for (auto &bucket : buckets) {
        bucket.clear();
        bucket.clear_arcs(D == Direction::Forward);
    }

    // Helper lambda: validate an arc and, if valid, add it.
    auto try_add_arc =
        [&](const VRPNode &curr_node, int from_bucket, const VRPNode &next_node,
            int to_bucket, const std::vector<double> &curr_node_base,
            const std::vector<double> &next_node_base,
            const std::vector<double> &res_inc, double cost_inc) -> bool {
        bool valid = true;
        if constexpr (D == Direction::Forward) {
            for (int r = 0; r < res_inc.size() && valid; ++r) {
                // For forward, check that adding the resource increment
                // from the from-bucket does not push us past the next
                // node's upper bound.
                if (numericutils::gt(buckets[from_bucket].lb[r] + res_inc[r],
                                     next_node.ub[r])) {
                    valid = false;
                } else {
                    double max_calc =
                        std::max(buckets[from_bucket].lb[r] + res_inc[r],
                                 next_node.lb[r]);
                    if (numericutils::lt(max_calc, buckets[to_bucket].lb[r]) ||
                        numericutils::gte(max_calc, buckets[to_bucket].lb[r] +
                                                        next_node_base[r])) {
                        valid = false;
                    }
                }
            }
        } else {
            for (int r = 0; r < res_inc.size() && valid; ++r) {
                // For backward, check that subtracting the resource
                // increment does not drop below next node's lb.
                if (numericutils::lt(buckets[from_bucket].ub[r] - res_inc[r],
                                     next_node.lb[r])) {
                    valid = false;
                } else {
                    double min_calc =
                        std::min(buckets[from_bucket].ub[r] - res_inc[r],
                                 next_node.ub[r]);
                    if (numericutils::gt(min_calc, buckets[to_bucket].ub[r]) ||
                        numericutils::lte(min_calc, buckets[to_bucket].ub[r] -
                                                        next_node_base[r])) {
                        valid = false;
                    }
                }
            }
        }
        if (valid) {
            // Add the arc to the global structure.
            {
                std::lock_guard<std::mutex> lock(buckets_mutex);
                add_arc<D>(from_bucket, to_bucket, res_inc, cost_inc);
                buckets[from_bucket].template add_bucket_arc<D>(
                    from_bucket, to_bucket, res_inc, cost_inc, false);
            }
        }
        return valid;
    };

    // Lambda to process a single node.
    auto process_node = [&](int node_id) {
        const auto &node = nodes[node_id];
        std::vector<double> res_inc(num_resources);
        // Local container for arcs (if further processing is needed).
        std::vector<std::pair<int, int>> local_arcs;
        local_arcs.reserve(nodes.size());

        const auto arcs = node.get_arcs<D>();
        // Use the pre-computed base intervals for the current node.
        const auto &curr_node_base = node_base_intervals[node.id];

        // For every bucket in the current node.
        for (int i = 0; i < num_buckets[node.id]; ++i) {
            int from_bucket = i + num_buckets_idx[node.id];
            // Process each outgoing arc.
            for (const auto &arc : nodes) {
                const auto &next_node = arc;
                // Avoid self-loop.
                if (node.id == next_node.id) continue;

                const auto &next_node_base = node_base_intervals[next_node.id];
                const double travel_cost = getcij(node.id, next_node.id);
                double cost_inc = travel_cost - next_node.cost;
                // Pre-calculate resource increments.
                for (int r = 0; r < num_resources; r++) {
                    res_inc[r] = node.consumption[r];
                    if (options.resources[r] == "time") {
                        res_inc[r] += travel_cost;
                    }
                }

                // Iterate over buckets in the next node.
                for (int j = 0; j < num_buckets[next_node.id]; ++j) {
                    int to_bucket = j + num_buckets_idx[next_node.id];
                    // if (from_bucket == to_bucket ||
                    //     is_bucket_fixed<D>(from_bucket, to_bucket)) {
                    //     continue;
                    // }
                    if (from_bucket == to_bucket) {
                        continue;
                    }
                    // // Validate and add arc if valid.
                    if (try_add_arc(node, from_bucket, next_node, to_bucket,
                                    curr_node_base, next_node_base, res_inc,
                                    cost_inc)) {
                        local_arcs.emplace_back(from_bucket, to_bucket);
                    }
                }
            }
        }
        // If needed, local_arcs can be used for debugging or further
        // processing.
    };

    // Create a list of tasks (each task is a node index).
    std::vector<int> tasks(nodes.size());
    std::iota(tasks.begin(), tasks.end(), 0);

    // Choose a chunk size based on a heuristic.
    constexpr int chunk_size = 10;

    // Set up the thread pool.
    const unsigned int thread_count =
        std::max(1u, std::thread::hardware_concurrency() / 2);
    exec::static_thread_pool pool(thread_count);
    auto sched = pool.get_scheduler();

    // Launch tasks in parallel over chunks.
    auto bulk_sender = stdexec::bulk(
        stdexec::just(), (tasks.size() + chunk_size - 1) / chunk_size,
        [&, chunk_size](std::size_t chunk_idx) {
            const size_t start_idx = chunk_idx * chunk_size;
            const size_t end_idx =
                std::min(start_idx + chunk_size, tasks.size());
            for (size_t idx = start_idx; idx < end_idx; ++idx) {
                process_node(tasks[idx]);
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
 * bucket graph. It then compares the cost of each label and keeps track of
 * the label with the lowest cost. The best label, along with its associated
 * bucket, is returned.
 *
 */
template <Direction D>
Label *BucketGraph::get_best_label(const std::vector<int> &topological_order,
                                   const std::vector<double> &c_bar,
                                   const std::vector<std::vector<int>> &sccs) {
    double best_cost = std::numeric_limits<double>::infinity();
    Label *best_label = nullptr;
    auto &buckets = assign_buckets<D>(fw_buckets, bw_buckets);

    // Iterate over all buckets, and select the label with the lowest cost.
    for (auto &bucket : buckets) {
        Label *label = bucket.get_best_label();
        if (!label) continue;
        if (label->cost < best_cost) {
            best_cost = label->cost;
            best_label = label;
        }
    }
    return best_label;
}

/**
 * Concatenates the label L with the bucket b and updates the best label
 * pbest.
 *
 */
template <Stage S, Symmetry SYM>
void BucketGraph::ConcatenateLabel(const Label *L, int &b,
                                   std::atomic<double> &best_cost) {
    // Determine the number of segments needed for the bucket visited
    // bitmap.
    const size_t n_segments = (fw_buckets_size + 63) / 64;
    std::vector<uint64_t> Bvisited(n_segments, 0);

    // Use thread_local storage for the bucket stack to avoid reallocations.
    static thread_local std::vector<int> bucket_stack;
    bucket_stack.reserve(R_SIZE);  // Reserve once at initialization
    bucket_stack.clear();          // Clear for reuse
    bucket_stack.push_back(b);

    // Choose symmetric buckets and cost-bar arrays based on symmetry.
    auto &other_buckets = assign_symmetry<SYM>(fw_buckets, bw_buckets);
    auto &other_c_bar = assign_symmetry<SYM>(fw_c_bar, bw_c_bar);

    // Cache frequently accessed values.
    const int L_node_id = L->node_id;
    const auto &L_resources = L->resources;
    const auto &L_last_node = nodes[L_node_id];
    const double L_cost = L->cost;
    const size_t bitmap_size = L->visited_bitmap.size();

    // Pre-compute if branching is active.
    const bool has_branching = !branching_duals->empty();

#if defined(SRC)
    // SRC mode setup: Only relevant in Stage::Four.
    decltype(cut_storage) cutter = nullptr;
    if constexpr (S > Stage::Three) {
        cutter = cut_storage;
    }
    const auto active_cuts = cutter->getActiveCuts();
#endif

    // Lambda to check if the visited bitmaps of L and L_bw overlap.
    auto overlapsVisited = [bitmap_size, this](const Label *L,
                                               const Label *L_bw) -> bool {
        // Make a local copy of the visited bitmap.
        auto visited_bitmap = L->visited_bitmap;

        // check if visited_bitmap overlaps
        for (size_t i = 0; i < bitmap_size; ++i) {
            if (visited_bitmap[i] & L_bw->visited_bitmap[i]) {
                return true;
            }
        }
        // Iterate in reverse over nodes_covered without allocating a new
        // vector.
        // for (auto it = L_bw->nodes_covered.rbegin();
        //      it != L_bw->nodes_covered.rend(); ++it) {
        //     uint16_t node_id = *it;
        //     if (node_id == L->node_id ||
        //         is_node_visited(visited_bitmap, node_id))
        //         return true;
        //     // Update the bitmap for each 64-bit word.
        //     for (size_t i = 0; i < visited_bitmap.size(); ++i) {
        //         uint64_t &current_visited = visited_bitmap[i];
        //         if (current_visited != 0)
        //             current_visited &= neighborhoods_bitmap[node_id][i];
        //     }
        //     set_node_visited(visited_bitmap, node_id);
        // }
        return false;
    };

// Lambda to apply SRC adjustments to total_cost (only for Stage::Four).
#if defined(SRC)
    auto applySRCAdjustments = [&](double total_cost, const Label *L,
                                   const Label *L_bw) -> double {
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
        return total_cost;
    };
#endif

    // Process buckets until the stack is empty.
    while (!bucket_stack.empty()) {
        const int current_bucket = bucket_stack.back();
        bucket_stack.pop_back();

        // Mark current_bucket as visited in the bitmap.
        const size_t segment = current_bucket >> 6;
        const uint64_t bit_mask = bit_mask_lookup[current_bucket & 63];
        Bvisited[segment] |= bit_mask;

        // Get the node id associated with this bucket.
        const int bucketLprimenode = other_buckets[current_bucket].node_id;
        double travel_cost = getcij(L_node_id, bucketLprimenode);

#if defined(RCC) || defined(EXACT_RCC)
        // Adjust travel cost with arc duals in Stage Four.
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

        // Early bound check.
        if ((S != Stage::Enumerate &&
             numericutils::gte(path_cost + bound, best_cost.load())) ||
            (S == Stage::Enumerate &&
             numericutils::gte(path_cost + bound, gap))) {
            continue;
        }

        // Retrieve labels in the current bucket.
        auto &bucket = other_buckets[current_bucket];
        auto labels = bucket.get_labels();
        if (labels.empty()) continue;

        // Process each label in the current bucket.
        for (const Label *L_bw : labels) {
            // Skip invalid labels with combined check
            if (L_bw == nullptr || L_bw->is_dominated ||
                L_bw->node_id == L_node_id) {
                continue;
            }

            // Check feasibility
            if (!check_feasibility(L, L_bw)) {
                continue;
            }

            double total_cost = path_cost + L_bw->cost;

#if defined(SRC)
            // Apply SRC adjustments if in Stage Four.
            total_cost = applySRCAdjustments(total_cost, L, L_bw);
#endif
            double latest_best = best_cost.load();

            // Filter based on total cost against best known cost or gap.
            bool cost_acceptable;
            if constexpr (S != Stage::Enumerate) {
                cost_acceptable = numericutils::lt(total_cost, latest_best);
            } else {
                cost_acceptable = numericutils::lt(total_cost, gap);
            }

            if (!cost_acceptable) {
                continue;
            }
            // Use critical section only when necessary
            {
                std::lock_guard<std::mutex> lock(merge_mutex);

                // Use atomic compare-exchange loop with exponential backoff
                double expected = best_cost.load();
                int backoff_counter = 0;

                while (numericutils::lt(total_cost, expected) &&
                       !best_cost.compare_exchange_weak(expected, total_cost)) {
                    // Simple exponential backoff to reduce contention
                    if (++backoff_counter > 3) {
                        for (int i = 0; i < (1 << (backoff_counter - 3)); ++i) {
                            _mm_pause();  // CPU hint to reduce power in spin
                                          // loops
                        }
                    }
                }

                // Add label if still an improvement
                if (numericutils::lte(total_cost, best_cost.load())) {
                    auto pbest = compute_label<S>(L, L_bw, total_cost);
                    merged_labels.push_back(pbest);
                }
            }
        }
        // Add neighbor buckets to the stack if they have not been visited.
        for (int b_prime : Phi_bw[current_bucket]) {
            const size_t seg_prime = b_prime >> 6;
            const uint64_t mask_prime = bit_mask_lookup[b_prime & 63];
            if (!(Bvisited[seg_prime] & mask_prime)) {
                bucket_stack.push_back(b_prime);
            }
        }
    }
}

/**
 * @brief Handles the computation of Strongly Connected Components (SCCs)
 * for the BucketGraph.
 *
 * This function processes the bucket graph to identify SCCs using Tarjan's
 * algorithm. It extends the bucket graph with arcs defined by the Phi sets,
 * computes the SCCs, and orders them topologically. It also sorts the
 * buckets within each SCC based on their lower or upper bounds, depending
 * on the direction. Finally, it splits the arcs for each SCC and removes
 * duplicates.
 *
 */
template <Direction D>
void BucketGraph::SCC_handler() {
    // Get necessary references.
    auto &Phi = assign_buckets<D>(Phi_fw, Phi_bw);
    auto &buckets = assign_buckets<D>(fw_buckets, bw_buckets);
    auto &bucket_graph = assign_buckets<D>(fw_bucket_graph, bw_bucket_graph);

    // Build an extended bucket graph from bucket_graph using Phi.
    ankerl::unordered_dense::map<int, std::vector<int>> extended_bucket_graph =
        bucket_graph;
    for (size_t i = 0; i < extended_bucket_graph.size(); ++i) {
        const auto &phi_set = Phi[i];
        if (phi_set.empty()) continue;
        for (auto phi_bucket : phi_set) {
            extended_bucket_graph[phi_bucket].push_back(i);
        }
    }

    // Compute SCCs using Tarjan's algorithm.
    SCC scc_finder;
    scc_finder.convertFromUnorderedMap(extended_bucket_graph);
    auto sccs = scc_finder.tarjanSCC();
    auto topological_order = scc_finder.topologicalOrderOfSCCs(sccs);

#ifdef VERBOSE
    // Optional: Print SCCs.
    constexpr auto blue = "\033[34m";
    constexpr auto reset = "\033[0m";
    fmt::print((D == Direction::Forward) ? "FW SCCs:\n" : "BW SCCs:\n");
    for (auto scc : topological_order) {
        fmt::print("{}({}) -> {}", blue, scc, reset);
        for (auto bucket : sccs[scc]) {
            fmt::print("{} ", bucket);
        }
        fmt::print("\n");
    }
    fmt::print("\n");
#endif

    // Order SCCs according to the topological order.
    std::vector<std::vector<int>> ordered_sccs;
    ordered_sccs.reserve(sccs.size());
    for (int scc : topological_order) {
        ordered_sccs.push_back(sccs[scc]);
    }

    // Sort each SCC: For Forward, sort by bucket lower bound; for Backward,
    // sort by bucket upper bound.
    auto sorted_sccs = sccs;  // Make a copy
    for (auto &scc : sorted_sccs) {
        if constexpr (D == Direction::Forward) {
            pdqsort(scc.begin(), scc.end(), [&buckets](int a, int b) {
                return buckets[a].lb[0] < buckets[b].lb[0];
            });
        } else {
            pdqsort(scc.begin(), scc.end(), [&buckets](int a, int b) {
                return buckets[a].ub[0] > buckets[b].ub[0];
            });
        }
    }

    // Ensure each VRP node has the appropriate SCC arc container allocated.
    for (auto &node : nodes) {
        if constexpr (D == Direction::Forward) {
            node.fw_arcs_scc.resize(sccs.size());
        } else {
            node.bw_arcs_scc.resize(sccs.size());
        }
    }

    // Lambda: Filter and collect arcs for a given bucket, node, and
    // direction.
    auto process_bucket_arcs = [&](int bucket, int scc_index) {
        int from_node_id = buckets[bucket].node_id;
        VRPNode &node = nodes[from_node_id];
        // Select the proper arc container and node arcs based on the
        // direction.
        const auto &node_arcs =
            (D == Direction::Forward) ? node.fw_arcs : node.bw_arcs;
        std::vector<Arc> &filtered_arcs = (D == Direction::Forward)
                                              ? node.fw_arcs_scc[scc_index]
                                              : node.bw_arcs_scc[scc_index];

        const auto &bucket_arcs = buckets[bucket].template get_bucket_arcs<D>();
        for (const auto &arc : bucket_arcs) {
            int to_node_id = buckets[arc.to_bucket].node_id;
            // Look for an arc with destination equal to to_node_id.
            auto it = std::find_if(
                node_arcs.begin(), node_arcs.end(),
                [&to_node_id](const Arc &a) { return a.to == to_node_id; });
            if (it != node_arcs.end()) {
                filtered_arcs.push_back(*it);
            }
        }
    };

    // For each SCC, process arcs for each bucket in the SCC.
    int scc_ctr = 0;
    for (const auto &scc : sccs) {
        for (int bucket : scc) {
            process_bucket_arcs(bucket, scc_ctr);
        }
        ++scc_ctr;
    }

    // Lambda: Sort and deduplicate arcs within each node's SCC arc
    // container.
    auto sortAndDedup = [](std::vector<Arc> &arcs) {
        pdqsort(arcs.begin(), arcs.end(), [](const Arc &a, const Arc &b) {
            return std::tie(a.from, a.to) < std::tie(b.from, b.to);
        });
        auto last = std::unique(arcs.begin(), arcs.end(),
                                [](const Arc &a, const Arc &b) {
                                    return a.from == b.from && a.to == b.to;
                                });
        arcs.erase(last, arcs.end());
    };

    // For each node, sort and deduplicate all arc lists.
    for (auto &node : nodes) {
        if constexpr (D == Direction::Forward) {
            for (auto &arcs : node.fw_arcs_scc) {
                sortAndDedup(arcs);
            }
        } else {
            for (auto &arcs : node.bw_arcs_scc) {
                sortAndDedup(arcs);
            }
        }
    }

    // Build union-find structure for SCCs using the ordered SCCs.
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
 * bucket index and the specified direction. It determines the node and
 * bounds of the current bucket, then calculates the opposite bucket index
 * using the appropriate direction.
 *
 */
template <Direction D>
int BucketGraph::get_opposite_bucket_number(int current_bucket_index,
                                            std::vector<double> &inc) {
    // Retrieve the current bucket based on the current direction.
    const auto &current_bucket = (D == Direction::Forward)
                                     ? fw_buckets[current_bucket_index]
                                     : bw_buckets[current_bucket_index];
    const int node = current_bucket.node_id;
    const auto &theNode = nodes[node];

    // Pre-compute the reference point for the given node using the main
    // resources.
    const size_t num_resources = options.main_resources.size();
    // Optionally, add an assertion or check that inc has the expected size.
    assert(inc.size() == num_resources && theNode.lb.size() == num_resources &&
           theNode.ub.size() == num_resources);

    std::vector<double> reference_point(num_resources);

    if constexpr (D == Direction::Forward) {
        // For Forward direction:
        // Use the maximum of inc and the node's lower bounds.
        // This ensures we do not fall below the allowed lower bound.
        for (size_t i = 0; i < num_resources; ++i) {
            reference_point[i] = std::max(inc[i], theNode.lb[i]);
            // Optionally, adjust with epsilon if needed:
            // reference_point[i] += numericutils::eps;
        }
        // Get the opposite bucket using the Backward tree.
        return get_bucket_number<Direction::Backward>(node, reference_point);
    } else {
        // For Backward direction:
        // Use the minimum of inc and the node's upper bounds.
        // This ensures we do not exceed the allowed upper bound.
        for (size_t i = 0; i < num_resources; ++i) {
            reference_point[i] = std::min(inc[i], theNode.ub[i]);
            // Optionally, adjust with epsilon if needed:
            // reference_point[i] -= numericutils::eps;
        }
        // Get the opposite bucket using the Forward tree.
        return get_bucket_number<Direction::Forward>(node, reference_point);
    }
}

/**
 * @brief Fixes the bucket arcs for the specified stage.
 *
 * This function performs the bucket arc fixing for the given stage. It
 * initializes necessary variables and runs labeling algorithms to compute
 * forward and backward reduced costs. Based on the computed gap, it
 * performs arc elimination in both forward and backward directions and
 * generates the necessary arcs.
 *
 */
template <Stage S>
void BucketGraph::bucket_fixing() {
    // Only execute if not already fixed.
    if (!fixed) {
        fmt::print("\033[34m_STARTING BUCKET FIXING PROCEDURE\033[0m\n");
        fixed = true;

        // Compute gap and check feasibility (if gap is negative, exit
        // early).
        gap = std::ceil(incumbent - (relaxation + std::min(0.0, min_red_cost)));
        if (gap < 0) {
            fmt::print(
                "\033[34m_BUCKET FIXING PROCEDURE CAN'T BE EXECUTED DUE TO "
                "GAP\033[0m\n");
            fmt::print("gap: {}\n", gap);
            return;
        }

        // Initialize common structures, arc scores, warm labels, etc.
        common_initialization();

        // Initialize forward and backward c_bar vectors.
        std::vector<double> forward_cbar(
            fw_buckets.size(), std::numeric_limits<double>::infinity());
        std::vector<double> backward_cbar(
            bw_buckets.size(), std::numeric_limits<double>::infinity());

        // Run labeling algorithms for Stage Four with full mode.
        run_labeling_algorithms<Stage::Four, Full::Full>(forward_cbar,
                                                         backward_cbar);
        // Save computed c_bar values.
        fw_c_bar = forward_cbar;
        bw_c_bar = backward_cbar;

        // Print info about bucket arc elimination.
        print_info("performing bucket arc elimination with theta = {}\n", gap);

        // Run forward and backward sections in parallel.
        PARALLEL_SECTIONS(
            work, bi_sched,
            SECTION {
                // Section 1: Process forward direction.
                BucketArcElimination<Direction::Forward>(gap);
                ObtainJumpBucketArcs<Direction::Forward>();
            },
            SECTION {
                // Section 2: Process backward direction.
                BucketArcElimination<Direction::Backward>(gap);
                ObtainJumpBucketArcs<Direction::Backward>();
            });

        // Generate arcs after elimination.
        // generate_arcs();
        fmt::print("\033[34m_BUCKET FIXING PROCEDURE FINISHED\033[0m\n");

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
    if (fixed) {
        return;
    }

    // Reset and initialize state.
    reset_pool();
    reset_fixed();
    common_initialization();

    // Pre-allocate c-bar vectors with infinity.
    std::vector<double> forward_cbar(fw_buckets.size(),
                                     std::numeric_limits<double>::infinity());
    std::vector<double> backward_cbar(bw_buckets.size(),
                                      std::numeric_limits<double>::infinity());

    gap = std::ceil(incumbent - (relaxation + std::min(0.0, min_red_cost)));

    // Run the appropriate labeling algorithms based on stage
    if (S == Stage::Three) {
        // Run the labeling algorithms for Stage::Two with Full::Partial mode.
        run_labeling_algorithms<Stage::Two, Full::Partial>(forward_cbar,
                                                           backward_cbar);
    } else if (S == Stage::Four) {
        fmt::print("\033[34m_STARTING ARC FIXING PROCEDURE\033[0m\n");
        fmt::print("gap: {}\n", gap);
        // Run the labeling algorithms for Stage::Four with Full::Full mode.
        run_labeling_algorithms<Stage::Four, Full::Full>(forward_cbar,
                                                         backward_cbar);
    }

    // Prepare containers for grouping labels by node.
    const size_t num_nodes = nodes.size();

    // Preallocate vectors with nullptr to avoid resizing
    std::vector<const Label *> min_fw_labels(num_nodes, nullptr);
    std::vector<const Label *> min_bw_labels(num_nodes, nullptr);

    // Find minimum cost label for each node directly from buckets
    // without creating intermediate label maps
    for (auto &bucket : fw_buckets) {
        for (const Label *label : bucket.get_labels()) {
            const size_t node_id = label->node_id;
            if (!min_fw_labels[node_id] ||
                label->cost < min_fw_labels[node_id]->cost) {
                min_fw_labels[node_id] = label;
            }
        }
    }

    for (auto &bucket : bw_buckets) {
        for (const Label *label : bucket.get_labels()) {
            const size_t node_id = label->node_id;
            if (!min_bw_labels[node_id] ||
                label->cost < min_bw_labels[node_id]->cost) {
                min_bw_labels[node_id] = label;
            }
        }
    }

    // Pre-compute the index of the "time" resource.
    const size_t time_index = std::distance(
        options.resources.begin(),
        std::find(options.resources.begin(), options.resources.end(), "time"));

    // Precompute lookup tables for common values
    std::vector<double> fw_time_resources(num_nodes, 0.0);
    std::vector<double> bw_time_resources(num_nodes, 0.0);
    std::vector<double> fw_costs(num_nodes, 0.0);
    std::vector<double> bw_costs(num_nodes, 0.0);

    for (size_t i = 0; i < num_nodes; ++i) {
        if (min_fw_labels[i]) {
            fw_time_resources[i] = min_fw_labels[i]->resources[time_index];
            fw_costs[i] = min_fw_labels[i]->cost;
        }
        if (min_bw_labels[i]) {
            bw_time_resources[i] = min_bw_labels[i]->resources[time_index];
            bw_costs[i] = min_bw_labels[i]->cost;
        }
    }

    // Cache node data to avoid repeated lookups
    std::vector<double> node_durations(num_nodes);
    std::vector<std::vector<double>> node_consumptions(
        num_nodes, std::vector<double>(options.resources.size()));

    for (size_t i = 0; i < num_nodes; ++i) {
        const VRPNode &node = nodes[i];
        node_durations[i] = node.duration;
        for (size_t r = 0; r < options.resources.size(); ++r) {
            node_consumptions[i][r] = node.consumption[r];
        }
    }

    size_t num_fixes = 0;
    const double gap_threshold = gap;  // Cache this value

    // Build a list of candidate node pairs to check
    struct NodePair {
        size_t i;
        size_t j;
    };

    std::vector<NodePair> candidate_pairs;
    candidate_pairs.reserve(num_nodes * num_nodes /
                            4);  // Estimate: 25% of pairs will be candidates

    for (size_t i = 0; i < num_nodes; ++i) {
        if (!min_fw_labels[i]) continue;

        for (size_t j = 0; j < num_nodes; ++j) {
            if (i == j || !min_bw_labels[j]) continue;
            candidate_pairs.push_back({i, j});
        }
    }

    // Process candidate pairs in batches to improve cache locality
    const size_t BATCH_SIZE = 256;
    std::vector<std::pair<size_t, size_t>> arcs_to_fix;
    arcs_to_fix.reserve(BATCH_SIZE);

    for (size_t start = 0; start < candidate_pairs.size();
         start += BATCH_SIZE) {
        const size_t end = std::min(start + BATCH_SIZE, candidate_pairs.size());
        arcs_to_fix.clear();

        for (size_t idx = start; idx < end; ++idx) {
            const auto &pair = candidate_pairs[idx];
            const size_t i = pair.i;
            const size_t j = pair.j;

            const double cost =
                getcij(min_fw_labels[i]->node_id, min_bw_labels[j]->node_id);

            // Early rejection using precomputed time resources (often most
            // restrictive)
            if (options.resources[time_index] == "time") {
                if (fw_time_resources[i] + cost +
                        node_durations[min_fw_labels[i]->node_id] >
                    bw_time_resources[j]) {
                    continue;
                }
            }

            // Check all other resources
            bool violated = false;
            const size_t fw_node_id = min_fw_labels[i]->node_id;

            for (size_t r = 0; r < options.resources.size(); ++r) {
                if (r != time_index) {
                    if (min_fw_labels[i]->resources[r] +
                            node_consumptions[fw_node_id][r] >
                        min_bw_labels[j]->resources[r]) {
                        violated = true;
                        break;
                    }
                }
            }

            // If all resource constraints are satisfied and the cost exceeds
            // the gap, fix the arc
            if (!violated &&
                (fw_costs[i] + cost + bw_costs[j] > gap_threshold)) {
                arcs_to_fix.emplace_back(i, j);
                ++num_fixes;
            }
        }

        // Fix arcs in batch
        for (const auto &[i, j] : arcs_to_fix) {
            fix_arc(i, j);
        }
    }

    if (S == Stage::Four) {
        fmt::print("Applied {} arc fixes\n", num_fixes);
        fmt::print("\033[34m_ARC FIXING PROCEDURE FINISHED\033[0m\n");
    }
}

template <Symmetry SYM>
void BucketGraph::set_adjacency_list() {
    // === Step 1: Clear Existing Arcs from Each Node ===
    for (auto &node : nodes) {
        node.clear_arcs();
    }

    // === Step 2: MST-Based Clustering ===
    MST mst_solver(nodes,
                   [this](int from, int to) { return getcij(from, to); });
    auto mst = mst_solver.compute_mst();

    // Collect and sort MST edge weights.
    std::vector<double> edge_weights;
    edge_weights.reserve(mst.size());
    for (const auto &[weight, from, to] : mst) {
        edge_weights.push_back(weight);
    }
    pdqsort(edge_weights.begin(), edge_weights.end());

    // Compute theta as the 90th percentile (scaled down by 100).
    size_t p90_index = static_cast<size_t>(0.9 * edge_weights.size());
    double theta = (edge_weights[p90_index]) / 100.0;
    print_info("Computed theta: {}\n", theta);

    // Cluster nodes and build mapping: job_id -> cluster_id.
    auto clusters = mst_solver.cluster(theta);
    print_info("Number of clusters: {}\n", clusters.size());
    std::vector<int> job_to_cluster(nodes.size(), -1);
    for (int cluster_id = 0; cluster_id < clusters.size(); ++cluster_id) {
        for (int job : clusters[cluster_id]) {
            job_to_cluster[job] = cluster_id;
        }
    }

    // === Step 3: Precompute Resource Information ===
    // Find the index for the "time" resource.
    auto time_it =
        std::find(options.resources.begin(), options.resources.end(), "time");
    const int time_resource_idx =
        (time_it != options.resources.end())
            ? std::distance(options.resources.begin(), time_it)
            : -1;

    // === Step 4: Process Arcs for Each Node ===

    // Helper lambda to compute resource increments.
    // Depending on symmetry, consumption is either taken as-is or averaged.
    auto compute_resource_increments = [&](const VRPNode &from_node,
                                           const VRPNode &to_node,
                                           double travel_cost,
                                           std::vector<double> &inc) {
        const int nres = options.resources.size();
        if constexpr (SYM == Symmetry::Asymmetric) {
            for (int r = 0; r < nres; ++r) {
                inc[r] = (r == time_resource_idx)
                             ? travel_cost + from_node.duration
                             : from_node.consumption[r];
            }
        } else {
            for (int r = 0; r < nres; ++r) {
                inc[r] =
                    (r == time_resource_idx)
                        ? (from_node.duration / 2.0 + travel_cost +
                           to_node.duration / 2.0)
                        : ((from_node.consumption[r] + to_node.consumption[r]) /
                           2.0);
            }
        }
    };

    // Lambda to add forward and reverse arcs from a given node.
    auto add_arcs_for_node = [&](const VRPNode &curr_node) {
        // Containers to accumulate arcs before insertion.
        using ArcTuple = std::tuple<double, int, std::vector<double>, double>;
        std::vector<ArcTuple> forward_arcs;
        std::vector<ArcTuple> reverse_arcs;
        forward_arcs.reserve(nodes.size());
        reverse_arcs.reserve(nodes.size());

        // Iterate over candidate next nodes.
        for (const auto &next_node : nodes) {
            // Skip if the candidate is a depot, the same node, or a pstep
            // depot.
            if (next_node.id == options.depot || curr_node.id == next_node.id ||
                (options.pstep && (next_node.id == options.pstep_depot ||
                                   next_node.id == options.pstep_end_depot))) {
                continue;
            }

            double travel_cost = getcij(curr_node.id, next_node.id);
            double cost_inc = travel_cost - next_node.cost;

            // Skip self-loops (or if bucket indexes match; here using node
            // id as bucket id).
            if (curr_node.id == next_node.id) continue;

            // Compute resource increments.
            std::vector<double> res_inc(options.resources.size());
            compute_resource_increments(curr_node, next_node, travel_cost,
                                        res_inc);

            // Check feasibility: ensure resource consumption does not
            // exceed next node's upper bound.
            bool feasible = true;
            for (int r = 0; r < options.resources.size(); ++r) {
                if (numericutils::gt(curr_node.lb[r] + res_inc[r],
                                     next_node.ub[r])) {
                    feasible = false;
                    break;
                }
            }
            if (!feasible) continue;

            // Compute priorities based on cluster membership.
            bool same_cluster =
                (job_to_cluster[curr_node.id] == job_to_cluster[next_node.id]);
            double forward_priority =
                (same_cluster ? 5.0 : 1.0) + 1.E-5 * next_node.start_time;
            double reverse_priority =
                (same_cluster ? 1.0 : 5.0) + 1.E-5 * curr_node.start_time;

            // Collect arcs for forward and reverse directions.
            forward_arcs.emplace_back(forward_priority, next_node.id, res_inc,
                                      cost_inc);
            reverse_arcs.emplace_back(reverse_priority, next_node.id, res_inc,
                                      cost_inc);
        }

        // Insert the arcs into the corresponding nodes.
        for (const auto &[priority, to_node, res_inc, cost_inc] :
             forward_arcs) {
            nodes[curr_node.id].template add_arc<Direction::Forward>(
                curr_node.id, to_node, res_inc, cost_inc, priority);
        }
        for (const auto &[priority, from_node, res_inc, cost_inc] :
             reverse_arcs) {
            nodes[from_node].template add_arc<Direction::Backward>(
                from_node, curr_node.id, res_inc, cost_inc, priority);
        }
    };

    // Process each node, skipping the end depot.
    for (const auto &node : nodes) {
        if (node.id != options.end_depot) {
            add_arcs_for_node(node);
        }
    }
}

/**
 * @brief Initializes the BucketGraph by clearing previous data and setting
 * up forward and backward buckets.
 *
 */
template <Direction D>
void BucketGraph::common_initialization() {
    // --- Pre-allocate and clear temporary vectors ---
    merged_labels.clear();
    merged_labels.reserve(50);

    // Retrieve active cuts and build a vector of ActiveCutInfo.
    auto &cutter = cut_storage;
    const auto active_cuts = cutter->getActiveCuts();
    std::vector<ActiveCutInfo> active_cuts_info;
    active_cuts_info.reserve(active_cuts.size());
    for (const auto &cut : active_cuts) {
        active_cuts_info.push_back(cut);
    }

    // --- Sort arcs by scores for each node ---
    auto &arc_scores = assign_buckets<D>(fw_arc_scores, bw_arc_scores);
    for (auto &node : nodes) {
        if (!arc_scores[node.id].empty()) {
            node.sort_arcs_by_scores<D>(arc_scores[node.id], nodes,
                                        active_cuts_info);
        }
        arc_scores[node.id].clear();
    }

    // --- Warm start: collect best labels from buckets ---
    auto &warm_labels = assign_buckets<D>(fw_warm_labels, bw_warm_labels);
    auto &buckets_size = assign_buckets<D>(fw_buckets_size, bw_buckets_size);
    auto &buckets = assign_buckets<D>(fw_buckets, bw_buckets);
    auto &c_bar = assign_buckets<D>(fw_c_bar, bw_c_bar);
    auto &bucket_splits = assign_buckets<D>(fw_bucket_splits, bw_bucket_splits);

    if (!merged_labels.empty() && options.warm_start && !just_fixed) {
        warm_labels.clear();
        warm_labels.reserve(buckets_size);
        for (int bucket = 0; bucket < buckets_size; ++bucket) {
            if (auto *label = buckets[bucket].get_best_label()) {
                warm_labels.push_back(label);
            }
        }
    }

    // --- Initialize c_bar vector ---
    c_bar.resize(buckets_size, std::numeric_limits<double>::infinity());

    // --- Initialize dominance check counters for forward direction ---
    if constexpr (Direction::Forward == D) {
        dominance_checks_per_bucket.assign(buckets_size + 1, 0);
        non_dominated_labels_per_bucket = 0;
    }

    // --- Clear all buckets ---
    for (size_t b = 0; b < buckets_size; b++) {
        buckets[b].clear();
    }

    // --- Warm start processing: sort warm_labels and compute reduced costs
    // ---
    if (options.warm_start && !just_fixed) {
        pdqsort(
            warm_labels.begin(), warm_labels.end(),
            [](const Label *a, const Label *b) { return a->cost < b->cost; });

        std::vector<Label *> processed_labels;
        const size_t process_size = std::min(
            warm_labels.size(),
            static_cast<size_t>(options.n_warm_start * warm_labels.size()));
        for (size_t i = 0; i < process_size; ++i) {
            auto label = warm_labels[i];
            // Uncomment if you want to skip non-fresh labels.
            // if (!label->fresh) continue;
            auto new_label = compute_red_cost(label, D == Direction::Forward);
            if (new_label != nullptr) {
                if constexpr (D == Direction::Forward) {
                    fw_buckets[new_label->vertex].add_sorted_label(new_label);
                } else {
                    bw_buckets[new_label->vertex].add_sorted_label(new_label);
                }
                processed_labels.push_back(new_label);
            }
        }
        warm_labels = std::move(processed_labels);
    }

    // --- Prepare interval arrays and compute per-resource splits for depot
    // ---
    const size_t num_intervals = options.main_resources.size();
    std::vector<double> base_intervals(num_intervals);
    std::vector<double> interval_starts(num_intervals);
    std::vector<double> interval_ends(num_intervals);
    // Compute per-resource split counts for the depot
    const auto &VRPNode = nodes[0];  // assuming node 0 is the depot
    std::vector<int> depot_splits(num_intervals);
    for (size_t r = 0; r < num_intervals; ++r) {
        // Use the same ratio as earlier:
        double full_range = R_max[r] - R_min[r];
        double node_range = VRPNode.ub[r] - VRPNode.lb[r];
        depot_splits[r] =
            std::max(1, static_cast<int>(std::round((node_range / full_range) *
                                                    intervals[r].interval)));
        // Compute the base interval for resource r.
        base_intervals[r] = (VRPNode.ub[r] - VRPNode.lb[r]) / depot_splits[r];
        interval_starts[r] = VRPNode.lb[r];
        interval_ends[r] = VRPNode.ub[r];
    }

    // --- Interval-based bucket generation ---
    std::vector<int> current_pos(num_intervals, 0);
    int offset = 0;
    // Lambda for generating interval combinations and creating depots.
    auto generate_combinations = [&](bool is_forward, int &offset) {
        auto &label_pool = is_forward ? label_pool_fw : label_pool_bw;
        auto &buckets = is_forward ? fw_buckets : bw_buckets;
        const auto depot_id = is_forward ? options.depot : options.end_depot;
        const int calculated_index_base =
            is_forward ? num_buckets_index_fw[options.depot]
                       : num_buckets_index_bw[options.end_depot];

        std::vector<double> interval_bounds(num_intervals);
        // Recursive lambda to generate all interval combinations.
        std::function<void(int)> process_intervals = [&](int depth) {
            if (depth == num_intervals) {
                // Acquire a new depot label.
                auto depot = label_pool->acquire();
                int calculated_index = calculated_index_base + offset;
                for (int r = 0; r < static_cast<int>(num_intervals); ++r) {
                    // For forward direction, start at the lower bound and
                    // add multiples of the base interval; adjust similarly
                    // for backward.
                    interval_bounds[r] =
                        is_forward ? interval_starts[r] +
                                         current_pos[r] * base_intervals[r]
                                   : interval_ends[r] -
                                         current_pos[r] * base_intervals[r];
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

            // For the current resource dimension, loop using its specific
            // split count.
            for (int k = 0; k < depot_splits[depth]; ++k) {
                current_pos[depth] = k;
                process_intervals(depth + 1);
            }
        };
        process_intervals(0);
    };

    // Generate combinations for the current direction.
    generate_combinations(D == Direction::Forward, offset);
}
