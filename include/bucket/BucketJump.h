/**
 * @file BucketJump.h
 * @brief Header file for the Bucket Graph's arc elimination and jump arc
 * utilities in Vehicle Routing Problems (VRP).
 *
 * This file provides functions for managing and eliminating arcs in the Bucket
 * Graph based on cost thresholds and other resource constraints. It includes
 * utilities for updating the set of visited buckets, performing arc
 * elimination, and eliminating jump arcs that do not contribute to improving
 * solutions.
 *
 * Key components:
 * - `UpdateBucketsSet`: Updates the set of visited buckets and checks
 * feasibility for adding arcs to the set.
 * - `BucketArcElimination`: Eliminates non-improving arcs from the Bucket Graph
 * based on a threshold (theta).
 *
 * The utilities use template parameters for direction (`Forward` or `Backward`)
 * to handle the arc elimination process in both directions of the Bucket Graph.
 * It leverages parallel processing using Intel's Threading Building Blocks
 * (TBB) for efficient computation.
 */

#pragma once
#include <execution>
#include <unordered_set>

#include "BucketGraph.h"
#include "Hashes.h"
#include "VRPNode.h"
#include "ankerl/unordered_dense.h"
/**
 * Updates the set of buckets based on the given parameters.
 *
 * This function inserts the current bucket into the visited set and then
 * performs various checks and updates based on the provided theta, label,
 * Bbidi, current_bucket, and Bvisited parameters.
 *
 */
template <Direction D>
void BucketGraph::UpdateBucketsSet(const double theta, const Label *label,
                                   ankerl::unordered_dense::set<int> &Bbidi,
                                   int &current_bucket,
                                   std::vector<uint64_t> &Bvisited) {
    // Select direction-specific data.
    auto &Phi_opposite = assign_buckets<D>(Phi_bw, Phi_fw);
    auto &buckets_opposite = assign_buckets<D>(bw_buckets, fw_buckets);
    auto &c_bar_opposite = assign_buckets<D>(bw_c_bar, fw_c_bar);

    // Cache label values to avoid repeated member lookups.
    const int bucketLnode = label->node_id;
    const double label_cost = label->cost;
    const auto &label_bitmap = label->visited_bitmap;
    const size_t n_bitmap = label_bitmap.size();

    // Precompute a flag for direction (avoid repeated ternary operators).
    constexpr bool forward = (D == Direction::Forward);

    // Lambda to check if the visited bitmaps of L and L_bw overlap.
    auto overlapsVisited = [n_bitmap, this](const Label *L,
                                            const Label *L_bw) -> bool {
        // Make a local copy of the visited bitmap.
        auto visited_bitmap = L->visited_bitmap;

        // check if visited_bitmap overlaps at all
        for (size_t i = 0; i < n_bitmap; ++i) {
            if ((visited_bitmap[i] & L_bw->visited_bitmap[i]) != 0) {
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

    // Inline helper function for bitmap conflict check.
    auto bitmapsConflict = [n_bitmap](const Label *L1,
                                      const Label *L2) -> bool {
        // Assume both L1 and L2 have the same sized visited_bitmap.
        for (size_t i = 0; i < n_bitmap; ++i) {
            if ((L1->visited_bitmap[i] & L2->visited_bitmap[i]) != 0) {
                return true;
            }
        }
        return false;
    };

    // Use a local DFS stack to traverse buckets.
    std::vector<int> bucket_stack;
    bucket_stack.reserve(10);
    bucket_stack.push_back(current_bucket);

    // Process buckets until the stack is empty.
    while (!bucket_stack.empty()) {
        const int curr_bucket = bucket_stack.back();
        bucket_stack.pop_back();

        // Mark the current bucket as visited.
        const size_t segment = curr_bucket >> 6;
        const size_t bit_pos = curr_bucket & 63;
        Bvisited[segment] |= (1ULL << bit_pos);

        // Get the opposite bucket’s label and node id.
        const int bucketLprimenode = buckets_opposite[curr_bucket].node_id;
        double cost = getcij(bucketLnode, bucketLprimenode);

        // Optionally adjust cost with arc duals if RCC mode is enabled.
        RCC_MODE_BLOCK(auto arc_dual =
                           arc_duals.getDual(bucketLnode, bucketLprimenode);
                       cost -= arc_dual;)

        // Skip if the accumulated cost plus lower bound exceeds theta.
        if (numericutils::gt(label_cost + cost + c_bar_opposite[curr_bucket],
                             theta))
            continue;

        // Process this bucket only if it hasn't been processed already.
        if (Bbidi.insert(curr_bucket).second) {
            const auto &opposite_labels =
                buckets_opposite[curr_bucket].get_labels();
            for (const auto &L_opposite : opposite_labels) {
                // Skip labels from the same node or with conflicting visited
                // bitmaps.
                // if (bucketLnode == L_opposite->node_id ||
                // bitmapsConflict(label, L_opposite))
                // continue;
                if (bucketLnode == L_opposite->node_id) continue;

                if constexpr (D == Direction::Forward) {
                    // Skip if the visited bitmaps overlap.
                    if (overlapsVisited(label, L_opposite)) continue;
                } else {
                    // Skip if the visited bitmaps overlap.
                    if (overlapsVisited(L_opposite, label)) continue;
                }

                // Determine the reference node for resource checks.
                const VRPNode &ref_node =
                    forward ? nodes[bucketLnode] : nodes[L_opposite->node_id];

                bool violated = false;
                // Use a local copy of resources if needed.
                const size_t num_resources = options.resources.size();
                for (size_t r = 0; r < num_resources; ++r) {
                    if (options.resources[r] == "time") {
                        // For time, add cost and the node's duration.
                        const double time_constraint =
                            forward ? label->resources[TIME_INDEX] + cost +
                                          ref_node.duration
                                    : L_opposite->resources[TIME_INDEX] + cost +
                                          ref_node.duration;
                        if (numericutils::gt(
                                time_constraint,
                                forward ? L_opposite->resources[TIME_INDEX]
                                        : label->resources[TIME_INDEX])) {
                            violated = true;
                            break;
                        }
                    } else {
                        const double resource_constraint =
                            forward
                                ? label->resources[r] + ref_node.consumption[r]
                                : L_opposite->resources[r] +
                                      ref_node.consumption[r];
                        if (numericutils::gt(resource_constraint,
                                             forward ? L_opposite->resources[r]
                                                     : label->resources[r])) {
                            violated = true;
                            break;
                        }
                    }
                }

                if (violated) continue;

                // If the merged cost is below theta, record this bucket.
                if (numericutils::lt(label_cost + cost + L_opposite->cost,
                                     theta)) {
                    Bbidi.emplace(curr_bucket);
                }
            }
        }

        // Add unvisited neighbor buckets.
        for (int b_prime : Phi_opposite[curr_bucket]) {
            const size_t seg_prime = b_prime >> 6;
            const size_t bit_prime = b_prime & 63;
            if ((Bvisited[seg_prime] & (1ULL << bit_prime)) == 0)
                bucket_stack.push_back(b_prime);
        }
    }
}

/**
 * @brief Eliminates bucket arcs based on a given threshold.
 *
 * This function performs bucket arc elimination in either the forward
 * or backward direction, depending on the template parameter `D`. It
 * resets all elements in the fixed buckets to 0, prints information
 * about the direction of elimination, and processes arcs to update the
 * local bucket arc map and fixed buckets.
 *
 */
template <Direction D>
void BucketGraph::BucketArcElimination(double theta) {
    // Select direction-specific containers.
    auto &buckets = assign_buckets<D>(fw_buckets, bw_buckets);
    auto &Phi = assign_buckets<D>(Phi_fw, Phi_bw);
    auto &Phi_opposite = assign_buckets<D>(Phi_bw, Phi_fw);
    auto &fixed_buckets = assign_buckets<D>(fw_fixed_buckets, bw_fixed_buckets);
    auto &buckets_size = assign_buckets<D>(fw_buckets_size, bw_buckets_size);
    auto &fixed_buckets_bitmap =
        assign_buckets<D>(fw_fixed_buckets_bitmap, bw_fixed_buckets_bitmap);

    // Reset fixed buckets bitmap.
    fixed_buckets_bitmap.assign(fixed_buckets_bitmap.size(), 0);
    const int n_buckets = buckets_size;
    // Compute the number of 64-bit segments for bitmaps.
    const size_t n_segments = (buckets_size + 63) / 64;

    // Define ArcMap to store arc information.
    using ArcMap =
        ankerl::unordered_dense::map<std::pair<std::pair<int, int>, int>,
                                     ankerl::unordered_dense::set<int>,
                                     arc_map_hash>;
    ArcMap local_B_Ba_b;
    int removed_arcs = 0;

    // Helper: create an arc key from two node IDs and a bucket.
    auto create_arc_key = [](int from, int to, int b) {
        return std::make_pair(std::make_pair(from, to), b);
    };

    // Helper lambda to quickly reset a bitmap vector.
    auto reset_bitmap = [&](std::vector<uint64_t> &bitmap) {
        // Using memset since our vector is contiguous.
        if (!bitmap.empty())
            std::memset(bitmap.data(), 0, bitmap.size() * sizeof(uint64_t));
    };

    // Lambda: Process jump arcs for bucket 'b'.
    auto process_jump_arcs = [&](int b) {
        const auto &jump_arcs = buckets[b].template get_jump_arcs<D>();
        if (jump_arcs.empty()) return;

        std::vector<uint64_t> Bvisited(n_segments, 0);
        auto labels = buckets[b].get_non_dominated_labels();

        for (const auto &a : jump_arcs) {
            auto increment = a.resource_increment;
            auto arc_key = create_arc_key(a.base_bucket, a.jump_bucket, b);
            int b_opposite =
                get_opposite_bucket_number<D>(a.jump_bucket, increment);
            auto &Bidi_map = local_B_Ba_b[arc_key];

            // Process each label in the current bucket.
            for (auto &L_item : labels) {
                Bidi_map.insert(b);
                // Reset Bvisited for each UpdateBucketsSet call.
                reset_bitmap(Bvisited);
                UpdateBucketsSet<D>(theta, L_item, Bidi_map, b_opposite,
                                    Bvisited);
            }
        }
    };

    // Lambda: Process bucket arcs for bucket 'b'.
    auto process_bucket_arcs = [&](int b) {
        const auto &bucket_arcs = buckets[b].template get_bucket_arcs<D>();
        if (bucket_arcs.empty()) return;

        auto labels = buckets[b].get_non_dominated_labels();
        std::vector<uint64_t> Bvisited(n_segments, 0);

        for (const auto &a : bucket_arcs) {
            // Build the resource increment vector.
            std::vector<double> increment(options.resources.size(), 0.0);
            for (size_t r = 0; r < options.resources.size(); ++r) {
                if constexpr (D == Direction::Forward)
                    increment[r] = buckets[b].lb[r] + a.resource_increment[r];
                else
                    increment[r] = buckets[b].ub[r] - a.resource_increment[r];
            }

            auto arc_key = create_arc_key(buckets[a.from_bucket].node_id,
                                          buckets[a.to_bucket].node_id, b);
            int b_opposite =
                get_opposite_bucket_number<D>(a.to_bucket, increment);
            auto &Bidi_map = local_B_Ba_b[arc_key];

            // Process each label in bucket 'b'.
            for (auto &L_item : labels) {
                Bidi_map.insert(b);
                reset_bitmap(Bvisited);
                UpdateBucketsSet<D>(theta, L_item, Bidi_map, b_opposite,
                                    Bvisited);
            }

            // Check if the opposite direction contains arcs.
            if (auto it = local_B_Ba_b.find(arc_key);
                it != local_B_Ba_b.end()) {
                auto &Bidi_map_opposite = it->second;
                bool contains = std::any_of(
                    Phi_opposite[b_opposite].begin(),
                    Phi_opposite[b_opposite].end(), [&](int b_prime) {
                        return Bidi_map_opposite.find(b_prime) !=
                               Bidi_map_opposite.end();
                    });
                if (!contains && Bidi_map_opposite.find(b_opposite) ==
                                     Bidi_map_opposite.end()) {
                    size_t bit_pos =
                        static_cast<size_t>(a.from_bucket) * n_buckets +
                        a.to_bucket;
                    fixed_buckets_bitmap[bit_pos / 64] |=
                        (1ULL << (bit_pos % 64));
                    ++removed_arcs;
                }
            }
        }
    };

    // Process each bucket.
    for (int b = 0; b < buckets_size; ++b) {
        // Process node arcs.
        const auto &node_arcs =
            nodes[buckets[b].node_id].template get_arcs<D>();
        for (const auto &a : node_arcs) {
            auto arc_key = std::make_pair(std::make_pair(a.from, a.to), b);
            auto &Bidi_map = local_B_Ba_b[arc_key];

            if (!Phi[b].empty()) {
                Bidi_map.insert(b);
                // Merge neighbor bucket arcs.
                for (const auto &b_prime : Phi[b]) {
                    auto neighbor_key =
                        std::make_pair(std::make_pair(a.from, a.to), b_prime);
                    if (auto it = local_B_Ba_b.find(neighbor_key);
                        it != local_B_Ba_b.end()) {
                        Bidi_map.insert(it->second.begin(), it->second.end());
                    }
                }
            }
        }

        // Process jump arcs and bucket arcs.
        process_jump_arcs(b);
        process_bucket_arcs(b);

        // iterate over fixed buckets and remove them
        for (int b_prime = 0; b_prime < buckets_size; ++b_prime) {
            if (is_bucket_fixed<D>(b, b_prime)) {
                buckets[b].template remove_bucket_arc<D>(b, b_prime);
            }
        }
    }

    // Print the total number of removed arcs.
    if constexpr (D == Direction::Forward)
        print_info("[Fw] removed arcs: {}\n", removed_arcs);
    else
        print_info("[Bw] removed arcs: {}\n", removed_arcs);
}

/**
 * @brief Obtains jump bucket arcs for the BucketGraph.
 *
 * This function iterates over each bucket in the BucketGraph and adds
 * jump arcs to the set of buckets. It checks if each arc exists in the
 * given Gamma vector for the current bucket. If an arc exists, it adds
 * the corresponding bucket to the set of buckets to add jump arcs to.
 * It then removes the non-component-wise minimal buckets from the set.
 * Finally, it adds jump arcs from the current bucket to each bucket in
 * the set.
 *
 */
template <Direction D>
void BucketGraph::ObtainJumpBucketArcs() {
    // Select direction-specific containers.
    auto &buckets = assign_buckets<D>(fw_buckets, bw_buckets);
    auto &fixed_buckets = assign_buckets<D>(fw_fixed_buckets, bw_fixed_buckets);
    auto &num_buckets_idx =
        assign_buckets<D>(num_buckets_index_fw, num_buckets_index_bw);
    auto &num_buckets = assign_buckets<D>(num_buckets_fw, num_buckets_bw);
    auto &Phi = assign_buckets<D>(Phi_fw, Phi_bw);
    auto &buckets_size = assign_buckets<D>(fw_buckets_size, bw_buckets_size);

    int arc_counter = 0;      // Count of jump arcs added
    int missing_counter = 0;  // Count of missing paths

    // Clear jump arcs for all nodes.
    for (auto &node : nodes) {
        node.clear_jump_arcs<D>();
    }

    // Process each bucket.
    for (int b = 0; b < buckets_size; ++b) {
        std::vector<int> B_bar;  // Temporary storage for valid adjacent
                                 // bucket indices

        const int current_node_id = buckets[b].node_id;
        const auto &bucket_arcs = buckets[b].template get_bucket_arcs<D>();
        const auto &orig_arcs = nodes[current_node_id].template get_arcs<D>();

        // Skip if there are no bucket arcs.
        if (bucket_arcs.empty()) {
            continue;
        }

        // For each original arc from the current node...
        for (const auto &orig_arc : orig_arcs) {
            const int from_node = orig_arc.from;
            const int to_node = orig_arc.to;
            bool have_path = false;

            // Check if the current bucket arcs contain the desired
            // path.
            for (const auto &gamma : bucket_arcs) {
                if (is_bucket_fixed<D>(gamma.from_bucket, gamma.to_bucket)) {
                    continue;
                }

                const int from_node_b = buckets[gamma.from_bucket].node_id;
                const int to_node_b = buckets[gamma.to_bucket].node_id;

                if (from_node == from_node_b && to_node == to_node_b) {
                    have_path = true;
                    break;  // Path found: no need to search further.
                }
            }

            // If no valid path was found, try to obtain a jump arc.
            if (!have_path) {
                ++missing_counter;

                // Determine bucket range for the current node.
                const int start_bucket = num_buckets_idx[current_node_id];
                const int node_buckets = num_buckets[current_node_id];

                // Lambda to process candidate adjacent buckets.
                auto process_bucket = [&](int current_b, int candidate_b) {
                    // Only proceed if candidate bucket is adjacent (per
                    // Phi).
                    const auto &phi_candidate = Phi[candidate_b];
                    if (std::find(phi_candidate.begin(), phi_candidate.end(),
                                  current_b) == phi_candidate.end())
                        return;

                    // Check candidate bucket's arcs.
                    const auto &candidate_arcs =
                        buckets[candidate_b].template get_bucket_arcs<D>();
                    for (const auto &gamma_candidate : candidate_arcs) {
                        if (is_bucket_fixed<D>(gamma_candidate.from_bucket,
                                               gamma_candidate.to_bucket))
                            continue;

                        const int from_node_candidate =
                            buckets[gamma_candidate.from_bucket].node_id;
                        const int to_node_candidate =
                            buckets[gamma_candidate.to_bucket].node_id;

                        // If the candidate arc matches the original
                        // arc...
                        if (from_node == from_node_candidate &&
                            to_node == to_node_candidate) {
                            B_bar.push_back(candidate_b);
                            std::vector<double> res =
                                gamma_candidate.resource_increment;

                            const double cost = gamma_candidate.cost_increment;

                            // Add jump arcs both in the bucket and at
                            // the node level.
                            buckets[current_b].template add_jump_arc<D>(
                                current_b, candidate_b, res, cost);
                            nodes[buckets[current_b].node_id]
                                .template add_jump_arc<D>(
                                    current_b, candidate_b, res, cost, to_node);
                            buckets[current_b].template add_bucket_arc<D>(
                                current_b, candidate_b, res, cost, true);
                            ++arc_counter;
                        }
                    }
                };

                // Process candidate buckets. For now both directions
                // use increasing index. (For backward direction, adjust
                // the loop as needed.)
                if constexpr (D == Direction::Forward) {
                    for (int candidate_b = b + 1;
                         candidate_b < start_bucket + node_buckets;
                         ++candidate_b) {
                        process_bucket(b, candidate_b);
                    }
                } else {
                    // Uncomment and adjust if backward should iterate
                    // in reverse order.
                    // for (int candidate_b = b - 1; candidate_b >=
                    // start_bucket;
                    // --candidate_b) {
                    // process_bucket(b, candidate_b);
                    // }
                    for (int candidate_b = b + 1;
                         candidate_b < start_bucket + node_buckets;
                         ++candidate_b) {
                        process_bucket(b, candidate_b);
                    }
                }
            }
        }
    }

    // Print summary of jump arcs added.
    CONDITIONAL(D, print_info("[Fw] added {} jump arcs\n", arc_counter),
                print_info("[Bw] added {} jump arcs\n", arc_counter));
}
