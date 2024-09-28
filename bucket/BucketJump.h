/**
 * @file BucketJump.h
 * @brief Header file for the Bucket Graph's arc elimination and jump arc utilities in Vehicle Routing Problems (VRP).
 *
 * This file provides functions for managing and eliminating arcs in the Bucket Graph based on cost thresholds and
 * other resource constraints. It includes utilities for updating the set of visited buckets, performing arc
 * elimination, and eliminating jump arcs that do not contribute to improving solutions.
 *
 * Key components:
 * - `UpdateBucketsSet`: Updates the set of visited buckets and checks feasibility for adding arcs to the set.
 * - `BucketArcElimination`: Eliminates non-improving arcs from the Bucket Graph based on a threshold (theta).
 *
 * The utilities use template parameters for direction (`Forward` or `Backward`) to handle the arc elimination
 * process in both directions of the Bucket Graph. It leverages parallel processing using Intel's Threading
 * Building Blocks (TBB) for efficient computation.
 */

#pragma once
#include "BucketGraph.h"
#include "Hashes.h"

#include <execution>
/**
 * Updates the set of buckets based on the given parameters.
 *
 * This function inserts the current bucket into the visited set and then performs various checks and updates
 * based on the provided theta, label, Bbidi, current_bucket, and Bvisited parameters.
 *
 */
template <Direction D>
void BucketGraph::UpdateBucketsSet(const double theta, const Label *label, std::unordered_set<int> &Bbidi,
                                   int &current_bucket, std::unordered_set<int> &Bvisited) {
    // Precompute values and assign references for the direction-specific data
    auto &Phi_opposite     = assign_buckets<D>(Phi_bw, Phi_fw);
    auto &buckets_opposite = assign_buckets<D>(bw_buckets, fw_buckets);
    auto &c_bar_opposite   = assign_buckets<D>(bw_c_bar, fw_c_bar);

    const int        bucketLjob = label->job_id;
    std::vector<int> bucket_stack;
    bucket_stack.reserve(10); // Pre-allocate space for the stack
    bucket_stack.push_back(current_bucket);

    // Lambda to compare bitmaps
    auto bitmaps_conflict = [&](const Label *L1, const Label *L2) {
        for (size_t i = 0; i < L1->visited_bitmap.size(); ++i) {
            if ((L1->visited_bitmap[i] & L2->visited_bitmap[i]) != 0) { return true; }
        }
        return false;
    };

    while (!bucket_stack.empty()) {
        int curr_bucket = bucket_stack.back();
        bucket_stack.pop_back();
        Bvisited.insert(curr_bucket); // Mark the current bucket as visited

        const int    bucketLprimejob = buckets_opposite[curr_bucket].job_id;
        const double cost            = getcij(bucketLjob, bucketLprimejob);

        // Stop exploring if the cost exceeds the threshold
        if (label->cost + cost + c_bar_opposite[curr_bucket] >= theta) { continue; }

        if (Bbidi.find(curr_bucket) == Bbidi.end()) {
            for (const auto &L : buckets_opposite[curr_bucket].get_labels()) {
                if (label->job_id == L->job_id || bitmaps_conflict(label, L)) {
                    continue; // Skip labels with the same job ID or conflicting bitmaps
                }

                const VRPJob &L_last_job      = (D == Direction::Forward) ? jobs[label->job_id] : jobs[L->job_id];
                const double  time_constraint = (D == Direction::Forward)
                                                    ? label->resources[TIME_INDEX] + cost + L_last_job.duration
                                                    : L->resources[TIME_INDEX] + cost + L_last_job.duration;

                if (time_constraint >
                    ((D == Direction::Forward) ? L->resources[TIME_INDEX] : label->resources[TIME_INDEX])) {
                    continue; // Skip if time constraint is violated
                }

                // Insert the current bucket into Bbidi if the total cost is below theta
                if (label->cost + cost + L->cost < theta) { Bbidi.emplace(curr_bucket); }
            }
        }

        // Add neighboring buckets to the stack
        for (int b_prime : Phi_opposite[curr_bucket]) {
            if (Bvisited.find(b_prime) == Bvisited.end()) { bucket_stack.push_back(b_prime); }
        }
    }
}

/**
 * @brief Eliminates bucket arcs based on a given threshold.
 *
 * This function performs bucket arc elimination in either the forward or backward direction,
 * depending on the template parameter `D`. It resets all elements in the fixed buckets to 0,
 * prints information about the direction of elimination, and processes arcs to update the
 * local bucket arc map and fixed buckets.
 *
 */
template <Direction D>
void BucketGraph::BucketArcElimination(double theta) {
    // Assign forward or backward buckets, adjacency lists, and fixed buckets based on direction
    auto &buckets       = assign_buckets<D>(fw_buckets, bw_buckets);
    auto &Phi           = assign_buckets<D>(Phi_fw, Phi_bw);
    auto &Phi_opposite  = assign_buckets<D>(Phi_bw, Phi_fw);
    auto &fixed_buckets = assign_buckets<D>(fw_fixed_buckets, bw_fixed_buckets);
    auto &buckets_size  = assign_buckets<D>(fw_buckets_size, bw_buckets_size);

    // Reset fixed_buckets in parallel
    std::for_each(std::execution::par, fixed_buckets.begin(), fixed_buckets.end(),
                  [](auto &fb) { std::fill(fb.begin(), fb.end(), 0); });

    // Print direction of arc elimination
    if constexpr (D == Direction::Forward) {
        print_info("[Fw] performing bucket arc elimination with theta = {}\n", theta);
    } else {
        print_info("[Bw] performing bucket arc elimination with theta = {}\n", theta);
    }

    // Map to store arcs and corresponding bucket sets
    // using ArcMap = std::unordered_map<std::pair<std::pair<int, int>, int>, std::unordered_set<int>>;
    using ArcMap = std::unordered_map<std::pair<std::pair<int, int>, int>, std::unordered_set<int>, arc_map_hash>;

    ArcMap           local_B_Ba_b;
    std::atomic<int> removed_arcs{0};

    // Lambda for processing jump arcs
    auto process_jump_arcs = [&](int b) {
        const auto &jump_arcs = buckets[b].template get_jump_arcs<D>();
        if (!jump_arcs.empty()) {
            for (const auto &a : jump_arcs) {
                auto  arc_key    = std::make_pair(std::make_pair(a.base_bucket, a.jump_bucket), b);
                int   b_opposite = get_opposite_bucket_number<D>(a.jump_bucket);
                auto &labels     = buckets[b].get_labels();

                std::unordered_set<int> Bvisited;
                for (auto &L_item : labels) {
                    auto &Bidi_map = local_B_Ba_b[arc_key];
                    Bidi_map.insert(b);
                    Bvisited.clear();
                    UpdateBucketsSet<D>(theta, L_item, Bidi_map, b_opposite, Bvisited);
                }
            }
        }
    };

    // Lambda for processing bucket arcs
    auto process_bucket_arcs = [&](int b) {
        const auto &bucket_arcs = buckets[b].template get_bucket_arcs<D>();

        for (const auto &a : bucket_arcs) {
            auto arc_key =
                std::make_pair(std::make_pair(buckets[a.from_bucket].job_id, buckets[a.to_bucket].job_id), b);
            int   b_opposite = get_opposite_bucket_number<D>(a.to_bucket);
            auto &labels     = buckets[b].get_labels();

            std::unordered_set<int> Bvisited;
            for (auto &L_item : labels) {
                auto &Bidi_map = local_B_Ba_b[arc_key];
                Bidi_map.insert(b);
                Bvisited.clear();
                UpdateBucketsSet<D>(theta, L_item, Bidi_map, b_opposite, Bvisited);
            }

            // Check if the arc exists in the opposite direction
            if (auto it = local_B_Ba_b.find(arc_key); it != local_B_Ba_b.end()) {
                auto &Bidi_map = it->second;
                bool  contains = false;
                for (const auto &b_prime : Phi_opposite[b_opposite]) {
                    if (Bidi_map.find(b_prime) != Bidi_map.end()) {
                        contains = true;
                        break;
                    }
                }
                if (!contains && Bidi_map.find(b_opposite) != Bidi_map.end()) { contains = true; }
                if (!contains) {
                    fixed_buckets[a.from_bucket][a.to_bucket] = 1;
                    ++removed_arcs;
                }
            }
        }
    };

    // Process each bucket
    for (int b = 0; b < buckets_size; ++b) {
        const auto &job_arcs = jobs[buckets[b].job_id].template get_arcs<D>();

        // Process arcs
        for (const auto &a : job_arcs) {
            auto  arc_key  = std::make_pair(std::make_pair(a.from, a.to), b);
            auto &Bidi_map = local_B_Ba_b[arc_key];

            if (!Phi[b].empty()) {
                Bidi_map.insert(b);
                for (const auto &b_prime : Phi[b]) {
                    auto neighbor_key = std::make_pair(std::make_pair(a.from, a.to), b_prime);
                    if (auto it = local_B_Ba_b.find(neighbor_key); it != local_B_Ba_b.end()) {
                        Bidi_map.insert(it->second.begin(), it->second.end());
                    }
                }
            }
        }

        // Process jump arcs and regular bucket arcs using the lambdas
        process_jump_arcs(b);
        process_bucket_arcs(b);
    }

    // Print the number of arcs removed
    CONDITIONAL(D, print_info("[Fw] removed arcs: {}\n", removed_arcs.load()),
                print_info("[Bw] removed arcs: {}\n", removed_arcs.load()));
}

/**
 * @brief Obtains jump bucket arcs for the BucketGraph.
 *
 * This function iterates over each bucket in the BucketGraph and adds jump arcs to the set of buckets.
 * It checks if each arc exists in the given Gamma vector for the current bucket.
 * If an arc exists, it adds the corresponding bucket to the set of buckets to add jump arcs to.
 * It then removes the non-component-wise minimal buckets from the set.
 * Finally, it adds jump arcs from the current bucket to each bucket in the set.
 *
 */
template <Direction D>
void BucketGraph::ObtainJumpBucketArcs() {
    // Assign forward or backward buckets, fixed buckets, bucket indices, and Phi (adjacency list) based on direction
    auto &buckets           = assign_buckets<D>(fw_buckets, bw_buckets);
    auto &fixed_buckets     = assign_buckets<D>(fw_fixed_buckets, bw_fixed_buckets);
    auto &num_buckets_index = assign_buckets<D>(num_buckets_index_fw, num_buckets_index_bw);
    auto &num_buckets       = assign_buckets<D>(num_buckets_fw, num_buckets_bw);
    auto &Phi               = assign_buckets<D>(Phi_fw, Phi_bw);
    auto &buckets_size      = assign_buckets<D>(fw_buckets_size, bw_buckets_size);

    int arc_counter     = 0; // Counter for jump arcs added
    int missing_counter = 0; // Counter for missing paths

    // Loop through all buckets
    for (int b = 0; b < buckets_size; ++b) {
        std::vector<int> B_bar; // Temporary storage for valid bucket indices

        // Cache the current bucket's job ID and arcs
        const int   current_job_id = buckets[b].job_id;
        const auto &arcs           = buckets[b].template get_bucket_arcs<D>();
        const auto &original_arcs  = jobs[current_job_id].template get_arcs<D>();

        if (arcs.empty()) { continue; } // Skip if no arcs in the current bucket

        // Process each original arc for the current job
        for (const auto &orig_arc : original_arcs) {
            const int from_job  = orig_arc.from;
            const int to_job    = orig_arc.to;
            bool      have_path = false;

            // Check if the path exists in the current bucket arcs
            for (const auto &gamma : arcs) {
                if (fixed_buckets[gamma.from_bucket][gamma.to_bucket] == 1) { continue; }

                const int from_job_b = buckets[gamma.from_bucket].job_id;
                const int to_job_b   = buckets[gamma.to_bucket].job_id;

                if (from_job == from_job_b && to_job == to_job_b) {
                    have_path = true;
                    break; // Path found, exit early
                }
            }

            // If no path was found, find a jump arc
            if (!have_path) {
                ++missing_counter; // Increment missing path counter

                // Cache the starting bucket and the number of buckets for this job
                const int start_bucket = num_buckets_index[current_job_id];
                const int job_buckets  = num_buckets[current_job_id];

                // Look through adjacent buckets
                for (int b_prime = b + 1; b_prime < start_bucket + job_buckets; ++b_prime) {
                    // Check if b_prime is adjacent to b in Phi
                    const auto &phi_b_prime = Phi[b_prime];
                    if (std::find(phi_b_prime.begin(), phi_b_prime.end(), b) == phi_b_prime.end()) {
                        continue; // Not adjacent, skip
                    }

                    // Check arcs in the adjacent bucket b_prime
                    const auto &arcs_prime = buckets[b_prime].template get_bucket_arcs<D>();
                    for (const auto &gamma_prime : arcs_prime) {
                        if (fixed_buckets[gamma_prime.from_bucket][gamma_prime.to_bucket] == 1) { continue; }

                        const int from_job_prime = buckets[gamma_prime.from_bucket].job_id;
                        const int to_job_prime   = buckets[gamma_prime.to_bucket].job_id;

                        // If the arc matches, add a jump arc
                        if (from_job == from_job_prime && to_job == to_job_prime) {
                            B_bar.push_back(b_prime); // Add the valid bucket to B_bar

                            // Add the jump arc with the resource and cost increments
                            // const double        res = gamma_prime.resource_increment;
                            const std::vector<double> res =
                                gamma_prime.resource_increment; // Wrap the value in a vector
                            const double cost = gamma_prime.cost_increment;

                            buckets[b].add_jump_arc(b, b_prime, res, cost, true);
                            ++arc_counter; // Increment the jump arc counter
                        }
                    }
                }
            }
        }
    }
    CONDITIONAL(D, print_info("[Fw] added {} jump arcs\n", arc_counter),
                print_info("[Bw] added {} jump arcs\n", arc_counter));
}
