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
    // Initialize a stack for bucket traversal and push the current bucket onto it
    std::vector<int> bucket_stack;
    bucket_stack.reserve(10);
    bucket_stack.push_back(current_bucket);

    // Traverse buckets using a depth-first approach with the stack
    while (!bucket_stack.empty()) {
        int curr_bucket = bucket_stack.back(); // Get the top bucket from the stack
        bucket_stack.pop_back();               // Pop the top bucket from the stack
        Bvisited.insert(curr_bucket);          // Mark the current bucket as visited

        // Get the opposite direction's Phi, bucket list, and cost bounds (c_bar)
        auto &Phi_opposite     = assign_buckets<D>(Phi_bw, Phi_fw);
        auto &buckets_opposite = assign_buckets<D>(bw_buckets, fw_buckets);
        auto &c_bar_opposite   = assign_buckets<D>(bw_c_bar, fw_c_bar);

        // Get the job IDs for the label and the current bucket in the opposite direction
        const int bucketLjob      = label->job_id;
        const int bucketLprimejob = buckets_opposite[curr_bucket].job_id;

        // Compute the cost between the current label's job and the opposite bucket's job
        const double cost = getcij(bucketLjob, bucketLprimejob);

        // If the label's total cost exceeds theta, stop exploring this path
        if (label->cost + cost + c_bar_opposite[curr_bucket] >= theta) { return; }

        // If the current bucket is not already in Bbidi, check labels in the opposite bucket
        if (Bbidi.find(curr_bucket) == Bbidi.end()) {
            for (const auto &L : buckets_opposite[curr_bucket].get_labels()) {
                if (label->job_id == L->job_id) continue; // Skip labels with the same job ID

                // Check if there are any conflicts in the visited bitmaps of the two labels
                for (size_t i = 0; i < label->visited_bitmap.size(); ++i) {
                    if ((label->visited_bitmap[i] & L->visited_bitmap[i]) != 0) { continue; }
                }

                // Direction-specific time index checks
                if constexpr (D == Direction::Forward) {
                    // Check if extending the label in the forward direction violates the time constraint
                    const VRPJob &L_last_job = jobs[label->job_id];
                    if (label->resources[TIME_INDEX] + cost + L_last_job.duration > L->resources[TIME_INDEX]) {
                        continue;
                    }
                } else {
                    // Check if extending the label in the backward direction violates the time constraint
                    const VRPJob &L_last_job = jobs[L->job_id];
                    if (L->resources[TIME_INDEX] + cost + L_last_job.duration > label->resources[TIME_INDEX]) {
                        continue;
                    }
                }

                // If the total cost is below theta, add the current bucket to Bbidi
                if (label->cost + cost + L->cost < theta) { Bbidi.insert(curr_bucket); }
            }
        }

        // Push neighboring buckets (from Phi_opposite) onto the stack if they haven't been visited
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
    // Assign forward or backward buckets, Phi (adjacency list), and fixed buckets based on the direction
    auto &buckets       = assign_buckets<D>(fw_buckets, bw_buckets);
    auto &Phi           = assign_buckets<D>(Phi_fw, Phi_bw);
    auto &Phi_opposite  = assign_buckets<D>(Phi_bw, Phi_fw);
    auto &fixed_buckets = assign_buckets<D>(fw_fixed_buckets, bw_fixed_buckets);

    // Vector to store jump arcs (Psi)
    std::vector<JumpArc> Psi;

    // Reset all elements of fixed_buckets to 0 in parallel
    for (auto &fb : fixed_buckets) {
        std::fill(fb.begin(), fb.end(), 0); // Efficiently reset all elements of the vector to 0
    }

    // Print the direction of the arc elimination process (forward or backward)
    if constexpr (D == Direction::Forward) {
        print_info("Performing forward bucket arc elimination with theta = {}\n", theta);
    } else {
        print_info("Performing backward bucket arc elimination with theta = {}\n", theta);
    }

    // Define a map to store arcs and the corresponding buckets that can traverse them
    using ArcMap = std::unordered_map<std::pair<std::pair<int, int>, int>, std::unordered_set<int>>;

    ArcMap           local_B_Ba_b;   // Local map to track arcs and associated buckets
    std::atomic<int> removed_arcs{}; // Counter for the number of removed arcs

    // Iterate through all backward buckets (can be forward if direction changes)
    for (auto b = 0; b < fw_buckets_size; ++b) {
        // Get the forward arcs for the job associated with the current bucket
        const auto arcs = jobs[buckets[b].job_id].template get_arcs<D>();

        // Process each arc for the current bucket
        for (const auto &a : arcs) {
            auto a_index = std::make_pair(std::make_pair(a.from, a.to), b); // Create a unique key for the arc

            // If the current bucket has neighboring buckets in Phi, update the local map
            if (!Phi[b].empty()) {
                auto &Bidi_map = local_B_Ba_b[a_index];
                Bidi_map.insert(b); // Insert the current bucket into the map

                // Check neighboring buckets and merge them into the current arc map
                for (const auto &b_prime : Phi[b]) {
                    auto a_index_interest = std::make_pair(std::make_pair(a.from, a.to), b_prime);
                    auto it               = local_B_Ba_b.find(a_index_interest);
                    if (it != local_B_Ba_b.end()) {
                        Bidi_map.insert(it->second.begin(), it->second.end()); // Merge bucket sets
                    }
                }
            } else {
                local_B_Ba_b[a_index] = {}; // If no neighbors, create an empty entry in the map
            }
        }

        // Process jump arcs for the current bucket
        auto jump_arcs = buckets[b].template get_jump_arcs<D>();
        if (!jump_arcs.empty()) {
            for (const auto &a : jump_arcs) {
                auto  a_index    = std::make_pair(std::make_pair(a.base_bucket, a.jump_bucket), b);
                auto  b_opposite = get_opposite_bucket_number<D>(a.jump_bucket); // Get the opposite bucket number
                auto &labels     = buckets[b].get_labels();

                std::unordered_set<int> Bvisited;
                // For each label in the current bucket, update the bucket set
                for (auto &L_item : labels) {
                    auto &Bidi_map = local_B_Ba_b[a_index];
                    Bidi_map.insert(b); // Insert current bucket into the map
                    Bvisited.clear();
                    UpdateBucketsSet<D>(theta, L_item, Bidi_map, b_opposite, Bvisited); // Update bucket set
                }
            }
        }

        // Process regular bucket arcs for the current bucket
        auto bucket_arcs = buckets[b].template get_bucket_arcs<D>();
        for (const auto &a : bucket_arcs) {
            // Create an index for the arc based on the job IDs and the bucket number
            auto a_index =
                std::make_pair(std::make_pair(buckets[a.from_bucket].job_id, buckets[a.to_bucket].job_id), b);
            auto  b_opposite = get_opposite_bucket_number<D>(a.to_bucket); // Get the opposite bucket number
            auto &labels     = buckets[b].get_labels();

            std::unordered_set<int> Bvisited;
            // For each label, update the bucket set using the opposite bucket
            for (auto &L_item : labels) {
                auto &Bidi_map = local_B_Ba_b[a_index];
                Bidi_map.insert(b); // Insert current bucket into the map
                Bvisited.clear();
                UpdateBucketsSet<D>(theta, L_item, Bidi_map, b_opposite, Bvisited); // Update bucket set
            }

            // Check if the arc exists in the opposite bucket's map
            auto it       = local_B_Ba_b.find(a_index);
            bool contains = false;

            if (it != local_B_Ba_b.end()) {
                auto &Bidi_map = it->second;
                // Check if any bucket in the opposite direction contains this arc
                for (const auto &b_prime : Phi_opposite[b_opposite]) {
                    if (Bidi_map.find(b_prime) != Bidi_map.end()) {
                        contains = true;
                        break;
                    }
                }
                // Check if the opposite bucket directly contains the arc
                if (!contains && Bidi_map.find(b_opposite) != Bidi_map.end()) { contains = true; }
            }

            // If the arc is not found in the opposite direction, mark it as fixed and increase the removal count
            if (!contains) {
                fixed_buckets[a.from_bucket][a.to_bucket] = 1;
                ++removed_arcs; // Increment the count of removed arcs
            }
        }
    }

    // Print the total number of arcs removed during the process
    print_info("Removed arcs: {}\n", removed_arcs.load());
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

    // Initialize resources and counters for arcs and missing paths
    std::vector<double> res             = {0.0};
    auto                cost            = 0.0;
    auto                missing_counter = 0; // Counter for missing paths

    int arc_counter = 0; // Counter for jump arcs added

    // Loop through all forward buckets (or backward, depending on direction)
    for (auto b = 0; b < fw_buckets_size; ++b) {
        std::vector<int> B_bar; // Temporary storage for valid bucket indices

        // Retrieve the current bucket arcs and the original arcs for the job
        const auto arcs          = buckets[b].template get_bucket_arcs<D>();
        const auto original_arcs = jobs[buckets[b].job_id].template get_arcs<D>();
        if (arcs.empty()) { continue; } // Skip if no arcs in the current bucket

        // Process each original arc (from the job's perspective)
        for (const auto orig_arc : original_arcs) {
            const auto from_job  = orig_arc.from; // Source job for the arc
            const auto to_job    = orig_arc.to;   // Destination job for the arc
            bool       have_path = false;         // Flag to check if the path exists

            // Check if the path exists in the current bucket arcs
            for (const auto &gamma : arcs) {
                // Skip fixed arcs
                if (fixed_buckets[gamma.from_bucket][gamma.to_bucket] == 1) { continue; }

                auto from_job_b = buckets[gamma.from_bucket].job_id;
                auto to_job_b   = buckets[gamma.to_bucket].job_id;

                // If the arc matches, mark the path as found
                if (from_job == from_job_b && to_job == to_job_b) {
                    have_path = true;
                    break;
                }
            }

            // If no path was found, try to find a jump arc
            if (!have_path) {
                missing_counter++; // Increment missing path counter

                // Start bucket for this job
                auto start_bucket = num_buckets_index[buckets[b].job_id];

                // Look through buckets that are adjacent to the current one in Phi
                for (auto b_prime = b + 1; b_prime < start_bucket + num_buckets[buckets[b].job_id]; ++b_prime) {
                    // Skip if the bucket is not adjacent (not in Phi)
                    if (std::find(Phi[b_prime].begin(), Phi[b_prime].end(), b) == Phi[b_prime].end()) { continue; }

                    // Check arcs in the adjacent bucket (b_prime)
                    auto arcs_prime = buckets[b_prime].template get_bucket_arcs<D>();
                    for (const auto &gamma_prime : arcs_prime) {
                        // Skip fixed arcs
                        if (fixed_buckets[gamma_prime.from_bucket][gamma_prime.to_bucket] == 1) { continue; }

                        auto from_job_prime = buckets[gamma_prime.from_bucket].job_id;
                        auto to_job_prime   = buckets[gamma_prime.to_bucket].job_id;

                        // If the arc matches the original one, add a jump arc
                        if (from_job == from_job_prime && to_job == to_job_prime) {
                            B_bar.push_back(b_prime); // Add the valid bucket to B_bar

                            // Retrieve the resource increment and cost increment from the jump arc
                            auto res  = gamma_prime.resource_increment;
                            auto cost = gamma_prime.cost_increment;

                            // Add a jump arc from the current bucket to the adjacent bucket
                            buckets[b].add_jump_arc(b, b_prime, res, cost, true);
                            arc_counter++; // Increment the jump arc counter
                        }
                    }
                }
            }
        }
    }

    // Print the number of jump arcs added (for debugging or logging purposes)
    print_info("Added {} jump arcs\n", arc_counter);
}
