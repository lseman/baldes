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

#include <tbb/concurrent_unordered_map.h>
#include <tbb/parallel_for.h> // Include TBB's parallel_for

/**
 * Updates the set of buckets based on the given parameters.
 *
 * This function inserts the current bucket into the visited set and then performs various checks and updates
 * based on the provided theta, label, Bbidi, current_bucket, and Bvisited parameters.
 *
 * @param theta The threshold value for the cost calculation.
 * @param label A pointer to the Label object.
 * @param Bbidi A reference to the set of integers representing the Bbidi set.
 * @param current_bucket The current bucket being processed.
 * @param Bvisited A reference to the set of integers representing the visited buckets.
 */
template <Direction D>
void BucketGraph::UpdateBucketsSet(const double theta, Label *&label, std::unordered_set<int> &Bbidi,
                                   int &current_bucket, std::unordered_set<int> &Bvisited) {
    std::stack<int> bucket_stack;
    bucket_stack.push(current_bucket);

    while (!bucket_stack.empty()) {
        int curr_bucket = bucket_stack.top();
        bucket_stack.pop();
        Bvisited.insert(curr_bucket);

        auto &Phi_opposite     = assign_buckets<D>(Phi_bw, Phi_fw);
        auto &buckets_opposite = assign_buckets<D>(bw_buckets, fw_buckets);
        auto &c_bar_opposite   = assign_buckets<D>(bw_c_bar, fw_c_bar);

        int    bucketLjob      = label->job_id;
        int    bucketLprimejob = buckets_opposite[curr_bucket].job_id;
        double cost            = getcij(bucketLjob, bucketLprimejob);

        if (label->cost + cost + c_bar_opposite[curr_bucket] >= theta) { return; }

        if (Bbidi.find(curr_bucket) == Bbidi.end()) {
            for (auto &L : buckets_opposite[curr_bucket].get_labels()) {
                if (label->job_id == L->job_id) continue;

                // Check visited bitmaps for conflicts
                for (size_t i = 0; i < label->visited_bitmap.size(); ++i) {
                    if ((label->visited_bitmap[i] & L->visited_bitmap[i]) != 0) { continue; }
                }

                if constexpr (D == Direction::Forward) {
                    const VRPJob &L_last_job = jobs[label->job_id];
                    if (label->resources[TIME_INDEX] + cost + L_last_job.duration > L->resources[TIME_INDEX]) {
                        continue;
                    }
                } else {
                    const VRPJob &L_last_job = jobs[L->job_id];
                    if (L->resources[TIME_INDEX] + cost + L_last_job.duration > label->resources[TIME_INDEX]) {
                        continue;
                    }
                }

                // Insert into Bbidi if condition is met
                if (label->cost + cost + L->cost < theta) { Bbidi.insert(curr_bucket); }
            }
        }

        // Push opposite bucket indices to the stack
        for (int b_prime : Phi_opposite[curr_bucket]) {
            if (Bvisited.find(b_prime) == Bvisited.end()) { bucket_stack.push(b_prime); }
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
 * @tparam D The direction of elimination (either `Direction::Forward` or `Direction::Backward`).
 * @param theta The threshold value used for bucket arc elimination.
 * @param Gamma A vector of `BucketArc` objects.
 */
template <Direction D>
void BucketGraph::BucketArcElimination(double theta) {
    auto &buckets       = assign_buckets<D>(fw_buckets, bw_buckets);
    auto &Phi           = assign_buckets<D>(Phi_fw, Phi_bw);
    auto &Phi_opposite  = assign_buckets<D>(Phi_bw, Phi_fw);
    auto &fixed_buckets = assign_buckets<D>(fw_fixed_buckets, bw_fixed_buckets);

    std::vector<JumpArc> Psi;
    for (auto &fb : fixed_buckets) {
        std::fill(fb.begin(), fb.end(), 0); // Efficiently reset all elements to 0
    }

    if constexpr (D == Direction::Forward) {
        print_info("Performing forward bucket arc elimination wtith theta = {}\n", theta);
    } else {
        print_info("Performing backward bucket arc elimination with theta = {}\n", theta);
    }

    using ArcMap = std::unordered_map<std::pair<std::pair<int, int>, int>, std::unordered_set<int>>;

    ArcMap           local_B_Ba_b;
    std::atomic<int> removed_arcs{};

    for (auto b = 0; b < bw_buckets_size; ++b) {
        auto arcs = jobs[buckets[b].job_id].template get_arcs<Direction::Forward>();

        for (const auto &a : arcs) {
            auto a_index = std::make_pair(std::make_pair(a.from, a.to), b);

            if (!Phi[b].empty()) {
                auto &Bidi_map = local_B_Ba_b[a_index];
                Bidi_map.insert(b);

                for (const auto &b_prime : Phi[b]) {
                    auto a_index_interest = std::make_pair(std::make_pair(a.from, a.to), b_prime);
                    auto it               = local_B_Ba_b.find(a_index_interest);
                    if (it != local_B_Ba_b.end()) { Bidi_map.insert(it->second.begin(), it->second.end()); }
                }
            } else {
                local_B_Ba_b[a_index] = {};
            }
        }

        // Checking bucket jump arcs
        auto jump_arcs = buckets[b].template get_jump_arcs<D>();
        if (!jump_arcs.empty()) {
            for (const auto &a : jump_arcs) {
                auto  a_index    = std::make_pair(std::make_pair(a.base_bucket, a.jump_bucket), b);
                auto  b_opposite = get_opposite_bucket_number<D>(a.jump_bucket);
                auto &labels     = buckets[b].get_labels();

                std::unordered_set<int> Bvisited;
                for (auto &L_item : labels) {
                    auto &Bidi_map = local_B_Ba_b[a_index];
                    Bidi_map.insert(b);
                    Bvisited.clear();
                    UpdateBucketsSet<D>(theta, L_item, Bidi_map, b_opposite, Bvisited);
                }
            }
        }

        // Checking bucket arcs
        auto bucket_arcs = buckets[b].template get_bucket_arcs<D>();
        for (const auto &a : bucket_arcs) {
            auto a_index =
                std::make_pair(std::make_pair(buckets[a.from_bucket].job_id, buckets[a.to_bucket].job_id), b);
            auto  b_opposite = get_opposite_bucket_number<D>(a.to_bucket);
            auto &labels     = buckets[b].get_labels();

            std::unordered_set<int> Bvisited;
            for (auto &L_item : labels) {
                auto &Bidi_map = local_B_Ba_b[a_index];
                Bidi_map.insert(b);
                Bvisited.clear();
                UpdateBucketsSet<D>(theta, L_item, Bidi_map, b_opposite, Bvisited);
            }

            auto it       = local_B_Ba_b.find(a_index);
            bool contains = false;

            if (it != local_B_Ba_b.end()) {
                auto &Bidi_map = it->second;
                for (const auto &b_prime : Phi_opposite[b_opposite]) {
                    if (Bidi_map.find(b_prime) != Bidi_map.end()) {
                        contains = true;
                        break;
                    }
                }
                if (!contains && Bidi_map.find(b_opposite) != Bidi_map.end()) { contains = true; }
            }

            if (!contains) {
                fixed_buckets[a.from_bucket][a.to_bucket] = 1;
                ++removed_arcs;
            }
        }
    }

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
 * @param Gamma The vector of BucketArcs to check for each bucket.
 */
template <Direction D>
void BucketGraph::ObtainJumpBucketArcs() {
    auto &buckets           = assign_buckets<D>(fw_buckets, bw_buckets);
    auto &fixed_buckets     = assign_buckets<D>(fw_fixed_buckets, bw_fixed_buckets);
    auto &num_buckets_index = assign_buckets<D>(num_buckets_index_fw, num_buckets_index_bw);
    auto &num_buckets       = assign_buckets<D>(num_buckets_fw, num_buckets_bw);
    auto &Phi               = assign_buckets<D>(Phi_fw, Phi_bw);

    std::vector<double> res             = {0.0};
    auto                cost            = 0.0;
    auto                missing_counter = 0;

    int arc_counter = 0;
    for (auto b = 0; b < fw_buckets_size; ++b) {
        std::vector<int> B_bar;

        auto arcs          = buckets[b].template get_bucket_arcs<D>();
        auto original_arcs = jobs[buckets[b].job_id].template get_arcs<D>();
        if (arcs.empty()) { continue; }

        for (const auto orig_arc : original_arcs) {
            auto from_job = orig_arc.from;

            auto to_job    = orig_arc.to;
            bool have_path = false;
            for (const auto &gamma : arcs) {
                if (fixed_buckets[gamma.from_bucket][gamma.to_bucket] == 1) { continue; }
                auto from_job_b = buckets[gamma.from_bucket].job_id;
                auto to_job_b   = buckets[gamma.to_bucket].job_id;
                if (from_job == from_job_b && to_job == to_job_b) {
                    have_path = true;
                    break;
                }
            }

            if (!have_path) {
                missing_counter++;
                auto start_bucket = num_buckets_index[buckets[b].job_id];
                for (auto b_prime = b + 1; b_prime < start_bucket + num_buckets[buckets[b].job_id]; ++b_prime) {
                    if (std::find(Phi[b_prime].begin(), Phi[b_prime].end(), b) == Phi[b_prime].end()) { continue; }
                    auto arcs_prime = buckets[b_prime].template get_bucket_arcs<D>();
                    for (const auto &gamma_prime : arcs_prime) {
                        if (fixed_buckets[gamma_prime.from_bucket][gamma_prime.to_bucket] == 1) { continue; }
                        auto from_job_prime = buckets[gamma_prime.from_bucket].job_id;
                        auto to_job_prime   = buckets[gamma_prime.to_bucket].job_id;
                        if (from_job == from_job_prime && to_job == to_job_prime) {
                            B_bar.push_back(b_prime);
                            auto res  = gamma_prime.resource_increment;
                            auto cost = gamma_prime.cost_increment;
                            buckets[b].add_jump_arc(b, b_prime, res, cost, true);
                            arc_counter++;
                        }
                    }
                }
            }
        }
    }
    // fmt::print("Missing paths: {}\n", missing_counter);

    print_info("Added {} jump arcs\n", arc_counter);
}
