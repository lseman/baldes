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
 * - `std::hash` specializations: Custom hash functions for `std::pair` to enable their use in unordered containers.
 *
 * The utilities use template parameters for direction (`Forward` or `Backward`) to handle the arc elimination
 * process in both directions of the Bucket Graph. It leverages parallel processing using Intel's Threading
 * Building Blocks (TBB) for efficient computation.
 */

#pragma once
#include "BucketGraph.h"

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
                    if (label->resources[0] + cost + L_last_job.duration > L->resources[0]) { continue; }
                } else {
                    const VRPJob &L_last_job = jobs[L->job_id];
                    if (L->resources[0] + cost + L_last_job.duration > label->resources[0]) { continue; }
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
 * Performs bucket arc elimination on the BucketGraph.
 *
 * This function eliminates non-improving bucket arcs based on the given threshold value (theta).
 * It takes a vector of BucketArcs (Gamma), a vector of JumpArcs (Psi), a vector of Labels (L),
 * and a vector of Labels (L_opposite) as input parameters.
 *
 * The function iterates over the buckets in lexicographic order and performs the following steps:
 * 1. Initializes a 2D vector B_Ba_b, where the first dimension is indexed by arcs (a) and the second by
 * buckets (b).
 * 2. Checks if the current bucket's Phi is not empty.
 * 3. Gets the arcs associated with the current bucket in the forward direction.
 * 4. Loops over the arcs and performs the following operations:
 *    a. Initializes the set for B_Ba_b[a][b].
 *    b. Computes the union of B_Ba_b'[a_index][b'] for b' in Phi[b].
 * 5. Gets the jump arcs associated with the current bucket.
 * 6. Loops over the jump arcs and performs the following operations:
 *    a. Clears the set for B_Ba_b[a][b].
 *    b. Updates the set with jump bucket arcs.
 * 7. Updates the set with regular bucket arcs.
 * 8. Eliminates non-improving bucket arcs by checking if the arc is present in B_Ba_b[a][b].
 *
 * @param theta The threshold value for eliminating non-improving bucket arcs.
 * @param Gamma The vector of BucketArcs.
 * @param Psi The vector of JumpArcs.
 * @param L The vector of Labels.
 * @param L_opposite The vector of Labels (opposite).
 */

namespace std {

/**
 * @brief Specialization of the std::hash template for std::pair<int, int>.
 *
 * This struct provides a hash function for pairs of integers, allowing them
 * to be used as keys in unordered containers such as std::unordered_map.
 *
 * @tparam std::pair<int, int> The type of the pair to be hashed.
 */
template <>
struct hash<std::pair<int, int>> {
    std::size_t operator()(const std::pair<int, int> &p) const noexcept {
        // Simple hash combination for two integers
        std::size_t h1 = std::hash<int>()(p.first);
        std::size_t h2 = std::hash<int>()(p.second);
        return h1 ^ (h2 << 1); // Combine the two hashes
    }
};

/**
 * @brief Specialization of std::hash for std::pair<std::pair<int, int>, int>.
 *
 * This struct provides a hash function for a pair consisting of another pair of integers
 * and an integer. It combines the hash values of the inner pair and the integer to produce
 * a single hash value.
 *
 * @tparam None Template specialization for std::pair<std::pair<int, int>, int>.
 */
template <>
struct hash<std::pair<std::pair<int, int>, int>> {
    std::size_t operator()(const std::pair<std::pair<int, int>, int> &p) const noexcept {
        // Reuse the hash for the inner pair
        std::size_t inner_hash = std::hash<std::pair<int, int>>()(p.first);
        std::size_t b_hash     = std::hash<int>()(p.second);

        // Combine the hashes
        return inner_hash ^ (b_hash << 1);
    }
};

} // namespace std

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
                    // auto it = Bidi_map.find(b_prime);
                    // if (it != Bidi_map.end()) { Bidi_map.insert(*it); }
                }
            } else {
                local_B_Ba_b[a_index] = {};
            }
        }

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
                // if (Bidi_map.find(b) != Bidi_map.end()) {
                for (const auto &b_prime : Phi_opposite[b_opposite]) {
                    if (Bidi_map.find(b_prime) != Bidi_map.end()) {
                        contains = true;
                        break;
                    }
                }
                if (!contains && Bidi_map.find(b_opposite) != Bidi_map.end()) { contains = true; }
                // ß}
            }

            if (!contains) {
                fixed_buckets[a.from_bucket][a.to_bucket] = 1;
                ++removed_arcs;
            }
        }
    }

    print_info("Removed arcs: {}\n", removed_arcs.load());
}
