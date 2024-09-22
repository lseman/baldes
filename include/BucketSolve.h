/**
 * @file BucketSolve.h
 * @brief Defines the BucketGraph class and its solving methods for optimization problems.
 *
 * This file contains the implementation of the `BucketGraph` class, which solves a multi-stage bi-labeling
 * optimization problem using bucket graphs. The `BucketGraph` class handles:
 *
 * - Solving bucket graph optimization problems using multi-stage labeling algorithms.
 * - Adaptive handling of terminal time adjustments based on label distribution.
 * - Adaptive stage transitions to refine solutions based on inner objectives and iteration counts.
 * - Integration of different arc types (standard, jump, fixed) during label extensions.
 * - Dual management and RCC separation for handling advanced routing optimization constraints.
 *
 * The `BucketGraph` class is designed to solve complex routing problems, leveraging multiple stages, parallelism,
 * and adaptive heuristics to optimize paths while respecting resource constraints.
 */

#pragma once

#include "BucketJump.h"
#include "Definitions.h"

#include "../cuts/SRC.h"
#include <cstring>

/**
 * Solves the bucket graph optimization problem using a multi-stage bi-labeling algorithm.
 *
 * The function iteratively adjusts the terminal time and handles different stages of the
 * bi-labeling algorithm to find the optimal paths. The stages are adaptive and transition
 * based on the inner objective values and iteration counts.
 *
 * @return A vector of pointers to Label objects representing the optimal paths.
 */
inline std::vector<Label *> BucketGraph::solve() {
    // Initialize the status as not optimal at the start
    status = Status::NotOptimal;

    //////////////////////////////////////////////////////////////////////
    // ADAPTIVE TERMINAL TIME
    //////////////////////////////////////////////////////////////////////
    // Adjust the terminal time dynamically based on the difference between the number of forward and backward labels
    for (auto split : q_star) {
        // If there are more backward labels than forward labels, increase the terminal time slightly
        if (((static_cast<double>(n_bw_labels) - static_cast<double>(n_fw_labels)) / static_cast<double>(n_fw_labels)) >
            0.05) {
            split += 0.05 * R_max[TIME_INDEX];
        }
        // If there are more forward labels than backward labels, decrease the terminal time slightly
        else if (((static_cast<double>(n_fw_labels) - static_cast<double>(n_bw_labels)) /
                  static_cast<double>(n_bw_labels)) > 0.05) {
            split -= 0.05 * R_max[TIME_INDEX];
        }
    }

    // Placeholder for the final paths (labels) and inner objective value
    std::vector<Label *> paths;
    double               inner_obj;

    //////////////////////////////////////////////////////////////////////
    // ADAPTIVE STAGE HANDLING
    //////////////////////////////////////////////////////////////////////
    // Stage 1: Apply a light heuristic (Stage::One)
    if (s1) {
        stage     = 1;
        paths     = bi_labeling_algorithm<Stage::One>(q_star); // Solve the problem with Stage 1 heuristic
        inner_obj = paths[0]->cost;

        // Transition from Stage 1 to Stage 2 if the objective improves or after 10 iterations
        if (inner_obj >= -1 || iter >= 10) {
            s1 = false;
            s2 = true; // Move to Stage 2
        }
    }
    // Stage 2: Apply a more expensive pricing heuristic (Stage::Two)
    else if (s2) {
        s2        = true;
        stage     = 2;
        paths     = bi_labeling_algorithm<Stage::Two>(q_star); // Solve the problem with Stage 2 heuristic
        inner_obj = paths[0]->cost;

        // Transition from Stage 2 to Stage 3 if the objective improves or after 800 iterations
        if (inner_obj >= -100 || iter > 800) {
            s2 = false;
            s3 = true; // Move to Stage 3
        }
    }
    // Stage 3: Apply a heuristic fixing approach (Stage::Three)
    else if (s3) {
        stage     = 3;
        paths     = bi_labeling_algorithm<Stage::Three>(q_star); // Solve the problem with Stage 3 heuristic
        inner_obj = paths[0]->cost;

        // Transition from Stage 3 to Stage 4 if the objective improves significantly
        if (inner_obj >= -1e-2) {
            s4         = true;
            s3         = false;
            transition = true; // Prepare for transition to Stage 4
        }
    }
    // Stage 4: Exact labeling algorithm with fixing enabled (Stage::Four)
    else if (s4) {
        stage = 4;

#ifdef FIX_BUCKETS
        // If transitioning to Stage 4, print a message and apply fixing
        if (transition) {
            print_cut("Transitioning to stage 4\n");
            fixed        = true;                                       // Enable fixing of buckets in Stage 4
            paths        = bi_labeling_algorithm<Stage::Four>(q_star); // Solve the problem with Stage 4
            transition   = false;                                      // End the transition period
            fixed        = false;
            min_red_cost = paths[0]->cost; // Update the minimum reduced cost
            iter++;
            return paths; // Return the final paths
        }
#endif
        // If not transitioning, continue with Stage 4 algorithm
        paths     = bi_labeling_algorithm<Stage::Four>(q_star);
        inner_obj = paths[0]->cost;

        // If the objective improves sufficiently, set the status to separation or optimal
        if (inner_obj >= -1e-1) {
            ss = true; // Enter separation mode (for SRC handling)
#if !defined(SRC) && !defined(SRC3)
            status = Status::Optimal; // If SRC is not defined, set status to optimal
            return paths;             // Return the optimal paths
#endif
            status = Status::Separation; // If SRC is defined, set status to separation
        }
    }

    iter++; // Increment the iteration counter

    return paths; // Return the final paths after processing
}

/**
 * Performs the labeling algorithm on the BucketGraph.
 *
 * @tparam D The direction of the algorithm (Forward or Backward).
 * @tparam S The stage of the algorithm (One or Two).
 * @param q_point The q-point used in the algorithm.
 * @return A vector of doubles representing the c_bar values for each bucket.
 */
template <Direction D, Stage S, Full F>
std::vector<double> BucketGraph::labeling_algorithm(std::vector<double> q_point, bool full) noexcept {

    // Assign the correct direction buckets, ordered SCCs, and other related structures depending on the direction D
    auto &buckets           = assign_buckets<D>(fw_buckets, bw_buckets);
    auto &ordered_sccs      = assign_buckets<D>(fw_ordered_sccs, bw_ordered_sccs);
    auto &topological_order = assign_buckets<D>(fw_topological_order, bw_topological_order);
    auto &sccs              = assign_buckets<D>(fw_sccs, bw_sccs);
    auto &Phi   = assign_buckets<D>(Phi_fw, Phi_bw); // Forward/backward adjacency list of strongly connected components
    auto &c_bar = assign_buckets<D>(fw_c_bar, bw_c_bar); // Lower bound on the cost of labels in each bucket
    auto &fixed_buckets = assign_buckets<D>(fw_fixed_buckets, bw_fixed_buckets);
    auto &n_labels      = assign_buckets<D>(n_fw_labels, n_bw_labels);         // Number of labels processed
    auto &sorted_sccs   = assign_buckets<D>(fw_sccs_sorted, bw_sccs_sorted);   // Sorted SCCs
    auto &n_buckets     = assign_buckets<D>(fw_buckets_size, bw_buckets_size); // Total number of buckets
    auto &stat_n_labels = assign_buckets<D>(stat_n_labels_fw, stat_n_labels_bw);
    auto &stat_n_dom    = assign_buckets<D>(stat_n_dom_fw, stat_n_dom_bw);

    // Reset the number of labels processed for this run
    n_labels = 0;

    // Set up a vector for tracking visited buckets, each segment represents 64 buckets
    const size_t          n_segments = n_buckets / 64 + 1;
    std::vector<uint64_t> Bvisited(n_segments, 0);

    bool all_ext;       // Flag to indicate if all labels have been extended
    bool dominated;     // Flag to check if a label is dominated
    bool domin_smaller; // Flag to check if a label is dominated by a smaller bucket

    // Iterate through each strongly connected component (SCC) in topological order
    for (const auto &scc_index : topological_order) {
        do {
            all_ext = true; // Assume all labels have been extended at the start
            for (const auto bucket : sorted_sccs[scc_index]) {
                auto bucket_labels = buckets[bucket].get_unextended_labels(); // Get unextended labels from this bucket
                for (Label *label : bucket_labels) {
                    domin_smaller = false;

                    // Clear the visited buckets vector for the current label
                    std::memset(Bvisited.data(), 0, Bvisited.size() * sizeof(uint64_t));

                    // Check if the label is dominated by any labels in smaller buckets
                    domin_smaller =
                        DominatedInCompWiseSmallerBuckets<D, S>(label, bucket, c_bar, Bvisited, ordered_sccs);

                    if (!domin_smaller) {
                    // Lambda function to process new labels after extension
                    auto process_new_label = [&](Label *new_label) {
                            // Partial mode: Skip if the label does not meet q_point requirements
                            if constexpr (F == Full::Partial) {
                                if constexpr (D == Direction::Forward) {
                                    if (label->resources[TIME_INDEX] > q_point[TIME_INDEX]) return;
                                } else {
                                    if (label->resources[TIME_INDEX] <= q_point[TIME_INDEX]) return;
                                }
                            }
                            stat_n_labels++; // Increment number of labels processed

                            int &to_bucket = new_label->vertex; // Get the bucket to which the new label belongs
                            dominated      = false;
                            const auto &to_bucket_labels =
                                buckets[to_bucket].get_labels(); // Get existing labels in the destination bucket

                            // Stage-specific dominance check
                            if constexpr (S == Stage::One) {
                                // If the new label has lower cost, remove dominated labels
                                for (auto *existing_label : to_bucket_labels) {
                                    if (label->cost < existing_label->cost) {
                                        buckets[to_bucket].remove_label(existing_label);
                                    } else {
                                        dominated = true;
                                        break;
                                    }
                                }
                            } else {
// General dominance check
#ifndef AVX
                                for (auto *existing_label : to_bucket_labels) {
                                    if (is_dominated<D, S>(new_label, existing_label)) {
                                        stat_n_dom++; // Increment dominated labels count
                                        dominated = true;
                                        break;
                                    }
                                }
#else
                                if (new_label->check_dominance_against_vector<D, S>(to_bucket_labels)) {
                                    stat_n_dom++; // Increment dominated labels count
                                    dominated = true;
                                }
#
                            }

                            if (!dominated) {
                                // Remove dominated labels from the bucket
                                if constexpr (S != Stage::Enumerate) {
                                    for (auto *existing_label : to_bucket_labels) {
                                        if (is_dominated<D, S>(existing_label, new_label)) {
                                            buckets[to_bucket].remove_label(existing_label);
                                        }
                                    }
                                }

                                n_labels++; // Increment the count of labels added

                                // Add the new label to the bucket
#ifdef SORTED_LABELS
                                buckets[to_bucket].add_sorted_label(new_label);
#elif LIMITED_BUCKETS
                                buckets[to_bucket].add_label_lim(new_label, BUCKET_CAPACITY);
#else
                                buckets[to_bucket].add_label(new_label);
#endif
                                all_ext = false; // Not all labels have been extended, continue processing
                            }
                        };

                        // Process regular arcs for label extension
                        const auto &arcs = jobs[label->job_id].get_arcs<D>(scc_index);
                        for (const auto &arc : arcs) {
                            Label *new_label = Extend<D, S, ArcType::Job, Mutability::Mut>(label, arc);
                            if (!new_label) {
#ifdef UNREACHABLE_DOMINANCE
                                set_job_unreachable(label->unreachable_bitmap, arc.to);
#endif
                                continue; // Skip if label extension failed
                            }
                            process_new_label(new_label); // Process the new label
                        }

#ifdef FIX_BUCKETS
                        // Process jump arcs if in Stage 4
                        if constexpr (S == Stage::Four) {
                            const auto &jump_arcs = buckets[bucket].template get_jump_arcs<D>();
                            for (const auto &jump_arc : jump_arcs) {
                                Label *new_label = Extend<D, S, ArcType::Jump, Mutability::Const>(label, jump_arc);
                                if (!new_label) { continue; } // Skip if label extension failed
                                process_new_label(new_label); // Process the new label
                            }
                        }
#endif
                    }

                    label->set_extended(true); // Mark the label as extended
                }
            }
        } while (!all_ext); // Continue until all labels have been extended

        // Update the cost bounds (c_bar) for the current SCC's buckets
        for (int bucket : sorted_sccs[scc_index]) {
            const auto &labels = buckets[bucket].get_labels();

            // Find the label with the minimum cost in the bucket
            if (!labels.empty()) {
                auto   min_label = std::min_element(labels.begin(), labels.end(),
                                                    [](const Label *a, const Label *b) { return a->cost < b->cost; });
                double min_cost  = (*min_label)->cost;
                c_bar[bucket]    = std::min(c_bar[bucket], min_cost); // Update the lower bound for the bucket
            }

            // Update the cost bound for the bucket's dependencies (Phi buckets)
            for (auto phi_bucket : Phi[bucket]) { c_bar[bucket] = std::min(c_bar[bucket], c_bar[phi_bucket]); }
        }
    }

    // Get the best label from the topological order
    Label *best_label = get_best_label<D>(topological_order, c_bar, sccs);

    // Store the best label for forward or backward direction
    if constexpr (D == Direction::Forward) {
        fw_best_label = best_label;
    } else {
        bw_best_label = best_label;
    }

    return c_bar; // Return the cost bounds for each bucket
}

/**
 * Performs the bi-labeling algorithm on the BucketGraph.
 *
 * @param q_star The vector of doubles representing the resource constraints.
 * @param S The stage of the algorithm to run (Stage::One, Stage::Two, or Stage::Three).
 * @return A vector of Label pointers representing the best labels obtained from the algorithm.
 */

template <Stage S>
std::vector<Label *> BucketGraph::bi_labeling_algorithm(std::vector<double> q_star) {

    // If in Stage 3, apply heuristic fixing based on q_star
    if constexpr (S == Stage::Three) {
        heuristic_fixing<S>(q_star);
    }
    // If in Stage 4, reset fixed buckets if it's the first reset
    else if constexpr (S == Stage::Four) {
        if (first_reset) {
            reset_fixed();
            first_reset = false; // Ensure this is only done once
        }
    }

#ifdef FIX_BUCKETS
    // If in Stage 4, apply bucket fixing based on q_star
    if constexpr (S == Stage::Four) { bucket_fixing<S>(q_star); }
#endif

    // Reset the label pool to ensure no leftover labels from previous runs
    reset_pool();
    // Perform any common initializations (data structures, etc.)
    common_initialization();

    // Initialize the cost bound vectors for forward and backward buckets
    std::vector<double> forward_cbar(fw_buckets.size());
    std::vector<double> backward_cbar(bw_buckets.size());

    // Run the labeling algorithm in both directions, but only partially (Full::Partial)
    run_labeling_algorithms<S, Full::Partial>(forward_cbar, backward_cbar, q_star);

    // Acquire the best label from the forward label pool (will later combine with backward)
    auto best_label = label_pool_fw.acquire();

    // Check if the best forward and backward labels can be combined into a feasible solution
    if (check_feasibility(fw_best_label, bw_best_label)) {
        // If feasible, compute and combine the best forward and backward labels into one
        best_label = compute_label(fw_best_label, bw_best_label);
    } else {
        // If not feasible, set the best label to have infinite cost (not usable)
        best_label->cost         = 0.0;
        best_label->real_cost    = std::numeric_limits<double>::infinity();
        best_label->jobs_covered = {};
    }

    // Add the best label (combined forward/backward path) to the merged label list
    merged_labels.push_back(best_label);

    // For Stage Enumerate, print a message when labels are concatenated
    if constexpr (S == Stage::Enumerate) { fmt::print("Labels generated, concatenating...\n"); }

    // Setup visited buckets tracking for forward buckets
    const size_t          n_segments = fw_buckets_size / 64 + 1;
    std::vector<uint64_t> Bvisited(n_segments, 0);

    // Iterate over all forward buckets
    for (auto bucket = 0; bucket < fw_buckets_size; ++bucket) {
        auto       &current_bucket = fw_buckets[bucket];          // Get the current bucket
        const auto &labels         = current_bucket.get_labels(); // Get labels in the current bucket

        // Process each label in the bucket
        for (const Label *L : labels) {

#ifndef ORIGINAL_ARCS
            // Get arcs corresponding to jobs for this label (Forward direction)
            const auto &to_arcs = jobs[L->job_id].get_arcs<Direction::Forward>();
#else
            // Alternative: Get arcs stored directly in the bucket
            const auto &to_arcs = fw_buckets[bucket].get_bucket_arcs(true);
#endif
            // Iterate over each arc from the current job
            for (const auto &arc : to_arcs) {
                const auto &to_job = arc.to;

                // Skip fixed arcs in Stage 3 if necessary
                if constexpr (S == Stage::Three) {
                    if (fixed_arcs[L->job_id][to_job] == 1) {
                        continue; // Skip if the arc is fixed
                    }
                }

                // Attempt to extend the current label using this arc
                auto L_prime = Extend<Direction::Forward, S, ArcType::Job, Mutability::Const>(L, arc);

                // Check if the new label is valid and respects the q_star constraints
                if (!L_prime || L_prime->resources[TIME_INDEX] <= q_star[TIME_INDEX]) {
                    continue; // Skip invalid labels or those that exceed q_star
                }

                // Get the bucket for the extended label
                auto b_prime = L_prime->vertex;

#ifdef FIX_BUCKETS
                // If in Stage 4, skip labels in fixed buckets
                if constexpr (S == Stage::Four) {
                    if (fw_fixed_buckets[bucket][b_prime] == 1) {
                        continue; // Skip if the bucket is fixed
                    }
                }
#endif

                // Clear visited buckets tracking for this new label extension
                std::memset(Bvisited.data(), 0, Bvisited.size() * sizeof(uint64_t));

                // Concatenate this new label with the best label found so far
                ConcatenateLabel<S>(L, b_prime, best_label, Bvisited, q_star);
            }
        }
    }

    // Sort the merged labels by cost, to prioritize cheaper labels
    pdqsort(merged_labels.begin(), merged_labels.end(),
            [](const Label *a, const Label *b) { return a->cost < b->cost; });

#ifdef RIH
    // If we are in Stage 2 or above, we run the RIH (Route Improvement Heuristic) in the background
    const int LABELS_MAX = 2; // Set a maximum of 2 labels to be improved

    if constexpr (S >= Stage::Two) {
        // Launch the RIH process asynchronously on a separate thread
        rih_thread = std::thread(&BucketGraph::async_rih_processing, this, merged_labels, LABELS_MAX);
        rih_thread.detach(); // Detach the thread so it runs in the background
    }
#endif

    // Return the final list of merged labels after processing
    return merged_labels;
}

/**
 * Extends the label L_prime with the given BucketArc gamma.
 *
 * @tparam D The direction of the extension (Forward or Backward).
 * @tparam S The stage of the extension (Stage::One, Stage::Two, or Stage::Three).
 * @param L_prime The label to be extended.
 * @param gamma The BucketArc to extend the label with.
 * @return A tuple containing a boolean indicating if the extension was successful and a pointer to the new
 * label.
 */

template <Direction D, Stage S, ArcType A, Mutability M>
inline Label *
BucketGraph::Extend(const std::conditional_t<M == Mutability::Mut, Label *, const Label *>          L_prime,
                    const std::conditional_t<A == ArcType::Bucket, BucketArc,
                                             std::conditional_t<A == ArcType::Jump, JumpArc, Arc>> &gamma) noexcept {
    // Get the forward or backward bucket structures, depending on the direction (D)
    auto &buckets       = assign_buckets<D>(fw_buckets, bw_buckets);
    auto &label_pool    = assign_buckets<D>(label_pool_fw, label_pool_bw);
    auto &fixed_buckets = assign_buckets<D>(fw_fixed_buckets, bw_fixed_buckets);

    // Precompute some values from the current label (L_prime) to avoid recalculating inside the loop
    const int    initial_job_id    = L_prime->job_id;
    auto         initial_resources = L_prime->resources; // Copy the current label's resources
    const double initial_cost      = L_prime->cost;      // Store the initial cost of the label

    int job_id = -1; // Initialize job ID
    // Determine the target job based on the arc type (Bucket, Job, or Jump)
    if constexpr (A == ArcType::Bucket) {
        job_id = buckets[gamma.to_bucket].job_id; // Use the job ID from the bucket
    } else if constexpr (A == ArcType::Job) {
        job_id = gamma.to; // Use the job ID from the arc
    } else if constexpr (A == ArcType::Jump) {
        job_id = buckets[gamma.jump_bucket].job_id; // Use the job ID from the jump bucket
        // Update resources based on jump bucket bounds
        for (size_t i = 0; i < initial_resources.size(); ++i) {
            if constexpr (D == Direction::Forward) {
                initial_resources[i] =
                    std::max(initial_resources[i], static_cast<double>(buckets[gamma.jump_bucket].lb[i]));
            } else {
                initial_resources[i] =
                    std::min(initial_resources[i], static_cast<double>(buckets[gamma.jump_bucket].ub[i]));
            }
        }
    }

    // Check if the arc between initial_job_id and job_id is fixed, and skip if so (in Stage 3)
    if constexpr (S == Stage::Three) {
        if constexpr (D == Direction::Forward) {
            if (fixed_arcs[initial_job_id][job_id] == 1) { return nullptr; }
        } else {
            if (fixed_arcs[job_id][initial_job_id] == 1) { return nullptr; }
        }
    }

    // Check if the job has already been visited for enumeration (Stage Enumerate)
    if constexpr (S == Stage::Enumerate) {
        if (is_job_visited(L_prime->visited_bitmap, job_id)) { return nullptr; }
    }

    // Perform 2-cycle elimination: if the job ID is the same as the current label's job, skip
    if (job_id == L_prime->job_id) { return nullptr; }

    // Check if job_id is in the neighborhood of initial_job_id and has already been visited
    size_t segment      = job_id / 64; // Determine the segment in the bitmap
    size_t bit_position = job_id % 64; // Determine the bit position in the segment
    if constexpr (S != Stage::Enumerate) {
        if ((neighborhoods_bitmap[initial_job_id][segment] & (1ULL << bit_position)) &&
            is_job_visited(L_prime->visited_bitmap, job_id)) {
            return nullptr; // Skip if the job is in the neighborhood and has been visited
        }
    }

    // Get the VRP job corresponding to job_id
    const VRPJob &VRPJob = jobs[job_id];

    // Initialize new resources based on the arc's resource increments
    std::vector<double> new_resources(initial_resources.size());
    for (size_t i = 0; i < initial_resources.size(); ++i) {
        if constexpr (D == Direction::Forward) {
            new_resources[i] =
                std::max(initial_resources[i] + gamma.resource_increment[i], static_cast<double>(VRPJob.lb[i]));
        } else {
            new_resources[i] =
                std::min(initial_resources[i] - gamma.resource_increment[i], static_cast<double>(VRPJob.ub[i]));
        }
    }

    // Ensure the new resources are within the job's resource bounds
    for (size_t i = 0; i < new_resources.size(); ++i) {
        if constexpr (D == Direction::Forward) {
            if (new_resources[i] > VRPJob.ub[i]) { return nullptr; } // Forward: Exceeds upper bound
        } else {
            if (new_resources[i] < VRPJob.lb[i]) { return nullptr; } // Backward: Below lower bound
        }
    }

    // Get the bucket number for the new job and resource state
    int to_bucket = get_bucket_number<D>(job_id, new_resources);

#ifdef FIX_BUCKETS
    // Skip if the bucket is fixed (in Stage 4) and not a jump arc
    if constexpr (S == Stage::Four && A != ArcType::Jump) {
        if (fixed_buckets[L_prime->vertex][to_bucket] == 1) { return nullptr; }
    }
#endif

#ifdef RCC
    // Get dual sum from RCC manager if available
    double cvrpsep_dual = 0.0;
    if constexpr (D == Direction::Forward) {
        cvrpsep_dual = rcc_manager->getCachedDualSumForArc(initial_job_id, job_id);
    } else {
        cvrpsep_dual = rcc_manager->getCachedDualSumForArc(job_id, initial_job_id);
    }
#endif

    // Compute travel cost between the initial and current jobs
    double travel_cost = getcij(initial_job_id, job_id);
    double new_cost    = initial_cost + travel_cost - VRPJob.cost;

#ifdef RCC
    // Adjust the cost using the cached dual sum from RCC (if applicable)
    new_cost -= cvrpsep_dual;
#endif

#ifdef KP_BOUND
    // Apply knapsack bound check in the forward direction (if applicable)
    if constexpr (D == Direction::Forward) {
        auto kpBound = knapsackBound(L_prime);
        if (kpBound > 0.0) {
            return nullptr; // Skip if knapsack bound is exceeded
        }
    }
#endif

    // Acquire a new label from the pool and initialize it with the new state
    auto new_label = label_pool.acquire();
    new_label->initialize(to_bucket, new_cost, new_resources, job_id);
    new_label->visited_bitmap = L_prime->visited_bitmap; // Copy visited bitmap from the original label
    set_job_visited(new_label->visited_bitmap, job_id);  // Mark the new job as visited

#ifdef UNREACHABLE_DOMINANCE
    // Copy unreachable bitmap (if applicable)
    new_label->unreachable_bitmap = L_prime->unreachable_bitmap;
#endif

    // Update real cost (for tracking the total travel cost)
    new_label->real_cost = L_prime->real_cost + travel_cost;

    // Set the parent label, depending on mutability
    if constexpr (M == Mutability::Mut) { new_label->parent = static_cast<const Label *>(L_prime); }

#if defined(SRC3) || defined(SRC)
    // Copy the SRC map from the original label (for SRC cuts)
    new_label->SRCmap = L_prime->SRCmap;
#endif

    // If not in enumeration stage, update visited bitmap to avoid redundant labels
    if constexpr (S != Stage::Enumerate) {
        size_t limit = new_label->visited_bitmap.size();
        for (size_t i = 0; i < limit; ++i) {
            uint64_t current_visited = new_label->visited_bitmap[i];

            if (!current_visited) continue; // Skip if no jobs were visited in this segment

            uint64_t neighborhood_mask = neighborhoods_bitmap[job_id][i]; // Get neighborhood mask for the current job
            uint64_t bits_to_clear     = current_visited & ~neighborhood_mask; // Determine which bits to clear

            if (i == job_id / 64) {
                bits_to_clear &= ~(1ULL << (job_id % 64)); // Ensure current job remains visited
            }

            new_label->visited_bitmap[i] &= ~bits_to_clear; // Clear irrelevant visited jobs
        }
    }

#if defined(SRC3) || defined(SRC)
    // Apply SRC (Subset Row Cuts) logic in Stages 3, 4, and Enumerate
    if constexpr (S == Stage::Three || S == Stage::Four || S == Stage::Enumerate) {
        auto          &cutter   = cut_storage;          // Access the cut storage manager
        auto          &SRCDuals = cutter->SRCDuals;     // Access the dual values for the SRC cuts
        const uint64_t bit_mask = 1ULL << bit_position; // Precompute bit shift for the job's position

        for (std::size_t idx = 0; idx < cutter->size(); ++idx) {
            auto it = cutter->begin();
            std::advance(it, idx);
            const auto &cut          = *it;
            const auto &baseSet      = cut.baseSet;
            const auto &baseSetorder = cut.baseSetOrder;
            const auto &neighbors    = cut.neighbors;
            const auto &multipliers  = cut.multipliers;

#if defined(SRC3)
            // Apply SRC3 logic: if the job is in the base set, increment the SRC map
            bool bitIsSet3 = baseSet[segment] & bit_mask;
            if (bitIsSet3) {
                new_label->SRCmap[idx]++;
                if (new_label->SRCmap[idx] % 2 == 0) { new_label->cost -= SRCDuals[idx]; }
            }
#endif

#if defined(SRC)
            // Apply SRC logic: Update the SRC map based on neighbors and base set
            bool bitIsSet  = neighbors[segment] & bit_mask;
            bool bitIsSet2 = baseSet[segment] & bit_mask;

            double &src_map_value = new_label->SRCmap[idx]; // Use reference to avoid multiple accesses
            if (bitIsSet) {
                src_map_value = L_prime->SRCmap[idx]; // Copy the original label's SRC map value
            } else {
                src_map_value = 0.0; // Reset the SRC map value
            }

            if (bitIsSet2) {
                src_map_value += multipliers[baseSetorder[job_id]];
                if (src_map_value >= 1) {
                    src_map_value -= 1;
                    new_label->cost -= SRCDuals[idx]; // Apply the SRC dual value if threshold is exceeded
                }
            }
#endif
        }
    }
#endif

    // Return the newly created and initialized label
    return new_label;
}

/**
 * @brief Checks if a label is dominated by a new label based on cost and resource conditions.
 *
 * @tparam D The direction of the graph traversal (Forward or Backward).
 * @tparam S The stage of the algorithm (One, Two, or Three).
 * @param new_label A pointer to the new label.
 * @param label A pointer to the label to be checked.
 * @return True if the label is dominated by the new label, false otherwise.
 */
template <Direction D, Stage S>
inline bool BucketGraph::is_dominated(Label *&new_label, Label *&label) noexcept {

    // Extract resources for the new label and the comparison label
    const auto &new_resources   = new_label->resources;
    const auto &label_resources = label->resources;

    double sumSRC = 0; // Variable to accumulate SRC dual values if applicable

#ifdef SRC
    // SRC logic (Subset Row Cuts) for Stage 3, 4, or Enumerate
    if constexpr (S == Stage::Three || S == Stage::Four || S == Stage::Enumerate) {
        const auto &SRCDuals = cut_storage->SRCDuals;
        // Check if the SRC dual values exist
        if (!SRCDuals.empty()) {
            // Iterate over the SRC dual values
            for (size_t i = 0; i < SRCDuals.size(); ++i) {
                // Add to sumSRC if the comparison label's SRC map is greater than the new label's SRC map
                if (label->SRCmap[i] > new_label->SRCmap[i]) { sumSRC += SRCDuals[i]; }
            }
        }
        // If the label's adjusted cost (cost - sumSRC) is greater than the new label's cost, it is not dominated
        if (label->cost - sumSRC > new_label->cost) { return false; }
    } else
#endif
    {
        // Simple cost check: if the comparison label has a higher cost, it is not dominated
        if (label->cost > new_label->cost) { return false; }
    }

    // Check resource conditions based on the direction (Forward or Backward)
    for (size_t i = 0; i < new_resources.size(); ++i) {
        if constexpr (D == Direction::Forward) {
            // In Forward direction: the comparison label must not have more resources than the new label
            if (label_resources[i] > new_resources[i]) { return false; }
        } else if constexpr (D == Direction::Backward) {
            // In Backward direction: the comparison label must not have fewer resources than the new label
            if (label_resources[i] < new_resources[i]) { return false; }
        }
    }

#ifndef UNREACHABLE_DOMINANCE
    // Check visited jobs (bitmap comparison) for Stages 3, 4, and Enumerate
    if constexpr (S == Stage::Three || S == Stage::Four || S == Stage::Enumerate) {
        // Iterate through the visited bitmap and ensure that the new label visits all jobs that the comparison label
        // visits
        for (size_t i = 0; i < label->visited_bitmap.size(); ++i) {
            // If the comparison label visits a job that the new label does not, it is not dominated
            if ((label->visited_bitmap[i] & ~new_label->visited_bitmap[i]) != 0) { return false; }
        }
    }
#else
    // Unreachable dominance logic: check visited and unreachable jobs in Stages 3, 4, and Enumerate
    if constexpr (S == Stage::Three || S == Stage::Four || S == Stage::Enumerate) {
        for (size_t i = 0; i < label->visited_bitmap.size(); ++i) {
            // Combine visited and unreachable jobs in the comparison label's bitmap
            auto combined_label_bitmap = label->visited_bitmap[i] | label->unreachable_bitmap[i];
            // Ensure the new label visits all jobs that the comparison label (or its unreachable jobs) visits
            if ((combined_label_bitmap & ~new_label->visited_bitmap[i]) != 0) { return false; }
        }
    }
#endif

#ifdef SRC3
    // Additional SRC3 logic for Stages 3, 4, and Enumerate
    if constexpr (S == Stage::Three || S == Stage::Four || S == Stage::Enumerate) {
        const auto &SRCDuals = cut_storage->SRCDuals;
        if (!SRCDuals.empty()) {
            sumSRC = 0; // Reset sumSRC for SRC3 logic
            // Iterate through SRC duals
            for (size_t i = 0; i < SRCDuals.size(); ++i) {
                // Check if the comparison label's SRC map is greater in modulo 2 sense
                if ((label->SRCmap[i] % 2) > (new_label->SRCmap[i] % 2)) { sumSRC += SRCDuals[i]; }
            }
            // If the adjusted cost of the comparison label is greater, it is not dominated
            if (label->cost + sumSRC > new_label->cost) { return false; }
        }
    }
#endif

    // If all conditions are met, return true, indicating that the new label is dominated by the comparison label
    return true;
}

/**
 * @brief Checks if element 'a' precedes element 'b' in the given strongly connected components (SCCs).
 *
 * This function takes a vector of SCCs and two elements 'a' and 'b' as input. It searches for 'a' and 'b' in
 * the SCCs and determines if 'a' precedes 'b' in the SCC list.
 *
 * @tparam T The type of elements in the SCCs.
 * @param sccs The vector of SCCs.
 * @param a The element 'a' to check.
 * @param b The element 'b' to check.
 * @return True if 'a' precedes 'b' in the SCC list, false otherwise.
 */
template <typename T>
inline bool precedes(const std::vector<std::vector<T>> &sccs, const T &a, const T &b) {
    auto it_scc_a = sccs.end();
    auto it_scc_b = sccs.end();

    for (auto it = sccs.begin(); it != sccs.end(); ++it) {
        const auto &scc = *it;

        // Use std::find once for each SCC, checking both a and b in the same iteration
        auto it_a = std::find(scc.begin(), scc.end(), a);
        auto it_b = std::find(scc.begin(), scc.end(), b);

        if (it_a != scc.end()) { it_scc_a = it; }
        if (it_b != scc.end()) { it_scc_b = it; }

        // If both are found in the same SCC, return false
        if (it_scc_a == it_scc_b && it_scc_a != sccs.end()) { return false; }

        // Early exit if both are found in different SCCs
        if (it_scc_a != sccs.end() && it_scc_b != sccs.end()) { return it_scc_a < it_scc_b; }
    }

    // If a and/or b are not found, return false
    return false;
}

/**
 * @brief Determines if a label is dominated in component-wise smaller buckets.
 *
 * This function checks if a given label is dominated by any other label in component-wise smaller buckets.
 * The dominance is determined based on the cost and order of the buckets.
 *
 * @tparam D The direction of the buckets.
 * @tparam S The stage of the buckets.
 * @param L A pointer to the label to be checked.
 * @param bucket The index of the current bucket.
 * @param c_bar The vector of cost values for each bucket.
 * @param Bvisited The set of visited buckets.
 * @param bucket_order The order of the buckets.
 * @return True if the label is dominated, false otherwise.
 */
template <Direction D, Stage S>
inline bool BucketGraph::DominatedInCompWiseSmallerBuckets(Label *L, int bucket, std::vector<double> &c_bar,
                                                           std::vector<uint64_t>               &Bvisited,
                                                           const std::vector<std::vector<int>> &bucket_order) noexcept {
    // Assign the appropriate buckets and Phi structures based on direction (D)
    auto &buckets = assign_buckets<D>(fw_buckets, bw_buckets);
    auto &Phi     = assign_buckets<D>(Phi_fw, Phi_bw);

    const int       b_L = L->vertex; // The vertex (bucket) associated with the label L
    std::stack<int> bucketStack;     // Stack to manage the traversal of buckets
    bucketStack.push(bucket);        // Start with the input bucket

    // Traverse the graph of buckets in a depth-first manner
    while (!bucketStack.empty()) {
        int currentBucket = bucketStack.top(); // Get the bucket at the top of the stack
        bucketStack.pop();                     // Remove it from the stack

        // Mark the current bucket as visited by updating the Bvisited bitmask
        const size_t segment      = currentBucket / 64; // Determine the segment for the current bucket
        const size_t bit_position = currentBucket % 64; // Determine the bit position within the segment
        Bvisited[segment] |= (1ULL << bit_position);    // Set the bit corresponding to the current bucket as visited

        // Check if the label's cost is lower than the cost bound (c_bar) for the current bucket
        // and if the current bucket precedes the label's bucket according to the bucket order
        if (L->cost < c_bar[currentBucket] && ::precedes<int>(bucket_order, currentBucket, b_L)) {
            return false; // The label is not dominated, return false early
        }

        // If the current bucket is different from the label's bucket, compare the labels in this bucket
        if (b_L != currentBucket) {
            const auto &bucket_labels = buckets[currentBucket].get_labels(); // Get the labels in the current bucket
            // Iterate over each label in the current bucket and check if it dominates L
            for (auto *label : bucket_labels) {
                if (is_dominated<D, S>(L, label)) {
                    return true; // If any label dominates L, return true
                }
            }
        }

        // Add the neighboring buckets (from Phi) to the stack if they haven't been visited yet
        for (const int b_prime : Phi[currentBucket]) {
            const size_t segment_prime      = b_prime / 64; // Determine the segment for the neighboring bucket
            const size_t bit_position_prime = b_prime % 64; // Determine the bit position within the segment

            // If the neighboring bucket hasn't been visited, push it onto the stack
            if ((Bvisited[segment_prime] & (1ULL << bit_position_prime)) == 0) { bucketStack.push(b_prime); }
        }
    }

    // If no domination was found in any smaller bucket, return false
    return false;
}
