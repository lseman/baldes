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

#include "BucketRes.h"

#include "Trees.h"
#include "cuts/SRC.h"
#include <cstring>

#ifdef __AVX2__
#include "BucketAVX.h"
#endif

#include "../third_party/small_vector.hpp"

/**
 * Solves the bucket graph optimization problem using a multi-stage bi-labeling algorithm.
 *
 * The function iteratively adjusts the terminal time and handles different stages of the
 * bi-labeling algorithm to find the optimal paths. The stages are adaptive and transition
 * based on the inner objective values and iteration counts.
 *
 */
template <Symmetry SYM>
inline std::vector<Label *> BucketGraph::solve(bool trigger) {
    // Initialize the status as not optimal at the start
    status = Status::NotOptimal;
    if (trigger) {
        transition = true;
        fixed      = false;
    }

    updateSplit(); // Update the split values for the bucket graph

    // Placeholder for the final paths (labels) and inner objective value
    std::vector<Label *> paths;

    //////////////////////////////////////////////////////////////////////
    // ADAPTIVE STAGE HANDLING
    //////////////////////////////////////////////////////////////////////
    // Stage 1: Apply a light heuristic (Stage::One)
    if (s1 && depth == 0) {
        stage = 1;
        paths = bi_labeling_algorithm<Stage::One>(); // Solve the problem with Stage 1 heuristic
        // inner_obj = paths[0]->cost;

        // Transition from Stage 1 to Stage 2 if the objective improves or after 10 iterations
        if (inner_obj >= -1 || iter >= 10) {
            s1 = false;
            s2 = true; // Move to Stage 2
        }
    }
    // Stage 2: Apply a more expensive pricing heuristic (Stage::Two)
    else if (s2 && depth == 0) {
        s2    = true;
        stage = 2;
        paths = bi_labeling_algorithm<Stage::Two>(); // Solve the problem with Stage 2 heuristic
        // inner_obj = paths[0]->cost;

        // Transition from Stage 2 to Stage 3 if the objective improves or after 800 iterations
        if (inner_obj >= -100 || iter > 500) {
            s2 = false;
            s3 = true; // Move to Stage 3
        }
    }
    // Stage 3: Apply a heuristic fixing approach (Stage::Three)
    else if (s3 && depth == 0) {
        stage = 3;
        paths = bi_labeling_algorithm<Stage::Three>(); // Solve the problem with Stage 3 heuristic
        // inner_obj = paths[0]->cost;

        // Transition from Stage 3 to Stage 4 if the objective improves significantly
        if (inner_obj >= -0.5) {
            s4         = true;
            s3         = false;
            transition = true; // Prepare for transition to Stage 4
        }
    }
    // Stage 4: Exact labeling algorithm with fixing enabled (Stage::Four)
    else {
        stage = 4;

#ifdef FIX_BUCKETS
        // If transitioning to Stage 4, print a message and apply fixing
        if (transition) {
            // print_cut("Transitioning to stage 4\n");
            bool original_fixed = fixed;                                // Store the original fixed status
            fixed               = true;                                 // Enable fixing of buckets in Stage 4
            paths               = bi_labeling_algorithm<Stage::Four>(); // Solve the problem with Stage 4
            transition          = false;                                // End the transition period
            fixed               = original_fixed;                       // Restore the original fixed status
            min_red_cost        = paths[0]->cost;                       // Update the minimum reduced cost
            iter++;
            return paths; // Return the final paths
        }
#endif
        // If not transitioning, continue with Stage 4 algorithm
        paths = bi_labeling_algorithm<Stage::Four>();
        // inner_obj = paths[0]->cost;

        auto rollback = updateStepSize(); // Update the step size for the bucket graph
        // auto rollback = false;
        //  auto rollback = false;
        if (rollback) {
            //    s4     = false; // Rollback to Stage 3 if necessary
            //    s3     = true;
            status = Status::Rollback; // Set status to rollback
            return paths;              // Return the paths after rollback
        }
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

inline std::vector<Label *> BucketGraph::solveHeuristic() {
    // Initialize the status as not optimal at the start
    status = Status::NotOptimal;

    updateSplit(); // Update the split values for the bucket graph

    // Placeholder for the final paths (labels) and inner objective value
    std::vector<Label *> paths;
    double               inner_obj;

    //////////////////////////////////////////////////////////////////////
    // ADAPTIVE STAGE HANDLING
    //////////////////////////////////////////////////////////////////////
    // Stage 1: Apply a light heuristic (Stage::One)
    if (s1) {
        stage     = 1;
        paths     = bi_labeling_algorithm<Stage::One>(); // Solve the problem with Stage 1 heuristic
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
        paths     = bi_labeling_algorithm<Stage::Two>(); // Solve the problem with Stage 2 heuristic
        inner_obj = paths[0]->cost;

        // Transition from Stage 2 to Stage 3 if the objective improves or after 800 iterations
        if (inner_obj >= -100 || iter > 800) { status = Status::Optimal; }
    }

    iter++; // Increment the iteration counter

    return paths; // Return the final paths after processing
}

/**
 * Performs the labeling algorithm on the BucketGraph.
 *
 */
template <Direction D, Stage S, Full F>
std::vector<double> BucketGraph::labeling_algorithm() {

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

    auto a_ctr = 0;
    // Iterate through each strongly connected component (SCC) in topological order
    for (const auto &scc_index : topological_order) {
        do {
            all_ext = true; // Assume all labels have been extended at the start
            for (const auto bucket : sorted_sccs[scc_index]) {
                auto bucket_labels = buckets[bucket].get_labels(); // Get unextended labels from this bucket
                // if (bucket_labels.empty()) { continue; }           // Skip empty buckets
                for (Label *label : bucket_labels) {
                    if (label->is_extended) { continue; }

                    if constexpr (F != Full::PSTEP && F != Full::TSP) {
                        if constexpr (F == Full::Partial) {
                            if constexpr (D == Direction::Forward) {
                                if (label->resources[options.main_resources[0]] >
                                    q_star[options.main_resources[0]]) { // + numericutils::eps) {
                                    label->set_extended(true);
                                    continue;
                                }
                            } else if constexpr (D == Direction::Backward) {
                                if (label->resources[options.main_resources[0]] <=
                                    q_star[options.main_resources[0]]) { // - numericutils::eps) {
                                    label->set_extended(true);
                                    continue;
                                }
                            }
                        }
                    }

                    domin_smaller = false;

                    // Clear visited buckets efficiently
                    if (n_segments <= 8) {
                        for (size_t i = 0; i < n_segments; ++i) { Bvisited[i] = 0; }
                    } else {
                        std::memset(Bvisited.data(), 0, n_segments * sizeof(uint64_t));
                    }

                    // Check if the label is dominated by any labels in smaller buckets
                    if constexpr (F != Full::TSP) {
                        domin_smaller =
                            DominatedInCompWiseSmallerBuckets<D, S>(label, bucket, c_bar, Bvisited, ordered_sccs);
                    }

                    if (!domin_smaller) {
                        // Lambda function to process new labels after extension
                        auto process_new_label = [&](Label *new_label) {
                            stat_n_labels++; // Increment number of labels processed

                            int &to_bucket = new_label->vertex; // Get the bucket to which the new label belongs
                            dominated      = false;
                            const auto &to_bucket_labels =
                                buckets[to_bucket].get_labels(); // Get existing labels in the destination bucket

                            if constexpr (F != Full::PSTEP && F != Full::TSP) {
                                if constexpr (S == Stage::Four) {
                                    // Track dominance checks for this bucket
                                    if constexpr (D == Direction::Forward) {
                                        dominance_checks_per_bucket[to_bucket] += to_bucket_labels.size();
                                    }
                                }
                            }
                            // Stage-specific dominance check
                            if constexpr (S == Stage::One) {
                                // If the new label has lower cost, remove dominated labels
                                for (auto *existing_label : to_bucket_labels) {
                                    if (label->cost < existing_label->cost) {

                                        // TODO: check if removing children is a good idea
                                        // Use a stack to track labels whose children need to be removed
                                        std::stack<Label *> stack;
                                        stack.push(existing_label);

                                        // Process all children recursively using the stack
                                        while (!stack.empty()) {
                                            Label *current_label = stack.top();
                                            stack.pop();

                                            // Remove all children of the current label
                                            for (auto *child : current_label->children) {
                                                buckets[child->vertex].remove_label(child);
                                                stack.push(child); // Add child's children to the stack
                                            }

                                            // Remove the current label itself
                                            buckets[current_label->vertex].remove_label(current_label);
                                        }
                                    } else {
                                        dominated = true;
                                        break;
                                    }
                                }
                            } else {

                                auto &mother_bucket             = buckets[to_bucket];
                                auto  check_dominance_in_bucket = [&](const std::vector<Label *> &labels) {
#ifndef __AVX2__
                                    for (auto *existing_label : labels) {
                                        if (is_dominated<D, S>(new_label, existing_label)) {
                                            stat_n_dom++; // Increment dominated labels count
                                            return true;
                                        }
                                    }
                                    return false;
#else
                                    return check_dominance_against_vector<D, S>(new_label, labels, cut_storage,
                                                                                options.resources.size());
#endif
                                };

                                // Call check_dominance on the mother bucket
                                dominated =
                                    mother_bucket.check_dominance(new_label, check_dominance_in_bucket, stat_n_dom);
                            }

                            if (!dominated) {
                                // Remove dominated labels from the bucket
                                if constexpr (S != Stage::Enumerate) {
                                    std::vector<Label *> labels_to_remove;
                                    for (auto *existing_label : to_bucket_labels) {
                                        if (is_dominated<D, S>(existing_label, new_label)) {
                                            labels_to_remove.push_back(existing_label);
                                        }
                                    }
                                    // Now remove all marked labels in one pass
                                    for (auto *label : labels_to_remove) { buckets[to_bucket].remove_label(label); }
                                }

                                n_labels++; // Increment the count of labels added

                                // Add the new label to the bucket
#ifdef SORTED_LABELS
                                buckets[to_bucket].add_sorted_label(new_label);
#elif LIMITED_BUCKETS
                                buckets[to_bucket].sorted_label(new_label, BUCKET_CAPACITY);
#else
                                buckets[to_bucket].add_label(new_label);
#endif
                                all_ext = false; // Not all labels have been extended, continue processing
                            }
                        };

                        // Process regular arcs for label extension
                        const auto &arcs = nodes[label->node_id].get_arcs<D>(scc_index);
                        for (const auto &arc : arcs) {
                            auto new_labels = Extend<D, S, ArcType::Node, Mutability::Mut, F>(label, arc);
                            if (new_labels.empty()) {
#ifdef UNREACHABLE_DOMINANCE
                                set_node_unreachable(label->unreachable_bitmap, arc.to);
#endif
                            } else {
                                for (auto label_ptr : new_labels) {
                                    process_new_label(label_ptr); // Process the new label
                                }
                            }
                        }
                    }

                    label->set_extended(true); // Mark the label as extended
                }
            }
        } while (!all_ext); // Continue until all labels have been extended

        // Update the cost bounds (c_bar) for the current SCC's buckets
        for (const int bucket : sorted_sccs[scc_index]) {
            // parallelize this

            // std::for_each(std::execution::par_unseq, sorted_sccs[scc_index].begin(), sorted_sccs[scc_index].end(),
            //   [&](int bucket) {
            // Update c_bar[bucket] with the minimum cost found in the bucket
            double current_cb = buckets[bucket].get_cb();
            c_bar[bucket]     = std::min(c_bar[bucket], current_cb);

            // Update the cost bound for dependencies in Phi[bucket]
            for (auto phi_bucket : Phi[bucket]) {
                int min_value = std::min(c_bar[bucket], c_bar[phi_bucket]);
                c_bar[bucket] = min_value;
            }
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
 */
template <Stage S, Symmetry SYM>
std::vector<Label *> BucketGraph::bi_labeling_algorithm() {

    // If in Stage 3, apply heuristic fixing based on q_star
    if constexpr (S == Stage::Three) {
        heuristic_fixing<S>();
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
    if constexpr (S == Stage::Four) {
        if (options.bucket_fixing) { bucket_fixing<S>(); }
    }
#endif

    // Reset the label pool to ensure no leftover labels from previous runs
    reset_pool();
    // Perform any common initializations (data structures, etc.)
    common_initialization();

    // Initialize the cost bound vectors for forward and backward buckets
    std::vector<double> forward_cbar(fw_buckets.size());
    std::vector<double> backward_cbar(bw_buckets.size());

    // Run the labeling algorithm in both directions, but only partially (Full::Partial)
    if constexpr (SYM == Symmetry::Asymmetric) {
        run_labeling_algorithms<S, Full::Partial>(forward_cbar, backward_cbar);
    } else {
        forward_cbar = labeling_algorithm<Direction::Forward, S, Full::Partial>();
    }

    // Acquire the best label from the forward label pool (will later combine with backward)
    auto best_label = label_pool_fw->acquire();

    if constexpr (SYM == Symmetry::Asymmetric) {
        // Check if the best forward and backward labels can be combined into a feasible solution
        if (check_feasibility(fw_best_label, bw_best_label)) {
            // If feasible, compute and combine the best forward and backward labels into one
            best_label = compute_label<S>(fw_best_label, bw_best_label);
        } else {
            // If not feasible, set the best label to have infinite cost (not usable)
            best_label->cost          = 0.0;
            best_label->real_cost     = std::numeric_limits<double>::infinity();
            best_label->nodes_covered = {};
        }
    } else {
        best_label->cost          = best_label->real_cost;
        best_label->real_cost     = std::numeric_limits<double>::infinity();
        best_label->nodes_covered = {};
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
        auto      &current_bucket = fw_buckets[bucket];          // Get the current bucket
        const auto labels         = current_bucket.get_labels(); // Get labels in the current bucket
        //
        if constexpr (S == Stage::Four) {
            non_dominated_labels_per_bucket += labels.size(); // Track non-dominated labels
        }
        // Process each label in the bucket
        for (const Label *L : labels) {
            // if (L->resources[TIME_INDEX] > q_star[TIME_INDEX]) { continue; } // Skip if label exceeds q_star

            // Get arcs corresponding to nodes for this label (Forward direction)
            const auto &to_arcs = nodes[L->node_id].get_arcs<Direction::Forward>();
            // Iterate over each arc from the current node
            for (const auto &arc : to_arcs) {
                const auto &to_node = arc.to;

                // Skip fixed arcs in Stage 3 if necessary
                if constexpr (S == Stage::Three || S == Stage::Eliminate) {
                    if (fixed_arcs[L->node_id][to_node] == 1) {
                        continue; // Skip if the arc is fixed
                    }
                }

                // Attempt to extend the current label using this arc
                auto extended_labels =
                    Extend<Direction::Forward, S, ArcType::Node, Mutability::Const, Full::Reverse>(L, arc);

                // Note: apparently without the second condition it work better in some cases
                // Check if the new label is valid and respects the q_star constraints
                if (extended_labels.empty()) {
                    continue; // Skip invalid labels or those that exceed q_star
                }

                // Iterate over the returned Label** array and add each valid label to the vector
                for (auto L_prime : extended_labels) {
                    // auto L_prime = *label_ptr; // Get the current label from the array
                    if (L_prime->resources[options.main_resources[0]] <= q_star[options.main_resources[0]]) {
                        continue; // Skip if the label exceeds q_star
                    }

                    // Get the bucket for the extended label
                    auto b_prime = L_prime->vertex;

                    // Clear visited buckets tracking for this new label extension
                    std::memset(Bvisited.data(), 0, Bvisited.size() * sizeof(uint64_t));

                    // Concatenate this new label with the best label found so far
                    ConcatenateLabel<S, SYM>(L, b_prime, best_label, Bvisited);
                }
            }
        }
    }

    // Sort the merged labels by cost, to prioritize cheaper labels
    pdqsort(merged_labels.begin(), merged_labels.end(),
            [](const Label *a, const Label *b) { return a->cost < b->cost; });

#ifdef SCHRODINGER
    // if merged_labels is bigger than 10, create Path related to the remaining ones
    // and add them to a std::vector<Path>
    if (merged_labels.size() > N_ADD) {
        std::vector<Path> paths;
        int               labels_size = merged_labels.size();
        for (size_t i = N_ADD; i < std::min(N_ADD + N_ADD, labels_size); ++i) {
            if (merged_labels[i]->nodes_covered.size() <= 3) { continue; }
            Path path = Path(merged_labels[i]->nodes_covered, merged_labels[i]->real_cost);
            paths.push_back(path);
        }
        sPool.add_paths(paths);
        sPool.iterate();
    }
#endif

    // Return the final list of merged labels after processing

#ifdef RIH
    std::vector<Label *> top_labels;
    top_labels.reserve(5); // Reserve memory for 5 elements
    for (size_t i = 0; i < std::min(5, static_cast<int>(merged_labels.size())); ++i) {
        top_labels.push_back(merged_labels[i]);
    }

    // auto new_labels = ils->perturbation(top_labels, nodes);
    ils->submit_task(top_labels, nodes);

#endif
    inner_obj = merged_labels[0]->cost;

    return merged_labels;
}

/**
 * Extends the label L_prime with the given BucketArc gamma.
 *
 */
template <Direction D, Stage S, ArcType A, Mutability M, Full F>
inline std::vector<Label *>
BucketGraph::Extend(const std::conditional_t<M == Mutability::Mut, Label *, const Label *>          L_prime,
                    const std::conditional_t<A == ArcType::Bucket, BucketArc,
                                             std::conditional_t<A == ArcType::Jump, JumpArc, Arc>> &gamma,
                    int                                                                             depth) noexcept {
    // Get the forward or backward bucket structures, depending on the direction (D)
    auto &buckets       = assign_buckets<D>(fw_buckets, bw_buckets);
    auto &label_pool    = assign_buckets<D>(label_pool_fw, label_pool_bw);
    auto &fixed_buckets = assign_buckets<D>(fw_fixed_buckets, bw_fixed_buckets);

    // Precompute some values from the current label (L_prime) to avoid recalculating inside the loop
    const int    initial_node_id   = L_prime->node_id;
    auto         initial_resources = L_prime->resources; // Copy the current label's resources
    const double initial_cost      = L_prime->cost;      // Store the initial cost of the label

    int node_id = -1; // Initialize node ID
    // Determine the target node based on the arc type (Bucket, Node, or Jump)
    if constexpr (A == ArcType::Bucket) {
        node_id = buckets[gamma.to_bucket].node_id; // Use the node ID from the bucket
    } else if constexpr (A == ArcType::Node) {
        node_id = gamma.to; // Use the node ID from the arc
    } else if constexpr (A == ArcType::Jump) {
        node_id = buckets[gamma.jump_bucket].node_id; // Use the node ID from the jump bucket
        // Update resources based on jump bucket bounds
        for (size_t i = 0; i < options.resources.size(); ++i) {
            if constexpr (D == Direction::Forward) {
                initial_resources[i] =
                    std::max(initial_resources[i], static_cast<double>(buckets[gamma.jump_bucket].lb[i]));
            } else {
                initial_resources[i] =
                    std::min(initial_resources[i], static_cast<double>(buckets[gamma.jump_bucket].ub[i]));
            }
        }
    }

    // Check if the arc between initial_node_id and node_id is fixed, and skip if so (in Stage 3)
    if constexpr (S == Stage::Three || S == Stage::Eliminate) {
        if constexpr (D == Direction::Forward) {
            if (fixed_arcs[initial_node_id][node_id] == 1) { return std::vector<Label *>(); }
        } else {
            if (fixed_arcs[node_id][initial_node_id] == 1) { return std::vector<Label *>(); }
        }
    }

    // Check if the node has already been visited for enumeration (Stage Enumerate)
    if (is_node_visited(L_prime->visited_bitmap, node_id)) { return std::vector<Label *>(); }

    // Perform 2-cycle elimination: if the node ID is the same as the current label's node, skip
    if (node_id == L_prime->node_id) { return std::vector<Label *>(); }

    // Check if node_id is in the neighborhood of initial_node_id and has already been visited
    size_t segment      = node_id >> 6; // Determine the segment in the bitmap
    size_t bit_position = node_id & 63; // Determine the bit position in the segment

    // Get the VRP node corresponding to node_id
    const VRPNode &VRPNode = nodes[node_id];

    // Initialize new resources based on the arc's resource increments and check feasibility
    std::vector<double> new_resources(options.resources.size());

    int n_visited = 0;
    if constexpr (F != Full::TSP) {
        //  Note: workaround
        size_t N = options.resources.size();
        if (!process_all_resources<D>(new_resources, initial_resources, gamma, VRPNode, N)) {
            return std::vector<Label *>(); // Handle failure case (constraint violation)
        }
    }
    if constexpr (F == Full::PSTEP || F == Full::TSP) {
        // counter the number of bits set in L_prime->visited_bitmap
        for (size_t i = 0; i < L_prime->visited_bitmap.size(); ++i) {
            n_visited += __builtin_popcountll(L_prime->visited_bitmap[i]);
        }

        if (n_visited > options.max_path_size) { return std::vector<Label *>(); }

        if (n_visited == options.max_path_size) {
            if (node_id != options.end_depot) { return std::vector<Label *>(); }
        }
    }

    // Get the bucket number for the new node and resource state
    int to_bucket = get_bucket_number<D>(node_id, new_resources);

#ifdef FIX_BUCKETS
    // Skip if the bucket is fixed (in Stage 4) and not a jump arc
    if constexpr (S == Stage::Four && A != ArcType::Jump) {
        if (fixed_buckets[L_prime->vertex][to_bucket] == 1) {
            if (depth > 1) { return std::vector<Label *>(); } // Skip if the bucket is fixed

            // Get jump arcs for the current node
            auto jump_arcs = nodes[L_prime->node_id].template get_jump_arcs<D>(node_id);

            // Use std::vector for dynamic label handling
            std::vector<Label *> label_vector;

            // Process each jump arc
            for (const auto &jump_arc : jump_arcs) {
                // Try to extend the label
                auto extended_labels = Extend<D, S, ArcType::Jump, Mutability::Const, F>(L_prime, jump_arc, depth + 1);

                if (extended_labels.empty()) {
                    // If extension fails, continue to the next arc
                    continue;
                }
                for (auto label_ptr : extended_labels) { label_vector.push_back(label_ptr); }
            }

            if (label_vector.empty()) { return std::vector<Label *>(); }

            // Return the vector as a dynamically sized array
            return label_vector; // std::vector provides direct access to its underlying array
        }
    }
#endif

    // print initial_node_id and node_id
    // Compute travel cost between the initial and current nodes
    const double travel_cost = getcij(initial_node_id, node_id);
    double       new_cost    = 0.0;
    if constexpr (F != Full::PSTEP) {
        new_cost = initial_cost + travel_cost - VRPNode.cost;
    } else {
        new_cost = initial_cost + travel_cost;
        // consider pstep duals
        auto arc_dual      = pstep_duals.getArcDualValue(initial_node_id, node_id); // eq (3.5)
        auto last_dual     = pstep_duals.getThreeTwoDualValue(node_id);             // eq (3.2)
        auto not_last_dual = pstep_duals.getThreeThreeDualValue(node_id);           // eq (3.3)

        auto old_last_dual     = pstep_duals.getThreeTwoDualValue(initial_node_id);
        auto old_not_last_dual = pstep_duals.getThreeThreeDualValue(initial_node_id);

        // since there was a new node visited, we give back the dual as a final node and add the dual corresponding to
        // the visit
        if (n_visited > 1 && initial_node_id != options.depot) { new_cost += old_last_dual + old_not_last_dual; }
        // suppose this is the last
        new_cost += -1.0 * last_dual;
        // Add the arc dual
        new_cost += arc_dual;
        // print the duals
    }

    // Compute branching duals
    if (branching_duals->size() > 0) {

        if constexpr (D == Direction::Forward) {
            new_cost += branching_duals->getDual(initial_node_id, node_id);
        } else {
            new_cost += branching_duals->getDual(node_id, initial_node_id);
        }
        auto b_duals = branching_duals->getDual(node_id);
        new_cost += b_duals;
    }

    RCC_MODE_BLOCK(if constexpr (S == Stage::Four) {
        if constexpr (D == Direction::Forward) {
            auto arc_dual = arc_duals.getDual(initial_node_id, node_id);
            new_cost -= arc_dual;
        } else if constexpr (D == Direction::Backward) {
            auto arc_dual = arc_duals.getDual(node_id, initial_node_id);
            new_cost -= arc_dual;
        }
    })

    // Acquire a new label from the pool and initialize it with the new state
    auto new_label = label_pool->acquire();
    new_label->initialize(to_bucket, new_cost, new_resources, node_id);
    new_label->vertex = to_bucket;

    // Lambda to create a Label** array with null-termination using the label pool
    // Function returning a single Label* in a vector (which can act like a Label**)
    auto create_label_array = [](Label *label) -> std::vector<Label *> {
        std::vector<Label *> label_array(1, nullptr); // Creates a vector with 1 element, initialized to nullptr
        label_array[0] = label;                       // Add the single label to the vector
        return label_array;
    };

    if constexpr (F == Full::Reverse) {
        return create_label_array(new_label); // Return the array (as Label**)
    }

#ifdef UNREACHABLE_DOMINANCE
    // Copy unreachable bitmap (if applicable)
    new_label->unreachable_bitmap = L_prime->unreachable_bitmap;
#endif

    // Update real cost (for tracking the total travel cost)
    new_label->real_cost = L_prime->real_cost + travel_cost;

    // Set the parent label, depending on mutability
    if constexpr (M == Mutability::Mut) {
        new_label->parent = L_prime;
        L_prime->children.push_back(new_label);
    }

    if constexpr (F != Full::PSTEP) {
        // If not in enumeration stage, update visited bitmap to avoid redundant labels
        if constexpr (S != Stage::Enumerate) {
            size_t limit = new_label->visited_bitmap.size();
            for (size_t i = 0; i < limit; ++i) {
                uint64_t current_visited = L_prime->visited_bitmap[i];

                if (!current_visited) continue; // Skip if no nodes were visited in this segment

                uint64_t neighborhood_mask =
                    neighborhoods_bitmap[node_id][i]; // Get neighborhood mask for the current node
                uint64_t bits_to_clear = current_visited & neighborhood_mask; // Determine which bits to clear

                new_label->visited_bitmap[i] = bits_to_clear; // Clear irrelevant visited nodes
            }
        }
    } else {
        new_label->visited_bitmap = L_prime->visited_bitmap;
    }
    set_node_visited(new_label->visited_bitmap, node_id); // Mark the new node as visited
    new_label->nodes_covered = L_prime->nodes_covered;    // Copy the list of covered nodes
    new_label->nodes_covered.push_back(node_id);          // Add the new node to the list of covered nodes

#if defined(SRC)
    new_label->SRCmap = L_prime->SRCmap;
    // Apply SRC (Subset Row Cuts) logic in Stages 3, 4, and Enumerate
    if constexpr (S == Stage::Four || S == Stage::Enumerate) {
        auto          &cutter   = cut_storage;
        auto          &SRCDuals = cutter->SRCDuals;
        const uint64_t bit_mask = 1ULL << bit_position;

#if defined(SRC)
        // Generate indices for parallel processing
#if defined(__cpp_lib_ranges)
        auto indices = std::views::iota(size_t{0}, cutter->size());
#else
        std::vector<size_t> indices(cutter->size());
        std::iota(indices.begin(), indices.end(), 0);
#endif

        // Lambda to process each cut and return cost update
        auto process_cut = [&](size_t idx) -> double {
            if (SRCDuals[idx] > -1e-3) { return 0.0; }

            const auto &cut           = cutter->getCut(idx);
            const bool  bitIsSet      = cut.neighbors[segment] & bit_mask;
            auto       &src_map_value = new_label->SRCmap[idx];

            if (!bitIsSet) {
                src_map_value = 0.0;
                return 0.0;
            }

            const bool bitIsSet2 = cut.baseSet[segment] & bit_mask;

            if (bitIsSet2) {
                const auto &multipliers = cut.p;
                const auto &den         = multipliers.den;
                src_map_value += multipliers.num[cut.baseSetOrder[node_id]];

                if (src_map_value >= den) {
                    src_map_value -= den;
                    return -SRCDuals[idx];
                }
            }

            return 0.0;
        };

#if defined(__cpp_lib_parallel_algorithm)
        // Parallel process all cuts and accumulate cost updates
        const double total_cost_update = std::transform_reduce(std::execution::par_unseq, indices.begin(),
                                                               indices.end(), 0.0, std::plus<>(), process_cut);
#else
        const double total_cost_update =
            std::transform_reduce(indices.begin(), indices.end(), 0.0, std::plus<>(), process_cut);
#endif

        // Apply the total cost update
        new_label->cost += total_cost_update;
#endif
    }
#endif

    // Usage of the lambda function
    auto result = create_label_array(new_label);

    // Return the array (as Label**)
    return result;
}

/**
 * @brief Checks if a label is dominated by a new label based on cost and resource conditions.
 *
 */
template <Direction D, Stage S>
inline bool BucketGraph::is_dominated(const Label *new_label, const Label *label) noexcept {

    // Extract resources for the new label and the comparison label
    const auto &new_resources   = new_label->resources;
    const auto &label_resources = label->resources;

    // Simple cost check: if the comparison label has a higher cost, it is not dominated
    if (label->cost > new_label->cost) { return false; }

    // Check resource conditions based on the direction (Forward or Backward)
    for (size_t i = 0; i < options.resources.size(); ++i) {
        if constexpr (D == Direction::Forward) {
            // In Forward direction: the comparison label must not have more resources than the new label
            if (label_resources[i] > new_resources[i]) { return false; }
        } else if constexpr (D == Direction::Backward) {
            // In Backward direction: the comparison label must not have fewer resources than the new label
            if (label_resources[i] < new_resources[i]) { return false; }
        }
    }

    // TODO:: check again this unreachable dominance
#ifndef UNREACHABLE_DOMINANCE
    // Check visited nodes (bitmap comparison) for Stages 3, 4, and Enumerate
    if constexpr (S == Stage::Three || S == Stage::Four || S == Stage::Enumerate) {
        // Iterate through the visited bitmap and ensure that the new label visits all nodes that the comparison
        // label visits
        for (size_t i = 0; i < label->visited_bitmap.size(); ++i) {
            // If the comparison label visits a node that the new label does not, it is not dominated
            if (((label->visited_bitmap[i] & new_label->visited_bitmap[i]) ^ label->visited_bitmap[i]) != 0) {
                return false;
            }
        }
    }
#else
    // Unreachable dominance logic: check visited and unreachable nodes in Stages 3, 4, and Enumerate
    if constexpr (S == Stage::Three || S == Stage::Four || S == Stage::Enumerate) {
        for (size_t i = 0; i < label->visited_bitmap.size(); ++i) {
            // Combine visited and unreachable nodes in the comparison label's bitmap
            auto combined_label_bitmap = label->visited_bitmap[i] | label->unreachable_bitmap[i];
            // Ensure the new label visits all nodes that the comparison label (or its unreachable nodes) visits
            if ((combined_label_bitmap & ~new_label->visited_bitmap[i]) != 0) { return false; }
        }
    }
#endif

    SRC_MODE_BLOCK(if constexpr (S == Stage::Four || S == Stage::Enumerate) {
        double sumSRC = 0;

        const auto &SRCDuals = cut_storage->SRCDuals;
        if (!SRCDuals.empty()) {
            const auto &labelSRCMap    = label->SRCmap;
            const auto &newLabelSRCMap = new_label->SRCmap;
            for (size_t i = 0; i < SRCDuals.size(); ++i) {
                const auto &den         = cut_storage->getCut(i).p.den;
                const auto  labelMod    = labelSRCMap[i];    // % den;
                const auto  newLabelMod = newLabelSRCMap[i]; // % den;
                if (labelMod > newLabelMod) { sumSRC += SRCDuals[i]; }
            }
        }
        if (label->cost - sumSRC > new_label->cost) { return false; }
    })

    // If all conditions are met, return true, indicating that the new label is dominated by the comparison label
    return true;
}

/**
 * @brief Checks if element 'a' precedes element 'b' in the given strongly connected components (SCCs).
 *
 * This function takes a vector of SCCs and two elements 'a' and 'b' as input. It searches for 'a' and 'b' in
 * the SCCs and determines if 'a' precedes 'b' in the SCC list.
 *
 */
template <typename T>
inline bool precedes(const std::vector<std::vector<int>> &sccs, const T a, const T b, UnionFind &uf) {
    // Step 2: Check if element `a`'s SCC precedes element `b`'s SCC
    size_t rootA = uf.getSubset(a);
    size_t rootB = uf.getSubset(b);

    // You can define precedence based on the comparison of the root components
    return rootA < rootB;
}
/**
 * @brief Determines if a label is dominated in component-wise smaller buckets.
 *
 * This function checks if a given label is dominated by any other label in component-wise smaller buckets.
 * The dominance is determined based on the cost and order of the buckets.
 *
 */
template <Direction D, Stage S>
inline bool BucketGraph::DominatedInCompWiseSmallerBuckets(const Label *L, int bucket, const std::vector<double> &c_bar,
                                                           std::vector<uint64_t>               &Bvisited,
                                                           const std::vector<std::vector<int>> &bucket_order) noexcept {
    // Assign the appropriate buckets and Phi structures based on direction (D)
    auto &buckets = assign_buckets<D>(fw_buckets, bw_buckets);
    auto &Phi     = assign_buckets<D>(Phi_fw, Phi_bw);

    const int        b_L = L->vertex; // The vertex (bucket) associated with the label L
    std::vector<int> bucketStack;     // Stack to manage the traversal of buckets
    bucketStack.reserve(10);
    bucketStack.push_back(bucket); // Start with the input bucket

    auto &uf = assign_buckets<D>(fw_union_find, bw_union_find);
    // Traverse the graph of buckets in a depth-first manner
    while (!bucketStack.empty()) {
        int currentBucket = bucketStack.back(); // Get the bucket at the top of the stack
        bucketStack.pop_back();                 // Remove it from the stack

        // Mark the current bucket as visited by updating the Bvisited bitmask
        const size_t segment      = currentBucket >> 6; // Determine the segment for the current bucket
        const size_t bit_position = currentBucket & 63; // Determine the bit position within the segment
        Bvisited[segment] |= (1ULL << bit_position);    // Set the bit corresponding to the current bucket as visited

        // Check if the label's cost is lower than the cost bound (c_bar) for the current bucket
        // and if the current bucket precedes the label's bucket according to the bucket order
        if (L->cost < c_bar[currentBucket] && ::precedes<int>(bucket_order, currentBucket, b_L, uf)) {
            return false; // The label is not dominated, return false early
        }

        // If the current bucket is different from the label's bucket, compare the labels in this bucket
        if (b_L != currentBucket) {
            const auto bucket_labels = buckets[currentBucket].get_labels(); // Get the labels in the current bucket
            auto      &mother_bucket = buckets[currentBucket];              // Get the mother bucket for the label
            auto       check_dominance_in_bucket = [&](const std::vector<Label *> &labels) {
#ifndef __AVX2__
                for (auto *existing_label : labels) {
                    if (is_dominated<D, S>(L, existing_label)) {
                        // stat_n_dom++; // Increment dominated labels count
                        return true;
                    }
                }
                return false;
#else
                return check_dominance_against_vector<D, S>(L, labels, cut_storage, options.resources.size());
#endif
            };

            int stat_n_dom = 0; // Initialize the count of dominated labels
            // Call check_dominance on the mother bucket
            bool dominated = mother_bucket.check_dominance(L, check_dominance_in_bucket, stat_n_dom);
            if (dominated) { return true; }
        }

        // Add the neighboring buckets (from Phi) to the stack if they haven't been visited yet
        for (const int b_prime : Phi[currentBucket]) {
            const size_t segment_prime      = b_prime >> 6; // Determine the segment for the neighboring bucket
            const size_t bit_position_prime = b_prime & 63; // Determine the bit position within the segment

            // If the neighboring bucket hasn't been visited, push it onto the stack
            if ((Bvisited[segment_prime] & (1ULL << bit_position_prime)) == 0) { bucketStack.push_back(b_prime); }
        }
    }

    // If no domination was found in any smaller bucket, return false
    return false;
}

/**
 * @brief Runs forward and backward labeling algorithms in parallel and synchronizes the results.
 *
 * This function creates tasks for forward and backward labeling algorithms using the provided
 * scheduling mechanism. The tasks are executed in parallel, and the results are synchronized
 * and stored in the provided vectors.
 *
 */
template <Stage state, Full fullness>
void BucketGraph::run_labeling_algorithms(std::vector<double> &forward_cbar, std::vector<double> &backward_cbar) {
    // Create tasks for forward and backward labeling algorithms

    auto forward_task = stdexec::schedule(bi_sched) |
                        stdexec::then([&]() { return labeling_algorithm<Direction::Forward, state, fullness>(); });

    auto backward_task = stdexec::schedule(bi_sched) |
                         stdexec::then([&]() { return labeling_algorithm<Direction::Backward, state, fullness>(); });

    // Execute the tasks in parallel and synchronize
    auto work = stdexec::when_all(std::move(forward_task), std::move(backward_task)) |
                stdexec::then([&](auto forward_result, auto backward_result) {
                    forward_cbar  = std::move(forward_result);
                    backward_cbar = std::move(backward_result);
                });

    stdexec::sync_wait(std::move(work));
}

/**
 * Computes a new label based on the given labels L and L_prime.
 *
 */
template <Stage S>
Label *BucketGraph::compute_label(const Label *L, const Label *L_prime) {
    double cij_cost = getcij(L->node_id, L_prime->node_id);
    double new_cost = L->cost + L_prime->cost + cij_cost;

    double real_cost = L->real_cost + L_prime->real_cost + cij_cost;

    if constexpr (S == Stage::Four) {
#if defined(RCC) || defined(EXACT_RCC)
        auto arc_dual = arc_duals.getDual(L->node_id, L_prime->node_id);
        new_cost -= arc_dual;
#endif
    }

    // Branching duals
    if (branching_duals->size() > 0) { new_cost -= branching_duals->getDual(L->node_id, L_prime->node_id); }

    // Directly acquire new_label and set the cost
    auto new_label       = label_pool_fw->acquire();
    new_label->cost      = new_cost;
    new_label->real_cost = real_cost;

    if constexpr (S == Stage::Four) {
        SRC_MODE_BLOCK(
            //  Check SRCDuals condition for specific stages
            auto sumSRC = 0.0; const auto &SRCDuals = cut_storage->SRCDuals; if (!SRCDuals.empty()) {
                size_t idx = 0;
                auto   sumSRC =
                    std::transform_reduce(SRCDuals.begin(), SRCDuals.end(), 0.0, std::plus<>(), [&](const auto &dual) {
                        size_t curr_idx = idx++;
                        auto   den      = cut_storage->getCut(curr_idx).p.den;
                        auto   sum      = (L->SRCmap[curr_idx] + L_prime->SRCmap[curr_idx]);
                        return (sum >= den) ? dual : 0.0;
                    });

                new_label->cost -= sumSRC;
            })
    }

    new_label->nodes_covered.clear();

    /*
    // Start by inserting backward list elements
    size_t forward_size = 0;
    auto   L_bw         = L_prime;
    for (; L_bw != nullptr; L_bw = L_bw->parent) {
        new_label->nodes_covered.push_back(L_bw->node_id); // Insert backward elements directly
        if (L_bw->parent == nullptr && L_bw->fresh == false) {
            for (size_t i = 0; i < L_bw->nodes_covered.size(); ++i) {
                new_label->nodes_covered.push_back(L_bw->nodes_covered[i]);
            }
        }
    }

    // Now insert forward list elements in reverse order without using std::reverse
    auto L_fw = L;
    for (; L_fw != nullptr; L_fw = L_fw->parent) {
        new_label->nodes_covered.insert(new_label->nodes_covered.begin(),
                                        L_fw->node_id); // Insert forward elements at the front
        if (L_fw->parent == nullptr && L_fw->fresh == false) {
            for (size_t i = 0; i < L_fw->nodes_covered.size(); ++i) {
                new_label->nodes_covered.insert(new_label->nodes_covered.begin(), L_fw->nodes_covered[i]);
            }
        }
    }
    */
    new_label->nodes_covered = L_prime->nodes_covered;
    // reverse the nodes_covered
    std::reverse(new_label->nodes_covered.begin(), new_label->nodes_covered.end());
    new_label->nodes_covered.insert(new_label->nodes_covered.begin(), L->nodes_covered.begin(), L->nodes_covered.end());

    return new_label;
}
