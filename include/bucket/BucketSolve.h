/**
 * @file BucketSolve.h
 * @brief Defines the BucketGraph class and its solving methods for optimization
 * problems.
 *
 * This file contains the implementation of the `BucketGraph` class, which
 * solves a multi-stage bi-labeling optimization problem using bucket graphs.
 * The `BucketGraph` class handles:
 *
 * - Solving bucket graph optimization problems using multi-stage labeling
 * algorithms.
 * - Adaptive handling of terminal time adjustments based on label distribution.
 * - Adaptive stage transitions to refine solutions based on inner objectives
 * and iteration counts.
 * - Integration of different arc types (standard, jump, fixed) during label
 * extensions.
 * - Dual management and RCC separation for handling advanced routing
 * optimization constraints.
 *
 * The `BucketGraph` class is designed to solve complex routing problems,
 * leveraging multiple stages, parallelism, and adaptive heuristics to optimize
 * paths while respecting resource constraints.
 */

#pragma once

#include <cstring>

#include "BucketJump.h"
#include "BucketRes.h"
#include "Definitions.h"
#include "Trees.h"
#include "cuts/SRC.h"

#ifdef __AVX2__
#include "BucketAVX.h"
#endif

#include "../third_party/small_vector.hpp"

/**
 * Solves the bucket graph optimization problem using a multi-stage bi-labeling
 * algorithm.
 *
 * The function iteratively adjusts the terminal time and handles different
 * stages of the bi-labeling algorithm to find the optimal paths. The stages are
 * adaptive and transition based on the inner objective values and iteration
 * counts.
 *
 */
template <Symmetry SYM>
inline std::vector<Label *> BucketGraph::solve(bool trigger) {
    // Initialize the status as not optimal at the start
    status = Status::NotOptimal;
    if (trigger) {
        transition = true;
        fixed = false;
    }

    updateSplit();  // Update the split values for the bucket graph

    // Placeholder for the final paths (labels) and inner objective value
    std::vector<Label *> paths;

    //////////////////////////////////////////////////////////////////////
    // ADAPTIVE STAGE HANDLING
    //////////////////////////////////////////////////////////////////////
    // Stage 1: Apply a light heuristic (Stage::One)
    if (s1 && depth == 0) {
        stage = 1;
        paths = bi_labeling_algorithm<Stage::One>();  // Solve the problem with
                                                      // Stage 1 heuristic
        // inner_obj = paths[0]->cost;

        // Transition from Stage 1 to Stage 2 if the objective improves or after
        // 10 iterations
        if (inner_obj >= -1 || iter >= 10) {
            s1 = false;
            s2 = true;  // Move to Stage 2
        }
    }
    // Stage 2: Apply a more expensive pricing heuristic (Stage::Two)
    else if (s2 && depth == 0) {
        s2 = true;
        stage = 2;
        paths = bi_labeling_algorithm<Stage::Two>();  // Solve the problem with
                                                      // Stage 2 heuristic
        // inner_obj = paths[0]->cost;

        // Transition from Stage 2 to Stage 3 if the objective improves or after
        // 800 iterations
        if (inner_obj >= -100 || iter > 500) {
            s2 = false;
            s3 = true;  // Move to Stage 3
        }
    }
    // Stage 3: Apply a heuristic fixing approach (Stage::Three)
    else if (s3 && depth == 0) {
        stage = 3;
        paths =
            bi_labeling_algorithm<Stage::Three>();  // Solve the problem with
                                                    // Stage 3 heuristic
        // inner_obj = paths[0]->cost;

        // Transition from Stage 3 to Stage 4 if the objective improves
        // significantly
        if (inner_obj >= -0.5) {
            s4 = true;
            s3 = false;
            transition = true;  // Prepare for transition to Stage 4
        }
    }
    // Stage 4: Exact labeling algorithm with fixing enabled (Stage::Four)
    else {
        stage = 4;

#ifdef FIX_BUCKETS
        // If transitioning to Stage 4, print a message and apply fixing
        if (transition) {
            // print_cut("Transitioning to stage 4\n");
            bool original_fixed = fixed;  // Store the original fixed status
            fixed = true;                 // Enable fixing of buckets in Stage 4
            paths = bi_labeling_algorithm<Stage::Four>();  // Solve the problem
                                                           // with Stage 4
            transition = false;             // End the transition period
            fixed = original_fixed;         // Restore the original fixed status
            min_red_cost = paths[0]->cost;  // Update the minimum reduced cost
            iter++;
            return paths;  // Return the final paths
        }
#endif
        // If not transitioning, continue with Stage 4 algorithm
        paths = bi_labeling_algorithm<Stage::Four>();
        // inner_obj = paths[0]->cost;

        auto rollback = false;
        // auto rollback = false;
        //   auto rollback = false;
        if (rollback) {
            //    s4     = false; // Rollback to Stage 3 if necessary
            //    s3     = true;
            status = Status::Rollback;  // Set status to rollback
            return paths;               // Return the paths after rollback
        }
        // If the objective improves sufficiently, set the status to separation
        // or optimal
        if (inner_obj > -1.0) {
            ss = true;  // Enter separation mode (for SRC handling)
#if !defined(SRC) && !defined(SRC3)
            status = Status::Optimal;  // If SRC is not defined, set status to
                                       // optimal
            return paths;              // Return the optimal paths
#endif
            status = Status::Separation;  // If SRC is defined, set status to
                                          // separation
        }
    }

    iter++;  // Increment the iteration counter

    return paths;  // Return the final paths after processing
}

inline std::vector<Label *> BucketGraph::solveHeuristic() {
    // Initialize the status as not optimal at the start
    status = Status::NotOptimal;

    updateSplit();  // Update the split values for the bucket graph

    // Placeholder for the final paths (labels) and inner objective value
    std::vector<Label *> paths;
    double inner_obj;

    //////////////////////////////////////////////////////////////////////
    // ADAPTIVE STAGE HANDLING
    //////////////////////////////////////////////////////////////////////
    // Stage 1: Apply a light heuristic (Stage::One)
    if (s1) {
        stage = 1;
        paths = bi_labeling_algorithm<Stage::One>();  // Solve the problem with
                                                      // Stage 1 heuristic
        inner_obj = paths[0]->cost;

        // Transition from Stage 1 to Stage 2 if the objective improves or after
        // 10 iterations
        if (inner_obj >= -5 || iter >= 10) {
            s1 = false;
            s2 = true;  // Move to Stage 2
        }
    }
    // Stage 2: Apply a more expensive pricing heuristic (Stage::Two)
    else if (s2) {
        s2 = true;
        stage = 2;
        paths = bi_labeling_algorithm<Stage::Two>();  // Solve the problem with
                                                      // Stage 2 heuristic
        inner_obj = paths[0]->cost;

        // Transition from Stage 2 to Stage 3 if the objective improves or after
        // 800 iterations
        if (inner_obj >= -100 || iter > 800) {
            status = Status::Optimal;
        }
    }

    iter++;  // Increment the iteration counter

    return paths;  // Return the final paths after processing
}

/**
 * Performs the labeling algorithm on the BucketGraph.
 *
 */
template <Direction D, Stage S, Full F>
std::vector<double> BucketGraph::labeling_algorithm() {
    // Use references to avoid repeated lookups
    auto &buckets = assign_buckets<D>(fw_buckets, bw_buckets);
    auto &ordered_sccs = assign_buckets<D>(fw_ordered_sccs, bw_ordered_sccs);
    auto &topological_order =
        assign_buckets<D>(fw_topological_order, bw_topological_order);
    auto &sccs = assign_buckets<D>(fw_sccs, bw_sccs);
    auto &Phi = assign_buckets<D>(Phi_fw, Phi_bw);
    auto &c_bar = assign_buckets<D>(fw_c_bar, bw_c_bar);
    auto &n_labels = assign_buckets<D>(n_fw_labels, n_bw_labels);
    auto &sorted_sccs = assign_buckets<D>(fw_sccs_sorted, bw_sccs_sorted);
    auto &n_buckets = assign_buckets<D>(fw_buckets_size, bw_buckets_size);
    auto &stat_n_labels = assign_buckets<D>(stat_n_labels_fw, stat_n_labels_bw);
    auto &stat_n_dom = assign_buckets<D>(stat_n_dom_fw, stat_n_dom_bw);

    n_labels = 0;

    // Pre-calculate segment info
    const size_t n_segments = (n_buckets + 63) / 64;
    std::vector<uint64_t> Bvisited(n_segments);

    // Preallocate vector for new labels to avoid repeated allocations
    std::vector<Label *> new_labels;
    new_labels.reserve(16);  // Adjust size based on typical usage

    // Process SCCs in topological order
    for (const auto &scc_index : topological_order) {
        bool all_ext;
        do {
            all_ext = true;

            // Process buckets in sorted order within SCC
            for (const auto bucket : sorted_sccs[scc_index]) {
                // Cache bucket labels reference
                const auto &bucket_labels = buckets[bucket].get_labels();
                const size_t n_bucket_labels = bucket_labels.size();

                for (size_t label_idx = 0; label_idx < n_bucket_labels;
                     ++label_idx) {
                    Label *label = bucket_labels[label_idx];
                    if (label->is_extended) continue;

                    // Early resource check for partial solutions
                    if constexpr (F != Full::PSTEP && F != Full::TSP) {
                        if constexpr (F == Full::Partial) {
                            const double main_resource =
                                label->resources[options.main_resources[0]];
                            const double q_star_value =
                                q_star[options.main_resources[0]];

                            if constexpr (D == Direction::Forward) {
                                if (main_resource > q_star_value) {
                                    label->set_extended(true);
                                    continue;
                                }
                            } else if constexpr (D == Direction::Backward) {
                                if (main_resource <= q_star_value) {
                                    label->set_extended(true);
                                    continue;
                                }
                            }
                        }
                    }

                    // Efficient visited bucket clearing
                    if (n_segments <= 8) {
                        for (size_t i = 0; i < n_segments; ++i) {
                            Bvisited[i] = 0;
                        }
                    } else {
                        std::memset(Bvisited.data(), 0,
                                    n_segments * sizeof(uint64_t));
                    }

                    // Check dominance in smaller buckets
                    bool domin_smaller = false;
                    if constexpr (F != Full::TSP) {
                        domin_smaller = DominatedInCompWiseSmallerBuckets<D, S>(
                            label, bucket, c_bar, Bvisited, ordered_sccs);
                    }

                    if (!domin_smaller) {
                        // Process arcs for current label
                        const auto &node_arcs =
                            nodes[label->node_id].get_arcs<D>(scc_index);

                        for (const auto &arc : node_arcs) {
                            new_labels.clear();  // Reuse vector
                            auto extended_labels =
                                Extend<D, S, ArcType::Node, Mutability::Mut, F>(
                                    label, arc);

                            if (extended_labels.empty()) {
#ifdef UNREACHABLE_DOMINANCE
                                set_node_unreachable(label->unreachable_bitmap,
                                                     arc.to);
#endif
                                continue;
                            }

                            // Process each new label
                            for (Label *new_label : extended_labels) {
                                stat_n_labels++;

                                const int to_bucket = new_label->vertex;
                                auto &mother_bucket = buckets[to_bucket];
                                const auto &to_bucket_labels =
                                    mother_bucket.get_labels();

                                // Stage-specific dominance checking
                                if constexpr (F != Full::PSTEP &&
                                              F != Full::TSP) {
                                    if constexpr (S == Stage::Four &&
                                                  D == Direction::Forward) {
                                        dominance_checks_per_bucket
                                            [to_bucket] +=
                                            to_bucket_labels.size();
                                    }
                                }

                                bool dominated = false;
                                if constexpr (S == Stage::One) {
                                    // Optimized Stage One processing
                                    for (auto *existing_label :
                                         to_bucket_labels) {
                                        if (label->cost <
                                            existing_label->cost) {
                                            mother_bucket.remove_label(
                                                existing_label);
                                        } else {
                                            dominated = true;
                                            break;
                                        }
                                    }
                                } else {
                                    // Use existing SIMD implementation for
                                    // other stages
                                    auto check_dominance_in_bucket =
                                        [&](const std::vector<Label *>
                                                &labels) {
#ifndef __AVX2__
                                            for (auto *existing_label :
                                                 labels) {
                                                if (is_dominated<D, S>(
                                                        new_label,
                                                        existing_label)) {
                                                    stat_n_dom++;
                                                    return true;
                                                }
                                            }
                                            return false;
#else
                                            return check_dominance_against_vector<
                                                D, S>(new_label, labels,
                                                      cut_storage,
                                                      options.resources.size());
#endif
                                        };

                                    dominated = mother_bucket.check_dominance(
                                        new_label, check_dominance_in_bucket,
                                        stat_n_dom);
                                }

                                if (!dominated) {
                                    if constexpr (S != Stage::Enumerate) {
                                        for (auto *existing_label :
                                             to_bucket_labels) {
                                            if (is_dominated<D, S>(
                                                    existing_label,
                                                    new_label)) {
                                                mother_bucket.remove_label(
                                                    existing_label);
                                            }
                                        }
                                    }

                                    n_labels++;
#ifdef SORTED_LABELS
                                    mother_bucket.add_sorted_label(new_label);
#elif LIMITED_BUCKETS
                                    mother_bucket.sorted_label(new_label,
                                                               BUCKET_CAPACITY);
#else
                                    mother_bucket.add_label(new_label);
#endif
                                    all_ext = false;
                                }
                            }
                        }
                    }

                    label->set_extended(true);
                }
            }

            // Update c_bar values efficiently
            for (const int bucket : sorted_sccs[scc_index]) {
                double min_c_bar = buckets[bucket].get_cb();
                for (const auto phi_bucket : Phi[bucket]) {
                    min_c_bar = std::min(min_c_bar, c_bar[phi_bucket]);
                }
                c_bar[bucket] = min_c_bar;
            }
        } while (!all_ext);
    }

    // Store best label
    Label *best_label = get_best_label<D>(topological_order, c_bar, sccs);
    if constexpr (D == Direction::Forward) {
        fw_best_label = best_label;
    } else {
        bw_best_label = best_label;
    }

    return c_bar;
}

/**
 * Performs the bi-labeling algorithm on the BucketGraph.
 *
 */
template <Stage S, Symmetry SYM>
std::vector<Label *> BucketGraph::bi_labeling_algorithm() {
    // Stage-specific initializations
    if constexpr (S == Stage::Three) {
        heuristic_fixing<S>();
    } else if constexpr (S == Stage::Four) {
        if (first_reset) {
            reset_fixed();
            first_reset = false;
        }
#ifdef FIX_BUCKETS
        if (options.bucket_fixing) {
            bucket_fixing<S>();
        }
#endif
    }

    reset_pool();
    common_initialization();

    std::vector<double> forward_cbar(fw_buckets.size());
    std::vector<double> backward_cbar(bw_buckets.size());

    if constexpr (SYM == Symmetry::Asymmetric) {
        run_labeling_algorithms<S, Full::Partial>(forward_cbar, backward_cbar);
    } else {
        forward_cbar =
            labeling_algorithm<Direction::Forward, S, Full::Partial>();
    }

    auto best_label = label_pool_fw->acquire();
    if constexpr (SYM == Symmetry::Asymmetric) {
        if (check_feasibility(fw_best_label, bw_best_label)) {
            best_label = compute_label<S>(fw_best_label, bw_best_label);
        } else {
            best_label->cost = 0.0;
            best_label->real_cost = std::numeric_limits<double>::infinity();
            best_label->nodes_covered.clear();
        }
    } else {
        best_label->cost = best_label->real_cost;
        best_label->real_cost = std::numeric_limits<double>::infinity();
        best_label->nodes_covered.clear();
    }
    merged_labels.push_back(best_label);

    if constexpr (S == Stage::Enumerate) {
        fmt::print("Labels generated, concatenating...\n");
    }

    const size_t n_segments = (fw_buckets_size + 63) / 64;
    std::vector<uint64_t> Bvisited(n_segments);

    double best_cost = std::numeric_limits<double>::infinity();
    for (int bucket = 0; bucket < fw_buckets_size; ++bucket) {
        const auto &bucket_labels = fw_buckets[bucket].get_labels();

        if constexpr (S == Stage::Four) {
            non_dominated_labels_per_bucket += bucket_labels.size();
        }

        for (const Label *L : bucket_labels) {
            const auto &to_arcs =
                nodes[L->node_id].get_arcs<Direction::Forward>();

            for (const auto &arc : to_arcs) {
                const int to_node = arc.to;

                if constexpr (S == Stage::Three || S == Stage::Eliminate) {
                    if (fixed_arcs[L->node_id][to_node] == 1) {
                        continue;
                    }
                }

                auto extended_labels =
                    Extend<Direction::Forward, S, ArcType::Node,
                           Mutability::Const, Full::Reverse>(L, arc);
                for (Label *L_prime : extended_labels) {
                    int bucket_to_process =
                        L_prime->vertex;  // Create mutable copy
                    std::memset(Bvisited.data(), 0,
                                n_segments * sizeof(uint64_t));
                    ConcatenateLabel<S, SYM>(L, bucket_to_process, best_cost,
                                             Bvisited);
                }
            }
        }
    }

    pdqsort(merged_labels.begin(), merged_labels.end(),
            [](const Label *a, const Label *b) { return a->cost < b->cost; });

#ifdef SCHRODINGER
    if (merged_labels.size() > N_ADD) {
        std::vector<Path> paths;
        const int labels_size = merged_labels.size();
        const int end_idx = std::min(N_ADD + N_ADD, labels_size);

        paths.reserve(end_idx - N_ADD);
        for (int i = N_ADD; i < end_idx; ++i) {
            if (merged_labels[i]->nodes_covered.size() <= 3) {
                continue;
            }
            paths.emplace_back(merged_labels[i]->nodes_covered,
                               merged_labels[i]->real_cost);
        }

        sPool.add_paths(paths);
        sPool.iterate();
    }
#endif

#ifdef RIH
    std::vector<Label *> top_labels;
    top_labels.reserve(N_ADD);

    const int n_candidates =
        std::min(N_ADD, static_cast<int>(merged_labels.size()));
    for (int i = 0; i < n_candidates; ++i) {
        if (merged_labels[i]->nodes_covered.size() <= 3) {
            continue;
        }
        top_labels.push_back(merged_labels[i]);
    }
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
inline std::vector<Label *> BucketGraph::Extend(
    const std::conditional_t<M == Mutability::Mut, Label *, const Label *>
        L_prime,
    const std::conditional_t<
        A == ArcType::Bucket, BucketArc,
        std::conditional_t<A == ArcType::Jump, JumpArc, Arc>> &gamma,
    int depth) noexcept {
    static thread_local std::vector<double> new_resources;
    new_resources.resize(options.resources.size());

    const int initial_node_id = L_prime->node_id;
    auto initial_resources = L_prime->resources;  // Make a copy as in original
    const double initial_cost = L_prime->cost;

    int node_id;
    if constexpr (A == ArcType::Bucket) {
        node_id =
            assign_buckets<D>(fw_buckets, bw_buckets)[gamma.to_bucket].node_id;
    } else if constexpr (A == ArcType::Node) {
        node_id = gamma.to;
    } else {
        auto &buckets = assign_buckets<D>(fw_buckets, bw_buckets);
        node_id = buckets[gamma.jump_bucket].node_id;

        // Update copy of resources for jump arcs
        const auto &bucket = buckets[gamma.jump_bucket];
        for (size_t i = 0; i < options.resources.size(); ++i) {
            if constexpr (D == Direction::Forward) {
                initial_resources[i] = std::max(
                    initial_resources[i], static_cast<double>(bucket.lb[i]));
            } else {
                initial_resources[i] = std::min(
                    initial_resources[i], static_cast<double>(bucket.ub[i]));
            }
        }
    }

    // Early rejection checks
    if (node_id == L_prime->node_id ||
        is_node_visited(L_prime->visited_bitmap, node_id)) {
        return {};
    }

    if constexpr (S == Stage::Three || S == Stage::Eliminate) {
        if ((D == Direction::Forward &&
             fixed_arcs[initial_node_id][node_id] == 1) ||
            (D == Direction::Backward &&
             fixed_arcs[node_id][initial_node_id] == 1)) {
            return {};
        }
    }

    // Process resources and check feasibility
    if constexpr (F != Full::TSP) {
        if (!process_all_resources<D>(new_resources, initial_resources, gamma,
                                      nodes[node_id],
                                      options.resources.size())) {
            return {};
        }
    }

    // Path size checks for PSTEP/TSP
    if constexpr (F == Full::PSTEP || F == Full::TSP) {
        int n_visited = 0;
        for (const auto &bitmap : L_prime->visited_bitmap) {
            n_visited += __builtin_popcountll(bitmap);
        }

        if (n_visited > options.max_path_size ||
            (n_visited == options.max_path_size &&
             node_id != options.end_depot)) {
            return {};
        }
    }

    const int to_bucket = get_bucket_number<D>(node_id, new_resources);

    // Handle fixed buckets
#ifdef FIX_BUCKETS
    if constexpr (S == Stage::Four && A != ArcType::Jump) {
        if (is_bucket_fixed<D>(L_prime->vertex, to_bucket)) {
            if (depth > 1) return {};

            static thread_local std::vector<Label *>
                label_vector;  // Reuse vector
            label_vector.clear();

            for (const auto &jump_arc :
                 nodes[L_prime->node_id].template get_jump_arcs<D>(node_id)) {
                auto extended_labels =
                    Extend<D, S, ArcType::Jump, Mutability::Const, F>(
                        L_prime, jump_arc, depth + 1);
                if (!extended_labels.empty()) {
                    label_vector.insert(label_vector.end(),
                                        extended_labels.begin(),
                                        extended_labels.end());
                }
            }
            return label_vector;
        }
    }
#endif

    // Cost computation
    double new_cost = initial_cost + getcij(initial_node_id, node_id);
    const auto &VRPNode = nodes[node_id];

    if constexpr (F != Full::PSTEP) {
        new_cost -= VRPNode.cost;
    } else {
        int n_visited = 0;
        for (const auto &bitmap : L_prime->visited_bitmap) {
            n_visited += __builtin_popcountll(bitmap);
        }

        if (n_visited > 1 && initial_node_id != options.depot) {
            new_cost += pstep_duals.getThreeTwoDualValue(initial_node_id) +
                        pstep_duals.getThreeThreeDualValue(initial_node_id);
        }
        new_cost += -pstep_duals.getThreeTwoDualValue(node_id) +
                    pstep_duals.getArcDualValue(initial_node_id, node_id);
    }

    // Branching duals
    if (!branching_duals->empty()) {
        new_cost += (D == Direction::Forward)
                        ? branching_duals->getDual(initial_node_id, node_id)
                        : branching_duals->getDual(node_id, initial_node_id);
        new_cost += branching_duals->getDual(node_id);
    }

    RCC_MODE_BLOCK(if constexpr (S == Stage::Four) {
        new_cost -= (D == Direction::Forward)
                        ? arc_duals.getDual(initial_node_id, node_id)
                        : arc_duals.getDual(node_id, initial_node_id);
    })

    // Handle Reverse case
    if constexpr (F == Full::Reverse) {
        if (new_resources[options.main_resources[0]] <=
            q_star[options.main_resources[0]]) {
            return {};
        }
        auto &label_pool = assign_buckets<D>(label_pool_fw, label_pool_bw);
        auto new_label = label_pool->acquire();
        new_label->vertex = to_bucket;
        return {new_label};
    }

    // Create and initialize new label
    auto &label_pool = assign_buckets<D>(label_pool_fw, label_pool_bw);
    auto new_label = label_pool->acquire();
    new_label->initialize(to_bucket, new_cost, new_resources, node_id);
    new_label->vertex = to_bucket;
    new_label->real_cost =
        L_prime->real_cost + getcij(initial_node_id, node_id);

    if constexpr (M == Mutability::Mut) {
        new_label->parent = L_prime;
    }

    // Handle visited bitmap
    if constexpr (F != Full::PSTEP) {
        if constexpr (S != Stage::Enumerate) {
            for (size_t i = 0; i < new_label->visited_bitmap.size(); ++i) {
                const uint64_t current_visited = L_prime->visited_bitmap[i];
                if (current_visited) {
                    new_label->visited_bitmap[i] =
                        current_visited & neighborhoods_bitmap[node_id][i];
                }
            }
        }
    } else {
        new_label->visited_bitmap = L_prime->visited_bitmap;
    }
    set_node_visited(new_label->visited_bitmap, node_id);

    // new_label->nodes_covered = L_prime->nodes_covered;    // Copy the list of
    // covered nodes new_label->nodes_covered.push_back(node_id);          //
    // Add the new node to the list of covered nodes

#if defined(SRC)
    new_label->SRCmap = L_prime->SRCmap;
    if constexpr (S == Stage::Four || S == Stage::Enumerate) {
        auto &cutter = cut_storage;
        std::span<const double> SRCDuals{cutter->SRCDuals};
        const size_t segment = node_id >> 6;
        const size_t bit_position = node_id & 63;
        const uint64_t bit_mask = 1ULL << bit_position;
        const size_t cut_size = cutter->size();

        static constexpr size_t SMALL_SIZE = 256;

#if defined(__cpp_lib_ranges)
        auto indices = std::views::iota(size_t{0}, cut_size);
#else
        static thread_local std::vector<size_t> indices;
        indices.resize(cut_size);
        std::iota(indices.begin(), indices.end(), 0);
#endif

        auto process_cut = [&](size_t idx) -> double {
            if (SRCDuals[idx] > -1e-3) {
                return 0.0;
            }

            const auto &cut = cutter->getCut(idx);
            if (!(cut.neighbors[segment] & bit_mask)) {
                new_label->SRCmap[idx] = 0.0;
                return 0.0;
            }

            if (cut.baseSet[segment] & bit_mask) {
                const auto &multipliers = cut.p;
                const auto &den = multipliers.den;
                auto &src_map_value = new_label->SRCmap[idx];
                src_map_value += multipliers.num[cut.baseSetOrder[node_id]];
                if (src_map_value >= den) {
                    src_map_value -= den;
                    return -SRCDuals[idx];
                }
            }
            return 0.0;
        };

        double total_cost_update;
        if (cut_size <= SMALL_SIZE) {
            total_cost_update = 0.0;
            for (size_t i = 0; i < cut_size; ++i) {
                total_cost_update += process_cut(i);
            }
        } else {
#if defined(__cpp_lib_parallel_algorithm)
            total_cost_update = std::transform_reduce(
                std::execution::par_unseq, std::begin(indices),
                std::end(indices), 0.0, std::plus<>(), process_cut);
#else
            total_cost_update =
                std::transform_reduce(std::begin(indices), std::end(indices),
                                      0.0, std::plus<>(), process_cut);
#endif
        }

        new_label->cost += total_cost_update;
    }
#endif

    // Return the array (as Label**)
    return std::vector<Label *>{new_label};
}

/**
 * @brief Checks if a label is dominated by a new label based on cost and
 * resource conditions.
 *
 */
template <Direction D, Stage S>
inline bool BucketGraph::is_dominated(const Label *new_label,
                                      const Label *label) noexcept {
    // Extract resources for the new label and the comparison label
    const auto &new_resources = new_label->resources;
    const auto &label_resources = label->resources;

    {
        // Simple cost check: if the comparison label has a higher cost, it is
        // not dominated
        if (label->cost > new_label->cost) {
            return false;
        }
    }

    // Check resource conditions based on the direction (Forward or Backward)
    for (size_t i = 0; i < options.resources.size(); ++i) {
        if constexpr (D == Direction::Forward) {
            // In Forward direction: the comparison label must not have more
            // resources than the new label
            if (label_resources[i] > new_resources[i]) {
                return false;
            }
        } else if constexpr (D == Direction::Backward) {
            // In Backward direction: the comparison label must not have fewer
            // resources than the new label
            if (label_resources[i] < new_resources[i]) {
                return false;
            }
        }
    }

    // TODO:: check again this unreachable dominance
#ifndef UNREACHABLE_DOMINANCE
    // Check visited nodes (bitmap comparison) for Stages 3, 4, and Enumerate
    if constexpr (S == Stage::Three || S == Stage::Four ||
                  S == Stage::Enumerate) {
        // Iterate through the visited bitmap and ensure that the new label
        // visits all nodes that the comparison label visits
        for (size_t i = 0; i < label->visited_bitmap.size(); ++i) {
            // If the comparison label visits a node that the new label does
            // not, it is not dominated
            if (((label->visited_bitmap[i] & new_label->visited_bitmap[i]) ^
                 label->visited_bitmap[i]) != 0) {
                return false;
            }
        }
    }
#else
    // Unreachable dominance logic: check visited and unreachable nodes in
    // Stages 3, 4, and Enumerate
    if constexpr (S == Stage::Three || S == Stage::Four ||
                  S == Stage::Enumerate) {
        for (size_t i = 0; i < label->visited_bitmap.size(); ++i) {
            // Combine visited and unreachable nodes in the comparison label's
            // bitmap
            auto combined_label_bitmap =
                label->visited_bitmap[i] | label->unreachable_bitmap[i];
            // Ensure the new label visits all nodes that the comparison label
            // (or its unreachable nodes) visits
            if ((combined_label_bitmap & ~new_label->visited_bitmap[i]) != 0) {
                return false;
            }
        }
    }
#endif

#ifdef SRC
    if constexpr (S == Stage::Four || S == Stage::Enumerate) {
        double sumSRC = 0;

        const auto &SRCDuals = cut_storage->SRCDuals;
        if (!SRCDuals.empty()) {
            const auto &labelSRCMap = label->SRCmap;
            const auto &newLabelSRCMap = new_label->SRCmap;
            for (size_t i = 0; i < SRCDuals.size(); ++i) {
                const auto &den = cut_storage->getCut(i).p.den;
                const auto labelMod = labelSRCMap[i];        // % den;
                const auto newLabelMod = newLabelSRCMap[i];  // % den;
                if (labelMod > newLabelMod) {
                    sumSRC += SRCDuals[i];
                }
            }
        }
        if (label->cost - sumSRC > new_label->cost) {
            return false;
        }
    }
#endif

    // If all conditions are met, return true, indicating that the new label is
    // dominated by the comparison label
    return true;
}

/**
 * @brief Checks if element 'a' precedes element 'b' in the given strongly
 * connected components (SCCs).
 *
 * This function takes a vector of SCCs and two elements 'a' and 'b' as input.
 * It searches for 'a' and 'b' in the SCCs and determines if 'a' precedes 'b' in
 * the SCC list.
 *
 */
template <typename T>
inline bool precedes(const std::vector<std::vector<int>> &sccs, const T a,
                     const T b, UnionFind &uf) {
    // Step 2: Check if element `a`'s SCC precedes element `b`'s SCC
    size_t rootA = uf.getSubset(a);
    size_t rootB = uf.getSubset(b);

    // You can define precedence based on the comparison of the root components
    return rootA < rootB;
}
/**
 * @brief Determines if a label is dominated in component-wise smaller buckets.
 *
 * This function checks if a given label is dominated by any other label in
 * component-wise smaller buckets. The dominance is determined based on the cost
 * and order of the buckets.
 *
 */
template <Direction D, Stage S>
inline bool BucketGraph::DominatedInCompWiseSmallerBuckets(
    const Label *__restrict__ L, int bucket,
    const std::vector<double> &__restrict__ c_bar,
    std::vector<uint64_t> &__restrict__ Bvisited,
    const std::vector<std::vector<int>> &__restrict__ bucket_order) noexcept {
    // Use references to avoid indirection
    auto &buckets = assign_buckets<D>(fw_buckets, bw_buckets);
    auto &Phi = assign_buckets<D>(Phi_fw, Phi_bw);
    auto &uf = assign_buckets<D>(fw_union_find, bw_union_find);

    // Cache frequently used values
    const int b_L = L->vertex;
    const double label_cost = L->cost;
    const size_t res_size = options.resources.size();

    // Pre-compute bit operation constants
    constexpr uint64_t one = 1ULL;
    constexpr uint64_t bit_mask_lookup[64] = {
        1ULL << 0,  1ULL << 1,  1ULL << 2,  1ULL << 3,  1ULL << 4,  1ULL << 5,
        1ULL << 6,  1ULL << 7,  1ULL << 8,  1ULL << 9,  1ULL << 10, 1ULL << 11,
        1ULL << 12, 1ULL << 13, 1ULL << 14, 1ULL << 15, 1ULL << 16, 1ULL << 17,
        1ULL << 18, 1ULL << 19, 1ULL << 20, 1ULL << 21, 1ULL << 22, 1ULL << 23,
        1ULL << 24, 1ULL << 25, 1ULL << 26, 1ULL << 27, 1ULL << 28, 1ULL << 29,
        1ULL << 30, 1ULL << 31, 1ULL << 32, 1ULL << 33, 1ULL << 34, 1ULL << 35,
        1ULL << 36, 1ULL << 37, 1ULL << 38, 1ULL << 39, 1ULL << 40, 1ULL << 41,
        1ULL << 42, 1ULL << 43, 1ULL << 44, 1ULL << 45, 1ULL << 46, 1ULL << 47,
        1ULL << 48, 1ULL << 49, 1ULL << 50, 1ULL << 51, 1ULL << 52, 1ULL << 53,
        1ULL << 54, 1ULL << 55, 1ULL << 56, 1ULL << 57, 1ULL << 58, 1ULL << 59,
        1ULL << 60, 1ULL << 61, 1ULL << 62, 1ULL << 63};

    // Stack-based implementation with fixed size for small bucket counts
    constexpr size_t MAX_STACK_SIZE = 128;
    int stack_buffer[MAX_STACK_SIZE];
    int stack_size = 1;
    stack_buffer[0] = bucket;

    // SIMD-optimized dominance check function
    auto check_dominance = [&](const std::vector<Label *> &labels) {
#ifdef __AVX2__
        return check_dominance_against_vector<D, S>(L, labels, cut_storage,
                                                    res_size);
#else
        const size_t n = labels.size();
        for (size_t i = 0; i < n; ++i) {
            if (is_dominated<D, S>(L, labels[i])) return true;
        }
        return false;
#endif
    };

    do {
        const int current_bucket = stack_buffer[--stack_size];

        // Optimize bit operations
        const size_t segment = current_bucket >> 6;
        const uint64_t bit = bit_mask_lookup[current_bucket & 63];
        Bvisited[segment] |= bit;

        // Early exit check
        if (__builtin_expect(
                (label_cost < c_bar[current_bucket] &&
                 ::precedes<int>(bucket_order, current_bucket, b_L, uf)),
                0)) {
            return false;
        }

        // Check dominance only if necessary
        if (b_L != current_bucket) {
            auto &mother_bucket = buckets[current_bucket];
            int stat_n_dom = 0;
            if (mother_bucket.check_dominance(L, check_dominance, stat_n_dom)) {
                return true;
            }
        }

        // Process neighbors using SIMD if available
        const auto &bucket_neighbors = Phi[current_bucket];
        const size_t n_neighbors = bucket_neighbors.size();

        // Regular processing
        for (size_t i = 0; i < n_neighbors; ++i) {
            const int b_prime = bucket_neighbors[i];
            const size_t seg_prime = b_prime >> 6;
            if (!(Bvisited[seg_prime] & bit_mask_lookup[b_prime & 63])) {
                // if (stack_size < MAX_STACK_SIZE) {
                stack_buffer[stack_size++] = b_prime;
                // } else {
                // fmt::print(
                // "Stack overflow in "
                // "DominatedInCompWiseSmallerBuckets\n");
                // return false;
                // }
            }
        }

    } while (stack_size > 0);

    return false;
}

/**
 * @brief Runs forward and backward labeling algorithms in parallel and
 * synchronizes the results.
 *
 * This function creates tasks for forward and backward labeling algorithms
 * using the provided scheduling mechanism. The tasks are executed in parallel,
 * and the results are synchronized and stored in the provided vectors.
 *
 */
template <Stage state, Full fullness>
void BucketGraph::run_labeling_algorithms(std::vector<double> &forward_cbar,
                                          std::vector<double> &backward_cbar) {
    // Create tasks for forward and backward labeling algorithms

    auto forward_task =
        stdexec::schedule(bi_sched) | stdexec::then([&]() {
            return labeling_algorithm<Direction::Forward, state, fullness>();
        });

    auto backward_task =
        stdexec::schedule(bi_sched) | stdexec::then([&]() {
            return labeling_algorithm<Direction::Backward, state, fullness>();
        });

    // Execute the tasks in parallel and synchronize
    auto work =
        stdexec::when_all(std::move(forward_task), std::move(backward_task)) |
        stdexec::then([&](auto forward_result, auto backward_result) {
            forward_cbar = std::move(forward_result);
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
    if (branching_duals->size() > 0) {
        new_cost -= branching_duals->getDual(L->node_id, L_prime->node_id);
    }

    // Directly acquire new_label and set the cost
    auto new_label = label_pool_fw->acquire();
    new_label->cost = new_cost;
    new_label->real_cost = real_cost;

    if constexpr (S == Stage::Four) {
        SRC_MODE_BLOCK(
            //  Check SRCDuals condition for specific stages
            auto sumSRC = 0.0; const auto &SRCDuals = cut_storage->SRCDuals;
            if (!SRCDuals.empty()) {
                size_t idx = 0;
                auto sumSRC = std::transform_reduce(
                    SRCDuals.begin(), SRCDuals.end(), 0.0, std::plus<>(),
                    [&](const auto &dual) {
                        size_t curr_idx = idx++;
                        auto den = cut_storage->getCut(curr_idx).p.den;
                        auto sum =
                            (L->SRCmap[curr_idx] + L_prime->SRCmap[curr_idx]);
                        return (sum >= den) ? dual : 0.0;
                    });

                new_label->cost -= sumSRC;
            })
    }

    new_label->nodes_covered.clear();

    // Start by inserting backward list elements
    size_t forward_size = 0;
    auto L_bw = L_prime;
    for (; L_bw != nullptr; L_bw = L_bw->parent) {
        new_label->nodes_covered.push_back(
            L_bw->node_id);  // Insert backward elements directly
        if (L_bw->parent == nullptr && L_bw->fresh == false) {
            for (size_t i = 0; i < L_bw->nodes_covered.size(); ++i) {
                new_label->nodes_covered.push_back(L_bw->nodes_covered[i]);
            }
        }
    }

    // Now insert forward list elements in reverse order without using
    // std::reverse
    auto L_fw = L;
    for (; L_fw != nullptr; L_fw = L_fw->parent) {
        new_label->nodes_covered.insert(
            new_label->nodes_covered.begin(),
            L_fw->node_id);  // Insert forward elements at the front
        if (L_fw->parent == nullptr && L_fw->fresh == false) {
            for (size_t i = 0; i < L_fw->nodes_covered.size(); ++i) {
                new_label->nodes_covered.insert(
                    new_label->nodes_covered.begin(), L_fw->nodes_covered[i]);
            }
        }
    }

    // new_label->nodes_covered = L_prime->nodes_covered;
    //  reverse the nodes_covered
    // std::reverse(new_label->nodes_covered.begin(),
    // new_label->nodes_covered.end());
    // new_label->nodes_covered.insert(new_label->nodes_covered.begin(),
    // L->nodes_covered.begin(), L->nodes_covered.end());

    return new_label;
}
