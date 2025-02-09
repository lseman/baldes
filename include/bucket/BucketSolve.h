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

#include <execution>

#include "../third_party/small_vector.hpp"
#include "BucketUtils.h"

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

inline auto computeThreshold(int iteration, double inner_obj,
                             const Stats &stats) {
    // Constants for threshold computation
    const double INITIAL_BASE = -15.0;
    const double DECAY_RATE = 0.003;
    const double MIN_THRESHOLD = -10.0;
    const double MAX_THRESHOLD = -1.5;
    const double STAGNATION_TARGET = -1.5;
    const int STAGNATION_WINDOW =
        10;  // Number of iterations to check for stagnation

    // Base threshold with exponential decay
    double base = INITIAL_BASE * std::exp(-DECAY_RATE * iteration);

    // Adjust based on recent progress
    double progress_factor = 1.0;
    if (stats.hasHistory()) {
        double avg_improvement = stats.getRecentImprovement(STAGNATION_WINDOW);
        progress_factor = std::max(0.8, std::min(1.2, avg_improvement));
    }

    // Add dynamic component based on current inner_obj value
    double dynamic_component = 0.01 * std::log(1 + std::abs(inner_obj));

    // Combine components to compute the initial threshold
    double threshold = base * progress_factor + dynamic_component;

    // Check for stagnation in the objective value
    static double prev_inner_obj = inner_obj;
    static int stagnation_counter = 0;

    if (std::abs(inner_obj - prev_inner_obj) < 1e-6) {
        stagnation_counter++;
    } else {
        stagnation_counter = 0;  // Reset if there's improvement
    }
    prev_inner_obj = inner_obj;

    // If stagnation is detected, gradually move the threshold toward -1.5
    if (stagnation_counter >= STAGNATION_WINDOW) {
        double stagnation_factor =
            static_cast<double>(stagnation_counter - STAGNATION_WINDOW) /
            STAGNATION_WINDOW;
        threshold =
            threshold + stagnation_factor * (STAGNATION_TARGET - threshold);
    }

    // Ensure the threshold stays within bounds
    threshold = std::max(MIN_THRESHOLD, std::min(MAX_THRESHOLD, threshold));
    // print_info("Computed threshold: {}\n", threshold);
    return threshold;
}
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
        if (status != Status::Rollback) {
            rollback = shallUpdateStep();
        }
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
        auto threshold = -1.5;  // computeThreshold(iter, inner_obj, stats);
        if (inner_obj > threshold) {
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
    // Get references to avoid repeated lookups
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
    auto &arc_scores = assign_buckets<D>(fw_arc_scores, bw_arc_scores);
    auto &best_label = assign_buckets<D>(fw_best_label, bw_best_label);

    n_labels = 0;

    // Pre-calculate segment info using bit shift for better performance
    const size_t n_segments = (n_buckets + 63) >> 6;
    std::vector<uint64_t> Bvisited(n_segments);

    // Process SCCs in topological order
    for (const auto &scc_index : topological_order) {
        bool all_ext;
        do {
            all_ext = true;

            // Process buckets in sorted order within SCC
            for (const auto bucket : sorted_sccs[scc_index]) {
                // Cache bucket labels reference
                const auto &bucket_labels = buckets[bucket].get_labels();

                for (Label *label : bucket_labels) {
                    if (label->is_extended || label->is_dominated) continue;

                    // Early resource check for partial solutions
                    if constexpr (F != Full::PSTEP && F != Full::TSP) {
                        if constexpr (F == Full::Partial) {
                            const auto main_resource =
                                label->resources[options.main_resources[0]];
                            const auto q_star_value =
                                q_star[options.main_resources[0]];

                            if (((D == Direction::Forward) &&
                                 (main_resource > q_star_value)) ||
                                ((D == Direction::Backward) &&
                                 (main_resource <= q_star_value))) {
                                label->set_extended(true);
                                continue;
                            }
                        }
                    }

                    // Efficient visited bucket clearing
                    if (n_segments <= 8) {
                        std::fill_n(Bvisited.begin(), n_segments, 0);
                    } else {
                        std::memset(Bvisited.data(), 0,
                                    n_segments * sizeof(uint64_t));
                    }

                    // Check dominance in smaller buckets
                    if constexpr (F != Full::TSP) {
                        // random change of callong the function
                        // auto change = rand() % 100;
                        // if (change < 95) {
                        if (DominatedInCompWiseSmallerBuckets<D, S>(
                                label, bucket, c_bar, Bvisited, stat_n_dom)) {
                            // buckets[label->vertex].lazy_flush(label);
                            // label->set_extended(true);
                            label->set_dominated(true);
                            continue;
                            // }
                        }
                    }

                    // Process arcs for current label
                    auto node_id = label->node_id;

                    const auto &node_arcs =
                        nodes[node_id].template get_arcs<D>();

                    for (const auto &arc : node_arcs) {
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
                        arc_scores[node_id][arc] += extended_labels.size();

                        // Process each new label
                        for (Label *new_label : extended_labels) {
                            ++stat_n_labels;

                            const int to_bucket = new_label->vertex;
                            auto &mother_bucket = buckets[to_bucket];
                            const auto &to_bucket_labels =
                                mother_bucket.get_labels();

                            bool dominated = false;
                            if constexpr (S == Stage::One) {
                                // Stage One processing with early exits
                                for (auto *existing_label : to_bucket_labels) {
                                    if (label->cost < existing_label->cost) {
                                        mother_bucket.remove_label(
                                            existing_label);
                                    } else {
                                        dominated = true;
                                        break;
                                    }
                                }
                            } else {
                                // SIMD dominance check
                                auto check_dominance_in_bucket =
                                    [&](const std::vector<Label *> &labels) {
#ifndef __AVX2__
                                        return check_dominance_against_vector<
                                            D, S>(new_label, labels,
                                                  cut_storage,
                                                  options.resources.size());
#else
                                        for (auto *existing_label : labels) {
                                            if (existing_label->is_dominated) {
                                                continue;
                                            }
                                            if (is_dominated<D, S>(
                                                    new_label,
                                                    existing_label)) {
                                                ++stat_n_dom;
                                                return true;
                                            }
                                        }
                                        return false;
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
                                        if (existing_label->is_dominated) {
                                            continue;
                                        }
                                        if (is_dominated<D, S>(existing_label,
                                                               new_label)) {
                                            // buckets[to_bucket].remove_label(
                                            //     existing_label);
                                            existing_label->set_dominated(true);
                                        }
                                    }
                                }

                                ++n_labels;
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
                    label->set_extended(true);
                }
                // buckets[bucket].flush();
            }
        } while (!all_ext);

        // Update c_bar values efficiently
        for (const int bucket : sorted_sccs[scc_index]) {
            double min_c_bar = buckets[bucket].get_cb();
            for (const auto phi_bucket : Phi[bucket]) {
                min_c_bar = std::min(min_c_bar, c_bar[phi_bucket]);
            }
            c_bar[bucket] = min_c_bar;
        }
    }

    // Store best label
    if (Label *best_label_run =
            get_best_label<D>(topological_order, c_bar, sccs)) {
        best_label = best_label_run;
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

        // if constexpr (S == Stage::Four) {
        //     non_dominated_labels_per_bucket += bucket_labels.size();
        // }

        for (const Label *L : bucket_labels) {
            if (L->is_dominated) {
                continue;
            }
            non_dominated_labels_per_bucket++;
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

    if constexpr (S == Stage::Four) {
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
    }
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

            static thread_local std::vector<Label *> label_vector;
            label_vector.clear();

            auto jump_arcs =
                nodes[L_prime->node_id].template get_jump_arcs<D>(node_id);

            for (const auto &jump_arc : jump_arcs) {
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

#ifdef CUSTOM_COST
    auto distance = getcij(initial_node_id, node_id);
    double new_cost =
        cost_calculator.calculate_cost(initial_cost,  // InitialCost
                                       distance,      // Distance
                                       node_id,       // NodeId
                                       new_resources  // Resources
        );
#else
    double new_cost = initial_cost + getcij(initial_node_id, node_id);
#endif

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
        // L_prime->children.push_back(new_label);
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
    new_label->nodes_covered = L_prime->nodes_covered;
    new_label->nodes_covered.push_back(node_id);

#if defined(SRC)
    using simd_double = std::experimental::native_simd<double>;
    using simd_abi = std::experimental::simd_abi::native<double>;
    static constexpr size_t SIMD_WIDTH = simd_double::size();
    static constexpr size_t SMALL_SIZE = 256;

    new_label->SRCmap = L_prime->SRCmap;

    if constexpr (S == Stage::Four || S == Stage::Enumerate) {
        auto &cutter = cut_storage;
        const size_t segment = node_id >> 6;
        const size_t bit_position = node_id & 63;
        const uint64_t bit_mask = bit_mask_lookup[bit_position];

        // Get active cuts once
        const auto active_cuts = cutter->getActiveCuts();
        const auto cut_size = cutter->activeSize();
        const size_t simd_cut_size = cut_size - (cut_size % SIMD_WIDTH);

        double total_cost_update = 0.0;

#ifdef __AVX2__
        if (cut_size <= SMALL_SIZE) {
#endif

            // Small size: direct processing of active cuts
            for (const auto &active_cut : active_cuts) {
                const auto &cut = *active_cut.cut_ptr;
                const size_t idx = active_cut.index;
                const double dual_value = active_cut.dual_value;

                if (cut.neighbors[segment] & bit_mask) {
                    if (cut.baseSet[segment] & bit_mask) {
                        const auto &multipliers = cut.p;
                        auto &src_map_value = new_label->SRCmap[idx];
                        src_map_value +=
                            multipliers.num[cut.baseSetOrder[node_id]];
                        if (src_map_value >= multipliers.den) {
                            src_map_value -= multipliers.den;
                            total_cost_update -= dual_value;
                        }
                    }
                } else {
                    new_label->SRCmap[idx] = 0.0;
                }
            }

#ifdef __AVX2__
        } else {
            // SIMD processing for larger sizes
            alignas(64) std::array<double, SIMD_WIDTH> dual_values;
            alignas(64) std::array<size_t, SIMD_WIDTH> indices;
            alignas(64) std::array<const Cut *, SIMD_WIDTH> cut_ptrs;

            for (size_t base_idx = 0; base_idx < simd_cut_size;
                 base_idx += SIMD_WIDTH) {
                // Prepare SIMD batch
                for (size_t j = 0; j < SIMD_WIDTH; ++j) {
                    const auto &active_cut = active_cuts[base_idx + j];
                    dual_values[j] = active_cut.dual_value;
                    indices[j] = active_cut.index;
                    cut_ptrs[j] = active_cut.cut_ptr;
                }

                simd_double duals_vec = load_simd_values(
                    std::span<const double>(dual_values.data(), SIMD_WIDTH), 0);

                // Process SIMD batch
                for (size_t j = 0; j < SIMD_WIDTH; ++j) {
                    const auto &cut = *cut_ptrs[j];
                    const size_t idx = indices[j];

                    if (cut.neighbors[segment] & bit_mask) {
                        if (cut.baseSet[segment] & bit_mask) {
                            const auto &multipliers = cut.p;
                            auto &src_map_value = new_label->SRCmap[idx];
                            src_map_value +=
                                multipliers.num[cut.baseSetOrder[node_id]];
                            if (src_map_value >= multipliers.den) {
                                src_map_value -= multipliers.den;
                                total_cost_update -= dual_values[j];
                            }
                        }
                    } else {
                        new_label->SRCmap[idx] = 0.0;
                    }
                }
            }

            // Handle remaining active cuts
            for (size_t i = simd_cut_size; i < cut_size; ++i) {
                const auto &active_cut = active_cuts[i];
                const auto &cut = *active_cut.cut_ptr;
                const size_t idx = active_cut.index;
                const double dual_value = active_cut.dual_value;

                if (cut.neighbors[segment] & bit_mask) {
                    if (cut.baseSet[segment] & bit_mask) {
                        const auto &multipliers = cut.p;
                        auto &src_map_value = new_label->SRCmap[idx];
                        src_map_value +=
                            multipliers.num[cut.baseSetOrder[node_id]];
                        if (src_map_value >= multipliers.den) {
                            src_map_value -= multipliers.den;
                            total_cost_update -= dual_value;
                        }
                    }
                } else {
                    new_label->SRCmap[idx] = 0.0;
                }
            }
        }
#endif

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
inline bool BucketGraph::is_dominated(const Label *__restrict new_label,
                                      const Label *__restrict label) noexcept {
    // Early exit on cost check
    const double cost_diff = label->cost - new_label->cost;
    if (cost_diff > 0) {
        return false;
    }

    // Resource comparison using vectorization-friendly code
    const auto *__restrict new_resources = new_label->resources.data();
    const auto *__restrict label_resources = label->resources.data();

    for (size_t i = 0; i < options.resources.size(); ++i) {
        if constexpr (D == Direction::Forward) {
            // In Forward direction: the comparison label must not have more
            // resources than the new label
            if (label_resources[i] > new_resources[i]) {
                return false;
            }
        } else if constexpr (D == Direction::Backward) {
            // In Backward direction: the comparison label must not have
            // fewer resources than the new label
            if (label_resources[i] < new_resources[i]) {
                return false;
            }
        }
    }

    // Check visited nodes (bitmap comparison) for Stages 3, 4, and
    // Enumerate
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

#ifdef SRC
    if constexpr (S == Stage::Four || S == Stage::Enumerate) {
        const auto *__restrict label_srcs = label->SRCmap.data();
        const auto *__restrict new_label_srcs = new_label->SRCmap.data();

        // Process active cuts in batches for better cache utilization
        double sum = 0.0;
        const auto &active_cuts = cut_storage->getActiveCuts();
        const size_t num_cuts = active_cuts.size();

        // Process cuts in batches
        for (auto &cut : active_cuts) {
            if (label_srcs[cut.index] > new_label_srcs[cut.index]) {
                sum += cut.dual_value;
                if (cost_diff - sum > 0) {
                    return false;
                }
            }
        }
    }
#endif

    return true;
}
/**
 * @brief Checks if element 'a' precedes element 'b' in the given strongly
 * connected components (SCCs).
 *
 * This function takes a vector of SCCs and two elements 'a' and 'b' as
 * input. It searches for 'a' and 'b' in the SCCs and determines if 'a'
 * precedes 'b' in the SCC list.
 *
 */
template <typename T>
inline bool precedes(
    const std::vector<std::vector<int>> &sccs, const T a, const T b,
    const UnionFind &uf,
    ankerl::unordered_dense::map<std::pair<T, T>, bool> &cache) {
    // Create a key for the cache
    std::pair<T, T> key = std::make_pair(a, b);

    // Check if the result is already in the cache
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;  // Return the cached result
    }

    // Step 2: Check if element `a`'s SCC precedes element `b`'s SCC
    size_t rootA = uf.getSubset(a);
    size_t rootB = uf.getSubset(b);

    // Compute the result
    bool result = rootA < rootB;

    // Cache the result for future use
    cache[key] = result;

    return result;
}
/**
 * @brief Determines if a label is dominated in component-wise smaller
 * buckets.
 *
 * This function checks if a given label is dominated by any other label in
 * component-wise smaller buckets. The dominance is determined based on the
 * cost and order of the buckets.
 *
 */
template <Direction D, Stage S>
inline bool BucketGraph::DominatedInCompWiseSmallerBuckets(
    const Label *__restrict__ L, int bucket,
    const std::vector<double> &__restrict__ c_bar,
    std::vector<uint64_t> &__restrict__ Bvisited, int &stat_n_dom) noexcept {
    const auto &buckets = assign_buckets<D>(fw_buckets, bw_buckets);
    const auto &Phi = assign_buckets<D>(Phi_fw, Phi_bw);
    const auto &uf = assign_buckets<D>(fw_union_find, bw_union_find);
    const auto &sccs = assign_buckets<D>(fw_ordered_sccs, bw_ordered_sccs);
    auto &union_cache = assign_buckets<D>(fw_union_cache, bw_union_cache);

    // Cache frequently used values
    const int b_L = L->vertex;
    const double label_cost = L->cost;
    const size_t res_size = options.resources.size();

    // Stack-based implementation with dynamic size
    static thread_local std::vector<int> stack_buffer;
    stack_buffer.reserve(R_SIZE);
    stack_buffer.push_back(bucket);

    const auto check_dominance = [&](const std::vector<Label *> &labels) {
#ifndef __AVX2__
        return check_dominance_against_vector<D, S>(L, labels, cut_storage,
                                                    res_size);
#endif
        {
            for (auto &existing_label : labels) {
                if (!existing_label->is_dominated &&
                    is_dominated<D, S>(L, existing_label)) {
                    ++stat_n_dom;
                    return true;
                }
            }
            return false;
        }
    };

    // Main processing loop with optimizations
    while (!stack_buffer.empty()) {
        const int current_bucket = stack_buffer.back();
        stack_buffer.pop_back();

        // Optimize bit operations using pre-computed masks
        const size_t segment = static_cast<size_t>(current_bucket) >> 6;
        const uint64_t bit = bit_mask_lookup[current_bucket & 63];
        Bvisited[segment] |= bit;

        // Early exit check with likely hint for better branch
        // prediction
        if (label_cost < c_bar[current_bucket] &&
            ::precedes(sccs, current_bucket, b_L, uf, union_cache)) {
            return false;
        }

        // Check dominance only if necessary
        if (b_L != current_bucket) {
            const auto &mother_bucket = buckets[current_bucket];
            if (mother_bucket.check_dominance(L, check_dominance, stat_n_dom)) {
                return true;
            }
        }

        // Process neighbors with vectorization hints
        const auto &neighbors = Phi[current_bucket];

        for (auto b_prime : neighbors) {
            const size_t word_idx_prime = static_cast<size_t>(b_prime) >> 6;
            const uint64_t mask = bit_mask_lookup[b_prime & 63];

            if (!(Bvisited[word_idx_prime] & mask)) {
                stack_buffer.push_back(b_prime);
            }
        }
    }

    return false;
}

/**
 * @brief Runs forward and backward labeling algorithms in parallel and
 * synchronizes the results.
 *
 * T#his function creates tasks for forward and backward labeling algorithms
 * using the provided scheduling mechanism. The tasks are executed in
 * parallel, and the results are synchronized and stored in the provided
 * vectors.
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
    // forward_cbar = labeling_algorithm<Direction::Forward, state,
    // fullness>(); backward_cbar = labeling_algorithm<Direction::Backward,
    // state, fullness>();
    //
    // #pragma omp parallel sections
    //     {
    // #pragma omp section
    //         {
    //             forward_cbar =
    //                 labeling_algorithm<Direction::Forward, state,
    //                 fullness>();
    //         }
    // #pragma omp section
    //         {
    //             backward_cbar =
    //                 labeling_algorithm<Direction::Backward, state,
    //                 fullness>();
    //         }
    //     }
}

/**
 * Computes a new label based on the given labels L and L_prime.
 *
 */
template <Stage S>
Label *BucketGraph::compute_label(const Label *L, const Label *L_prime,
                                  double red_cost) {
    double cij_cost = getcij(L->node_id, L_prime->node_id);
    double real_cost = L->real_cost + L_prime->real_cost + cij_cost;

    // Directly acquire new_label and set the cost
    auto new_label = label_pool_fw->acquire();
    new_label->cost = red_cost;
    new_label->real_cost = real_cost;
    new_label->nodes_covered.clear();

    new_label->nodes_covered = L_prime->nodes_covered;
    //     reverse the nodes_covered
    std::reverse(new_label->nodes_covered.begin(),
                 new_label->nodes_covered.end());
    new_label->nodes_covered.insert(new_label->nodes_covered.begin(),
                                    L->nodes_covered.begin(),
                                    L->nodes_covered.end());

    return new_label;
}
