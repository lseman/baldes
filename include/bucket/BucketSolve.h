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
template <Symmetry SYM>
inline std::vector<Label *> BucketGraph::solve(bool trigger) {
    // Start with a nonoptimal status.
    status = Status::NotOptimal;
    if (trigger) {
        transition = true;
        fixed = false;
    }

    updateSplit();  // Update split values for the bucket graph.
    std::vector<Label *> paths;

    //////////////////////////////////////////////////////////////////////
    // Adaptive Stage Handling
    //////////////////////////////////////////////////////////////////////

    // ----- Stage 1: Light Heuristic -----
    if (s1 && depth == 0) {
        stage = 1;
        paths = bi_labeling_algorithm<Stage::One>();
        // inner_obj = paths[0]->cost; // (if available)
        if (inner_obj >= -1 || iter >= 10) {
            s1 = false;
            s2 = true;  // Transition to Stage 2.
        }
    }
    // ----- Stage 2: Expensive Pricing Heuristic -----
    else if (s2 && depth == 0) {
        stage = 2;
        paths = bi_labeling_algorithm<Stage::Two>();
        // inner_obj = paths[0]->cost; // (if available)
        if (inner_obj >= -10 || iter > 500) {
            s2 = false;
            s3 = true;  // Transition directly to Stage 4.
        }
    }
    // ----- Stage 3: Heuristic Fixing Approach -----
    else if (s3 && depth == 0) {
        stage = 3;
        paths = bi_labeling_algorithm<Stage::Three>();
        // inner_obj = paths[0]->cost; // (if available)
        if (inner_obj >= -0.5) {
            s3 = false;
            s4 = true;
            transition = true;  // Prepare for Stage 4 transition.
        }
    }
    // ----- Stage 4: Exact Labeling with Fixing Enabled -----
    else if (s4 && depth == 0) {
        stage = 4;
#ifdef FIX_BUCKETS
        if (transition) {
            // During a transition to Stage 4, fix buckets temporarily.
            bool original_fixed = fixed;
            fixed = true;
            paths = bi_labeling_algorithm<Stage::Four>();
            transition = false;
            fixed = original_fixed;
            min_red_cost = paths[0]->cost;  // Update minimum reduced cost.
            iter++;
            return paths;
        }
#endif
        // Standard Stage 4 processing.
        paths = bi_labeling_algorithm<Stage::Four>();
        // inner_obj = paths[0]->cost; // (if available)
        bool rollback = false;
        if (status != Status::Rollback) {
            rollback = shallUpdateStep();
        }
        if (rollback) {
            status = Status::Rollback;
            return paths;
        }
        // Compute a new threshold for separation based on stats.
        threshold = stats.computeThreshold(iter, inner_obj);
        // If objective is sufficiently high, enter separation mode.
        if (inner_obj > threshold) {
            ss = true;
#if !defined(SRC) && !defined(SRC3)
            status = Status::Optimal;
            return paths;
#endif
            status = Status::Separation;
        }
    }
    // ----- Stage 5: Enumeration -----
    else if (s5 && depth == 0) {
        stage = 5;
        print_info("Starting enumeration with gap {}\n", gap);
        paths = bi_labeling_algorithm<Stage::Enumerate>();
        print_info("Finished enumeration with {} paths\n", paths.size());
        status = Status::Optimal;
    }

    // Increment the iteration counter.
    iter++;
    return paths;
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
    // Get references to internal bucket structures to avoid repeated lookups.
    auto &buckets = assign_buckets<D>(fw_buckets, bw_buckets);
    const auto &ordered_sccs =
        assign_buckets<D>(fw_ordered_sccs, bw_ordered_sccs);
    const auto &topological_order =
        assign_buckets<D>(fw_topological_order, bw_topological_order);
    const auto &sccs = assign_buckets<D>(fw_sccs, bw_sccs);
    const auto &Phi = assign_buckets<D>(Phi_fw, Phi_bw);
    auto &c_bar = assign_buckets<D>(fw_c_bar, bw_c_bar);
    auto &n_labels = assign_buckets<D>(n_fw_labels, n_bw_labels);
    const auto &sorted_sccs = assign_buckets<D>(fw_sccs_sorted, bw_sccs_sorted);
    const auto &n_buckets = assign_buckets<D>(fw_buckets_size, bw_buckets_size);
    auto &stat_n_labels = assign_buckets<D>(stat_n_labels_fw, stat_n_labels_bw);
    auto &stat_n_dom = assign_buckets<D>(stat_n_dom_fw, stat_n_dom_bw);
    auto &arc_scores = assign_buckets<D>(fw_arc_scores, bw_arc_scores);
    auto &best_label = assign_buckets<D>(fw_best_label, bw_best_label);

    // Reset the total label count.
    n_labels = 0;

    // Pre-calculate bitmap segments for dominance-checking.
    const size_t n_segments = (n_buckets + 63) >> 6;
    std::vector<uint64_t> Bvisited(n_segments, 0);

    // Process each strongly connected component (SCC) in topological order.
    for (const auto &scc_index : topological_order) {
        bool all_ext = false;
        do {
            all_ext = true;  // Assume all labels are extended unless a new
                             // extension is made.

            // Process each bucket in sorted order within the current SCC.
            const auto &scc_buckets = sorted_sccs[scc_index];
            for (const auto bucket : scc_buckets) {
                // Retrieve all labels from the current bucket.
                const auto &bucket_labels = buckets[bucket].get_active_labels();

                // Process each label in this bucket.
                for (Label *label : bucket_labels) {
                    // Skip labels that are already extended or dominated.
                    // if (label->is_extended || label->is_dominated) continue;

                    // For partial solutions (if not PSTEP or TSP), check
                    // resource feasibility.
                    if constexpr (F != Full::PSTEP && F != Full::TSP) {
                        if constexpr (F == Full::Partial) {
                            const auto main_resource =
                                label->resources[options.main_resources[0]];
                            const auto q_star_value =
                                std::floor(q_star[options.main_resources[0]]);
                            if ((D == Direction::Forward &&
                                 main_resource > q_star_value) ||
                                (D == Direction::Backward &&
                                 main_resource <= q_star_value)) {
                                label->set_extended(true);
                                continue;
                            }
                        }
                    }

                    // Clear Bvisited bitmap for this label's dominance check.
                    if (n_segments <= 8)
                        std::fill(Bvisited.begin(), Bvisited.end(), 0);
                    else
                        std::memset(Bvisited.data(), 0,
                                    n_segments * sizeof(uint64_t));

                    // Check if this label is dominated by labels in smaller
                    // buckets.
                    if constexpr (F != Full::TSP) {
                        // if constexpr (S > Stage::Three) {
                        if (DominatedInCompWiseSmallerBuckets<D, S>(
                                label, bucket, c_bar, Bvisited, stat_n_dom)) {
                            // label->set_dominated(true);
                            continue;
                        }
                        // }
                    }

                    // Process outgoing arcs for the current label.
                    const int node_id = label->node_id;
                    const auto &node_arcs =
                        nodes[node_id].template get_arcs<D>();

                    for (const auto &arc : node_arcs) {
                        // Extend the label along the arc.
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
                        // Update arc score with the number of extended labels.
                        arc_scores[node_id][arc] += extended_labels.size();

                        // Process each new label produced by the extension.
                        for (Label *new_label : extended_labels) {
                            ++stat_n_labels;
                            const int to_bucket = new_label->vertex;
                            auto &mother_bucket = buckets[to_bucket];
                            const auto &to_bucket_labels =
                                mother_bucket.get_non_dominated_labels();

                            bool dominated = false;
                            if constexpr (S == Stage::One) {
                                // In Stage One, use an early exit: mark labels
                                // with higher cost as dominated.
                                for (auto *existing_label : to_bucket_labels) {
                                    if (label->cost < existing_label->cost)
                                        existing_label->set_dominated(true);
                                    else {
                                        dominated = true;
                                        break;
                                    }
                                }
                            } else {
                                // For later stages, use a SIMD-friendly
                                // dominance check.
                                auto check_dominance_in_bucket =
                                    [&](const std::vector<Label *> &labels,
                                        int &stat_n_dom) {
#ifndef __AVX2__
                                        if (labels.size() >= 32)
                                            return check_dominance_against_vector<
                                                D, S>(new_label, labels,
                                                      cut_storage,
                                                      options.resources.size(),
                                                      stat_n_dom);
                                        else
#endif
                                        {
                                            for (auto *existing_label :
                                                 labels) {
                                                if (!existing_label
                                                         ->is_dominated &&
                                                    is_dominated<D, S>(
                                                        new_label,
                                                        existing_label)) {
                                                    ++stat_n_dom;
                                                    return true;
                                                }
                                            }
                                            return false;
                                        }
                                    };
                                dominated = mother_bucket.check_dominance(
                                    new_label, check_dominance_in_bucket,
                                    stat_n_dom);
                            }

                            if (!dominated) {
                                // Recursively mark labels dominated by
                                // new_label.
                                auto set_dominated_recursive =
                                    [](auto &self, Label *lbl) -> void {
                                    lbl->set_dominated(true);
                                    for (auto *child : lbl->children)
                                        self(self, child);
                                };

                                for (auto *existing_label : to_bucket_labels) {
                                    if (existing_label->is_dominated) continue;
                                    if (is_dominated<D, S>(existing_label,
                                                           new_label)) {
                                        existing_label->set_dominated(true);
                                        set_dominated_recursive(
                                            set_dominated_recursive,
                                            existing_label);
                                    }
                                }
                                ++n_labels;
#ifdef SORTED_LABELS
                                mother_bucket.add_sorted_label(new_label);
#elif defined(LIMITED_BUCKETS)
                                mother_bucket.sorted_label(new_label,
                                                           BUCKET_CAPACITY);
#else
                                mother_bucket.add_label(new_label);
#endif
                                // A new extension was successfully added.
                                all_ext = false;
                            }
                        }  // End for each extended label.
                    }  // End for each arc.

                    // Mark the label as extended (processed).
                    label->set_extended(true);
                }  // End for each label in bucket.
            }  // End for each bucket in current SCC.

            // Update c_bar values for all buckets in the current SCC.
            for (const int bucket : sorted_sccs[scc_index]) {
                double min_c_bar = buckets[bucket].get_cb();
                for (const auto phi_bucket : Phi[bucket]) {
                    min_c_bar = std::min(min_c_bar, c_bar[phi_bucket]);
                }
                c_bar[bucket] = min_c_bar;
            }
        } while (!all_ext);

    }  // End for each SCC (in topological order).

    // Store the best label found from the labeling process.
    if (Label *best_label_run =
            get_best_label<D>(topological_order, c_bar, sccs)) {
        best_label = best_label_run;
    }

    // Return the final c_bar vector.
    return c_bar;
}

/**
 * Performs the bi-labeling algorithm on the BucketGraph.
 *
 */

template <Stage S, Symmetry SYM>
std::vector<Label *> BucketGraph::bi_labeling_algorithm() {
    // === Stage-specific Initializations ===
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

    // Reset pools and perform common initialization.
    reset_pool();
    common_initialization();

    // Pre-allocate c-bar vectors for forward and backward buckets.
    std::vector<double> forward_cbar(fw_buckets.size());
    std::vector<double> backward_cbar(bw_buckets.size());

    // Run the labeling algorithms.
    if constexpr (SYM == Symmetry::Asymmetric) {
        run_labeling_algorithms<S, Full::Partial>(forward_cbar, backward_cbar);
    } else {
        forward_cbar =
            labeling_algorithm<Direction::Forward, S, Full::Partial>();
    }

    // Acquire an initial best label from the forward label pool.
    auto best_label = label_pool_fw->acquire();
    if constexpr (SYM == Symmetry::Asymmetric) {
        // Compute the best label if the forward and backward labels are
        // feasible.
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

    // Declare atomics to track the best cost and non-dominated label count.
    std::atomic<double> best_cost{0.0};
    std::atomic<size_t> non_dominated_labels_per_bucket{0};

    // === Parallel Bucket Processing ===
    // Choose a chunk size based on hardware concurrency.
    const size_t chunk_size =
        fw_buckets_size / std::thread::hardware_concurrency();
    const size_t n_chunks = (fw_buckets_size + chunk_size - 1) / chunk_size;

    auto bulk_sender = stdexec::bulk(
        stdexec::just(), n_chunks,
        [this, chunk_size, &best_cost,
         &non_dominated_labels_per_bucket](std::size_t chunk_idx) {
            const size_t start_bucket = chunk_idx * chunk_size;
            const size_t end_bucket =
                std::min(start_bucket + chunk_size,
                         static_cast<size_t>(fw_buckets_size));

            // Process each bucket within the current chunk.
            for (size_t bucket = start_bucket; bucket < end_bucket; ++bucket) {
                // Process non-dominated labels in the current bucket.
                const auto &bucket_labels = fw_buckets[bucket].get_labels();
                for (const Label *L : bucket_labels) {
                    if (L->is_dominated) continue;

                    non_dominated_labels_per_bucket.fetch_add(
                        1, std::memory_order_relaxed);

                    // Retrieve the outgoing arcs for the current label.
                    const auto &to_arcs =
                        nodes[L->node_id].get_arcs<Direction::Forward>();
                    for (const auto &arc : to_arcs) {
                        const int to_node = arc.to;
                        if constexpr (S == Stage::Three ||
                                      S == Stage::Eliminate) {
                            if (is_arc_fixed(L->node_id, to_node)) continue;
                        }

                        // Extend the label along the current arc.
                        auto extended_labels =
                            Extend<Direction::Forward, S, ArcType::Node,
                                   Mutability::Const, Full::Reverse>(L, arc);

                        // Process each extended label.
                        for (Label *L_prime : extended_labels) {
                            int bucket_to_process =
                                L_prime->vertex;  // mutable copy if needed

                            // Concatenate the label with bucket information,
                            // updating best_cost.
                            ConcatenateLabel<S, SYM>(L, bucket_to_process,
                                                     best_cost);
                        }
                    }
                }
            }
        });

    // Submit the bulk work to the merge scheduler and wait for completion.
    auto work = stdexec::starts_on(merge_sched, bulk_sender);
    stdexec::sync_wait(std::move(work));

    // === Post-processing and Sorting ===
    // Sort merged labels based on cost (ascending order).
    pdqsort(merged_labels.begin(), merged_labels.end(),
            [](const Label *a, const Label *b) { return a->cost < b->cost; });

#ifdef SCHRODINGER
    // If using the Schrödinger approach, process additional paths.
    if (merged_labels.size() > N_ADD) {
        std::vector<Path> paths;
        const int labels_size = merged_labels.size();
        const int end_idx = std::min(N_ADD + N_ADD, labels_size);
        paths.reserve(end_idx - N_ADD);
        for (int i = N_ADD; i < end_idx; ++i) {
            if (merged_labels[i]->nodes_covered.size() <= 3) continue;
            paths.emplace_back(merged_labels[i]->nodes_covered,
                               merged_labels[i]->real_cost);
        }
        sPool.add_paths(paths);
        sPool.iterate();
    }
#endif

#ifdef RIH
    // If using RIH stage modifications, submit the top labels to ILS.
    if constexpr (S == Stage::Four) {
        std::vector<Label *> top_labels;
        top_labels.reserve(N_ADD);
        const int n_candidates =
            std::min(N_ADD, static_cast<int>(merged_labels.size()));
        for (int i = 0; i < n_candidates; ++i) {
            if (merged_labels[i]->nodes_covered.size() <= 3) continue;
            top_labels.push_back(merged_labels[i]);
        }
        ils->submit_task(top_labels, nodes);
    }
#endif

    // Store the cost of the best (first) merged label as the inner objective.
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
    // Prepare temporary resource storage (resize to current resources size)
    static thread_local std::vector<double> new_resources;
    const size_t n_resources = options.resources.size();
    new_resources.resize(n_resources);

    // Cache initial values from L_prime
    const int initial_node_id = L_prime->node_id;
    auto initial_resources = L_prime->resources;  // Copy resources from L_prime
    const double initial_cost = L_prime->cost;

    int node_id;
    if constexpr (A == ArcType::Bucket) {
        node_id =
            assign_buckets<D>(fw_buckets, bw_buckets)[gamma.to_bucket].node_id;
    } else if constexpr (A == ArcType::Node) {
        node_id = gamma.to;
    } else {  // Jump arc
        auto &buckets = assign_buckets<D>(fw_buckets, bw_buckets);
        node_id = buckets[gamma.jump_bucket].node_id;
        const auto &bucket = buckets[gamma.jump_bucket];
        // Update resource bounds based on jump bucket properties
        for (size_t i = 0; i < n_resources; ++i) {
            if constexpr (D == Direction::Forward) {
                initial_resources[i] = std::max(
                    initial_resources[i], static_cast<double>(bucket.lb[i]));
            } else {
                initial_resources[i] = std::min(
                    initial_resources[i], static_cast<double>(bucket.ub[i]));
            }
        }
    }

    // Early rejection: if we loop back or the node was already visited.
    if (node_id == L_prime->node_id ||
        is_node_visited(L_prime->visited_bitmap, node_id))
        return {};

    // For stages where fixed arcs matter.
    if constexpr (S == Stage::Three || S == Stage::Eliminate) {
        if ((D == Direction::Forward &&
             is_arc_fixed(initial_node_id, node_id)) ||
            (D == Direction::Backward &&
             is_arc_fixed(node_id, initial_node_id)))
            return {};
    }

    // Process resources (feasibility check)
    if constexpr (F != Full::TSP) {
        if (!process_all_resources<D>(new_resources, initial_resources, gamma,
                                      nodes[node_id], n_resources))
            return {};
    }

    // For PSTEP/TSP: Check path length constraints.
    if constexpr (F == Full::PSTEP || F == Full::TSP) {
        int n_visited = 0;
        for (const auto &bitmap : L_prime->visited_bitmap)
            n_visited += __builtin_popcountll(bitmap);
        if (n_visited > options.max_path_size ||
            (n_visited == options.max_path_size &&
             node_id != options.end_depot))
            return {};
    }

    const int to_bucket = get_bucket_number<D>(node_id, new_resources);

#ifdef FIX_BUCKETS
    // If bucket is fixed (for Stage::Four) and not a jump arc, try jump arcs
    // recursively.
    if constexpr (S == Stage::Four && A != ArcType::Jump) {
        if (is_bucket_fixed<D>(L_prime->vertex, to_bucket)) {
            // if (depth > 1) return {};
            static thread_local std::vector<Label *> label_vector;
            label_vector.clear();
            auto jump_arcs =
                nodes[L_prime->node_id].template get_jump_arcs<D>(node_id);
            for (const auto &jump_arc : jump_arcs) {
                auto extended_labels =
                    Extend<D, S, ArcType::Jump, Mutability::Const, F>(
                        L_prime, jump_arc, depth + 1);
                if (!extended_labels.empty())
                    label_vector.insert(label_vector.end(),
                                        extended_labels.begin(),
                                        extended_labels.end());
            }
            return label_vector;
        }
    }
#endif

#ifdef CUSTOM_COST
    const auto distance = getcij(initial_node_id, node_id);
    double new_cost = cost_calculator.calculate_cost(initial_cost, distance,
                                                     node_id, new_resources);
#else
    double new_cost = initial_cost + getcij(initial_node_id, node_id);
#endif

    const auto &VRPNode = nodes[node_id];
    if constexpr (F != Full::PSTEP) {
        new_cost -= VRPNode.cost;
    } else {
        int n_visited = 0;
        for (const auto &bitmap : L_prime->visited_bitmap)
            n_visited += __builtin_popcountll(bitmap);
        if (n_visited > 1 && initial_node_id != options.depot)
            new_cost += pstep_duals.getThreeTwoDualValue(initial_node_id) +
                        pstep_duals.getThreeThreeDualValue(initial_node_id);
        new_cost += -pstep_duals.getThreeTwoDualValue(node_id) +
                    pstep_duals.getArcDualValue(initial_node_id, node_id);
    }

    // Apply branching dual adjustments if present.
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

    // Handle Reverse mode early.
    if constexpr (F == Full::Reverse) {
        if (new_resources[options.main_resources[0]] <=
            std::floor(q_star[options.main_resources[0]]))
            return {};
        // auto &label_pool = assign_buckets<D>(label_pool_fw, label_pool_bw);
        auto new_label = new Label();
        if (to_bucket == -1) {
            fmt::print("Warning: to_bucket is -1\n");
        }
        new_label->vertex = to_bucket;
        return {new_label};
    }

    // Create and initialize a new label.
    auto &label_pool = assign_buckets<D>(label_pool_fw, label_pool_bw);
    auto new_label = label_pool->acquire();
    new_label->initialize(to_bucket, new_cost, new_resources, node_id);
    new_label->vertex = to_bucket;
    new_label->real_cost =
        L_prime->real_cost + getcij(initial_node_id, node_id);

    if constexpr (M == Mutability::Mut) L_prime->children.push_back(new_label);

    // Update visited bitmap: intersect parent's visited bits with neighborhood
    // mask.
    if constexpr (F != Full::PSTEP) {
        for (size_t i = 0; i < new_label->visited_bitmap.size(); ++i) {
            const uint64_t current_visited = L_prime->visited_bitmap[i];
            if (current_visited)
                new_label->visited_bitmap[i] =
                    current_visited & neighborhoods_bitmap[node_id][i];
        }
    } else {
        new_label->visited_bitmap = L_prime->visited_bitmap;
    }
    set_node_visited(new_label->visited_bitmap, node_id);

    // Update the path (nodes covered) by copying parent's path and adding
    // current node.
    new_label->nodes_covered = L_prime->nodes_covered;
    new_label->nodes_covered.push_back(node_id);

#if defined(SRC)
    if constexpr (S == Stage::Four || S == Stage::Enumerate) {
        new_label->SRCmap = L_prime->SRCmap;
        auto &cutter = cut_storage;
        const auto active_cuts = cutter->getActiveCuts();
        const auto n_cuts = cutter->activeSize();

        if (n_cuts == 0) return {new_label};

        double total_cost_update = 0.0;
        const size_t segment = node_id >> 6;
        const size_t bit_position = node_id & 63;

        auto &masks = cutter->getSegmentMasks();
        const auto cut_limit_mask = masks.get_cut_limit_mask();
        uint64_t valid_cuts = masks.get_valid_cut_mask(segment, bit_position);
        valid_cuts &= cut_limit_mask;  // Safeguard

        while (valid_cuts) {
            const int cut_idx = __builtin_ctzll(valid_cuts);
            const auto &active_cut = active_cuts[cut_idx];
            const auto &cut = *active_cut.cut_ptr;
            auto &src_map_value = new_label->SRCmap[active_cut.index];
            src_map_value += cut.p.num[cut.baseSetOrder[node_id]];
            if (src_map_value >= cut.p.den) {
                src_map_value -= cut.p.den;
                total_cost_update -= active_cut.dual_value;
            }
            valid_cuts &= (valid_cuts - 1);
        }

        uint64_t to_clear =
            (~masks.get_neighbor_mask(segment, bit_position)) & cut_limit_mask;
        while (to_clear) {
            const int cut_idx = __builtin_ctzll(to_clear);
            if (cut_idx >= n_cuts) {
                fmt::print("Warning: Invalid clear_idx {} >= n_cuts {}\n",
                           cut_idx, n_cuts);
                break;
            }
            new_label->SRCmap[active_cuts[cut_idx].index] = 0.0;
            to_clear &= (to_clear - 1);
        }
        new_label->cost += total_cost_update;
    }
#endif

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
    constexpr double TOLERANCE = 1e-3;

    // A label cannot dominate if its cost is higher.
    const double cost_diff = label->cost - new_label->cost;
    if (cost_diff > TOLERANCE) {
        return false;
    }

    // Compare resource vectors.
    const auto *__restrict new_res = new_label->resources.data();
    const auto *__restrict lbl_res = label->resources.data();
    const size_t n_res = options.resources.size();

    if constexpr (D == Direction::Forward) {
        // In the Forward direction, each resource value of 'label' must not
        // exceed that of 'new_label'
        for (size_t i = 0; i < n_res; ++i) {
            if (lbl_res[i] > new_res[i]) {
                return false;
            }
        }
    } else if constexpr (D == Direction::Backward) {
        // In the Backward direction, each resource value of 'label' must not be
        // less than that of 'new_label'
        for (size_t i = 0; i < n_res; ++i) {
            if (lbl_res[i] < new_res[i]) {
                return false;
            }
        }
    }

    // visits of label.
    if constexpr (S == Stage::Three || S == Stage::Four ||
                  S == Stage::Enumerate) {
        const size_t n_bitmap = label->visited_bitmap.size();
        for (size_t i = 0; i < n_bitmap; ++i) {
            // Every visited node in 'label' must also be visited in
            // 'new_label'.
            if ((label->visited_bitmap[i] & new_label->visited_bitmap[i]) !=
                label->visited_bitmap[i]) {
                return false;
            }
        }
    }

#ifdef SRC
    // For Stage Four or Enumerate, apply additional SRC-based cost adjustments.
    if constexpr (S == Stage::Four || S == Stage::Enumerate) {
        const auto *__restrict lbl_srcs = label->SRCmap.data();
        const auto *__restrict new_lbl_srcs = new_label->SRCmap.data();
        const auto &active_cuts = cut_storage->getActiveCuts();
        double dual_sum = 0.0;

        // Accumulate the dual values from active cuts if label's SRC exceeds
        // new_label's.
        for (const auto &cut : active_cuts) {
            if (lbl_srcs[cut.index] > new_lbl_srcs[cut.index]) {
                dual_sum += cut.dual_value;
                // If the cost difference minus the accumulated duals is still
                // positive, new_label is not sufficiently dominating.
                if (cost_diff - dual_sum > TOLERANCE) {
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
    const std::vector<std::vector<int>> &sccs,
    const std::vector<int> &topological_order, const T a, const T b,
    const UnionFind &uf,
    ankerl::unordered_dense::map<std::pair<T, T>, bool> &cache) {
    // Create a cache key
    const std::pair<T, T> key{a, b};
    if (const auto it = cache.find(key); it != cache.end()) {
        return it->second;
    }

    // Use the UnionFind's built-in comparison which already has the correct
    // ordering
    bool result = uf.compareSubsets(a, b);
    cache.emplace(key, result);
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
    // Get direction-specific data once.
    const auto &buckets = assign_buckets<D>(fw_buckets, bw_buckets);
    const auto &Phi = assign_buckets<D>(Phi_fw, Phi_bw);
    const auto &uf = assign_buckets<D>(fw_union_find, bw_union_find);
    const auto &sccs = assign_buckets<D>(fw_ordered_sccs, bw_ordered_sccs);
    auto &union_cache = assign_buckets<D>(fw_union_cache, bw_union_cache);

    const auto &topological_order =
        assign_buckets<D>(fw_topological_order, bw_topological_order);

    // Cache key label properties.
    const int b_L = L->vertex;
    const double label_cost = L->cost;
    const size_t res_size = options.resources.size();

    // Use a thread-local stack to traverse buckets.
    static thread_local std::vector<int> stack_buffer;
    stack_buffer.clear();  // Clear previous state.
    stack_buffer.reserve(R_SIZE);
    stack_buffer.push_back(bucket);

    // Precompute pointer to bit mask lookup for fast bit operations.
    const uint64_t *bit_mask_lookup_ptr = bit_mask_lookup.data();

    // Lambda to check if any label in a bucket dominates L.
    const auto check_dominance = [&](const std::vector<Label *> &labels,
                                     int &stat_n_dom) -> bool {
#ifndef __AVX2__
        if (labels.size() >= 32) {
            return check_dominance_against_vector<D, S>(L, labels, cut_storage,
                                                        res_size, stat_n_dom);
        } else
#endif
        {
            for (auto *existing_label : labels) {
                if (!existing_label->is_dominated &&
                    is_dominated<D, S>(L, existing_label)) {
                    ++stat_n_dom;
                    return true;
                }
            }
            return false;
        }
    };

    // Main processing loop: traverse buckets using DFS.
    while (!stack_buffer.empty()) {
        const int current_bucket = stack_buffer.back();
        stack_buffer.pop_back();

        // Mark current_bucket as visited using precomputed bit masks.
        const size_t segment = static_cast<size_t>(current_bucket) >> 6;
        const uint64_t bit = bit_mask_lookup_ptr[current_bucket & 63];
        Bvisited[segment] |= bit;

        // Early exit: if label cost is lower than the bucket's c_bar and
        // the bucket precedes L in the SCC order, then L is not dominated.
        if (label_cost < c_bar[current_bucket] &&
            ::precedes(sccs, topological_order, current_bucket, b_L, uf,
                       union_cache)) {
            return false;
        }

        // Check for dominance in this bucket if it's not the label's own
        // bucket.
        if (b_L != current_bucket) {
            const auto &mother_bucket = buckets[current_bucket];
            if (mother_bucket.check_dominance(L, check_dominance, stat_n_dom)) {
                return true;
            }
        }

        // Process neighbor buckets.
        const auto &neighbors = Phi[current_bucket];
        for (auto b_prime : neighbors) {
            const size_t word_idx = static_cast<size_t>(b_prime) >> 6;
            const uint64_t mask = bit_mask_lookup_ptr[b_prime & 63];
            if (!(Bvisited[word_idx] & mask)) {
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
    // Schedule the forward labeling algorithm task on the scheduler.
    auto forward_task =
        stdexec::schedule(bi_sched) | stdexec::then([&]() {
            return labeling_algorithm<Direction::Forward, state, fullness>();
        });

    // Schedule the backward labeling algorithm task on the scheduler.
    auto backward_task =
        stdexec::schedule(bi_sched) | stdexec::then([&]() {
            return labeling_algorithm<Direction::Backward, state, fullness>();
        });

    // Combine the forward and backward tasks.
    auto combined_work =
        stdexec::when_all(std::move(forward_task), std::move(backward_task)) |
        stdexec::then([&](auto forward_result, auto backward_result) {
            // Store the results into the provided vectors.
            forward_cbar = std::move(forward_result);
            backward_cbar = std::move(backward_result);
        });

    // Wait for all tasks to complete.
    stdexec::sync_wait(std::move(combined_work));
}

/**
 * Computes a new label based on the given labels L and L_prime.
 *
 */
template <Stage S>
Label *BucketGraph::compute_label(const Label *L, const Label *L_prime,
                                  double red_cost) {
    // Compute cost values
    double cij_cost = getcij(L->node_id, L_prime->node_id);
    double real_cost = L->real_cost + L_prime->real_cost + cij_cost;

    // Acquire a new label from the pool and initialize its cost fields.
    auto new_label = new Label();
    new_label->cost = red_cost;
    new_label->real_cost = real_cost;

    // Efficiently combine nodes_covered: preallocate sufficient space to avoid
    // reallocation.
    new_label->nodes_covered.clear();
    new_label->nodes_covered.reserve(L->nodes_covered.size() +
                                     L_prime->nodes_covered.size());

    // First add the nodes from L.
    new_label->nodes_covered.insert(new_label->nodes_covered.end(),
                                    L->nodes_covered.begin(),
                                    L->nodes_covered.end());

    // Then add the nodes from L_prime in reverse order using reverse iterators.
    new_label->nodes_covered.insert(new_label->nodes_covered.end(),
                                    L_prime->nodes_covered.rbegin(),
                                    L_prime->nodes_covered.rend());

    return new_label;
}
