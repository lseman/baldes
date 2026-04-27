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
        fixed      = false;
    }

    updateSplit(); // Update split values for the bucket graph.
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
            s2 = true; // Transition to Stage 2.
        }
    }
    // ----- Stage 2: Expensive Pricing Heuristic -----
    else if (s2 && depth == 0) {
        stage = 2;
        paths = bi_labeling_algorithm<Stage::Two>();
        // inner_obj = paths[0]->cost; // (if available)
        if (inner_obj >= -10 || iter > 500) {
            s2 = false;
            s3 = true; // Transition directly to Stage 4.
        }
    }
    // ----- Stage 3: Heuristic Fixing Approach -----
    else if (s3 && depth == 0) {
        stage = 3;
        paths = bi_labeling_algorithm<Stage::Three>();
        // inner_obj = paths[0]->cost; // (if available)
        if (inner_obj >= -0.5) {
            s3         = false;
            s4         = true;
            transition = true; // Prepare for Stage 4 transition.
        }
    }
    // ----- Stage 4: Exact Labeling with Fixing Enabled -----
    else if (s4 && depth == 0) {
        stage = 4;
#ifdef FIX_BUCKETS
        if (transition) {
            // During a transition to Stage 4, fix buckets temporarily.
            bool original_fixed = fixed;
            fixed               = true;
            paths               = bi_labeling_algorithm<Stage::Four>();
            transition          = false;
            fixed               = original_fixed;
            min_red_cost        = paths[0]->cost; // Update minimum reduced cost.
            iter++;
            return paths;
        }
#endif
        // Standard Stage 4 processing.
        paths = bi_labeling_algorithm<Stage::Four>();
        // inner_obj = paths[0]->cost; // (if available)
        bool rollback = false;
        if (status != Status::Rollback) {
            // rollback = shallUpdateStep();
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

    updateSplit(); // Update the split values for the bucket graph

    // Placeholder for the final paths (labels) and inner objective value
    std::vector<Label *> paths;
    double               inner_obj;

    //////////////////////////////////////////////////////////////////////
    // ADAPTIVE STAGE HANDLING
    //////////////////////////////////////////////////////////////////////
    // Stage 1: Apply a light heuristic (Stage::One)
    if (s1) {
        stage = 1;
        paths = bi_labeling_algorithm<Stage::One>(); // Solve the problem with
                                                     // Stage 1 heuristic
        inner_obj = paths[0]->cost;

        // Transition from Stage 1 to Stage 2 if the objective improves or after
        // 10 iterations
        if (inner_obj >= -5 || iter >= 10) {
            s1 = false;
            s2 = true; // Move to Stage 2
        }
    }
    // Stage 2: Apply a more expensive pricing heuristic (Stage::Two)
    else if (s2) {
        s2    = true;
        stage = 2;
        paths = bi_labeling_algorithm<Stage::Two>(); // Solve the problem with
                                                     // Stage 2 heuristic
        inner_obj = paths[0]->cost;

        // Transition from Stage 2 to Stage 3 if the objective improves or after
        // 800 iterations
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
    // Cache bucket and auxiliary structure references to avoid repeated
    // lookups.
    auto       &buckets           = assign_buckets<D>(fw_buckets, bw_buckets);
    const auto &topological_order = assign_buckets<D>(fw_topological_order, bw_topological_order);
    const auto &sccs              = assign_buckets<D>(fw_sccs, bw_sccs);
    const auto &Phi               = assign_buckets<D>(Phi_fw, Phi_bw);
    auto       &c_bar             = assign_buckets<D>(fw_c_bar, bw_c_bar);
    auto       &n_labels          = assign_buckets<D>(n_fw_labels, n_bw_labels);
    const auto &sorted_sccs       = assign_buckets<D>(fw_sccs_sorted, bw_sccs_sorted);
    const auto &n_buckets         = assign_buckets<D>(fw_buckets_size, bw_buckets_size);
    auto       &stat_n_labels     = assign_buckets<D>(stat_n_labels_fw, stat_n_labels_bw);
    auto       &stat_n_dom        = assign_buckets<D>(stat_n_dom_fw, stat_n_dom_bw);
    auto       &arc_scores        = assign_buckets<D>(fw_arc_scores, bw_arc_scores);
    auto       &best_label        = assign_buckets<D>(fw_best_label, bw_best_label);

    const bool profile_labeling = options.profile_labeling;

    // Reset the total label count.
    n_labels = 0;

    // Pre-calculate bitmap segments for dominance-checking.
    const size_t          n_segments = (n_buckets + 63) >> 6;
    std::vector<uint64_t> Bvisited(n_segments, 0);
    std::vector<uint32_t> touched_segments;
    touched_segments.reserve(std::min<size_t>(n_segments, 64));
    std::vector<int> bucket_pos_in_scc(n_buckets, -1);

    // Process only newly queued labels while preserving the bucket order
    // inside each SCC.
    std::vector<std::vector<Label *>> pending_labels_by_bucket;
    std::vector<size_t>               pending_label_cursor;
    std::vector<uint8_t>              bucket_is_active;
    std::vector<int>                  active_bucket_heap;

    // Precompute main resource index and q_star value for partial solutions.
    constexpr bool is_partial     = (F == Full::Partial);
    int            main_res_index = 0;
    double         q_star_value   = 0.0;
    if constexpr (is_partial) {
        main_res_index = options.main_resources[0];
        q_star_value   = std::floor(q_star[main_res_index]);
    }

    // Reuse a thread-local vector for collecting labels to dominate.
    static thread_local std::vector<Label *> tl_labels_to_dominate;
    tl_labels_to_dominate.clear();
    tl_labels_to_dominate.reserve(64); // Pre-reserve space to avoid reallocations

    // Process each strongly connected component (SCC) in topological order.
    for (const auto &scc_index : topological_order) {
        const auto &scc_buckets = sorted_sccs[scc_index];
        if (scc_buckets.empty()) continue;

        pending_labels_by_bucket.clear();
        pending_labels_by_bucket.resize(scc_buckets.size());
        pending_label_cursor.assign(scc_buckets.size(), 0);
        bucket_is_active.assign(scc_buckets.size(), 0);
        active_bucket_heap.clear();

        auto activate_bucket = [&](int bucket_pos) {
            if (bucket_pos < 0 || bucket_is_active[bucket_pos]) return;
            bucket_is_active[bucket_pos] = 1;
            active_bucket_heap.push_back(bucket_pos);
            std::push_heap(active_bucket_heap.begin(), active_bucket_heap.end(), std::greater<int>());
        };

        for (size_t bucket_idx = 0; bucket_idx < scc_buckets.size(); ++bucket_idx) {
            const int bucket          = scc_buckets[bucket_idx];
            bucket_pos_in_scc[bucket] = static_cast<int>(bucket_idx);

            const auto &bucket_labels  = buckets[bucket].get_sorted_labels();
            const auto &extra_labels   = buckets[bucket].get_extra_labels();
            auto       &pending_labels = pending_labels_by_bucket[bucket_idx];
            pending_labels.reserve(bucket_labels.size() + extra_labels.size());

            for (Label *label : bucket_labels) {
                if (!label->is_extended && !label->is_dominated) pending_labels.push_back(label);
            }
            for (Label *label : extra_labels) {
                if (!label->is_extended && !label->is_dominated) pending_labels.push_back(label);
            }

            if (!pending_labels.empty()) activate_bucket(static_cast<int>(bucket_idx));
        }

        while (!active_bucket_heap.empty()) {
            std::pop_heap(active_bucket_heap.begin(), active_bucket_heap.end(), std::greater<int>());
            const int bucket_idx = active_bucket_heap.back();
            active_bucket_heap.pop_back();

            const int bucket         = scc_buckets[bucket_idx];
            auto     &pending_labels = pending_labels_by_bucket[bucket_idx];
            size_t   &pending_cursor = pending_label_cursor[bucket_idx];

            while (pending_cursor < pending_labels.size()) {
                Label *label = pending_labels[pending_cursor++];

                // Skip labels already processed.
                if (__builtin_expect(label->is_extended || label->is_dominated, 0)) continue;

                // For partial solutions, check resource feasibility.
                if constexpr (F != Full::PSTEP && F != Full::TSP) {
                    if constexpr (is_partial) {
                        const auto main_resource = label->resources[main_res_index];
                        if ((D == Direction::Forward && main_resource > q_star_value) ||
                            (D == Direction::Backward && main_resource <= q_star_value)) {
                            label->set_extended(true);
                            continue;
                        }
                    }
                }

                for (uint32_t segment_idx : touched_segments) { Bvisited[segment_idx] = 0; }
                touched_segments.clear();

                // Check if the label is dominated by labels in smaller
                // buckets.
                if constexpr (F != Full::TSP) {
                    if (profile_labeling) profile_record_dominance_check(D, S);
                    if (DominatedInCompWiseSmallerBuckets<D, S>(label, bucket, c_bar, Bvisited, touched_segments,
                                                                stat_n_dom))
                        continue;
                }

                // Retrieve outgoing arcs for the current label.
                const auto &node_arcs = buckets[bucket].template get_bucket_arcs<D>();
                if (__builtin_expect(node_arcs.empty(), 0)) {
                    label->set_extended(true);
                    continue;
                }

                // Prefetch first few arcs
                if (!node_arcs.empty()) {
                    for (size_t i = 0; i < std::min(size_t(4), node_arcs.size()); ++i) {
                        __builtin_prefetch(&node_arcs[i], 0, 3);
                    }
                }

                for (size_t arc_idx = 0; arc_idx < node_arcs.size(); ++arc_idx) {
                    const auto &arc = node_arcs[arc_idx];

                    // Prefetch next arc if available
                    if (arc_idx + 4 < node_arcs.size()) { __builtin_prefetch(&node_arcs[arc_idx + 4], 0, 3); }

                    auto new_label = Extend<D, S, ArcType::Bucket, Mutability::Mut, F>(label, arc);

                    // Process each new label produced by the extension.
                    if (new_label != nullptr) {
                        ++stat_n_labels;
                        const int to_bucket         = new_label->vertex;
                        auto     &mother_bucket     = buckets[to_bucket];
                        uint64_t  bucket_scan_count = 0;

                        if (profile_labeling) { profile_record_new_label(D, S); }

                        // Prefetch the mother bucket and its labels
                        __builtin_prefetch(&mother_bucket, 0, 3);

                        const auto &to_bucket_labels = mother_bucket.get_sorted_labels();
                        const auto &to_bucket_extra  = mother_bucket.get_extra_labels();

                        // Prefetch first few destination bucket labels
                        if (!to_bucket_labels.empty()) {
                            for (size_t i = 0; i < std::min(size_t(4), to_bucket_labels.size()); ++i) {
                                __builtin_prefetch(to_bucket_labels[i], 0, 3);
                            }
                        }

                        bool dominated = false;
                        if constexpr (S == Stage::One) {
                            // Stage One: mark higher-cost labels as
                            // dominated.
                            for (auto *existing_label : to_bucket_labels) {
                                ++bucket_scan_count;
                                if (existing_label->is_dominated) continue;
                                if (label->cost < existing_label->cost)
                                    existing_label->set_dominated(true);
                                else {
                                    dominated = true;
                                    break;
                                }
                            }
                            if (!dominated) {
                                for (auto *existing_label : to_bucket_extra) {
                                    ++bucket_scan_count;
                                    if (existing_label->is_dominated) continue;
                                    if (label->cost < existing_label->cost)
                                        existing_label->set_dominated(true);
                                    else {
                                        dominated = true;
                                        break;
                                    }
                                }
                            }
                            if (profile_labeling) {
                                profile_record_dominance_check(D, S);
                                profile_record_inner_bin_scan(D, S, bucket_scan_count);
                                if (D == Direction::Forward) {
                                    dominance_checks_per_bucket[to_bucket] += static_cast<int>(bucket_scan_count);
                                }
                            }
                        } else {
                            uint64_t inner_scan_count = 0;

                            // RC-bracketed single-pass dominance (port from
                            // RouteOpt dominance.hpp doDominance). The sorted
                            // tier is RC-sorted; we walk it once, dispatching
                            // by cost relation to new_label:
                            //   existing.cost < new.cost - eps -> only check
                            //     is_dominated(existing, new) (cheap label
                            //     can dominate expensive one only if
                            //     resources/visited line up). If true,
                            //     new_label is dominated -> stop.
                            //   existing.cost > new.cost + eps -> only check
                            //     is_dominated(new, existing). If true, mark
                            //     existing dominated and continue.
                            //   in bracket -> check both directions.
                            const double cost_lo = new_label->cost - numericutils::eps;
                            const double cost_hi = new_label->cost + numericutils::eps;

                            // Forward path: walk sorted committed labels once.
                            const size_t to_bucket_size = to_bucket_labels.size();
                            size_t       i              = 0;
                            for (; i < to_bucket_size; ++i) {
                                if (i + 8 < to_bucket_size) { __builtin_prefetch(to_bucket_labels[i + 8], 0, 3); }
                                Label *cur = to_bucket_labels[i];
                                ++inner_scan_count;
                                if (__builtin_expect(cur->is_dominated, 0)) continue;
                                const double cur_cost = cur->cost;
                                if (cur_cost < cost_lo) {
                                    // cheaper -> only "cur dominates new" is
                                    // possible.
                                    if (is_dominated<D, S>(new_label, cur)) {
                                        ++stat_n_dom;
                                        dominated = true;
                                        break;
                                    }
                                } else if (cur_cost > cost_hi) {
                                    // pricier -> only "new dominates cur".
                                    if (is_dominated<D, S>(cur, new_label)) { cur->set_dominated(true); }
                                } else {
                                    // bracket -> check both.
                                    if (is_dominated<D, S>(new_label, cur)) {
                                        ++stat_n_dom;
                                        dominated = true;
                                        break;
                                    }
                                    if (is_dominated<D, S>(cur, new_label)) { cur->set_dominated(true); }
                                }
                            }

                            // Extra (unsorted) tier: same trichotomy without
                            // sort assumptions. Only walked if not dominated
                            // by the sorted tier.
                            if (!dominated) {
                                for (Label *cur : to_bucket_extra) {
                                    ++inner_scan_count;
                                    if (__builtin_expect(cur->is_dominated, 0)) continue;
                                    const double cur_cost = cur->cost;
                                    if (cur_cost < cost_lo) {
                                        if (is_dominated<D, S>(new_label, cur)) {
                                            ++stat_n_dom;
                                            dominated = true;
                                            break;
                                        }
                                    } else if (cur_cost > cost_hi) {
                                        if (is_dominated<D, S>(cur, new_label)) { cur->set_dominated(true); }
                                    } else {
                                        if (is_dominated<D, S>(new_label, cur)) {
                                            ++stat_n_dom;
                                            dominated = true;
                                            break;
                                        }
                                        if (is_dominated<D, S>(cur, new_label)) { cur->set_dominated(true); }
                                    }
                                }
                            }
                            if (profile_labeling) {
                                profile_record_dominance_check(D, S);
                                profile_record_inner_bin_scan(D, S, inner_scan_count);
                                if (D == Direction::Forward) {
                                    dominance_checks_per_bucket[to_bucket] += static_cast<int>(inner_scan_count);
                                }
                            }
                        }

                        if (!dominated) {
                            if (profile_labeling) profile_record_non_dominated_label(D, S);
                            if (D == Direction::Forward) { ++non_dominated_labels_per_bucket; }
                            ++n_labels;
#ifdef SORTED_LABELS
                            mother_bucket.add_sorted_label(new_label);
#elif defined(LIMITED_BUCKETS)
                            mother_bucket.sorted_label(new_label, BUCKET_CAPACITY);
#else
                            mother_bucket.add_label(new_label);
#endif
                            const int to_bucket_pos = bucket_pos_in_scc[to_bucket];
                            if (to_bucket_pos != -1) {
                                pending_labels_by_bucket[to_bucket_pos].push_back(new_label);
                                activate_bucket(to_bucket_pos);
                            }
                        }
                    }
                }
                label->set_extended(true);
            }

            bucket_is_active[bucket_idx] = 0;
        }

        for (const int bucket : scc_buckets) { bucket_pos_in_scc[bucket] = -1; }

        // Update c_bar values for all buckets in the current SCC.
        auto sorted_buckets = sorted_sccs[scc_index];
        if constexpr (D == Direction::Forward) {
            pdqsort(sorted_buckets.begin(), sorted_buckets.end(),
                    [&](int a, int b) { return buckets[a].get_lb() < buckets[b].get_lb(); });
        } else {
            pdqsort(sorted_buckets.begin(), sorted_buckets.end(),
                    [&](int a, int b) { return buckets[a].get_ub() > buckets[b].get_ub(); });
        }

        // Prefetch data for c_bar updates
        if (!sorted_buckets.empty()) {
            for (size_t i = 0; i < std::min(size_t(4), sorted_buckets.size()); ++i) {
                __builtin_prefetch(&buckets[sorted_buckets[i]], 0, 3);
                __builtin_prefetch(&Phi[sorted_buckets[i]], 0, 3);
            }
        }

        // Batch processing of c_bar updates for improved cache locality.
        for (size_t i = 0; i < sorted_buckets.size(); ++i) {
            const int bucket = sorted_buckets[i];

            // Prefetch data for next iterations
            if (i + 4 < sorted_buckets.size()) {
                __builtin_prefetch(&buckets[sorted_buckets[i + 4]], 0, 3);
                __builtin_prefetch(&Phi[sorted_buckets[i + 4]], 0, 3);
            }

            double      min_c_bar   = buckets[bucket].get_cb();
            const auto &phi_buckets = Phi[bucket];

            // Prefetch first few phi buckets
            if (!phi_buckets.empty()) {
                for (size_t j = 0; j < std::min(size_t(4), phi_buckets.size()); ++j) {
                    __builtin_prefetch(&c_bar[phi_buckets[j]], 0, 3);
                }
            }

            // Process phi buckets in blocks of 4 for better cache efficiency
            size_t j = 0;
            for (; j + 3 < phi_buckets.size(); j += 4) {
                // Prefetch next block
                if (j + 8 < phi_buckets.size()) {
                    __builtin_prefetch(&c_bar[phi_buckets[j + 4]], 0, 3);
                    __builtin_prefetch(&c_bar[phi_buckets[j + 5]], 0, 3);
                    __builtin_prefetch(&c_bar[phi_buckets[j + 6]], 0, 3);
                    __builtin_prefetch(&c_bar[phi_buckets[j + 7]], 0, 3);
                }

                // Compute min across 4 values at once
                double val0 = c_bar[phi_buckets[j]];
                double val1 = c_bar[phi_buckets[j + 1]];
                double val2 = c_bar[phi_buckets[j + 2]];
                double val3 = c_bar[phi_buckets[j + 3]];

                min_c_bar = std::min(min_c_bar, std::min(std::min(val0, val1), std::min(val2, val3)));
            }

            // Process remaining phi buckets
            for (; j < phi_buckets.size(); ++j) { min_c_bar = std::min(min_c_bar, c_bar[phi_buckets[j]]); }

            c_bar[bucket] = min_c_bar;
        }
    }

    // Store the best label from the labeling process.
    if (Label *best_label_run = get_best_label<D>(topological_order, c_bar, sccs)) best_label = best_label_run;

    return c_bar;
}

/* Performs the bi-labeling algorithm on the BucketGraph.
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
            // heuristic_fixing<S>();

            bucket_fixing<S>();
        }
#endif
        // heuristic_fixing<S>();
    }

    if (options.warm_start && !just_fixed) {
        capture_warm_start_labels<Direction::Forward>();
        capture_warm_start_labels<Direction::Backward>();
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
        forward_cbar = labeling_algorithm<Direction::Forward, S, Full::Partial>();
    }

    // Acquire an initial best label from the forward label pool.
    auto best_label = label_pool_fw->acquire();
    if constexpr (SYM == Symmetry::Asymmetric) {
        // Compute the best label if the forward and backward labels are
        // feasible.
        if (check_feasibility(fw_best_label, bw_best_label)) {
            best_label = compute_label<S>(fw_best_label, bw_best_label);
        } else {
            best_label->cost      = 0.0;
            best_label->real_cost = std::numeric_limits<double>::infinity();
            best_label->clearRoute();
        }
    } else {
        best_label->cost      = best_label->real_cost;
        best_label->real_cost = std::numeric_limits<double>::infinity();
        best_label->clearRoute();
    }
    merged_labels.push_back(best_label);

    // Declare atomics to track the best cost and non-dominated label
    // count.
    std::atomic<double> best_cost{0.0};
    std::atomic<size_t> non_dominated_labels_per_bucket{0};

    // === Parallel Bucket Processing ===
    // Choose a chunk size based on hardware concurrency.
    const size_t chunk_size = fw_buckets_size / MERGE_SCHED_CONCURRENCY; // 1/4 of buckets per core
    const size_t n_chunks   = (fw_buckets_size + chunk_size - 1) / chunk_size;

    auto bulk_sender = stdexec::bulk(
        stdexec::just(), n_chunks,
        [this, chunk_size, &best_cost, &non_dominated_labels_per_bucket](std::size_t chunk_idx) {
            const size_t start_bucket = chunk_idx * chunk_size;
            const size_t end_bucket   = std::min(start_bucket + chunk_size, static_cast<size_t>(fw_buckets_size));

            // Process each bucket within the current chunk.
            for (size_t bucket = start_bucket; bucket < end_bucket; ++bucket) {
                // Process non-dominated labels in the current bucket.
                const auto &bucket_labels = fw_buckets[bucket].get_labels();
                for (const Label *L : bucket_labels) {
                    if (L->is_dominated) continue;

                    non_dominated_labels_per_bucket.fetch_add(1, std::memory_order_relaxed);

                    // Retrieve the outgoing arcs for the current label.
                    // const auto &to_arcs =
                    // nodes[L->node_id].get_arcs<Direction::Forward>();
                    const auto &to_arcs = fw_buckets[bucket].template get_bucket_arcs<Direction::Forward>();
                    for (const auto &arc : to_arcs) {
                        const int to_node = fw_buckets[arc.to_bucket].node_id;
                        if constexpr (S >= Stage::Three || S == Stage::Eliminate) {
                            if (is_arc_fixed(L->node_id, to_node)) continue;
                        }

                        // Extend the label along the current arc.
                        auto L_prime =
                            Extend<Direction::Forward, S, ArcType::Bucket, Mutability::Const, Full::Reverse>(L, arc);

                        // Process each extended label.
                        if (L_prime != -1) {
                            int bucket_to_process = L_prime;

                            // Concatenate the label with bucket
                            // information, updating best_cost.
                            ConcatenateLabel<S, SYM>(L, bucket_to_process, best_cost);
                        }
                    }
                }
            }
        });

    // Submit the bulk work to the merge scheduler and wait for
    // completion.
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
        const int         labels_size = merged_labels.size();
        const int         end_idx     = std::min(N_ADD + N_ADD, labels_size);
        paths.reserve(end_idx - N_ADD);
        for (int i = N_ADD; i < end_idx; ++i) {
            const auto &route = merged_labels[i]->getRoute();
            if (route.size() <= 3) continue;
            paths.emplace_back(route, merged_labels[i]->real_cost);
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
        const int n_candidates = std::min(N_ADD, static_cast<int>(merged_labels.size()));
        for (int i = 0; i < n_candidates; ++i) {
            if (merged_labels[i]->nodes_covered.size() <= 3) continue;
            top_labels.push_back(merged_labels[i]);
        }
        ils->submit_task(top_labels, nodes);
    }
#endif

    // Store the cost of the best (first) merged label as the inner
    // objective.
    inner_obj = merged_labels[0]->cost;
    return merged_labels;
}

/**
 * Extends the label L_prime with the given BucketArc gamma.
 *
 */
template <Direction D, Stage S, ArcType A, Mutability M, Full F>
inline auto BucketGraph::Extend(const std::conditional_t<M == Mutability::Mut, Label *, const Label *>          L_prime,
                                const std::conditional_t<A == ArcType::Bucket, BucketArc,
                                                         std::conditional_t<A == ArcType::Jump, JumpArc, Arc>> &gamma,
                                int depth) noexcept {
    // Pre-allocated result vector to avoid allocations for the common case of
    // 0-1 results
    static thread_local std::vector<Label *> result_vector;
    Label                                   *result = nullptr;

    // Prepare temporary resource storage (resize to current resources size)
    static thread_local std::vector<double> new_resources;
    const size_t                            n_resources = options.resources.size();
    new_resources.resize(n_resources);

    // Cache initial values from L_prime
    const int    initial_node_id   = L_prime->node_id;
    const auto  &initial_resources = L_prime->resources; // Use reference instead of copy
    const double initial_cost      = L_prime->cost;

    // Determine target node ID based on arc type
    int node_id;
    if constexpr (A == ArcType::Bucket) {
        node_id = assign_buckets<D>(fw_buckets, bw_buckets)[gamma.to_bucket].node_id;
    } else if constexpr (A == ArcType::Node) {
        node_id = gamma.to;
    } else { // Jump arc
        auto &buckets      = assign_buckets<D>(fw_buckets, bw_buckets);
        node_id            = buckets[gamma.jump_bucket].node_id;
        const auto &bucket = buckets[gamma.jump_bucket];

        // Fast copy the resources
        std::memcpy(new_resources.data(), initial_resources.data(), n_resources * sizeof(double));

        // Update resource bounds based on jump bucket properties
        for (size_t i = 0; i < n_resources; ++i) {
            if constexpr (D == Direction::Forward) {
                new_resources[i] = std::max(new_resources[i], static_cast<double>(bucket.lb[i]));
            } else {
                new_resources[i] = std::min(new_resources[i], static_cast<double>(bucket.ub[i]));
            }
        }
    }

    // Early rejection: if we loop back or the node was already visited
    if (node_id == initial_node_id || is_node_visited(L_prime->visited_bitmap, node_id)) {
        if constexpr (F == Full::Reverse)
            return -1;
        else
            return result;
    }

    // For stages where fixed arcs matter
    if constexpr (S >= Stage::Three || S == Stage::Eliminate) {
        if ((D == Direction::Forward && is_arc_fixed(initial_node_id, node_id)) ||
            (D == Direction::Backward && is_arc_fixed(node_id, initial_node_id))) {
            if constexpr (F == Full::Reverse)
                return -1;
            else
                return result;
        }
    }

    // Process resources (feasibility check)
    if constexpr (F != Full::TSP) {
        if constexpr (A != ArcType::Jump) {
            // Only copy resources for non-jump arcs (jump arcs already copied)
            std::memcpy(new_resources.data(), initial_resources.data(), n_resources * sizeof(double));

            if (!process_all_resources<D>(new_resources, initial_resources, gamma, nodes[node_id], n_resources)) {
                if constexpr (F == Full::Reverse)
                    return -1;
                else
                    return result;
            }
        } else {
            auto       &buckets = assign_buckets<D>(fw_buckets, bw_buckets);
            const auto &bucket  = buckets[gamma.jump_bucket];
            for (size_t i = 0; i < n_resources; ++i) {
                if constexpr (D == Direction::Forward) {
                    new_resources[i] =
                        std::max(new_resources[i] + gamma.resource_increment[i], static_cast<double>(bucket.lb[i]));
                    if (numericutils::gt(new_resources[i], bucket.ub[i])) {
                        if constexpr (F == Full::Reverse)
                            return -1;
                        else
                            return nullptr;
                    }

                } else {
                    new_resources[i] =
                        std::min(new_resources[i] - gamma.resource_increment[i], static_cast<double>(bucket.ub[i]));
                    if (numericutils::lt(new_resources[i], bucket.lb[i])) {
                        if constexpr (F == Full::Reverse)
                            return -1;
                        else
                            return nullptr;
                    }
                }
            }
        }
    }

    // For PSTEP/TSP: Check path length constraints
    if constexpr (F == Full::PSTEP || F == Full::TSP) {
        const int next_path_len = L_prime->path_len + 1;
        if (next_path_len > options.max_path_size ||
            (next_path_len == options.max_path_size && node_id != options.end_depot)) {
            if constexpr (F == Full::Reverse)
                return -1;
            else
                return result;
        }
    }

    // Get bucket number for the new label
    int to_bucket = -1;
    if constexpr (A == ArcType::Bucket) {
        to_bucket = get_bucket_number<D>(node_id, new_resources);
    } else if constexpr (A == ArcType::Jump) {
        to_bucket = gamma.jump_bucket;
    } else if constexpr (A == ArcType::Node) {
        to_bucket = get_bucket_number<D>(node_id, new_resources);
    }

    // Handle Reverse mode early
    if constexpr (F == Full::Reverse) {
        if (new_resources[options.main_resources[0]] <= std::floor(q_star[options.main_resources[0]])) return -1;

        return to_bucket;
    }

    // Calculate new cost
#ifdef CUSTOM_COST
    const auto distance = getcij(initial_node_id, node_id);
    double     new_cost = cost_calculator.calculate_cost(initial_cost, distance, node_id, new_resources);
#else
    // Cache the distance value to avoid repeated calls
    const auto distance = getcij(initial_node_id, node_id);
    double     new_cost = initial_cost + distance;
#endif

    const auto &VRPNode = nodes[node_id];

    if constexpr (F != Full::PSTEP) {
        new_cost -= VRPNode.cost;
    } else {
        if (L_prime->path_len > 1 && initial_node_id != options.depot) {
            const auto three_two_dual = pstep_duals.getThreeTwoDualValue(initial_node_id);
            new_cost += three_two_dual + pstep_duals.getThreeThreeDualValue(initial_node_id);
        }
        new_cost += -pstep_duals.getThreeTwoDualValue(node_id) + pstep_duals.getArcDualValue(initial_node_id, node_id);
    }

    // Apply branching dual adjustments if present - use local variables to
    // avoid repeated pointer dereferencing
    if (!branching_duals->empty()) {
        if (D == Direction::Forward) {
            new_cost += branching_duals->getDual(initial_node_id, node_id);
        } else {
            new_cost += branching_duals->getDual(node_id, initial_node_id);
        }
        new_cost += branching_duals->getDual(node_id);
    }

    RCC_MODE_BLOCK(if constexpr (S == Stage::Four) {
        if (D == Direction::Forward) {
            new_cost -= arc_duals.getDual(initial_node_id, node_id);
        } else {
            new_cost -= arc_duals.getDual(node_id, initial_node_id);
        }
    })

    // Create and initialize a new label
    auto &label_pool = assign_buckets<D>(label_pool_fw, label_pool_bw);
    auto  new_label  = label_pool->acquire();
    new_label->initialize(to_bucket, new_cost, new_resources, node_id);
    new_label->vertex    = to_bucket;
    new_label->real_cost = L_prime->real_cost + distance; // Use cached distance
    new_label->path_len  = L_prime->path_len + 1;

    // Update visited bitmap: intersect parent's visited bits with neighborhood
    // mask
    if constexpr (F != Full::PSTEP) {
        for (size_t i = 0; i < new_label->visited_bitmap.size(); ++i) {
            const uint64_t current_visited = L_prime->visited_bitmap[i];
            if (current_visited) { new_label->visited_bitmap[i] = current_visited & neighborhoods_bitmap[node_id][i]; }
        }
    } else {
        // Use memcpy for direct bitmap copy (likely faster than assignment for
        // large bitmaps)
        std::memcpy(new_label->visited_bitmap.data(), L_prime->visited_bitmap.data(),
                    new_label->visited_bitmap.size() * sizeof(uint64_t));
    }

    set_node_visited(new_label->visited_bitmap, node_id);

    // Reuse pooled vector capacity when extending the route.
    auto &extended_route = new_label->nodes_covered;
    extended_route.clear();
    extended_route.reserve(L_prime->nodes_covered.size() + 1);
    extended_route.insert(extended_route.end(), L_prime->nodes_covered.begin(), L_prime->nodes_covered.end());
    extended_route.push_back(node_id);

#if defined(SRC)
    if constexpr (S == Stage::Four || S == Stage::Enumerate) {
        new_label->SRCmap      = L_prime->SRCmap;
        auto      &cutter      = cut_storage;
        const auto active_cuts = cutter->getActiveCuts();
        const auto n_cuts      = cutter->activeSize();
        if (__builtin_expect(n_cuts == 0, 0)) { // Branch prediction: unlikely
            if constexpr (F == Full::Reverse) {
                return -1;
            } else {
                return new_label;
            }
        }

        double       total_cost_update = 0.0;
        const size_t segment           = node_id >> 6;
        const size_t bit_position      = node_id & 63;

        // Prefetch critical data structures
        auto &masks = cutter->getSegmentMasks();
        __builtin_prefetch(&masks, 0, 3);
        __builtin_prefetch(active_cuts.data(), 0, 3);
        __builtin_prefetch(new_label->SRCmap.data(), 1, 3); // 1 = read-write

#if !defined(SRC_MEMORY_MODE_ARC)
        for (const auto &update : cutter->getSRCNodeUpdates(node_id)) {
            auto &src_map_value = new_label->SRCmap[update.active_idx];
            src_map_value += update.add;
            const bool overflow = src_map_value >= update.den;
            src_map_value -= overflow ? update.den : 0;
            total_cost_update -= overflow ? update.dual : 0;
        }

        for (const auto active_idx : cutter->getSRCNodeClears(node_id)) { new_label->SRCmap[active_idx] = 0; }
#else
        const auto     cut_limit_mask = masks.get_cut_limit_mask();
        const uint64_t valid_cuts     = masks.get_valid_cut_mask(segment, bit_position) & cut_limit_mask;

        // Skip processing if no valid cuts
        if (__builtin_expect(valid_cuts != 0,
                             1)) { // Branch prediction: likely
            // Process valid cuts (4 at a time if possible)
            uint64_t remaining_cuts = valid_cuts;

            // Process blocks of 4 cuts if available (manual loop unrolling)
            while (__builtin_popcountll(remaining_cuts) >= 4) {
                for (int i = 0; i < 4; i++) {
                    const int   cut_idx    = __builtin_ctzll(remaining_cuts);
                    const auto &active_cut = active_cuts[cut_idx];
                    const auto &cut        = *active_cut.cut_ptr;

                    // Prefetch the next cut data we'll need
                    if (i < 3) {
                        uint64_t  next_bits    = remaining_cuts & (remaining_cuts - 1);
                        const int next_cut_idx = __builtin_ctzll(next_bits);
                        __builtin_prefetch(&active_cuts[next_cut_idx], 0, 3);
                        __builtin_prefetch(active_cuts[next_cut_idx].cut_ptr, 0, 3);
                    }

#if defined(SRC_MEMORY_MODE_ARC)
                    auto &src_map_value = new_label->SRCmap[active_cut.index];
                    if (L_prime->node_id != -1 && cut.isSRCset(L_prime->node_id, node_id)) {
                        const double num_value = cut.p.num[cut.baseSetOrder[node_id]];
                        src_map_value += num_value;
                        const bool overflow = src_map_value >= cut.p.den;
                        src_map_value -= overflow ? cut.p.den : 0;
                        total_cost_update -= overflow ? active_cut.dual_value : 0;
                    }
#else
                    auto        &src_map_value = new_label->SRCmap[active_cut.index];
                    const double num_value     = cut.p.num[cut.baseSetOrder[node_id]];
                    src_map_value += num_value;
                    const bool overflow = src_map_value >= cut.p.den;
                    src_map_value -= overflow ? cut.p.den : 0;
                    total_cost_update -= overflow ? active_cut.dual_value : 0;
#endif

                    remaining_cuts &= (remaining_cuts - 1); // Clear processed bit
                }
            }

            // Process remaining cuts (less than 4)
            while (remaining_cuts) {
                const int   cut_idx    = __builtin_ctzll(remaining_cuts);
                const auto &active_cut = active_cuts[cut_idx];
                const auto &cut        = *active_cut.cut_ptr;

#if defined(SRC_MEMORY_MODE_ARC)
                auto &src_map_value = new_label->SRCmap[active_cut.index];
                if (L_prime->node_id != -1 && cut.isSRCset(L_prime->node_id, node_id)) {
                    const double num_value = cut.p.num[cut.baseSetOrder[node_id]];
                    src_map_value += num_value;
                    if (src_map_value >= cut.p.den) {
                        src_map_value -= cut.p.den;
                        total_cost_update -= active_cut.dual_value;
                    }
                }
#else
                auto &src_map_value = new_label->SRCmap[active_cut.index];
                src_map_value += cut.p.num[cut.baseSetOrder[node_id]];
                if (src_map_value >= cut.p.den) {
                    src_map_value -= cut.p.den;
                    total_cost_update -= active_cut.dual_value;
                }
#endif
                remaining_cuts &= (remaining_cuts - 1);
            }
        }

        // Process cuts to clear
        uint64_t to_clear = (~masks.get_neighbor_mask(segment, bit_position)) & cut_limit_mask;

        if (__builtin_expect(to_clear != 0, 1)) { // Branch prediction: likely
            // Use vectorization for clearing multiple cuts when possible
            constexpr int vector_size = 8; // Process 8 cuts at a time if possible

            while (__builtin_popcountll(to_clear) >= vector_size) {
                for (int i = 0; i < vector_size; i++) {
                    const int cut_idx                             = __builtin_ctzll(to_clear);
                    new_label->SRCmap[active_cuts[cut_idx].index] = 0.0;
                    to_clear &= (to_clear - 1);
                }
            }

            // Process remaining cuts to clear
            while (to_clear) {
                const int cut_idx                             = __builtin_ctzll(to_clear);
                new_label->SRCmap[active_cuts[cut_idx].index] = 0.0;
                to_clear &= (to_clear - 1);
            }
        }
#endif

        new_label->cost += total_cost_update;
    }
#endif
    if constexpr (F != Full::Reverse) {
        return new_label;
    } else {
        return -1;
    }
}
/**
 * @brief Checks if a label is dominated by a new label based on cost
 * and resource conditions.
 *
 */
template <Direction D, Stage S>
inline bool BucketGraph::is_dominated(const Label *__restrict new_label, const Label *__restrict label) noexcept {
    // A label cannot dominate if its cost is higher.
    // const double cost_diff = label->cost - new_label->cost;
    if (numericutils::gt(label->cost, new_label->cost)) { return false; }

    // Compare resource vectors.
    const auto *__restrict new_res = new_label->resources.data();
    const auto *__restrict lbl_res = label->resources.data();

    // Prefetch resource data
    __builtin_prefetch(new_res, 0,
                       3); // 0 = read only, 3 = high temporal locality
    __builtin_prefetch(lbl_res, 0, 3);

    const size_t n_res = options.resources.size();
    if constexpr (D == Direction::Forward) {
        // In the Forward direction, each resource value of 'label' must
        // not exceed that of 'new_label'. Check resource 0 last because bucket
        // traversal already gives it partial ordering.
        for (size_t i = 1; i < n_res; ++i) {
            if (numericutils::gt(lbl_res[i], new_res[i])) { return false; }
        }
        if (n_res > 0 && numericutils::gt(lbl_res[0], new_res[0])) { return false; }
    } else if constexpr (D == Direction::Backward) {
        // In the Backward direction, each resource value of 'label'
        // must not be less than that of 'new_label'. Check resource 0 last.
        for (size_t i = 1; i < n_res; ++i) {
            if (numericutils::lt(lbl_res[i], new_res[i])) { return false; }
        }
        if (n_res > 0 && numericutils::lt(lbl_res[0], new_res[0])) { return false; }
    }

    // visits of label.
    if constexpr (S == Stage::Three || S == Stage::Four || S == Stage::Enumerate) {
        const size_t n_bitmap = label->visited_bitmap.size();

        // Prefetch bitmap data
        __builtin_prefetch(label->visited_bitmap.data(), 0, 3);
        __builtin_prefetch(new_label->visited_bitmap.data(), 0, 3);

        for (size_t i = 0; i < n_bitmap; ++i) {
            // Every visited node in 'label' must also be visited in
            // 'new_label'.
            if ((label->visited_bitmap[i] & new_label->visited_bitmap[i]) != label->visited_bitmap[i]) { return false; }
        }
    }
#ifdef SRC
    // For Stage Four or Enumerate, apply additional SRC-based cost
    // adjustments.
    if constexpr (S == Stage::Four || S == Stage::Enumerate) {
        const double local_cost_diff        = label->cost - new_label->cost;
        const auto *__restrict lbl_srcs     = label->SRCmap.data();
        const auto *__restrict new_lbl_srcs = new_label->SRCmap.data();

        // Prefetch SRCmap data
        __builtin_prefetch(lbl_srcs, 0, 3);
        __builtin_prefetch(new_lbl_srcs, 0, 3);

        const auto  &active_cuts = cut_storage->getActiveCuts();
        double       dual_sum    = 0.0;
        const size_t n           = active_cuts.size();
        for (size_t i = 0; i < n; ++i) {
            const auto &cut = active_cuts[i];
            // Branch prediction hint: likely false.
            if (__builtin_expect(lbl_srcs[cut.index] > new_lbl_srcs[cut.index], 0)) {
                dual_sum += cut.dual_value;
                if (__builtin_expect((numericutils::gt(local_cost_diff, dual_sum)), 0)) { return false; }
            }
        }
    }
#endif
    return true;
}

/**
 * @brief Checks if element 'a' precedes element 'b' in the given
 * strongly connected components (SCCs).
 *
 * This function takes a vector of SCCs and two elements 'a' and 'b' as
 * input. It searches for 'a' and 'b' in the SCCs and determines if 'a'
 * precedes 'b' in the SCC list.
 *
 */

template <typename T>
inline bool precedes(const T a, const T b, const UnionFind &uf,
                     ankerl::unordered_dense::map<std::pair<T, T>, bool> &cache) {
    // Create a cache key
    const std::pair<T, T> key{a, b};
    if (const auto it = cache.find(key); it != cache.end()) { return it->second; }

    // Use the UnionFind's built-in comparison which already has the
    // correct ordering
    bool result = uf.compareSubsets(a, b);
    cache.emplace(key, result);
    return result;
}
/**
 * @brief Determines if a label is dominated in component-wise smaller
 * buckets.
 *
 * This function checks if a given label is dominated by any other label
 * in component-wise smaller buckets. The dominance is determined based
 * on the cost and order of the buckets.
 *
 */
template <Direction D, Stage S>
inline bool BucketGraph::DominatedInCompWiseSmallerBuckets(const Label *__restrict__ L, int bucket,
                                                           const std::vector<double> &__restrict__ c_bar,
                                                           std::vector<uint64_t> &__restrict__ Bvisited,
                                                           std::vector<uint32_t> &touched_segments,
                                                           uint                  &stat_n_dom) noexcept {
    // Cache direction-specific data
    const auto &buckets           = assign_buckets<D>(fw_buckets, bw_buckets);
    const auto &Phi               = assign_buckets<D>(Phi_fw, Phi_bw);
    const auto &bucket_scc_rank   = assign_buckets<D>(fw_bucket_scc_rank, bw_bucket_scc_rank);
    const auto &num_buckets_index = assign_buckets<D>(num_buckets_index_fw, num_buckets_index_bw);
    const auto &num_buckets       = assign_buckets<D>(num_buckets_fw, num_buckets_bw);
    const auto &rc2_bin           = assign_buckets<D>(fw_rc2_bin, bw_rc2_bin);
    const auto &rc2_till_this_bin = assign_buckets<D>(fw_rc2_till_this_bin, bw_rc2_till_this_bin);
    const bool  profile_labeling  = options.profile_labeling;

    // Cache frequently used label properties
    const int    b_L        = L->vertex;
    const double label_cost = L->cost;
    const int    label_rank = bucket_scc_rank[b_L];

    // Use static thread-local stack to avoid repeated allocations
    static thread_local std::vector<int> stack_buffer;
    if (stack_buffer.capacity() < R_SIZE) { stack_buffer.reserve(R_SIZE); }
    stack_buffer.clear();
    stack_buffer.push_back(bucket);

    // Direct pointer to bit mask lookup for fast bit operations
    const uint64_t *const bit_mask_lookup_ptr = bit_mask_lookup.data();

    // Inline dominance check function to avoid lambda overhead
    auto inline_check_dominance = [&](std::span<Label *const> labels, uint &n_dom) -> bool {
        const auto scan_end = std::upper_bound(labels.begin(), labels.end(), L->cost + numericutils::eps,
                                               [](double cost, const Label *label) { return cost < label->cost; });
        labels              = labels.first(static_cast<size_t>(scan_end - labels.begin()));
        if (profile_labeling) profile_record_inner_bin_scan(D, S, labels.size());
#ifdef __AVX2__
        if (labels.size() >= AVX_LIM) return check_dominance_against_vector<D, S>(L, labels, cut_storage, n_dom);
#endif
        const size_t size = labels.size();
        size_t       i    = 0;
        // Loop unrolling: process 4 labels at a time
        for (; i + 3 < size; i += 4) {
            if (is_dominated<D, S>(L, labels[i]) || is_dominated<D, S>(L, labels[i + 1]) ||
                is_dominated<D, S>(L, labels[i + 2]) || is_dominated<D, S>(L, labels[i + 3])) {
                ++n_dom;
                return true;
            }
        }
        // Process remaining labels
        for (; i < size; ++i) {
            if (is_dominated<D, S>(L, labels[i])) {
                ++n_dom;
                return true;
            }
        }
        return false;
    };

    auto inline_check_extra_dominance = [&](std::span<Label *const> labels, uint &n_dom) -> bool {
        if (profile_labeling) profile_record_inner_bin_scan(D, S, labels.size());
        for (Label *label : labels) {
            if (label->is_dominated) continue;
            if (numericutils::gt(label->cost, L->cost + numericutils::eps)) continue;
            if (is_dominated<D, S>(L, label)) {
                ++n_dom;
                return true;
            }
        }
        return false;
    };

    // Main DFS traversal loop
    while (!stack_buffer.empty()) {
        const int current_bucket = stack_buffer.back();
        stack_buffer.pop_back();

        const size_t   segment = static_cast<size_t>(current_bucket) >> 6;
        const uint64_t bit     = bit_mask_lookup_ptr[current_bucket & 63];

        if (unlikely(Bvisited[segment] & bit)) continue;

        // Mark as visited
        if (Bvisited[segment] == 0) { touched_segments.push_back(static_cast<uint32_t>(segment)); }
        Bvisited[segment] |= bit;

        // Fast path: if label cost is lower and bucket precedes L in the SCC
        // order, return early
        if (likely(label_cost < c_bar[current_bucket] && bucket_scc_rank[current_bucket] < label_rank)) return false;

        // Whole-bucket prune: if no label in this bucket or any predecessor bin
        // at this node can dominate L in the second resource, skip it entirely.
        if (options.resources.size() > 1) {
            const auto &bucket       = buckets[current_bucket];
            const int   node_id      = bucket.node_id;
            const int   local_bucket = current_bucket - num_buckets_index[node_id];
            if (local_bucket >= 0 && local_bucket < num_buckets[node_id]) {
                const double label_rc2 = L->resources[1];
                if constexpr (D == Direction::Forward) {
                    if (numericutils::gt(rc2_till_this_bin[node_id][local_bucket], label_rc2)) { continue; }
                } else {
                    if (numericutils::lt(rc2_till_this_bin[node_id][local_bucket], label_rc2)) { continue; }
                }
            }
        }

        // Skip dominance check for L's own bucket
        if (b_L != current_bucket) {
            const auto &mother_bucket = buckets[current_bucket];
            if (mother_bucket.check_dominance(L, inline_check_dominance, inline_check_extra_dominance, stat_n_dom))
                return true;
        }

        // Process neighbor buckets using pointer arithmetic and prefetching
        const auto  &neighbors    = Phi[current_bucket];
        const size_t n_neighbors  = neighbors.size();
        const int   *neighbor_ptr = neighbors.data();
        const int   *neighbor_end = neighbor_ptr + n_neighbors;
        for (; neighbor_ptr != neighbor_end; ++neighbor_ptr) {
            const int   b_prime = *neighbor_ptr;
            const auto &bucket  = buckets[b_prime];
            const int   node_id = bucket.node_id;
            const int   local_b = b_prime - num_buckets_index[node_id];
            if (options.resources.size() > 1 && local_b >= 0 && local_b < num_buckets[node_id]) {
                const double label_rc2 = L->resources[1];
                if constexpr (D == Direction::Forward) {
                    if (numericutils::gt(rc2_till_this_bin[node_id][local_b], label_rc2)) { continue; }
                } else {
                    if (numericutils::lt(rc2_till_this_bin[node_id][local_b], label_rc2)) { continue; }
                }
            }

            const size_t   word_idx = static_cast<size_t>(b_prime) >> 6;
            const uint64_t mask     = bit_mask_lookup_ptr[b_prime & 63];
            __builtin_prefetch(&Bvisited[word_idx], 0,
                               1); // Prefetch Bvisited for this segment
            if (!(Bvisited[word_idx] & mask)) stack_buffer.push_back(b_prime);
        }
    }

    return false;
}

/**
 * @brief Runs forward and backward labeling algorithms in parallel and
 * synchronizes the results.
 *
 * T#his function creates tasks for forward and backward labeling
 * algorithms using the provided scheduling mechanism. The tasks are
 * executed in parallel, and the results are synchronized and stored in
 * the provided vectors.
 *
 */
template <Stage state, Full fullness>
void BucketGraph::run_labeling_algorithms(std::vector<double> &forward_cbar, std::vector<double> &backward_cbar) {
    // Schedule the forward labeling algorithm task on the scheduler.
    auto forward_task = stdexec::schedule(bi_sched) |
                        stdexec::then([&]() { return labeling_algorithm<Direction::Forward, state, fullness>(); });

    // Schedule the backward labeling algorithm task on the scheduler.
    auto backward_task = stdexec::schedule(bi_sched) |
                         stdexec::then([&]() { return labeling_algorithm<Direction::Backward, state, fullness>(); });

    // Combine the forward and backward tasks.
    auto combined_work = stdexec::when_all(std::move(forward_task), std::move(backward_task)) |
                         stdexec::then([&](auto forward_result, auto backward_result) {
                             // Store the results into the provided vectors.
                             forward_cbar  = std::move(forward_result);
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
Label *BucketGraph::compute_label(const Label *L, const Label *L_prime, double red_cost) {
    // Compute cost values
    double      cij_cost       = getcij(L->node_id, L_prime->node_id);
    double      real_cost      = L->real_cost + L_prime->real_cost + cij_cost;
    const auto &forward_route  = L->nodes_covered;
    const auto &backward_route = L_prime->nodes_covered;

    // Acquire a new label from the pool and initialize its cost fields.
    auto new_label       = label_pool_fw->acquire();
    new_label->cost      = red_cost;
    new_label->real_cost = real_cost;
    new_label->parent    = nullptr;
    new_label->path_len  = L->path_len + L_prime->path_len;

    // Reuse pooled vector capacity and avoid zero-filling before overwrite.
    auto &merged_route = new_label->nodes_covered;
    merged_route.clear();
    merged_route.reserve(forward_route.size() + backward_route.size());
    merged_route.insert(merged_route.end(), forward_route.begin(), forward_route.end());
    merged_route.insert(merged_route.end(), backward_route.rbegin(), backward_route.rend());

    return new_label;
}
