/**
 * @file BucketSolve.h
 * @brief Defines the BucketGraph solving methods for bucket-based optimization.
 *
 */

#pragma once

#include <cstring>

#include "pricing/BucketJump.h"
#include "pricing/BucketRes.h"
#include "core/Definitions.h"
#include "cuts/SRC.h"

#if !defined(BALDES_DISABLE_SIMD) && __has_include(<experimental/simd>)
#include <experimental/simd>
#define BALDES_HAS_SIMD 1
#endif

#ifdef __AVX2__
#include "pricing/BucketAVX.h"
#endif

#include <execution>

#include "../../third_party/small_vector.hpp"
#include "pricing/BucketConcat.h"
#include "pricing/BucketPricing.h"
#include "pricing/BucketPricingPass.h"
#include "pricing/BucketTopology.h"
#include "pricing/BucketUtils.h"

inline std::vector<Label *> BucketGraph::solveHeuristic() {
    // Initialize the status as not optimal at the start
    status = Status::NotOptimal;

    // updateSplit(); // Update the split values for the bucket graph

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

        if (inner_obj >= -5) {
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

        if (inner_obj >= -100) { status = Status::Optimal; }
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

                // Extend first, then process labels grouped by destination
                // bucket. This keeps the destination bucket and its SoA cache
                // hot across the whole group and amortizes staged-tier merges.
                static thread_local std::vector<Label *> destination_batch;
                destination_batch.clear();
                destination_batch.reserve(node_arcs.size());
                for (size_t arc_idx = 0; arc_idx < node_arcs.size(); ++arc_idx) {
                    const auto &arc = node_arcs[arc_idx];

                    // Prefetch next arc if available
                    if (arc_idx + 4 < node_arcs.size()) { __builtin_prefetch(&node_arcs[arc_idx + 4], 0, 3); }

                    auto new_label = Extend<D, S, ArcType::Bucket, Mutability::Mut, F>(label, arc);
                    if (new_label != nullptr) destination_batch.push_back(new_label);
                }

                std::stable_sort(destination_batch.begin(), destination_batch.end(), [](const Label *a, const Label *b) {
                    return a->vertex < b->vertex;
                });

                for (Label *new_label : destination_batch) {
                    // Process each new label produced by the extension.
                    if (new_label != nullptr) {
                        if constexpr (S == Stage::Enumerate) {
                            if (n_labels >= enumeration_policy.max_labels_per_direction) {
                                enumeration_failed.store(true, std::memory_order_relaxed);
                                label->set_extended(true);
                                continue;
                            }
                        }
                        ++stat_n_labels;
                        const int to_bucket         = new_label->vertex;
                        auto     &mother_bucket     = buckets[to_bucket];
                        uint64_t  bucket_scan_count = 0;

                        if (profile_labeling) { profile_record_new_label(D, S); }

                        // Prefetch the mother bucket and its labels
                        __builtin_prefetch(&mother_bucket, 0, 3);

                        mother_bucket.flush_extra_labels_if_large();
                        const auto &to_bucket_labels = mother_bucket.get_sorted_labels();
                        const auto &to_bucket_extra  = mother_bucket.get_extra_labels();
                        constexpr bool uses_visited_dominance =
                            S == Stage::Three || S == Stage::Four || S == Stage::Enumerate;
                        const uint64_t new_visited_signature = new_label->visited_signature();
#if defined(BALDES_HAS_SIMD)
                        if constexpr (uses_visited_dominance) {
                            if (!to_bucket_labels.empty()) mother_bucket.ensure_label_cache();
                        }
#endif

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
                                if (new_label->cost < existing_label->cost)
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
                                    if (new_label->cost < existing_label->cost)
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

#if defined(SRC)
                            constexpr bool src_adjusted_dominance = (S == Stage::Four || S == Stage::Enumerate);
#else
                            constexpr bool src_adjusted_dominance = false;
#endif
                            // RC-bracketed single-pass dominance (RouteOpt
                            // dominance.hpp doDominance style). The committed
                            // tier is RC-sorted; for non-SRC stages, raw cost
                            // decides which dominance direction is possible:
                            //   cur.cost < new.cost - eps: only cur can
                            //     dominate new, so test resource/path dominance
                            //     of new by cur; stop if true.
                            //   cur.cost > new.cost + eps: only new can
                            //     dominate cur, so test resource/path dominance
                            //     of cur by new; mark cur dominated if true.
                            //   near-equal costs: test both full predicates.
                            // SRC stages skip this bracketing because SRC dual
                            // compensation can overturn raw reduced-cost order.
                            const double cost_lo = new_label->cost - numericutils::eps;
                            const double cost_hi = new_label->cost + numericutils::eps;

                            auto label_cost_less = [](const Label *label, double value) { return label->cost < value; };

                            auto scan_new_dominated_by = [&](size_t begin, size_t end) {
#if defined(BALDES_HAS_SIMD)
                                if constexpr (!src_adjusted_dominance) {
                                    namespace stdx = std::experimental;
                                    using simd_d   = stdx::native_simd<double>;
                                    using mask_d   = typename simd_d::mask_type;

                                    constexpr size_t simd_width = simd_d::size();
                                    const simd_d     eps(numericutils::eps);
                                    size_t           i = begin;
                                    for (; i + simd_width <= end; i += simd_width) {
                                        mask_d candidate(true);
                                        if constexpr (D == Direction::Forward) {
                                            for (size_t r = 1; r < options.resources.size(); ++r) {
                                                const simd_d res(mother_bucket.soa_resources[r].data() + i,
                                                                 stdx::element_aligned);
                                                candidate &= res <= (simd_d(new_label->resources[r]) + eps);
                                            }
                                            if (!options.resources.empty()) {
                                                const simd_d res0(mother_bucket.soa_resources[0].data() + i,
                                                                  stdx::element_aligned);
                                                candidate &= res0 <= (simd_d(new_label->resources[0]) + eps);
                                            }
                                        } else {
                                            for (size_t r = 1; r < options.resources.size(); ++r) {
                                                const simd_d res(mother_bucket.soa_resources[r].data() + i,
                                                                 stdx::element_aligned);
                                                candidate &= res >= (simd_d(new_label->resources[r]) - eps);
                                            }
                                            if (!options.resources.empty()) {
                                                const simd_d res0(mother_bucket.soa_resources[0].data() + i,
                                                                  stdx::element_aligned);
                                                candidate &= res0 >= (simd_d(new_label->resources[0]) - eps);
                                            }
                                        }
                                        if (!stdx::any_of(candidate)) {
                                            inner_scan_count += simd_width;
                                            continue;
                                        }
                                        for (size_t lane = 0; lane < simd_width; ++lane) {
                                            ++inner_scan_count;
                                            if (!candidate[lane]) continue;
                                            if constexpr (uses_visited_dominance) {
                                                if ((mother_bucket.soa_visited_signatures[i + lane] &
                                                     ~new_visited_signature) != 0) {
                                                    if (profile_labeling) profile_record_signature_rejection(D, S);
                                                    continue;
                                                }
                                            }
                                            Label *cur = to_bucket_labels[i + lane];
                                            if (__builtin_expect(cur->is_dominated, 0)) continue;
                                            if (dominates_resource_path<D, S>(new_label, cur)) {
                                                ++stat_n_dom;
                                                dominated = true;
                                                return;
                                            }
                                        }
                                    }
                                    begin = i;
                                }
#endif
                                for (size_t i = begin; i < end; ++i) {
                                    if (i + 8 < end) { __builtin_prefetch(to_bucket_labels[i + 8], 0, 3); }
                                    Label *cur = to_bucket_labels[i];
                                    ++inner_scan_count;
                                    if (__builtin_expect(cur->is_dominated, 0)) continue;
                                    if (dominates_resource_path<D, S>(new_label, cur)) {
                                        ++stat_n_dom;
                                        dominated = true;
                                        break;
                                    }
                                }
                            };

                            auto scan_existing_dominated_by_new = [&](size_t begin, size_t end) {
#if defined(BALDES_HAS_SIMD)
                                if constexpr (!src_adjusted_dominance) {
                                    namespace stdx = std::experimental;
                                    using simd_d   = stdx::native_simd<double>;
                                    using mask_d   = typename simd_d::mask_type;

                                    constexpr size_t simd_width = simd_d::size();
                                    const simd_d     eps(numericutils::eps);
                                    size_t           i = begin;
                                    for (; i + simd_width <= end; i += simd_width) {
                                        mask_d candidate(true);
                                        if constexpr (D == Direction::Forward) {
                                            for (size_t r = 1; r < options.resources.size(); ++r) {
                                                const simd_d res(mother_bucket.soa_resources[r].data() + i,
                                                                 stdx::element_aligned);
                                                candidate &= res >= (simd_d(new_label->resources[r]) - eps);
                                            }
                                            if (!options.resources.empty()) {
                                                const simd_d res0(mother_bucket.soa_resources[0].data() + i,
                                                                  stdx::element_aligned);
                                                candidate &= res0 >= (simd_d(new_label->resources[0]) - eps);
                                            }
                                        } else {
                                            for (size_t r = 1; r < options.resources.size(); ++r) {
                                                const simd_d res(mother_bucket.soa_resources[r].data() + i,
                                                                 stdx::element_aligned);
                                                candidate &= res <= (simd_d(new_label->resources[r]) + eps);
                                            }
                                            if (!options.resources.empty()) {
                                                const simd_d res0(mother_bucket.soa_resources[0].data() + i,
                                                                  stdx::element_aligned);
                                                candidate &= res0 <= (simd_d(new_label->resources[0]) + eps);
                                            }
                                        }
                                        if (!stdx::any_of(candidate)) {
                                            inner_scan_count += simd_width;
                                            continue;
                                        }
                                        for (size_t lane = 0; lane < simd_width; ++lane) {
                                            ++inner_scan_count;
                                            if (!candidate[lane]) continue;
                                            if constexpr (uses_visited_dominance) {
                                                if ((new_visited_signature &
                                                     ~mother_bucket.soa_visited_signatures[i + lane]) != 0) {
                                                    if (profile_labeling) profile_record_signature_rejection(D, S);
                                                    continue;
                                                }
                                            }
                                            Label *cur = to_bucket_labels[i + lane];
                                            if (__builtin_expect(cur->is_dominated, 0)) continue;
                                            if (dominates_resource_path<D, S>(cur, new_label)) {
                                                cur->set_dominated(true);
                                            }
                                        }
                                    }
                                    begin = i;
                                }
#endif
                                for (size_t i = begin; i < end; ++i) {
                                    if (i + 8 < end) { __builtin_prefetch(to_bucket_labels[i + 8], 0, 3); }
                                    Label *cur = to_bucket_labels[i];
                                    ++inner_scan_count;
                                    if (__builtin_expect(cur->is_dominated, 0)) continue;
                                    if (dominates_resource_path<D, S>(cur, new_label)) { cur->set_dominated(true); }
                                }
                            };

                            auto scan_bracket = [&](size_t begin, size_t end) {
#if defined(BALDES_HAS_SIMD)
                                if constexpr (!src_adjusted_dominance) {
                                    namespace stdx = std::experimental;
                                    using simd_d   = stdx::native_simd<double>;
                                    using mask_d   = typename simd_d::mask_type;

                                    constexpr size_t simd_width = simd_d::size();
                                    const simd_d     eps(numericutils::eps);
                                    size_t           i = begin;
                                    for (; i + simd_width <= end; i += simd_width) {
                                        mask_d cur_dominates_new(true);
                                        mask_d new_dominates_cur(true);
                                        if constexpr (D == Direction::Forward) {
                                            for (size_t r = 1; r < options.resources.size(); ++r) {
                                                const simd_d res(mother_bucket.soa_resources[r].data() + i,
                                                                 stdx::element_aligned);
                                                cur_dominates_new &= res <= (simd_d(new_label->resources[r]) + eps);
                                                new_dominates_cur &= res >= (simd_d(new_label->resources[r]) - eps);
                                            }
                                            if (!options.resources.empty()) {
                                                const simd_d res0(mother_bucket.soa_resources[0].data() + i,
                                                                  stdx::element_aligned);
                                                cur_dominates_new &= res0 <= (simd_d(new_label->resources[0]) + eps);
                                                new_dominates_cur &= res0 >= (simd_d(new_label->resources[0]) - eps);
                                            }
                                        } else {
                                            for (size_t r = 1; r < options.resources.size(); ++r) {
                                                const simd_d res(mother_bucket.soa_resources[r].data() + i,
                                                                 stdx::element_aligned);
                                                cur_dominates_new &= res >= (simd_d(new_label->resources[r]) - eps);
                                                new_dominates_cur &= res <= (simd_d(new_label->resources[r]) + eps);
                                            }
                                            if (!options.resources.empty()) {
                                                const simd_d res0(mother_bucket.soa_resources[0].data() + i,
                                                                  stdx::element_aligned);
                                                cur_dominates_new &= res0 >= (simd_d(new_label->resources[0]) - eps);
                                                new_dominates_cur &= res0 <= (simd_d(new_label->resources[0]) + eps);
                                            }
                                        }
                                        const mask_d candidate = cur_dominates_new | new_dominates_cur;
                                        if (!stdx::any_of(candidate)) {
                                            inner_scan_count += simd_width;
                                            continue;
                                        }
                                        for (size_t lane = 0; lane < simd_width; ++lane) {
                                            ++inner_scan_count;
                                            if (!candidate[lane]) continue;
                                            if constexpr (uses_visited_dominance) {
                                                if (cur_dominates_new[lane] &&
                                                    (mother_bucket.soa_visited_signatures[i + lane] &
                                                     ~new_visited_signature) != 0) {
                                                    cur_dominates_new[lane] = false;
                                                    if (profile_labeling) profile_record_signature_rejection(D, S);
                                                }
                                                if (new_dominates_cur[lane] &&
                                                    (new_visited_signature &
                                                     ~mother_bucket.soa_visited_signatures[i + lane]) != 0) {
                                                    new_dominates_cur[lane] = false;
                                                    if (profile_labeling) profile_record_signature_rejection(D, S);
                                                }
                                                if (!cur_dominates_new[lane] && !new_dominates_cur[lane]) continue;
                                            }
                                            Label *cur = to_bucket_labels[i + lane];
                                            if (__builtin_expect(cur->is_dominated, 0)) continue;
                                            if (cur_dominates_new[lane] && is_dominated<D, S>(new_label, cur)) {
                                                ++stat_n_dom;
                                                dominated = true;
                                                return;
                                            }
                                            if (new_dominates_cur[lane] && is_dominated<D, S>(cur, new_label)) {
                                                cur->set_dominated(true);
                                            }
                                        }
                                    }
                                    begin = i;
                                }
#endif
                                for (size_t i = begin; i < end; ++i) {
                                    if (i + 8 < end) { __builtin_prefetch(to_bucket_labels[i + 8], 0, 3); }
                                    Label *cur = to_bucket_labels[i];
                                    ++inner_scan_count;
                                    if (__builtin_expect(cur->is_dominated, 0)) continue;
                                    if (is_dominated<D, S>(new_label, cur)) {
                                        ++stat_n_dom;
                                        dominated = true;
                                        break;
                                    }
                                    if (is_dominated<D, S>(cur, new_label)) { cur->set_dominated(true); }
                                }
                            };

                            // Forward path: walk sorted committed labels once.
                            const size_t to_bucket_size = to_bucket_labels.size();
                            if constexpr (src_adjusted_dominance) {
                                for (size_t i = 0; i < to_bucket_size; ++i) {
                                    if (i + 8 < to_bucket_size) { __builtin_prefetch(to_bucket_labels[i + 8], 0, 3); }
                                    Label *cur = to_bucket_labels[i];
                                    ++inner_scan_count;
                                    if (__builtin_expect(cur->is_dominated, 0)) continue;
                                    if (is_dominated<D, S>(new_label, cur)) {
                                        ++stat_n_dom;
                                        dominated = true;
                                        break;
                                    }
                                    if (is_dominated<D, S>(cur, new_label)) { cur->set_dominated(true); }
                                }
                            } else {
                                static constexpr size_t kLinearCostBracketScanLimit = 32;
                                if (to_bucket_size <= kLinearCostBracketScanLimit) {
                                    for (size_t i = 0; i < to_bucket_size; ++i) {
                                        if (i + 8 < to_bucket_size) {
                                            __builtin_prefetch(to_bucket_labels[i + 8], 0, 3);
                                        }
                                        Label *cur = to_bucket_labels[i];
                                        ++inner_scan_count;
                                        if (__builtin_expect(cur->is_dominated, 0)) continue;
                                        const double cur_cost = cur->cost;
                                        if (cur_cost < cost_lo) {
                                            if (dominates_resource_path<D, S>(new_label, cur)) {
                                                ++stat_n_dom;
                                                dominated = true;
                                                break;
                                            }
                                        } else if (cur_cost > cost_hi) {
                                            if (dominates_resource_path<D, S>(cur, new_label)) {
                                                cur->set_dominated(true);
                                            }
                                        } else {
                                            if (is_dominated<D, S>(new_label, cur)) {
                                                ++stat_n_dom;
                                                dominated = true;
                                                break;
                                            }
                                            if (is_dominated<D, S>(cur, new_label)) { cur->set_dominated(true); }
                                        }
                                    }
                                } else {
#if defined(BALDES_HAS_SIMD)
                                    mother_bucket.ensure_label_cache();
#endif
                                    const auto bracket_begin_it = std::lower_bound(
                                        to_bucket_labels.begin(), to_bucket_labels.end(), cost_lo, label_cost_less);
                                    const auto expensive_begin_it = std::upper_bound(
                                        to_bucket_labels.begin(), to_bucket_labels.end(), cost_hi,
                                        [](double value, const Label *label) { return value < label->cost; });
                                    const size_t bracket_begin =
                                        static_cast<size_t>(std::distance(to_bucket_labels.begin(), bracket_begin_it));
                                    const size_t expensive_begin = static_cast<size_t>(
                                        std::distance(to_bucket_labels.begin(), expensive_begin_it));

                                    scan_new_dominated_by(0, bracket_begin);
                                    if (!dominated) { scan_bracket(bracket_begin, expensive_begin); }
                                    if (!dominated) { scan_existing_dominated_by_new(expensive_begin, to_bucket_size); }
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
                                    if constexpr (src_adjusted_dominance) {
                                        if (is_dominated<D, S>(new_label, cur)) {
                                            ++stat_n_dom;
                                            dominated = true;
                                            break;
                                        }
                                        if (is_dominated<D, S>(cur, new_label)) { cur->set_dominated(true); }
                                    } else if (cur_cost < cost_lo) {
                                        if (dominates_resource_path<D, S>(new_label, cur)) {
                                            ++stat_n_dom;
                                            dominated = true;
                                            break;
                                        }
                                    } else if (cur_cost > cost_hi) {
                                        if (dominates_resource_path<D, S>(cur, new_label)) { cur->set_dominated(true); }
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
                            if constexpr (D == Direction::Forward) {
                                ++non_dominated_labels_per_bucket;
                            } else {
                                ++non_dominated_labels_per_bucket_bw;
                            }
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

        for (const int bucket : scc_buckets) {
            bucket_pos_in_scc[bucket] = -1;
            buckets[bucket].compact_dominated_labels();
        }

        // Update c_bar values for all buckets in the current SCC. The topology
        // builder already stores these buckets in direction-specific resource
        // order, so avoid copying and sorting them on every labeling pass.
        const auto &sorted_buckets = sorted_sccs[scc_index];

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

    // Determine target node ID based on arc type. A bucket jump arc is stored
    // as a marked BucketArc whose to_bucket is the jump bucket and whose
    // jump_to_node is the head of the original arc.
    int  node_id;
    int  jump_bucket_id = -1;
    bool is_jump_arc    = false;
    if constexpr (A == ArcType::Bucket) {
        if (gamma.jump) {
            if (gamma.jump_to_node < 0) {
                if constexpr (F == Full::Reverse)
                    return -1;
                else
                    return result;
            }
            node_id        = gamma.jump_to_node;
            jump_bucket_id = gamma.to_bucket;
            is_jump_arc    = true;
        } else {
            node_id = assign_buckets<D>(fw_buckets, bw_buckets)[gamma.to_bucket].node_id;
        }
    } else if constexpr (A == ArcType::Node) {
        node_id = gamma.to;
    } else { // Jump arc
        auto &buckets  = assign_buckets<D>(fw_buckets, bw_buckets);
        node_id        = gamma.to_job;
        jump_bucket_id = gamma.jump_bucket;
        is_jump_arc    = true;
        if (node_id < 0 || jump_bucket_id < 0 || jump_bucket_id >= static_cast<int>(buckets.size())) {
            if constexpr (F == Full::Reverse)
                return -1;
            else
                return result;
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
        auto process_jump_resources = [&]() -> bool {
            auto &buckets = assign_buckets<D>(fw_buckets, bw_buckets);
            if (jump_bucket_id < 0 || jump_bucket_id >= static_cast<int>(buckets.size())) return false;
            const auto &jump_bucket = buckets[jump_bucket_id];

            std::array<double, R_SIZE> jumped_resources = initial_resources;
            for (size_t i = 0; i < n_resources; ++i) {
                if constexpr (D == Direction::Forward) {
                    jumped_resources[i] = std::max(jumped_resources[i], static_cast<double>(jump_bucket.lb[i]));
                } else {
                    jumped_resources[i] = std::min(jumped_resources[i], static_cast<double>(jump_bucket.ub[i]));
                }
            }
            return process_all_resources<D>(new_resources, jumped_resources, gamma, nodes[node_id], n_resources);
        };

        if constexpr (A != ArcType::Jump) {
            // Only copy resources for non-jump arcs (jump arcs already copied)
            std::memcpy(new_resources.data(), initial_resources.data(), n_resources * sizeof(double));

            const bool feasible = is_jump_arc ? process_jump_resources()
                                              : process_all_resources<D>(new_resources, initial_resources, gamma,
                                                                         nodes[node_id], n_resources);
            if (!feasible) {
                if constexpr (F == Full::Reverse)
                    return -1;
                else
                    return result;
            }
        } else {
            if (!process_jump_resources()) {
                if constexpr (F == Full::Reverse)
                    return -1;
                else
                    return result;
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
        to_bucket = get_bucket_number<D>(node_id, new_resources);
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
    auto  new_label  = label_pool->acquire(to_bucket);
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
        auto      &cutter = cut_storage;
        const auto n_cuts = cutter->activeSize();
        if (__builtin_expect(n_cuts == 0, 0)) { // Branch prediction: unlikely
            if constexpr (F == Full::Reverse) {
                return -1;
            } else {
                return new_label;
            }
        }

        new_label->SRCmap            = L_prime->SRCmap;
        const auto active_cuts       = cutter->getActiveCuts();
        double     total_cost_update = 0.0;

        // Prefetch critical data structures
        __builtin_prefetch(active_cuts.data(), 0, 3);
        __builtin_prefetch(new_label->SRCmap.data(), 1, 3); // 1 = read-write

#if !defined(SRC_MEMORY_MODE_ARC)
        auto &masks = cutter->getSegmentMasks();
        __builtin_prefetch(&masks, 0, 3);
        for (const auto &update : cutter->getSRCNodeUpdates(node_id)) {
            auto &src_map_value = new_label->SRCmap[update.active_idx];
            src_map_value += update.add;
            const bool overflow = src_map_value >= update.den;
            src_map_value -= overflow ? update.den : 0;
            total_cost_update -= overflow ? update.dual : 0;
        }

        for (const auto active_idx : cutter->getSRCNodeClears(node_id)) { new_label->SRCmap[active_idx] = 0; }
#else
        for (const auto &active_cut : active_cuts) {
            const auto &cut           = *active_cut.cut_ptr;
            auto       &src_map_value = new_label->SRCmap[active_cut.index];
            if (!cut.isSRCMemoryArc(L_prime->node_id, node_id)) { src_map_value = 0; }
            if (cut.isSRCset(node_id)) {
                src_map_value += cut.srcMultiplier(node_id);
                if (src_map_value >= cut.p.den) {
                    src_map_value -= cut.p.den;
                    total_cost_update -= active_cut.dual_value;
                }
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
inline bool BucketGraph::dominates_resource_path(const Label *__restrict new_label,
                                                 const Label *__restrict label) noexcept {
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
    return true;
}

template <Direction D, Stage S>
inline bool BucketGraph::is_dominated(const Label *__restrict new_label, const Label *__restrict label) noexcept {
    // A label cannot dominate if its cost is higher.
    // const double cost_diff = label->cost - new_label->cost;
#if defined(SRC)
    if constexpr (!(S == Stage::Four || S == Stage::Enumerate)) {
        if (numericutils::gt(label->cost, new_label->cost)) { return false; }
    }
#else
    if (numericutils::gt(label->cost, new_label->cost)) { return false; }
#endif

    if (!dominates_resource_path<D, S>(new_label, label)) { return false; }

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
            if (__builtin_expect(lbl_srcs[cut.index] > new_lbl_srcs[cut.index], 0)) { dual_sum += cut.dual_value; }
        }
        return !numericutils::gt(local_cost_diff, dual_sum);
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
    const auto  &label_res  = L->resources;
    const size_t n_res      = options.resources.size();
    const uint64_t label_visited_signature = L->visited_signature();
    constexpr bool uses_visited_dominance =
        S == Stage::Three || S == Stage::Four || S == Stage::Enumerate;

    // Use static thread-local stack to avoid repeated allocations
    static thread_local std::vector<int> stack_buffer;
    if (stack_buffer.capacity() < R_SIZE) { stack_buffer.reserve(R_SIZE); }
    stack_buffer.clear();
    stack_buffer.push_back(bucket);

    // Direct pointer to bit mask lookup for fast bit operations
    const uint64_t *const bit_mask_lookup_ptr = bit_mask_lookup.data();

    // Read-only dominance scan: ask only "does some predecessor label cur
    // dominate L?". Per Sadykov & Vanderbeck (2021), this routine MUST NOT
    // mutate labels in compwise-smaller buckets — they belong to already-
    // processed (or concurrently processed) buckets, and marking them
    // dominated mid-walk silently drops queued pending labels. Cost is a
    // necessary condition for cur-dominates-L, so once cur->cost > L.cost
    // (sorted bin) we can break.
    const double cost_hi_break = label_cost + numericutils::eps;
#if defined(SRC)
    constexpr bool src_adjusted_dominance = (S == Stage::Four || S == Stage::Enumerate);
#else
    constexpr bool src_adjusted_dominance = false;
#endif

    auto inline_check_dominance = [&](const BucketLabelSoAView &view, uint &n_dom) -> bool {
        const size_t size = view.labels.size();
        if (profile_labeling) profile_record_inner_bin_scan(D, S, size);
#if defined(BALDES_HAS_SIMD)
        if constexpr (!src_adjusted_dominance) {
            namespace stdx = std::experimental;
            using simd_d   = stdx::native_simd<double>;
            using mask_d   = typename simd_d::mask_type;

            constexpr size_t simd_width = simd_d::size();
            const simd_d     cost_limit(cost_hi_break);
            const simd_d     eps(numericutils::eps);

            size_t i = 0;
            for (; i + simd_width <= size; i += simd_width) {
                const simd_d costs(view.costs.data() + i, stdx::element_aligned);
                mask_d       candidate = costs <= cost_limit;
                if (!stdx::any_of(candidate)) {
                    if (view.costs[i] > cost_hi_break) return false;
                    continue;
                }

                if constexpr (D == Direction::Forward) {
                    for (size_t r = 1; r < n_res; ++r) {
                        const simd_d res(view.resources[r].data() + i, stdx::element_aligned);
                        candidate &= res <= (simd_d(label_res[r]) + eps);
                    }
                    if (n_res > 0) {
                        const simd_d res0(view.resources[0].data() + i, stdx::element_aligned);
                        candidate &= res0 <= (simd_d(label_res[0]) + eps);
                    }
                } else {
                    for (size_t r = 1; r < n_res; ++r) {
                        const simd_d res(view.resources[r].data() + i, stdx::element_aligned);
                        candidate &= res >= (simd_d(label_res[r]) - eps);
                    }
                    if (n_res > 0) {
                        const simd_d res0(view.resources[0].data() + i, stdx::element_aligned);
                        candidate &= res0 >= (simd_d(label_res[0]) - eps);
                    }
                }

                if (!stdx::any_of(candidate)) continue;
                for (size_t lane = 0; lane < simd_width; ++lane) {
                    if (!candidate[lane]) continue;
                    if constexpr (uses_visited_dominance) {
                        if ((view.visited_signatures[i + lane] & ~label_visited_signature) != 0) {
                            if (profile_labeling) profile_record_signature_rejection(D, S);
                            continue;
                        }
                    }
                    Label *cur = view.labels[i + lane];
                    if (cur->is_dominated) continue;
                    if (is_dominated<D, S>(L, cur)) {
                        ++n_dom;
                        return true;
                    }
                }
            }

            for (; i < size; ++i) {
                if (view.costs[i] > cost_hi_break) break;

                bool resource_candidate = true;
                if constexpr (D == Direction::Forward) {
                    for (size_t r = 1; r < n_res; ++r) {
                        if (numericutils::gt(view.resources[r][i], label_res[r])) {
                            resource_candidate = false;
                            break;
                        }
                    }
                    if (resource_candidate && n_res > 0 && numericutils::gt(view.resources[0][i], label_res[0])) {
                        resource_candidate = false;
                    }
                } else {
                    for (size_t r = 1; r < n_res; ++r) {
                        if (numericutils::lt(view.resources[r][i], label_res[r])) {
                            resource_candidate = false;
                            break;
                        }
                    }
                    if (resource_candidate && n_res > 0 && numericutils::lt(view.resources[0][i], label_res[0])) {
                        resource_candidate = false;
                    }
                }
                if (!resource_candidate) continue;

                if constexpr (uses_visited_dominance) {
                    if ((view.visited_signatures[i] & ~label_visited_signature) != 0) {
                        if (profile_labeling) profile_record_signature_rejection(D, S);
                        continue;
                    }
                }

                Label *cur = view.labels[i];
                if (cur->is_dominated) continue;
                if (is_dominated<D, S>(L, cur)) {
                    ++n_dom;
                    return true;
                }
            }
            return false;
        }
#endif
        for (size_t i = 0; i < size; ++i) {
            // Bin is RC-sorted ascending; once cur is more expensive than L,
            // no later label can dominate L (cost ordering).
            if constexpr (!src_adjusted_dominance) {
                if (view.costs[i] > cost_hi_break) break;
            }

            bool resource_candidate = true;
            if constexpr (D == Direction::Forward) {
                for (size_t r = 1; r < n_res; ++r) {
                    if (numericutils::gt(view.resources[r][i], label_res[r])) {
                        resource_candidate = false;
                        break;
                    }
                }
                if (resource_candidate && n_res > 0 && numericutils::gt(view.resources[0][i], label_res[0])) {
                    resource_candidate = false;
                }
            } else {
                for (size_t r = 1; r < n_res; ++r) {
                    if (numericutils::lt(view.resources[r][i], label_res[r])) {
                        resource_candidate = false;
                        break;
                    }
                }
                if (resource_candidate && n_res > 0 && numericutils::lt(view.resources[0][i], label_res[0])) {
                    resource_candidate = false;
                }
            }
            if (!resource_candidate) continue;

            if constexpr (uses_visited_dominance) {
                if ((view.visited_signatures[i] & ~label_visited_signature) != 0) {
                    if (profile_labeling) profile_record_signature_rejection(D, S);
                    continue;
                }
            }

            Label *cur = view.labels[i];
            if (cur->is_dominated) continue;
            if (is_dominated<D, S>(L, cur)) {
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
            // extra is unsorted -> no break, just skip.
            if constexpr (!src_adjusted_dominance) {
                if (label->cost > cost_hi_break) continue;
            }
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
            if (mother_bucket.check_dominance_soa(L, inline_check_dominance, inline_check_extra_dominance, stat_n_dom))
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
