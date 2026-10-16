/**
 * @file BucketSolve.h
 * @brief Defines the BucketGraph solving methods for bucket-based optimization.
 *
 */

#pragma once

#include <bit>
#include <cstring>

#include "pricing/bucket_graph/ArcElimination.h"
#include "pricing/bucket_graph/Resources.h"
#include "core/Definitions.h"
#include "cuts/rank1/Rank1Cuts.h"

#if !defined(BALDES_DISABLE_SIMD) && __has_include(<experimental/simd>)
#include <experimental/simd>
#define BALDES_HAS_SIMD 1
#endif

#ifdef __AVX2__
#include "pricing/bucket_graph/Dominance.h"
#endif

#include <execution>

#include "../../third_party/small_vector.hpp"
#include "pricing/bucket_graph/Concatenation.h"
#include "pricing/bucket_graph/Pricing.h"
#include "pricing/bucket_graph/PricingPass.h"
#include "pricing/bucket_graph/Topology.h"
#include "pricing/bucket_graph/Utilities.h"

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
    // Dominance brackets need no SRC widening: with the limited-memory R1C
    // rule in dominates_cost_state the SRC allowance is never positive, so a
    // dominator is never more expensive than the label it dominates.
    constexpr double src_compensation_bound = 0.0;

    // Reset the total label count.
    n_labels = 0;

    // Retain direction-local work buffers across pricing stages. Forward and
    // backward labeling use distinct storage because they can run in parallel.
    auto &scratch = assign_buckets<D>(fw_labeling_scratch, bw_labeling_scratch);

    const size_t n_segments = (n_buckets + 63) >> 6;
    scratch.bucket_visited.assign(n_segments, 0);
    auto &Bvisited = scratch.bucket_visited;
    auto &touched_segments = scratch.touched_segments;
    touched_segments.clear();
    touched_segments.reserve(std::min<size_t>(n_segments, 64));
    scratch.bucket_pos_in_scc.assign(n_buckets, -1);
    auto &bucket_pos_in_scc = scratch.bucket_pos_in_scc;

    // Process only newly queued labels while preserving the bucket order
    // inside each SCC.
    auto &pending_labels_by_bucket = scratch.pending_labels_by_bucket;
    auto &pending_label_cursor = scratch.pending_label_cursor;
    auto &bucket_is_active = scratch.bucket_is_active;
    auto &active_bucket_heap = scratch.active_bucket_heap;

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

        // Retain the per-position vector capacities across SCCs. Clearing the
        // outer vector destroyed every inner allocation on each component,
        // even though the same buffers are reused throughout every pricing
        // pass.
        if (pending_labels_by_bucket.size() < scc_buckets.size()) {
            pending_labels_by_bucket.resize(scc_buckets.size());
        }
        for (size_t i = 0; i < scc_buckets.size(); ++i) {
            pending_labels_by_bucket[i].clear();
        }
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

            // Extension order within a connected component:
            // filter the bucket's pending labels first (stage cut-off and
            // dominance by component-wise smaller buckets), then extend
            // arc-outer / label-inner so each bucket arc's data (resource
            // increments, head node bounds, ng row, costs) stays hot across
            // labels. Bucket arcs never lead back into this bucket, so a
            // round's own insertions cannot add labels to it.
            const auto &node_arcs = buckets[bucket].template get_bucket_arcs<D>();
            static thread_local std::vector<Label *>                     round_labels;
            static thread_local std::vector<std::pair<Label *, Label *>> destination_batch;

            while (pending_cursor < pending_labels.size()) {
                round_labels.clear();
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

                    if (__builtin_expect(node_arcs.empty(), 0)) {
                        label->set_extended(true);
                        continue;
                    }
                    round_labels.push_back(label);
                }
                if (round_labels.empty()) break;

                for (size_t arc_idx = 0; arc_idx < node_arcs.size(); ++arc_idx) {
                    const auto &arc = node_arcs[arc_idx];
                    if (arc_idx + 1 < node_arcs.size()) { __builtin_prefetch(&node_arcs[arc_idx + 1], 0, 3); }

                    // Extend every round label along this arc before mutating
                    // the destination buckets; insertion still compares later
                    // siblings against earlier survivors.
                    destination_batch.clear();
                    for (Label *label : round_labels) {
                        auto new_label = Extend<D, S, ArcType::Bucket, Mutability::Mut, F>(label, arc);
                        if (new_label != nullptr) destination_batch.emplace_back(label, new_label);
                    }

                for (const auto &extension : destination_batch) {
                    Label *const label     = extension.first;
                    Label *const new_label = extension.second;
                    // Process each new label produced by the extension.
                    if (new_label != nullptr) {
                        if (new_label->is_dominated) continue;
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

                        mother_bucket.flush_extra_labels_if_large();
                        const auto &to_bucket_labels = mother_bucket.get_sorted_labels();
                        const auto &to_bucket_extra  = mother_bucket.get_extra_labels();
                        constexpr bool uses_visited_dominance =
                            S == Stage::Three || S == Stage::Four || S == Stage::Enumerate;
                        const uint64_t new_visited_signature = new_label->visited_signature();
                        // Enumeration must keep every route under the gap, so a
                        // label is only dominated by one with the same visited
                        // set (as in enumeration storage); both subset
                        // tests then mean equality.
                        auto visited_equal = [&](size_t index) noexcept {
                            if (mother_bucket.soa_visited_signatures[index] != new_visited_signature) return false;
                            for (size_t word = 0; word < Label::bitmap_words; ++word) {
                                if (mother_bucket.soa_visited_words[word][index] != new_label->visited_bitmap[word]) {
                                    return false;
                                }
                            }
                            return true;
                        };
                        auto existing_visited_subset_of_new = [&](size_t index) noexcept {
                            if constexpr (S == Stage::Enumerate) return visited_equal(index);
                            const uint64_t existing_signature = mother_bucket.soa_visited_signatures[index];
                            if ((existing_signature & ~new_visited_signature) != 0) { return false; }
                            if (Label::exact_visited_pair(existing_signature, new_visited_signature)) return true;
                            for (size_t word = 0; word < Label::bitmap_words; ++word) {
                                if ((mother_bucket.soa_visited_words[word][index] &
                                     ~new_label->visited_bitmap[word]) != 0) {
                                    return false;
                                }
                            }
                            return true;
                        };
                        auto new_visited_subset_of_existing = [&](size_t index) noexcept {
                            if constexpr (S == Stage::Enumerate) return visited_equal(index);
                            const uint64_t existing_signature = mother_bucket.soa_visited_signatures[index];
                            if ((new_visited_signature & ~existing_signature) != 0) { return false; }
                            if (Label::exact_visited_pair(existing_signature, new_visited_signature)) return true;
                            for (size_t word = 0; word < Label::bitmap_words; ++word) {
                                if ((new_label->visited_bitmap[word] &
                                     ~mother_bucket.soa_visited_words[word][index]) != 0) {
                                    return false;
                                }
                            }
                            return true;
                        };
// SoA cache maintained incrementally; always valid.

                        // Prefetch first few destination bucket labels
                        if (!to_bucket_labels.empty()) {
                            for (size_t i = 0; i < std::min(size_t(4), to_bucket_labels.size()); ++i) {
                                __builtin_prefetch(to_bucket_labels[i], 0, 3);
                            }
                        }

                        bool dominated = false;
                        if constexpr (S == Stage::One) {
                            // Stage One is cost-only. Reject against the
                            // cached bucket minimum in O(1); only a new record
                            // minimum needs to invalidate older labels.
                            dominated = mother_bucket.apply_stage_one_dominance(new_label, bucket_scan_count);
                            if constexpr (D == Direction::Forward) {
                                regen_dominance_checks_fw += bucket_scan_count;
                            } else {
                                regen_dominance_checks_bw += bucket_scan_count;
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
                            // SRC states only ever add to the dominator's
                            // side (see dominates_cost_state), so raw cost
                            // order is decisive in every stage.
                            const double cost_lo =
                                new_label->cost - src_compensation_bound - numericutils::eps;
                            const double cost_hi =
                                new_label->cost + src_compensation_bound + numericutils::eps;

                            auto label_cost_less = [](const Label *label, double value) { return label->cost < value; };

                            // When every SRC dual is non-positive (and always
                            // outside SRC stages), raw cost decides which
                            // direction is possible even inside the bracket:
                            // cur can only dominate new if it is not more
                            // expensive, and vice versa. The 2*eps slack keeps
                            // these filters looser than dominates_cost_state.
                            constexpr bool src_one_sided = true;
                            const double cur_may_dominate_max     = new_label->cost + 2 * numericutils::eps;
                            const double cur_may_be_dominated_min = new_label->cost - 2 * numericutils::eps;

                            // Full two-way test for a bracketed pair. Returns
                            // true when new_label is dominated by cur.
                            auto check_bracketed_pair = [&](Label *cur) -> bool {
                                const double cur_cost = cur->cost;
                                if (!src_one_sided || cur_cost <= cur_may_dominate_max) {
                                    if (is_dominated<D, S>(new_label, cur)) return true;
                                }
                                if (!src_one_sided || cur_cost >= cur_may_be_dominated_min) {
                                    if (is_dominated<D, S>(cur, new_label)) cur->set_dominated(true);
                                }
                                return false;
                            };

                            auto scan_new_dominated_by = [&](size_t begin, size_t end) {
#if defined(BALDES_HAS_SIMD)
                                {
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
                                                if (!existing_visited_subset_of_new(i + lane)) {
                                                    if (profile_labeling) profile_record_signature_rejection(D, S);
                                                    continue;
                                                }
                                            }
                                            Label *cur = to_bucket_labels[i + lane];
                                            if (__builtin_expect(cur->is_dominated, 0)) continue;
                                            ++stat_n_dom;
                                            dominated = true;
                                            return;
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
                                {
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
                                                if (!new_visited_subset_of_existing(i + lane)) {
                                                    if (profile_labeling) profile_record_signature_rejection(D, S);
                                                    continue;
                                                }
                                            }
                                            Label *cur = to_bucket_labels[i + lane];
                                            if (__builtin_expect(cur->is_dominated, 0)) continue;
                                            cur->set_dominated(true);
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
                                {
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
                                        if (src_one_sided) {
                                            const simd_d cost(mother_bucket.soa_costs.data() + i,
                                                              stdx::element_aligned);
                                            cur_dominates_new &= cost <= simd_d(cur_may_dominate_max);
                                            new_dominates_cur &= cost >= simd_d(cur_may_be_dominated_min);
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
                                                    !existing_visited_subset_of_new(i + lane)) {
                                                    cur_dominates_new[lane] = false;
                                                    if (profile_labeling) profile_record_signature_rejection(D, S);
                                                }
                                                if (new_dominates_cur[lane] &&
                                                    !new_visited_subset_of_existing(i + lane)) {
                                                    new_dominates_cur[lane] = false;
                                                    if (profile_labeling) profile_record_signature_rejection(D, S);
                                                }
                                                if (!cur_dominates_new[lane] && !new_dominates_cur[lane]) continue;
                                            }
                                            Label *cur = to_bucket_labels[i + lane];
                                            if (__builtin_expect(cur->is_dominated, 0)) continue;
                                            const bool eq_resources = equality_resources_match<D>(cur, new_label);
                                            if (cur_dominates_new[lane] && eq_resources &&
                                                dominates_cost_state<S>(new_label, cur)) {
                                                ++stat_n_dom;
                                                dominated = true;
                                                return;
                                            }
                                            if (new_dominates_cur[lane] && eq_resources &&
                                                dominates_cost_state<S>(cur, new_label)) {
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
                                    if (check_bracketed_pair(cur)) {
                                        ++stat_n_dom;
                                        dominated = true;
                                        break;
                                    }
                                }
                            };

                            // Forward path: walk sorted committed labels once.
                            const size_t to_bucket_size = to_bucket_labels.size();
                            {
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
                                        } else if (check_bracketed_pair(cur)) {
                                            ++stat_n_dom;
                                            dominated = true;
                                            break;
                                        }
                                    }
                                } else {
                                    // SoA cache maintained incrementally; always valid.
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
                                    if (cur_cost < cost_lo) {
                                        if (dominates_resource_path<D, S>(new_label, cur)) {
                                            ++stat_n_dom;
                                            dominated = true;
                                            break;
                                        }
                                    } else if (cur_cost > cost_hi) {
                                        if (dominates_resource_path<D, S>(cur, new_label)) { cur->set_dominated(true); }
                                    } else if (check_bracketed_pair(cur)) {
                                        ++stat_n_dom;
                                        dominated = true;
                                        break;
                                    }
                                }
                            }
                            if constexpr (D == Direction::Forward) {
                                regen_dominance_checks_fw += inner_scan_count;
                            } else {
                                regen_dominance_checks_bw += inner_scan_count;
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
                }
                for (Label *label : round_labels) label->set_extended(true);
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
    Label *result = nullptr;

    // Resource state after the extension. Only the first n_resources slots
    // are meaningful; the rest stay zero.
    std::array<double, R_SIZE> new_resources{};
    const size_t               n_resources = options.resources.size();

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
    // Arc NG memory cycle check.
    // If physical-arc NG memories are enabled, use them for tighter cycle detection.
    // The arc NG memory tracks which elementarity sets are forbidden on each physical arc.
    if constexpr (F != Full::PSTEP && F != Full::TSP) {
        if (arc_ng_memory.enabled && A == ArcType::Bucket) {
            // Get the physical arc ID from the bucket arc.
            int phys_id = -1;
            if constexpr (A == ArcType::Bucket) {
                phys_id = gamma.physical_arc_id;
            }
            if (phys_id >= 0) {
                auto it = arc_ng_memory.arcs.find(phys_id);
                if (it != arc_ng_memory.arcs.end()) {
                    // Check if current label state intersects with arc's visited sets.
                    const auto& visited = it->second.visited_sets;
                    bool forbidden = false;
                    for (int set_id : visited) {
                        if (static_cast<size_t>(set_id) < Label::visited_sets.size() &&
                            Label::visited_sets[set_id]) {
                            forbidden = true;
                            break;
                        }
                    }
                    if (forbidden) {
                        return result;  // Reject: would create a cycle
                    }
                    // Update label state: merge arc's visited sets into label.
                    for (int set_id : visited) {
                        if (static_cast<size_t>(set_id) < Label::visited_sets.size()) {
                            Label::visited_sets[set_id] = true;
                        }
                    }
                }
            }
        }
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

            // Buckets only bound the main (bucketed) resources, which occupy
            // the leading resource slots (as in get_bucket_number); the other
            // resources are not lifted by a jump.
            std::array<double, R_SIZE> jumped_resources = initial_resources;
            const size_t n_bucketed = std::min({n_resources, jump_bucket.lb.size(), jump_bucket.ub.size()});
            for (size_t i = 0; i < n_bucketed; ++i) {
                if constexpr (D == Direction::Forward) {
                    jumped_resources[i] = std::max(jumped_resources[i], static_cast<double>(jump_bucket.lb[i]));
                } else {
                    jumped_resources[i] = std::min(jumped_resources[i], static_cast<double>(jump_bucket.ub[i]));
                }
            }
            return process_all_resources<D>(new_resources, jumped_resources, gamma, nodes[node_id], n_resources);
        };

        if constexpr (A != ArcType::Jump) {
            bool feasible;
            if (is_jump_arc) {
                feasible = process_jump_resources();
            } else if (n_resources == 1 &&
                       static_cast<ResourceType>(options.resource_type[0]) == ResourceType::Disposable) {
                // The overwhelmingly common VRPTW kernel: one disposable
                // time resource. Avoid the generic resource loop and switch.
                feasible = process_disposable_resource<D>(new_resources[0], initial_resources[0],
                                                          gamma.resource_increment[0], nodes[node_id].lb[0],
                                                          nodes[node_id].ub[0]);
            } else {
                feasible = process_all_resources<D>(new_resources, initial_resources, gamma, nodes[node_id],
                                                    n_resources);
            }
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

    const auto &direction_buckets = assign_buckets<D>(fw_buckets, bw_buckets);
    if (to_bucket < 0 || to_bucket >= static_cast<int>(direction_buckets.size())) {
        if constexpr (F == Full::Reverse)
            return -1;
        else
            return result;
    }

    // Handle Reverse mode early
    if constexpr (F == Full::Reverse) {
        if (new_resources[options.main_resources[0]] <= std::floor(q_star[options.main_resources[0]])) return -1;

        return to_bucket;
    }

    // Calculate new cost
#ifdef CUSTOM_COST
    const auto distance = getcij(initial_node_id, node_id);
    double     new_cost = cost_calculator.calculate_cost(
        initial_cost, distance, node_id, std::vector<double>(new_resources.begin(), new_resources.begin() + n_resources));
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

    // Exact pricing can use the already-computed backward labels as a
    // current-cost completion lower bound. Account conservatively for every
    // possible future SRC discount before rejecting the extension. This runs
    // before allocation and before bitmap/route/SRC state copies.
    if constexpr (D == Direction::Forward && F == Full::Partial &&
                  (S == Stage::Four || S == Stage::Enumerate)) {
        if (forward_completion_bounds_ready) {
            const int opposite_bucket = get_opposite_bucket_number<D>(to_bucket, new_resources);
            if (opposite_bucket >= 0 && opposite_bucket < static_cast<int>(bw_c_bar.size())) {
                const double completion = bw_c_bar[opposite_bucket];
                if (std::isfinite(completion)) {
                    completion_bound_attempts.fetch_add(1, std::memory_order_relaxed);
                    const double cutoff = S == Stage::Enumerate
                                              ? std::min(gap, enumeration_route_cutoff.load(std::memory_order_relaxed))
                                              : 0.0;
                    const double base_cost = new_cost - pricing_src_compensation_bound;
                    if (base_cost + completion >= cutoff - numericutils::eps ||
                        !backward_completion_exists(L_prime, node_id, opposite_bucket, base_cost, cutoff)) {
                        completion_bound_rejections.fetch_add(1, std::memory_order_relaxed);
                        return result;
                    }
                }
            }
        }
    }

    // Create and initialize a new label
    auto &label_pool = assign_buckets<D>(label_pool_fw, label_pool_bw);
    auto  new_label  = label_pool->acquire(to_bucket);
    new_label->initialize(to_bucket, new_cost, new_resources, node_id);
    new_label->vertex    = to_bucket;
    new_label->real_cost = L_prime->real_cost + distance; // Use cached distance
    new_label->path_len  = L_prime->path_len + 1;

    // Update visited bitmap: intersect parent's visited bits with neighborhood
    // mask. Enumeration keeps the full visited set instead (elementary
    // routes; see the Enumerate dominance rule in dominates_resource_path).
    if constexpr (F != Full::PSTEP && S != Stage::Enumerate) {
        // Branch-free so the compiler can vectorize it; ng rows are padded
        // to Label::bitmap_words (see calculate_neighborhoods).
        const uint64_t *__restrict ng     = neighborhoods_bitmap[node_id].data();
        const uint64_t *__restrict parent = L_prime->visited_bitmap.data();
        uint64_t *__restrict visited      = new_label->visited_bitmap.data();
        for (size_t i = 0; i < Label::bitmap_words; ++i) { visited[i] = parent[i] & ng[i]; }
    } else {
        // Use memcpy for direct bitmap copy (likely faster than assignment for
        // large bitmaps)
        std::memcpy(new_label->visited_bitmap.data(), L_prime->visited_bitmap.data(),
                    new_label->visited_bitmap.size() * sizeof(uint64_t));
    }

    set_node_visited(new_label->visited_bitmap, node_id);
    if constexpr (S == Stage::Enumerate) {
        // Full (elementary) memory: the node-local ng word is not exact here.
        new_label->visited_sig = new_label->conservative_visited_signature();
    } else {
        finalize_visited_signature<F>(new_label, node_id);
    }

    // Keep the extension O(1). Routes are materialized only for labels that
    // survive pricing and become column candidates.
    new_label->parent = L_prime;

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
        double     total_cost_update = 0.0;

#if !defined(SRC_MEMORY_MODE_ARC)
        for (const auto &update : cutter->getSRCNodeUpdates(node_id)) {
            auto          &slot     = new_label->SRCmap[update.active_idx];
            unsigned       value    = slot + update.add;
            const bool     overflow = value >= update.den;
            value -= overflow ? update.den : 0;
            slot = static_cast<uint8_t>(value);
            total_cost_update -= overflow ? update.dual : 0;
        }

        for (const auto active_idx : cutter->getSRCNodeClears(node_id)) { new_label->SRCmap[active_idx] = 0; }
#else
        const auto &active_cuts = cutter->getActiveCuts();
        for (const auto &active_cut : active_cuts) {
            const auto &cut           = *active_cut.cut_ptr;
            auto       &slot          = new_label->SRCmap[active_cut.index];
            unsigned    src_map_value = slot;
            if (!cut.isSRCMemoryArc(L_prime->node_id, node_id)) { src_map_value = 0; }
            if (cut.isSRCset(node_id)) {
                src_map_value += cut.srcMultiplier(node_id);
                if (src_map_value >= static_cast<unsigned>(cut.p.den)) {
                    src_map_value -= cut.p.den;
                    total_cost_update -= active_cut.dual_value;
                }
            }
            slot = static_cast<uint8_t>(src_map_value);
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
    if (!equality_resources_match<D>(label, new_label)) { return false; }

    // visits of label.
    if constexpr (S == Stage::Three || S == Stage::Four || S == Stage::Enumerate) {
        const uint64_t label_signature = label->visited_signature();
        const uint64_t new_signature   = new_label->visited_signature();
        if constexpr (S == Stage::Enumerate) {
            // Enumeration: dominance only between labels with the same
            // visited set, so no route under the gap is lost.
            return label_signature == new_signature && label->visited_bitmap == new_label->visited_bitmap;
        }
        if ((label_signature & ~new_signature) != 0) return false;
        if (Label::exact_visited_pair(label_signature, new_signature)) return true;

        const size_t n_bitmap = label->visited_bitmap.size();

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
    if (!dominates_resource_path<D, S>(new_label, label)) { return false; }
    return dominates_cost_state<S>(new_label, label);
}

template <Stage S>
inline bool BucketGraph::dominates_cost_state(const Label *__restrict new_label,
                                              const Label *__restrict label) noexcept {
    // A label cannot dominate if its cost is higher.
#if defined(SRC)
    if constexpr (!(S == Stage::Four || S == Stage::Enumerate)) {
        if (numericutils::gt(label->cost, new_label->cost)) { return false; }
    }
#else
    if (numericutils::gt(label->cost, new_label->cost)) { return false; }
#endif

#ifdef SRC
    // For Stage Four or Enumerate, apply additional SRC-based cost
    // adjustments.
    if constexpr (S == Stage::Four || S == Stage::Enumerate) {
        const double local_cost_diff = label->cost - new_label->cost;
        // The SRC allowance below is never positive, so a more expensive
        // label can never dominate.
        if (numericutils::gt(local_cost_diff, 0.0)) return false;

        // Limited-memory R1C dominance (per-cut signed charge):
        // a cut with dual sigma is charged (red_cost -= sigma) when its state
        // overflows, so a higher state reaches the charge first. The
        // dominator must absorb
        //   sigma < 0 on cuts where its state is higher (it may pay first),
        //   sigma > 0 on cuts where new_label's state is higher (new_label
        //              may collect the discount first),
        // i.e. local_cost_diff <= sum_{label>new} min(sigma,0)
        //                         - sum_{new>label} max(sigma,0).
        // Inactive slots have a zero dual, so whole chunks can be compared
        // without masking.
        const uint8_t *label_state = label->SRCmap.data();
        const uint8_t *new_state   = new_label->SRCmap.data();
        double         dual_sum    = 0.0;
        for (size_t chunk = 0; chunk < src_dense_chunks; ++chunk) {
            const size_t base = chunk * 32;
            uint32_t     mask = src_greater_mask32(label_state + base, new_state + base);
            while (mask) {
                dual_sum += std::min(src_dense_duals[base + std::countr_zero(mask)], 0.0);
                mask &= mask - 1;
            }
            if (!src_duals_nonpositive) {
                uint32_t lower = src_greater_mask32(new_state + base, label_state + base);
                while (lower) {
                    dual_sum -= std::max(src_dense_duals[base + std::countr_zero(lower)], 0.0);
                    lower &= lower - 1;
                }
            }
        }
        return !numericutils::gt(local_cost_diff, dual_sum);
    }
#endif
    return true;
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
    // necessary condition for cur-dominates-L in every stage (the SRC
    // allowance in dominates_cost_state is never positive).
    const double cost_hi_break = label_cost + numericutils::eps;
    const double bounded_cost_hi = cost_hi_break;

    auto inline_check_dominance = [&](const BucketLabelSoAView &view, uint &n_dom) -> bool {
        const size_t size = view.labels.size();
        if (profile_labeling) profile_record_inner_bin_scan(D, S, size);
        auto visited_subset = [&](size_t index) noexcept {
            if constexpr (S == Stage::Enumerate) {
                // Enumeration: same visited set only (see the insertion scan).
                if (view.visited_signatures[index] != label_visited_signature) return false;
                for (size_t word = 0; word < Label::bitmap_words; ++word) {
                    if (view.visited_words[word][index] != L->visited_bitmap[word]) return false;
                }
                return true;
            }
            const uint64_t signature = view.visited_signatures[index];
            if ((signature & ~label_visited_signature) != 0) return false;
            if (Label::exact_visited_pair(signature, label_visited_signature)) return true;
            for (size_t word = 0; word < Label::bitmap_words; ++word) {
                if ((view.visited_words[word][index] & ~L->visited_bitmap[word]) != 0) return false;
            }
            return true;
        };
#if defined(BALDES_HAS_SIMD)
        {
            namespace stdx = std::experimental;
            using simd_d   = stdx::native_simd<double>;
            using mask_d   = typename simd_d::mask_type;

            constexpr size_t simd_width = simd_d::size();
            const simd_d     cost_limit(bounded_cost_hi);
            const simd_d     eps(numericutils::eps);

            size_t i = 0;
            for (; i + simd_width <= size; i += simd_width) {
                const simd_d costs(view.costs.data() + i, stdx::element_aligned);
                mask_d       candidate = costs <= cost_limit;
                if (!stdx::any_of(candidate)) {
                    if (view.costs[i] > bounded_cost_hi) return false;
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
                        if (!visited_subset(i + lane)) {
                            if (profile_labeling) profile_record_signature_rejection(D, S);
                            continue;
                        }
                    }
                    Label *cur = view.labels[i + lane];
                    if (cur->is_dominated) continue;
                    if (equality_resources_match<D>(cur, L) && dominates_cost_state<S>(L, cur)) {
                        ++n_dom;
                        return true;
                    }
                }
            }

            for (; i < size; ++i) {
                if (view.costs[i] > bounded_cost_hi) break;

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
                    if (!visited_subset(i)) {
                        if (profile_labeling) profile_record_signature_rejection(D, S);
                        continue;
                    }
                }

                Label *cur = view.labels[i];
                if (cur->is_dominated) continue;
                if (equality_resources_match<D>(cur, L) && dominates_cost_state<S>(L, cur)) {
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
            if (view.costs[i] > bounded_cost_hi) break;

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
            if (label->cost > bounded_cost_hi) continue;
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

        // Subtree prune: c_bar[b] bounds the cost of every label in b and in
        // all buckets reachable from b through Phi, so none of them can
        // dominate L on cost. Sibling subtrees still on the stack are not
        // covered by this bound and must still be scanned.
        if (likely(bounded_cost_hi < c_bar[current_bucket] &&
                   bucket_scc_rank[current_bucket] < label_rank))
            continue;

        // Whole-bucket prune: if no label in this bucket or any predecessor bin
        // at this node can dominate L in the second resource, skip it entirely.
        if (options.main_resources.size() > 1) {
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
            if constexpr (S == Stage::One) {
                // Phi contains only component-wise better resource buckets.
                // Stage One has no path-state dominance, so the bucket's
                // cached minimum cost is a complete dominance certificate.
                if (profile_labeling && !mother_bucket.is_empty()) {
                    profile_record_inner_bin_scan(D, S, 1);
                }
                if (mother_bucket.get_cb() <= label_cost) {
                    ++stat_n_dom;
                    return true;
                }
            } else {
                if (mother_bucket.check_dominance_soa(L, inline_check_dominance, inline_check_extra_dominance,
                                                      stat_n_dom)) {
                    return true;
                }
            }
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
            if (options.main_resources.size() > 1 && local_b >= 0 && local_b < num_buckets[node_id]) {
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
 * @brief Exact completion test for a forward extension into node_id
 * (completion check in exact mode).
 *
 * The c_bar test in Extend bounds only the cost of the best backward
 * completion. This walks the backward bucket closure of opposite_bucket
 * (opposite_bucket plus its Phi predecessors, pruned by bw_c_bar like the
 * concatenation scan) and looks for one backward label at node_id that is
 * cheap enough under the same cost convention and whose ng-memory meets the
 * extended forward memory only at node_id. The forward memory is the
 * extension's (parent & N(node_id)), computed without allocating the label.
 *
 * Returns false only when every candidate is ruled out; a backward label
 * dominated away is covered by its dominator (cheaper, memory subset). The
 * scan is budgeted: once kLabelBudget labels have been examined the
 * completion is assumed to exist, so the extension is kept.
 */
inline bool BucketGraph::backward_completion_exists(const Label *parent, int node_id, int opposite_bucket,
                                                    double base_cost, double cutoff) const {
    constexpr size_t kLabelBudget = 256;
    const double     limit        = cutoff - numericutils::eps;

    static thread_local std::vector<int>      stack;
    static thread_local std::vector<uint64_t> seen;
    static thread_local std::vector<uint32_t> touched;
    const size_t n_segments = (bw_buckets.size() + 63) >> 6;
    if (seen.size() < n_segments) seen.resize(n_segments, 0);
    stack.clear();
    touched.clear();

    std::array<uint64_t, Label::bitmap_words> forward_memory;
    const auto &ng_row = neighborhoods_bitmap[node_id];
    for (size_t w = 0; w < Label::bitmap_words; ++w) {
        forward_memory[w] = parent->visited_bitmap[w] & (w < ng_row.size() ? ng_row[w] : 0);
    }
    forward_memory[static_cast<size_t>(node_id) >> 6] &= ~(1ULL << (node_id & 63));

    size_t scanned = 0;
    bool   found   = false;
    // Returns true when the scan can stop (completion found or budget hit).
    auto examine = [&](const Label *backward) noexcept {
        if (backward->is_dominated || base_cost + backward->cost >= limit) return false;
        if (++scanned > kLabelBudget || !visited_overlap(forward_memory, backward->visited_bitmap)) {
            found = true;
            return true;
        }
        return false;
    };

    stack.push_back(opposite_bucket);
    while (!stack.empty() && !found) {
        const int      b       = stack.back();
        const size_t   segment = static_cast<size_t>(b) >> 6;
        const uint64_t bit     = 1ULL << (b & 63);
        stack.pop_back();
        if (seen[segment] & bit) continue;
        if (seen[segment] == 0) touched.push_back(static_cast<uint32_t>(segment));
        seen[segment] |= bit;

        // bw_c_bar[b] bounds every label in b and in its Phi closure.
        if (base_cost + bw_c_bar[b] >= limit) continue;

        const auto &bucket = bw_buckets[b];
        for (const Label *backward : bucket.get_sorted_labels()) {
            if (base_cost + backward->cost >= limit) break; // cost-sorted tier
            if (examine(backward)) break;
        }
        if (found) break;
        for (const Label *backward : bucket.get_extra_labels()) {
            if (examine(backward)) break;
        }
        if (found) break;

        for (int predecessor : Phi_bw[b]) {
            if (!(seen[static_cast<size_t>(predecessor) >> 6] & (1ULL << (predecessor & 63)))) {
                stack.push_back(predecessor);
            }
        }
    }

    for (uint32_t segment : touched) seen[segment] = 0;
    return found;
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
    if constexpr (state == Stage::Four || state == Stage::Enumerate) {
        prepare_src_pricing_state();
    } else {
        pricing_src_compensation_bound = 0.0;
    }

    if constexpr (state == Stage::Four || state == Stage::Enumerate) {
        ++completion_exact_calls;
        bool use_backward_first = false;
#if defined(BALDES_COMPLETION_SCHEDULE_BACKWARD_FIRST)
        use_backward_first = true;
#elif defined(BALDES_COMPLETION_SCHEDULE_PARALLEL)
        use_backward_first = false;
#else
        // AUTO starts parallel, probes backward-first every eight exact calls,
        // and periodically rechecks parallel even after choosing sequential.
        const uint64_t phase = completion_exact_calls % 8;
        use_backward_first = auto_prefer_backward_first ? phase != 0 : phase == 4;
#endif

        if (use_backward_first) {
            completion_bound_attempts.store(0, std::memory_order_relaxed);
            completion_bound_rejections.store(0, std::memory_order_relaxed);
            const auto backward_start = std::chrono::steady_clock::now();
            forward_completion_bounds_ready = false;
            backward_cbar = labeling_algorithm<Direction::Backward, state, fullness>();
            const auto backward_end = std::chrono::steady_clock::now();
            bw_c_bar = backward_cbar;
            forward_completion_bounds_ready = true;
            forward_cbar = labeling_algorithm<Direction::Forward, state, fullness>();
            const auto forward_end = std::chrono::steady_clock::now();
            forward_completion_bounds_ready = false;

            const double backward_ms =
                std::chrono::duration<double, std::milli>(backward_end - backward_start).count();
            const double forward_ms =
                std::chrono::duration<double, std::milli>(forward_end - backward_end).count();
            last_pricing_timings.backward_labeling_ms = backward_ms;
            last_pricing_timings.forward_labeling_ms  = forward_ms;
            last_sequential_completion_ms = backward_ms + forward_ms;

#if defined(BALDES_COMPLETION_SCHEDULE_AUTO)
            const double attempts =
                static_cast<double>(completion_bound_attempts.load(std::memory_order_relaxed));
            const double rejected =
                static_cast<double>(completion_bound_rejections.load(std::memory_order_relaxed));
            const double rejection_ratio = attempts > 0.0 ? rejected / attempts : 0.0;
            const double estimated_unpruned_forward_ms = forward_ms / std::max(0.10, 1.0 - rejection_ratio);
            const double estimated_parallel_ms = std::max(backward_ms, estimated_unpruned_forward_ms);
            auto_prefer_backward_first = last_sequential_completion_ms < 0.95 * estimated_parallel_ms;
#endif
            return;
        }
    }

    forward_completion_bounds_ready = false;
    double forward_ms = 0.0;
    double backward_ms = 0.0;
    // Schedule the forward labeling algorithm task on the scheduler.
    auto forward_task = stdexec::schedule(bi_sched) | stdexec::then([&]() {
                            const auto start = std::chrono::steady_clock::now();
                            auto result = labeling_algorithm<Direction::Forward, state, fullness>();
                            forward_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                            return result;
                        });

    // Schedule the backward labeling algorithm task on the scheduler.
    auto backward_task = stdexec::schedule(bi_sched) | stdexec::then([&]() {
                             const auto start = std::chrono::steady_clock::now();
                             auto result = labeling_algorithm<Direction::Backward, state, fullness>();
                             backward_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                             return result;
                         });

    // Combine the forward and backward tasks.
    auto combined_work = stdexec::when_all(std::move(forward_task), std::move(backward_task)) |
                         stdexec::then([&](auto forward_result, auto backward_result) {
                             // Store the results into the provided vectors.
                             forward_cbar  = std::move(forward_result);
                             backward_cbar = std::move(backward_result);
                         });

    // Wait for all tasks to complete.
    stdexec::sync_wait(std::move(combined_work));
    last_pricing_timings.forward_labeling_ms  = forward_ms;
    last_pricing_timings.backward_labeling_ms = backward_ms;
    last_parallel_completion_ms = std::max(forward_ms, backward_ms);
#if defined(BALDES_COMPLETION_SCHEDULE_AUTO)
    if constexpr (state == Stage::Four || state == Stage::Enumerate) {
        if (completion_exact_calls % 8 == 0 && std::isfinite(last_sequential_completion_ms)) {
            auto_prefer_backward_first = last_sequential_completion_ms < 0.95 * last_parallel_completion_ms;
        }
    }
#endif
}

/**
 * Computes a new label based on the given labels L and L_prime.
 *
 */
template <Stage S>
Label *BucketGraph::compute_label(const Label *L, const Label *L_prime, double red_cost) {
    // L_prime == nullptr: L is a complete forward route (it reached the end
    // depot without crossing the half-way point).
    double      cij_cost       = L_prime ? getcij(L->node_id, L_prime->node_id) : 0.0;
    double      real_cost      = L->real_cost + (L_prime ? L_prime->real_cost + cij_cost : 0.0);
    std::vector<uint16_t> forward_route;
    std::vector<uint16_t> backward_route;
    L->materializeRoute(forward_route);
    if (L_prime) L_prime->materializeRoute(backward_route);

    // Acquire a new label from the pool and initialize its cost fields.
    auto new_label       = label_pool_fw->acquire();
    new_label->cost      = red_cost;
    new_label->real_cost = real_cost;
    new_label->parent    = nullptr;
    new_label->path_len  = L->path_len + (L_prime ? L_prime->path_len : 0);

    auto &merged_route = new_label->mutableRoute();
    merged_route.reserve(forward_route.size() + backward_route.size());
    merged_route.insert(merged_route.end(), forward_route.begin(), forward_route.end());
    merged_route.insert(merged_route.end(), backward_route.rbegin(), backward_route.rend());

    return new_label;
}
