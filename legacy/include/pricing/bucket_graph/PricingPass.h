/**
 * @file BucketPricingPass.h
 * @brief Bi-directional pricing pass orchestration.
 */

#pragma once

#include "pricing/bucket_graph/Concatenation.h"
#include "pricing/bucket_graph/BucketGraph.h"

template <Stage S, Symmetry SYM>
std::vector<Label *> BucketGraph::bi_labeling_algorithm() {
    using Clock = std::chrono::steady_clock;
    last_pricing_timings = {};

    auto phase_start = Clock::now();
    prepare_pricing_stage<S>();
    last_pricing_timings.preparation_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - phase_start).count();

    phase_start = Clock::now();
    initialize_pricing_pass<S>();
    last_pricing_timings.initialization_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - phase_start).count();

    BucketPricingPass pass(fw_buckets.size(), bw_buckets.size());
    phase_start = Clock::now();
    run_directional_pricing<S, SYM>(pass.bounds);
    last_pricing_timings.directional_wall_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - phase_start).count();

    merged_labels.push_back(make_initial_merged_label<S, SYM>());
    phase_start = Clock::now();
    concatenate_pricing_pass<S, SYM>(pass);
    last_pricing_timings.concatenation_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - phase_start).count();

    phase_start = Clock::now();
    auto result = finalize_pricing_pass<S>();
    last_pricing_timings.finalization_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - phase_start).count();
    return result.labels;
}

template <Stage S>
void BucketGraph::prepare_pricing_stage() {
    if constexpr (S == Stage::Three) {
        heuristic_fixing<S>();
    } else if constexpr (S == Stage::Four) {
        if (first_reset) {
            reset_fixed();
            first_reset = false;
        }
#ifdef FIX_BUCKETS
        if (options.bucket_fixing) { bucket_fixing<S>(); }
#endif
    }

    if (options.warm_start && !just_fixed) {
        capture_warm_start_labels<Direction::Forward>();
        capture_warm_start_labels<Direction::Backward>();
    }
}

template <Stage S>
void BucketGraph::initialize_pricing_pass() {
    reset_pool();
    common_initialization();
    if constexpr (S == Stage::Enumerate) {
        merged_labels.reserve(static_cast<size_t>(enumeration_policy.max_routes) + 1);
    }
}

template <Stage S, Symmetry SYM>
void BucketGraph::run_directional_pricing(BucketDirectionalBounds &bounds) {
    if constexpr (SYM == Symmetry::Asymmetric) {
        run_labeling_algorithms<S, Full::Partial>(bounds.forward, bounds.backward);
    } else {
        // SYMMETRIC: forward labeling only, backward prefixes from forward labels.
        // The binary skips backward labeling when the graph is
        // symmetric (unique opposite arcs, identical pure costs & variable coeffs).
        // Backward prefixes are formed from forward labels using half-dual arc costs.
        if constexpr (S == Stage::Four || S == Stage::Enumerate) {
            prepare_src_pricing_state();
        } else {
            pricing_src_compensation_bound = 0.0;
        }
        const auto start = std::chrono::steady_clock::now();
        bounds.forward = labeling_algorithm<Direction::Forward, S, Full::Partial>();
        last_pricing_timings.forward_labeling_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

        // Build backward prefixes from forward labels using half-dual costs.
        // Each forward label at vertex v becomes a backward prefix for all
        // opposite arcs (v -> u) with cost = reduced_cost + symmetric_arc_cost(c_uv, dual[v], dual[u]).
        const size_t n_nodes = nodes.size();
        node_duals.assign(n_nodes, 0.0);
        if constexpr (S == Stage::Four || S == Stage::Enumerate) {
            // Extract node duals from SRC cut duals and branching duals.
            // For the symmetric half-dual cost, we need per-node dual values.
            // Branching duals provide node-level duals; cut duals are per-cut.
            if (branching_duals && !branching_duals->empty()) {
                for (size_t n = 0; n < n_nodes; ++n) {
                    node_duals[n] = branching_duals->getDual(n);
                }
            }
            // Otherwise node_duals stays zero (no branching duals available).
        }

        // Store forward labels as backward prefixes.
        for (size_t bucket = 0; bucket < fw_buckets.size(); ++bucket) {
            const auto &labels = fw_buckets[bucket].get_sorted_labels();
            for (const Label *L : labels) {
                if (L->is_dominated || !L->is_extended) continue;
                auto *bw_label = label_pool_bw->acquire();
                *bw_label = *L;
                const int head = L->node_id;
                if (head >= 0 && static_cast<size_t>(head) < n_nodes) {
                    bw_label->cost += node_duals[head];
                }
                bw_label->is_extended = false;
                bw_label->vertex = static_cast<int>(bucket);
                fw_buckets[bucket].add_extra_label(bw_label);
            }
        }

        // Bounds for backward direction derived from forward labels.
        bounds.backward.assign(bw_buckets.size(), std::numeric_limits<double>::infinity());
        for (size_t bucket = 0; bucket < fw_buckets.size(); ++bucket) {
            if (bucket < bw_buckets.size()) {
                const auto &labels = fw_buckets[bucket].get_sorted_labels();
                if (!labels.empty()) {
                    bounds.backward[bucket] = labels.front()->cost;
                }
            }
        }
        last_pricing_timings.backward_labeling_ms = 0.0;
    }
}

template <Stage S, Symmetry SYM>
Label *BucketGraph::make_initial_merged_label() {
    auto label       = label_pool_fw->acquire();
    label->cost      = 0.0;
    label->real_cost = std::numeric_limits<double>::infinity();
    label->clearRoute();
    return label;
}

template <Stage S, Symmetry SYM>
void BucketGraph::concatenate_from_forward_arc(const Label *label, const BucketArc &arc, BucketPricingPass &pass,
                                               std::vector<ConcatenationCandidate> &chunk_candidates) {
    const int to_node = arc.jump ? arc.jump_to_node : fw_buckets[arc.to_bucket].node_id;
    if (to_node < 0) return;
    if constexpr (S >= Stage::Three || S == Stage::Eliminate) {
        if (is_arc_fixed(label->node_id, to_node)) return;
    }

    auto extended_bucket =
        Extend<Direction::Forward, S, ArcType::Bucket, Mutability::Const, Full::Reverse>(label, arc);
    if (extended_bucket == -1) return;

    SpliceState splice_state;
    splice_state.resources = label->resources;

    std::array<double, R_SIZE> base_resources = label->resources;
    if (arc.jump) {
        // Buckets only bound the main (leading) resources.
        const auto  &jump_bucket = fw_buckets[arc.to_bucket];
        const size_t n_bucketed  = std::min(options.resources.size(), jump_bucket.lb.size());
        for (size_t r = 0; r < n_bucketed; ++r) {
            base_resources[r] = std::max(base_resources[r], jump_bucket.lb[r]);
        }
    }

    std::vector<double> splice_resources(options.resources.size());
    if (!process_all_resources<Direction::Forward>(splice_resources, base_resources, arc, nodes[to_node],
                                                   options.resources.size())) {
        return;
    }

    for (size_t r = 0; r < options.resources.size(); ++r) { splice_state.resources[r] = splice_resources[r]; }

#if defined(SRC)
    if constexpr (S == Stage::Four || S == Stage::Enumerate) {
        if (cut_storage && cut_storage->activeSize() > 0) {
            splice_state.SRCmap  = label->SRCmap;
            splice_state.has_src = true;
        }
    }
#endif

    // SYMMETRIC: backward prefixes stored as extra labels in forward buckets.
    // Use the forward bucket number directly (no coordinate conversion).
    // ASYMMETRIC: convert to backward bucket numbering (opposite physical layout).
    int bucket_to_process = extended_bucket;
    if constexpr (SYM == Symmetry::Asymmetric) {
        bucket_to_process = get_bucket_number<Direction::Backward>(to_node, splice_resources);
    }
    concatenate_label_from_bucket<S, SYM>(label, bucket_to_process, pass.best_cost, chunk_candidates, &splice_state);
}

template <Stage S, Symmetry SYM>
void BucketGraph::concatenate_pricing_pass(BucketPricingPass &pass) {
    concatenation_labels_tested.store(0, std::memory_order_relaxed);
    concatenation_labels_accepted.store(0, std::memory_order_relaxed);

    // Only forward labels at or below the half-way point can start a
    // crossing arc (the Full::Reverse extension rejects arcs that end at or
    // below it); a label already past it lies on a route whose crossing arc
    // is earlier, so joining it again only duplicates that route. Only the
    // labels created by a crossing extension are joined.
    const int    main_res      = options.main_resources.empty() ? 0 : options.main_resources[0];
    const double half_way      = q_star.size() > static_cast<size_t>(main_res)
                                     ? std::floor(q_star[static_cast<size_t>(main_res)])
                                     : std::numeric_limits<double>::infinity();

    const size_t chunk_size = std::max<size_t>(1, fw_buckets_size / MERGE_SCHED_CONCURRENCY);
    const size_t n_chunks   = (fw_buckets_size + chunk_size - 1) / chunk_size;

    auto bulk_sender = stdexec::bulk(
        stdexec::just(), n_chunks,
        [this, chunk_size, &pass, main_res, half_way](std::size_t chunk_idx) {
            const size_t start_bucket = chunk_idx * chunk_size;
            const size_t end_bucket   = std::min(start_bucket + chunk_size, static_cast<size_t>(fw_buckets_size));
            std::vector<ConcatenationCandidate> chunk_candidates;
            chunk_candidates.reserve(64);

            for (size_t bucket = start_bucket; bucket < end_bucket; ++bucket) {
                if constexpr (S == Stage::Four) {
                    if (pricing_truncated.load(std::memory_order_relaxed)) break;
                }
                const auto &bucket_labels = fw_buckets[bucket].get_labels();
                for (const Label *L : bucket_labels) {
                    if constexpr (S == Stage::Four) {
                        if (pricing_truncated.load(std::memory_order_relaxed)) break;
                    }
                    if (L->is_dominated) continue;

                    pass.non_dominated_labels.fetch_add(1, std::memory_order_relaxed);
                    if (L->resources[static_cast<size_t>(main_res)] > half_way) continue;

                    // A forward label at the end depot at or below the
                    // half-way point is a complete route that never crossed
                    // it, so no join can produce it: a label that reaches the
                    // sink becomes a solution candidate directly.
                    if (L->node_id == options.end_depot) {
                        collect_complete_forward_route<S>(L, pass.best_cost, chunk_candidates);
                        continue;
                    }

                    const auto &to_arcs = fw_buckets[bucket].template get_bucket_arcs<Direction::Forward>();
                    for (const auto &arc : to_arcs) {
                        if constexpr (S == Stage::Four) {
                            if (pricing_truncated.load(std::memory_order_relaxed)) break;
                        }
                        concatenate_from_forward_arc<S, SYM>(L, arc, pass, chunk_candidates);
                    }
                }
            }

            publish_concatenation_candidates<S>(chunk_candidates, pass.best_cost);
        });

    auto work = stdexec::starts_on(merge_sched, bulk_sender);
    stdexec::sync_wait(std::move(work));
    last_concatenation_labels_tested   = concatenation_labels_tested.load(std::memory_order_relaxed);
    last_concatenation_labels_accepted = concatenation_labels_accepted.load(std::memory_order_relaxed);
}

template <Stage S>
BucketGraph::BucketPricingResult BucketGraph::finalize_pricing_pass() {
    pdqsort(merged_labels.begin(), merged_labels.end(),
            [](const Label *a, const Label *b) { return a->cost < b->cost; });

#ifdef SCHRODINGER
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
    if constexpr (S == Stage::Four) {
        std::vector<Label *> top_labels;
        top_labels.reserve(N_ADD);
        const int n_candidates = std::min(N_ADD, static_cast<int>(merged_labels.size()));
        for (int i = 0; i < n_candidates; ++i) {
            if (merged_labels[i]->getRoute().size() <= 3) continue;
            top_labels.push_back(merged_labels[i]);
        }
        ils->submit_task(top_labels, nodes);
    }
#endif

    inner_obj = merged_labels[0]->cost;
    return make_bucket_pricing_result(merged_labels);
}
