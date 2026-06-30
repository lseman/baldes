/**
 * @file BucketPricingPass.h
 * @brief Bi-directional pricing pass orchestration.
 */

#pragma once

#include "BucketConcat.h"
#include "BucketGraph.h"

template <Stage S, Symmetry SYM>
std::vector<Label *> BucketGraph::bi_labeling_algorithm() {
    prepare_pricing_stage<S>();
    initialize_pricing_pass<S>();

    BucketPricingPass pass(fw_buckets.size(), bw_buckets.size());
    run_directional_pricing<S, SYM>(pass.bounds);

    merged_labels.push_back(make_initial_merged_label<S, SYM>());
    concatenate_pricing_pass<S, SYM>(pass);

    return finalize_pricing_pass<S>().labels;
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
        bounds.forward = labeling_algorithm<Direction::Forward, S, Full::Partial>();
    }
}

template <Stage S, Symmetry SYM>
Label *BucketGraph::make_initial_merged_label() {
    auto best_label = label_pool_fw->acquire();
    if constexpr (SYM == Symmetry::Asymmetric) {
        if (fw_best_label->node_id != bw_best_label->node_id &&
            !visited_overlap(fw_best_label->visited_bitmap, bw_best_label->visited_bitmap) &&
            check_feasibility(fw_best_label, bw_best_label)) {
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
    return best_label;
}

template <Stage S, Symmetry SYM>
void BucketGraph::concatenate_from_forward_arc(const Label *label, const BucketArc &arc, BucketPricingPass &pass) {
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
        const auto &jump_bucket = fw_buckets[arc.to_bucket];
        for (size_t r = 0; r < options.resources.size(); ++r) {
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

    int bucket_to_process = extended_bucket;
    concatenate_label_from_bucket<S, SYM>(label, bucket_to_process, pass.best_cost, &splice_state);
}

template <Stage S, Symmetry SYM>
void BucketGraph::concatenate_pricing_pass(BucketPricingPass &pass) {
    concatenation_labels_tested.store(0, std::memory_order_relaxed);
    concatenation_labels_accepted.store(0, std::memory_order_relaxed);

    const size_t chunk_size = std::max<size_t>(1, fw_buckets_size / MERGE_SCHED_CONCURRENCY);
    const size_t n_chunks   = (fw_buckets_size + chunk_size - 1) / chunk_size;

    auto bulk_sender = stdexec::bulk(
        stdexec::just(), n_chunks,
        [this, chunk_size, &pass](std::size_t chunk_idx) {
            const size_t start_bucket = chunk_idx * chunk_size;
            const size_t end_bucket   = std::min(start_bucket + chunk_size, static_cast<size_t>(fw_buckets_size));

            for (size_t bucket = start_bucket; bucket < end_bucket; ++bucket) {
                const auto &bucket_labels = fw_buckets[bucket].get_labels();
                for (const Label *L : bucket_labels) {
                    if (L->is_dominated) continue;

                    pass.non_dominated_labels.fetch_add(1, std::memory_order_relaxed);

                    const auto &to_arcs = fw_buckets[bucket].template get_bucket_arcs<Direction::Forward>();
                    for (const auto &arc : to_arcs) { concatenate_from_forward_arc<S, SYM>(L, arc, pass); }
                }
            }
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
            if (merged_labels[i]->nodes_covered.size() <= 3) continue;
            top_labels.push_back(merged_labels[i]);
        }
        ils->submit_task(top_labels, nodes);
    }
#endif

    inner_obj = merged_labels[0]->cost;
    return make_bucket_pricing_result(merged_labels);
}
