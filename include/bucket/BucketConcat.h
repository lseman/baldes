/**
 * @file BucketConcat.h
 * @brief Forward/backward label concatenation for bucket pricing.
 */

#pragma once

#include "BucketGraph.h"
#include "utils/NumericUtils.h"

template <Stage S>
void BucketGraph::publish_concatenation_candidates(const Label *forward_label,
                                                   std::span<const ConcatenationCandidate> candidates,
                                                   std::atomic<double> &best_cost) {
    if (candidates.empty()) return;

    std::lock_guard<std::mutex> lock(merge_mutex);

    double current_best = best_cost.load(std::memory_order_relaxed);
    for (const auto &candidate : candidates) {
        const Label *candidate_label = candidate.label;
        const double candidate_cost  = candidate.cost;

        if constexpr (S == Stage::Enumerate) {
            const size_t route_limit = static_cast<size_t>(std::max(1, enumeration_policy.max_routes)) + 1;
            const double cutoff      = std::min(gap, enumeration_route_cutoff.load(std::memory_order_relaxed));
            if (!numericutils::lt(candidate_cost, cutoff)) continue;

            if (numericutils::lt(candidate_cost, current_best)) {
                best_cost.store(candidate_cost, std::memory_order_relaxed);
                current_best = candidate_cost;
            }

            if (merged_labels.size() < route_limit) {
                auto pbest = compute_label<S>(forward_label, candidate_label, candidate_cost);
                merged_labels.push_back(pbest);
                std::push_heap(merged_labels.begin(), merged_labels.end(),
                               [](const Label *a, const Label *b) { return a->cost < b->cost; });
            } else {
                if (merged_labels.empty() || !numericutils::lt(candidate_cost, merged_labels.front()->cost)) continue;

                auto pbest = compute_label<S>(forward_label, candidate_label, candidate_cost);
                std::pop_heap(merged_labels.begin(), merged_labels.end(),
                              [](const Label *a, const Label *b) { return a->cost < b->cost; });
                merged_labels.back() = pbest;
                std::push_heap(merged_labels.begin(), merged_labels.end(),
                               [](const Label *a, const Label *b) { return a->cost < b->cost; });
            }

            if (merged_labels.size() >= route_limit && !merged_labels.empty()) {
                enumeration_route_cutoff.store(merged_labels.front()->cost, std::memory_order_relaxed);
            }
        } else {
            if (numericutils::lt(candidate_cost, current_best)) {
                best_cost.store(candidate_cost, std::memory_order_relaxed);
                current_best = candidate_cost;
            }

            if (numericutils::lte(candidate_cost, current_best)) {
                auto pbest = compute_label<S>(forward_label, candidate_label, candidate_cost);
                merged_labels.push_back(pbest);
            }
        }
    }
}

inline void BucketGraph::push_unvisited_phi_neighbors(std::span<const int> neighbors,
                                                      ConcatenationScratch &scratch) const {
    if (scratch.bucket_stack.capacity() < scratch.bucket_stack.size() + neighbors.size()) {
        scratch.bucket_stack.reserve(scratch.bucket_stack.size() + neighbors.size());
    }
    for (int b_prime : neighbors) {
        const size_t   segment = static_cast<size_t>(b_prime) >> 6;
        const uint64_t mask    = bit_mask_lookup[b_prime & 63];
        if (!(scratch.visited_buckets[segment] & mask)) { scratch.bucket_stack.push_back(b_prime); }
    }
}

template <Stage S>
bool BucketGraph::collect_concatenation_candidate(const Label *forward_label, const Label *backward_label,
                                                  double path_cost, const SpliceState *splice_state,
                                                  [[maybe_unused]] std::span<const ActiveCutInfo> active_cuts,
                                                  std::atomic<double> &best_cost, ConcatenationScratch &scratch) {
    ++scratch.stats.labels_tested;

    const int forward_node_id = forward_label->node_id;
    if (backward_label == nullptr || backward_label->is_dominated || backward_label->node_id == forward_node_id) {
        return false;
    }

    double total_cost = path_cost + backward_label->cost;
    if constexpr (!(S == Stage::Four || S == Stage::Enumerate)) {
        if (!numericutils::lt(total_cost, best_cost.load(std::memory_order_relaxed))) { return false; }
    }

    if (visited_overlap(forward_label->visited_bitmap, backward_label->visited_bitmap)) { return false; }
    if (!check_feasibility(forward_label, backward_label, splice_state)) { return false; }

#if defined(SRC)
    if constexpr (S == Stage::Four || S == Stage::Enumerate) {
        for (const auto &active_cut : active_cuts) {
            const auto  &cut  = *active_cut.cut_ptr;
            const size_t idx  = active_cut.index;
            const double dual = active_cut.dual_value;
#if defined(SRC_MEMORY_MODE_ARC)
            if (!cut.isSRCMemoryArc(forward_label->node_id, backward_label->node_id)) continue;
#endif
            const bool cut_hit = splice_state && splice_state->has_src
                                     ? splice_state->SRCmap[idx] + backward_label->SRCmap[idx] >= cut.p.den
                                     : forward_label->SRCmap[idx] + backward_label->SRCmap[idx] >= cut.p.den;
            if (cut_hit) { total_cost -= dual; }
        }
    }
#endif

    bool cost_acceptable;
    if constexpr (S != Stage::Enumerate) {
        cost_acceptable = numericutils::lt(total_cost, best_cost.load(std::memory_order_relaxed));
    } else {
        const double cutoff = std::min(gap, enumeration_route_cutoff.load(std::memory_order_relaxed));
        cost_acceptable     = numericutils::lt(total_cost, cutoff);
    }

    if (!cost_acceptable) { return false; }
    scratch.candidates.push_back({backward_label, total_cost});
    ++scratch.stats.labels_accepted;
    return true;
}

template <Stage S, Symmetry SYM>
void BucketGraph::concatenate_label_from_bucket(const Label *L, int b, std::atomic<double> &best_cost,
                                                const SpliceState *splice_state) {
    const size_t n_segments = (fw_buckets_size + 63) / 64;
    static thread_local ConcatenationScratch scratch;
    scratch.reset(n_segments, b);

    auto &other_buckets = assign_symmetry<SYM>(fw_buckets, bw_buckets);
    auto &other_c_bar   = assign_symmetry<SYM>(fw_c_bar, bw_c_bar);
    auto &other_phi     = assign_symmetry<SYM>(Phi_fw, Phi_bw);

    const int    L_node_id     = L->node_id;
    const double L_cost        = L->cost;
    const bool   has_branching = !branching_duals->empty();

    std::span<const ActiveCutInfo> active_cuts;
#if defined(SRC)
    if constexpr (S > Stage::Three) { active_cuts = cut_storage->getActiveCuts(); }
#endif

    while (!scratch.bucket_stack.empty()) {
        const int current_bucket = scratch.bucket_stack.back();
        scratch.bucket_stack.pop_back();

        const size_t   segment  = current_bucket >> 6;
        const uint64_t bit_mask = bit_mask_lookup[current_bucket & 63];
        scratch.visited_buckets[segment] |= bit_mask;

        const int bucketLprimenode = other_buckets[current_bucket].node_id;
        double    travel_cost      = getcij(L_node_id, bucketLprimenode);

#if defined(RCC) || defined(EXACT_RCC)
        if constexpr (S == Stage::Four) { travel_cost -= arc_duals.getDual(L_node_id, bucketLprimenode); }
#endif

        if (has_branching) { travel_cost -= branching_duals->getDual(L_node_id, bucketLprimenode); }

        const double path_cost = L_cost + travel_cost + (splice_state ? splice_state->cost_delta : 0.0);
        const double bound     = other_c_bar[current_bucket];

        double prune_limit = best_cost.load(std::memory_order_relaxed);
        if constexpr (S == Stage::Enumerate) {
            prune_limit = std::min(gap, enumeration_route_cutoff.load(std::memory_order_relaxed));
        }

        if ((S != Stage::Enumerate && numericutils::gte(path_cost + bound, prune_limit)) ||
            (S == Stage::Enumerate && numericutils::gte(path_cost + bound, prune_limit))) {
            continue;
        }

        const auto &bucket       = other_buckets[current_bucket];
        const auto &labels       = bucket.get_sorted_labels();
        const auto &extra_labels = bucket.get_extra_labels();
        const auto  label_count  = labels.size() + extra_labels.size();
        if (label_count == 0) continue;
        scratch.prepare_candidates(label_count);

        for (const Label *L_bw : labels) {
            collect_concatenation_candidate<S>(L, L_bw, path_cost, splice_state, active_cuts, best_cost, scratch);
        }
        for (const Label *L_bw : extra_labels) {
            collect_concatenation_candidate<S>(L, L_bw, path_cost, splice_state, active_cuts, best_cost, scratch);
        }

        publish_concatenation_candidates<S>(L, scratch.candidates, best_cost);
        push_unvisited_phi_neighbors(other_phi[current_bucket], scratch);
    }

    concatenation_labels_tested.fetch_add(scratch.stats.labels_tested, std::memory_order_relaxed);
    concatenation_labels_accepted.fetch_add(scratch.stats.labels_accepted, std::memory_order_relaxed);
}
