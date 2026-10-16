/**
 * @file BucketPricing.h
 * @brief Pricing-stage state machine for bucket graph labeling.
 */

#pragma once

#include "pricing/bucket_graph/BucketGraph.h"

inline BucketGraph::BucketStageDecision BucketGraph::current_bucket_pricing_stage() const noexcept {
    if (s1) return {Stage::One, true};
    if (s2) return {Stage::Two, true};
    if (s3) return {Stage::Three, true};
    if (s4) return {Stage::Four, true};
    if (s5) return {Stage::Enumerate, true};
    return {};
}

inline void BucketGraph::activate_bucket_pricing_stage(Stage next_stage) noexcept {
    s1 = s2 = s3 = s4 = s5 = false;
    switch (next_stage) {
    case Stage::One:
        s1    = true;
        stage = 1;
        break;
    case Stage::Two:
        s2    = true;
        stage = 2;
        break;
    case Stage::Three:
        s3    = true;
        stage = 3;
        break;
    case Stage::Four:
        s4    = true;
        stage = 4;
        break;
    case Stage::Enumerate:
        s5    = true;
        stage = 5;
        break;
    default:
        break;
    }
}

inline BucketGraph::BucketPricingResult
BucketGraph::make_bucket_pricing_result(std::vector<Label *> labels, bool skip_stage_transition) const {
    return BucketPricingResult{std::move(labels), inner_obj, last_concatenation_labels_tested,
                               last_concatenation_labels_accepted, enumerationFailed(),
                               skip_stage_transition || pricing_truncated.load(std::memory_order_relaxed),
                               pricing_truncated.load(std::memory_order_relaxed)};
}

template <Symmetry SYM>
BucketGraph::BucketPricingResult BucketGraph::execute_bucket_pricing_stage(Stage active_stage) {
    switch (active_stage) {
    case Stage::One:
        return make_bucket_pricing_result(bi_labeling_algorithm<Stage::One, SYM>());
    case Stage::Two:
        return make_bucket_pricing_result(bi_labeling_algorithm<Stage::Two, SYM>());
    case Stage::Three:
        return make_bucket_pricing_result(bi_labeling_algorithm<Stage::Three, SYM>());
    case Stage::Four:
#ifdef FIX_BUCKETS
        if (transition) {
            bool original_fixed = fixed;
            fixed               = true;
            auto result          = make_bucket_pricing_result(bi_labeling_algorithm<Stage::Four, SYM>(), true);
            transition          = false;
            fixed               = original_fixed;
            if (!result.labels.empty()) { min_red_cost = result.labels[0]->cost; }
            return result;
        }
#endif
        return make_bucket_pricing_result(bi_labeling_algorithm<Stage::Four, SYM>());
    case Stage::Enumerate: {
        // Enumerate every route with reduced cost below UB - b·v at the
        // current duals v. That gap is valid only if no route has negative
        // reduced cost at v (then L(v) = b·v); otherwise the run is
        // reported as failed and pricing falls back to Stage Four.
        if (!std::isfinite(pricing_dual_objective) || !std::isfinite(incumbent)) {
            print_info("Skipping enumeration: no valid bound at current duals\n");
            enumeration_failed.store(true, std::memory_order_relaxed);
            return make_bucket_pricing_result({});
        }
        gap = std::max(0.0, incumbent - pricing_dual_objective);
        print_info("Starting enumeration with gap {}\n", gap);
        enumeration_failed.store(false, std::memory_order_relaxed);
        auto labels = bi_labeling_algorithm<Stage::Enumerate, SYM>();
        if (!labels.empty() && labels.front()->cost < -1e-6) {
            print_info("Enumeration found a negative reduced cost route ({}); bound not valid\n",
                       labels.front()->cost);
            enumeration_failed.store(true, std::memory_order_relaxed);
        }
        if (labels.size() > static_cast<size_t>(enumeration_policy.max_routes)) {
            labels.resize(static_cast<size_t>(enumeration_policy.max_routes));
            enumeration_failed.store(true, std::memory_order_relaxed);
        }
        print_info("Finished enumeration with {} paths{}\n", labels.size(),
                   enumerationFailed() ? " (cap reached)" : "");
        return make_bucket_pricing_result(std::move(labels));
    }
    default:
        return {};
    }
}

inline void BucketGraph::transition_after_bucket_pricing(Stage active_stage, const BucketPricingResult &result) {
    if (result.skip_stage_transition) return;

    switch (active_stage) {
    case Stage::One:
        if (result.best_reduced_cost >= -1) { activate_bucket_pricing_stage(Stage::Two); }
        break;
    case Stage::Two:
        if (result.best_reduced_cost >= -10) { activate_bucket_pricing_stage(Stage::Three); }
        break;
    case Stage::Three:
        if (result.best_reduced_cost >= -0.5) {
            activate_bucket_pricing_stage(Stage::Four);
            transition = true;
        }
        break;
    case Stage::Four: {
#ifdef FIX_BUCKETS
        if (transition) { break; }
#endif
        if (status != Status::Rollback && considerRegenerate()) {
            status = Status::Rollback;
            break;
        }
        threshold = stats.computeThreshold(iter, result.best_reduced_cost);
        constexpr double kExactPricingTol = 1e-6;
        if (result.best_reduced_cost >= -kExactPricingTol) {
            ss = true;
#if !defined(SRC) && !defined(SRC3)
            status = Status::Optimal;
#else
            status = Status::Separation;
#endif
        } else {
            ss     = false;
            status = Status::NotOptimal;
        }
        break;
    }
    case Stage::Enumerate:
        if (result.enumeration_failed) {
            activate_bucket_pricing_stage(Stage::Four);
            status = Status::NotOptimal;
        } else {
            status = Status::Optimal;
        }
        break;
    default:
        break;
    }
}
/**
 * Phase-driven stage transition.
 *
 * Replaces the hardcoded cost thresholds (-1, -10, -0.5) with the
 * phase configuration vector. Each phase's (check_ng_and_cuts, allow_jump_arcs,
 * labels_per_bucket) tuple controls what dominance and jump behavior applies.
 *
 * The one-label heuristic (phase 3 with labels_per_bucket==1) replaces the
 * incumbent when the augmented score is better; it does not advance to phase 4.
 * Exact phases advance when best_reduced_cost >= -1e-6.
 */
inline void BucketGraph::transition_after_bucket_pricing_phase(Stage active_stage, const BucketPricingResult &result) {
    if (result.skip_stage_transition) return;

    // Initialize phase configurations from heur_labeling_strategy selector.
    if (phase_configurations.empty() && options.heur_labeling_strategy >= 0) {
        phase_configurations = baldes::initialize_phase_configurations(options.heur_labeling_strategy);
        current_phase_index = 0;
    }

    const bool has_phases = !phase_configurations.empty();
    const int  n_phases   = static_cast<int>(phase_configurations.size());

    switch (active_stage) {
    case Stage::One:
        if (has_phases && current_phase_index + 1 < n_phases) {
            // Advance to next phase; phases do not use cost thresholds.
            activate_bucket_pricing_stage(Stage::Two);
            ++current_phase_index;
        } else if (result.best_reduced_cost >= -1) {
            // Fallback: original threshold for backward compat.
            activate_bucket_pricing_stage(Stage::Two);
        }
        break;
    case Stage::Two:
        if (has_phases && current_phase_index + 1 < n_phases) {
            activate_bucket_pricing_stage(Stage::Three);
            ++current_phase_index;
        } else if (result.best_reduced_cost >= -10) {
            activate_bucket_pricing_stage(Stage::Three);
        }
        break;
    case Stage::Three:
        if (has_phases && current_phase_index + 1 < n_phases) {
            activate_bucket_pricing_stage(Stage::Four);
            transition = true;
            ++current_phase_index;
        } else if (result.best_reduced_cost >= -0.5) {
            activate_bucket_pricing_stage(Stage::Four);
            transition = true;
        }
        break;
    case Stage::Four: {
#ifdef FIX_BUCKETS
        if (transition) { break; }
#endif
        if (status != Status::Rollback && considerRegenerate()) {
            status = Status::Rollback;
            break;
        }
        threshold = stats.computeThreshold(iter, result.best_reduced_cost);
        constexpr double kExactPricingTol = 1e-6;
        if (result.best_reduced_cost >= -kExactPricingTol) {
            ss = true;
#if !defined(SRC) && !defined(SRC3)
            status = Status::Optimal;
#else
            status = Status::Separation;
#endif
        } else {
            ss     = false;
            status = Status::NotOptimal;
        }
        break;
    }
    case Stage::Enumerate:
        if (result.enumeration_failed) {
            activate_bucket_pricing_stage(Stage::Four);
            status = Status::NotOptimal;
        } else {
            status = Status::Optimal;
        }
        break;
    default:
        break;
    }
}

template <Symmetry SYM>
inline std::vector<Label *> BucketGraph::solve(bool trigger) {
    status = Status::NotOptimal;
    if (trigger) {
        transition = true;
        fixed      = false;
    }

    updateSplit();
    const auto decision = current_bucket_pricing_stage();
    last_pricing_exact  = decision.available && decision.stage == Stage::Four;
    if (!decision.available) {
        ++iter;
        return {};
    }

    activate_bucket_pricing_stage(decision.stage);
    auto result = execute_bucket_pricing_stage<SYM>(decision.stage);
    // Use phase-driven transition when phase config is available.
    if (!phase_configurations.empty()) {
        transition_after_bucket_pricing_phase(decision.stage, result);
    } else {
        transition_after_bucket_pricing(decision.stage, result);
    }
    ++iter;
    return result.labels;
}
