/**
 * @file BucketPricing.h
 * @brief Pricing-stage state machine for bucket graph labeling.
 */

#pragma once

#include "pricing/BucketGraph.h"

inline BucketGraph::BucketStageDecision BucketGraph::current_bucket_pricing_stage() const noexcept {
    if (depth != 0) return {};
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
        print_info("Starting enumeration with gap {}\n", gap);
        enumeration_failed.store(false, std::memory_order_relaxed);
        auto labels = bi_labeling_algorithm<Stage::Enumerate, SYM>();
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

template <Symmetry SYM>
inline std::vector<Label *> BucketGraph::solve(bool trigger) {
    status = Status::NotOptimal;
    if (trigger) {
        transition = true;
        fixed      = false;
    }

    updateSplit();
    const auto decision = current_bucket_pricing_stage();
    if (!decision.available) {
        ++iter;
        return {};
    }

    activate_bucket_pricing_stage(decision.stage);
    auto result = execute_bucket_pricing_stage<SYM>(decision.stage);
    transition_after_bucket_pricing(decision.stage, result);
    ++iter;
    return result.labels;
}
