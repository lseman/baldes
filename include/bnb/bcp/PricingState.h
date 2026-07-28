#pragma once

#include <algorithm>

#include "bnb/bcp/MasterSolution.h"
#include "core/Definitions.h"
#include "core/Logger.h"
#include "pricing/bucket_graph/BucketGraph.h"

namespace baldes::bcp {

struct PricingState {
    bool enumerate            = false;
    int  exact_pricing_passes = 0;
    int  hgs_interval         = 5;
    int  failed_enumerations  = 0;
    int  retry_enumeration_at = 0;

    void recordPricingStage(int stage) noexcept {
        if (stage == 4) ++exact_pricing_passes;
    }

    bool handleEnumerationFailure(BucketGraph *bucket_graph) {
        if (!bucket_graph->enumerationFailed()) return false;
        enumerate = false;
        ++failed_enumerations;
        retry_enumeration_at = exact_pricing_passes + std::clamp(failed_enumerations, 1, 10);
        bucket_graph->clearEnumerationFailure();
        print_info("Enumeration cap reached; falling back to exact pricing\n");
        return true;
    }

    [[nodiscard]] static double lowerBound(double lp_objective, int route_count, double pricing_objective) noexcept {
        return lp_objective + route_count * std::min(0.0, pricing_objective);
    }

    [[nodiscard]] static bool cutsStable(const BucketGraph *bucket_graph, bool non_violated_cuts,
                                         int non_violated_count) noexcept {
        return non_violated_cuts && bucket_graph->A_MAX == N_SIZE &&
               non_violated_count >= bucket_graph->enumeration_policy.min_stable_cut_passes;
    }

    bool maybeStartEnumeration(BucketGraph *bucket_graph, double pricing_lower_bound, bool cuts_stable,
                               bool no_negative_pricing, double pricing_objective) {
        if (enumerate || exact_pricing_passes < retry_enumeration_at) return false;
        if (!bucket_graph->shouldAttemptEnumeration(pricing_lower_bound, cuts_stable, exact_pricing_passes,
                                                    no_negative_pricing)) {
            return false;
        }

        const double relative_gap = bucket_graph->relative_gap_to_incumbent(pricing_lower_bound);
        print_info("{} and relative gap {:.4f}; trying enumeration (pricing {:.6g})\n",
                   no_negative_pricing ? "No negative pricing/cuts" : "Stable cuts/small gap", relative_gap,
                   pricing_objective);
        bucket_graph->enableEnumeration(pricing_lower_bound);
        enumerate = true;
        return true;
    }

    [[nodiscard]] bool shouldForceCuts(int columns_added, double pricing_objective, int stage) const noexcept {
        return stage == 4 && columns_added == 0 && !hasNegativeReducedCost(pricing_objective);
    }

    [[nodiscard]] bool shouldRunHGS(int iteration, int stage, int columns_added) const noexcept {
        if (enumerate) return false;
        if (columns_added == 0) return true;
        return stage >= 4 && iteration % hgs_interval == 0;
    }
};

} // namespace baldes::bcp
