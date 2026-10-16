#pragma once

#include <cmath>
#include <cstdint>
#include <string>

#include "bnb/bcp/MasterSolution.h"
#include "core/Logger.h"

namespace baldes::bcp {

inline void printCgProgress(int iteration, double lp_objective, double pricing_objective, int cut_count,
                            int rcc_cut_count, int columns_added, int stage, double lagrangian_gap,
                            double current_alpha, double trust_region_value, double gap, double integer_solution,
                            double bucket_graph_threshold, std::uint64_t concatenations_tested = 0,
                            std::uint64_t concatenations_accepted = 0) {
    constexpr int threshold = 1'000'000;
    const std::string lagrangian_gap_text =
        lagrangian_gap > threshold ? "∞" : fmt::format("{:10.4f}", lagrangian_gap);

    std::string integer_solution_text;
    if (integer_solution > threshold) {
        integer_solution_text = "∞";
    } else if (std::abs(integer_solution - std::round(integer_solution)) <= kVarIntegralityTol) {
        integer_solution_text = fmt::format("{:.0f}", integer_solution);
    } else {
        integer_solution_text = fmt::format("{:.6f}", integer_solution);
    }

    fmt::print("| It.: {:4} | Obj.: {:8.2f} | Price: {:9.2f} | SRC: {:3} "
               "| RCC: {:3} | Paths: {:3} | Stage: {:1} | Cat: {:5}/{:<5} | "
               "Lag.: {:>10} | α: {:4.2f} | tr: {:2.2f} | gap: {:2.4f} "
               "| Int.: {:>4} | Th: {:4.2f} |\n",
               iteration, lp_objective, pricing_objective, cut_count, rcc_cut_count, columns_added, stage,
               concatenations_accepted, concatenations_tested, lagrangian_gap_text, current_alpha,
               trust_region_value, gap, integer_solution_text, bucket_graph_threshold);

    Logger::log("| It.: {:4} | Obj.: {:8.2f} | Price: {:9.2f} | SRC: {:3} "
                "| RCC: {:3} | Paths: {:3} | Stage: {:1} | Cat: {:5}/{:<5} | "
                "Lag.: {:10.4f} | α: {:4.2f} | tr: {:2.2f} | gap: {:2.4f} |\n",
                iteration, lp_objective, pricing_objective, cut_count, rcc_cut_count, columns_added, stage,
                concatenations_accepted, concatenations_tested, lagrangian_gap, current_alpha, trust_region_value,
                gap);
}

} // namespace baldes::bcp
