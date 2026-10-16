/**
 * @file PhaseConfiguration.h
 * @brief Phase configuration for RCSP pricing phases.
 *
 * Each phase has three fields:
 *   - byte0: check NG and cut dominance (bool)
 *   - byte1: allow jump arcs (bool)
 *   - signed long: label limit per bucket (not column count)
 *
 * Derived byte2 = first && second && limit == LONG_MAX (exact flag).
 * The selector at solver offset 0x44 (UserControlParameters RCSPheurLabelingStrategy)
 * selects the phase vector. Default (0 or >5):
 *   Phase 1: (true, true, LONG_MAX)  — exact, NG+cut dom, jump arcs, unlimited
 *   Phase 2: (false, false, LONG_MAX) — heuristic, no dom, no jump, unlimited
 *   Phase 3: (false, false, 1)        — one-label augmented-score heuristic
 */

#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace baldes {

struct PhaseConfiguration {
    bool check_ng_and_cuts = true;
    bool allow_jump_arcs = true;
    std::int64_t labels_per_bucket = std::numeric_limits<std::int64_t>::max();

    bool exact() const noexcept {
        return check_ng_and_cuts && allow_jump_arcs &&
               labels_per_bucket == std::numeric_limits<std::int64_t>::max();
    }
    void validate() const {
        if (labels_per_bucket < 1) throw std::invalid_argument("positive phase label limit required");
    }
};

/**
 * @brief Initialize phase configuration from binary selector tables.
 *
 * If supplied is non-empty, validates and returns it directly (custom config).
 * Otherwise, dispatches on the heuristic labeling strategy selector:
 *   0 or >5: standard 3-phase vector
 *   1: (true,true,LONG_MAX), (true,false,4), (false,false,1)
 *   2: (true,false,16), (true,false,4), (false,false,1)
 *   3: (true,false,8), (true,false,4), (false,false,1)
 *   4: (true,true,LONG_MAX), (false,false,8), (false,false,1)
 *   5: (true,true,LONG_MAX), (true,false,LONG_MAX), (false,false,8)
 */
inline std::vector<PhaseConfiguration> initialize_phase_configurations(
    int selector, const std::vector<PhaseConfiguration> &supplied = {}) {
    if (!supplied.empty()) {
        for (const auto &phase : supplied) phase.validate();
        return supplied;
    }
    constexpr auto unlimited = std::numeric_limits<std::int64_t>::max();
    switch (selector) {
    case 1: return {{true, true, unlimited}, {true, false, 4}, {false, false, 1}};
    case 2: return {{true, false, 16}, {true, false, 4}, {false, false, 1}};
    case 3: return {{true, false, 8}, {true, false, 4}, {false, false, 1}};
    case 4: return {{true, true, unlimited}, {false, false, 8}, {false, false, 1}};
    case 5: return {{true, true, unlimited}, {true, false, unlimited}, {false, false, 8}};
    default: return {{true, true, unlimited}, {false, false, unlimited}, {false, false, 1}};
    }
}

} // namespace baldes
