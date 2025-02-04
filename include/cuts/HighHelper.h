#pragma once

using yzzLong = Bitset<N_SIZE>;
using cutLong = yzzLong;

struct PlanValidationResult {
    // Pack booleans together to reduce memory footprint
    bool is_valid : 1;
    const std::tuple<std::vector<int>, int, int> *plan;

    // Constructor for direct initialization
    constexpr PlanValidationResult(
        bool valid, const std::tuple<std::vector<int>, int, int> *p)
        : is_valid(valid), plan(p) {}

    // Static size validation to allow compile-time optimization
    static constexpr bool isValidSize(int size) noexcept { return size >= 3; }

    // Main validation function optimized for branch prediction
    static PlanValidationResult validate(
        const int plan_idx, const int size,
        const ankerl::unordered_dense::map<
            int, std::vector<std::tuple<std::vector<int>, int, int>>>
            &map) noexcept {
        // Early size check (likely to be predicted well by CPU)
        if (unlikely(!isValidSize(size))) {
            return {false, nullptr};
        }

        // Find size in map
        auto size_it = map.find(size);
        if (unlikely(size_it == map.end())) {
            return {false, nullptr};
        }

        // Check plan index bounds
        const auto &plans = size_it->second;
        if (unlikely(static_cast<size_t>(plan_idx) >= plans.size())) {
            return {false, nullptr};
        }

        // Return valid result with plan pointer
        return {true, &plans[plan_idx]};
    }

    // Optional: Add cache-friendly batch validation
    static std::vector<PlanValidationResult> validateBatch(
        const std::vector<std::pair<int, int>> &plan_sizes,
        const ankerl::unordered_dense::map<
            int, std::vector<std::tuple<std::vector<int>, int, int>>>
            &map) noexcept {
        std::vector<PlanValidationResult> results;
        results.reserve(plan_sizes.size());  // Avoid reallocation

        for (const auto &[plan_idx, size] : plan_sizes) {
            results.push_back(validate(plan_idx, size, map));
        }

        return results;
    }
};

struct R1c {
    std::pair<std::vector<int>, int> info_r1c{};  // cut and plan
    int rhs{};
    std::vector<int> arc_mem{};  // Store only memory positions
};

struct Rank1MultiLabel {
    std::vector<int> c;
    std::vector<int> w_no_c;
    int plan_idx{};
    double vio{};
    char search_dir{};

    Rank1MultiLabel(std::vector<int> c, std::vector<int> w_no_c, int plan_idx,
                    double vio, char search_dir)
        : c(std::move(c)),
          w_no_c(std::move(w_no_c)),
          plan_idx(plan_idx),
          vio(vio),
          search_dir(search_dir) {}

    Rank1MultiLabel() = default;
};

struct MoveResult {
    double violation_score;
    std::variant<std::monostate, int, std::pair<int, int>> operation_data;

    MoveResult(double score = -std::numeric_limits<double>::max())
        : violation_score(score) {}
};

struct State {
    int begin;
    int target_remainder;
    std::vector<int> remaining_counts;
    std::vector<int> memory_indices;

    State(int b, int t, std::vector<int> rc, std::vector<int> mi)
        : begin(b),
          target_remainder(t),
          remaining_counts(std::move(rc)),
          memory_indices(std::move(mi)) {}
};

// New members for R1C separation
struct R1CMemory {
    std::vector<int> C;  // Customer subset
    int p_idx;           // Index of optimal vector p
    yzzLong arc_memory;  // Arc memory as bitset
    double violation;    // Cut violation

    R1CMemory(std::vector<int> c, int p, double v)
        : C(std::move(c)), p_idx(p), violation(v) {}
};
