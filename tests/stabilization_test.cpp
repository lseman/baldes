#include "stabilization/Stabilization.h"

#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

constexpr double kTolerance = 1e-12;
constexpr double kNoBound   = std::numeric_limits<double>::quiet_NaN();

bool close(double lhs, double rhs) { return std::abs(lhs - rhs) <= kTolerance; }

void test_equality_duals_remain_unrestricted() {
    DualSolution  center{-2.0};
    Stabilization stabilization(0.5, center);

    const DualSolution result = stabilization.getStabDualSol(DualSolution{-4.0});

    assert(result.size() == 1);
    assert(close(result[0], -3.0));
}

void test_new_master_rows_are_not_smoothed_against_zero() {
    DualSolution  center{1.0};
    Stabilization stabilization(0.5, center);

    const DualSolution result = stabilization.getStabDualSol(DualSolution{3.0, -5.0});

    assert(result.size() == 2);
    assert(close(result[0], 2.0));
    assert(close(result[1], -5.0));
}

void test_projection_respects_constraint_senses() {
    DualSolution  center(3, 0.0);
    Stabilization stabilization(0.5, center);
    stabilization.constraint_senses = {'<', '>', '='};

    DualSolution duals{2.0, -3.0, -4.0};
    stabilization.project_onto_dual_domain(duals);

    assert(close(duals[0], 0.0));
    assert(close(duals[1], 0.0));
    assert(close(duals[2], -4.0));
}

// Master layout: one row per customer (nodes 1..N_SIZE-2), the vehicle row,
// then any cut rows.
ModelData make_master(std::size_t cut_rows) {
    ModelData model;
    model.b.assign(N_SIZE - 2, 1.0);
    model.sense.assign(N_SIZE - 2, '>');
    model.b.push_back(25.0);
    model.sense.push_back('<');
    for (std::size_t i = 0; i < cut_rows; ++i) {
        model.b.push_back(7.0);
        model.sense.push_back('<');
    }
    return model;
}

void test_subgradient_uses_pricing_columns_and_keeps_sign() {
    const ModelData model = make_master(1);

    Label column;
    column.mutableRoute() = {0, 1, N_SIZE - 1};
    std::vector<Label *> pricing_columns{&column};

    DualSolution  center(model.b.size(), 0.0);
    Stabilization stabilization(0.5, center);
    stabilization.updateNumK(1);
    stabilization.update_subgradient(model, center, pricing_columns);

    assert(stabilization.subgradient.size() == model.b.size());
    assert(close(stabilization.subgradient[0], 0.0));         // node 1 covered once
    assert(close(stabilization.subgradient[1], 1.0));         // node 2 uncovered
    assert(close(stabilization.subgradient[N_SIZE - 2], 0.0)); // vehicle row
    assert(close(stabilization.subgradient[N_SIZE - 1], 0.0)); // cut row
    assert(close(stabilization.subgradient_norm, std::sqrt(static_cast<double>(N_SIZE - 3))));
}

// g·(pi_out - pi_in) > 0: the bound still rises toward pi_out, so alpha
// must decrease.
void test_alpha_decreases_when_bound_rises_toward_out_point() {
    ModelData model;
    model.b     = {1.0};
    model.sense = {'>'};

    Label column; // covers no customer, so g = b = (1)
    column.mutableRoute() = {0, N_SIZE - 1};
    std::vector<Label *> pricing_columns{&column};

    DualSolution  center{0.0};
    Stabilization stabilization(0.5, center);
    stabilization.updateNumK(1);
    stabilization.update_stabilization_after_pricing_optim(model, DualSolution{3.0}, kNoBound, pricing_columns);

    assert(close(stabilization.alpha, 0.5 / 1.1));
    assert(close(stabilization.cur_alpha, 0.5 / 1.1));
}

// g·(pi_out - pi_in) < 0: pi_out overshoots, so alpha must increase.
void test_alpha_increases_when_out_point_overshoots() {
    ModelData model;
    model.b     = {1.0};
    model.sense = {'>'};

    Label column; // covers customer 1 twice, so g = (1 - 2) = (-1)
    column.mutableRoute() = {0, 1, 1, N_SIZE - 1};
    std::vector<Label *> pricing_columns{&column};

    DualSolution  center{0.0};
    Stabilization stabilization(0.5, center);
    stabilization.updateNumK(1);
    stabilization.update_stabilization_after_pricing_optim(model, DualSolution{3.0}, kNoBound, pricing_columns);

    assert(close(stabilization.alpha, 0.55));
    assert(close(stabilization.cur_alpha, 0.55));
}

void test_inconsistent_dimensions_are_rejected() {
    ModelData model;
    model.b     = {1.0};
    model.sense = {'='};

    DualSolution  center{0.0, 0.0};
    Stabilization stabilization(0.5, center);

    bool threw = false;
    try {
        stabilization.update_subgradient(model, center, {});
    } catch (const std::invalid_argument &) { threw = true; }
    assert(threw);
}

void test_clear_alpha_disables_smoothing() {
    DualSolution  center{1.0};
    Stabilization stabilization(0.5, center);
    stabilization.clearAlpha();

    const DualSolution result = stabilization.getStabDualSol(DualSolution{3.0});

    assert(close(stabilization.alpha, 0.0));
    assert(close(stabilization.cur_alpha, 0.0));
    assert(close(result[0], 3.0));
}

void test_missing_pricing_direction_does_not_decay_alpha() {
    ModelData model;
    model.b     = {1.0};
    model.sense = {'='};

    DualSolution  center{1.0};
    Stabilization stabilization(0.5, center);
    stabilization.updateNumK(1);
    stabilization.update_stabilization_after_pricing_optim(model, DualSolution{3.0}, kNoBound, {});

    assert(close(stabilization.alpha, 0.5));
    assert(close(stabilization.cur_alpha, 0.5));
    assert(close(stabilization.base_alpha, 0.5));
}

// The center moves to the separation point only when its Lagrangian bound
// beats the center's (Wentges).
void test_center_moves_only_when_bound_improves() {
    ModelData model;
    model.b     = {1.0};
    model.sense = {'>'};

    DualSolution  center{0.0};
    Stabilization stabilization(0.5, center);

    // pi_sep = 0.5 * 0 + 0.5 * 2 = 1, with an improving bound.
    stabilization.getStabDualSol(DualSolution{2.0});
    stabilization.update_stabilization_after_pricing_optim(model, DualSolution{2.0}, 10.0, {});
    stabilization.update_stabilization_after_iter(DualSolution{2.0});
    assert(close(stabilization.cur_stab_center[0], 1.0));

    // pi_sep = 0.5 * 1 + 0.5 * 3 = 2, with a worse bound: center stays.
    stabilization.getStabDualSol(DualSolution{3.0});
    stabilization.update_stabilization_after_pricing_optim(model, DualSolution{3.0}, 5.0, {});
    stabilization.update_stabilization_after_iter(DualSolution{3.0});
    assert(close(stabilization.cur_stab_center[0], 1.0));
}

// A mispriced round leaves alpha to the misprice schedule.
void test_mispriced_round_does_not_adapt_alpha() {
    ModelData model;
    model.b     = {1.0};
    model.sense = {'>'};

    Label column;
    column.mutableRoute() = {0, N_SIZE - 1};
    std::vector<Label *> pricing_columns{&column};

    DualSolution  center{0.0};
    Stabilization stabilization(0.5, center);
    stabilization.updateNumK(1);
    stabilization.update_stabilization_after_pricing_optim(model, DualSolution{3.0}, kNoBound, pricing_columns,
                                                           true);

    assert(close(stabilization.alpha, 0.5));
    assert(close(stabilization.cur_alpha, 0.5));
}

} // namespace

int main() {
    test_equality_duals_remain_unrestricted();
    test_new_master_rows_are_not_smoothed_against_zero();
    test_projection_respects_constraint_senses();
    test_subgradient_uses_pricing_columns_and_keeps_sign();
    test_inconsistent_dimensions_are_rejected();
    test_clear_alpha_disables_smoothing();
    test_missing_pricing_direction_does_not_decay_alpha();
    test_alpha_decreases_when_bound_rises_toward_out_point();
    test_alpha_increases_when_out_point_overshoots();
    test_center_moves_only_when_bound_improves();
    test_mispriced_round_does_not_adapt_alpha();
}
