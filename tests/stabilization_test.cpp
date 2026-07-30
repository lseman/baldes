#include "stabilization/Stabilization.h"

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

constexpr double kTolerance = 1e-12;

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

void test_subgradient_uses_pricing_columns_and_keeps_sign() {
    ModelData model;
    model.b     = {1.0, 1.0, 7.0};
    model.sense = {'=', '=', '<'};

    Label column;
    column.mutableRoute() = {0, 1, N_SIZE - 1};
    std::vector<Label *> pricing_columns{&column};

    DualSolution  center(3, 0.0);
    Stabilization stabilization(0.5, center);
    stabilization.updateNumK(1);
    stabilization.update_subgradient(model, center, pricing_columns);

    assert(stabilization.subgradient.size() == 3);
    assert(close(stabilization.subgradient[0], 0.0));
    assert(close(stabilization.subgradient[1], 1.0));
    assert(close(stabilization.subgradient[2], 0.0));
    assert(close(stabilization.subgradient_norm, 1.0));
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
    stabilization.update_stabilization_after_pricing_optim(model, DualSolution{3.0}, 0.0, {});

    assert(close(stabilization.alpha, 0.5));
    assert(close(stabilization.cur_alpha, 0.5));
    assert(close(stabilization.base_alpha, 0.5));
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
}
