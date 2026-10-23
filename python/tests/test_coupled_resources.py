"""Direct physical-graph resource contracts; execution held during implementation."""
from pybaldes import ResourceModel, Solver


def test_piecewise_dependencies_share_the_existing_solver_and_original_objective():
    solver = Solver(4, 0)
    for tail in range(3):
        solver.new_arc(tail, tail + 1, -10.0)
    resources = ResourceModel()
    parent = resources.add_cumulative_cost(0.0, [1.0] * 3, [2.0, -1.0, 3.0])
    child = resources.add_piecewise_linear_dependent(
        parent, 0.0, 0.0, 10.0,
        [[(0.0, 0.0), (1.0, 2.0), (2.0, 1.0), (4.0, 3.0)]] * 3,
    )
    resources.add_dependent(child, 10.0, -20.0, 20.0, [0.0] * 3, [-1.0] * 3)
    resources.install(solver)
    minimum, routes = solver.price(10)
    assert minimum == -21.0
    assert routes[0].original_cost == -21.0
    assert routes[0].nodes() == [0, 1, 2, 3]


def test_base_piecewise_map_and_state_cost_without_cumulative_resource():
    solver = Solver(4, 0)
    for tail in range(3):
        solver.new_arc(tail, tail + 1, -10.0)
    resources = ResourceModel()
    base = resources.add_piecewise_linear_resource(
        0.0, 0.0, 4.0, [[(0.0, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 4.0)]] * 3,
    )
    resources.add_piecewise_linear_cost(
        base, [[(0.0, 4.0), (1.0, -3.0), (2.0, 1.0), (4.0, -2.0)]] * 3,
    )
    resources.install(solver)
    minimum, routes = solver.price(10)
    assert minimum == -34.0
    assert routes[0].original_cost == -34.0


def test_interval_front_keeps_one_shared_initial_choice_across_all_arcs():
    solver = Solver(3, 0)
    solver.new_arc(0, 1, -10.0)
    solver.new_arc(1, 2, -10.0)
    model = ResourceModel()
    # Independent arc minima would charge -4; jointly x + (-x) costs zero.
    model.add_interval_cost(
        0.0, 4.0, 0.0, 4.0, [0.0, 0.0],
        [[(0.0, 0.0), (4.0, 4.0)], [(0.0, 0.0), (4.0, -4.0)]],
    )
    model.install(solver)
    minimum, routes = solver.price(10)
    assert minimum == -20.0
    assert routes[0].original_cost == -20.0
    assert routes[0].nodes() == [0, 1, 2]


def test_paired_relation_optimizes_waiting_and_retains_original_cost():
    solver = Solver(3, 0)
    solver.new_arc(0, 1, -20.0)
    solver.new_arc(1, 2, -20.0)
    model = ResourceModel()
    model.add_paired_piecewise_linear(
        0.0, 0.0, [(0.0, 0.0), (1.0, 0.0)],
        arrivals=[[(0.0, 1.0), (8.0, 9.0)]] * 2,
        increments=[[(0.0, 2.0), (8.0, 2.0)],
                    [(0.0, 1.0), (8.0, 1.0)]],
        lowers=[[(3.0, 0.0), (8.0, 0.0)], [(6.0, 0.0), (10.0, 0.0)]],
        uppers=[[(3.0, 100.0), (8.0, 100.0)], [(6.0, 100.0), (10.0, 100.0)]],
        waiting=[3, 2], mode="cost_bounded",
    )
    model.install(solver)
    minimum, routes = solver.price(10)
    # Wait two units on each arc: 2 + 3*2 + 1 + 2*2 = 13.
    assert minimum == -27.0
    assert routes[0].original_cost == -27.0
    assert routes[0].nodes() == [0, 1, 2]


def test_paired_relation_rejects_mismatched_vectors_and_unknown_modes():
    import pytest
    model = ResourceModel()
    common = dict(initial_lower=0.0, initial_upper=0.0,
                  initial=[(0.0, 0.0), (1.0, 0.0)], arrivals=[],
                  increments=[], lowers=[], uppers=[], waiting=[])
    with pytest.raises(RuntimeError, match="vector lengths differ"):
        model.add_paired_piecewise_linear(**(common | {"waiting": [0]}))
    with pytest.raises(RuntimeError, match="unknown paired resource mode"):
        model.add_paired_piecewise_linear(**common, mode="unknown")
