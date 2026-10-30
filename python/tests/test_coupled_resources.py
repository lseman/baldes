"""Direct physical-graph resource contracts."""
import pytest
from pybaldes import ResourceModel, Solver


def test_custom_resource_uses_zero_initial_state_and_is_a_numeric_parent():
    solver = Solver(3, 0)
    solver.new_arc(0, 1, -20.)
    solver.new_arc(1, 2, -20.)
    model = ResourceModel()
    parent = model.add_custom_resource([2., 1.], [3., 4.], forward_bound=7., backward_bound=3.)
    model.add_piecewise_linear_cost(parent, [[(0., 0.), (10., 10.)]] * 2)
    model.install(solver)
    minimum, routes = solver.price(10)
    # Custom cost 2*3+3*4=18, parent-dependent charges 2+3=5, arc cost -40.
    assert minimum == -17.
    assert routes[0].original_cost == -17.
    assert routes[0].nodes() == [0, 1, 2]


def test_custom_resource_json_kind_survives_network_round_trip_and_solves():
    import pytest
    from pybaldes import NetworkProblem, PricingArc, PricingNetwork, SharedRow, has_clp
    if not has_clp:
        pytest.skip("Clp build required")
    definition = dict(kind="custom_resource", increments=[2., 1.], weights=[3., 4.],
                      forward_bound=7., backward_bound=3.)
    model = NetworkProblem([SharedRow("eq", 1.)], [PricingNetwork(
        1, 3, (0., 1.), [PricingArc(0, 1, 0., entries=[(0, 1.)]), PricingArc(1, 2, 0.)],
        coupled_resources=[definition],
    )])
    restored = NetworkProblem.from_json(model.to_json())
    assert restored == model
    assert restored.networks[0].coupled_resources[0]["kind"] == "custom_resource"
    result = restored.solve(stabilize=False)
    assert result.status == "Optimal"
    assert result.lp == pytest.approx(18.)
    assert result.routes[0].cost == pytest.approx(18.)


def test_custom_resource_rejects_invalid_data_when_installed():
    import pytest
    for increments, weights, options in [
        ([], [1.], {}), ([1.], [], {}), ([float("inf")], [1.], {}),
        ([1.], [float("nan")], {}), ([1.], [1.], dict(forward_bound=-1.)),
        ([1.], [1.], dict(backward_bound=float("inf"))),
    ]:
        solver = Solver(2, 0)
        solver.new_arc(0, 1, 0.)
        model = ResourceModel()
        model.add_custom_resource(increments, weights, **options)
        with pytest.raises(RuntimeError):
            model.install(solver)


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


def test_computed_front_domain_is_an_explicit_installation_choice():
    import pytest
    from pybaldes import EncodedFunction
    solver = Solver(3, 0)
    solver.new_arc(0, 1, -10.)
    solver.new_arc(1, 2, -10.)
    model = ResourceModel()
    point = 0.30000000000000004
    model.add_slope_interval_cost(
        .1, .1, 0., 1., [.2, 0.],
        [EncodedFunction.slope_segments([(3., point)], "endpoint_tolerance"),
         EncodedFunction.slope_segments([(0., point)], "endpoint_tolerance")],
        policy="front_algebra",
    )
    model.install(solver, front_program_domain=True)
    minimum, routes = solver.price(10)
    assert minimum == pytest.approx(-17.)
    assert routes[0].original_cost == pytest.approx(-17.)
    # Invalid opt-in must leave the live, already installed model intact.
    with pytest.raises(RuntimeError, match="native raw resources"):
        ResourceModel().install(solver, front_program_domain=True)
    assert solver.price(10)[0] == pytest.approx(-17.)


def test_paired_front_install_rejects_missing_arc_operands_and_preserves_the_live_solver():
    solver = Solver(2, 0)
    solver.new_arc(0, 1, -10.0)
    assert solver.price(10)[0] == -10.0
    model = ResourceModel()
    paired = model.add_paired_front(mode="cost_bounded")
    model.set_paired_front_program(paired, {
        "source_bounds": [0.0, 1.0], "terminal_bounds": [0.0, 1.0],
        "root": {"encoding": "affine", "intercept": 0.0, "slope": 0.0},
        "arcs": [],
    })
    with pytest.raises(RuntimeError):
        model.install(solver)
    assert solver.price(10)[0] == -10.0
