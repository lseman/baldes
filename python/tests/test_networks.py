import gc
import json
from dataclasses import replace
from pathlib import Path

import pytest

from pybaldes import (
    NetworkProblem, NetworkRowCoefficients, PricingArc, PricingNetwork,
    PricingResource, SharedRow, has_clp, solve,
)

pytestmark = pytest.mark.skipif(not has_clp, reason="Clp build required")
FIXTURE = Path(__file__).resolve().parents[2] / "examples/networks/triangle.json"


def triangle():
    return NetworkProblem.from_file(FIXTURE)


@pytest.mark.parametrize("direction,first_side", [("upper-first", 1), ("lower-first", 0)])
def test_named_native_branch_direction_reaches_trial_evaluation(direction, first_side):
    result = triangle().solve(
        tree=True, stabilize=False, network_branching="phased",
        mapped_variables=({"name": "count42", "priority": 1000.,
                           "networks": [{"network": 42, "per_route": 1.}]},),
        network_branch_directions={"count42": direction},
    )
    assert result.status == "Optimal"
    assert result.incumbent == pytest.approx(2.)
    assert result.bound == pytest.approx(2.)
    observations = [e for e in result.mapped_branch_evaluations if e.name == "count42"]
    assert observations
    assert observations[0].side == first_side


def test_named_branch_directions_reject_root_calls_and_unknown_directions():
    with pytest.raises(ValueError, match="requires tree=true"):
        triangle().solve(network_branch_directions={"count42": "lower-first"})
    with pytest.raises(ValueError, match="unknown variant"):
        triangle().solve(tree=True, network_branching="phased",
                         network_branch_directions={"count42": "up"})
    with pytest.raises(ValueError, match="require phased"):
        triangle().solve(tree=True, node_limit=0,
                         network_branch_directions={"count42": "lower-first"})
    with pytest.raises(ValueError, match="nonempty generator name"):
        triangle().solve(tree=True, node_limit=0, network_branching="phased",
                         network_branch_directions={" ": "lower-first"})


@pytest.mark.parametrize("schedule", ["full-sweep", "cyclic", "sorted", "promising-first", "unvisited-first"])
def test_network_pricing_schedules_across_fresh_and_snapshot_solves(schedule):
    root = triangle().solve(network_pricing_schedule=schedule)
    assert root.status == "Optimal"
    assert root.lp == pytest.approx(1.5)
    child = root.snapshot.branch_assignment(0, 7, True)
    assert child.solve(network_pricing_schedule=schedule).lp == pytest.approx(2.)
    assert root.snapshot.solve(network_pricing_schedule=schedule).lp == pytest.approx(1.5)
    assert triangle().solve(tree=True, network_pricing_schedule=schedule).incumbent == pytest.approx(2.)


def test_unknown_network_pricing_schedule_is_rejected():
    with pytest.raises(ValueError, match="full-sweep.*cyclic.*sorted"):
        triangle().solve(network_pricing_schedule="unknown")


@pytest.mark.parametrize("bands", [
    [{"half_width": .5, "penalty": None}],
    [{"half_width": .5, "penalty": 1.}],
    [{"half_width": .1, "penalty": .5}, {"half_width": .5, "penalty": 1.}],
])
def test_explicit_proximal_bands_keep_original_network_objectives(bands):
    proximal = {"bands": bands}
    root = triangle().solve(proximal=proximal, network_pricing_schedule="promising-first")
    assert root.lp == pytest.approx(1.5)
    assert root.snapshot.branch_assignment(0, 7, True).solve(proximal=proximal).lp == pytest.approx(2.)
    assert triangle().solve(tree=True, proximal=proximal).incumbent == pytest.approx(2.)
    with pytest.raises(ValueError, match="proximal"):
        triangle().solve(proximal={"bands": [{"half_width": -1., "penalty": None}]})


def test_native_queue_budget_is_validated():
    assert triangle().solve(network_pricing_schedule="unvisited-first", network_unpromising_limit=2).lp == pytest.approx(1.5)
    with pytest.raises(ValueError, match="network_unpromising_limit"):
        triangle().solve(network_unpromising_limit=0)


def test_multipoint_proximal_support_planes_preserve_network_solutions():
    proximal = {"bands": [{"half_width": .5, "penalty": 1.}], "multi_point": True}
    root = triangle().solve(proximal=proximal, network_pricing_schedule="unvisited-first")
    assert root.lp == pytest.approx(1.5)
    assert root.snapshot.solve(proximal=proximal).lp == pytest.approx(1.5)
    assert triangle().solve(tree=True, proximal=proximal).incumbent == pytest.approx(2.)


@pytest.mark.parametrize("five_piece", [False, True])
def test_automatic_curvature_needs_no_manually_initialized_state(five_piece):
    proximal = {"curvature": {"five_piece": five_piece}}
    root = triangle().solve(proximal=proximal)
    assert root.lp == pytest.approx(1.5)
    assert root.snapshot.solve(proximal=proximal).lp == pytest.approx(1.5)
    assert root.snapshot.branch_assignment(0, 7, True).solve(proximal=proximal).lp == pytest.approx(2.)
    dynamic = root.snapshot.with_row(SharedRow("le", 10.), (), dynamic=True)
    assert dynamic.solve(proximal=proximal).lp == pytest.approx(1.5)
    assert triangle().solve(tree=True, proximal=proximal).incumbent == pytest.approx(2.)


def test_model_json_round_trip_and_shared_root():
    model = triangle()
    assert NetworkProblem.from_json(model.to_json()) == model
    root = model.solve(stabilize=False)
    assert root.status == "Optimal" and root.mode == "root"
    assert root.lp == pytest.approx(1.5)
    assert {n for n, calls in root.oracle_calls if calls} == {7, 42}
    assert len([r for r in root.routes if r.arcs == (0, 1)]) == 2
    assert root.snapshot is not None
    data = json.loads(root.to_json())
    assert data["lp"] == pytest.approx(1.5)
    assert "snapshot" not in data
    assert data["incumbent"] is None


@pytest.mark.parametrize("bound_mode", ["numerical", "certified"])
def test_assignment_tree_and_limits(bound_mode):
    result = solve(triangle(), tree=True, bound_mode=bound_mode)
    assert result.status == "Optimal" and result.mode == "tree"
    assert result.incumbent == pytest.approx(2)
    assert result.bound == pytest.approx(2)
    assert result.nodes == 6 and result.open_nodes == 0
    assert result.tree_processed_nodes == 7
    assert result.strong_branching_trials == 0
    assert result.pricing_calls > 0
    assert {n for n, calls in result.oracle_calls if calls} == {7, 42}
    assert result.snapshot is None
    assert all(r.value.is_integer() for r in result.routes)
    limited = triangle().solve(tree=True, node_limit=1)
    assert limited.status == "NodeLimit" and limited.open_nodes == 2
    assert limited.incumbent is None and limited.bound == pytest.approx(1.5)
    zero = triangle().solve(tree=True, node_limit=0)
    assert zero.bound is None and json.loads(zero.to_json())["bound"] is None


@pytest.mark.parametrize("policy", ["resource-first", "multiplicity-first", "phased"])
def test_explicit_network_branch_order_in_fresh_and_snapshot_trees(policy):
    model = triangle()
    fresh = model.solve(tree=True, network_branching=policy, stabilize=False)
    assert fresh.status == "Optimal"
    assert fresh.incumbent == pytest.approx(2)
    root = model.solve(stabilize=False)
    resumed = root.snapshot.solve(tree=True, network_branching=policy, stabilize=False)
    assert resumed.status == "Optimal"
    assert resumed.incumbent == pytest.approx(2)
    assert root.snapshot.solve(stabilize=False).lp == pytest.approx(1.5)


def test_invalid_network_branch_order_is_rejected():
    with pytest.raises(ValueError, match="variant|branch"):
        triangle().solve(tree=True, network_branching="unknown")


def test_python_nodes_branch_restore_and_own_their_state():
    root = triangle().solve()
    candidates = root.snapshot.fractional_assignments()
    candidate = next(c for c in candidates if c.shared_row == 0 and c.network == 7)
    assert candidate.value == pytest.approx(0.5)
    parent = root.snapshot
    assigned = parent.branch_assignment(0, 7, True)
    excluded = parent.branch_assignment(0, 7, False)
    del root
    gc.collect()
    for _ in range(2):
        assert assigned.solve().lp == pytest.approx(2)
        assert excluded.solve().lp == pytest.approx(2)
        assert parent.solve().lp == pytest.approx(1.5)


def test_additive_rows_and_fixed_routes_reach_future_nodes():
    problem = triangle()
    root = problem.solve()
    coeffs = tuple(NetworkRowCoefficients(n.id, (0.,) * len(n.arcs), 1.) for n in problem.networks)
    cut = root.snapshot.with_row(SharedRow("ge", 2.), coeffs).solve()
    assert cut.lp == pytest.approx(2)
    pair = next(r for r in cut.routes if r.network == 42 and r.arcs == (0, 1))
    fixed = cut.snapshot.fix_route(pair.route_index)
    child = fixed.with_row(SharedRow("eq", 2.), coeffs).branch_assignment(0, 7, False).solve()
    assert child.lp == pytest.approx(2)
    assert len(child.fixed_routes) == 1 and child.fixed_routes[0].network == 42
    assert sum(r.value for r in child.routes) == pytest.approx(1)
    assert child.snapshot.fractional_assignments() == ()
    assert root.snapshot.solve().lp == pytest.approx(1.5)


def test_python_construction_and_resource_inputs():
    model = NetworkProblem(
        [SharedRow("eq", 1)],
        [PricingNetwork(19, 3, (0, 1), [
            PricingArc(0, 1, 2., consumption=[1.], entries=[(0, 1.)]),
            PricingArc(1, 2, 1., consumption=[0.]),
        ], resources=[PricingResource(bounds=[(0., 2.)] * 3)])],
    )
    assert "time_windows" not in json.loads(model.to_json())["networks"][0]["resources"][0]
    result = model.solve()
    assert result.status == "Optimal" and result.lp == pytest.approx(3.)
    assert result.routes[0].network == 19
    assert result.routes[0].resources == (1.,)
    infeasible_resource = replace(model.networks[0].resources[0], bounds=((0., 0.),) * 3)
    invalid = replace(model, networks=(replace(model.networks[0], resources=(infeasible_resource,)),))
    # The pricing domain has no feasible covering route: it must not report an optimum.
    try:
        output = invalid.solve()
    except ValueError:
        return
    assert output.status != "Optimal"


@pytest.mark.parametrize("options", [
    {"max_iterations": 0}, {"time_limit": float("nan")}, {"time_limit": -1},
    {"bound_mode": "wrong"}, {"log_interval": 0}, {"artificial_cost": 0},
    {"pricing_options": {"unknown": 1}},
])
def test_invalid_options(options):
    with pytest.raises((ValueError, OverflowError)):
        triangle().solve(**options)


def test_invalid_models_and_snapshot_operations():
    model = triangle()
    duplicate = replace(model, networks=(model.networks[0], replace(model.networks[1], id=7)))
    with pytest.raises(ValueError, match="identity"):
        duplicate.solve()
    bad_arc = replace(model.networks[0].arcs[0], entries=((99, 1.),))
    bad_network = replace(model.networks[0], arcs=(bad_arc,) + model.networks[0].arcs[1:])
    with pytest.raises(ValueError):
        replace(model, networks=(bad_network, model.networks[1])).solve()
    node = model.solve().snapshot
    with pytest.raises(ValueError):
        node.branch_assignment(0, 99, True)
    assert node.solve(tree=True).incumbent == pytest.approx(2.)
    with pytest.raises(ValueError, match="unknown field"):
        node.solve(typo=True)
    with pytest.raises(TypeError):
        node.fix_route(0, quantity=0.5)
    with pytest.raises(ValueError):
        node.fix_route(0, quantity=2**53 + 1)
    with pytest.raises(ValueError):
        node.with_row(SharedRow("eq", 1), (NetworkRowCoefficients(7, (), 1.),))
    assert node.solve().lp == pytest.approx(1.5)


def test_json_resources_keep_exact_incumbent_validation():
    import math
    resource = PricingResource(bounds=((1., 1.),) * 3)
    problem = NetworkProblem((SharedRow("eq", 1.),), (PricingNetwork(
        7, 3, (0., 1.), (
            PricingArc(0, 1, 1., consumption=(math.ulp(1.) / 2,), entries=((0, 1.),)),
            PricingArc(1, 2, 0., consumption=(0.,)),
        ), resources=(resource,),
    ),))
    root = problem.solve()
    with pytest.raises(ValueError, match="elementary path"):
        root.snapshot.fix_route(root.routes[0].route_index)
    tree = problem.solve(tree=True)
    assert tree.status == "Incomplete" and tree.incumbent is None


def test_enumeration_reports_resume_and_unsupported_mip_builds():
    from pybaldes import has_highs
    parent = triangle().solve()
    assert parent.snapshot.solve_enumerated_mip() is None
    completed = parent.snapshot.solve(enumerate=True)
    assert completed.lp == pytest.approx(1.5)
    report = completed.enumeration_reports[0]
    assert report.complete and report.routes == 9
    repeated = completed.snapshot.solve(enumerate=True)
    assert repeated.enumeration_reports[0].attempts == 1
    partial = parent.snapshot.solve(enumerate=True, enumeration_route_limit=5)
    assert not partial.enumeration_reports[0].complete
    assert partial.enumeration_reports[0].transferred == 0
    assert partial.snapshot.solve().lp == pytest.approx(1.5)
    assert parent.snapshot.solve().enumeration_reports[0].attempts == 0
    assert json.loads(completed.to_json())["enumeration_reports"][0]["complete"]
    if not has_highs:
        with pytest.raises(ValueError, match="highs"):
            completed.snapshot.solve_enumerated_mip()
        with pytest.raises(ValueError, match="highs"):
            triangle().solve(tree=True, enumerated_mip=True)


@pytest.mark.skipif(not __import__("pybaldes").has_highs, reason="HiGHS required")
def test_full_integer_transfer_and_mip_retry():
    completed = triangle().solve(enumerate=True)
    mip = completed.snapshot.solve_enumerated_mip()
    assert mip.mode == "mip" and mip.status == "Optimal"
    assert mip.incumbent == pytest.approx(2.)
    limited = completed.snapshot.solve_enumerated_mip(node_limit=0)
    assert limited.status == "NodeLimit" and limited.incumbent is None
    cutoff = completed.snapshot.solve_enumerated_mip(objective_cutoff=2.)
    assert cutoff.status == "ObjectiveCutoff" and cutoff.incumbent is None
    assert completed.snapshot.solve_enumerated_mip().incumbent == pytest.approx(2.)
    tree = triangle().solve(tree=True, enumerated_mip=True)
    assert tree.status == "Optimal" and tree.incumbent == pytest.approx(2.)
    assert tree.nodes == 1 and tree.enumerated_mip_calls == 1


def test_equality_transfer_filters_waiting_relaxation_before_certification():
    model = NetworkProblem.from_json(json.dumps({
        "rows": [{"sense": "eq", "rhs": 1}],
        "networks": [{
            "id": 7, "vertices": 4, "multiplicity": [0, 1],
            "resources": [{"kind": "equality", "initial": 1,
                           "bounds": [[0, 10], [5, 10], [0, 10], [0, 10]]}],
            "arcs": [
                {"tail": 0, "head": 1, "cost": 1, "consumption": [2], "entries": [[0, 1]]},
                {"tail": 1, "head": 3, "cost": 0, "consumption": [0]},
                {"tail": 0, "head": 2, "cost": 3, "consumption": [1], "entries": [[0, 1]]},
                {"tail": 2, "head": 3, "cost": 0, "consumption": [0]},
            ],
        }],
    }))
    result = model.solve(enumerate=True, bound_mode="certified")
    assert result.status == "Optimal"
    assert result.lp == pytest.approx(3.) and result.bound == pytest.approx(3.)
    assert result.enumeration_reports[0].complete
    assert result.enumeration_reports[0].routes == 1
    assert all(r.arcs == (2, 3) for r in result.routes if r.value > 0)
    if __import__("pybaldes").has_highs:
        mip = result.snapshot.solve_enumerated_mip()
        assert mip.status == "Optimal" and mip.incumbent == pytest.approx(3.)


@pytest.mark.parametrize("kind", ["multiplicity", "arc"])
@pytest.mark.parametrize("bound_mode", ["numerical", "certified"])
def test_generic_integer_tree_without_fractional_assignment_candidates(kind, bound_mode):
    if kind == "multiplicity":
        model = NetworkProblem([SharedRow("ge", 0.5)], [PricingNetwork(
            7, 3, (0, 3), [
                PricingArc(0, 1, 1., entries=[(0, 1.)]),
                PricingArc(1, 2, 0.),
            ],
        )])
    else:
        model = NetworkProblem([SharedRow("eq", 1.), SharedRow("ge", 0.5)], [PricingNetwork(
            7, 4, (0, 3), [
                PricingArc(0, 1, 0., entries=[(0, 1.)]),
                PricingArc(1, 3, 0.),
                PricingArc(0, 2, 1., entries=[(0, 1.), (1, 1.)]),
                PricingArc(2, 3, 0.),
            ],
        )])
    root = model.solve(stabilize=False, bound_mode=bound_mode)
    assert root.lp == pytest.approx(0.5)
    assert root.snapshot.fractional_assignments() == ()
    tree = model.solve(tree=True, stabilize=False, bound_mode=bound_mode)
    assert tree.status == "Optimal" and tree.incumbent == pytest.approx(1.)
    assert tree.nodes == 3 and tree.open_nodes == 0
    assert all(r.value.is_integer() for r in tree.routes)


@pytest.mark.skipif(not __import__("pybaldes").has_highs, reason="HiGHS required")
def test_cutoff_tree_report_and_missing_incumbent_preserve_original_feasibility():
    result = triangle().solve(
        tree=True, enumerated_mip=True, mip_objective_cutoff=1.75,
        enumeration_route_limit=3, node_limit=1, stabilize=False,
    )
    assert result.status == "NodeLimit"
    assert result.incumbent is None and result.open_nodes > 0
    assert result.enumerated_mip_calls == 1
    scoped = [r for r in result.enumeration_reports if r.coverage_cutoff is not None]
    assert len(scoped) == 1 and scoped[0].complete
    assert scoped[0].coverage_cutoff == 1.75 and scoped[0].routes == 3
    data = json.loads(result.to_json())
    assert any(r["coverage_cutoff"] == 1.75 for r in data["enumeration_reports"])
    root = triangle().solve(stabilize=False)
    assert root.lp == pytest.approx(1.5)
    assert all(r.coverage_cutoff is None for r in root.enumeration_reports)
    with pytest.raises(ValueError, match="requires enumerated MIP"):
        triangle().solve(mip_objective_cutoff=3.)


@pytest.mark.parametrize("waiting", [False, True])
@pytest.mark.parametrize("complete", [False, True])
def test_grouped_resource_children_and_restricted_trees(waiting, complete):
    from pybaldes import NetworkResourceGroupMember
    networks = []
    for ident, costs, extra in [(7, (1., 4.), False), (42, (2., 3.), True)]:
        sink = 4 if extra else 3
        bounds = [(0., 3.)] * (sink + 1)
        if not waiting:
            bounds[1], bounds[2] = (0., .5), (0., 2.5)
        arcs = []
        for vertex, cost, q in [(1, costs[0], .5), (2, costs[1], 2.5)]:
            arcs.extend([
                PricingArc(0, vertex, cost, consumption=(q,), entries=((0, 1.),)),
                PricingArc(vertex, sink, 0.),
            ])
        if extra:
            arcs.extend([PricingArc(0, 3, 1., entries=((1, 1.),)), PricingArc(3, sink, 0.)])
        networks.append(PricingNetwork(ident, sink + 1, (0, 3), arcs,
                                       resources=(PricingResource(bounds=bounds),)))
    model = NetworkProblem((SharedRow("eq", 1.), SharedRow("eq", 1.)), networks)
    parent = model.solve(stabilize=False, enumerate=complete)
    members = (NetworkResourceGroupMember(7, [0, 2]), NetworkResourceGroupMember(42, [0, 2]))
    for upper in [False, True]:
        child = parent.snapshot.with_resource_group_branch(0, 0, 1.5, upper, members)
        expected = 4. if upper and not waiting else 2.
        solved = child.solve(stabilize=False, enumerate=True)
        assert solved.lp == pytest.approx(expected)
        assert child.solve(tree=True, stabilize=False).incumbent == pytest.approx(expected)
        if complete:
            capped = child.with_row(SharedRow("le", .25),
                                    (NetworkRowCoefficients(7, (1., 0., 0., 0.)),)).solve()
            if waiting:
                assert capped.lp == pytest.approx(2.75)
    with pytest.raises(ValueError, match="duplicate"):
        parent.snapshot.with_resource_group_branch(0, 0, 1.5, True, members + members[:1])
    assert parent.snapshot.solve().lp == pytest.approx(2.)


@pytest.mark.parametrize("strong", [False, True])
def test_registered_cross_network_aggregate_branching_preserves_original_tree(strong):
    mappings = ({
        "name": "total-route-count", "priority": 1.,
        "networks": [{"network": 7, "per_route": 1.},
                     {"network": 42, "per_route": 1.}],
    },)
    result = triangle().solve(tree=True, mapped_variables=mappings,
                              network_branching="resource-first",
                              mapped_strong_branching=strong)
    assert result.status == "Optimal"
    assert result.incumbent == pytest.approx(2.)
    assert (result.strong_branching_trials > 0) == strong
    payload = json.loads(result.to_json())
    assert payload["strong_branching_trials"] == result.strong_branching_trials
    assert all(e.name == "total-route-count" for e in result.mapped_branch_evaluations)


def test_native_tree_selection_and_processed_node_accounting():
    native = triangle().solve(tree=True, network_branching="multiplicity-first",
                              network_tree_search="default")
    best_bound = triangle().solve(tree=True, network_branching="multiplicity-first",
                                  network_tree_search="best-bound")
    strategy2 = triangle().solve(tree=True, network_branching="multiplicity-first",
                                 network_tree_search="bound-priority")
    strategy3 = triangle().solve(tree=True, network_branching="multiplicity-first",
                                 network_tree_search="depth-bound-priority")
    assert strategy3.status == "Optimal"
    assert strategy3.incumbent == pytest.approx(2.)
    strategy2_unrounded = triangle().solve(tree=True, network_branching="multiplicity-first",
                                          network_tree_search="bound-priority",
                                          network_strategy2_rounding="unrounded")
    strategy2_exact = triangle().solve(tree=True, network_branching="multiplicity-first",
                                       network_tree_search="bound-priority",
                                       network_strategy2_rounding="exact-ceil")
    assert native.status == best_bound.status == strategy2.status == strategy2_exact.status == "Optimal"
    assert native.incumbent == pytest.approx(2.)
    assert best_bound.incumbent == pytest.approx(2.)
    assert strategy2.incumbent == pytest.approx(2.)
    assert strategy2_unrounded.incumbent == pytest.approx(2.)
    assert strategy2_unrounded.tree_processed_nodes == strategy2.tree_processed_nodes
    assert strategy2_exact.incumbent == pytest.approx(2.)
    assert native.tree_processed_nodes >= native.nodes
    assert best_bound.tree_processed_nodes >= best_bound.nodes
    assert json.loads(native.to_json())["tree_processed_nodes"] == native.tree_processed_nodes
    assert json.loads(strategy2.to_json())["tree_processed_nodes"] == strategy2.tree_processed_nodes
    overflow = triangle().solve(tree=True, network_branching="multiplicity-first",
                                network_tree_search="default", network_open_node_limit=1)
    assert overflow.status == "Optimal"
    assert overflow.incumbent == pytest.approx(2.)
    all_overflow = triangle().solve(tree=True, network_branching="multiplicity-first",
                                    network_tree_search="default", network_open_node_limit=0)
    assert all_overflow.status == "Incomplete"
    assert all_overflow.incumbent is None
    assert all_overflow.bound == pytest.approx(1.5)
    assert all_overflow.open_nodes == 2
    assert all_overflow.tree_processed_nodes == 1


@pytest.mark.parametrize("depth", [-1, 0, 1])
def test_signed_stabilization_depth_and_promising_budget_are_public(depth):
    root = triangle().solve(stabilization_max_tree_depth=depth,
                            proximal={"curvature": {}}, network_promising_limit=2)
    assert root.lp == pytest.approx(1.5)
    with pytest.raises(ValueError, match="network_promising_limit"):
        triangle().solve(network_promising_limit=0)


@pytest.mark.parametrize("proximal", [
    {"explicit": {"box_step": True}},
    {"explicit": {}},
    {"explicit": {"five_piece": True}},
    {"bands": [{"half_width": 0., "penalty": 0.8999999761581421},
               {"half_width": 0., "penalty": 0.8999999761581421}],
     "multi_point": True, "rhs_row_scaling": True},
])
def test_native_explicit_defaults_preserve_root_resume_child_and_tree(proximal):
    root = triangle().solve(proximal=proximal)
    assert root.lp == pytest.approx(1.5)
    assert root.snapshot.solve(proximal=proximal).lp == pytest.approx(1.5)
    assert root.snapshot.branch_assignment(0, 7, True).solve(proximal=proximal).lp == pytest.approx(2.)
    assert triangle().solve(tree=True, proximal=proximal).incumbent == pytest.approx(2.)


def test_explicit_front_program_model_round_trip_and_certified_original_objective():
    model_file = Path(__file__).resolve().parents[2] / "tests/fixtures/coupled/native_machine_network.json"
    model = NetworkProblem.from_json(model_file.read_text())
    assert model.networks[0].front_program_domain is True
    assert NetworkProblem.from_json(model.to_json()) == model
    root = model.solve(stabilize=False, bound_mode="certified")
    assert root.lp == pytest.approx(1003.)
    result = model.solve(tree=True, stabilize=False, bound_mode="certified")
    assert result.status == "Optimal"
    assert result.incumbent == pytest.approx(1003.)
    assert result.bound == pytest.approx(1003.)


def test_bounded_walk_model_round_trip_and_certified_negative_cycle():
    model_file = Path(__file__).resolve().parents[2] / "tests/fixtures/coupled/bounded_walk_network.json"
    model = NetworkProblem.from_json(model_file.read_text())
    assert model.networks[0].route_domain == {"kind": "bounded_walks", "max_arcs": 4}
    assert NetworkProblem.from_json(model.to_json()) == model
    root = model.solve(stabilize=False, bound_mode="certified")
    assert root.lp == pytest.approx(-3.)
    result = model.solve(tree=True, stabilize=False, bound_mode="certified")
    assert result.status == "Optimal"
    assert result.incumbent == pytest.approx(-3.)
    assert result.bound == pytest.approx(-3.)


@pytest.mark.parametrize("gap", [0., 1e-6, 0.4])
def test_native_branch_scoring_gap_keeps_original_proof_bounds(gap):
    result = triangle().solve(tree=True, network_branching="phased",
                              network_scoring_gap_tolerance=gap)
    assert result.status == "Optimal"
    assert result.incumbent == pytest.approx(2.)
    assert result.bound == pytest.approx(2.)


def test_negative_native_branch_scoring_gap_is_rejected():
    with pytest.raises(ValueError, match="scoring gap tolerance"):
        triangle().solve(tree=True, network_scoring_gap_tolerance=-1.)


@pytest.mark.parametrize("mode", ["numerical", "certified"])
def test_native_multiphase_screens_are_distinct_from_processed_exact_trials(mode):
    mappings = json.loads((Path(__file__).resolve().parents[2] / "tests/fixtures/multi_network_native_phased_mappings.json").read_text())
    result = triangle().solve(tree=True, network_branching="resource-first",
                              mapped_variables=tuple(mappings), stabilize=False,
                              bound_mode=mode)
    assert result.status == "Optimal"
    assert result.incumbent == pytest.approx(2.)
    assert result.bound == pytest.approx(2.)
    assert result.nodes == 1
    assert result.tree_processed_nodes == 3
    assert result.strong_branching_trials == 6
    assert result.iterations == 18
    assert sum(calls for _, calls in result.oracle_calls) == 2


@pytest.mark.parametrize("mode", ["numerical", "certified"])
def test_shared_inactive_tree_paths_keep_original_bounds(mode):
    result = triangle().solve(tree=True, network_share_inactive_columns=True,
                              bound_mode=mode)
    assert result.status == "Optimal"
    assert result.incumbent == 2.
    assert abs(result.bound - 2.) < 1e-8


def test_shared_inactive_paths_require_a_tree():
    with pytest.raises(ValueError, match="requires tree=true"):
        triangle().solve(network_share_inactive_columns=True)


@pytest.mark.parametrize("mode", ["numerical", "certified"])
def test_half_incident_mapping_preserves_integer_tree_bounds(mode):
    mapping = {
        "name": "ESA_7_1",
        "priority": 1000.,
        "networks": [{
            "network": 7,
            "arcs": [[0, 0.5], [1, 0.5], [6, 0.5], [7, 0.5]],
        }],
    }
    result = triangle().solve(tree=True, mapped_variables=(mapping,),
                              network_branching="resource-first", stabilize=False,
                              bound_mode=mode)
    assert result.status == "Optimal"
    assert result.incumbent == pytest.approx(2.)
    assert result.bound == pytest.approx(2.)


def test_unpaired_half_mapping_is_rejected_before_branching():
    mapping = {"name": "fractional-route", "networks": [{"network": 7,
               "arcs": [[0, 0.5]]}]}
    with pytest.raises(ValueError, match="integer on every graph path"):
        triangle().solve(tree=True, mapped_variables=(mapping,), stabilize=False)


@pytest.mark.parametrize("mode", ["numerical", "certified"])
def test_native_plain_near_zero_admission_matches_physical_entries(mode):
    fixture = Path(__file__).resolve().parents[2] / "tests/fixtures/multi_network_native_tree.json"
    model = NetworkProblem.from_json(fixture.read_text())
    result = model.solve(tree=True, stabilize=False, bound_mode=mode)
    assert result.status == "Optimal"
    assert result.incumbent == pytest.approx(2.)
    assert result.bound == pytest.approx(2.)
    assert result.nodes == 6
    assert result.tree_processed_nodes == 7
    assert result.iterations == 17
    assert result.pricing_calls == 16
    assert dict(result.oracle_calls) == {7: 4, 42: 7}


@pytest.mark.parametrize("mode", ["numerical", "certified"])
def test_native_packing_registration_without_manual_mappings(mode):
    fixture = Path(__file__).resolve().parents[2] / "tests/fixtures/multi_network_native_packing.json"
    packing = json.loads(fixture.read_text())
    result = triangle().solve(tree=True, packing=packing, stabilize=False,
                              bound_mode=mode)
    assert result.status == "Optimal"
    assert result.incumbent == pytest.approx(2.)
    assert result.bound == pytest.approx(2.)
    assert result.tree_processed_nodes == 3
    assert result.strong_branching_trials == 6


def test_native_packing_requires_tree_and_rejects_endpoint_labels():
    packing = {"priority": 1000., "networks": [{"network": 7, "vertices": [[0, 1]]}]}
    with pytest.raises(ValueError, match="requires tree=true"):
        triangle().solve(packing=packing)
    with pytest.raises(ValueError, match="source and sink"):
        triangle().solve(tree=True, packing=packing, node_limit=0)


@pytest.mark.parametrize("mode", ["numerical", "certified"])
def test_native_packing_separator_is_bypassed_without_strong_branching(mode):
    fixture = Path(__file__).resolve().parents[2] / "tests/fixtures/multi_network_native_packing.json"
    packing = json.loads(fixture.read_text())
    model_fixture = fixture.with_name("multi_network_native_tree.json")
    result = NetworkProblem.from_json(model_fixture.read_text()).solve(
        tree=True, packing=packing, mapped_strong_branching=False,
        stabilize=False, bound_mode=mode)
    assert result.status == "Optimal"
    assert result.incumbent == pytest.approx(2.)
    assert result.bound == pytest.approx(2.)
    assert result.tree_processed_nodes == 7
    assert result.strong_branching_trials == 0


@pytest.mark.parametrize("mode", ["numerical", "certified"])
def test_finite_variable_box_matches_native_negative_path(mode):
    fixture = Path(__file__).resolve().parents[2] / "tests/fixtures/multi_network_native_negative_path.json"
    boxes = json.loads(fixture.with_name("multi_network_native_negative_boxes.json").read_text())
    result = NetworkProblem.from_json(fixture.read_text()).solve(
        stabilize=False, bound_mode=mode, variable_boxes=tuple(boxes))
    assert result.status == "Optimal"
    assert result.lp == pytest.approx(-1.)
    assert result.bound == pytest.approx(-1.)
    assert dict(result.oracle_calls) == {7: 1}

    unbounded = NetworkProblem.from_json(fixture.read_text()).solve(stabilize=False, bound_mode=mode)
    assert unbounded.bound == pytest.approx(-1.)
    assert dict(unbounded.oracle_calls) == {7: 3}


@pytest.mark.parametrize("mode", ["numerical", "certified"])
@pytest.mark.parametrize("fixture,boxes,objective,calls", [
    ("multi_network_native_positive_lower.json", "multi_network_native_positive_boxes.json", 0., 0),
    ("multi_network_native_positive_lower.json", "multi_network_native_negative_boxes.json", 0., 3),
    ("multi_network_native_cyclic_box.json", "multi_network_native_cyclic_boxes.json", -2., 1),
])
def test_positive_lower_and_cyclic_native_boxes(mode, fixture, boxes, objective, calls):
    folder = Path(__file__).resolve().parents[2] / "tests/fixtures"
    model = NetworkProblem.from_json((folder / fixture).read_text())
    result = model.solve(stabilize=False, bound_mode=mode,
                         variable_boxes=tuple(json.loads((folder / boxes).read_text())))
    assert result.status == "Optimal"
    assert result.lp == pytest.approx(objective)
    assert result.bound == pytest.approx(objective)
    assert dict(result.oracle_calls) == {7: calls}


@pytest.mark.parametrize("mode", ["numerical", "certified"])
@pytest.mark.parametrize("registered", [False, True])
def test_required_zero_cost_quantity_has_a_real_route_witness(mode, registered):
    folder = Path(__file__).resolve().parents[2] / "tests/fixtures"
    data = json.loads((folder / "multi_network_native_negative_path.json").read_text())
    data["networks"][0]["multiplicity"] = [1, 1]
    for arc in data["networks"][0]["arcs"]:
        arc["cost"] = 0.
    boxes = tuple(json.loads((folder / "multi_network_native_negative_boxes.json").read_text())) if registered else ()
    result = NetworkProblem.from_json(json.dumps(data)).solve(
        stabilize=False, bound_mode=mode, variable_boxes=boxes)
    assert result.status == "Optimal"
    assert result.lp == pytest.approx(0.)
    assert result.bound == pytest.approx(0.)
    assert dict(result.oracle_calls) == {7: 1}
    assert result.routes


@pytest.mark.parametrize("mode", ["numerical", "certified"])
def test_inactive_scan_option_applies_to_root_and_resumed_nodes(mode):
    for enabled in [False, True]:
        root = triangle().solve(stabilize=False, bound_mode=mode,
                                network_search_inactive_columns=enabled)
        assert root.status == "Optimal"
        assert root.lp == pytest.approx(1.5)
        child = root.snapshot.branch_assignment(0, 7, True)
        result = child.solve(stabilize=False, bound_mode=mode,
                             network_search_inactive_columns=enabled)
        assert result.status == "Optimal"
        assert result.lp == pytest.approx(2.)
        assert result.bound == pytest.approx(2.)


def test_original_variable_model_survives_forks_resume_fixed_and_tree():
    model = NetworkProblem.from_file(Path(__file__).resolve().parents[2] /
                                     "tests/fixtures/multi_network_original_variables.json")
    assert NetworkProblem.from_json(model.to_json()).pricing_variables
    root = model.solve(stabilize=False)
    assert root.lp == -1.
    assert root.original_variables == ({"network": 7, "id": 0, "lower": 1., "upper": 1.},)
    assert root.snapshot.solve(stabilize=False).lp == -1.
    route = next(r for r in root.routes if r.value > 0.5)
    fixed = root.snapshot.fix_route(route.route_index, 1).solve(stabilize=False)
    assert fixed.lp == -1.
    assert fixed.original_variables == root.original_variables
    child = root.snapshot.with_row(SharedRow("le", 0.), (
        NetworkRowCoefficients(7, (1., 0., 1., 0.)),))
    result = child.solve(stabilize=False)
    assert result.lp == 0.
    assert result.original_variables[0]["upper"] == 0.
    assert root.snapshot.solve(stabilize=False).lp == -1.
    assert model.solve(tree=True, stabilize=False).original_variables == root.original_variables
    assert root.snapshot.solve(tree=True, stabilize=False).original_variables == root.original_variables


def test_original_integer_registration_reaches_fresh_and_snapshot_trials():
    data = json.loads(FIXTURE.read_text())
    data["pricing_variables"] = [{"network": 7, "id": 4, "cost": 0.,
                                 "arcs": [[0, 1.]], "integer": True, "priority": 1000.}]
    model = NetworkProblem.from_json(json.dumps(data))
    root = model.solve(stabilize=False)
    assert root.original_variables[0]["lower"] == pytest.approx(0.5)
    for result in [model.solve(tree=True, network_branching="phased", stabilize=False),
                   root.snapshot.solve(tree=True, network_branching="phased", stabilize=False)]:
        assert result.incumbent == 2.
        assert any(e.name == "X_7_4" for e in result.mapped_branch_evaluations)
        activity = result.original_variables[0]
        assert activity["lower"] == activity["upper"]
        assert activity["lower"] in (0., 1.)


@pytest.mark.skipif(not __import__("pybaldes").has_highs, reason="HiGHS required")
def test_original_variable_registry_reports_enumerated_mip_activity():
    model = NetworkProblem.from_file(Path(__file__).resolve().parents[2] /
                                     "tests/fixtures/multi_network_original_variables.json")
    completed = model.solve(enumerate=True, stabilize=False)
    result = completed.snapshot.solve_enumerated_mip()
    assert result.incumbent == -1.
    assert result.original_variables == ({"network": 7, "id": 0, "lower": 1., "upper": 1.},)


@pytest.mark.parametrize("mode", ["numerical", "certified"])
def test_original_variable_boxes_survive_root_resume_and_tree(mode):
    model = NetworkProblem.from_file(Path(__file__).resolve().parents[2] /
                                     "tests/fixtures/multi_network_original_variables.json")
    boxes = ({"network": 7, "id": 0, "lower": 0., "upper": 1.},)
    plain = model.solve(stabilize=False, bound_mode=mode)
    boxed = model.solve(stabilize=False, bound_mode=mode, original_variable_boxes=boxes)
    assert boxed.lp == boxed.bound == -1.
    assert sum(c for _, c in boxed.oracle_calls) < sum(c for _, c in plain.oracle_calls)
    resumed = boxed.snapshot.solve(stabilize=False, bound_mode=mode, original_variable_boxes=boxes)
    assert resumed.lp == -1.
    assert resumed.original_variables == boxed.original_variables
    assert model.solve(tree=True, stabilize=False, bound_mode=mode,
                       original_variable_boxes=boxes).incumbent == -1.
    with pytest.raises(ValueError, match="unproved"):
        model.solve(time_limit=0., original_variable_boxes=({"network": 7, "id": 0, "upper": 0.5},))
    with pytest.raises(ValueError, match="identity"):
        model.solve(original_variable_boxes=({"network": 7, "id": 100, "upper": 1.},))


@pytest.mark.parametrize("mode", ["numerical", "certified"])
def test_original_variables_retain_cost_space_with_implicit_or_explicit_boxes(mode):
    model = NetworkProblem.from_file(Path(__file__).resolve().parents[2] /
                                     "tests/fixtures/multi_network_original_cancellation.json")
    boxes = tuple({"network": 7, "id": i, "upper": 1.} for i in [0, 1])
    implicit = model.solve(stabilize=False, bound_mode=mode)
    assert implicit.lp == implicit.bound == 0.
    assert implicit.oracle_calls == ((7, 3),)
    assert implicit.snapshot.solve(stabilize=False, bound_mode=mode).lp == 0.
    result = model.solve(stabilize=False, bound_mode=mode, original_variable_boxes=boxes)
    assert result.lp == 0.
    assert result.oracle_calls == ((7, 3),)


@pytest.mark.parametrize("mode", ["numerical", "certified"])
def test_signed_original_boxes_preserve_negative_activity_and_zero_upper(mode):
    data = json.loads((Path(__file__).resolve().parents[2] /
                       "tests/fixtures/multi_network_original_variables.json").read_text())
    variable = data["pricing_variables"][0]
    variable.update(cost=1., integer=False, arcs=[[0, -0.5], [2, -0.5]])
    model = NetworkProblem.from_json(json.dumps(data))
    for upper in [0., 4.]:
        boxes = ({"network": 7, "id": 0, "lower": -4., "upper": upper},)
        result = model.solve(stabilize=False, bound_mode=mode, original_variable_boxes=boxes)
        assert result.lp == result.bound == -0.5
        assert result.oracle_calls == ((7, 3),)
        assert result.original_variables == ({"network": 7, "id": 0,
                                              "lower": -0.5, "upper": -0.5},)
        resumed = result.snapshot.solve(stabilize=False, bound_mode=mode,
                                        original_variable_boxes=boxes)
        assert resumed.lp == resumed.bound == -0.5
        assert resumed.original_variables == result.original_variables


@pytest.mark.parametrize("mode", ["numerical", "certified"])
def test_original_cost_rounding_preserves_raw_objective_through_row_edits(mode):
    data = json.loads((Path(__file__).resolve().parents[2] /
                       "tests/fixtures/multi_network_original_variables.json").read_text())
    variable = data["pricing_variables"][0]
    variable.update(cost=-0.1234567893, integer=False, arcs=[[0, 0.5], [2, 0.5]])
    model = NetworkProblem.from_json(json.dumps(data))
    result = model.solve(stabilize=False, bound_mode=mode)
    raw = -0.1234567893 * 0.5
    assert result.lp == raw
    assert result.bound <= raw
    assert result.bound == pytest.approx(raw, abs=1e-12)
    assert result.original_variables[0]["lower"] == result.original_variables[0]["upper"] == 0.5
    edited = result.snapshot.with_row(
        SharedRow("le", 0.25),
        (NetworkRowCoefficients(7, (0.5, 0., 0.5, 0.)),),
    ).solve(stabilize=False, bound_mode=mode)
    assert edited.lp == pytest.approx(raw * 0.5, abs=1e-12)
    assert edited.bound <= edited.lp
    assert edited.bound == pytest.approx(edited.lp, abs=1e-12)
    assert edited.original_variables[0]["lower"] == edited.original_variables[0]["upper"] == 0.25

@pytest.mark.parametrize("bound_mode", ["numerical", "certified"])
def test_scalar_and_array_pricings_use_the_same_solve_path(bound_mode):
    network = PricingNetwork(7, 3, (0, 1), (PricingArc(0, 1, -2), PricingArc(1, 2, 0)))
    results = [solve(model, bound_mode=bound_mode, stabilize=False) for model in
               (network, [network], (network,), NetworkProblem((), network), NetworkProblem((), (network,)))]
    for result in results:
        assert result.status == "Optimal"
        assert result.lp == -2
        assert result.bound == -2
        assert result.oracle_calls == results[0].oracle_calls
        assert result.routes == results[0].routes
    second = replace(network, id=42, arcs=(PricingArc(0, 1, -3), PricingArc(1, 2, 0)))
    assert solve([network, second], bound_mode=bound_mode, stabilize=False).lp == -5


def test_modern_frontend_exports_one_solve_operation():
    import pybaldes
    assert callable(pybaldes.solve)
    assert callable(pybaldes._core._solve)
    for name in ("solve_root", "solve_networks"):
        assert not hasattr(pybaldes, name)
        assert not hasattr(pybaldes._core, "_" + name)


_ACTIVITY_MODELS = json.loads((Path(__file__).resolve().parents[2] /
                               "tests/fixtures/metasolver_registered_activities.json").read_text())["cases"]


@pytest.mark.parametrize("case", _ACTIVITY_MODELS, ids=lambda case: case["name"])
@pytest.mark.parametrize("mode", ["numerical", "certified"])
@pytest.mark.parametrize("singular", [False, True])
def test_registered_accumulated_variables_retain_intrinsic_activity(case, mode, singular):
    from copy import deepcopy
    from fractions import Fraction
    data = deepcopy(case["model"])
    if singular:
        data["networks"] = data["networks"][0]
    model = NetworkProblem.from_json(json.dumps(data))
    root = model.solve(stabilize=False, bound_mode=mode)
    assert root.status == "Optimal"
    assert root.lp == pytest.approx(case["oracle"]["objective_float"])
    theta = float(Fraction(case["oracle"]["fraction_first_path"]))
    expected = theta * case["activity_values_by_path"][0] + (1 - theta) * case["activity_values_by_path"][1]
    value = next(v for v in root.original_variables if v["id"] == 37)
    assert value["lower"] == pytest.approx(expected)
    assert value["upper"] == pytest.approx(expected)
    resumed = root.snapshot.solve(stabilize=False, bound_mode=mode)
    assert resumed.status == "Optimal"
    assert resumed.lp == pytest.approx(root.lp)
    assert next(v for v in resumed.original_variables if v["id"] == 37)["upper"] == pytest.approx(expected)


def test_explicit_activity_entries_roundtrip_in_the_common_network_model():
    from copy import deepcopy
    case = next(case for case in _ACTIVITY_MODELS if case["name"] == "custom-mixed-row")
    data = deepcopy(case["model"])
    data["networks"][0]["activity_entries"] = [{"resource": 1, "row": 0, "coefficient": 1.}]
    data["pricing_variables"][2]["entries"] = []
    model = NetworkProblem.from_json(json.dumps(data))
    assert model.networks[0].activity_entries
    restored = NetworkProblem.from_json(model.to_json())
    assert restored.networks[0].activity_entries == model.networks[0].activity_entries
    result = restored.solve(stabilize=False, bound_mode="certified")
    assert result.status == "Optimal"
    assert result.lp == pytest.approx(case["oracle"]["objective_float"])


@pytest.mark.parametrize("case", _ACTIVITY_MODELS, ids=lambda case: case["name"])
@pytest.mark.parametrize("mode", ["numerical", "certified"])
def test_original_accumulated_boxes_are_verified_intrinsic_envelopes(case, mode):
    model = NetworkProblem.from_json(json.dumps(case["model"]))
    lower, upper = min(case["activity_values_by_path"]), max(case["activity_values_by_path"])
    box = {"network": 42, "id": 37, "lower": lower, "upper": upper}
    root = model.solve(stabilize=False, bound_mode=mode, original_variable_boxes=[box])
    assert root.status == "Optimal"
    assert root.lp == pytest.approx(case["oracle"]["objective_float"])
    resumed = root.snapshot.solve(stabilize=False, bound_mode=mode)
    assert resumed.status == "Optimal"
    assert resumed.lp == pytest.approx(root.lp)
    with pytest.raises(ValueError, match="envelope is unproved"):
        model.solve(stabilize=False, bound_mode=mode,
                    original_variable_boxes=[dict(box, upper=upper - 1)])


@pytest.mark.parametrize("name, expected", [("custom-mixed-row", -9.), ("cumulative-mixed-row", -10.)])
@pytest.mark.parametrize("mode", ["numerical", "certified"])
def test_integer_accumulated_registration_reaches_the_shared_tree(name, expected, mode):
    from copy import deepcopy
    case = next(case for case in _ACTIVITY_MODELS if case["name"] == name)
    data = deepcopy(case["model"])
    data["pricing_variables"][2]["integer"] = True
    data["pricing_variables"][2]["priority"] = 100.
    model = NetworkProblem.from_json(json.dumps(data))
    root = model.solve(stabilize=False, bound_mode=mode)
    assert root.lp == pytest.approx(case["oracle"]["objective_float"])
    for result in (model.solve(tree=True, stabilize=False, bound_mode=mode),
                   root.snapshot.solve(tree=True, stabilize=False, bound_mode=mode)):
        assert result.status == "Optimal"
        assert result.incumbent == pytest.approx(expected)
        value = next(v for v in result.original_variables if v["id"] == 37)
        assert value["lower"] == value["upper"] == case["activity_values_by_path"][0]


_PREPARED_RESOURCE_MODELS = json.loads((Path(__file__).resolve().parents[2] /
    "tests/fixtures/coupled/metasolver-paired-preparation.json").read_text())["cases"]
_PREPARED_RESOURCE_MODELS += json.loads((Path(__file__).resolve().parents[2] /
    "tests/fixtures/coupled/metasolver-paired-pwl-preparation.json").read_text())["cases"]
_PREPARED_RESOURCE_MODELS += json.loads((Path(__file__).resolve().parents[2] /
    "tests/fixtures/coupled/metasolver-paired-feasibility-preparation.json").read_text())["cases"]
_PREPARED_RESOURCE_MODELS += json.loads((Path(__file__).resolve().parents[2] /
    "tests/fixtures/coupled/metasolver-spd-preparation.json").read_text())["cases"]
_PREPARED_RESOURCE_MODELS += json.loads((Path(__file__).resolve().parents[2] /
    "tests/fixtures/coupled/metasolver-fictive-preparation.json").read_text())["cases"]


@pytest.mark.parametrize("case", _PREPARED_RESOURCE_MODELS, ids=lambda case: case["name"])
@pytest.mark.parametrize("mode", ["numerical", "certified"])
def test_actual_resource_factory_models_keep_original_column_bounds(case, mode):
    model = NetworkProblem.from_json(json.dumps(case["model"]))
    result = model.solve(stabilize=False, bound_mode=mode)
    expected = case["independent_objective"]
    assert result.status == "Optimal"
    assert result.lp == pytest.approx(expected)
    assert result.bound <= expected
    assert result.bound == pytest.approx(expected)


def test_selected_paired_values_survive_python_output_and_resume():
    case = _PREPARED_RESOURCE_MODELS[0]
    model = NetworkProblem.from_json(json.dumps(case["model"]))
    result = model.solve(stabilize=False)
    resumed = result.snapshot.solve(stabilize=False)
    for solved in (result, resumed):
        assert solved.routes
        for route in solved.routes:
            expected = 1.0 if len(route.arcs) == 1 else 3.0
            assert route.coupled_values == ((0, expected),)


@pytest.mark.parametrize("representation", ["raw", "control_point"])
@pytest.mark.parametrize("case_index", [0, 18])
@pytest.mark.parametrize("coefficient", [0.0, 0.5, 1.0, 2.0, 8.0])
@pytest.mark.parametrize("mode", ["numerical", "certified"])
def test_paired_original_variable_reports_selected_value_with_nonnegative_weight(case_index, coefficient, mode, representation):
    selected_case = (36 if case_index == 0 else 39) if representation == "control_point" else case_index
    data = json.loads(json.dumps(_PREPARED_RESOURCE_MODELS[selected_case]["model"]))
    final_value = 3.0
    if representation == "control_point":
        data["networks"][0]["front_program_domain"] = False
        resource = data["networks"][0]["coupled_resources"][0]
        def knots(function):
            value, start = function["pairs"][0]
            points = [[start, value]]
            for slope, end in function["pairs"][1:]:
                value += slope * (end - start)
                start = end
                points.append([start, value])
            return {"points": points}
        resource["initial"] = knots(resource["initial"])
        for arc in resource["arcs"]:
            for key in ("arrival", "increment", "lower", "upper"):
                arc[key] = knots(arc[key])
        resource["kind"] = "paired_piecewise_linear"
        del resource["policy"]
        final_value = 5.0
    data["pricing_variables"] = [{"network": 42, "id": 901, "cost": coefficient,
                                  "accumulated_resource": 0}]
    model = NetworkProblem.from_json(json.dumps(data))
    root = model.solve(stabilize=False, bound_mode=mode)
    expected = min(0.0, -20.0 + final_value * coefficient)
    gamma = final_value if expected < 0 else 0.0
    for result in (root, root.snapshot.solve(stabilize=False, bound_mode=mode),
                   model.solve(tree=True, stabilize=False, bound_mode=mode)):
        assert result.original_variables == ({"network": 42, "id": 901,
                                               "lower": gamma, "upper": gamma},)
    assert root.lp == pytest.approx(expected)
    if expected < 0:
        route = next(r for r in root.routes if r.value > 0)
        fixed = root.snapshot.fix_route(route.route_index, 1).solve(stabilize=False, bound_mode=mode)
        assert fixed.original_variables == root.original_variables


@pytest.mark.parametrize("case_index", [0, 18])
@pytest.mark.parametrize("coefficient", [0.5, 2.0])
@pytest.mark.parametrize("bound_mode", ["numerical", "certified"])
def test_explicit_route_controls_survive_pool_resume_fixing_and_original_reporting(case_index, coefficient, bound_mode):
    from pybaldes import PairedControl
    data = json.loads(json.dumps(_PREPARED_RESOURCE_MODELS[case_index]["model"]))
    data["networks"][0]["front_program_domain"] = False
    data["networks"][0]["coupled_resources"][0]["policy"] = "strict_domain"
    data["pricing_variables"] = [{"network": 42, "id": 901, "cost": coefficient,
                                  "accumulated_resource": 0}]
    model = NetworkProblem.from_json(json.dumps(data))
    root = model.solve(stabilize=False, bound_mode=bound_mode)
    assert root.status == "Optimal"
    original = next(route for route in root.routes if len(route.arcs) == 2)
    generated = root.snapshot.optimize_route_control(original.route_index, 0, terminal=(4.0, 4.0), maximize=True)
    assert generated.complete and generated.maximize
    assert generated.work_used > 0 and generated.lp_iterations > 0
    assert generated.dependent == 4.0 and generated.lower_bound <= -4.0
    control = generated.control
    assert isinstance(control, PairedControl)
    maximum = root.snapshot.optimize_route_control(original.route_index, 0, maximize=True)
    assert maximum.complete and maximum.dependent == 8.0
    incomplete = root.snapshot.optimize_route_control(original.route_index, 0, max_cells=0)
    assert not incomplete.complete and incomplete.control is None
    empty = root.snapshot.optimize_route_control(original.route_index, 0, terminal=(9.0, 9.0))
    assert empty.complete and empty.control is None
    expanded = root.snapshot.with_route_controls(original.route_index, (control,)).solve(
        stabilize=False, bound_mode=bound_mode)
    assert expanded.lp == pytest.approx(-20 + 3*coefficient)
    alternative = next(route for route in expanded.routes if route.coupled_controls)
    assert alternative.coupled_controls == (control,)
    assert isinstance(alternative.coupled_controls[0].base, tuple)
    assert alternative.coupled_values == ((0, 4.0),)
    assert alternative.cost == -20 + 4*coefficient
    fixed = expanded.snapshot.fix_route(alternative.route_index).solve(stabilize=False, bound_mode=bound_mode)
    assert fixed.status == "Optimal"
    assert fixed.lp == pytest.approx(alternative.cost)
    assert fixed.fixed_routes[0].coupled_controls == (control,)
    assert fixed.original_variables == ({"network": 42, "id": 901, "lower": 4.0, "upper": 4.0},)
    resumed = fixed.snapshot.solve(stabilize=False, bound_mode=bound_mode)
    assert resumed.original_variables == fixed.original_variables
    assert resumed.fixed_routes[0].coupled_controls == (control,)
    tree = fixed.snapshot.solve(tree=True, stabilize=False, bound_mode=bound_mode)
    assert tree.status == "Optimal"
    assert tree.incumbent == pytest.approx(alternative.cost)
    assert tree.original_variables == fixed.original_variables
    assert tree.routes[0].coupled_controls == (control,)
    with pytest.raises(ValueError, match="infeasible or uncertified"):
        root.snapshot.with_route_controls(original.route_index, (PairedControl(0, (0.5, 1.0, 2.0)),))


@pytest.mark.parametrize("case_index", [0, 18])
@pytest.mark.parametrize("bound_mode", ["numerical", "certified"])
def test_paired_shared_rows_generate_and_retain_interior_controls(case_index, bound_mode):
    data = json.loads(json.dumps(_PREPARED_RESOURCE_MODELS[case_index]["model"]))
    network = data["networks"][0]
    network["front_program_domain"] = False
    network["coupled_resources"][0]["policy"] = "strict_domain"
    data["rows"] = [{"sense": "ge", "rhs": 4.0}]
    data["pricing_variables"] = [{"network": 42, "id": 901, "cost": 1.0,
                                  "accumulated_resource": 0, "entries": [[0, 1.0]]}]
    root = NetworkProblem.from_json(json.dumps(data)).solve(stabilize=False, bound_mode=bound_mode)
    assert root.status == "Optimal"
    assert root.lp == pytest.approx(-16.0)
    assert root.bound <= -16.0 + 1e-10
    assert root.bound == pytest.approx(-16.0)
    assert root.original_variables[0]["lower"] == pytest.approx(4.0)
    alternative = next(r for r in root.routes if r.coupled_values == ((0, 4.0),) and len(r.arcs) == 2)
    assert alternative.coupled_controls
    fixed = root.snapshot.fix_route(alternative.route_index).solve(stabilize=False, bound_mode=bound_mode)
    assert fixed.status == "Optimal"
    assert fixed.lp == pytest.approx(-16.0)
    assert fixed.original_variables[0]["lower"] == pytest.approx(4.0)
    resumed = root.snapshot.solve(stabilize=False, bound_mode=bound_mode)
    assert resumed.lp == pytest.approx(-16.0)


@pytest.mark.parametrize("representation", ["slope", "points"])
@pytest.mark.parametrize("bound_mode", ["numerical", "certified"])
@pytest.mark.parametrize("upper", [None, 4.0])
def test_negative_paired_objectives_price_controls_and_preserve_fixed_certificates(representation, bound_mode, upper):
    data = json.loads(json.dumps(_PREPARED_RESOURCE_MODELS[0]["model"]))
    network = data["networks"][0]
    network["front_program_domain"] = False
    resource = network["coupled_resources"][0]
    resource["policy"] = "strict_domain"
    if representation == "points":
        def knots(function):
            pairs = function["pairs"]
            value, start = pairs[0]
            points = [[start, value]]
            for slope, end in pairs[1:]:
                value += slope * (end - start)
                start = end
                points.append([start, value])
            return {"points": points}
        resource["initial"] = knots(resource["initial"])
        for arc in resource["arcs"]:
            for key in ("arrival", "increment", "lower", "upper"):
                arc[key] = knots(arc[key])
        resource["kind"] = "paired_piecewise_linear"
        del resource["policy"]
    variable = {"network": 42, "id": 901, "cost": -1.0, "accumulated_resource": 0}
    if upper is not None:
        data["rows"] = [{"sense": "le", "rhs": upper}]
        variable["entries"] = [[0, 1.0]]
    data["pricing_variables"] = [variable]
    root = NetworkProblem.from_json(json.dumps(data)).solve(stabilize=False, bound_mode=bound_mode)
    gamma = upper if upper is not None else 8.0
    expected = -20.0 - gamma
    assert root.status == "Optimal"
    assert root.lp == pytest.approx(expected)
    assert root.bound <= expected + 1e-10
    assert root.bound == pytest.approx(expected)
    assert root.original_variables[0]["lower"] == pytest.approx(gamma)
    route = next(r for r in root.routes if len(r.arcs) == 2 and r.coupled_values == ((0, gamma),))
    assert route.coupled_controls
    fixed = root.snapshot.fix_route(route.route_index).solve(stabilize=False, bound_mode=bound_mode)
    assert fixed.status == "Optimal"
    assert fixed.lp == pytest.approx(expected)
    assert fixed.bound == pytest.approx(expected)
    assert fixed.original_variables[0]["lower"] == pytest.approx(gamma)


@pytest.mark.parametrize("case_index", [0, 18])
@pytest.mark.parametrize("bound_mode", ["numerical", "certified"])
@pytest.mark.parametrize("cost", [-1.0, 1.0])
def test_paired_boxes_enclose_alternative_controls(case_index, bound_mode, cost):
    data = json.loads(json.dumps(_PREPARED_RESOURCE_MODELS[case_index]["model"]))
    network = data["networks"][0]
    network["front_program_domain"] = False
    network["coupled_resources"][0]["policy"] = "strict_domain"
    data["pricing_variables"] = [{"network": 42, "id": 901, "cost": cost, "accumulated_resource": 0}]
    model = NetworkProblem.from_json(json.dumps(data))
    box = {"network": 42, "id": 901, "lower": 0.0, "upper": 8.0}
    root = model.solve(stabilize=False, bound_mode=bound_mode, original_variable_boxes=[box])
    expected = -20.0 + cost * (8.0 if cost < 0 else 3.0)
    assert root.status == "Optimal"
    assert root.lp == pytest.approx(expected)
    assert root.bound == pytest.approx(expected)
    assert root.snapshot.solve(stabilize=False, bound_mode=bound_mode).lp == pytest.approx(expected)
    with pytest.raises(ValueError, match="envelope is unproved"):
        model.solve(stabilize=False, bound_mode=bound_mode, original_variable_boxes=[dict(box, upper=3.0)])


@pytest.mark.parametrize("case_index", [0, 4])
@pytest.mark.parametrize("bound_mode", ["numerical", "certified"])
@pytest.mark.parametrize("cost", [-1.0, 0.0])
def test_integer_paired_bindings_reach_fresh_and_snapshot_trees(case_index, bound_mode, cost):
    cases = json.loads((Path(__file__).resolve().parents[2] /
        "tests/fixtures/coupled/metasolver-paired-objective-mapping.json").read_text())["cases"]
    data = json.loads(json.dumps(cases[case_index]["model"]))
    for i, arc in enumerate(data["networks"][0]["arcs"]):
        arc["cost"] = 100.0 if i == 0 else (20.0 if i in (1, 3) else 0.0)
        if i in (1, 3):
            arc["entries"] = [[0, 1.0]]
    data["rows"] = [{"sense": "ge", "rhs": 0.25}]
    data["pricing_variables"] = [{"network": 42, "id": 901, "cost": cost,
                                  "accumulated_resource": 0, "integer": True, "priority": 100.0}]
    model = NetworkProblem.from_json(json.dumps(data))
    root = model.solve(stabilize=False, bound_mode=bound_mode)
    assert root.status == "Optimal"
    assert root.lp == pytest.approx(5.0 + 0.5 * cost)
    assert root.original_variables[0]["lower"] == pytest.approx(0.5)
    for tree in (model.solve(tree=True, stabilize=False, bound_mode=bound_mode, network_branching="resource-first"),
                 root.snapshot.solve(tree=True, stabilize=False, bound_mode=bound_mode, network_branching="resource-first")):
        assert tree.status == "Optimal"
        assert tree.incumbent == pytest.approx(20.0 + 2.0 * cost)
        assert tree.original_variables[0]["lower"] == pytest.approx(2.0)


def test_paired_integer_endpoints_do_not_authorize_continuous_activity_registration():
    data = json.loads(json.dumps(_PREPARED_RESOURCE_MODELS[0]["model"]))
    data["networks"][0]["front_program_domain"] = False
    data["networks"][0]["coupled_resources"][0]["policy"] = "strict_domain"
    data["pricing_variables"] = [{"network": 42, "id": 901, "cost": 0.0,
                                  "accumulated_resource": 0, "integer": True}]
    with pytest.raises(ValueError, match="exactly integer"):
        NetworkProblem.from_json(json.dumps(data)).solve()


@pytest.mark.parametrize("case_index", [0, 4])
@pytest.mark.parametrize("bound_mode", ["numerical", "certified"])
@pytest.mark.parametrize("cost", [-1.0, 1.0])
def test_invariant_paired_controls_have_complete_enumerated_snapshots(case_index, bound_mode, cost):
    cases = json.loads((Path(__file__).resolve().parents[2] /
        "tests/fixtures/coupled/metasolver-paired-objective-mapping.json").read_text())["cases"]
    data = json.loads(json.dumps(cases[case_index]["model"]))
    data["rows"] = [{"sense": "ge", "rhs": 1.0}]
    data["pricing_variables"] = [{"network": 42, "id": 901, "cost": cost,
                                  "accumulated_resource": 0, "entries": [[0, 1.0]]}]
    parent = NetworkProblem.from_json(json.dumps(data)).solve(stabilize=False, bound_mode=bound_mode)
    complete = parent.snapshot.solve(enumerate=True, stabilize=False, bound_mode=bound_mode)
    assert complete.status == "Optimal"
    assert complete.enumeration_reports[0].complete
    assert complete.enumeration_reports[0].routes == 3
    assert complete.lp == pytest.approx(-20.0 + 2 * cost)
    assert complete.bound == pytest.approx(complete.lp)
    route = next(r for r in complete.routes if len(r.arcs) == 2)
    assert route.coupled_controls
    fixed = complete.snapshot.fix_route(route.route_index).solve(stabilize=False, bound_mode=bound_mode)
    assert fixed.status == "Optimal"
    assert fixed.lp == pytest.approx(complete.lp)
    assert fixed.original_variables[0]["lower"] == 2.0
    assert parent.enumeration_reports[0].attempts == 0


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("cost", [-1.0, 1.0])
def test_shared_python_row_api_adds_paired_activities_to_complete_universes(dynamic, cost):
    cases = json.loads((Path(__file__).resolve().parents[2] /
        "tests/fixtures/coupled/metasolver-paired-objective-mapping.json").read_text())["cases"]
    data = json.loads(json.dumps(cases[0]["model"]))
    data["pricing_variables"][0]["cost"] = cost
    root = NetworkProblem.from_json(json.dumps(data)).solve(stabilize=False, enumerate=True, bound_mode="certified")
    assert root.enumeration_reports[0].complete
    coefficients = (NetworkRowCoefficients(42, (0.5,) * 5, per_route=1.0, activities=((0, 1.0),)),)
    child = root.snapshot.with_row(SharedRow("ge", 4.0), coefficients, dynamic=dynamic).solve(
        stabilize=False, bound_mode="certified")
    assert child.status == "Optimal"
    assert child.lp == pytest.approx(-20.0 + 2 * cost)
    assert child.enumeration_reports[0].complete
    impossible = root.snapshot.with_row(SharedRow("ge", 5.0), coefficients, dynamic=dynamic).solve(
        stabilize=False, bound_mode="certified")
    assert impossible.status == "Infeasible"
    assert root.snapshot.solve(stabilize=False).lp == pytest.approx(root.lp)


def test_python_paired_activity_rows_reject_widening_a_continuous_universe():
    data = json.loads(json.dumps(_PREPARED_RESOURCE_MODELS[0]["model"]))
    data["networks"][0]["front_program_domain"] = False
    data["networks"][0]["coupled_resources"][0]["policy"] = "strict_domain"
    root = NetworkProblem.from_json(json.dumps(data)).solve(stabilize=False, enumerate=True)
    assert root.enumeration_reports[0].complete
    coefficients = (NetworkRowCoefficients(42, (0.0,) * 5, activities=((0, 1.0),)),)
    with pytest.raises(ValueError, match="cannot widen"):
        root.snapshot.with_row(SharedRow("ge", 4.0), coefficients)
    assert root.snapshot.solve(stabilize=False).lp == pytest.approx(-17.0)

@pytest.mark.parametrize("representation", ["slope", "points"])
@pytest.mark.parametrize("bound_mode", ["numerical", "certified"])
@pytest.mark.parametrize("singular", [False, True])
def test_network_graph_coefficients_preserve_relative_objectives_across_frontends(representation, bound_mode, singular):
    from pathlib import Path
    data = json.loads(Path("tests/fixtures/coupled/negative-relative-network.json").read_text())
    network = data["networks"][0]
    if representation == "points":
        def knots(function):
            value, start = function["pairs"][0]
            points = [[start, value]]
            for slope, end in function["pairs"][1:]:
                value += slope * (end - start)
                start = end
                points.append([start, value])
            return {"points": points}
        resource = network["coupled_resources"][0]
        resource["initial"] = knots(resource["initial"])
        for arc in resource["arcs"]:
            for key in ("arrival", "increment", "lower", "upper"):
                arc[key] = knots(arc[key])
        resource["kind"] = "paired_piecewise_linear"
        del resource["policy"]
    if singular:
        data["networks"] = network
    problem = NetworkProblem.from_json(json.dumps(data))
    assert problem.networks[0].coupled_objective_coefficients == ((0, -1.0),)
    restored = NetworkProblem.from_json(problem.to_json())
    assert restored == problem
    root = restored.solve(stabilize=False, bound_mode=bound_mode)
    assert root.status == "Optimal"
    assert root.lp == pytest.approx(-26.0)
    assert root.bound == pytest.approx(-26.0)
    assert root.snapshot.solve(stabilize=False, bound_mode=bound_mode).lp == pytest.approx(-26.0)
    route = next(r for r in root.routes if r.coupled_values == ((0, 8.0),) and len(r.arcs) == 2)
    assert route.coupled_controls
    fixed = root.snapshot.fix_route(route.route_index).solve(stabilize=False, bound_mode=bound_mode)
    assert fixed.status == "Optimal"
    assert fixed.lp == pytest.approx(-26.0)
    tree = root.snapshot.solve(tree=True, stabilize=False, bound_mode=bound_mode)
    assert tree.status == "Optimal"
    assert tree.incumbent == pytest.approx(-26.0)


@pytest.mark.parametrize("coefficients", [((0, -1.), (0, -2.)), ((9, 1.),)])
def test_network_graph_coefficient_validation_uses_the_graph_contract(coefficients):
    from pathlib import Path
    data = json.loads(Path("tests/fixtures/coupled/negative-relative-network.json").read_text())
    data["networks"][0]["coupled_objective_coefficients"] = coefficients
    with pytest.raises(ValueError, match="invalid accumulated objective coefficient"):
        NetworkProblem.from_json(json.dumps(data)).solve()


def test_network_graph_coefficients_reject_two_cost_owners():
    from pathlib import Path
    data = json.loads(Path("tests/fixtures/coupled/negative-relative-network.json").read_text())
    data["pricing_variables"] = [dict(network=42, id=901, cost=-1., accumulated_resource=0)]
    with pytest.raises(ValueError, match="not both"):
        NetworkProblem.from_json(json.dumps(data)).solve()

@pytest.mark.parametrize("points", [False, True])
@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("bound_mode", ["numerical", "certified"])
def test_objective_only_control_endpoints_survive_complete_transfer(points, relative, bound_mode):
    from pathlib import Path
    data = json.loads(Path("tests/fixtures/coupled/negative-relative-network.json").read_text())
    resource = data["networks"][0]["coupled_resources"][0]
    if not relative:
        resource["normalization"] = "absolute_final_value"
    if points:
        def knots(function):
            value, start = function["pairs"][0]
            result = [[start, value]]
            for slope, end in function["pairs"][1:]:
                value += slope * (end - start)
                start = end
                result.append([start, value])
            return {"points": result}
        resource["initial"] = knots(resource["initial"])
        for arc in resource["arcs"]:
            for key in ("arrival", "increment", "lower", "upper"):
                arc[key] = knots(arc[key])
        resource["kind"] = "paired_piecewise_linear"
        del resource["policy"]
    root = NetworkProblem.from_json(json.dumps(data)).solve(stabilize=False, bound_mode=bound_mode)
    complete = root.snapshot.solve(enumerate=True, stabilize=False, bound_mode=bound_mode)
    assert complete.enumeration_reports[0].complete
    assert complete.enumeration_reports[0].routes == 3
    expected = -26. if relative else -28.
    assert complete.status == "Optimal"
    assert complete.lp == pytest.approx(expected)
    assert complete.bound == pytest.approx(expected)
    route = next(r for r in complete.routes if r.coupled_values == ((0, 8.),) and len(r.arcs) == 2)
    fixed = complete.snapshot.fix_route(route.route_index).solve(stabilize=False, bound_mode=bound_mode)
    assert fixed.status == "Optimal"
    assert fixed.lp == pytest.approx(expected)
    if not relative:
        from pybaldes import SharedRow, NetworkRowCoefficients
        with pytest.raises(ValueError, match="paired rows cannot widen"):
            complete.snapshot.with_row(SharedRow("le", 4.), [
                NetworkRowCoefficients(42, (0.,) * len(data["networks"][0]["arcs"]), activities=((0, 1.),))])

@pytest.mark.parametrize("points", [False, True])
@pytest.mark.parametrize("cost,sense,coefficient,rhs,gamma", [
    (-1., "ge", 1., 5., 8.), (1., "ge", -1., -4., 3.),
    (0., "ge", 1., 5., 8.),
])
def test_monotone_paired_rows_preserve_feasibility_through_endpoint_transfer(points, cost, sense, coefficient, rhs, gamma):
    data = json.loads(json.dumps(_PREPARED_RESOURCE_MODELS[0]["model"]))
    network = data["networks"][0]
    network["front_program_domain"] = False
    resource = network["coupled_resources"][0]
    resource["policy"] = "strict_domain"
    if points:
        def knots(function):
            value, start = function["pairs"][0]
            result = [[start, value]]
            for slope, end in function["pairs"][1:]:
                value += slope * (end - start)
                start = end
                result.append([start, value])
            return {"points": result}
        resource["initial"] = knots(resource["initial"])
        for arc in resource["arcs"]:
            for key in ("arrival", "increment", "lower", "upper"):
                arc[key] = knots(arc[key])
        resource["kind"] = "paired_piecewise_linear"
        del resource["policy"]
    data["rows"] = [dict(sense=sense, rhs=rhs)]
    data["pricing_variables"] = [dict(network=42, id=901, cost=cost,
        accumulated_resource=0, entries=[[0, coefficient]])]
    root = NetworkProblem.from_json(json.dumps(data)).solve(stabilize=False, bound_mode="certified")
    complete = root.snapshot.solve(enumerate=True, stabilize=False, bound_mode="certified")
    assert complete.enumeration_reports[0].complete
    assert complete.status == "Optimal"
    assert complete.lp == pytest.approx(-20. + cost * gamma)
    assert complete.bound == pytest.approx(complete.lp)
    assert complete.original_variables[0]["lower"] == pytest.approx(gamma)

@pytest.mark.parametrize("cost,sense,rhs,gamma", [(-1., "ge", 14., 8.), (1., "le", 7., 3.)])
@pytest.mark.parametrize("dynamic", [False, True])
def test_compatible_new_activity_rows_reuse_projected_controls_and_fixed_copies(cost, sense, rhs, gamma, dynamic):
    from pybaldes import SharedRow, NetworkRowCoefficients
    data = json.loads(json.dumps(_PREPARED_RESOURCE_MODELS[0]["model"]))
    network = data["networks"][0]
    network["front_program_domain"] = False
    network["multiplicity"] = [0.,2.]
    network["coupled_resources"][0]["policy"] = "strict_domain"
    data["pricing_variables"] = [dict(network=42,id=901,cost=cost,accumulated_resource=0)]
    root = NetworkProblem.from_json(json.dumps(data)).solve(stabilize=False, bound_mode="certified", enumerate=True)
    assert root.enumeration_reports[0].complete
    terms = [NetworkRowCoefficients(42, (0.,) * len(data["networks"][0]["arcs"]), activities=((0,1.),))]
    child = root.snapshot.with_row(SharedRow(sense,rhs), terms, dynamic=dynamic).solve(stabilize=False, bound_mode="certified")
    expected = 2. * (-20. + cost * gamma)
    assert child.status == "Optimal"
    assert child.lp == pytest.approx(expected)
    assert child.enumeration_reports[0].complete
    route = next(r for r in child.routes if r.coupled_values == ((0,gamma),) and len(r.arcs) == 2)
    fixed = child.snapshot.fix_route(route.route_index)
    repeated = fixed.with_row(SharedRow(sense,rhs), terms, dynamic=dynamic).solve(stabilize=False, bound_mode="certified")
    assert repeated.status == "Optimal"
    assert repeated.lp == pytest.approx(expected)
    assert repeated.enumeration_reports[0].complete
    with pytest.raises(ValueError, match="paired rows cannot widen"):
        root.snapshot.with_row(SharedRow("eq",2.*gamma), terms, dynamic=dynamic)
    assert root.snapshot.solve(stabilize=False).lp == pytest.approx(expected)


@pytest.mark.parametrize("bound_mode", ["numerical", "certified"])
@pytest.mark.parametrize("shape", ["one", "array", "multiple"])
@pytest.mark.parametrize("copies", [1, 2])
def test_joint_paired_rows_recover_interior_integer_witnesses(bound_mode, shape, copies):
    data = json.loads((Path(__file__).resolve().parents[2] /
        "tests/fixtures/coupled/joint-interior-model.json").read_text())
    data["networks"][0]["multiplicity"] = [copies, copies]
    if shape == "one":
        data["networks"] = data["networks"][0]
    elif shape == "multiple":
        data["networks"].append({"id": 7, "vertices": 2, "source": 0, "sink": 1,
            "multiplicity": [1, 1], "arcs": [{"tail": 0, "head": 1, "cost": 2.}]})
    model = NetworkProblem.from_json(json.dumps(data))
    root = model.solve(stabilize=False, bound_mode=bound_mode)
    expected = -10. * copies + (2. if shape == "multiple" else 0.)
    for tree in (model.solve(tree=True, stabilize=False, bound_mode=bound_mode),
                 root.snapshot.solve(tree=True, stabilize=False, bound_mode=bound_mode)):
        assert tree.status == "Optimal"
        assert tree.incumbent == expected
        assert tree.bound <= expected and tree.bound == pytest.approx(expected)
        primary = next(route for route in tree.routes if route.network == 42)
        assert primary.value == copies
        assert primary.coupled_values == ((0, 4. / copies), (1, 4. / copies))
        assert len(primary.coupled_controls) == 2
