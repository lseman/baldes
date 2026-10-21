import gc
import json
from dataclasses import replace
from pathlib import Path

import pytest

from pybaldes import (
    NetworkProblem, NetworkRowCoefficients, PricingArc, PricingNetwork,
    PricingResource, SharedRow, has_clp, solve_networks,
)

pytestmark = pytest.mark.skipif(not has_clp, reason="Clp build required")
FIXTURE = Path(__file__).resolve().parents[2] / "examples/networks/triangle.json"


def triangle():
    return NetworkProblem.from_file(FIXTURE)


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
    result = solve_networks(triangle(), tree=True, bound_mode=bound_mode)
    assert result.status == "Optimal" and result.mode == "tree"
    assert result.incumbent == pytest.approx(2)
    assert result.bound == pytest.approx(2)
    assert result.nodes == 3 and result.open_nodes == 0
    assert result.pricing_calls > 0
    assert {n for n, calls in result.oracle_calls if calls} == {7, 42}
    assert result.snapshot is None
    assert all(r.value.is_integer() for r in result.routes)
    limited = triangle().solve(tree=True, node_limit=1)
    assert limited.status == "NodeLimit" and limited.open_nodes == 2
    assert limited.incumbent is None and limited.bound == pytest.approx(1.5)
    zero = triangle().solve(tree=True, node_limit=0)
    assert zero.bound is None and json.loads(zero.to_json())["bound"] is None


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
    with pytest.raises(ValueError):
        node.solve(tree=True)
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
