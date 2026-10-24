"""Explicit function encodings; runtime execution deferred during implementation."""
import json
import math

import pytest

from pybaldes import EncodedFunction, NetworkProblem, ResourceModel, Solver, has_clp


def test_prepared_slope_encoding_preserves_fractional_literal_arithmetic():
    x0, y0 = 61.42565465487604, 7.245618290940143
    slope, x1, argument = 9.462315279587411, 68.72567520394728, 64.1889633870269
    raw = EncodedFunction.slope_segments([(y0, x0), (slope, x1)])
    endpoint = y0 + slope * (x1 - x0)
    knots = EncodedFunction.knots([(x0, y0), (x1, endpoint)])
    assert raw.value(argument) == 33.39291672938855
    assert knots.value(argument) == 33.39291672938856
    assert raw.value(argument) != knots.value(argument)
    assert raw.encoding == "slope_segments" and raw.policy == "strict_domain"
    assert knots.encoding == "knots" and knots.policy is None


def test_affine_preparation_and_boundary_policy_are_explicit():
    affine = EncodedFunction.affine(2.0, 3.0, 0.0, 10.0)
    assert affine.value(0.0) == 2.0
    assert affine.value(0.25) == 2.75
    assert affine.value(10.0) == 32.0
    assert affine.value(-0.00001) is None
    tolerant = EncodedFunction.affine(2.0, 3.0, 0.0, 10.0, "native_tolerance")
    assert tolerant.value(-0.00001) == 2.0
    assert tolerant.value(10.00001) == 32.0
    assert tolerant.value(10.001) is None
    singleton = EncodedFunction.slope_segments([(1e12, 3.0)])
    assert singleton.value(3.0) == 1e12
    assert singleton.value(3.1) is None


@pytest.mark.parametrize("function", [
    lambda: EncodedFunction.knots([(0.0, 1.0), (2.0, 3.0)]),
    lambda: EncodedFunction.slope_segments([(2.0, 0.0), (3.0, 5.0), (-1.0, 10.0)]),
    lambda: EncodedFunction.affine(2.0, 3.0, 0.0, 10.0, "native_tolerance"),
])
def test_json_compatible_definition_round_trip_retains_encoding_and_policy(function):
    original = function()
    wire = json.loads(json.dumps(original.to_dict(), allow_nan=False))
    restored = EncodedFunction.from_dict(wire)
    assert restored.to_dict() == original.to_dict()
    assert restored.encoding == original.encoding
    assert restored.policy == original.policy
    for x in [-0.00001, 0.0, 0.125, 1.0, 2.0, 5.0, 10.0, 10.00001]:
        assert restored.value(x) == original.value(x)
    if original.policy == "native_tolerance":
        assert wire["policy"] == "native_tolerance"


@pytest.mark.parametrize("build", [
    lambda: EncodedFunction.knots([(0.0, 0.0)]),
    lambda: EncodedFunction.slope_segments([]),
    lambda: EncodedFunction.slope_segments([(0.0, 1.0), (2.0, 1.0)]),
    lambda: EncodedFunction.slope_segments([(math.nan, 0.0)]),
    lambda: EncodedFunction.affine(0.0, 1.0, 3.0, 2.0),
    lambda: EncodedFunction.affine(0.0, math.inf, 0.0, 1.0),
    lambda: EncodedFunction.affine(0.0, 1.0, 0.0, 1.0, "unknown"),
])
def test_invalid_encoding_inputs_raise_value_errors(build):
    with pytest.raises(ValueError):
        build()


@pytest.mark.parametrize("wire", [
    {"encoding": "unknown", "function": {}},
    {"encoding": "knots", "function": {"points": [[0.0, 0.0], [1.0, 1.0]], "extra": 0}},
    {"encoding": "knots", "function": {"points": [[0.0, 0.0], [1.0, 1.0]]}, "extra": 0},
    {"encoding": "slope_segments", "function": {"pairs": [[0.0, 0.0], [1.0, 1.0]]}},
    {"encoding": "slope_segments", "function": {"pairs": [[0.0, 0.0], [1.0, 1.0]]}, "policy": "StrictDomain"},
    {"encoding": "slope_segments", "function": {"pairs": [[0.0, 1.0], [1.0, 0.0]]}, "policy": "strict_domain"},
])
def test_wire_validation_rejects_unknown_fields_missing_policy_and_invalid_order(wire):
    with pytest.raises(ValueError):
        EncodedFunction.from_dict(wire)


@pytest.mark.parametrize("pairs", [[], [[0.0]], [[0.0, 1.0, 2.0]], [["bad", 0.0]]])
@pytest.mark.parametrize("encoding, field", [("knots", "points"), ("slope_segments", "pairs")])
def test_wire_pairs_reject_malformed_sequences(encoding, field, pairs):
    wire = {"encoding": encoding, "function": {field: pairs}}
    if encoding == "slope_segments":
        wire["policy"] = "strict_domain"
    with pytest.raises(ValueError):
        EncodedFunction.from_dict(wire)


@pytest.mark.parametrize("function", [
    lambda: EncodedFunction.knots([(0.0, 1.0), (2.0, 3.0)]),
    lambda: EncodedFunction.slope_segments([(2.0, 0.0), (3.0, 5.0)]),
])
def test_dictionary_round_trip_accepts_python_pairs(function):
    original = function()
    assert EncodedFunction.from_dict(original.to_dict()).to_dict() == original.to_dict()


def test_literal_value_rejects_nonfinite_argument():
    function = EncodedFunction.affine(0.0, 1.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        function.value(math.nan)
    with pytest.raises(AttributeError):
        function.policy = "native_tolerance"


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_encoded_numeric_maps_dependencies_and_costs_share_original_objective(mode):
    solver = Solver(3, 0)
    solver.new_arc(0, 1, 1.0)
    solver.new_arc(1, 2, 1.0)
    solver.set_bidirectional_mode(mode)
    model = ResourceModel()
    base = model.add_encoded_piecewise_linear_resource(
        2.0, 0.0, 4.0, [EncodedFunction.affine(0.0, 0.5, 0.0, 4.0)] * 2,
    )
    dependent = model.add_encoded_piecewise_linear_dependent(
        base, 0.0, 0.0, 10.0, [EncodedFunction.affine(0.0, 2.0, 0.0, 4.0)] * 2,
    )
    model.add_encoded_piecewise_linear_cost(
        dependent, [EncodedFunction.knots([(0.0, 0.0), (10.0, 10.0)])] * 2,
    )
    model.install(solver)
    # Pricing returns negative reduced-cost columns; the dual leaves the
    # original objective unchanged while making this route improving.
    solver.update_reduced_costs([1.0, 1.0], [0.0, 10.0, 0.0])
    minimum, routes = solver.price(10)
    # Base 2 -> 1 -> .5; dependent 0 -> 2 -> 3; charges 2 + 3.
    assert minimum == -3.0
    assert routes[0].original_cost == 7.0
    assert routes[0].nodes() == [0, 1, 2]


@pytest.mark.parametrize("policy, feasible", [("strict_domain", False), ("native_tolerance", True)])
def test_tolerated_input_remains_feasible_during_resource_preprocessing(policy, feasible):
    solver = Solver(2, 0)
    solver.new_arc(0, 1, -10.0)
    model = ResourceModel()
    base = model.add_encoded_piecewise_linear_resource(
        -0.00001, -1.0, 100.0,
        [EncodedFunction.slope_segments([(2.0, 0.0), (3.0, 10.0)], policy)],
    )
    model.add_encoded_piecewise_linear_cost(base, [EncodedFunction.affine(0.0, 1.0, 0.0, 10.0)])
    model.install(solver)
    minimum, routes = solver.price(10)
    assert bool(routes) is feasible
    if feasible:
        assert minimum == -8.0 and routes[0].original_cost == -8.0


def _encoded_network(network_id, arc_cost):
    half = EncodedFunction.affine(0.0, 0.5, 0.0, 4.0, "native_tolerance").to_dict()
    twice = EncodedFunction.affine(0.0, 2.0, 0.0, 4.0).to_dict()
    identity = EncodedFunction.knots([(0.0, 0.0), (10.0, 10.0)]).to_dict()
    return {
        "id": network_id, "vertices": 3, "multiplicity": [0, 1],
        "resources": [{"kind": "disposable", "bounds": [[0.0, 2.0]] * 3}],
        "arcs": [
            {"tail": 0, "head": 1, "cost": arc_cost, "consumption": [1.0], "entries": [[0, 1.0]]},
            {"tail": 1, "head": 2, "cost": arc_cost, "consumption": [1.0]},
        ],
        "coupled_resources": [
            {"kind": "encoded_piecewise_linear_resource", "initial": 2.0, "bounds": {"lower": 0.0, "upper": 4.0}, "functions": [half] * 2},
            {"kind": "encoded_piecewise_linear_dependent", "parent": 0, "initial": 0.0, "bounds": {"lower": 0.0, "upper": 10.0}, "functions": [twice] * 2},
            {"kind": "encoded_piecewise_linear_cost", "parent": 1, "functions": [identity] * 2},
        ],
    }


@pytest.mark.skipif(not has_clp, reason="Clp build required")
def test_encoded_json_phase_i_recovery_and_sibling_restoration_keep_resource_costs():
    # No initial columns: satisfying the covering row requires phase-I pricing.
    wire = {"rows": [{"sense": "eq", "rhs": 1.0}], "networks": [_encoded_network(7, 1.0), _encoded_network(8, 5.0)]}
    model = NetworkProblem.from_json(json.dumps(wire, allow_nan=False))
    serialized = json.loads(model.to_json())
    assert serialized["networks"][0]["coupled_resources"][0]["functions"][0]["policy"] == "native_tolerance"
    parent = model.solve(bound_mode="numerical", stabilize=False)
    assert parent.status == "Optimal" and parent.lp == pytest.approx(7.0)
    assert parent.snapshot is not None
    assigned = parent.snapshot.branch_assignment(0, 8, True)
    excluded = parent.snapshot.branch_assignment(0, 7, False)
    for node in [assigned, excluded]:
        child = node.solve(bound_mode="numerical", stabilize=False)
        assert child.status == "Optimal" and child.lp == pytest.approx(15.0)
    restored = parent.snapshot.solve(bound_mode="numerical", stabilize=False)
    assert restored.status == "Optimal" and restored.lp == pytest.approx(7.0)


@pytest.mark.parametrize("front_policy, leaf_policy", [
    ("strict_domain", "strict_domain"),
    ("native_front_algebra", "native_tolerance"),
])
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_slope_interval_cost_preserves_explicit_encoding_and_operator_policy(front_policy, leaf_policy, mode):
    solver = Solver(3, 0)
    solver.new_arc(0, 1, -10.0)
    solver.new_arc(1, 2, -10.0)
    solver.set_bidirectional_mode(mode)
    model = ResourceModel()
    model.add_slope_interval_cost(
        0.0, 4.0, 0.0, 4.0, [0.0, 0.0],
        [EncodedFunction.affine(2.0, 1.0, 0.0, 4.0, leaf_policy),
         EncodedFunction.affine(1.0, -0.5, 0.0, 4.0, leaf_policy)],
        policy=front_policy,
    )
    model.install(solver)
    minimum, routes = solver.price(10)
    assert minimum == -17.0
    assert routes[0].original_cost == -17.0
    assert routes[0].nodes() == [0, 1, 2]


@pytest.mark.parametrize("functions, policy", [
    ([lambda: EncodedFunction.knots([(0.0, 0.0), (1.0, 1.0)])], "strict_domain"),
    ([lambda: EncodedFunction.affine(0.0, 1.0, 0.0, 1.0)], "native_front_algebra"),
    ([lambda: EncodedFunction.affine(0.0, 1.0, 0.0, 1.0, "native_tolerance")], "strict_domain"),
    ([lambda: EncodedFunction.affine(0.0, 1.0, 0.0, 1.0)], "native_tolerance"),
])
def test_slope_interval_cost_rejects_lossy_conversion_and_policy_loss(functions, policy):
    with pytest.raises(ValueError):
        ResourceModel().add_slope_interval_cost(
            0.0, 1.0, 0.0, 1.0, [0.0], [build() for build in functions], policy=policy,
        )


@pytest.mark.parametrize("initial, bounds, increments", [
    ((0.0, 2.0), (0.0, 1.0), [0.0]),
    ((0.0, 1.0), (0.0, math.inf), [0.0]),
    ((0.0, 1.0), (0.0, 1.0), [math.nan]),
    ((0.0, 1.0), (0.0, 1.0), []),
])
def test_slope_interval_cost_validates_windows_and_arc_vectors(initial, bounds, increments):
    with pytest.raises(ValueError):
        ResourceModel().add_slope_interval_cost(
            *initial, *bounds, increments, [EncodedFunction.affine(0.0, 1.0, 0.0, 1.0)],
        )


@pytest.mark.skipif(not has_clp, reason="Clp build required")
@pytest.mark.parametrize("policy", ["strict_domain", "native_front_algebra"])
def test_slope_interval_cost_network_json_retains_operator_policy_and_phase_i_cost(policy):
    # SlopeIntervalCost contains prepared function records, with one front policy
    # for the whole operator chain, rather than tagged numeric leaf definitions.
    network = _encoded_network(7, 1.0)
    network["coupled_resources"] = [{
        "kind": "slope_interval_cost", "initial": {"lower": 0.0, "upper": 4.0},
        "bounds": {"lower": 0.0, "upper": 4.0}, "increments": [0.0, 0.0],
        "functions": [{"pairs": [[2.0, 0.0], [1.0, 4.0]]},
                      {"pairs": [[1.0, 0.0], [-0.5, 4.0]]}],
        "policy": policy,
    }]
    model = NetworkProblem.from_json(json.dumps({
        "rows": [{"sense": "eq", "rhs": 1.0}], "networks": [network],
    }))
    restored = json.loads(model.to_json())
    assert restored["networks"][0]["coupled_resources"][0]["policy"] == policy
    result = model.solve(bound_mode="numerical", stabilize=False)
    assert result.status == "Optimal"
    assert result.lp == pytest.approx(5.0)


def _slope_pair_arguments(leaf_policy="strict_domain"):
    def affine(a, b, upper=20.0):
        return EncodedFunction.affine(a, b, 0.0, upper, leaf_policy)
    return {
        "initial_lower": 0.0, "initial_upper": 4.0,
        "initial": affine(5.0, 2.0, 4.0),
        "arrivals": [affine(1.0, 1.0)] * 2,
        "increments": [affine(1.0, 0.0)] * 2,
        "lowers": [affine(0.0, 0.0, 21.0)] * 2,
        "uppers": [affine(100.0, 0.0, 21.0)] * 2,
        "waiting": [0, 0],
    }


@pytest.mark.parametrize("front_policy, leaf_policy", [
    ("strict_domain", "strict_domain"),
    ("native_front_algebra", "native_tolerance"),
])
@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("normalization, expected", [
    ("absolute_final_value", -13.0),
    ("final_minus_initial_minimum", -18.0),
])
def test_slope_paired_keeps_initial_value_and_normalization_explicit(front_policy, leaf_policy, mode, normalization, expected):
    solver = Solver(3, 0)
    solver.new_arc(0, 1, -10.0)
    solver.new_arc(1, 2, -10.0)
    solver.set_bidirectional_mode(mode)
    model = ResourceModel()
    model.add_slope_paired(
        **_slope_pair_arguments(leaf_policy), mode="cost_bounded",
        policy=front_policy, normalization=normalization,
    )
    model.install(solver)
    minimum, routes = solver.price(10)
    # Initial minimum5; two positive increments1 yield absolute7 or delta2.
    assert minimum == expected
    assert routes[0].original_cost == expected
    assert routes[0].nodes() == [0, 1, 2]


def test_slope_paired_default_normalization_charges_absolute_final_value():
    solver = Solver(3, 0)
    solver.new_arc(0, 1, -10.0)
    solver.new_arc(1, 2, -10.0)
    model = ResourceModel()
    model.add_slope_paired(**_slope_pair_arguments(), mode="cost_bounded")
    model.install(solver)
    minimum, routes = solver.price(10)
    assert minimum == -13.0 and routes[0].original_cost == -13.0


@pytest.mark.parametrize("field", ["initial", "arrivals", "increments", "lowers", "uppers"])
@pytest.mark.parametrize("bad_encoding", ["knots", "mismatched_leaf_policy"])
def test_slope_paired_rejects_lossy_or_mismatched_function_in_every_role(field, bad_encoding):
    arguments = _slope_pair_arguments()
    bad = (EncodedFunction.knots([(0.0, 0.0), (21.0, 21.0)]) if bad_encoding == "knots"
           else EncodedFunction.affine(0.0, 1.0, 0.0, 21.0, "native_tolerance"))
    arguments[field] = bad if field == "initial" else [bad] * 2
    with pytest.raises(ValueError):
        ResourceModel().add_slope_paired(**arguments, mode="cost_bounded")


@pytest.mark.parametrize("overrides", [
    {"mode": "unknown"}, {"normalization": "delta"}, {"policy": "native_tolerance"},
    {"waiting": [0]}, {"increments": []}, {"initial_upper": math.inf},
])
def test_slope_paired_rejects_invalid_modes_normalization_and_arc_shapes(overrides):
    arguments = _slope_pair_arguments()
    arguments.update(overrides)
    with pytest.raises(ValueError):
        ResourceModel().add_slope_paired(**arguments)


@pytest.mark.skipif(not has_clp, reason="Clp build required")
@pytest.mark.parametrize("normalization, expected", [
    ("absolute_final_value", 9.0), ("final_minus_initial_minimum", 4.0),
])
@pytest.mark.parametrize("policy", ["strict_domain", "native_front_algebra"])
def test_slope_paired_network_phase_i_and_sibling_restore_retain_normalization(normalization, expected, policy):
    def network(network_id, arc_cost):
        value = _encoded_network(network_id, arc_cost)
        arguments = _slope_pair_arguments()
        value["coupled_resources"] = [{
            "kind": "slope_paired", "initial_base": {"lower": 0.0, "upper": 4.0},
            "initial": arguments["initial"].to_dict()["function"],
            "arcs": [{
                "arrival": arguments["arrivals"][i].to_dict()["function"],
                "increment": arguments["increments"][i].to_dict()["function"],
                "lower": arguments["lowers"][i].to_dict()["function"],
                "upper": arguments["uppers"][i].to_dict()["function"], "waiting": 0,
            } for i in range(2)],
            "mode": "cost_bounded", "normalization": normalization, "policy": policy,
        }]
        return value
    model = NetworkProblem.from_json(json.dumps({
        "rows": [{"sense": "eq", "rhs": 1.0}],
        "networks": [network(7, 1.0), network(8, 5.0)],
    }))
    wire = json.loads(model.to_json())
    assert wire["networks"][0]["coupled_resources"][0]["normalization"] == normalization
    assert wire["networks"][0]["coupled_resources"][0]["policy"] == policy
    root = model.solve(bound_mode="numerical", stabilize=False)
    assert root.status == "Optimal" and root.lp == pytest.approx(expected)
    assert root.snapshot is not None
    for child in [root.snapshot.branch_assignment(0, 8, True), root.snapshot.branch_assignment(0, 7, False)]:
        solved = child.solve(bound_mode="numerical", stabilize=False)
        assert solved.status == "Optimal" and solved.lp == pytest.approx(expected + 8.0)
    restored = root.snapshot.solve(bound_mode="numerical", stabilize=False)
    assert restored.status == "Optimal" and restored.lp == pytest.approx(expected)
