import gc
import importlib.metadata
import json
import math

import pytest

import pybaldes
from pybaldes import Customer, Instance, solve_root


def triangle():
    return Instance("triangle", 2, [
        Customer(0, 0, due=100), Customer(1, 0, 1, due=100),
        Customer(0, 1, 1, due=100), Customer(1, 1, 1, due=100),
    ], vehicles=3)


def assert_routes(result, count, capacity):
    coverage = [0.0] * (count + 1)
    objective = 0.0
    for route in result.routes:
        assert route.nodes[0] == 0
        assert route.nodes[-1] == count + 1
        assert len(route.arcs) == len(route.nodes) - 1
        assert len(route.nodes[1:-1]) <= capacity
        for node in route.nodes[1:-1]:
            coverage[node] += route.value
        objective += route.cost * route.value
    assert coverage[1:] == pytest.approx([1.0] * count)
    assert objective == pytest.approx(result.lp)


def test_package_identity():
    assert importlib.metadata.version("pybaldes") == pybaldes.__version__
    assert pybaldes.Solver.__module__ == "pybaldes._core"
    assert pybaldes.has_clp


@pytest.mark.parametrize("problem_type", ["vrptw", "cvrp"])
@pytest.mark.parametrize("rank_one_cuts,fixing", [(False, False), (True, False), (False, True), (True, True)])
def test_capacity_root(problem_type, rank_one_cuts, fixing):
    instance = triangle()
    plain = solve_root(instance, problem_type=problem_type, reduced_cost_fixing=False)
    result = instance.solve_root(
        problem_type=problem_type, capacity_cuts=True, rank_one_cuts=rank_one_cuts,
        reduced_cost_fixing=fixing,
    )
    assert plain.lp == pytest.approx(1.5 * (2 + math.sqrt(2)))
    assert result.status == "Optimal"
    assert result.lp == pytest.approx(4 + math.sqrt(2))
    assert result.bound == pytest.approx(result.lp)
    assert result.lp_before_cuts == pytest.approx(plain.lp)
    assert result.cuts > 0 and result.cut_rounds > 0
    assert result.history and result.pricing_calls > 0
    assert len(result.duals) >= 3
    assert_routes(result, 3, 2)
    serialized = json.loads(result.to_json())
    assert serialized["lp"] == result.lp
    assert serialized["routes"][0]["nodes"] == list(result.routes[0].nodes)


def test_solomon_file_and_time_window_semantics(tmp_path):
    path = tmp_path / "tiny.txt"
    path.write_text(
        "tiny\nVEHICLE\nNUMBER CAPACITY\n2 2\nCUSTOMER\n"
        "CUST NO. X Y DEMAND READY DUE SERVICE\n"
        "0 0 0 0 0 100 0\n1 1 0 1 0 1.1 0\n2 0 1 1 0 1.1 0\n"
    )
    instance = Instance.from_file(path)
    assert instance.name == "tiny"
    assert instance.capacity == 2 and instance.vehicles == 2
    assert len(instance.customers) == 3
    assert len(instance.truncated(1).customers) == 2
    assert len(instance.customers) == 3
    vrptw = solve_root(path, reduced_cost_fixing=False)
    cvrp = solve_root(path, problem_type="cvrp", reduced_cost_fixing=False)
    assert vrptw.lp == pytest.approx(4)
    assert cvrp.lp == pytest.approx(2 + math.sqrt(2))
    assert solve_root(path, customers=1, reference_ng=True, reduced_cost_fixing=False).lp == pytest.approx(2)
    with pytest.raises(ValueError):
        Instance.from_file(tmp_path / "absent.txt")


def test_cvrp_ignores_service_and_time_windows():
    instance = Instance("service", 2, [
        Customer(0, 0, due=0),
        Customer(1, 0, 1, ready=100, due=100, service=100),
        Customer(0, 1, 1, ready=100, due=100, service=100),
    ])
    result = solve_root(instance, problem_type="cvrp", reduced_cost_fixing=False)
    assert result.lp == pytest.approx(2 + math.sqrt(2))


@pytest.mark.parametrize("options", [
    {"problem_type": "evrp"}, {"rounding": "bad"}, {"ng": 0},
    {"time_limit": float("nan")}, {"time_limit": -1}, {"max_iterations": 0},
    {"upper_bound": float("nan")}, {"upper_bound": -1},
])
def test_invalid_options(options):
    with pytest.raises(ValueError):
        solve_root(triangle(), **options)


def test_invalid_instance_and_pricing_options():
    with pytest.raises(ValueError):
        solve_root(Instance("bad", -1, [Customer(0, 0), Customer(1, 0, 1)]))
    with pytest.raises(ValueError):
        triangle().truncated(0)
    with pytest.raises(TypeError):
        Instance("bad", 1, [(0, 0)])
    with pytest.raises(RuntimeError):
        solve_root(triangle(), pricing_options={"does_not_exist": 1})


@pytest.mark.parametrize("rounding,expected", [("none", math.sqrt(2) * 2), ("nearest", 2), ("tenth", 2.8)])
def test_rounding(rounding, expected):
    instance = Instance("diagonal", 1, [Customer(0, 0), Customer(1, 1, 1)])
    result = solve_root(instance, problem_type="cvrp", rounding=rounding, reduced_cost_fixing=False)
    assert result.lp == pytest.approx(expected)


def test_low_level_pricing_validation_and_route_ownership():
    solver = pybaldes.Solver(3, 1)
    solver.set_source(0)
    solver.set_sink(2)
    resource = solver.new_resource("load", 0)
    first = solver.new_arc(0, 1, -2)
    second = solver.new_arc(1, 2, 1)
    solver.set_arc_consumption(first, resource, 1)
    solver.set_arc_consumption(second, resource, 1)
    for node in range(3):
        solver.set_vertex_bounds(node, resource, 0, 2)
        solver.set_ng_neighborhood(node, [node])
    with pytest.raises(ValueError):
        solver.update_reduced_costs([-2, 1], [0])
    for maximum in [0, -1]:
        with pytest.raises(ValueError):
            solver.price(maximum)
        with pytest.raises(ValueError):
            solver.enumerate(maximum)
    solver.update_reduced_costs([-2, 1], [0, 0, 0])
    cost, route = solver.solve()
    assert cost == route.cost == -1
    assert route.nodes() == [0, 1, 2]
    assert route.arcs() == [first, second]
    assert route.resource_consumption(resource) == 2
    del solver
    gc.collect()
    assert route.cost == -1 and route.nodes() == [0, 1, 2]
