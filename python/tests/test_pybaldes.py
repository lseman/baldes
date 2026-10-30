import gc
import importlib.metadata
import json
import math

import pytest

import pybaldes
from pybaldes import Customer, Instance, solve


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


def test_bound_modes_keep_numerical_estimates_separate_from_certificates():
    numerical = solve(triangle(), reduced_cost_fixing=False, stabilize=False)
    certified = solve(triangle(), bound_mode="certified", reduced_cost_fixing=False, stabilize=False)
    assert numerical.bound_mode == "numerical"
    assert numerical.bound == numerical.numerical_bound
    assert numerical.certified_bound == -math.inf
    assert json.loads(numerical.to_json())["certified_bound"] is None
    assert certified.bound_mode == "certified"
    assert math.isfinite(certified.certified_bound)
    assert certified.bound == certified.certified_bound
    assert certified.numerical_bound == pytest.approx(numerical.numerical_bound)
    assert certified.lp == pytest.approx(numerical.lp)
    assert_routes(numerical, 3, 2)
    assert_routes(certified, 3, 2)
    with pytest.raises(ValueError, match="bound mode"):
        solve(triangle(), bound_mode="invalid")


@pytest.mark.parametrize("problem_type", ["vrptw", "cvrp"])
@pytest.mark.parametrize("rank_one_cuts,fixing", [(False, False), (True, False), (False, True), (True, True)])
def test_capacity_root(problem_type, rank_one_cuts, fixing):
    instance = triangle()
    plain = solve(instance, problem_type=problem_type, reduced_cost_fixing=False)
    result = instance.solve(
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
    assert result.oracle_calls > 0
    assert result.oracle_calls <= result.pricing_calls + sum(r.pricing_calls for r in result.feasibility)
    assert len(result.duals) >= 3
    assert_routes(result, 3, 2)
    serialized = json.loads(result.to_json())
    assert serialized["lp"] == result.lp
    assert serialized["oracle_calls"] == result.oracle_calls
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
    vrptw = solve(path, reduced_cost_fixing=False)
    cvrp = solve(path, problem_type="cvrp", reduced_cost_fixing=False)
    assert vrptw.lp == pytest.approx(4)
    assert cvrp.lp == pytest.approx(2 + math.sqrt(2))
    assert solve(path, customers=1, reference_ng=True, reduced_cost_fixing=False).lp == pytest.approx(2)
    with pytest.raises(ValueError):
        Instance.from_file(tmp_path / "absent.txt")


def test_cvrp_ignores_service_and_time_windows():
    instance = Instance("service", 2, [
        Customer(0, 0, due=0),
        Customer(1, 0, 1, ready=100, due=100, service=100),
        Customer(0, 1, 1, ready=100, due=100, service=100),
    ])
    result = solve(instance, problem_type="cvrp", reduced_cost_fixing=False)
    assert result.lp == pytest.approx(2 + math.sqrt(2))


@pytest.mark.parametrize("options", [
    {"problem_type": "evrp"}, {"rounding": "bad"}, {"ng": 0},
    {"time_limit": float("nan")}, {"time_limit": -1}, {"max_iterations": 0},
    {"upper_bound": float("nan")}, {"upper_bound": -1},
])
def test_invalid_options(options):
    with pytest.raises(ValueError):
        solve(triangle(), **options)


def test_invalid_instance_and_pricing_options():
    with pytest.raises(ValueError):
        solve(Instance("bad", -1, [Customer(0, 0), Customer(1, 0, 1)]))
    with pytest.raises(ValueError):
        triangle().truncated(0)
    with pytest.raises(TypeError):
        Instance("bad", 1, [(0, 0)])
    with pytest.raises(RuntimeError):
        solve(triangle(), pricing_options={"does_not_exist": 1})


@pytest.mark.parametrize("rounding,expected", [("none", math.sqrt(2) * 2), ("nearest", 2), ("tenth", 2.8)])
def test_rounding(rounding, expected):
    instance = Instance("diagonal", 1, [Customer(0, 0), Customer(1, 1, 1)])
    result = solve(instance, problem_type="cvrp", rounding=rounding, reduced_cost_fixing=False)
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


def test_kohl_time_resource_closes_r101_root_gap():
    from pathlib import Path
    instance = Instance.from_file(Path(__file__).resolve().parents[2] / 'instances/Solomon/R101.txt').truncated(10)
    plain = solve(instance, stabilize=False, reduced_cost_fixing=False)
    kohl = solve(instance, two_path_cuts=True, stabilize=False, reduced_cost_fixing=False)
    assert plain.lp == pytest.approx(261.30231910442023)
    assert kohl.status == 'Optimal'
    assert kohl.lp == pytest.approx(269.533140900765)
    assert kohl.cuts > 0


def test_kohl_rejects_cvrp_without_time_resource():
    with pytest.raises(ValueError, match='time windows'):
        solve(triangle(), problem_type='cvrp', two_path_cuts=True)

@pytest.mark.parametrize("columns", ["best", "all", 1])
def test_hgs_columns_and_ipm_master(columns):
    result = solve(
        triangle(), rank_one_cuts=True, reduced_cost_fixing=False,
        master_solver="ipm", hgs_time=0.03, hgs_columns=columns, hgs_seed=7,
    )
    assert result.status == "Optimal"
    assert result.ipm_iterations > 0
    assert result.lp == pytest.approx(4 + math.sqrt(2), abs=1e-6)
    assert all(step.alpha == 0 for step in result.history)
    assert_routes(result, 3, 2)


@pytest.mark.parametrize("options", [
    {"master_solver": "wrong"}, {"hgs_columns": "wrong"},
    {"hgs_columns": 0}, {"hgs_columns": -1}, {"hgs_columns": 1.5},
    {"hgs_time": 0}, {"hgs_time": -1}, {"hgs_time": math.nan},
])
def test_invalid_hgs_and_master_options(options):
    with pytest.raises(ValueError):
        solve(triangle(), **options)


def test_cg_progress_is_opt_in_and_goes_to_stderr(capfd):
    quiet = solve(triangle(), reduced_cost_fixing=False)
    assert "COLUMN GENERATION" not in capfd.readouterr().err
    logged = solve(triangle(), reduced_cost_fixing=False, verbose=True, log_interval=1)
    captured = capfd.readouterr()
    assert "COLUMN GENERATION" in captured.err
    assert "RMP obj" in captured.err and "CG Optimal" in captured.err
    assert "COLUMN GENERATION" not in captured.out
    assert logged.lp == pytest.approx(quiet.lp)
    with pytest.raises(ValueError, match="positive"):
        solve(triangle(), log_interval=0)


def test_highs_master_selection_respects_the_build_features():
    if pybaldes.has_highs:
        result = solve(triangle(), master_solver="highs", rank_one_cuts=True, reduced_cost_fixing=False)
        assert result.status == "Optimal"
        assert result.lp == pytest.approx(4 + math.sqrt(2), abs=1e-6)
        assert_routes(result, 3, 2)
    else:
        with pytest.raises(ValueError, match="highs build feature"):
            solve(triangle(), master_solver="highs")


def test_general_scalar_resource_dimensions():
    dimensions = pybaldes.resource_dimensions
    solver = pybaldes.Solver(3, dimensions)
    for resource in range(dimensions):
        assert solver.new_resource(f"r{resource}", 0) == resource
        for vertex in range(3):
            solver.set_vertex_bounds(vertex, resource, 0.0, 10.0)
    first = solver.new_arc(0, 1, -2.0)
    second = solver.new_arc(1, 2, -3.0)
    for resource in range(dimensions):
        solver.set_arc_consumption(first, resource, 1.0)
        solver.set_arc_consumption(second, resource, 2.0)
    cost, route = solver.solve()
    assert cost == -5.0
    assert [route.resource_consumption(r) for r in range(dimensions)] == [3.0] * dimensions


def test_resource_capacity_is_fixed_at_build_time():
    with pytest.raises(RuntimeError):
        pybaldes.Solver(3, pybaldes.resource_dimensions + 1)


@pytest.mark.parametrize("scaling", ["auto", "geometric", "equilibrium", "ruiz", "off"])
@pytest.mark.parametrize("keep", [False, True])
@pytest.mark.parametrize("rank_one", [False, True])
def test_clp_runtime_policy_preserves_routing_solution(scaling, keep, rank_one):
    result = solve(
        triangle(), clp_scaling=scaling, clp_keep_work_areas=keep,
        reduced_cost_fixing=False, rank_one_cuts=rank_one,
    )
    assert result.status == "Optimal"
    expected = 4 + math.sqrt(2) if rank_one else 3 + 1.5 * math.sqrt(2)
    assert result.lp == pytest.approx(expected, abs=1e-6)
    assert_routes(result, 3, 2)


def test_invalid_clp_scaling_is_rejected():
    with pytest.raises(ValueError, match="Clp scaling"):
        solve(triangle(), clp_scaling="invalid")


def test_master_worker_budget_defaults_and_validation():
    import inspect

    default = solve(triangle(), reduced_cost_fixing=False)
    explicit = solve(triangle(), master_threads=0, reduced_cost_fixing=False)
    assert default.lp == pytest.approx(explicit.lp)
    for threads in [-1, 2**64, 2**31]:
        with pytest.raises((ValueError, OverflowError)):
            solve(triangle(), master_threads=threads)
    result = solve(triangle(), master_threads=1, reduced_cost_fixing=False)
    assert result.status == "Optimal"


def test_root_scaling_defaults_to_off():
    import inspect

    assert inspect.signature(solve).parameters["clp_scaling"].default == "off"
    default = solve(triangle(), reduced_cost_fixing=False)
    explicit = solve(triangle(), reduced_cost_fixing=False, clp_scaling="off")
    assert default.status == explicit.status == "Optimal"
    assert default.lp == pytest.approx(explicit.lp, abs=1e-8)


def test_direct_rust_solver_validates_ids_and_preserves_phase_keywords():
    solver = pybaldes.Solver(2, 1)
    for call in [
        lambda: solver.set_source(-1),
        lambda: solver.set_sink(2),
        lambda: solver.set_resource_type(1, 0),
        lambda: solver.set_resource_type(0, 2),
        lambda: solver.set_initial_resource(0, math.inf),
        lambda: solver.set_ng_neighborhood(0, [0, 0]),
        lambda: solver.attach_elementarity_to_node(0, 999),
        lambda: solver.get_phase(index=-1),
        lambda: solver.get_enumerated(index=0),
    ]:
        with pytest.raises(RuntimeError):
            call()
    assert solver.get_phase(index=0).labels_per_bucket > 0
    assert solver.enumerated_count == 0
    arc = solver.new_arc(0, 1, -3)
    solver.set_arc_consumption(arc, 0, 1)
    solver.set_vertex_bounds(0, 0, 0, 2)
    solver.set_vertex_bounds(1, 0, 0, 2)
    solver.build_default_resource_buckets()
    minimum, routes = solver.price(10)
    assert minimum == -3
    assert routes[0].arcs() == [arc]
    # The default pricing schedule starts with a heuristic phase.
    assert not routes[0].is_certified
    _, _, exact_routes = solver.enumerate(10)
    assert exact_routes[0].is_certified
    assert routes[0].resource_consumption(0) == 1
    with pytest.raises(RuntimeError):
        routes[0].resource_consumption(1)


def test_direct_rust_solver_can_move_to_another_python_thread():
    from concurrent.futures import ThreadPoolExecutor

    solver = pybaldes.Solver(2, 0)
    solver.new_arc(0, 1, -1)
    with ThreadPoolExecutor(max_workers=1) as pool:
        cost, route = pool.submit(solver.solve).result()
    assert cost == -1
    assert route.nodes() == [0, 1]


def rank_one_path():
    # 0 -> 1 -> 2 -> 3 -> 4 is the only route; every customer arc costs -1.
    solver = pybaldes.Solver(5, 1)
    solver.set_source(0)
    solver.set_sink(4)
    solver.set_initial_resource(0, 0.)
    arcs = []
    for tail in range(4):
        arc = solver.new_arc(tail, tail + 1, -1. if tail < 3 else 0.)
        solver.set_arc_consumption(arc, 0, 1.)
        arcs.append(arc)
    for vertex in range(5):
        solver.set_vertex_bounds(vertex, 0, 0., 10.)
    return solver, arcs


def test_rank_one_cut_duals_follow_arc_memory():
    solver, arcs = rank_one_path()
    support = [(1, 1), (2, 1), (3, 1)]
    solver.set_rank_one_cuts([(support, 2, 1, arcs, -0.75)])
    minimum, routes, exact = solver.price_with_status(5, exact=True)
    assert exact
    assert routes[0].nodes() == [0, 1, 2, 3, 4]
    # Full memory: floor(3 / 2) = 1 unit of the dual.
    assert minimum == pytest.approx(-3. + 0.75)
    # Without memory the state resets on every arc, so the cut is never paid.
    solver.set_rank_one_cuts([(support, 2, 1, [], -0.75)])
    assert solver.price_with_status(5, exact=True)[0] == pytest.approx(-3.)
    solver.set_rank_one_cuts([])
    assert solver.price_with_status(5, exact=True)[0] == pytest.approx(-3.)
    for bad in ([(support, 0, 1, arcs, -1.)], [([(1, 0)], 2, 0, arcs, -1.)],
                [(support, 2, 1, [99], -1.)], [(support, 2, 1, arcs, math.nan)]):
        with pytest.raises((ValueError, RuntimeError)):
            solver.set_rank_one_cuts(bad)
