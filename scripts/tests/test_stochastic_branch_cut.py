"""Independent tiny-oracle checks of capacity separation and BC-Theta recourse."""
import importlib.util
import itertools
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "stochastic_branch_cut.py"
spec = importlib.util.spec_from_file_location("stochastic_branch_cut", SCRIPT)
bc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bc)


def data(capacity, supports, costs, vehicles):
    return {"capacity": capacity, "vehicles": vehicles, "costs": costs,
            "demands": [{"customer": j, "support": [d for d, _ in support],
                         "probabilities": [p for _, p in support]} for j, support in enumerate(supports, 1)]}


def scenario(d, route, at, load, policy):
    if at == len(route):
        return 0.
    i, j = route[at - 1] if at else 0, route[at]
    customer = d["demands"][j - 1]
    def continuation(q):
        value = 0.
        for demand, probability in zip(customer["support"], customer["probabilities"]):
            remaining = q - demand
            cost = 0.
            if remaining < 0:
                remaining += d["capacity"]
                cost = d["costs"][j][0] + d["costs"][0][j]
            value += probability * (cost + scenario(d, route, at + 1, remaining, policy))
        return value
    refill = d["costs"][i][0] + d["costs"][0][j] - d["costs"][i][j] + continuation(d["capacity"])
    return min(refill, continuation(load)) if policy == "p3" else (refill if load == 0 else continuation(load))


def exhaustive(d, policy):
    n = len(d["demands"])
    means = bc.mean_demands(d)
    best = {}
    for length in range(1, n + 1):
        for route in itertools.permutations(range(1, n + 1), length):
            if sum(means[j - 1] for j in route) > d["capacity"]:
                continue
            route = list(route)
            travel = sum(d["costs"][i][j] for i, j in zip([0] + route, route + [0]))
            cost = travel + scenario(d, route, 0, d["capacity"], policy)
            key = frozenset(route)
            best[key] = min(best.get(key, float("inf")), cost)
    def partition(remaining, vehicles):
        if not remaining:
            return 0.
        if not vehicles:
            return float("inf")
        first = min(remaining)
        return min((cost + partition(remaining - subset, vehicles - 1)
                    for subset, cost in best.items() if first in subset and subset <= remaining), default=float("inf"))
    return partition(frozenset(range(1, n + 1)), d["vehicles"])


@pytest.fixture(scope="module", autouse=True)
def license_available():
    gp = pytest.importorskip("gurobipy")
    try:
        with gp.Env(params={"OutputFlag": 0}) as env, gp.Model(env=env) as model:
            model.optimize()
    except gp.GurobiError as error:
        pytest.skip(f"Gurobi license unavailable: {error}")


@pytest.mark.parametrize("policy", ["p1", "p3"])
@pytest.mark.parametrize("fractional", [False, True])
def test_recourse_cuts_match_independent_scenario_and_partition_oracle(policy, fractional):
    d = data(10, [[(4, .5), (8, .5)], [(4, 1.)]], [[0., 1., 10.], [1., 0., 10.], [10., 10., 0.]], 2)
    answer = bc.solve(d, policy, 10, fractional=fractional)
    assert answer["status"] == "Optimal"
    assert answer["objective"] == pytest.approx(exhaustive(d, policy), abs=1e-6)
    assert answer["lower_bound"] == pytest.approx(answer["objective"], abs=1e-6)
    assert answer["cuts"]["theta"] > 0


@pytest.mark.parametrize("policy", ["p1", "p3"])
def test_subtours_capacity_and_directed_costs(policy):
    d = data(4, [[(0, .5), (2, .5)], [(2, 1.)], [(1, 1.)], [(2, 1.)]],
             [[(j + 5 - i) % 5 for j in range(5)] for i in range(5)], 2)
    answer = bc.solve(d, policy, 10)
    assert answer["status"] == "Optimal"
    assert answer["objective"] == pytest.approx(exhaustive(d, policy), abs=1e-6)
    assert answer["cuts"]["lazy_capacity"] + answer["cuts"]["fractional_capacity"] > 0


@pytest.mark.parametrize("policy", ["p1", "p3"])
def test_nonmetric_negative_recourse_is_not_clamped_to_zero(policy):
    d = data(2, [[(2, 1.)], [(0, 1.)]], [[0., 0., 0.], [0., 0., 10.], [0., 20., 0.]], 1)
    answer = bc.solve(d, policy, 10)
    assert answer["recourse_shift_per_route"] > 0
    assert answer["status"] == "Optimal"
    assert answer["objective"] == pytest.approx(exhaustive(d, policy), abs=1e-6)
    assert answer["objective"] == pytest.approx(0., abs=1e-6)


def test_zero_demand_cycles_are_eliminated():
    d = data(1, [[(0, 1.)]] * 3, [[0 if i == j else (2 if i == 0 or j == 0 else 0) for j in range(4)] for i in range(4)], 1)
    answer = bc.solve(d, "p3", 10, fractional=False)
    assert answer["status"] == "Optimal"
    assert answer["objective"] == pytest.approx(4.)
    assert sorted(j for r in answer["routes"] for j in r) == [1, 2, 3]


def test_fleet_infeasibility_and_time_limit_have_no_false_optimum():
    d = data(2, [[(1, 1.)]] * 3, [[0 if i == j else 1 for j in range(4)] for i in range(4)], 1)
    answer = bc.solve(d, "p3", 10)
    assert answer["status"] == "Infeasible"
    assert answer["objective"] is None
    answer = bc.solve(d, "p3", 0)
    assert answer["status"] == "TimeLimit"
    assert answer["objective"] is None
