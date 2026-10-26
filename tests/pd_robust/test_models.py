"""Licensed Gurobi integration checks, including the printed-load counterexample."""
from pathlib import Path

import pytest

gp = pytest.importorskip("gurobipy")

from pd_robust.data import Settings, read_instance
from pd_robust.models import SolveOptions, _callback, build_model, solve
from test_data_separation import fixture, scenario_feasible, solutions


@pytest.fixture(scope="module")
def license_available():
    try:
        with gp.Env(params={"OutputFlag": 0}):
            pass
    except gp.GurobiError as error:
        pytest.skip(f"Gurobi license unavailable: {error}")


@pytest.mark.parametrize("gq,gt", [(0, 0), (1, 1), (2, 2)])
@pytest.mark.parametrize("method", ["compact", "branch-cut", "compact-guarded", "onboard-compact"])
def test_models_match_exhaustive_set_partition(license_available, gq, gt, method):
    p = fixture(capacity=10, close=7, gamma_q=gq, gamma_t=gt)
    expected = min(sum(p.cost(arc) for route in selected for arc in zip(route, route[1:]))
                   for selected in solutions(p))
    result = solve(p, method, SolveOptions(seconds=10, warm_start=False))
    assert result["status"] == "Optimal", result
    assert result["valid"], result
    assert abs(result["objective"] - expected) < 1e-6
    assert all(scenario_feasible(p, route) for route in result["routes"])


@pytest.mark.parametrize("method", ["compact", "branch-cut", "onboard-compact"])
def test_individually_infeasible_request_is_not_reported_as_a_solution(license_available, method):
    p = fixture(capacity=5, gamma_q=1)
    result = solve(p, method, SolveOptions(seconds=5))
    assert result["status"] == "Infeasible"
    assert result["objective"] is None


def counterexample():
    path = Path(__file__).resolve().parents[2] / "pd_robust/cases/compact_counterexample.txt"
    return read_instance(path, Settings(1, 0, 0.3, 0))


@pytest.mark.parametrize("method,expected", [("compact", gp.GRB.OPTIMAL),
                                            ("compact-guarded", gp.GRB.INFEASIBLE),
                                            ("branch-cut", gp.GRB.INFEASIBLE),
                                            ("onboard-compact", gp.GRB.INFEASIBLE)])
@pytest.mark.parametrize("literal_big_m", [False, True])
def test_printed_paper_load_model_counterexample(license_available, method, expected, literal_big_m):
    p = counterexample()
    route = [0, 1, 2, 4, 3, 5, 6, 7]
    with pytest.raises(ValueError, match="11.0 > 10.0"):
        p.validate([route])
    with gp.Env(params={"OutputFlag": 0}) as env:
        built = build_model(p, method, SolveOptions(seconds=10, warm_start=False,
                                                   literal_big_m=literal_big_m), env=env)
        with built.model as model:
            selected = set(zip(route, route[1:]))
            for arc, var in built.x.items():
                var.LB = var.UB = int(arc in selected)
            if method in ("branch-cut", "compact-guarded"):
                model.optimize(lambda m, where: _callback(built, m, where))
                assert built.callback_error is None
            else:
                model.optimize()
            assert model.Status == expected


def test_counterexample_audit_is_reproducible(license_available):
    from pd_robust.audit import audit
    report = audit()
    assert report["worst_onboard_after_node3"] > report["capacity"]
    assert report["methods"]["compact"]["status"] == "Optimal"
