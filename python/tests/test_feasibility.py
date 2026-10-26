import json

import pytest

from pybaldes import Customer, Instance, solve


@pytest.mark.parametrize("distance", [1.0, 1e13])
def test_root_feasibility_diagnostics_survive_typed_results_and_json(distance):
    instance = Instance(
        "phase-i", 1,
        [Customer(0, 0, due=1e15), Customer(distance, 0, 1, due=1e15)],
        vehicles=1,
    )
    result = solve(
        instance, problem_type="cvrp", rounding="none", stabilize=False,
        reduced_cost_fixing=False,
    )
    assert result.status == "Optimal"
    assert result.lp == pytest.approx(2 * distance)
    assert result.routes[0].cost == pytest.approx(2 * distance)
    serialized = json.loads(result.to_json())
    assert isinstance(serialized["feasibility"], list)
    if distance > 1e12:
        assert result.feasibility
        report = result.feasibility[-1]
        assert report.status == "feasible"
        assert report.artificial_activity <= 1e-6
        assert report.iterations > 0 and report.pricing_calls > 0
        assert serialized["feasibility"][-1]["status"] == "feasible"
    else:
        assert result.feasibility == ()
