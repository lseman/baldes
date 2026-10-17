import math
from pathlib import Path

import pytest

import pybaldes
from pybaldes import Customer, Instance, solve_root

INSTANCES = Path(__file__).resolve().parents[2] / "instances" / "cvrplib"


def test_from_file_reads_cvrplib():
    instance = Instance.from_file(INSTANCES / "E-n22-k4.vrp")
    assert instance.name == "E-n22-k4"
    assert instance.rounding == "nearest"
    assert instance.distances is None
    assert len(instance.customers) == 22 and instance.customers[0].demand == 0
    assert all(math.isinf(c.due) for c in instance.customers)


@pytest.mark.skipif(not pybaldes.has_clp, reason="built without Clp")
def test_cvrplib_root_uses_the_file_convention():
    result = solve_root(INSTANCES / "E-n22-k4.vrp", problem_type="cvrp", capacity_cuts=True,
                        reduced_cost_fixing=False, reference_ng=True)
    assert result.lp == pytest.approx(375.0)
    unrounded = solve_root(INSTANCES / "E-n22-k4.vrp", problem_type="cvrp", capacity_cuts=True,
                           reduced_cost_fixing=False, reference_ng=True, rounding="none")
    assert unrounded.lp == pytest.approx(375.2797868, abs=1e-6)
    with pytest.raises(ValueError, match="no time windows"):
        solve_root(INSTANCES / "E-n22-k4.vrp")


@pytest.mark.skipif(not pybaldes.has_clp, reason="built without Clp")
def test_explicit_distances_survive_truncation():
    customers = [Customer(0, 0), Customer(0, 0, 3), Customer(0, 0, 3), Customer(0, 0, 3)]
    distances = [[0, 2, 3, 4], [2, 0, 5, 6], [3, 5, 0, 7], [4, 6, 7, 0]]
    instance = Instance("explicit", 10, customers, distances=distances)
    small = instance.truncated(2)
    assert small.distances == ((0, 2, 3), (2, 0, 5), (3, 5, 0))
    # Customers 1 and 2 share a route: 2 + 5 + 3 = 10 < 2 * 2 + 2 * 3.
    result = small.solve_root(problem_type="cvrp", reduced_cost_fixing=False)
    assert result.lp == pytest.approx(10.0)
