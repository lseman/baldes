"""Optional multi-time-windows wheel feature."""
import pytest
from pybaldes import Solver

pytestmark = pytest.mark.skipif(
    not hasattr(Solver, "set_vertex_time_windows"),
    reason="requires a wheel built with multi-time-windows",
)


def test_multiple_windows_wait_and_reject_bounds_inside_a_gap():
    solver = Solver(3, 1)
    for vertex in range(3):
        solver.set_vertex_bounds(vertex, 0, 0.0, 20.0)
    first = solver.new_arc(0, 1, -2.0)
    last = solver.new_arc(1, 2, -1.0)
    solver.set_arc_consumption(first, 0, 4.0)
    solver.set_arc_consumption(last, 0, 2.0)
    solver.set_vertex_time_windows(1, 0, [(8.0, 12.0), (0.0, 2.0)])
    minimum, routes = solver.price(100)
    assert minimum == -3.0
    assert routes[0].resource_consumption(0) == 10.0
    with pytest.raises(RuntimeError):
        solver.set_vertex_time_windows(1, 0, [])
    solver.set_vertex_bounds(1, 0, 3.0, 7.0)
    assert solver.solve() is None


def test_network_resource_windows_round_trip_and_waiting():
    from dataclasses import replace
    from pybaldes import (
        NetworkProblem, PricingArc, PricingNetwork, PricingResource, SharedRow, has_clp,
    )
    if not has_clp:
        pytest.skip("Clp build required")
    resource = PricingResource(
        bounds=[(0., 20.)] * 3,
        time_windows=[None, [[0., 2.], [8., 12.]], None],
    )
    assert resource.time_windows == (None, ((0., 2.), (8., 12.)), None)
    model = NetworkProblem([SharedRow("eq", 1.)], [PricingNetwork(
        7, 3, (0., 1.), [
            PricingArc(0, 1, 2., consumption=[4.], entries=[(0, 1.)]),
            PricingArc(1, 2, 1., consumption=[2.]),
        ], resources=[resource],
    )])
    assert NetworkProblem.from_json(model.to_json()) == model
    result = model.solve(stabilize=False)
    assert result.status == "Optimal"
    assert result.lp == pytest.approx(3.)
    assert result.routes[0].resources == (10.,)
    # Bounds entirely inside the gap must not turn the union into its hull.
    blocked = replace(resource, bounds=((0., 20.), (3., 7.), (0., 20.)))
    invalid = replace(model, networks=(replace(model.networks[0], resources=(blocked,)),))
    try:
        output = invalid.solve(stabilize=False)
    except ValueError:
        return
    assert output.status != "Optimal"
