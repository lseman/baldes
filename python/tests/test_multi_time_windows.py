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
