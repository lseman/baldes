# pyBaldes quickstart

[pyBaldes documentation](README.md) · [Installation](installation.md)

## Solve a small routing relaxation

This example uses only in-memory data. No instance download is required:

```python
from pybaldes import Customer, Instance

instance = Instance(
    name="triangle",
    capacity=2,
    customers=[
        Customer(0, 0, due=100),  # depot at index 0
        Customer(1, 0, demand=1, due=100),
        Customer(0, 1, demand=1, due=100),
        Customer(1, 1, demand=1, due=100),
    ],
    vehicles=3,
)
result = instance.solve_root(
    problem_type="vrptw",
    capacity_cuts=True,
    time_limit=30,
    reduced_cost_fixing=False,
)
print(result.status, result.lp, result.bound)
for route in result.routes:
    print(route.value, route.cost, route.nodes)
```

With a converged solve, the cut-strengthened objective is approximately
`5.4142135624`. Route values are LP column weights. A value such as `0.5` means
the relaxation uses half a column; it is not half a vehicle to dispatch.
`Optimal` describes convergence of this LP, not proof of an integer VRP optimum.

The returned graph uses source 0 and a separate terminal depot. For this example,
customer IDs are 1, 2, 3 and the terminal ID is 4. Do not index
`instance.customers[4]`; map the terminal back to depot 0 for visualization.

## Read a bundled instance

Run from the repository root:

```python
import pybaldes

result = pybaldes.solve_root(
    "instances/Solomon/R101.txt",
    customers=10,
    rounding="none",
    time_limit=30,
)
print(result.status, result.numerical_bound, result.certified_bound)
```

For a CVRPLIB file, explicitly select the capacity-only model:

```python
from pybaldes import Instance

instance = Instance.from_file("instances/cvrplib/A-n32-k5.vrp")
result = instance.solve_root(problem_type="cvrp", time_limit=30)
```

Python defaults to the instance's distance convention. CLI routing defaults to
truncation to tenths. Match `rounding` explicitly when comparing the interfaces.

## Export a result

```python
from pathlib import Path

Path("root-result.json").write_text(result.to_json(), encoding="utf-8")
for step in result.history:
    print(step.iteration, step.lp, step.minimum_reduced_cost, step.exact)
```

Unknown numerical quantities can be infinite in Python. `RootResult.to_json()`
converts non-finite floats to JSON `null`; `to_dict()` preserves them.

## Price a path directly

Use the low-level interface when you own the master or need a constrained path:

```python
from pybaldes import Solver

solver = Solver(3, 1)
solver.set_source(0)
solver.set_sink(2)
load = solver.new_resource("load", 0)
a = solver.new_arc(0, 1, -2.0)
b = solver.new_arc(1, 2, 1.0)
solver.set_arc_consumption(a, load, 1.0)
solver.set_arc_consumption(b, load, 1.0)
for vertex in range(3):
    solver.set_vertex_bounds(vertex, load, 0.0, 2.0)
    solver.set_ng_neighborhood(vertex, [vertex])
minimum, routes, exact = solver.price_with_status(10, exact=True)
assert exact
assert minimum == -1.0
assert routes[0].nodes() == [0, 1, 2]
print(routes[0].original_cost, routes[0].resource_consumption(load))
```

This solves the configured pricing domain. It does not create customer-covering
rows, solve a routing master, or independently certify a master lower bound.
See [pricing](pricing.md) for repeated dual updates and [networks](networks.md)
for a master managed by pyBaldes.
