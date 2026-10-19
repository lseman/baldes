# Python: pybaldes

[Documentation index](README.md) · [Command-line usage](cli.md)

Commands below run from the repository root. Python 3.9+, Rust/Cargo and a
C++14 compiler are required to build the default package from source.

The Python distribution and import are both named `pybaldes`. It contains the
low-level `Solver`, `Route`, and `Phase` bindings and direct Rust-backed VRPTW/CVRP
root solving. The default build includes the Clp master, built from the same
`vendor/` submodules as the Rust CLI:

```bash
git submodule update --init
python -m pip install .
```

```python
import pybaldes

result = pybaldes.solve_root(
    "instances/R101.txt",
    problem_type="vrptw",  # or "cvrp" for CVRPLIB .vrp files
    customers=25,
    capacity_cuts=True,
    rank_one_cuts=True,
    time_limit=120,
)
print(result.status, result.lp, result.bound, result.cuts)
for route in result.routes:
    print(route.value, route.cost, route.nodes)
```

`Instance.from_file(path)` reads Solomon-format and CVRPLIB `.vrp` files. You can also build inputs
in memory:

```python
from pybaldes import Customer, Instance

instance = Instance("small", capacity=2, customers=[
    Customer(0, 0),  # depot, zero demand
    Customer(1, 0, demand=1),
    Customer(0, 1, demand=1),
])
result = instance.solve_root(problem_type="cvrp", capacity_cuts=True)
```

CVRP ignores time windows and service times; in-memory VRPTW data requires finite
`ready`/`due` windows, including the depot. `vehicles` is retained as input
metadata; the current root model uses the customer count as its fleet upper
bound, matching the Rust CLI.

## Solve options and results

Options include `ng`, `reference_ng`, `rounding` (`none`, `nearest`, `tenth`, `ceil`),
`stabilize`, `reduced_cost_fixing`, `upper_bound`, `max_iterations`,
`max_cut_rounds`, and a `pricing_options` dictionary of BALDES option values.
Disabling reduced-cost fixing also disables enumeration. Results include positive
LP routes, duals, iteration history, cut/fixing counts, and timers;
`result.to_dict()` and `result.to_json()` export them. `to_dict()` retains
infinite values; `to_json()` converts non-finite values to `null`.
Cut counts include generated rows later retired.
These are root LP relaxations; `Optimal` does not certify an integer VRP optimum.

| Option | Default | Meaning |
| --- | --- | --- |
| `master_solver` | `"clp"` | Select `"clp"`, native `"ipm"`, or `"highs"` (requires its build feature); IPM disables stabilization |
| `verbose` | `False` | Print column-generation progress to stderr |
| `log_interval` | `10` | Positive iteration interval for progress when verbose |
| `hgs_time` | `None` | Positive HGS search budget in seconds before the root solve, separate from `time_limit` |
| `hgs_columns` | `"best"` | Seed the best plan, `"all"` validated encountered routes, or a positive integer of cheapest new routes |
| `hgs_seed` | `0` | Random seed for HGS seeding |

For example, seed a native IPM root solve with HGS and log progress:

```python
result = pybaldes.solve_root(
    "instances/R101.txt", customers=25,
    master_solver="ipm", hgs_time=2, hgs_columns="all", verbose=True,
)
print(result.to_json())
```

Python root solves enable Kohl with `two_path_cuts=True`. This attaches time
resource 0 for capacity-cut registration;
ordinary capacity cuts retain the default resource -1. Both use the capacity
family's priority 3 at root and 1 below it, complement form and persistent subset pool.

## Build variants and development

For pricing alone, build without Clp:

```bash
python -m pip install . --config-settings=build-args=--no-default-features
```

`pybaldes.has_clp` reports backend availability; root solving raises a clear error
in a pricing-only build. The wheel uses Python's stable ABI for Python 3.9+.
For local development, `maturin develop --release` installs the mixed package.
See the [Python validation guide](testing.md#python) for test commands.

To include HiGHS, install its shared library as described in the
[build guide](build.md), then install with its feature:

```bash
python -m pip install . --config-settings=build-args="--features highs"
```
