# Routing inputs and options

[pyBaldes documentation](README.md) · [Results](results.md)

## Customer and Instance

`Customer(x, y, demand=0.0, ready=0.0, due=math.inf, service=0.0)` is a frozen
value object. Coordinates define Euclidean distances unless an explicit matrix
is supplied. Time windows constrain service start; service duration contributes
to the next transition. The depot must be the first customer and have zero demand.

`Instance(name, capacity, customers, vehicles=1, rounding="none", distances=None)`
normalizes customers and matrix rows into tuples. Every customer must be a
`Customer`. Native solving performs additional numeric/model validation.

| Field | Contract |
| --- | --- |
| `name` | Model label |
| `capacity` | Positive vehicle capacity |
| `customers` | Depot followed by customers in internal index order |
| `vehicles` | Input metadata; the routing root uses customer count as its fleet upper bound |
| `rounding` | Distance convention used when no solve override is supplied |
| `distances` | Optional full depot/customer matrix; supplied weights determine distances |

For VRPTW, supply finite `ready` and `due` values, including for the depot.
The default infinite `due` is convenient for CVRP but is not a valid VRPTW
window. CVRP ignores time windows and service durations.

```python
from pybaldes import Customer, Instance

instance = Instance("matrix", 2, [
    Customer(0, 0), Customer(1, 0, demand=1), Customer(0, 1, demand=1),
], distances=[
    [0, 2, 3],
    [2, 0, 4],
    [3, 4, 0],
])
result = instance.solve(problem_type="cvrp")
```

`Instance.from_file(path)` accepts a string or path-like object and reads Solomon
or CVRPLIB `.vrp` data. Constraint type is inferred when solving;
use `problem_type` only to override it. `Instance.truncated(N)` returns a new instance
containing the first N customers plus the depot, truncating both matrix axes.
It requires `0 < N < len(instance.customers)` and preserves the original object.

## Distance conventions

| Value | Coordinate-derived distance |
| --- | --- |
| `"none"` | Unrounded Euclidean distance |
| `"nearest"` | TSPLIB nearest integer (`nint`) |
| `"tenth"` | Truncate to tenths |
| `"ceil"` | Round upward to an integer |

Solomon files default to `none`; CVRPLIB `EUC_2D` defaults to `nearest` and
`CEIL_2D` to `ceil`. Explicit CVRPLIB full/triangular edge-weight matrices retain
the supplied weights. A solve's `rounding=None` uses `instance.rounding`.
Match these conventions before comparing objective values. CVRPLIB route-length
constraints (`DISTANCE`) are not part of this routing interface.

## solve

`solve(instance, *, ...)` accepts an `Instance`, file path string, or path-like
object. `instance.solve(**options)` delegates to the same function. All
options below are keyword-only. The call requires a Clp-enabled build.

### Model and pricing

| Keyword | Default | Behavior |
| --- | --- | --- |
| `problem_type` | `None` | Infer from finite due times; `"vrptw"` or `"cvrp"` explicitly overrides it |
| `customers` | `None` | Keep the first N customers before solving |
| `ng` | `8` | NG neighborhood size; constrained by compiled memory |
| `reference_ng` | `False` | Use reference nearest-customer neighborhood construction |
| `rounding` | `None` | Override coordinate distance convention |
| `pricing_options` | `None` | Mapping of supported BALDES names to numeric values |
| `capacity_cuts` | `False` | Enable rounded capacity cuts |
| `two_path_cuts` | `False` | Enable capacity-family Kohl separation with time resource; VRPTW only |
| `rank_one_cuts` | `False` | Enable limited-memory rank-one cuts |
| `stabilize` | `True` | Dual-price smoothing; disabled by IPM |
| `stabilization_max_tree_depth` | `10000` | Shared stabilization depth gate; routing Python calls solve the root |
| `reduced_cost_fixing` | `True` | Reduced-cost fixing and route enumeration; disabling disables both |
| `upper_bound` | `math.inf` | Supplied incumbent objective used for fixing; use a valid feasible bound |

NG pricing can include repeated customers. Exact pricing of this configured
relaxation does not make every route elementary. Rank-one cut memory also
depends on build features. Inspect supported controls with
`Solver.option_names()` and `Solver.option_support(name)`; see the
[pricing option reference](../reference/options.md) for scope and values.

### Master and bound policy

| Keyword | Default | Behavior |
| --- | --- | --- |
| `master_solver` | `"clp"` | `"clp"`, `"ipm"`, `"highs"`, or `"gurobi"`; last two need features |
| `master_threads` | `0` | HiGHS/Gurobi worker budget; 0 uses available CPUs; Clp/IPM ignore it |
| `bound_mode` | `"numerical"` | Numerical bounds or independent `"certified"` validation |
| `clp_scaling` | `"off"` | `"off"`, `"equilibrium"`, `"auto"`, `"geometric"`, or `"ruiz"` |
| `clp_keep_work_areas` | `True` | Retain native work buffers across LP solves |

HiGHS uses a process-shared native scheduler: keep its worker budget consistent
within one process and configure it before the first solve. CPU affinity limits
the available worker count. A higher budget does not guarantee faster solving.
Selecting IPM disables stabilization. Gurobi additionally needs its runtime
library and a valid license. All backend choices still require the Clp model store.

### Heuristic seeding, limits, and logging

| Keyword | Default | Behavior |
| --- | --- | --- |
| `hgs_time` | `None` | Positive HGS time budget in seconds before column generation |
| `hgs_columns` | `"best"` | Best plan, `"all"` validated encountered routes, or positive integer count of cheapest new routes |
| `hgs_seed` | `0` | HGS random seed |
| `time_limit` | `120.0` | Root solve budget in seconds; separate from HGS |
| `max_iterations` | `100000` | Column-generation iteration limit |
| `max_cut_rounds` | `None` | Optional cut-round limit |
| `verbose` | `False` | Emit progress to stderr |
| `log_interval` | `10` | Positive interval between progress rows |

```python
from pybaldes import solve

result = solve(
    "instances/Solomon/R101.txt", customers=10, rounding="none",
    master_solver="ipm", hgs_time=0.5, hgs_columns="all", hgs_seed=7,
    capacity_cuts=True, bound_mode="certified", time_limit=30,
    verbose=True, log_interval=5,
)
print(result.status, result.bound, result.incumbent)
```

The HGS budget is additional work, so `time_limit` alone is not an end-to-end
wall-time cap. Failure to obtain a certificate leaves the certified bound unknown;
it never substitutes the numerical estimate. See [results](results.md).
