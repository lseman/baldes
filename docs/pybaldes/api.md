# pyBaldes API reference

[pyBaldes documentation](README.md)

The public package exports the names below. The `math` module supplies `math.inf`
in default expressions. Import them from `pybaldes`.
Underscore-prefixed native functions are implementation details. Signatures
below follow the Python wrappers and binding declarations in this checkout;
`self` and `cls` are shown explicitly. Build flags can affect native availability.

## Package metadata

| Name | Value or meaning |
| --- | --- |
| `__version__` | Python package version; separate from the BALDES 2.0 project name |
| `has_clp`, `has_highs`, `has_gurobi` | Compiled backend flags |
| `resource_dimensions` | Maximum active scalar coordinates in this extension |
| `ng_memory_capacity` | Compiled per-vertex NG memory capacity |

## Routing constructors and results

See [routing](routing.md) for field contracts and [results](results.md) for
objective/bound interpretation. Frozen dataclasses use these declared fields:


```python
Customer(
    x: float,
    y: float,
    demand: float = 0.0,
    ready: float = 0.0,
    due: float = math.inf,
    service: float = 0.0
)
```

```python
Instance(
    name: str,
    capacity: float,
    customers: Tuple[Customer, ...],
    vehicles: int = 1,
    rounding: str = 'none',
    distances: Optional[Tuple[Tuple[float, ...], ...]] = None
)
```

```python
RootRoute(
    value: float,
    cost: float,
    nodes: Tuple[int, ...],
    arcs: Tuple[int, ...]
)
```

```python
Iteration(
    iteration: int,
    lp: float,
    minimum_reduced_cost: float,
    new_columns: int,
    columns: int,
    exact: bool,
    phase: int,
    pricing_calls: int,
    alpha: float,
    misprices: int
)
```

```python
FeasibilityReport(
    status: str,
    iterations: int,
    pricing_calls: int,
    simplex_iterations: int,
    ipm_iterations: int,
    seconds: float,
    artificial_activity: Optional[float],
    lower_bound: Optional[float],
    certified_infeasible: bool
)
```

```python
RootResult(
    status: str,
    lp: float,
    bound: float,
    bound_mode: str,
    numerical_bound: float,
    certified_bound: float,
    lp_before_cuts: float,
    incumbent: float,
    cuts: int,
    cut_rounds: int,
    iterations: int,
    pricing_calls: int,
    columns: int,
    master_columns: int,
    fixing_calls: int,
    enumerated_routes: int,
    pruned_by_bound: bool,
    duals: Tuple[float, ...],
    seconds: float,
    pricing_seconds: float,
    solver_seconds: float,
    routes: Tuple[RootRoute, ...],
    history: Tuple[Iteration, ...],
    simplex_iterations: int = 0,
    ipm_iterations: int = 0,
    feasibility: Tuple[FeasibilityReport, ...] = (),
    oracle_calls: int = 0
)
```

## Unified solve and routing methods

```python
solve(problem, *, rows=(), time_limit=120.0, max_iterations=100000,
      stabilize=True, stabilization_max_tree_depth=10000,
      bound_mode="numerical", pricing_options=None, **options
) -> Union[RootResult, NetworkResult]
```

`problem` accepts an `Instance`, a path, a `PricingNetwork`, an array of
`PricingNetwork` objects, or a `NetworkProblem`. A `.json` path reads a network
model; other paths read routing input. `rows=` supplies shared rows when passing
a scalar network or array; a complete `NetworkProblem` already owns its rows.
One network and a one-element array normalize to the same model and defaults.

Model-specific keywords are documented in [routing options](routing.md#solve)
and [network options](networks.md#solve-controls). Routing input returns a
`RootResult`; network input returns a `NetworkResult`. Routing constraint type
is inferred by default; `problem_type="vrptw"` or `"cvrp"` overrides it.

```python
# Instance
from_file(cls, path: Union[str, PathLike]) -> Instance
truncated(self, customers: int) -> Instance
solve(self, **options) -> RootResult
```

```python
# RootResult
to_dict(self) -> dict
to_json(self, *, indent: Optional[int]=2) -> str
```

## Native pricing signatures

See [pricing](pricing.md) for state reuse and exactness. These declarations
include the optional `set_vertex_time_windows` method, which only exists in a
build with `multi-time-windows`. Type annotations do not perform all native
validation. `new_resource` configures an already allocated resource slot.


### Solver

```python
__init__(self, n_vertices: int, n_resources: int) -> None
set_source(self, v: int) -> None
set_sink(self, v: int) -> None
set_graph_id(self, id: int) -> None
new_arc(self, tail: int, head: int, cost: float) -> int
new_resource(self, _name: str, kind: int) -> int
set_resource_type(self, resource_id: int, kind: int) -> None
set_arc_consumption(self, arc_id: int, resource_id: int, value: float) -> None
set_vertex_bounds(self, vertex_id: int, resource_id: int, lower: float, upper: float) -> None
set_vertex_time_windows(self, vertex_id: int, resource_id: int, windows: list[tuple[float, float]]) -> None
set_initial_resource(self, resource_id: int, value: float) -> None
set_ng_neighborhood(self, vertex_id: int, nodes: list[int]) -> None
add_elementarity_set(self, nodes: list[int]) -> int
attach_elementarity_to_node(self, vertex_id: int, set_id: int) -> None
set_pricing_controls(self, cutoff: float, max_labels: int, max_columns: int) -> None
set_option(self, name: str, value: float) -> None
get_option(self, name: str) -> float
@staticmethod
option_names() -> list[str]
@staticmethod
option_support(name: str) -> tuple[bool, str]
build_default_resource_buckets(self) -> None
update_reduced_costs(self, costs: list[float], duals: list[float]) -> None
set_reduced_costs(self, costs: list[float]) -> None
set_bidirectional_mode(self, mode: int) -> None
solve(self) -> Optional[tuple[float, Route]]
price(self, maximum: int) -> tuple[float, list[Route]]
price_with_status(self, maximum: int, *, exact: bool=False) -> tuple[float, list[Route], bool]
enumerate(self, maximum: int) -> tuple[float, int, list[Route]]
phase_count: int  # read-only property
get_phase(self, index: int) -> Phase
enumerated_count: int  # read-only property
get_enumerated(self, index: int) -> Route
```

### Route

```python
cost: float  # read-only property
original_cost: float  # read-only property
coupled_values: list[tuple[int, float]]  # selected paired descriptor/value pairs
nb_nodes: int  # read-only property
is_certified: bool  # read-only property
nodes(self) -> list[int]
arcs(self) -> list[int]
resource_consumption(self, resource_id: int) -> float
```

### Phase

```python
check_ng: bool  # read-only property
allow_jump_arcs: bool  # read-only property
labels_per_bucket: int  # read-only property
```

## EncodedFunction signatures

```python
@staticmethod
knots(points: list[tuple[float, float]]) -> EncodedFunction
@staticmethod
slope_segments(prepared_pairs: list[tuple[float, float]], policy: str='strict_domain') -> EncodedFunction
@staticmethod
affine(intercept: float, slope: float, lower: float, upper: float, policy: str='strict_domain') -> EncodedFunction
@staticmethod
from_dict(definition: dict[str, Any]) -> EncodedFunction
to_dict(self) -> dict[str, Any]
value(self, argument: float) -> Optional[float]
encoding: str  # read-only property
policy: Optional[str]  # read-only property
```

## ResourceModel signatures

```python
add_paired_piecewise_linear(self, initial_lower: float, initial_upper: float, initial: list[tuple[float, float]], arrivals: list[list[tuple[float, float]]], increments: list[list[tuple[float, float]]], lowers: list[list[tuple[float, float]]], uppers: list[list[tuple[float, float]]], waiting: list[int], mode: str='feasibility') -> int
add_interval_cost(self, initial_lower: float, initial_upper: float, lower: float, upper: float, increments: list[float], functions: list[list[tuple[float, float]]]) -> int
add_slope_interval_cost(self, initial_lower: float, initial_upper: float, lower: float, upper: float, increments: list[float], functions: list[EncodedFunction], policy: str='strict_domain') -> int
add_slope_paired(self, initial_lower: float, initial_upper: float, initial: EncodedFunction, arrivals: list[EncodedFunction], increments: list[EncodedFunction], lowers: list[EncodedFunction], uppers: list[EncodedFunction], waiting: list[int], mode: str='feasibility', policy: str='strict_domain', normalization: str='absolute_final_value') -> int
__init__(self) -> None
install(self, solver: Solver, *, front_program_domain: bool=False) -> None
set_paired_waiting_base_resource(self, resource: int, base_resource: int | None) -> None
set_paired_machine_inputs(self, resource: int, arcs: list[dict[str, Any]] | None) -> None
set_paired_machine_root(self, resource: int, root: dict[str, Any] | None) -> None
set_paired_machine_terminal_bounds(self, resource: int, bounds: tuple[float, float] | None) -> None
add_selection(self, initial: int, allowed: list[int], vertices: list[int] | None=None, relaxed: bool=False) -> int
add_simultaneous_pickup_delivery(self, capacity: float, deliveries: list[float], pickups: list[float], tolerance: float=1e-06) -> int
add_cumulative_cost(self, initial: float, increments: list[float], weights: list[float], sink_upper: float | None=None, source_upper: float | None=None) -> int
add_custom_resource(self, increments: list[float], weights: list[float], forward_bound: float | None=None, backward_bound: float | None=None) -> int
add_piecewise_linear_resource(self, initial: float, lower: float, upper: float, functions: list[list[tuple[float, float]]]) -> int
add_piecewise_linear_cost(self, parent: int, functions: list[list[tuple[float, float]]]) -> int
add_piecewise_linear_dependent(self, parent: int, initial: float, lower: float, upper: float, functions: list[list[tuple[float, float]]]) -> int
add_encoded_piecewise_linear_resource(self, initial: float, lower: float, upper: float, functions: list[EncodedFunction]) -> int
add_encoded_piecewise_linear_cost(self, parent: int, functions: list[EncodedFunction]) -> int
add_encoded_piecewise_linear_dependent(self, parent: int, initial: float, lower: float, upper: float, functions: list[EncodedFunction]) -> int
add_dependent(self, parent: int, initial: float, lower: float, upper: float, offsets: list[float], factors: list[float]) -> int
```

`set_paired_machine_inputs` stores explicit prepared inputs for a
`front_algebra` slope-paired resource in physical-arc insertion order. Each
arc dictionary contains `tail_limit`, `head_limit`, `waiting_lower`,
`waiting_upper`, `waiting_rate`, and the five function fields
`forward_mapping`, `inverse`, `sum_vector`, `dependent_lower`, and
`dependent_upper`. Each function is a dictionary `{"pairs": [...]}` of
prepared `(value, coordinate)` and `(slope, endpoint)` pairs. Pass `None` to
remove a mapping. Prepared-pair validation occurs at the setter; the physical
arc count and remaining graph constraints are checked atomically at `install`.
An optional `backward` dictionary supplies `tail_lower`, `tail_upper`,
`forward_at_tail_lower`, `mapped_upper`, and a prepared `dependent` function.
With `front_program_domain=True`, the partial machine path also requires a
separately prepared dependent root supplied by `set_paired_machine_root` in the
same `{"pairs": [...]}` format. Both setters accept the recovered two-pair
zero-width and three-pair leading-zero-width shapes. Unsupported machine
operations return an incomplete result; these inputs do not certify a route.
Use `set_paired_machine_terminal_bounds` with the explicit terminal base window
to install a backward machine root from the prepared sink-arc operands.

## Network constructors and results

See [networks](networks.md) for coefficient semantics and snapshot lifetime.
Type aliases `Tuple`, `Optional`, `Mapping`, `Union`, and `Any` below refer to
`typing`. `field(...)` is dataclass metadata, not an argument to pass yourself.
Construct result objects through a solve rather than manually filling their counters.


```python
SharedRow(
    sense: str,
    rhs: float
)
```

```python
PricingResource(
    kind: str = 'disposable',
    initial: Optional[float] = None,
    bounds: Optional[Tuple[Tuple[float, float], ...]] = None,
    time_windows: Optional[Tuple[Optional[Tuple[Tuple[float, float], ...]], ...]] = None
)
```

```python
PricingArc(
    tail: int,
    head: int,
    cost: float,
    consumption: Tuple[float, ...] = (),
    bounds: Tuple[Optional[Tuple[float, float]], ...] = (),
    entries: Tuple[Tuple[int, float], ...] = ()
)
```

```python
PricingNetwork(
    id: int,
    vertices: int,
    multiplicity: Tuple[float, float],
    arcs: Tuple[PricingArc, ...],
    resources: Tuple[PricingResource, ...] = (),
    source: int = 0,
    sink: Optional[int] = None,
    route_entries: Tuple[Tuple[int, float], ...] = (),
    activity_entries: Tuple[Mapping[str, Any], ...] = (),
    ng: Optional[Tuple[Tuple[int, ...], ...]] = None,
    coupled_resources: Tuple[Mapping[str, Any], ...] = (),
    elementary: bool = False,
    front_program_domain: bool = False,
    route_domain: Mapping[str, Any] = field(default_factory=lambda: {'kind': 'physical_simple'}),
    coupled_objective_coefficients: Tuple[Tuple[int, float], ...] = (),
    paired_waiting_base_resources: Tuple[Tuple[int, int], ...] = ()
)
```

```python
SynchronizedServiceMember(
    network: int,
    vertex: int
)
```

```python
SynchronizedService(
    id: int,
    resource: int,
    vehicles_required: int,
    reward: float,
    start: Tuple[int, int],
    members: Tuple[SynchronizedServiceMember, ...]
)
```

```python
NetworkProblem(
    rows: Tuple[SharedRow, ...],
    networks: Union[PricingNetwork, Tuple[PricingNetwork, ...]],
    pricing_variables: Tuple[Mapping[str, Any], ...] = ()
)
```

```python
NetworkRowCoefficients(
    network: int,
    arcs: Tuple[float, ...],
    per_route: float = 0.0,
    activities: Tuple[Tuple[int, float], ...] = ()
)
```

```python
NetworkResourceGroupMember(
    network: int,
    arcs: Tuple[int, ...]
)
```

```python
NetworkAssignment(
    shared_row: int,
    network: int,
    value: float
)
```

```python
NetworkRoute(
    network: int,
    route_index: Optional[int],
    value: float,
    cost: float,
    nodes: Tuple[int, ...],
    arcs: Tuple[int, ...],
    resources: Tuple[float, ...],
    resource_profile: Tuple[Tuple[float, ...], ...]
)
```

```python
NetworkEnumerationReport(
    attempts: int,
    complete: bool,
    networks: Tuple[Tuple[int, int], ...],
    routes: int,
    transferred: int,
    failure: Optional[str],
    coverage_cutoff: Optional[float] = None
)
```

```python
NetworkResult(
    mode: str,
    status: str,
    bound_mode: str,
    lp: Optional[float],
    bound: Optional[float],
    numerical_bound: Optional[float],
    certified_bound: Optional[float],
    incumbent: Optional[float],
    routes: Tuple[NetworkRoute, ...],
    fixed_routes: Tuple[NetworkRoute, ...],
    duals: Tuple[float, ...],
    iterations: int,
    pricing_calls: int,
    oracle_calls: Tuple[Tuple[int, int], ...],
    nodes: int,
    open_nodes: int,
    seconds: float,
    enumeration_reports: Tuple[NetworkEnumerationReport, ...],
    enumerated_mip_calls: int,
    enumerated_mip_nodes: int,
    enumerated_mip_failures: int,
    tree_processed_nodes: int = 0,
    strong_branching_trials: int = 0,
    mapped_history_ranked: int = 0,
    mapped_branch_evaluations: Tuple[MappedBranchEvaluation, ...] = (),
    original_variables: Tuple[Mapping[str, Any], ...] = (),
    snapshot: Optional[NetworkNode] = field(default=None, repr=False, compare=False)
)
```

`mapped_branch_evaluations` contains read-only records with `name`, `depth`,
`side`, `stage`, `kind`, `parent`, `estimate` and `distance`. Their record class is
an implementation type; inspect the returned records rather than importing it
from the package root.

## Network model and snapshot methods

Fresh network inputs use the same `solve` operation above. Its `**options`
include `tree`, network scheduling/branching, enumeration, mapped variables,
proximal stabilization and synchronization controls; see the complete
[network option tables](networks.md#solve-controls). `NetworkNode.solve` resumes
its retained model and accepts the corresponding snapshot options.

```python
# NetworkProblem
from_json(cls, text: str) -> NetworkProblem
from_file(cls, path: Union[str, PathLike]) -> NetworkProblem
to_json(self, *, indent: Optional[int]=2) -> str
solve(self, **options: Any) -> NetworkResult
```

```python
# NetworkNode
solve(self, **options: Any) -> NetworkResult
branch_assignment(self, shared_row: int, network: int, assigned: bool) -> NetworkNode
with_row(self, row: SharedRow, coefficients: Tuple[NetworkRowCoefficients, ...]) -> NetworkNode
with_resource_group_branch(self, packing_row: int, resource: int, threshold: float, upper: bool, members: Tuple[NetworkResourceGroupMember, ...]) -> NetworkNode
with_synchronized_start(self, id: int, lower: int, upper: int) -> NetworkNode
fix_route(self, route_index: int, quantity: int=1) -> NetworkNode
solve_enumerated_mip(self, *, time_limit: float=1.0, node_limit: int=100000, objective_cutoff: Optional[float]=None) -> Optional[NetworkResult]
fractional_assignments(self, tolerance: float=1e-06) -> Tuple[NetworkAssignment, ...]
```

```python
# NetworkResult
to_json(self, *, indent: Optional[int]=2) -> str
```

## Implementation sources

- [Package exports](../../python/pybaldes/__init__.py)
- [Routing wrappers](../../python/pybaldes/routing.py)
- [Network wrappers](../../python/pybaldes/networks.py)
- [Native type declarations](../../python/pybaldes/_core.pyi)
- [Pricing bindings](../../python/rust/src/lib.rs)
- [Resource bindings](../../python/rust/src/coupled.rs)
- [Unified solve wrapper](../../python/pybaldes/solve.py)
- [Native input dispatch](../../python/rust/src/frontend/mod.rs)
- [Routing bindings](../../python/rust/src/frontend/routing.rs)
- [Network bindings](../../python/rust/src/frontend/networks.rs)

The wrapper source determines routing/network defaults. Native binding source
determines feature gates and runtime behavior. Type declarations aid editors;
they do not supersede build flags or native validation.
