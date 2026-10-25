# PyBaldes API reference

[PyBaldes documentation](README.md)

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
    feasibility: Tuple[FeasibilityReport, ...] = ()
)
```

## Routing function and methods

```python
solve_root(instance: Union[Instance, str, PathLike], *, problem_type: str='vrptw', customers: Optional[int]=None, ng: int=8, reference_ng: bool=False, rounding: Optional[str]=None, capacity_cuts: bool=False, two_path_cuts: bool=False, rank_one_cuts: bool=False, stabilize: bool=True, master_solver: str='clp', master_threads: int=0, bound_mode: str='numerical', clp_scaling: str='equilibrium', clp_keep_work_areas: bool=True, verbose: bool=False, log_interval: int=10, hgs_time: Optional[float]=None, hgs_columns: Union[str, int]='best', hgs_seed: int=0, reduced_cost_fixing: bool=True, upper_bound: float=math.inf, time_limit: float=120.0, max_iterations: int=10000, max_cut_rounds: Optional[int]=None, pricing_options: Optional[Mapping[str, float]]=None) -> RootResult
```

```python
# Instance
from_file(cls, path: Union[str, PathLike]) -> Instance
truncated(self, customers: int) -> Instance
solve_root(self, **options) -> RootResult
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
install(self, solver: Solver) -> None
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
    ng: Optional[Tuple[Tuple[int, ...], ...]] = None,
    coupled_resources: Tuple[Mapping[str, Any], ...] = (),
    elementary: bool = False
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
    networks: Tuple[PricingNetwork, ...]
)
```

```python
NetworkRowCoefficients(
    network: int,
    arcs: Tuple[float, ...],
    per_route: float = 0.0
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
    snapshot: Optional[NetworkNode] = field(default=None, repr=False, compare=False)
)
```

## Network function and methods

```python
solve_networks(problem: NetworkProblem, *, tree: bool=False, network_pricing_schedule: str='full-sweep', network_unpromising_limit: int=1, network_branching: str='resource-first', enumerate: bool=False, enumeration_route_limit: int=10000, enumeration_work_limit: int=10000000, enumeration_stored_arc_limit: int=1000000, enumeration_time_limit: float=1.0, enumerated_mip: bool=False, mip_time_limit: float=1.0, mip_node_limit: int=100000, mip_objective_cutoff: Optional[float]=None, time_limit: float=120.0, max_iterations: int=100000, node_limit: int=10000, stabilize: bool=True, proximal: Optional[Mapping[str, Any]]=None, bound_mode: str='numerical', artificial_cost: float=2.76, log_interval: Optional[int]=None, pricing_options: Optional[Mapping[str, float]]=None, synchronized_services: Tuple[SynchronizedService, ...]=()) -> NetworkResult
```

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
- [Routing bindings](../../python/rust/src/root.rs)
- [Network bindings](../../python/rust/src/mp.rs)

The wrapper source determines routing/network defaults. Native binding source
determines feature gates and runtime behavior. Type declarations aid editors;
they do not supersede build flags or native validation.
