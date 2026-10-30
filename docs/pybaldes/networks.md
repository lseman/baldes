# Shared-master network models

[pyBaldes documentation](README.md) · [Results](results.md)

Use independent pricing networks when several route families contribute to the
same master constraints. Network IDs identify fleets or families; they need not
be consecutive. Arc IDs and vertex IDs are local to a network. Shared-row IDs
are positions in `NetworkProblem.rows`, starting at zero.

## One network or a collection

Use the same `solve` function for scalar and array inputs:

```python
from pybaldes import PricingArc, PricingNetwork, solve

network = PricingNetwork(
    id=7, vertices=3, multiplicity=(0, 1),
    arcs=(PricingArc(0, 1, -2), PricingArc(1, 2, 0)),
)
a = solve(network, stabilize=False)
b = solve([network], stabilize=False)
assert a.lp == b.lp == -2
assert a.oracle_calls == b.oracle_calls
```

Pass `rows=rows` with a scalar/array, or use `NetworkProblem(rows, networks)`.
The latter accepts a scalar network or an array and always stores a tuple.
`pricing_variables` optionally registers original pricing-variable mappings;
see the [model compilation contract](../solver/models.md#original-pricing-variable-compilation-rust).

## Build a complete model

```python
from pybaldes import (
    NetworkProblem, PricingArc, PricingNetwork, PricingResource, SharedRow,
)

problem = NetworkProblem(
    rows=[SharedRow("eq", 1.0)],
    networks=[PricingNetwork(
        id=19, vertices=3, multiplicity=(0.0, 1.0),
        arcs=[
            PricingArc(0, 1, 2.0, consumption=(1.0,), entries=((0, 1.0),)),
            PricingArc(1, 2, 1.0, consumption=(0.0,)),
        ],
        resources=[PricingResource(bounds=((0.0, 2.0),) * 3)],
    )],
)
root = problem.solve(stabilize=False)
assert root.status == "Optimal" and root.lp == 3.0
print(root.routes[0].network, root.routes[0].nodes)
```

`NetworkRoute.coupled_values` is an immutable tuple of paired descriptor/value
pairs retained with the scalar `resource_profile`. Values describe the current
selected physical dependent state, independently of the objective coefficient;
they are not full control trajectories or standalone feasibility certificates.

The only covering route costs 3, contributes one unit to shared row 0, consumes
one load unit, and uses one copy of network 19's allowed route multiplicity.
`problem.solve(**options)` delegates to `solve(problem, **options)`.
Both require Clp.

## Model reference

| Type | Fields and defaults |
| --- | --- |
| `SharedRow` | `sense`, `rhs`; sense is `"eq"`, `"le"`, or `"ge"` |
| `PricingResource` | `kind="disposable"`, `initial=None`, `bounds=None`, `time_windows=None` |
| `PricingArc` | `tail`, `head`, `cost`, `consumption=()`, `bounds=()`, `entries=()` |
| `PricingNetwork` | `id`, `vertices`, `multiplicity`, `arcs`, `resources=()`, `source=0`, `sink=None`, `route_entries=()`, `activity_entries=()`, `ng=None`, `coupled_resources=()`, `coupled_objective_coefficients=()`, `paired_waiting_base_resources=()`, `paired_machine_inputs=()`, `paired_machine_roots=()`, `paired_machine_terminal_bounds=()`, `paired_machine_waiting_rates=()`, `paired_front_programs=()`, `paired_owner_registration_bounds=()`, `elementary=False`, `route_domain=None`, `front_program_domain=False` |
| `NetworkProblem` | `rows`, scalar/array `networks`, optional `pricing_variables` |

Scalar `kind` accepts `"disposable"` or `"equality"`. Resource `bounds` has one
`(lower, upper)` pair per vertex when supplied; `initial` overrides the source
initial value. Optional windows are described in [resources](resources.md).
`PricingArc.consumption` follows scalar coordinate order. Arc `bounds` entries
are optional `(lower, upper)` pairs constraining head states. `entries` contains
`(shared_row_index, coefficient)` pairs for that arc.

For a route p in network k, an additive shared-row coefficient is the sum of its
arc entries plus `route_entries` once per route. Multiplicity `(lower, upper)`
bounds the sum of route variables in that network. Shared coverage rows and
fleet multiplicity are different constraints. `sink=None` selects the last
vertex. `ng` supplies per-vertex neighborhoods; `elementary=True` selects an
elementary network domain. `coupled_resources` accepts tagged descriptors.

`paired_front_programs` supplies a sequence of complete program dictionaries,
with one `resource` index per entry. Declare the corresponding coupled resource
as `{"kind": "paired_front", "mode": "cost_unbounded"}` (or `feasibility` /
`cost_bounded`). Each program supplies source/terminal windows, a root and one
arc mapping per physical arc. See [paired front programs](../extensions/coupled-resources.md#paired-front-programs-and-waiting)
for the complete schema. Preparation selects the front-program domain.

`paired_waiting_base_resources=((coupled_index, scalar_index), ...)` derives
head waiting windows from the intersection of scalar vertex and arc bounds.
Invalid indices, duplicate coupled indices, empty intersections and disconnected
scalar time windows are rejected.

The lower-level `paired_machine_inputs`, `paired_machine_roots` and
`paired_machine_terminal_bounds` fields accept separately prepared operands.
Do not combine those fields with a complete program for the same resource.
Changing preparation invalidates existing pricing state.

`coupled_objective_coefficients=((resource_index, coefficient), ...)` sets
finite paired owner objective coefficients of either sign; omitted resources
use one. Raw front pricing keeps its unit charge and minimum-front selection.
Original route costs apply the owner coefficient to the recorded selected
dependent value without initial-minimum subtraction. Each resource has one
objective owner: omit a graph coefficient when an original `pricing_variables`
entry binds that resource.

The Rust interface additionally supports independent master variables and
imported-column domains. Those registrations are not fields in the CLI/Python
network JSON schema. `pricing_variables` registers original pricing-variable
identities; it does not add arbitrary static master variables. See the
[shared model guide](../solver/models.md) and [column imports](../solver/column-imports.md)
for the Rust-only contracts.

An original variable can use `accumulated_resource` instead of `arcs` to bind
the intrinsic charge of a custom/cumulative coupled descriptor. Its `cost`
replaces the descriptor's unit objective coefficient; `entries` adds that
intrinsic activity to shared rows. Reported activity remains independent of the
objective coefficient. Accumulated registrations support continuous variables,
integer registrations with complete exact-integer path proofs, and explicit
original-variable boxes whose envelopes can be certified
over the complete bounded graph relaxation. Direct network
`activity_entries` use objects with `resource`, `row` and `coefficient` fields.
See [original variable compilation](../solver/models.md#original-pricing-variable-compilation-rust)
for the shared contract.

`accumulated_resource` also accepts cost-bearing paired fronts with any finite
`cost`. The reported value comes from the selected column, independently of the
objective coefficient. Paired `entries` and network `activity_entries` use that
selected value. Their dual terms do not change raw front pricing. When owner
terms differ from raw pricing, completion alone cannot certify the owner
objective; a solve may report `Stalled` with an unknown bound.

Paired integer registration is retained as metadata. It does not constrain each
path's selected activity to an integer or supply a projected integer branching
rule. Feasibility-only paired fronts cannot bind an original variable.

`paired_owner_registration_bounds=((resource_index, (lower, upper)), ...)`
supplies owner preprocessing bounds, with `None` for an unbounded endpoint.
These inputs are separate from per-path resource windows and shared activity
rows. A positive propagated lower bound can fail the pricing callback with an
explicit unsupported-bound error. They do not truncate individual path values.

Only one paired program can be installed, alone or with standard graph
resources. Combinations with other coupled descriptors, including custom and
cumulative resources, and a second paired program return explicit unsupported
composition errors. These combinations are solver limitations.

## Model JSON

```python
from pathlib import Path
from pybaldes import NetworkProblem

text = problem.to_json(indent=2)
restored = NetworkProblem.from_json(text)
assert restored == problem
Path("model.json").write_text(text, encoding="utf-8")
restored = NetworkProblem.from_file("model.json")
```

The same model format is accepted by `baldes solve model.json`. JSON rejects
non-finite numeric literals; an unlimited multiplicity upper bound uses `null`
as described in the shared model contract. Synchronization definitions are solve options, not fields in
`NetworkProblem.to_json()`; save them separately for reproducibility.

## Solve controls

All options to `solve` are keyword-only.

| Keyword | Default | Meaning |
| --- | --- | --- |
| `tree` | `False` | Solve a root LP or search a shared-master integer tree |
| `network_pricing_schedule` | `"full-sweep"` | `full-sweep`, `cyclic`, `sorted`, `promising-first`, `unvisited-first` |
| `network_unpromising_limit` | `1` | Positive queue batch budget |
| `network_branching` | `"multiplicity-first"` | `multiplicity-first`, `resource-first` or `phased` branching policy |
| `stabilize` | `True` | Dual smoothing when no explicit proximal configuration replaces it |
| `proximal` | `None` | Explicit proximal bands and optional multi-point policy |
| `bound_mode` | `"numerical"` | Numerical or certified lower bounds |
| `artificial_cost` | `2.76` | Positive artificial-column configuration; not an incumbent bound |
| `time_limit` | `120.0` | Solve time budget in seconds |
| `max_iterations` | `100000` | Column-generation iteration limit |
| `node_limit` | `10000` | Tree node limit |
| `log_interval` | `None` | Optional positive progress interval |
| `pricing_options` | `None` | Numeric BALDES option mapping |
| `synchronized_services` | `()` | Common-start optional service definitions |

A schedule changes which networks are priced first; early improving columns do
not establish that all networks have completed pricing. The solver coordinates
the complete pricing checkpoint needed for bounds.

Example proximal configuration:

```python
root = problem.solve(
    network_pricing_schedule="promising-first",
    proximal={"bands": [{"half_width": 0.5, "penalty": 1.0}], "multi_point": True},
)
```

`penalty=None` selects a hard band. Use positive validated widths/penalties;
see [proximal pricing](../solver/models.md#proximal-pricing) for the mathematical
contract. Root and tree APIs have different option sets from routing
`solve`; there is no `master_solver` keyword here.

### Additional tree and stabilization controls

| Keyword | Default | Behavior |
| --- | --- | --- |
| `network_promising_limit` | `1000000` | Promising-network queue budget |
| `stabilization_max_tree_depth` | `10000` | Disable stabilization at this depth |
| `network_tree_search` | `"default"` | `default`, `best-bound`, `bound-priority`, `depth-bound-priority` |
| `network_strategy2_rounding` | `"unrounded"` | Queue-key rounding: `unrounded`, `guarded-tolerance`, `exact-ceil` |
| `network_open_node_limit` | `1000` | Primary frontier capacity before overflow |
| `network_scoring_gap_tolerance` | `1e-6` | Absolute branch-score gap tolerance |
| `network_search_inactive_columns` | `False` | Reprice retained inactive columns before another oracle call |
| `network_share_inactive_columns` | `False` | Reprice columns shared by tree nodes/trials |
| `network_branch_priorities` | `None` | Named branch-family priority mapping |
| `network_branch_directions` | `None` | Named generator directions for phased branching |
| `mapped_variables` | `()` | Registered original aggregate branch variables |
| `packing` | `None` | Packing-assignment registration |
| `variable_boxes` / `original_variable_boxes` | `()` | Verified variable envelopes |
| `mapped_strong_branching` | `True` | Evaluate admitted mapped branch proposals in trials |

These controls configure the shared lifecycle's model registration. Network
count does not select another engine. Routing-specific Python keywords belong
to [routing input](routing.md); a scalar `PricingNetwork` uses these model controls.

## Resume, branch, and add rows

A root result owns a `NetworkNode` snapshot. Child operations return new nodes
and preserve their parent. Handles are bound to their creating Python thread.

```python
from pybaldes import NetworkRowCoefficients, SharedRow

root = problem.solve()
parent = root.snapshot
assert parent is not None
# Add an additive row requiring at least one route across the networks.
coefficients = tuple(
    NetworkRowCoefficients(n.id, (0.0,) * len(n.arcs), per_route=1.0)
    for n in problem.networks
)
child = parent.with_row(SharedRow("ge", 1.0), coefficients)
child_result = child.solve(stabilize=False)
assert parent.solve().lp == root.lp
```

`NetworkRowCoefficients(network, arcs, per_route=0.0, activities=())` supplies
one coefficient per physical arc and optional `(coupled_descriptor, coefficient)`
activity terms. The constant is charged once per route. Supported deterministic
custom/cumulative activities use the same `with_row` lifecycle, including
`dynamic=True`, for existing and future columns. Paired front activities are
rejected because their selected split-dependent values do not define this
shared-row contract.

`parent.fractional_assignments(tolerance=1e-6)` returns `NetworkAssignment`
objects with `shared_row`, `network`, and `value`. A candidate can be branched:

```python
for candidate in parent.fractional_assignments():
    assigned = parent.branch_assignment(candidate.shared_row, candidate.network, True)
    excluded = parent.branch_assignment(candidate.shared_row, candidate.network, False)
    print(assigned.solve().status, excluded.solve().status)
    break
```

Use candidate IDs from the solved snapshot and supported original assignment
rows. Arbitrary shared rows need not define assignment disjunctions.
`fix_route(route_index, quantity=1)` fixes a route using the snapshot's route
handle. `quantity` must be a positive Python integer at most `2**53`; booleans
and fractional values are rejected. Use a non-None `route_index` from that
snapshot's routes, not an index into an unrelated result.

`with_resource_group_branch(packing_row, resource, threshold, upper, members)`
accepts `NetworkResourceGroupMember(network, arcs)` objects. The original packing
row must establish at most one occurrence. The operation has resource replay
and row compatibility preconditions, including enumeration for some resource-load
rows. See [grouped branches](../solver/models.md#grouped-accumulated-resource-branches)
for the shared boundary convention and restrictions.

## Integer search

```python
from pybaldes import NetworkProblem

# Bundled example, run from the repository root.
triangle = NetworkProblem.from_file("examples/networks/triangle.json")
root = triangle.solve(stabilize=False)
integer = triangle.solve(tree=True, time_limit=30, node_limit=1000)
assert root.lp == 1.5
assert integer.incumbent == 2.0
print(integer.status, integer.bound, integer.open_nodes)
```

`root.snapshot.solve(tree=True, ...)` starts a tree from that node's inherited
state. Automatic search supports its documented branching families and may
return `Incomplete` when they do not resolve the model. A tree result has no
resumable node snapshot. An incumbent remains distinct from a lower bound.

## Enumeration and integer transfer

| Keyword | Default | Meaning |
| --- | --- | --- |
| `enumerate` | `False` | Attempt full route-universe transfer at a valid pricing checkpoint |
| `enumeration_route_limit` | `10000` | Candidate route budget across networks |
| `enumeration_work_limit` | `10000000` | Collection/proof work budget |
| `enumeration_stored_arc_limit` | `1000000` | Stored route-arc budget |
| `enumeration_time_limit` | `1.0` | Per-attempt time budget in seconds |
| `enumerated_mip` | `False` | Use optional HiGHS integer transfer |
| `mip_time_limit` | `1.0` | Per-MIP time budget |
| `mip_node_limit` | `100000` | Per-MIP node budget |
| `mip_objective_cutoff` | `None` | Optional finite improvement cutoff |

```python
root = triangle.solve(enumerate=True)
for report in root.enumeration_reports:
    print(report.complete, report.routes, report.failure, report.coverage_cutoff)
```

Full transfer commits only when every enabled network completes. Unsupported
resources, ambiguous replay, or exhausted limits leave normal pricing available
and do not install a partial universe. A failed attempt may be retried by a later
solve with different limits. `NetworkNode.solve_enumerated_mip(time_limit=1.0,
node_limit=100000, objective_cutoff=None)` solves an available transferred copy
and may return `None` when unprepared. HiGHS support is required for integer solving.

With `tree=True, enumerated_mip=True`, the solver can use full transfer or a
cutoff-bounded improvement domain. Inspect each report's `coverage_cutoff`:
`None` means full-domain coverage; a number restricts coverage to improvements
below it. Failure to improve an arbitrary cutoff is not original-model
infeasibility and does not supply a feasible incumbent. Independent certificates
and numerical MIP bounds remain distinct. See the
[enumeration contract](../solver/models.md#complete-enumeration-and-integer-transfer).

## Synchronized optional service

`SynchronizedService(id, resource, vehicles_required, reward, start, members)`
defines a job served by distinct vehicle copies at a common integer service start.
Members are `SynchronizedServiceMember(network, vertex)` values. Service is
optional and its reward is counted once in the master, so exclude that reward
from route arc costs and remove ordinary unit-cover rows for this job.

```python
from pybaldes import SynchronizedService, SynchronizedServiceMember

fleets = NetworkProblem((), tuple(
    PricingNetwork(
        id=n, vertices=3, multiplicity=(0, 1),
        arcs=(PricingArc(0, 1, 0), PricingArc(1, 2, 0)),
        resources=(PricingResource(initial=0, bounds=((0, 0), window, (0, 100))),),
    ) for n, window in ((7, (0, 9)), (42, (5, 10)))
))
service = SynchronizedService(
    id=17, resource=0, vehicles_required=2, reward=1,
    start=(0, 10),
    members=(SynchronizedServiceMember(7, 1), SynchronizedServiceMember(42, 1)),
)
result = fleets.solve(tree=True, synchronized_services=(service,))
assert result.incumbent == -1
assert all(r.resource_profile[1][0] == 5 for r in result.routes)
```

The synchronized resource must be disposable with nonnegative consumption.
Put service duration on outgoing arcs. Start bounds are inclusive integers in
`[-2**52, 2**52]`; the unit determines the time grid. The synchronized axis is
a graph scalar resource; separate coupled resources may coexist. Resource-load
rows on the same axis are refreshed when synchronized start domains change.
The routing registration accepts bounded walks without cuts; its automatic cut
families require a hard vertex elementarity group on every interior vertex.
That restriction belongs to routing cut hooks and is independent of network count.
Root LP columns need not form a synchronized schedule; the tree validates the
joint integer witness.

A snapshot's service definitions are immutable. Narrow a start domain with
`with_synchronized_start(id, lower, upper)`; parents remain reusable. Complete
MIP is deferred while common start domains are non-singleton. The
[synchronization contract](../solver/models.md#synchronized-service-across-vehicles)
explains coverage equations and schedule validation.


Original-variable pricing envelopes can be passed to `solve` or a node solve:

```python
result = problem.solve(
    original_variable_boxes=({"network": 7, "id": 0, "lower": 0, "upper": 1},),
)
```

Each identity must occur in `problem.pricing_variables`. The bound must already
follow from the pricing graph; an unproved envelope raises an error. These
bounds guide pricing admission without adding a route restriction. Original
variable costs remain separate during this calculation even if they cancel on
shared physical arcs. See the [compilation contract](../solver/models.md#original-pricing-variable-compilation-rust).


Paired front routes retain their selected `coupled_values`. The selected value
must match a successful split join of the retained program. Explicit source
controls and route-control optimization are unsupported.
