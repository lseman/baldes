# Multi-pricing: one master, multiple pricing subproblems

[Documentation index](../README.md)

With `clp`, `master::mp::solve` solves a shared root and returns an owned
node snapshot that can be resumed or forked. Its pricing graphs share master
rows. Each `Network` owns a graph, a unique nonnegative ID and
nonnegative multiplicity bounds. Rust models may use a finite lower bound and
positive-infinite upper bound. Local arc IDs may overlap across
networks: route identities include the network ID and resource schedule.

The coordinator is called **multi-pricing** because one master coordinates
multiple pricing subproblems. A **network** is an individual subproblem's graph
and route family. Subproblems may share a vehicle type or represent different
types, depots or resources; heterogeneity is optional. This is independent of
using several CPU cores for a single pricing problem.

`Problem.rows` defines the shared constraints. `Network.arc_entries` contains
one list of `(shared_row, coefficient)` pairs per physical arc, in graph order.
`route_entries` adds coefficients charged once per route. The coordinator adds
the network multiplicity rows separately; callers do not include them in the
shared-row indices. Arc costs define original route costs. Variable-cost
mapping used internally by pricing does not modify the supplied graphs.
Physical multiplicity rows appear in input-network order: a finite upper row
comes first, then a lower row when the lower bound is positive. An infinite
upper bound adds no row. Shared rows follow these multiplicity rows in their
original order.

In Rust, use `Window { lower: 0., upper: f64::INFINITY }` for an unlimited route
count; `Window::new` remains the finite scalar-resource window constructor.
Original shared rows can prove a finite remaining count, which tightens future
columns and bound compensation and survives forks/fixing. If a negative reduced
cost has no finite count bound, the estimate remains unknown while pricing can
continue. A zero reduced cost does not multiply infinity. Truly unbounded
negative-cost route directions report an unbounded backend solve. JSON uses
`"multiplicity": [0, null]` for an unlimited upper count. The lower count must
remain finite and nonnegative.

The coordinator maintains independent pricing state, combines columns into one
restricted master and aggregates each network's multiplicity contribution and
stabilization subgradient. `Output.oracle_calls` counts individual oracle calls;
`Output.master.pricing_calls` counts batches. `Output.values` gives the current
restricted-master quantity for every route in `Output.routes`.

Phase estimates guide stabilization. Published lower bounds additionally use
graph-walk relaxations with downward arithmetic. If a negative cycle prevents
a resource-free bound, nonnegative scalar consumption and bounded resource
windows can justify a finite walk-length limit. The proof can combine dimensions
with different units: it sums their individual positive-arc occurrence limits.
Arcs consuming zero on every selected dimension must form an acyclic graph.
Thus different cycles may consume different resources. The fallback uses a bounded work
budget. Without a justified limit, or after arithmetic failure, budget exhaustion
or an unavailable required path, the certificate remains unknown. A finite
resource-aware bound can still be too weak to close the gap. In certified mode, unavailable certificates cannot close
the gap; numerical mode retains its separate exact-pricing estimate. Finite
artificial penalties
also do not certify original-model infeasibility. A stalled restricted LP can
still contain artificial variables; its objective and route quantities are
not then an original feasible solution.

## Node lifecycle

`Output<D>::snapshot` owns the solved master, generated-column registry, row
maps and one independent pricing solver per network. Its methods borrow the
parent and return a new child or solve result; solving either child leaves the
parent and sibling unchanged. Oracle counters in each output describe that
solve, while column tags stay stable across descendants.

- `snapshot.solve(options)` resumes all networks with a new work budget.
- `snapshot.with_row(row, coefficients)` appends a shared additive row.
  `RowCoefficients` identifies a network, gives one coefficient per original
  physical arc and a once-per-route coefficient. Omitted networks contribute
  zero. Retained master columns, the generated pool and future pricing all use
  the row. Callers are responsible for the validity of supplied cuts.
- `snapshot.fractional_assignments(tolerance)` aggregates each network's use of
  unit-cover shared rows. Assignment membership comes from row coefficients,
  so matching local arc or vertex IDs in different networks cannot alias.
- `snapshot.branch_assignment(shared_row, network_id, assigned)` builds the
  `use >= 1` or `use <= 0` child. Assignment rows must be unit equalities with
  nonnegative integer contributions. Both children price every enabled network.
- `snapshot.fix_route(route_index, quantity)` fixes additional integer copies
  of a registered original-feasible elementary route. It replays resources,
  subtracts every shared/branch/cut and multiplicity contribution, and carries
  the cost as an objective offset. Rows appended later also subtract this fixed
  contribution. `Output.fixed_routes` reports the fixed portion separately;
  `Output.values` contains only remaining quantities.

For example, after obtaining a coupled root:

```rust
let candidate = root.snapshot.fractional_assignments(1e-6)?[0];
let assigned = root.snapshot.branch_assignment(
    candidate.shared_row, candidate.network, true,
)?;
let excluded = root.snapshot.branch_assignment(
    candidate.shared_row, candidate.network, false,
)?;
let assigned_result = assigned.solve(options.clone())?;
let excluded_result = excluded.solve(options.clone())?;
let restored_parent = root.snapshot.solve(options)?;
```

Fresh roots still require an empty column registry; descendants own their warm
metadata internally. Pass pricing options through `Options.pricing_options`.
Scalar resource capacity is a const generic, defaulting to two; see
[scalar resources](../reference/resources.md).

## Assignment tree

`solve_assignment_tree(problem, AssignmentTreeOptions)` runs a best-bound tree
using network-aware snapshots. It tries assignment rows, declared integer master
variables, total network multiplicities, then original physical-arc activities. It returns network-tagged
integer route quantities, an incumbent, selected bound mode, frontier bound and
node/work-limit status. Integer witnesses replay original graph resources,
shared rows and original multiplicities with exact sums; objectives round upward.
Fixed routes are included in this replay. A rounded labeling resource profile
alone never authorizes an incumbent or fixed route.

Rust callers can run `snapshot.solve_assignment_tree(options)` after installing
shared nonlinear rows, fixing routes or restricting master variables. It forks
the snapshot and retains cuts, bounds, fixed contributions and complete route
universes. A zero-node run preserves only the inherited bound of the requested
bound mode; a numerical bound is not promoted to a certificate.

Every fallback split is an integer-valid floor/ceiling disjunction on an
original quantity. Continuous master variables are never branched as integers.
These families still do not cover every fractional route decomposition; such a
node remains `Incomplete`. An exact whole-original-flow contradiction can close
a node. Positive artificials, a stalled LP or incomplete certification alone do
not prove infeasibility. Complete-master integer deductions retain the explicit
integer-domain policy.
Node/time limits retain the open-frontier bound. Bounds on fixed-route nodes
are newly certified only when their RHS, offset and residual multiplicity
reductions are exact; otherwise valid inherited certificates remain available,
but a new original-domain bound stays unknown.

## Finite column imports

Rust callers can combine routing networks with explicitly declared finite
original-vector column domains using `mp::imported::solve`. See
[finite column imports](column-imports.md) for coordinate links, column fixing,
multiplicity bounds, identity mapping and complete-domain MIP.

## Independent master variables

`solve_with_master_variables(problem, variables, options)` adds independent
`MasterVariable` definitions to the shared master. Each definition has its own
ID, objective cost, signed bounds (finite, one-sided or free), integrality flag and shared-row
coefficients. IDs are independent of network and arc IDs. Root LPs relax
integrality; complete-domain MIP retains it. `Output.master_values` returns
original values by ID, separately from network-tagged routes.

For a model with no pricing subproblems, Rust callers can use
`solve_master(rows, variables, options)`. It solves the continuous relaxation and
returns the same snapshot API. Signed, one-sided, free and integer variables
retain their usual contracts. The empty pricing family is complete immediately:
`snapshot.solve_enumerated_mip(options)` needs no preceding enumeration, and
`snapshot.solve_assignment_tree(options)` branches integer master variables.
No dummy network is required. A `Problem::<D>` with an empty `networks` list is
also accepted by `solve_with_master_variables`, for any compile-time `D`.
An empty model has objective zero; original-row contradictions are available to
the tree as preprocessing proofs. This Rust API does not add master-variable
fields to the CLI/Python JSON schema.

Snapshots support `with_master_variable_bounds(id, bounds)` for restrictions or
fixing. `with_master_row(row, network_coefficients, master_coefficients)` couples
these variables to additive network activities;
`with_rank_one_master_row(row, floor_coefficients, master_coefficients)` couples
them to nonlinear floor rows. Existing parent restrictions remain in force.
All operations preserve independent sibling snapshots. The corresponding tree
entry point is `solve_assignment_tree_with_master_variables`.

Signed variables retain their original coefficients and costs. Their bounds
participate in joint propagation, and bounds tightened by original constraints
reach static master columns and publication certificates. Inferred positive
lower bounds and negative upper bounds are also published as original-variable
rows, so the LP, warm snapshots and MIP enforce those bounds. A bounded exact
row-reduction pass also combines shared constraints with linear equalities that
link master variables. For example, `a + b >= 1.5` and `m = a + b` imply
`m >= 1.5`; the optional integer policy strengthens this to `m >= 2` when `m`
is integer. The pass retains support-reducing sums/differences only, rejects
rounded coefficients/RHS values, and reserves most work for ordinary propagation.
It does not treat nonlinear row envelopes as equalities. Continuous master
values remain continuous when route quantities are made integer in MIP and
witness validation.

Infinite endpoints represent mathematical domains. An adverse dual point on an
unbounded component can leave the published bound unknown. Original shared rows
may supply a finite proof-only bound even with preprocessing disabled; this does
not change the snapshot or replace infinity with a finite modeling cap.

## Original activities, multiplicities and nonlinear rows

Rust snapshots expose `with_activity_bounds(bounds, coefficients)` to restrict
an aggregate original variable across networks. Each `RowCoefficients` supplies
arc coefficients and a once-per-route coefficient for its network. Equal lower
and upper bounds fix the activity. `with_multiplicity(network, bounds)` bounds
the total route count, including fixed copies. Both operations fork the node
and preserve restrictions in future pricing, descendants and enumeration.

Activity bounds constrain the aggregate master activity. For example, an upper
bound of `0.25` remains a fractional activity cap in the continuous relaxation.
It is not an arc-availability flag. Opting into integer-domain preprocessing can
strengthen a proved integer activity cap below one to zero. Pricing supports
are defined separately by the graph and registered restrictions.

`with_rank_one_row(row, coefficients)` installs a shared nonlinear row. Each
`RankOneCoefficients` defines a network's coefficient as
`floor(sum of visited vertex weights / denominator)`. Weights count each visit,
lie between zero and denominator minus one, and must be zero at both terminals.
Supported denominators are 1–7. The caller supplies a valid row. Fixed routes subtract their nonlinear
coefficient from its RHS, including when the row is installed after fixing.
Pricing uses full memory; master coefficients retain the actual physical arcs.
`with_registered_cut_row(row, coefficients)` accepts `RegisteredCutCoefficients`
for existing packing, covering, limited-memory, arc-weighted and strong k-path
descriptors. `with_registered_cut_master_row` also couples independent master
variables. Use physical vertex/arc IDs for the selected network and consistent
graph-specific supports. Weights must be nonnegative and below the denominator
(1–7); unsupported original-set mappings and incompatible descriptors are
rejected. The caller supplies a valid inequality; these methods do not separate
cuts automatically. Fixed-route shifts and future-column coefficients use the
registered memory contract.

`with_composite_cut_row(row, coefficients)` sums several registered descriptors
into one shared row, including multiple terms from the same network.
`with_composite_cut_master_row` couples this sum to independent master variables.
Repeated descriptors contribute repeatedly. Pricing applies the shared row dual
to every term; replay and fixed-route shifts use their summed coefficient.
This API expresses a sum, not an indicator that any term is present. The original
`with_registered_cut_row` still rejects duplicate network descriptors.

These Rust methods are not currently exposed in the JSON or Python model schema.

`with_presence_row(row, coefficients)` installs a caller-supplied valid nonlinear
row over original-variable events. Each `PresenceCoefficients` supplies a
network ID, a `PresenceRow` and event lists aligned with physical graph arcs.
`Mode::PairCount` counts satisfied stored pairs, including duplicates;
`Mode::AnyPair` contributes at most one. Repeated emission of an event does not
multiply its contribution. `with_presence_master_row` also couples independent
master variables. All row senses are supported. Fixing replays the actual
coefficient, including rows appended after the fix; pricing, snapshots and
complete enumeration/MIP use the same event definition.

Original-domain certificates keep active presence rows together in a bounded
walk-state search. It charges actual event-pair increments using directed
arithmetic, including duplicate pairs, repeated events and either dual sign.
Event IDs are local to each row. Sparse event compression avoids allocating by
the largest event ID. Resource-derived walk-length bounds can cover adverse
cycles; otherwise an adverse cycle leaves this improvement unknown. Work or
time exhaustion falls back to the existing conservative relaxation. These
certificates do not require complete enumeration.

Original-domain preprocessing drops nonnegative presence terms from `<=`
relaxations and bounds them by the maximum coefficient times multiplicity for
`>=` relaxations. When static terms exhaust the upper side of a `<=` or equality
row, preprocessing searches source-to-sink walks with zero presence coefficient.
It considers all pairs in that descriptor jointly, including events spread over
several arcs, and forbids physical arcs absent from every such walk. The search
ignores resources and elementarity, so its walk domain contains every feasible
route. Cycles and repeated events are supported; equivalent incidence states
at the same vertex are merged. Sparse event IDs do not allocate the entire
declared event domain. All searches share the preprocessing work budget. An
unfinished search publishes no additional walk-support exclusions; previously
proved exclusions remain valid. These deductions participate in the joint
flow/static-variable fixed point across pricing subproblems.

Certified bounds can be
looser than the numerical LP because their original-domain relaxation drops
positive presence penalties when joint presence-state certification exhausts its
work/time allowance; such a run can remain `Stalled` until a stronger
proof or complete enumeration closes the gap.

Registered cut rows also support original-domain zero-coefficient propagation.
When directed bounds prove that static contributions exhaust a nonnegative
row's `<=` budget, every registered kernel in that row must be zero. A bounded
walk search keeps the kernels' packing remainders jointly, including vertex or
physical-arc memory resets. Covering and strong-entry kernels reject support
hits; repeated composite descriptors keep their sum semantics. Arcs absent
from all zero-coefficient source-to-sink walks are forbidden before enumeration.
The search ignores route resources and elementarity, so its walk domain is a
safe superset. Interrupted searches publish no partial walk exclusions.

These deductions share the flow/resource/presence work budget and repeat to a
joint fixed point. Signed static compensation or a positive fractional row
budget prevents unjustified zero deductions. This is a necessary zero-budget
rule, not a general nonlinear optimizer or automatic cut separator.

`with_resource_knapsack_row(row, coefficients)` installs a shared `<=` row
using `ResourceKnapsackCoefficients` (network, resource coordinate, graph scale
and monotone nonnegative step cut). `with_resource_knapsack_master_row` also
couples independent master variables. This API supports nonnegative
consumption, initial usage and vertex/arc windows on a common power-of-two grid.
The grid spacing must lie between `2^-18` and `1`, and each value must use at
most `2^52` grid units. Integer schedules retain their existing range; fractions
such as `0.5` and `0.125` are supported when the complete schedule fits this
range. Finer binary fractions, negative values, and schedules requiring more
grid units are rejected before installation.
Disposable resources may wait; equality resources must meet their windows
without waiting. Coefficients use the earliest forward scalar schedule replayed
from physical arcs; cached route profiles cannot change them. Pricing uses graph
scaling and unrounded nonpositive duals.

Fractional schedules, nonzero initial usage, waiting windows, intermediate
capacity changes and equality resources select forward-only pricing for the
affected network. This
preserves the coefficient contract even if a caller requests bidirectional
search. Zero-initial integer disposable loads with uniform capacity retain the
existing bidirectional option.

Fixed routes shift these rows, and descendants, enumeration and mixed MIP
retain them. Complete-domain preprocessing can fix route quantities from the
step coefficients. Before enumeration, a row that permits only zero resource
penalty can also forbid original arcs whose consumption guarantees a positive
step coefficient. Directed bounds on static variables must first prove that no
negative static contribution can compensate those penalties. Flow conservation
then propagates the arc fixings. The original-domain flow and resource-zero
rules repeat while deductions enable each other, sharing one work budget.
`preprocessing_report().propagation.work_used` reports the logical work spent.
Incomplete budgeted passes retain valid partial deductions without claiming a
fixed point. Before complete enumeration, the original-domain certificate
may drop their nonnegative pricing penalty for a weaker valid lower bound;
this need not close the LP gap. These methods require caller-supplied valid
inequalities and do not automatically separate resource-knapsack cuts.

Node solves automatically propagate a joint arc-flow relaxation across networks.
It includes shared activities, flow conservation and route multiplicities;
nonlinear rows contribute safe activity envelopes. Proved-zero arc usage forbids
that arc in future pricing and fixes affected retained columns to zero. After
complete enumeration, a second pass can also deduce zero route quantities from
nonlinear coefficients. Missing future columns never justify these deductions.

Use `options.preprocessing.enabled = false` to disable propagation, or adjust
`options.preprocessing.work_limit`. `snapshot.preprocess()` runs the default
policy on a fork; `preprocess_with_options()` supplies a separate budget.
`preprocessing_report()` exposes aggregate arc and multiplicity bounds, work
completion, route caps, zero fixings, and reasons for skipping unsafe arithmetic.
Positive aggregate arc bounds are not per-route resource windows. After complete
transfer, proved positive route-quantity caps are installed in the master and
preserved through cleanup, regeneration and descendants. Fixing copies of a route
reduces its remaining cap. Upper-active columns may have negative reduced costs;
the numerical and certified bounds account for these caps. Inexact coefficient
aggregation or fixed-route reductions skip deductions. Exhausted work preserves
valid partial deductions. The default policy preserves the continuous LP and reports preprocessing
infeasibility without independently pruning a tree node. Set
`options.preprocessing.integer_domains = true` to use original integer route,
arc-usage and multiplicity domains, together with each master variable's declared
integrality. Continuous master variables remain continuous. With this explicit
policy, a proved contradiction can close an integer tree node even when its LP
relaxation is feasible. Reports distinguish inferred master-variable bounds
from network activity bounds. Disabling propagation later does not widen a
snapshot that already contains deductions; reuse an earlier ancestor for its
original continuous relaxation.

## Limitations

The supported lifecycle supports additive rows and explicit full-memory floor
rows through Rust. Automatic nonlinear cut separation,
reduced-cost-pruned enumeration with proved coverage,
other branching families and broader resource-aware
certificates remain open. Rust and Python expose the supported
shared-master node lifecycle. The existing single-network routing entry
points retain their current behavior.

## CLI

The `multi-pricing` command accepts an explicit JSON model, rather than a Solomon
instance. It uses Clp and supports both shared roots and the assignment tree:

```sh
cargo build --offline --features clp --bin baldes
target/debug/baldes multi-pricing examples/networks/triangle.json --json root.json
target/debug/baldes multi-pricing examples/networks/triangle.json --tree --json tree.json
# JSON alone on stdout, suitable for a pipe:
target/debug/baldes multi-pricing examples/networks/triangle.json --tree --json -
```

The example has shared LP optimum `1.5` and integer optimum `2`. `--time`,
`--iterations`, `--nodes`, `--bound-mode`, `--no-stabilization`,
`--artificial-cost`, `--option NAME=VALUE`, and `--log-interval` control solves.
Run `baldes multi-pricing --help` for details. The previous `networks` command remains an alias. Unsupported routing flags are errors.
A successful command means the run completed and produced a result: inspect
`status` and `open_nodes` for `NodeLimit`, `TimeLimit` or `Incomplete` outcomes.

A model contains `rows` and `networks`. Rows use `sense: "eq" | "le" | "ge"`
and a finite `rhs`. Each network has a unique nonnegative `id`, a `vertices`
count, `[lower, upper]` fleet `multiplicity`, and an `arcs` list. Optional
fields are `source` (default zero), `sink` (default last vertex), `resources`,
`route_entries` and `ng`. Vertex indices and shared-row indices are zero based.
Arc IDs are assigned consecutively in each network's input order, starting at
zero; overlapping IDs in different networks are supported.

Each arc provides `tail`, `head`, `cost` and optional `entries` pairs
`[shared_row, coefficient]`. Its optional `consumption` and `bounds` arrays
must match the resource count when nonempty. Arc bounds contain either
`[lower, upper]` or `null` for each resource. Missing consumption is zero.
Each resource has `kind: "disposable"` (default) or `"equality"`, an optional
`initial` value, and optional `bounds` with one `[lower, upper]` pair per vertex.
Default resource bounds are `[0, 100000]`. Network `route_entries` are charged
once per route. Optional `ng` contains one neighborhood list per vertex.
Resources may be empty; the maximum count is the build's scalar capacity.
Unknown fields, nonfinite values and inconsistent dimensions are rejected.

Result routes retain `network`, local `arcs`, `nodes`, cost, resource states and
`value`. A root's `route_index` identifies its registry entry for later fixing;
zero-valued registered routes are retained too. `fixed_routes` lists fixed
quantities separately. Unknown/nonfinite bounds and absent objectives are JSON
`null`, not `Infinity`. Root mode reports an LP relaxation; tree mode reports
an independently replayed integer witness when available. Their `mode` and
`bound_mode` fields preserve that distinction.

## Python

The dedicated [PyBaldes network guide](../pybaldes/networks.md) covers Python model construction, solve options, snapshots, and synchronization. The sections here specify the shared Rust/CLI/Python model contract.

Install the extension with Clp support (the Python build's default). The typed
inputs use the same JSON schema as the CLI, allowing a single model to drive
both interfaces:

```python
from pybaldes import NetworkProblem, solve_networks

problem = NetworkProblem.from_file("examples/networks/triangle.json")
root = solve_networks(problem)
assert root.lp == 1.5
integer = problem.solve(tree=True)
assert integer.incumbent == 2.0

candidate = root.snapshot.fractional_assignments()[0]
left = root.snapshot.branch_assignment(
    candidate.shared_row, candidate.network, True,
)
right = root.snapshot.branch_assignment(
    candidate.shared_row, candidate.network, False,
)
assert left.solve().lp == 2.0
assert right.solve().lp == 2.0
assert root.snapshot.solve().lp == 1.5
```

Models can also be constructed with `NetworkProblem`, `SharedRow`,
`PricingNetwork`, `PricingArc` and `PricingResource` dataclasses. Use
`problem.to_json()` to export them for the CLI. `NetworkResult.to_json()`
serializes owned result data and excludes its native snapshot.

Root snapshots additionally expose `with_row(SharedRow(...), coefficients)`
with `NetworkRowCoefficients(network, arcs, per_route=0)`, and
`fix_route(route_index, quantity=1)`. Child operations return independent
`NetworkNode` objects; `node.solve(**options)` returns a new result and snapshot.
Quantities fixed in Python must be positive integers. Node handles belong to
one Python thread and root/node calls keep the GIL. Fresh `tree=True` calls
release the GIL and return owned results without a resumable snapshot.

The root/tree solve accepts `time_limit`, `max_iterations`, `node_limit`,
`stabilize`, `bound_mode`, `artificial_cost`, `log_interval` and
`pricing_options`. Resuming a snapshot accepts node options; `tree=True`
requires a fresh model. Tree iteration, pricing-batch and individual oracle
counts aggregate the evaluated nodes; `nodes` and `open_nodes` report tree work.

## Complete enumeration and integer transfer

Enumeration is opt-in. `Options.enumeration` in Rust, `enumerate=True` in
Python, or `--enumerate` in the CLI attempts a bounded full-domain transfer
at an accurate exact pricing checkpoint with no improving columns or positive
artificials. This can strengthen an NG master to
its elementary-route master. Every enabled network must complete before any
columns are committed. Existing valid columns and their bounds remain; cyclic
columns outside the integer domain are fixed to zero. All current physical
rows, including additive cuts/branches, remain in place. Descendants reuse the
complete route universe and evaluate new row coefficients on it.

Transfer supports nonempty physically elementary paths with disposable and
scalar equality resources. Equality paths are collected through a disposable
superset, then checked against the original equality transitions; paths feasible
only by waiting are excluded. Equality resources with a non-fixed sink
window use forward pricing even if another direction was requested; backward
pricing requires a fixed terminal resource value. Graphs containing direct source/sink routes
decline transfer because empty-route integer semantics are not yet defined. Collection uses a downward resource relaxation
that cannot discard an exact-feasible path, then independently replays original
resources. Negative-consumption equality domains remain unavailable through
this collector. Unsupported resources, ambiguous canonical replay, global work/path/
storage limits or timeout leave the transfer unavailable. Normal pricing remains
usable. Limits apply across all enabled networks, not independently per network.
The default candidate limit is 10,000, work limit 10,000,000, stored-arc limit
1,000,000 and attempt budget one second. Disabled networks contribute no routes.

`enumeration_reports` distinguishes no attempt, complete transfer and failure.
Each report records attempts, per-network counts, route count, newly transferred
columns and the failure reason. A repeated transfer on a complete snapshot does
not duplicate columns. Failed transfers do not retain a partial universe.
Each solve attempts at most one adaptive transfer, then resumes CG within the
original time and iteration limits. Backend transfer failures retain the original
master and warm start. A later solve can retry with different enumeration limits.
The explicit Rust `snapshot.enumerate(...)` method attempts transfer immediately.

No reduced-cost/objective cutoff removes paths during this collection; coverage
is the full supported domain, including incumbent routes.

Rust callers can separately request `snapshot.enumerate_below(cutoff, &limits)`
to collect routes needed by integer solutions strictly below a finite objective
cutoff. This returns an immutable `CutoffEnumeration`, with `coverage_cutoff()`,
`report()`, `routes()`, `dual_point()` and `cost_bounds()`. Limits cover all
networks and proof preparation. Missing original-domain bounds, unsupported
resource replay and exhausted budgets return an error; the source snapshot
remains usable. This API currently requires a snapshot that has not already
installed a full enumerated universe.

With `highs`, call `view.solve_mip(&MipOptions::default())` on that view. Its
`ObjectiveCutoff` outcome means no integer solution improves the effective
cutoff; it does not mean the original model is infeasible. Time/node limits remain explicit. A cutoff supplied by the caller
does not itself establish a feasible incumbent. The view cannot be resumed as
an original-domain LP or attached to a different snapshot. Its rows, fixed
contributions and caps are frozen; derive a new view from each changed node.
The adaptive CG handoff above continues to use full-domain enumeration.
When tree `enumerated_mip` is enabled and full transfer is unavailable, the tree
tries cutoff enumeration using its validated incumbent or the smaller supplied
MIP cutoff. A failed attempt leaves normal branching available. Numerical mode
can close a node from a sufficient MIP bound; certified mode uses returned
witnesses without promoting a numerical MIP bound to a certificate.

The CLI accepts `--mip-cutoff X` with `--tree --enumerated-mip`. The matching
JSON solve option is `mip_objective_cutoff` (finite number, or null for no supplied
cutoff). Reports include `coverage_cutoff`: null denotes full-domain coverage;
a number denotes coverage only below that objective. A cutoff without an
incumbent cannot close the original tree merely because its restricted pool
has no improving integer solution.
Python's `solve_networks(..., tree=True, enumerated_mip=True,
mip_objective_cutoff=...)` uses the same policy. Each
`NetworkEnumerationReport.coverage_cutoff` identifies the scope of its pool;
`None` means full-domain coverage.

With the optional `highs` feature, the tree can solve a complete transferred
copy as an integer model. Artificial columns are fixed to zero, and route
bounds, fleet rows, shared rows, inherited cuts/branches, fixed-route RHS
contributions and objective offsets are retained. If LP cleanup retired a route,
the private MIP model restores it from the complete universe and reapplies its
cap before solving. The original snapshot remains unchanged. Positive integer routes are
replayed against the original graph and original rows before an incumbent is
published. Source masters and siblings remain unchanged after any MIP outcome.

```sh
cargo build --offline --features clp,highs --bin baldes
target/debug/baldes multi-pricing examples/networks/triangle.json --tree --enumerated-mip --json -
# Force enumeration to decline and normal pricing/branching to continue:
target/debug/baldes multi-pricing examples/networks/triangle.json --tree --enumerated-mip --enumeration-route-limit 5 --json -
```

`--enumerated-mip` requires `--tree` and HiGHS support; it also enables enumeration.
`--enumeration-time`, `--mip-time` and `--mip-nodes` control per-attempt budgets.
MIP timeout/node limits/backend failure do not authorize pricing termination.
`enumerated_mip_calls`, `enumerated_mip_nodes` and `enumerated_mip_failures` report
work and fallback. Full-MIP numerical proof outcomes may close numerical-mode
nodes; they never become certified-mode lower-bound certificates. A cutoff
without a validated witness is not an incumbent or an infeasibility claim.

Python uses the same options and reports:

```python
from pybaldes import NetworkProblem
problem = NetworkProblem.from_file("examples/networks/triangle.json")
root = problem.solve(enumerate=True)
assert root.enumeration_reports[0].complete
integer = root.snapshot.solve_enumerated_mip(time_limit=1.0)
assert integer.status == "Optimal" and integer.incumbent == 2.0
# Or combine node enumeration and MIP with the tree:
tree = problem.solve(tree=True, enumerated_mip=True)
assert tree.nodes == 1 and tree.incumbent == 2.0
```

`node.solve(enumerate=True, enumeration_route_limit=..., enumeration_work_limit=...,
enumeration_stored_arc_limit=..., enumeration_time_limit=...)` prepares a resumed
node and returns a new snapshot. `solve_enumerated_mip` returns `None` when the
node has not completed transfer. Its optional `objective_cutoff` filters improving
witnesses without restricting enumeration; `ObjectiveCutoff` remains distinct
from `Infeasible`. MIP calls on existing Python snapshots keep the GIL and their
thread ownership. Fresh tree calls retain the existing GIL-release behavior.

For Python, build with `maturin develop --features highs` (Clp is enabled by
default). Clp-only builds still support complete enumeration; they reject MIP
requests explicitly.

## Grouped accumulated-resource branches

A group can span several pricing networks. Name its original shared packing row,
one resource coordinate, a stored threshold, and the physical arc IDs belonging
to the group in each network. The packing row must prove at most one occurrence:
its sense is `eq` or `le`, its RHS is at most one, all original contributions are
nonnegative, and each member arc has a coefficient of at least one. Signed
master-variable domains must also preserve nonnegative packing contributions.

```rust,ignore
use baldes::master::mp::ResourceGroupMember;
let members = vec![
    ResourceGroupMember { network: 7, arcs: vec![0, 2] },
    ResourceGroupMember { network: 42, arcs: vec![0, 2] },
];
let lower = parent.with_resource_group_branch(0, 0, 1.5, false, &members)?;
let upper = parent.with_resource_group_branch(0, 0, 1.5, true, &members)?;
```

Both children include the boundary `threshold - 1e-6`. The lower child requires
resource usage at most that boundary; the upper child requires usage at least
it. Disposable resources can wait to satisfy an upper window. Equality resources
cannot wait. Restrictions cover every occurrence of each listed physical arc,
including parallel alternatives. Other networks and arcs remain available.

Shared trees also generate disposable-resource groups automatically from original
unit packing rows. They combine physical occurrences across networks, account for
ancestor windows and use feasible forward/backward schedule intervals to select
a threshold. Rust callers can inspect the current solved node with
`parent.resource_group_candidates(1e-6)?` and attach either side with
`candidate.apply(&parent, upper)?`. The tree tries these candidates before its
assignment and aggregate-activity fallbacks. A failed attachment leaves the
parent reusable and causes the tree to try another disjunction.

Children own their restrictions through solves, further branches, route fixing,
enumeration and MIP. Parent and sibling snapshots remain reusable. Retained
schedules refresh under their existing route tags and caps. Fixed schedules
remain fixed; a contradictory child is infeasible. If exact feasibility can be
proved but a scalar schedule cannot be reconstructed, attachment returns an
error and preserves the parent.

Resource-load rows on a disposable coordinate that an upper branch can re-time
require complete enumeration before attachment. This keeps the rows' original
path-load semantics. Call `parent.enumerate(&limits)` first; the same condition
applies when adding such a row after a branch. Rows on other coordinates and
path-based additive or registered rows keep their usual workflow.

Python exposes the same owned operation:

```python
from pybaldes import NetworkResourceGroupMember
members = (NetworkResourceGroupMember(7, (0, 2)),
           NetworkResourceGroupMember(42, (0, 2)))
child = parent.with_resource_group_branch(0, 0, 1.5, True, members)
result = child.solve(stabilize=False)
tree = child.solve(tree=True, stabilize=False)
```

For the CLI, pass a JSON array of these restrictions with
`--resource-branches FILE`. Each object has `packing_row`, `resource`,
`threshold`, `upper` and `members`; each member has `network` and `arcs`.
The model is restricted before the requested root or tree solve.

```sh
cargo run --release --features clp --bin baldes -- multi-pricing \
  examples/networks/resource_groups.json \
  --resource-branches examples/networks/resource_group_upper.json \
  --tree --bound-mode certified --json -
```

## Synchronized service across vehicles

Pass `synchronized_services` to a native solve to require simultaneous service
of a job by several distinct vehicle copies. A member identifies the service
vertex in one pricing network; each service has at most one member per network.
A network with multiplicity two can supply two interchangeable vehicles using
two copies of the same elementary route.

```python
from pybaldes import SynchronizedService, SynchronizedServiceMember

service = SynchronizedService(
    id=17,
    resource=0,                 # scalar service-start time
    vehicles_required=2,
    reward=1.0,                 # counted once, not once per vehicle
    start=(0, 1440),            # inclusive integer domain
    members=(SynchronizedServiceMember(network=7, vertex=17),
             SynchronizedServiceMember(network=42, vertex=8)),
)
result = problem.solve(tree=True, synchronized_services=(service,))
```

The native master appends a binary `served` variable and the coverage equality
`sum(visits) = vehicles_required * served`. Per-network linking rows additionally
enforce `visits_in_network <= min(fleet_size, vehicles_required) * served`.
This is optional service: an unserved job has no visits. Keep its reward out of
incoming arc costs and remove ordinary unit-cover rows for that job; the solver
retains all caller-supplied costs and constraints. Physical fleet limits remain
in force. Rust callers set `mp::Options.synchronized_services`; JSON solve options
accept the same array of definitions. The CLI accepts
`--synchronized-services services.json` alongside `--tree`.

Synchronization uses an independent disposable resource with nonnegative arc
consumption. Its value at the member vertex must represent service start; put
service duration in outgoing arc consumption. Disjoint service-start windows
work with `multi-time-windows`. Start domains contain integers between
`-2**52` and `2**52`, inclusive; choose time units to obtain the desired grid.
Ordinary pricing-grid limits also apply to resource values.
Coupled resources in participating networks and resource-load rows on the
synchronized axis are rejected.

Root solves are LP relaxations and may contain unsynchronized columns. The
native tree propagates common earliest starts across all selected routes and
checks the complete joint schedule using exact resource replay. When that
cannot certify a witness, it splits a common integer start domain and clips
all participating networks. Restrictions survive snapshot forks, retained and
inactive columns, enumeration and fixed routes. Published integer route
profiles include the common starts. Time and node limits retain open nodes and
a relaxation bound; they never turn a failed synchronization check into an
incumbent or an infeasibility proof.

An LP snapshot can be restricted explicitly with
`node.with_synchronized_start(id=17, lower=300, upper=600)`. Its synchronization
definitions are immutable. Complete-domain MIP is deferred while any common
start domain is non-singleton, since its route-only matrix is then a relaxation.

The DSN adapter `mtw/synchronized_baldes.py` builds grouped antenna networks,
charges one reward per request, and independently checks returned schedules:

```python
from synchronized_baldes import solve_synchronized
result = solve_synchronized(data, time_limit=300)
```

A runnable two-fleet example is included:

```sh
cargo run --release --features clp -- networks \
  examples/networks/synchronization.json \
  --synchronized-services examples/networks/synchronization-services.json \
  --tree --json -
```

Both vehicles start service at time 5; the minimization objective is -1.
