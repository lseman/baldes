# Multi-pricing: one master, multiple pricing subproblems

[Documentation index](../README.md)

With `clp`, `master::model::solve` solves a shared root and returns an owned
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

## Network pricing schedules

`Options.network_pricing_schedule` selects `NetworkPricingSchedule` in Rust.
The JSON/Python option is `network_pricing_schedule`; the CLI flag is
`--network-pricing-schedule`. All three accept these choices:

| Schedule | Network visitation |
| --- | --- |
| `full-sweep` (default) | Visit every enabled network in input order. |
| `cyclic` | Resume after the last visited network, wrap around, and return after a network generates a previously unregistered negative-reduced-cost route. |
| `sorted` | Visit unseen networks first, then ascending recent minimum reduced cost, breaking ties by input order. Return on the same new-route condition. |
| `promising-first` | Persistent primary/secondary queues following recovered strategy 2; normal solved networks enter the secondary queue. |
| `unvisited-first` | Persistent queues following recovered strategy 3; unvisited primary entries precede newly promising entries. |

For example, `problem.solve(network_pricing_schedule="cyclic")` in Python, or
`baldes solve MODEL.json --network-pricing-schedule sorted`. The option
also applies to snapshot solves and the nodes of an assignment tree.

A partial batch returns columns without an aggregate bound correction or
subgradient. It is never exact, even when each visited oracle used an exact
phase. A traversal that finds no new negative route visits every enabled network;
only a complete batch can supply a fresh global pricing estimate or establish
pricing convergence. Cached scores affect visitation order only: minima and
coverage are recomputed at the current dual point. Disabled networks are skipped.
The cursor and scores reset on a new solve, a formulation fork or a phase change.
An installed complete enumerated universe continues to use its full scan.
Early return requires reduced cost below minus the configured CG tolerance;
insignificant negative routes do not interrupt the scan.

The sorted score is an explicit BALDES heuristic. The two native queue policies
use `network_unpromising_limit` (CLI `--network-unpromising-limit`, default 1),
which limits normal solved visits after a new improving column appears. Empty
sweeps still visit all networks. Queue order persists across phase changes;
forks and new solves reset it. Normal modern native pricing returns zero even
when it generates columns; negative reduced cost does not imply the native
positive-status promising classification. That positive branch is recovered
statically but has no demonstrated normal-modern public reachability.
Native strategy 1 is deliberately excluded: its pinned public control aborts
with invalid-pointer corruption. These schedules change work distribution;
speedups and identical native trajectories are not promised.

## Proximal pricing

`cg::Options.proximal` selects explicit `ProximalParameters` in Rust. One hard
band gives box-step stabilization, one finite-penalty band gives a three-piece
penalty, and two bands give five pieces. In JSON/Python:

```python
result = problem.solve(
    proximal={"bands": [{"half_width": 0.5, "penalty": 1.0}]},
    network_pricing_schedule="promising-first",
)
```

`penalty: null` means a hard band. `multi_point: true` enables the private
support-plane epigraph and requires a finite outer penalty. Use
`max_points` (positive, default 1) to bound retained support planes; inactive
unprotected planes age out after more than two zero-activity solves. CLI callers can use
`--proximal-width 0.5 --proximal-penalty 1`, omit the penalty for a hard box, or
pass a complete bands object with `--proximal FILE.json`. Proximal pricing
replaces smoothing; `--no-stabilization` disables both.

Auxiliary columns, support rows and beta variables belong to a private LP copy.
Its objective cannot update the original bound or incumbent. An unsuccessful
proximal pricing pass retries the original master duals before completing a
phase. Explicit modes reset their center on phase changes; automatic curvature
retains its accepted center and radii. Backends that cannot
clone the master, or unfinished private solves, fall back to original duals.

## Automatic curvature

Automatic three-piece stabilization needs no manually prepared state:

```python
result = problem.solve(proximal={"curvature": {}})
# Five-piece automatic bands:
result = problem.solve(proximal={"curvature": {"five_piece": True}})
```

CLI flags are `--automatic-curvature` and `--automatic-curvature-five-piece`.
`--proximal FILE.json` accepts the same object. Automatic curvature is a separate
mode from explicit `bands` and `multi_point`; conflicting selections are rejected.
`--no-stabilization` disables the selected mode. Existing defaults remain smoothing.

| Curvature option | Default | Meaning |
| --- | --- | --- |
| `five_piece` | `false` | Maintain inner and outer radii instead of one radius. |
| `kappa` | `1` | Positive: initialize curvature from the first positive ordinary-row mean divided by Kappa. |
| `curvature` | `1` | Configured starting curvature when Kappa is nonpositive. |
| `advance_rate` | `1` | Radius adaptation rate, with native stored-float precision. |
| `artificial_activity_factor` | `10` | Multiply curvature when stabilization auxiliaries remain active at a stopping pass. |
| `initial` | `null` | Optional explicit prepared state; ordinary use needs no override. |

The controller measures original master duals against the previous accepted
center before pricing, and consumes that measurement once after the iteration.
Retries and a later center move cannot replace the saved mean. Positive-mean
bootstrap initializes the radii and curvature automatically. No eligible rows
means no fabricated feedback. Active dynamic rows use the same bands but do
not enter the ordinary-row mean; generated network multiplicity rows are excluded
from both measurement and automatic proximal variables. Root controllers use
actual base/branch/cut row metadata. Custom CG controllers can implement
`curvature_rows`; row sense or append position does not define participation.

`node.with_row(...)` adds an ordinary row. Rust `with_dynamic_row(...)` and
Python `node.with_row(..., dynamic=True)` label an added dynamic row explicitly.
An active private auxiliary above absolute activity `1e-6` can request a bounded
no-new-column retry; these updates share the existing penalty/time/iteration
budgets. Equal-width bands retain separate activity variables.

Automatic geometry survives pricing-phase and ordinary schedule restarts.
Compatible warm starts retain the accepted center and exact global radii/curvature
without applying the explicit-mode child factor. New child rows start at center
zero; incompatible coordinate geometry is discarded while a restricted child
can retain global radius information. Resume is distinct from creating a child.
Warm checkpoints validate configuration, physical row roles and original matrix
prefixes; ordinary newly generated columns do not invalidate them. Auxiliary
LP columns, support planes and objective values never enter those checkpoints
or authorize an original bound/incumbent.

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

`solve(problem, ModelTreeOptions)` uses network-aware
snapshots. It tries proved shared resource-group splits,
assignment rows, declared integer master variables, total network multiplicities,
original physical-arc activities, then synchronized-service splits. With
`ModelTreeOptions::time_window_selection`, arc activities are summed over
window copies and time-window selection counts follow them; see
[branching on time-window selection](multi-time-windows.md#branching-on-time-window-selection).
It returns network-tagged
integer route quantities, an incumbent, selected bound mode, frontier bound and
node/work-limit status. Integer witnesses replay original graph resources,
shared rows and original multiplicities with exact sums; objectives round upward.
Fixed routes are included in this replay. A rounded labeling resource profile
alone never authorizes an incumbent or fixed route.

`ModelTreeOptions::branching` selects the family and child order.
`NetworkBranchingPolicy::MultiplicityFirst` is the default: it tries fractional
network route counts first and visits upper children first, following the
observed public native controls. `ResourceFirst` preserves the resource and
assignment order above, with lower children first.
It does not change disjunctions or proof bounds, and does not promise the same
LP basis, generated columns or total node count as another backend. Select it
with `--network-branching multiplicity-first`, JSON solve option
`"network_branching": "multiplicity-first"`, or Python
`solve(..., tree=True, network_branching="multiplicity-first")`.

Tree selection is independent of branch-family selection.
`TreeSearchPolicy::Default` (default) processes the deepest open
node first, then the lowest inherited numerical LP bound, then creation order. It uses the
native 1,000-node primary frontier and its overflow queue. Select the
bound-first queue with `TreeSearchPolicy::BestBound`, CLI
`--network-tree-search best-bound`, JSON `"network_tree_search": "best-bound"`,
or Python `network_tree_search="best-bound"`. Queue comparisons use the numerical
channel even in certified mode; the selected certification bound still controls
pruning and published proof bounds. Results retain `nodes` as the
number of ordinary tree-node solves (excluding separate strong-branching trials).
`tree_processed_nodes` also counts exact-pricing branching trials and fresh
nodes closed by an inherited bound. Restricted-LP and heuristic screens are
included in `strong_branching_trials` and do not increment that processed count. Reusing a completed exact trial does not count it twice.
All traversal policies budget outer tree selections. A phased branching pass
can finish its trials past that limit, as in the native solver; its processed
count can therefore exceed the tree node limit.
`network_open_node_limit` controls the primary frontier capacity (native
default 1000); overflow nodes use the native secondary priority queue. The CLI
equivalent is `--network-open-node-limit N`.
The native outer loop also requires a nonempty primary queue. If only overflow
nodes remain (including capacity zero after the root), the solve returns
`Incomplete` with the unresolved nodes and their bound retained.
Native bound comparisons treat
values as tied within `1e-6 + 1e-10 * max(abs(a), abs(b))`; BALDES reproduces
that pairwise rule and the native vector-heap operations, including overflow
priority. Since tolerance-based pairwise ties are not transitive, this policy
uses a custom heap rather than Rust's ordering-based `BinaryHeap`.
`network_tree_search="depth-bound-priority"` selects deeper nodes first and prefers
the higher numerical LP bound at equal depth, followed by creation order.
Its overflow queue retains the usual lower-bound preference, as in native.
`network_tree_search="bound-priority"` selects native strategy 2. It orders
queued children by the numerical LP dual bound, then the native
child primal-bound sentinels, depth, parent treatment rank and creation
order. The child primal sentinels are equal before evaluation, so depth and
parent rank decide ties. The default `network_strategy2_rounding="unrounded"`
matches the native continuous-objective bound contract with nonpositive
`integerValuedBound`. Integer-objective models can select `"guarded-tolerance"`
for tolerant integer rounding, or `"exact-ceil"` for forced integer rounding.
CLI uses `--network-strategy2-rounding unrounded|guarded-tolerance|exact-ceil`; JSON uses
`"network_strategy2_bound_rounding"`. These are ranking keys only and do not
alter proof bounds.

Rust callers can run `snapshot.solve(options)` after installing
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
`snapshot.solve(options)` branches integer master variables.
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
entry point is `solve_with_master_variables`.

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
target/debug/baldes solve examples/networks/triangle.json --json root.json
target/debug/baldes solve examples/networks/triangle.json --tree --json tree.json
# JSON alone on stdout, suitable for a pipe:
target/debug/baldes solve examples/networks/triangle.json --tree --json -
```

The example has shared LP optimum `1.5` and integer optimum `2`. `--time`,
`--iterations`, `--nodes`, `--bound-mode`, `--no-stabilization`,
`--artificial-cost`, `--option NAME=VALUE`, and `--log-interval` control solves.
Run `baldes solve --help` for details. The previous `networks` command remains an alias. Unsupported routing flags are errors.
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

The dedicated [pyBaldes network guide](../pybaldes/networks.md) covers Python model construction, solve options, snapshots, and synchronization. The sections here specify the shared Rust/CLI/Python model contract.

Install the extension with Clp support (the Python build's default). The typed
inputs use the same JSON schema as the CLI, allowing a single model to drive
both interfaces:

```python
from pybaldes import NetworkProblem, solve

problem = NetworkProblem.from_file("examples/networks/triangle.json")
root = solve(problem)
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
pricing requires a fixed terminal resource value. Direct source-to-sink arcs are
included as one-arc routes in complete transfer and use the same original row
and objective accounting as other paths. Collection uses a downward resource relaxation
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

Coupled objective costs are included in proof preparation. A bounded complete
scalar-path superset can provide a directed route-wide resource-cost lower
bound, which joins the cutoff's once-per-route constant. This preserves routes
whose large scalar cost is offset by a negative resource charge. Full-universe
column bounds likewise use certified resource-cost lower endpoints. Path and
certificate work share the attempt's budget and deadline; unavailable replay
or incomplete coverage leaves the proof unknown. The original integer-route
domain is physically vertex-simple, and this fallback does not certify arbitrary
repeated-walk objective programs. It can be too expensive for large networks.

PWL paired resources and strict-domain slope pairs have independent directed
feasibility witnesses. A cost enclosure can remain non-singleton; finite-MIP
proof outcomes still require exact objective encoding. Front-algebra
paired operations, standalone raw slope-interval fronts and plugins retain
their explicit unsupported-proof behavior.

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
Python's `solve(..., tree=True, enumerated_mip=True,
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
target/debug/baldes solve examples/networks/triangle.json --tree --enumerated-mip --json -
# Force enumeration to decline and normal pricing/branching to continue:
target/debug/baldes solve examples/networks/triangle.json --tree --enumerated-mip --enumeration-route-limit 5 --json -
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
use baldes::master::model::ResourceGroupMember;
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
The synchronized coordinate must be a graph scalar resource. Separate coupled
resource domains may coexist on participating routes. Resource-load rows on
the synchronized axis are replayed when a start-domain branch changes route
profiles; fixed-route contributions update the residual row RHS as well.

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

### Policy controls and registered original variables

`stabilization_max_tree_depth` uses the recovered signed native boundary:
activation requires actual branch depth strictly below the limit (default
10000). A zero or negative limit disables smoothing/proximal participation from
the root. Resumes and ordinary row additions preserve depth. The corresponding
CLI flag is `--stabilization-max-tree-depth`.

`network_promising_limit` exposes the independent native positive-status budget
(default 1000000), alongside `network_unpromising_limit` (default 1). A normal
modern oracle still has status zero, even when it produces columns. Negative
reduced cost does not imply a promising status.

Register original integer aggregate variables explicitly when their meaning
spans networks, route constants or static integer master variables:

Arc coefficients may be integers or exact half-integers. Half-integer mappings
require a graph parity proof that every source-to-sink walk contributes an
integer value, including walks with repeated arcs. This supports splitting a
vertex assignment equally across its incoming and outgoing arcs. Per-route and
static-master coefficients still require integers.

```python
result = problem.solve(
    tree=True,
    mapped_variables=({
        "name": "total-route-count",
        "priority": 1.,
        "networks": [
            {"network": 7, "per_route": 1.},
            {"network": 42, "per_route": 1.},
        ],
    },),
)
```

Each network contribution also accepts sparse `arcs: [[arc_id, coefficient], ...]`;
`masters: [[variable_id, coefficient], ...]` adds static integer variables.
Coefficients must be bounded integers. Names identify generators; they do not
infer membership. `MappedBranchVariable::from_shared_row` derives selected
network/master contributions of a known original packing or linking row.
Branches install floor/ceiling bounds through the shared-row lifecycle, including
fixed routes and future columns.

Registered candidates support phased restricted-LP, heuristic-pricing and exact
pricing trials, using the routing strong-branching controls. Trial objective
estimates and history rank candidates; estimates do not become proof bounds.
Trial solutions that pass original-model integer witness validation can improve
the global incumbent, including solutions found in restricted-master trials.
Their objective is recomputed from original route and master-variable costs.
Set `mapped_strong_branching=False` to select by priority/fractionality alone.
Branch scores use the numerical parent bound even in certified mode. Their
continuous-objective cutoff is the parent incumbent minus half
`network_scoring_gap_tolerance` (default `1e-6`), matching the native absolute-gap
scoring margin. This setting affects scores, not published proof bounds or the
tree's stopping tolerance. Rust uses `scoring_gap_tolerance` in
`ModelTreeOptions`; CLI uses `--network-scoring-gap-tolerance`.
Trial witnesses update the tree incumbent after the evaluation pass returns;
they do not replace the parent cutoff during the same pass.
Results expose `strong_branching_trials`, `mapped_history_ranked` and
`mapped_branch_evaluations`. The CLI accepts `--mapped-variables FILE` and
`--no-mapped-strong-branching`; see
[`mapped_variables.json`](../../examples/networks/mapped_variables.json).

Recovered explicit proximal defaults are available through
`proximal={"explicit": {}}` (three-piece), or
`{"explicit": {"five_piece": True}}`. These derive initial widths from
the first positive ordinary-row mean when Kappa is positive, retain native
stored-float controls, and use the original row RHS for private artificial
scaling. Existing `bands` configurations remain explicit overrides. Rust
constructors are `explicit_three_piece()`, `explicit_five_piece()` and
`explicit_multipoint()`. CLI equivalents are `--proximal-explicit-three-piece`,
`--proximal-explicit-five-piece` and `--proximal-explicit-multipoint`.

Unstabilized MP pricing admits near-zero arc costs using the recovered native
`1e-6` sign precision only when a finite walk bound proves no improving path.
Negative bound corrections are retained. An unavailable pricing subgradient
remains unavailable, and a weak aggregate estimate falls back to real pricing.
Cyclic NG relaxations without a hard walk cap keep their pricing calls.

MP trees retain original-model integer witnesses during restricted-master solves.
When a valid numerical or certified bound closes the witness's incumbent gap,
the node can finish before another pricing pass; this is integer-domain pruning,
not an LP-pricing convergence claim. Retained witnesses survive later master
changes and are revalidated before publication. Root LP solves keep this behavior
disabled by default; Rust `Options.integer_primal_termination` enables it explicitly.

`network_branching="phased"` admits original resource-group, assignment,
integer master-variable, route-count, arc and time-window proposals into the same
phased evaluator as registered aggregates. Family priorities can be supplied in
`network_branch_priorities`; defaults are the routing adapter preset, rather
than a claim that all native models register the same priorities. Route-count
generators do not enter retained native-style ranking history. Rust callers can
also select completed-subtree estimates through `StrongBranchingOptions`.
Named generators can select `upper-first` or `lower-first` through
`network_branch_directions`, for example `{"count42": "lower-first"}` for a
registered aggregate called `count42`, or `{"ESA_7_1": "lower-first"}` for a
registered packing assignment. The Rust field is
`ModelTreeOptions.branch_directions`; CLI reads the same JSON map with
`--network-branch-directions FILE`, and Python accepts the map as a keyword.
Directions require `phased` branching (also selected by packing
registration) and a tree solve. A name can remain dormant until its generator
becomes fractional. Direction controls both trial construction and committed
child admission; it does not override the frontier's depth/bound comparisons.
Phased proposals default to upper first. Recorded side zero always
means the lower branch and side one the upper branch, regardless of execution
order; their history distances and proof outputs remain attached to those sides.
Mapped activity evaluation retains signed cancellation and multiplication
residuals when combining route copies, fixed routes and integer master terms.
The resulting LP activity selects candidates and scores trials; it does not
replace an independently published proof bound.
Trials use independent children. Surviving candidates keep their private LP,
columns and pricing workspaces between phases. Each priced phase restores the
full parent schedule before applying its own pricing floor, so a heuristic
slice cannot remove the later exact phase. Selected children can reuse completed exact
pricing results and their ordinary bound provenance, including the selected
certificate channel in certified mode. Restricted-master, heuristic and
incomplete trials contribute ranking estimates and validated feasible solutions.
Selected children retain those LP columns, basis and pricing workspaces even
when the local pass stops before exact completion. Their full pricing schedules
are restored, and ordinary tree treatment performs a fresh pricing scan.
Numerical and certified lower-bound improvements published by an exact-pricing
trial survive even when its final status is incomplete. Each channel remains
independent; heuristic and restricted-LP objectives cannot replace inherited
proof bounds. Partial estimates never become proof bounds or completed child
outputs. Discarded candidates remain isolated.
Phased trials follow their generator's direction, including mapped and
packing-assignment proposals. An exact infeasible trial is retained only with an
original-domain phase-one proof. Its closed state participates in native queue
ordering and outer selection budgets, but it is neither re-priced nor counted
as unresolved work. A solver status or positive artificial penalty alone does
not authorize this closure. If the outer treatment budget is exhausted before phased
branching starts, the current node stays unresolved without running trials.
A candidate admitted before the strong-pass deadline attempts both sides. If
that deadline expires during its first side, the sibling receives zero remaining
time and retains its inherited proof bounds. This attempt still counts in trial
and exact-processed statistics, while no LP iterations or pricing calls are
invented. A zero-iteration output preserves the preceding LP estimate, or the
unknown-value sentinel for a fresh side.

Set `network_share_inactive_columns=true` in JSON/Python, or pass
`--tree --network-share-inactive-columns`, to share generated physical paths
between tree nodes and strong trials. Rust exposes
`ModelTreeOptions.share_inactive_columns`. It defaults to false. Each
recipient rebuilds the path's schedule, original cost and row coefficients;
existing route caps and enumeration universes still apply. Imported paths stay
inactive until recipient repricing accepts them. Sharing does not transfer a
donor's proof bounds, solved LP or pricing completion. Root-only calls reject
the flat option because there are no other tree nodes to share with.

`finite_box_step()` / `--proximal-finite-box-step` follows native functiontype1,
including its finite stored-float cap1.01; the existing `box_step(width)` remains
a mathematical hard-band override. Public native interval names differ from the
logical band ordering: `inner_half_width` controls the last active pair (default10)
and `outer_half_width` the first pair (default100). Native multipoint defaults to
two zero-width bands with stored-float caps0.9. These details are retained in the native policy evidence.

Networks can declare `route_domain={"kind": "bounded_walks", "max_arcs": 4}`
to allow repeated interior vertices within a finite arc limit. The default is
`{"kind": "physical_simple"}`. The limit belongs to the model: pricing,
certified bounds, enumeration, fixed routes and resumed children preserve it.
Negative cycles may improve a bounded route, so its certified lower bound uses
the declared arc limit. A runtime pricing limit cannot silently narrow this
domain. Direct source-to-sink arcs are valid one-arc routes when `max_arcs >= 1`;
source/sink revisits are unsupported in this mode. Explicit vertex elementarity
groups can keep customers unique while allowing station revisits; their count
must fit the compiled NG memory capacity. `Graph::add_arc_elementarity_set`
adds hard groups of physical arcs, limiting a route to one arc from each group.
Pricing, canonical enumeration and original-route validation enforce the same
groups for bounded walks. The single-network root master supports uncutted
bounded walks. Root cut families and dynamic NG strengthening are supported
when the bounded-walk graph is acyclic or every interior vertex has a hard
elementarity group, either of which guarantees physically simple routes;
otherwise they are rejected for bounded walks.
Synchronized services additionally require their member vertex to belong to a
declared hard vertex elementarity group, so revisiting a service cannot count
one vehicle as two participants.

`front_program_domain=True` is a separate, explicit model choice for
coupled-resource fronts. It defines feasibility using the computed
floating-point front program. Its independent machine checker reproduces the
program's rounding operations; it does not prove equivalence to exact-real
source constraints. Leave the option false for the default source-domain
contract. Ambiguous source boundaries and nonfinite machine operations remain
uncertified. See the retained proof-scope counterexamples.

Packing-assignment branching can be registered with `packing` in solve
options, or `--packing FILE` in the multi-pricing CLI. For example:

```json
{"priority": 1000, "networks": [
  {"network": 7, "vertices": [[1, 1], [2, 2], [3, 0], [4, 1], [5, 2]]}
]}
```

Each pair is `[vertex_index, packing_index]`; repeated vertex setters replace
its preceding label. Source and sink labels are rejected. Packing labels are
independent of elementarity and additive shared-row membership. Rust callers
use `PackingBranching::new` and `NetworkPackingAssignment::new`; Python
callers pass the same dictionary as `packing=...` with `tree=True`.
Registration selects `phased` branching and replaces inferred shared-row
assignment proposals. Other families and their priorities remain available.

Each network/group expression charges half on every incident arc endpoint;
thus an arc with both endpoints in that group contributes one. Only activities
strictly between the tree tolerance and one minus that tolerance are admitted.
Candidates prefer activities closest to one half, then pricing-configuration
identity and group index; one configuration per packing group survives before
the ordinary strong-branching quota is applied. Configuration identity remains
stable through option clones, but allocation ties can differ across processes
and from the native binary. This does not promise identical symmetric branch
IDs or universal search trajectories. Explicit mappings may coexist, with
names distinct from the generated `ESA_<network>_<group>` names.

With `mapped_strong_branching=False` (CLI `--no-mapped-strong-branching`),
the packing separator is bypassed; ordinary count/original-activity
branching continues. Explicit `mapped_variables` retain their separate
rank-without-trials behavior.

Unstabilized network pricing accepts explicitly registered variable-box
bounds through Rust/JSON `variable_boxes`, Python
`variable_boxes=(...)`, or CLI `--variable-boxes FILE`. For example:

```json
[{"network": 7, "variables": [{"arc": 0, "upper": 1}, {"arc": 1, "upper": 1}]}]
```

Omitted variables/upper bounds remain unbounded; `lower` defaults to zero.
These are verified pricing envelopes, not new route restrictions. A finite cap
must already contain the graph's DAG/walk domain or follow from a forbidden arc.
Positive lower bounds are accepted when every complete graph walk within the
hard walk limit must traverse that arc. Upper-envelope checks use source/sink
connectivity and shortest return cycles, so a cyclic graph can still prove unit
caps on bridges and tighter repeat caps within its walk limit. Bounds requiring
unproved resource-specific deductions are rejected. Supplied
loose bounds retain their value; graph-derived caps do not silently replace
native inputs. Without registration, the existing orthant/near-zero policy
continues. Negative corrections are retained, without an invented
subgradient. A bound-only skip must close the complete current master gap or
accompany a fresh improving column; otherwise full pricing resumes. Aggregate
master activity bounds are not per-route variable bounds. Grouped resource rows
and other omitted nonlinear terms bypass these additive admission tests.


Required network multiplicity uses a bound-only correction rather than an
attaining zero selection. Admission requires a closing complete master gap or
fresh improving columns; an initial required route still needs a real witness.
The implicit nonnegative-arc box supports this test without extra registration.

Inactive-column reduced-cost scanning is opt-in. Set Rust
`Options::search_inactive_columns`, JSON/Python
`network_search_inactive_columns`, or CLI `--network-search-inactive-columns`
to reprice retained inactive paths before invoking the pricing oracles. This
applies to fresh roots and resumed snapshots. Tree `share_inactive_columns`
(JSON/Python `network_share_inactive_columns`) additionally transfers eligible
paths between nodes/trials and enables scanning for the root, ordinary nodes,
and strong trials. Retaining a path does not itself activate it or transfer a
proof bound. With scanning disabled, ordinary pricing still discovers improving
paths and supplies the required pricing certificates.

### Shared pricing coordination

The routing root frontend and the multi-network frontend use one pricing
coordinator. Routing supplies a one-network collection; network models supply
all enabled networks. Scheduling, deadlines, admission, complete-batch checks,
quantity corrections and subgradients use the same implementation. Model
adapters retain cut/NG updates, route-to-row coefficients and original-domain
certificate callbacks. Retained route domains also enter this coordinator;
explicit finite masters keep their individual column caps.

Routing pricing now uses independent raw-cost pricing, as network pricing does,
rather than treating correction of one rounded route as a global raw optimum.
The routing result's `oracle_calls` counts actual oracle entries, including
phase-I entries and excluding bound-only skips. It is separate from CG
`pricing_calls`; cached and raw sweeps form one oracle entry. This metric is
available in Rust, CLI JSON and Python results.

### Original pricing-variable compilation (Rust)

`compile_pricing_variables(problem, definitions)` compiles scoped
`OriginalPricingVariable` registrations into a physical problem and a
`PricingVariableRegistry`. Each registration supplies its network and model ID,
original cost, shared-row entries, and `(physical_arc_id, coefficient)` setters.
The final setter for an arc replaces earlier setters for that variable.
Different variables, base arc costs, and base row entries remain additive.
Expansion uses binary64 coefficients; certificates apply to that compiled model.
The registry also retains original-space cost caching. Call
`registry.configure_pricing(&mut options)` for a Rust root solve; the registry's
box/tree configuration and JSON/CLI/Python option builders install it
automatically. Each original reduced cost is rounded to1e-8 before applying its
association coefficients, matching the reference cache order. Base arc costs
remain separate. The raw compiled physical objective is priced independently
for proof safety; a rounded-cache optimum alone cannot certify the raw model.
Immutable metadata is shared across node forks and retained by default resumes;
window compilation copies its physical arc associations. Appended shared rows
contribute their explicit physical-arc dual terms. Replacing the compiled
coefficient prefix falls back to ordinary physical caching and raw pricing.

Signed associations retain their algebra in costs, shared rows and reported
activities. A negative lower endpoint of an original-variable envelope remains
negative for an optional network: choosing no route does not exclude its
negative-activity routes. Graph feasibility still determines which activities
are possible.

The registry retains canonical expressions independently of route tags.
`registry.activity(network, id, &output)` reports an outward `Bounds` enclosure
at the current LP point, including fixed copies. `integer: true` requests an
original aggregate branch registration after the existing integer-path proof;
call `registry.configure_tree(&mut tree_options)` to add those registrations.
It rejects name conflicts before editing the options. Generated names are
`X_<network>_<id>` and can be used with generator direction controls.

Original-variable bounds must be encoded in the pricing graph. This compiler
does not turn them into a global master restriction or add an implicit resource.
Optional original-variable pricing envelopes can now be attached through the
registry with `configure_boxes(&mut options, &[OriginalVariableBound { ... }])`.
The JSON/Python option is `original_variable_boxes`; the CLI accepts
`--original-variable-boxes FILE`, containing an array such as:

```json
[{"network": 7, "id": 0, "lower": 0, "upper": 1}]
```

These endpoints must already enclose the registered activity on every feasible
pricing walk. They do not impose a new graph restriction. The verifier uses
weighted DAG walk ranges, bounded cyclic walks, or matching scalar-resource
bounds; it rejects unsupported or unaffordable proofs. Omitted upper bounds are
unbounded. The declared endpoints remain unchanged after verification.

Admission evaluates the original variables independently, even when their costs
cancel on shared physical arcs. It adds an outward residual bound for base arc
costs, row edits and rounding. Overlapping associations therefore do not double
count physical costs. Existing witness, phase and complete-master-gap gates
still apply; unsupported nonlinear/coupled pricing falls back to the oracle.
Registry-backed node snapshots, tree forks and window copies retain these
expressions. Rust `SolveOptions` callers with original-variable references use
`node_options_with_registry` or `tree_options_with_registry`; the builders
without a registry reject unresolved references.

Model JSON now accepts a top-level `pricing_variables` array with the same
registration fields. The CLI and Python model frontend compile it once before
solving. Python `NetworkProblem.pricing_variables` carries these declarations;
its node handles retain the registry across row edits, fixing, resumes and
subsequent tree solves. Results report `original_variables` entries with
`network`, `id`, `lower`, and `upper` activity bounds. Tree/MIP results report
activities only when an incumbent is available. Fresh-root and tree compilation
use the original physical arc IDs; restored tree paths are used for reporting.
Rust callers needing both the physical problem and registry can use
`format::Model::compile()`; `problem()` returns only the compiled problem.
