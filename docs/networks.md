# Shared master with multiple pricing networks

With `clp`, `master::networks::solve` solves a shared root and returns an owned
node snapshot that can be resumed or forked. Its pricing graphs share master
rows. Each `Network` owns a graph, a unique nonnegative ID and
finite nonnegative multiplicity bounds. Local arc IDs may overlap across
networks: route identities include the network ID and resource schedule.

`Problem.rows` defines the shared constraints. `Network.arc_entries` contains
one list of `(shared_row, coefficient)` pairs per physical arc, in graph order.
`route_entries` adds coefficients charged once per route. The coordinator adds
the network multiplicity rows separately; callers do not include them in the
shared-row indices. Arc costs define original route costs. Variable-cost
mapping used internally by pricing does not modify the supplied graphs.
Physical multiplicity rows appear in input-network order: each upper row is
followed immediately by its lower row when the lower bound is positive. Shared
rows follow these multiplicity rows in their original order.

The coordinator maintains independent pricing state, combines columns into one
restricted master and aggregates each network's multiplicity contribution and
stabilization subgradient. `Output.oracle_calls` counts individual oracle calls;
`Output.master.pricing_calls` counts batches. `Output.values` gives the current
restricted-master quantity for every route in `Output.routes`.

Phase estimates guide stabilization. Published lower bounds additionally use a
resource-free graph-walk relaxation with downward arithmetic. A relevant
negative cycle, arithmetic overflow or unavailable required path leaves this
certificate unknown. In certified mode, unavailable certificates cannot close
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
[scalar resources](resources.md).

## Assignment tree

`solve_assignment_tree(problem, AssignmentTreeOptions)` runs a best-bound tree
using the network-aware snapshots and assignment rows. It returns network-tagged
integer route quantities, an incumbent, selected bound mode, frontier bound and
node/work-limit status. Integer witnesses replay original graph resources,
shared rows and original multiplicities with exact sums; objectives round upward.
Fixed routes are included in this replay. A rounded labeling resource profile
alone never authorizes an incumbent or fixed route.

Assignment branching does not cover every integer model: a single network can
have integral customer assignments and fractional route quantities. Such a node
returns `Incomplete` and remains open for another branching family. Positive
artificials or incomplete certification likewise do not prove infeasibility.
Node/time limits retain the open-frontier bound. Bounds on fixed-route nodes
are newly certified only when their RHS, offset and residual multiplicity
reductions are exact; otherwise valid inherited certificates remain available,
but a new original-domain bound stays unknown.

## Limitations

The supported lifecycle uses additive rows. Automatic nonlinear cut separation,
reduced-cost-pruned enumeration with proved coverage, adaptive in-loop
transfer eligibility, other branching families, stronger resource-aware
certificates and C dispatch remain open. The existing
single-network routing entry points retain their current behavior.

## CLI

The `networks` command accepts an explicit JSON model, rather than a Solomon
instance. It uses Clp and supports both shared roots and the assignment tree:

```sh
cargo build --offline --features clp --bin baldes
target/debug/baldes networks examples/networks/triangle.json --json root.json
target/debug/baldes networks examples/networks/triangle.json --tree --json tree.json
# JSON alone on stdout, suitable for a pipe:
target/debug/baldes networks examples/networks/triangle.json --tree --json -
```

The example has shared LP optimum `1.5` and integer optimum `2`. `--time`,
`--iterations`, `--nodes`, `--bound-mode`, `--no-stabilization`,
`--artificial-cost`, `--option NAME=VALUE`, and `--log-interval` control solves.
Run `baldes networks --help` for details. Unsupported routing flags are errors.
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
before column generation starts/resumes. This can strengthen an NG master to
its elementary-route master. Every enabled network must complete before any
columns are committed. Existing valid columns and their bounds remain; cyclic
columns outside the integer domain are fixed to zero. All current physical
rows, including additive cuts/branches, remain in place. Descendants reuse the
complete route universe and evaluate new row coefficients on it.

The first implementation supports nonempty physically elementary paths with
**disposable scalar resources**. Graphs containing direct source/sink routes
decline transfer because empty-route integer semantics are not yet defined. Collection uses a downward resource relaxation
that cannot discard an exact-feasible path, then independently replays original
resources. Unsupported resources, ambiguous canonical replay, global work/path/
storage limits or timeout leave the transfer unavailable. Normal pricing remains
usable. Limits apply across all enabled networks, not independently per network.
The default candidate limit is 10,000, work limit 10,000,000, stored-arc limit
1,000,000 and attempt budget one second. Disabled networks contribute no routes.

`enumeration_reports` distinguishes no attempt, complete transfer and failure.
Each report records attempts, per-network counts, route count, newly transferred
columns and the failure reason. A repeated transfer on a complete snapshot does
not duplicate columns. Failed transfers do not retain a partial universe.

No reduced-cost/objective cutoff removes paths during this collection; coverage
is the full supported domain, including incumbent routes. This is deliberately
separate from adaptive reduced-cost enumeration: that optimization still needs
a coverage proof and in-CG eligibility policy before it can replace this path.

With the optional `highs` feature, the tree can solve a complete transferred
copy as an integer model. Artificial columns are fixed to zero, and route
bounds, fleet rows, shared rows, inherited cuts/branches, fixed-route RHS
contributions and objective offsets are retained. Positive integer routes are
replayed against the original graph and original rows before an incumbent is
published. Source masters and siblings remain unchanged after any MIP outcome.

```sh
cargo build --offline --features clp,highs --bin baldes
target/debug/baldes networks examples/networks/triangle.json --tree --enumerated-mip --json -
# Force enumeration to decline and normal pricing/branching to continue:
target/debug/baldes networks examples/networks/triangle.json --tree --enumerated-mip --enumeration-route-limit 5 --json -
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
