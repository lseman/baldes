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

## Remaining integration

The supported lifecycle uses additive rows. Automatic nonlinear cut separation,
all-network enumeration/MIP transfer, other branching families, stronger
resource-aware certificates, and CLI/C/Python dispatch remain open. The existing
single-network routing entry points retain their current behavior.

## Validation

The registered integration suite exercises shared covering rows, fractional and
required multiplicities, disabled networks, overlapping local IDs, once-per-route
coefficients, both assignment children, sibling/parent restore, additive cuts,
fixed routes, generated-pool cleanup, smoothing, nondefault resource dimensions,
new columns required only in a child, and conservative incomplete outcomes.
The two-network triangle has analytic LP value `1.5`; exhaustive binary-route
enumeration gives integer value `2`, matching both children and the tree.

```sh
cargo test --offline --features clp --test networks
```
