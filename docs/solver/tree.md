# Branch-and-price tree and node lifecycle

[Documentation index](../README.md) · [Branching](branching.md) ·
[Enumerated MIP](enumerated-mip.md)

The shared tree driver accepts routing registrations, explicit network models
and imported-column adapters. Frontier selection, budgets and dispatch have one
implementation; model hooks interpret nodes and supply branching decisions.
This guide describes the routing registration. See [network models](models.md#assignment-tree)
and [architecture](../guides/architecture.md) for the other input contracts.

The routing registration solves a set-partitioning master with one pricing graph.
It requires nonnegative arc costs and unique public arc IDs. The base rows are
a fleet upper bound and one equality per customer; the tree validates this
structure before searching.

## Run a tree

```sh
cargo run --release --features clp --bin baldes -- \
  solve instances/Solomon/R101.txt --customers 10 --tree --time 60
```

`--nodes N` sets the tree node budget (default 10,000). `--json FILE` writes
validated incumbent routes, bounds, statuses, and work counters. Use
`--no-enumeration` to disable both reduced-cost fixing and enumeration.

### Optional parallel nodes

Add `--tree-threads N` to evaluate up to `N` open nodes concurrently; the
default is `1`, using the sequential search. Rust callers set
`TreeOptions::threads`. The root is evaluated first, so parallel work begins
once branching supplies multiple open nodes.

Workers receive independent parent matrices, cuts, pricing state and route
identities. Clp basis statuses are copied; other LP backends rebuild their
internal solver state. Native model pointers and `Rc` snapshots never cross
threads. The coordinator merges a bounded batch in dispatch order, updates
the incumbent and branching history, and runs strong branching and diving.
New incumbents reach workers at the next batch. Parallel search may therefore
visit more nodes or select a different optimal route set than sequential search.

The node limit and deadline apply to the whole tree, including dispatched
nodes. Every batch finishes before returning, and unfinished nodes retain
their lower bounds. `parallel_batches` reports batches containing multiple
nodes. Worker count is separate from pricing threads and `--master-threads`;
choose those budgets together to avoid oversubscribing the machine.

A complete Rust entry point, requiring `clp`:

```rust
use baldes::{
    master::{routing::{self, RoutingContext}, tree::{self, TreeOptions}},
    inputs::{Instance, Rounding},
};
use std::time::Duration;

fn main() -> baldes::Result<()> {
    let mut instance = Instance::read("instances/Solomon/R101.txt")?;
    instance.truncate(10)?;
    let graph = instance.graph(8, Rounding::None)?;
    let problem = RoutingContext::set_partitioning(graph, instance.vehicles as f64)?;
    let mut options = TreeOptions::default();
    options.time_limit = Duration::from_secs(60);
    options.root.cg.multiplicity = (instance.customers.len() - 1) as f64;
    let result = tree::solve(problem, options)?;
    println!("{:?}: incumbent={}, bound={}",
        result.status, result.incumbent, result.bound);
    Ok(())
}
```

The CLI tree uses `Instance::vehicles` as its fleet bound. Root-only CLI and
`Instance::solve` use the customer count instead. A custom Rust caller
chooses the bound in `RoutingContext::set_partitioning` and configures the
column-generation multiplicity separately.

## Node processing

The search selects a node by its lower bound and evaluates it within the
remaining global budget. Its main steps are:

1. Construct the root master, or fork the solved parent and append the child's
   constraints.
2. Run column generation, pricing, cut rounds, and enabled fixing/enumeration.
3. Share valid elementary routes with the incumbent heuristic and other nodes.
4. Search restricted columns for a feasible integer solution; optionally run
   HiGHS over a complete enumerated universe.
5. Close the node when its selected bound meets the incumbent gap, or select
   a branching candidate and create children. Strong trials and diving use
   isolated evaluations.

With multiple time windows, `options.branching.time_window_selection` runs the
tree on window copies of the arcs and branches on time-window selection counts
after arc flows; see
[branching on time-window selection](../extensions/multi-time-windows.md#branching-on-time-window-selection).

Time limits are cooperative across solver operations. A backend call can
finish after its requested budget. Limits preserve bounds for unfinished nodes.

`options.root.bound_mode` selects `BoundMode::Numerical` by default, or
`BoundMode::Certified` for independent original-domain validation. The CLI
equivalent is `--bound-mode numerical|certified`. Numerical bounds authorize
ordinary pruning within the configured tolerances. Certified mode never uses
a numerical estimate to close a node: an unsuccessful certificate attempt
leaves the bound unknown. `TreeOutput::bound_mode` and JSON label the bound
and the guarantee of gap-based termination. Warm starts retain the two bounds
separately, so changing modes cannot promote a parent's numerical estimate.

## What a child inherits

| State | Child handling |
| --- | --- |
| Clp master | Deep-copy model, basis, solution, scaling, and simplex state |
| Branch and cut rows | Preserve physical rows and coefficient mappings; append new branch rows |
| Existing route columns | Retain coefficients and upper bounds; retire routes outside the new domain |
| Pricing graph | Carry bucket preparation, fixed bucket arcs, NG state, options, and adaptive state |
| Labels | Rebuild for the child's duals and active branch resources |
| Enumeration | Retain/filter routes where compatible and preserve branch-distinct arc paths |
| Sibling state | Own independent mutable models and pricing state |

When enumeration changes the master to full-memory cut coefficients, existing
columns are updated as well as new ones. Reusing limited-memory coefficients
for only part of the enumerated master would change the model.

## Root snapshots and manual inheritance

Set `RootOptions::capture_state = true` to retain a solved `NodeSnapshot` in
`RootOutput::snapshot`. The tree arranges snapshot capture for its own nodes.
Manual callers use `routing::inherit_branches(parent, additional_branches, options)`
for typed constraints, or `routing::inherit` for arc-flow tuples. Supply only the
**additional** branches; the parent already contains its ancestor rows.
To retain the child's snapshot for another generation, enable capture on that
solve too.

Snapshots retain both restricted-master columns and the generated column pool,
including cached columns that have not entered the master. Child branch rows
and cut cleanup update cached coefficients as well as loaded coefficients.
Regeneration checks the child's current path restrictions; a cached route is
not automatically admissible. Manually constructed column-generation
`WarmStart` values provide `generated_pool` in creation order; an empty pool is
initialized from the loaded columns.

The snapshot exposes `bound_mode()`, `branch_depth()`, `active_cuts()`, and
`enumerated_routes()`. A route count alone does not establish completeness;
the enumerated-MIP adapter separately checks the completion flag.

## Incumbents, proofs, and status

| Source | Supplies an incumbent? | Certifies the routing lower bound? |
| --- | --- | --- |
| Validated elementary route solution | Yes | No, by its objective alone |
| Restricted-column integer heuristic | Yes, after validation | No |
| Exact node pricing | When the resulting routes form an integer witness | Numerical estimate in normal mode; independent certificate required in certified mode |
| Heuristic strong trial or dive | Can find a witness | No proof for the original node |
| Complete enumerated HiGHS MIP | Yes, after validation | Only within saved enumeration coverage and for proved outcomes |

A cost-only `--upper-bound` can assist root fixing, but the tree needs actual
feasible routes before accepting an incumbent. It does not invent a solution
from the supplied number.

`Optimal` means the configured gap is closed under the selected bound mode. `TimeLimit`,
`NodeLimit`, and `Incomplete` retain unfinished-node bounds. Check whether a
finite incumbent and routes are present before treating a limited run as a
solution. JSON encodes nonfinite numbers as `null`.

The restricted-column heuristic uses `TreeOptions::mip_node_limit` and
`mip_time_limit`; CLI `--mip-nodes` budgets that heuristic. Optional HiGHS node
MIP has separate controls and counters; see [Enumerated MIP](enumerated-mip.md).

## Limits and supported models

Every root, child, and priced trial runs node preprocessing before column
generation. A worklist propagates continuous bounds through the original
arc-flow master rows and flow conservation, including equivalent outgoing
cover rows. Child nodes also propagate active inherited capacity, complement,
and two-path cut rows using their exact additive arc coefficients and current
physical row bounds. Retired cut rows are excluded. Contradictions certify node infeasibility without LP solving or
pricing. Zero upper bounds retire inherited columns and restrict both pricing
directions and complete enumeration. Positive aggregate lower bounds remain
enforced by their original master rows; they do not require every route to
contain the arc. Parent and sibling state remains independent.

`RootOutput::preprocessing` reports bound updates, processed rows,
infeasibility, and whether the bounded worklist reached completion. Reaching
the work limit keeps valid deductions and continues ordinary node evaluation.
The reusable `master::preprocessing` engine accepts signed sparse rows and
infinite variable bounds, but callers must supply a complete formulation.
Applying it to an unfinished restricted-column master would be unsound.

Ryan–Foster preprocessing closes transitive together groups, excludes arcs
between apart groups, and fixes all incident arcs of an internally contradictory
group to zero. Covering and assignment rows then determine whether that group
is mandatory and the node is infeasible. Pricing resources still enforce the
complete route-level rules; preprocessing does not replace them.

Preprocessing currently covers the single-network routing model. The bounded
activity worklist may leave contradictions that require solving the LP.
Decomposition-wide local/global multiplicity links remain outside this adapter.

## Explicit fixed partial solutions

The Rust API can fix additional integer quantities of physical elementary routes
in a captured node:

```rust,ignore
use baldes::master::routing::{self, FixedRoute};

let child = routing::inherit_fixed_routes(
    parent_snapshot,
    &[FixedRoute { arcs: route_arc_ids, quantity: 1 }],
    options,
)?;
```

Quantities must be positive. Paths are reconstructed and checked against graph
resources and the node's pricing constraints. Calls are cumulative: fixing more
routes in a returned snapshot retains its existing fixed part. Over-covering a
customer or violating a later branch produces an infeasible residual node.
Parents and siblings retain their original row bounds, basis and fixed state.

Base and inherited active cut rows are shifted by each added route contribution.
The fixed elementary part uses full-memory coefficients for rank-one and strong
k-path rows; remaining LP columns can retain their valid weaker limited-memory
coefficients. Separation sees fixed routes alongside residual LP routes, and new
cut RHS values are shifted once. This also handles capacity/complement and mapped
resource-knapsack rows. Remaining fleet multiplicity and zero-arc propagation
apply to the residual routes.

LP objectives, Lagrangian bounds, incumbents and enumerated MIP outcomes include
the fixed cost. `RootOutput::routes` and `values` retain their residual-column
meaning. `fixed_routes` and `fixed_cost` expose the fixed part;
`RootOutput::route_values()` combines it with positive residual route values.
Check solver status and artificial values before accepting an LP witness.
Enumerated MIP route values already include fixed routes when a witness exists.

A fresh fixed-route solve discards inherited cutoff-restricted enumeration and
bucket fixing. It also starts with the caller's incumbent cutoff, because a
parent incumbent need not satisfy the fixed restriction. Automatic enumeration
and MIP gates retain their existing behavior; fixing routes does not force an
integer solve. The API does not choose routes to fix automatically, and currently
has no CLI or Python entry point.

This single-network routing tree does not provide multiple pricing networks or
cut callbacks during the integer solve. Python exposes routing root solving; use
Rust or the CLI for this tree. For a Python shared-master integer tree, use
[pyBaldes network models](../pybaldes/networks.md#integer-search), whose supported
model and branching families differ. `Infeasible` means all nodes were closed without a feasible
incumbent, under the selected bound mode and solver tolerances.

`Optimal` uses the configured absolute and relative gap tolerances. The reported
lower bound retains the minimum node bound across open and tolerance-closed
subproblems; closing a node does not replace its lower bound with the incumbent.
The final bound can therefore remain below the incumbent after successful gap
closure. If the retained bound cannot justify closure, the status is `Incomplete`.

Vehicle-count upper branches reuse the fleet row when its coefficients match
the branch. Repeated bounds keep the tighter RHS, while branch depth still
counts every recorded decision. Fixed route quantities reduce that RHS once.
A custom fleet row with different coefficients receives a separate branch row.

The root master receives explicit `root.initial_routes`. Child masters inherit
their parent's restricted model and generated column pool. Routes collected
globally for integer search, including automatic singletons and routes from
other evaluations, do not preload the child's restricted model. Equal-bound
child insertion visits the upward branch before its downward sibling.

For normal tree evaluations, a compatible preceding LP can supply numerical
state after its trailing branch columns and rows are removed. Exact ordered
matrix/cost/sense/offset checks precede transfer, and the target node keeps its
own row and column bounds. Other layouts retain the independent parent fork;
strong trials and dives remain isolated. Stabilization centers belong to the
parent, while a tightened vehicle upper bound sets the remaining route
multiplicity.

Warm metadata retains the parent's numerical and certified lower bounds
separately. Stabilization gap gating compares the selected inherited bound
with the preceding artificial-free restricted-LP bound
within each CG round. This differs from the global integer incumbent used for
pruning and enumeration.

Before mutating cuts, an artificial-free completed pricing round retains a
checkpoint of the master, route registry, cut/pricing state, stabilization and
fixing-gap cache. A later pricing timeout that explicitly requests saved-point
rollback restores that checkpoint and preserves any improved incumbent. Child
nodes start with a fresh saved-point flag. Unsaved timeouts, label limits and
other errors still propagate. Recovery uses the saved pricing bound; the
unfinished cut round's restricted LP objective supplies no new certificate.
Failed pricing attempts remain included in the recovered node's pricing-call
count and pricing time, while its bound and warm state come from the checkpoint.
