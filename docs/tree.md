# Branch-and-price tree and node lifecycle

[Documentation index](README.md) · [Branching](branching.md) ·
[Enumerated MIP](enumerated-mip.md)

The Rust tree solves a routing set-partitioning master with one pricing graph.
It requires nonnegative arc costs and unique public arc IDs. The base rows are
a fleet upper bound and one equality per customer; the tree validates this
structure before searching.

## Run a tree

```sh
cargo run --release --features clp --bin baldes -- \
  vrptw instances/R101.txt --customers 10 --tree --time 60
```

`--nodes N` sets the tree node budget (default 10,000). `--json FILE` writes
validated incumbent routes, bounds, statuses, and work counters. Use
`--no-enumeration` to disable both reduced-cost fixing and enumeration.

A complete Rust entry point, requiring `clp`:

```rust
use baldes::{
    master::{root::RootProblem, tree::{self, TreeOptions}},
    vrptw::{Instance, Rounding},
};
use std::time::Duration;

fn main() -> baldes::Result<()> {
    let mut instance = Instance::read("instances/R101.txt")?;
    instance.truncate(10)?;
    let graph = instance.graph(8, Rounding::None)?;
    let problem = RootProblem::set_partitioning(graph, instance.vehicles as f64)?;
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
`Instance::solve_root` use the customer count instead. A custom Rust caller
chooses the bound in `RootProblem::set_partitioning` and configures the
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
5. Close the node when its certified bound meets the incumbent gap, or select
   a branching candidate and create children. Strong trials and diving use
   isolated evaluations.

Time limits are cooperative across solver operations. A backend call can
finish after its requested budget. Limits preserve bounds for unfinished nodes.

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
Manual callers use `root::inherit_branches(parent, additional_branches, options)`
for typed constraints, or `root::inherit` for arc-flow tuples. Supply only the
**additional** branches; the parent already contains its ancestor rows.
To retain the child's snapshot for another generation, enable capture on that
solve too.

The snapshot exposes `branch_depth()`, `active_cuts()`, and
`enumerated_routes()`. A route count alone does not establish completeness;
the enumerated-MIP adapter separately checks the completion flag.

## Incumbents, proofs, and status

| Source | Supplies an incumbent? | Certifies the routing lower bound? |
| --- | --- | --- |
| Validated elementary route solution | Yes | No, by its objective alone |
| Restricted-column integer heuristic | Yes, after validation | No |
| Exact node pricing | When the resulting routes form an integer witness | Yes, through its certified bound |
| Heuristic strong trial or dive | Can find a witness | No proof for the original node |
| Complete enumerated HiGHS MIP | Yes, after validation | Only within saved enumeration coverage and for proved outcomes |

A cost-only `--upper-bound` can assist root fixing, but the tree needs actual
feasible routes before accepting an incumbent. It does not invent a solution
from the supplied number.

`Optimal` means the configured numerical gap is closed. `TimeLimit`,
`NodeLimit`, and `Incomplete` retain unfinished-node bounds. Check whether a
finite incumbent and routes are present before treating a limited run as a
solution. JSON encodes nonfinite numbers as `null`.

The restricted-column heuristic uses `TreeOptions::mip_node_limit` and
`mip_time_limit`; CLI `--mip-nodes` budgets that heuristic. Optional HiGHS node
MIP has separate controls and counters; see [Enumerated MIP](enumerated-mip.md).

## Limits and validation

Every root, child, and priced trial runs node preprocessing before column
generation. A worklist propagates continuous bounds through the original
arc-flow master rows and flow conservation, including equivalent outgoing
cover rows. Contradictions certify node infeasibility without LP solving or
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

Preprocessing currently covers the single-network routing model.
Nonlinear Ryan–Foster rules retain their
existing pricing resources; separator cuts and decomposition-specific
local/global multiplicity bounds are not propagated by this adapter. There is
no integer rounding, so propagation preserves the node LP relaxation.

The tree does not yet provide multiple pricing networks or cut callbacks
during the integer solve. Python exposes root solving; use Rust or the CLI
for the tree. `Infeasible` means all nodes were certified closed without a
feasible incumbent.

- [Node snapshots and inheritance](../src/master/root/mod.rs)
- [Tree driver](../src/master/tree/mod.rs)
- [Lifecycle tests](../tests/node_lifecycle.rs): fixing carry, sibling isolation,
  full-memory updates, grandchildren, and failed setup
- [Tree tests](../tests/branch_price.rs): exhaustive route universes, limits,
  feasible witnesses, and branch/cut preservation

```sh
cargo test --features clp --test node_lifecycle --test branch_price
```
