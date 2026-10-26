# Enumerated node MIP with HiGHS

[Documentation index](../README.md) · [Tree lifecycle](tree.md) ·
[BALDES options](../reference/options.md)

BALDES can solve a complete enumerated node master with **HiGHS**, while keeping
Clp for column generation. The integer solve uses a separate HiGHS instance
and leaves the source LP, its solution/basis, and sibling snapshots untouched.

## Controls and API

Build and enable the optional tree path:

```sh
cargo run --features clp,highs --bin baldes -- \
  solve instances/Solomon/R101.txt --customers 10 --tree --enumerated-mip
```

`TreeOptions::enumerated_mip = Some(MipOptions::default())` enables the same
path in Rust. It is disabled by default to preserve the existing tree schedule.
A build without `highs` rejects this request explicitly. Disabling enumeration
leaves no complete universe to send to the MIP solver.

`MipOptions` defaults to 2,000,000 nodes, 360,000 seconds, and no objective
cutoff. The tree additionally caps time at 98% of its remaining global budget.
These budgets are separate from the restricted-column heuristic's
`TreeOptions::mip_node_limit` / `mip_time_limit`.

For a generic integer model, the `highs` feature alone is sufficient. This
complete example builds a two-variable binary master:

```rust
use baldes::master::{
    solvers::{highs::{self, HighsMaster}, mip::MipOptions},
    LpMaster, RowSense,
};

fn main() -> baldes::Result<()> {
    let mut master = HighsMaster::new()?;
    let row = master.add_row(RowSense::GreaterEqual, 1.5, &[])?;
    for _ in 0..2 {
        let col = master.add_column(1.0, &[(row, 1.0)])?;
        master.set_upper(col, 1.0)?;
    }
    let result = highs::solve_integer(&master, &[0, 1], &MipOptions::default())?;
    println!("{:?}: {:?}", result.status, result.solution);
    Ok(())
}
```

This copies every current physical row, coefficient, cost, and upper bound
through `LpMaster`. Columns have lower bound zero in this interface. Unlisted
columns remain continuous. Integer values and all row activities are checked
before returning a feasible witness. Coefficients small enough for HiGHS to
discard are rejected explicitly instead of silently changing the matrix.

For a routing snapshot, `NodeSnapshot::solve_enumerated_mip` prepares the full
route master and applies the end-of-node route-count gate. It returns `None`
if enumeration is incomplete or over the limit. This snapshot API requires
`clp,highs`; the generic `highs::solve_integer` API needs only `highs`.

## Enumeration gate

The route universe must be complete and its route count must not exceed the
configured threshold. Equality and a completed empty universe are accepted.
`BALDESmaxNumOfEnumSolutionsForMIP` defaults to 10,000;
`BALDESmaxNumOfEnumSolsForEndOfNodeMIP` defaults to zero and is raised to at least
the normal threshold. An incomplete universe never passes the gate. Setting
only the end-of-node option to zero does not disable the solve, because the normal threshold still
raises it. The gate counts retained enumerated routes before adding any
previously priced elementary witness routes to the integer model.

## Preserving node semantics

The snapshot adapter copies the current physical row order, including base
rows, inherited typed branches, and every active capacity, rank-one, strong
k-path and route-load knapsack row. It rebuilds route columns using current
base/branch coefficients and **full-memory** cut coefficients. It preserves
existing route upper bounds, including zero bounds. Artificial columns are
omitted, so finite artificial penalties cannot masquerade as feasibility.

Previously priced elementary routes are independently reconstructed and
validated before being retained as possible witnesses. Enumeration can have
removed a previously found incumbent route at its cutoff; retaining validated
old routes avoids losing that witness. Every added route still receives every
branch and active cut row.

`EnumeratedMipOutput::coverage_cutoff` records the incumbent under which the
universe was restricted (or the existing 1e12 fallback). A proof about this
matrix alone cannot exclude solutions above that cutoff. The tree uses at most
the minimum of the proved MIP bound and that coverage cutoff. It records a
validated routing witness before closing a gap. Without a witness, a finite
coverage cutoff cannot establish global infeasibility.

For snapshots with [fixed partial routes](tree.md#explicit-fixed-partial-solutions),
the integer model uses residual row bounds and a fixed objective offset. Returned
bounds and objectives include that offset, and `route_values` includes the fixed
routes when a witness is available, including a completed empty remainder.

## Outcomes

| Outcome | Interpretation |
| --- | --- |
| `Optimal` | HiGHS proved the integer model optimal and supplied a validated improving witness |
| `Infeasible` | HiGHS proved the transferred integer model infeasible; node interpretation still depends on enumeration coverage |
| `ObjectiveCutoff` | The proved integer optimum does not improve the caller's cutoff; no feasible incumbent is invented from that cutoff |
| `TimeLimit` / `NodeLimit` | Unfinished solve; any validated improving incumbent is retained, but the tree does not treat the limit as a proof |
| `Incomplete` | Other or ambiguous statuses, including unbounded-or-infeasible and unusable witnesses; no node-closing proof |

A finite `objective_cutoff` filters improving witnesses after solving; it is
not installed as a HiGHS search cutoff. The source model is copied before
solving, so a supplied cutoff never substitutes for a feasible incumbent.
Coefficients with nonzero magnitude at most `1e-12` are rejected to avoid
silent matrix changes in HiGHS.

HiGHS relative and absolute MIP gaps are set to zero. Its model status is
retained in `MipOutput::solver_status`, with solution, dual bound, node count,
simplex iterations, and elapsed time. Tree JSON adds `enumerated_mip_calls` and
`enumerated_mip_nodes`; its existing `mip_nodes` includes this work too.

## Limitations

BALDES currently runs the optional solve after node evaluation. A handoff
inside column generation, new cut separation during integer solving, and
multiple pricing networks remain outside this implementation. Active rows
are all retained.

## Custom CG enumeration controllers

The built-in routing controller transfers enumerated columns through an atomic
LP copy. If enumeration, regeneration, insertion or remapping fails, it restores
pricing state and continues ordinary CG. `cg::Output::enumeration_failure`
retains the last transfer error; it does not itself change the solve status.
Incomplete retained enumerations are discarded before pricing or inspection.
Within a CG solve, a successful handoff is not repeated for an unchanged
formulation. Reported row, coefficient or domain changes permit another handoff.
Retained generated columns are synchronized before the transaction commits.

A custom Rust `Controller` that implements `enumerated_universe` must also
implement `begin_enumeration_transaction` and `finish_enumeration_transaction`.
The begin hook saves all mutable state used by enumeration, regeneration,
column remapping and retained-pool synchronization, then returns `true`.
The finish hook commits on `true` and
restores that saved state on `false`; restoration must be infallible. A successful
fixing pass returning no universe may commit valid partial deductions.
Controllers without these hooks decline the optional transfer. Custom LP
backends must implement `LpMaster::try_clone_for_transaction`; Clp and HiGHS
already provide independent copies. The live LP is replaced only after every
fallible operation succeeds.
