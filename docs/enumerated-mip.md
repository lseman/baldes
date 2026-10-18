# Enumerated node MIP with HiGHS

BALDES can solve a complete enumerated node master with **HiGHS**, while keeping
Clp for column generation. There is no custom branch-and-bound around Clp.
The integer solve uses a separate HiGHS instance and leaves the source LP,
its solution/basis, and sibling snapshots untouched.

## Controls and API

Build and enable the optional tree path:

```sh
cargo run --features clp,highs --bin baldes -- \
  vrptw instances/R101.txt --customers 10 --tree --enumerated-mip
```

`TreeOptions::enumerated_mip = Some(MipOptions::default())` enables the same
path in Rust. It is disabled by default to preserve the existing tree schedule.
A build without `highs` rejects this request explicitly. Disabling enumeration
leaves no complete universe to send to the MIP solver.

`MipOptions` defaults to 2,000,000 nodes, 360,000 seconds, and no objective
cutoff. The tree additionally caps time at 98% of its remaining global budget.
These budgets are separate from the restricted-column heuristic's
`TreeOptions::mip_node_limit` / `mip_time_limit`.

For an existing LP backend:

```rust,ignore
use baldes::master::solvers::{highs, mip::MipOptions};
let result = highs::solve_integer(&master, &integer_column_indices, &MipOptions::default())?;
```

This copies every current physical row, coefficient, cost, and upper bound
through `LpMaster`. Columns have lower bound zero in this interface. Unlisted
columns remain continuous. Integer values and all row activities are checked
before returning a feasible witness. Coefficients small enough for HiGHS to
discard are rejected explicitly instead of silently changing the matrix.

For a routing snapshot, `NodeSnapshot::solve_enumerated_mip` prepares the full
route master and applies the end-of-node route-count gate. It returns `None`
if enumeration is incomplete or over the limit. It is available with `highs`.

## Enumeration gate

The route universe must be complete and its route count must not exceed the
configured threshold. Equality and a completed empty universe are accepted.
`RCSPmaxNumOfEnumSolutionsForMIP` defaults to 10,000;
`RCSPmaxNumOfEnumSolsForEndOfNodeMIP` defaults to zero and is raised to at least
the normal threshold. An incomplete universe never passes the gate.

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

## Outcomes

| Outcome | Interpretation |
| --- | --- |
| `Optimal` | HiGHS proved the integer model optimal and supplied a validated improving witness |
| `Infeasible` | HiGHS proved the transferred integer model infeasible; node interpretation still depends on enumeration coverage |
| `ObjectiveCutoff` | The proved integer optimum does not improve the caller's cutoff; no feasible incumbent is invented from that cutoff |
| `TimeLimit` / `NodeLimit` | Unfinished solve; any validated improving incumbent is retained, but the tree does not treat the limit as a proof |
| `Incomplete` | Other or ambiguous statuses, including unbounded-or-infeasible and unusable witnesses; no node-closing proof |

HiGHS relative and absolute MIP gaps are set to zero. Its model status is
retained in `MipOutput::solver_status`, with solution, dual bound, node count,
simplex iterations, and elapsed time. Tree JSON adds `enumerated_mip_calls` and
`enumerated_mip_nodes`; its existing `mip_nodes` includes this work too.

## Validation and remaining gaps

`tests/highs_mip.rs` covers generic Clp/HiGHS model transfer, source isolation,
rows and upper bounds, mixed integrality, infeasibility, cutoffs, zero budgets,
50 independent exhaustive integer comparisons, threshold boundaries and time
reserve, and complete snapshot solves across all five existing cut-family
configurations. Child tests enforce vehicle count, together/apart, arc-flow and
assignment rows and then re-solve the parent to check isolation. Tree and CLI
tests exercise opt-in MIP, disabled enumeration, exhausted budgets, and missing
backend support.

The complete all-feature workspace suite passed **155 tests**, including nine
HiGHS MIP tests and the CLI test. Default/all-feature build checks passed;
Clp-only option and CLI checks passed too. HiGHS headers/library used for these
checks are version 1.15.1.

BALDES currently runs the optional solve after node evaluation. A handoff
inside column generation, new cut separation during integer solving, and
multiple pricing networks remain outside this implementation. Active rows
are all retained.
