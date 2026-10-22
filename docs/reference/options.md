# BALDES pricing options

[Documentation index](../README.md) · [Cuts](../solver/cuts.md) ·
[Enumerated MIP](../solver/enumerated-mip.md)

The public option names start with `BALDES`. The standard build exposes
**39 supported pricing options**. Builds with both `clp` and `highs` expose
**two additional enumerated-MIP controls**. Each option below has a consumer;
some apply only to a particular operation or a configured cut family.

Rust exposes the supported metadata through `OPTION_TABLE` and availability
through `Options::support(name)`. Python lists supported names with
`Solver.option_names()` and reports scope with `Solver.option_support(name)`.
The C option-count/name functions use the same supported table. Listings are
specific to the features enabled in the build.

Use `Solver::set_option` to update a live solver. `Options` itself is a validated
option-value container; changing `solver.options` directly does not synchronize
controls, phase schedules or engine state. Failed live-solver setters preserve
the prior option values, controls, phases and prepared engine.

## Configuration and runtime overrides

- `BALDESlabelDominanceStrategy` accepts 0 or 1 and reaches labeling through
  `Solver::dominance_mode`. The dedicated C strategy setter uses the same
  path, so its value agrees with the named option getter. Full-label fixing
  still selects its required dominance behavior.
- `BALDESuseCapacityCuts` and `BALDEScapCutsMaxNumPerRound` override an explicitly
  configured root capacity separator and apply after inherited cut state is
  installed. Omitting an override preserves the configuration or parent's
  settings. Enabling the option alone does not create demands or a separator.
- `BALDEScapacityCutsSeparator` selects the available built-in separator for
  nonzero values. Value 0 requests external CVRPSEP, which is unavailable and
  is rejected explicitly. Positive capacity cut limits are required.

## Supported controls

Paths below name consumers, not unconditional guarantees. For example, NG
initialization options are consumed by the explicit C initialization calls;
`inputs::Instance::graph` receives its NG size as an argument. Cut options need
the corresponding cut family. The staged `Solver::price` convenience method
installs its own heuristic column schedule, while `price_phase` uses the
configured per-iteration limit. Knapsack root overrides currently apply when
constructing that separator; changing them in an inherited node does not
reconfigure its existing separator.

| Option | Kind | Default | Consumer / scope |
| --- | --- | --- | --- |
| `BALDESstopCutGenTimeThresholdInPricing` | Double | `10.0` | [src/pricing/controller.rs](../../src/pricing/controller.rs) |
| `BALDEShardTimeThresholdInPricing` | Double | `20.0` | [src/pricing/mod.rs](../../src/pricing/mod.rs); [src/pricing/solve.rs](../../src/pricing/solve.rs); [src/master/tree/mod.rs](../../src/master/tree/mod.rs) |
| `BALDESredCostFixingTimeThreshold` | Double | `100.0` | [src/pricing/fixing_workflow.rs](../../src/pricing/fixing_workflow.rs); [src/master/root/controller/enumeration.rs](../../src/master/root/controller/enumeration.rs) |
| `BALDESnumberOfBucketsPerVertex` | Int | `25.0` | [src/pricing/mod.rs](../../src/pricing/mod.rs) |
| `BALDESdynamicBucketSteps` | Int | `1.0` | [src/pricing/controller.rs](../../src/pricing/controller.rs); [src/pricing/solve.rs](../../src/pricing/solve.rs) |
| `BALDESuseBidirectionalSearch` | Int | `2.0` | [src/pricing/mod.rs](../../src/pricing/mod.rs) |
| `BALDESapplyReducedCostFixing` | Int | `1.0` | [src/pricing/fixing_workflow.rs](../../src/pricing/fixing_workflow.rs) |
| `BALDESmaxNumOfColsPerIteration` | Int | `30.0` | [src/pricing/mod.rs](../../src/pricing/mod.rs); [src/pricing/solve.rs](../../src/pricing/solve.rs) |
| `BALDESmaxNumOfColsPerExactIteration` | Int | `150.0` | [src/pricing/mod.rs](../../src/pricing/mod.rs) |
| `BALDESmaxNumOfLabelsInEnumeration` | Int | `1000000.0` | [src/pricing/fixing_workflow.rs](../../src/pricing/fixing_workflow.rs); [src/pricing/universe.rs](../../src/pricing/universe.rs) |
| `BALDESmaxNumOfEnumeratedSolutions` | Int | `1000000.0` | [src/pricing/fixing_workflow.rs](../../src/pricing/fixing_workflow.rs); [src/pricing/universe.rs](../../src/pricing/universe.rs) |
| `BALDESinitNGneighbourhoodSize` | Int | `8.0` | [src/graph/ng.rs](../../src/graph/ng.rs) |
| `BALDESmaxNGneighbourhoodSize` | Int | `0.0` | [src/master/root/controller/relaxation.rs](../../src/master/root/controller/relaxation.rs) |
| `BALDESrankOneCutsMaxNumRows` | Int | `5.0` | [src/cuts/registry/rank_one.rs](../../src/cuts/registry/rank_one.rs) |
| `BALDESrankOneCutsMaxNumPerRound` | Int | `100.0` | [src/cuts/registry/rank_one.rs](../../src/cuts/registry/rank_one.rs); [src/cuts/kpath/mod.rs](../../src/cuts/kpath/mod.rs) |
| `BALDESrankOneCutsMemoryType` | Int | `2.0` | [src/cuts/registry/rank_one.rs](../../src/cuts/registry/rank_one.rs) |
| `BALDESrankOneCutsLSnumIterations` | Int | `1000.0` | [src/cuts/registry/rank_one.rs](../../src/cuts/registry/rank_one.rs) |
| `BALDESredCostFixingFalseGap` | Double | `0.0` | [src/pricing/fixing_workflow.rs](../../src/pricing/fixing_workflow.rs) |
| `BALDESmaxNumOfLabelsInHeurEnumeration` | Int | `0.0` | [src/pricing/fixing_workflow.rs](../../src/pricing/fixing_workflow.rs) |
| `BALDEScheckDominInOtherBuckets` | Bool | `1.0` | [src/pricing/mod.rs](../../src/pricing/mod.rs) |
| `BALDESheurLabelingStrategy` | Int | `0.0` | [src/pricing/mod.rs](../../src/pricing/mod.rs) |
| `BALDESlabelDominanceStrategy` | Int | `0.0` | [src/pricing/mod.rs](../../src/pricing/mod.rs); label dominance mode (0 or 1), shared with the C strategy setter |
| `BALDESdynBuckStepAdjustRatioThreshold` | Double | `500.0` | [src/pricing/controller.rs](../../src/pricing/controller.rs) |
| `BALDESdynBuckStepAdjustNumBuckArcsThreshold` | Double | `10000.0` | [src/pricing/controller.rs](../../src/pricing/controller.rs) |
| `BALDESdynamicNGmode` | Int | `-1.0` | [src/master/root/controller/relaxation.rs](../../src/master/root/controller/relaxation.rs) |
| `BALDESmaxNumOfColumnsInDynamicNG` | Int | `100.0` | [src/master/root/controller/relaxation.rs](../../src/master/root/controller/relaxation.rs) |
| `BALDESmaxCycleSizeInDynamicNG` | Int | `5.0` | [src/master/root/controller/relaxation.rs](../../src/master/root/controller/relaxation.rs) |
| `BALDESmaxNGaverNeighbourhoodSize` | Int | `0.0` | [src/master/root/controller/relaxation.rs](../../src/master/root/controller/relaxation.rs) |
| `BALDESuseCapacityCuts` | Bool | `1.0` | [src/master/root/mod.rs](../../src/master/root/mod.rs); explicit option overrides a configured capacity separator and is inherited by children |
| `BALDEScapacityCutsSeparator` | Int | `1.0` | [src/pricing/mod.rs](../../src/pricing/mod.rs); accepts the available built-in separator; external CVRPSEP (0) is rejected |
| `BALDEScapCutsMaxNumPerRound` | Int | `100.0` | [src/master/root/mod.rs](../../src/master/root/mod.rs); explicit per-round override for configured capacity separation, including inherited nodes |
| `BALDESresConsKnapsackCutsMode` | Int | `-1.0` | [src/master/root/mod.rs](../../src/master/root/mod.rs); [src/cuts/knapsack/mod.rs](../../src/cuts/knapsack/mod.rs) |
| `BALDESresConsKnapsackCutsMaxNumPerRound` | Int | `100.0` | [src/master/root/mod.rs](../../src/master/root/mod.rs); [src/cuts/knapsack/mod.rs](../../src/cuts/knapsack/mod.rs) |
| `BALDESrankOneCutsArcMemoryType` | Int | `2.0` | [src/cuts/registry/rank_one.rs](../../src/cuts/registry/rank_one.rs) |
| `BALDESrankOneCutsSpDependentMemory` | Bool | `0.0` | [src/cuts/registry/rank_one.rs](../../src/cuts/registry/rank_one.rs) |
| `BALDESrankOneCutsMaxNumTriplets` | Int | `10000.0` | [src/cuts/registry/rank_one.rs](../../src/cuts/registry/rank_one.rs) |
| `BALDESrankOneCutsTypeToSeparate` | Int | `0.0` | [src/cuts/registry/rank_one.rs](../../src/cuts/registry/rank_one.rs) |
| `BALDESuseCovSetsForSepOneRowPackingCuts` | Bool | `0.0` | [src/cuts/registry/rank_one.rs](../../src/cuts/registry/rank_one.rs) |
| `BALDESuseExactComplBoundsInEnumeration` | Bool | `1.0` | [src/pricing/universe.rs](../../src/pricing/universe.rs) |

`BALDEShardTimeThresholdInPricing` is measured in seconds. Pricing uses fifty
times this threshold before a rollback point is saved and the threshold itself
afterward. The explicit `Controls::time_limit` remains a cap in either case.
Zero disallows a labeling sweep; cached inspection can complete without one.
Raw variable-cost validation shares the effective budget with cached pricing.

The two additional consumers with `clp,highs` are:

| Option | Kind | Default | Consumer / scope |
| --- | --- | --- | --- |
| `BALDESmaxNumOfEnumSolutionsForMIP` | Int | `10000.0` | Normal threshold used to normalize the end-of-node threshold; immediate in-CG MIP handoff is not implemented |
| `BALDESmaxNumOfEnumSolsForEndOfNodeMIP` | Int | `0.0` | End-of-node route-count gate, raised to at least the normal threshold; requires complete enumeration and opt-in HiGHS solving |

These appear in the public option list only when both `clp` and `highs` are
enabled. They apply to opt-in tree MIP solving; see
[enumerated integer solving](../solver/enumerated-mip.md).

## Examples and verification

Rust exposes metadata independently of constructing a graph:

```rust
use baldes::{core::options::Options, pricing::Solver};

fn main() -> baldes::Result<()> {
    println!("{:?}", Options::support("BALDESlabelDominanceStrategy")?);
    let mut solver = Solver::<2>::new(3, 1)?;
    solver.set_option("BALDESlabelDominanceStrategy", 1.0)?;
    Ok(())
}
```

Python exposes the same support query and runtime setter:

```python
from pybaldes import Solver

names = Solver.option_names()
implemented, scope = Solver.option_support("BALDESlabelDominanceStrategy")
solver = Solver(3, 1)
solver.set_option("BALDESlabelDominanceStrategy", 1.0)
```

CLI overrides are repeatable:

```sh
cargo run --release --features clp --bin baldes -- \
  vrptw instances/R101.txt --customers 10 \
  --option BALDESlabelDominanceStrategy=1 --time 60
```

[`OPTION_TABLE`](../../src/core/options.rs) is the source of truth for names,
types, defaults, and feature-dependent support. It lists only controls supported
in the current build.
Bool values are numeric zero/one; integer options
require integral values. Live setters also validate operation-specific modes
and limits before changing solver state.

## HGS initial columns and master backends

Run HGS for a separate, positive search budget before solving the root or tree:

```sh
cargo run --release --features clp --bin baldes -- \
  cvrp instances/cvrplib/E-n22-k4.vrp --hgs-time 10 --hgs-columns all --tree
```

`--hgs-columns best` (default) inserts routes from the best feasible HGS solution.
`all` retains distinct routes seen during population insertion, including those
later discarded or lost at a restart. A positive integer such as
`--hgs-columns 100` inserts up to 100 cheapest new feasible routes, ranked by
original arc-cost sum, with arc IDs breaking ties. Validation and deduplication
happen before ranking; routes already supplied as initial columns do not consume
this quota.
If fewer routes are available, all available routes are inserted. This ranking
uses route cost, not reduced cost or whole-solution quality. Every candidate
is replayed against the actual graph, using its resource bounds and arc costs;
HGS's integer demands do not establish original-model feasibility. HGS uses the
CLI's rounding mode, which defaults to truncated tenths (`--rounding tenth`)
in both HGS and the target graph; `--rounding none` keeps unrounded
distances. Explicit distance matrices retain their supplied precision. A complete best plan
supplies an incumbent only after original-model validation. `--seed` controls
the random seed. HGS runs before the solver's `--time` budget starts; its search
limit is cooperative, checked between HGS operations. Keeping all encountered
routes uses more memory than keeping only the best plan.

Select native IPM for restricted-master solves with `--master ipm`. This
selection automatically disables dual stabilization, including at descendant
nodes. The default remains `--master clp`. IPM solves are rebuilt from the
current sparse matrix and return their own primal values, duals, and reduced
costs. Clp still stores the mutable model and snapshots, so the routing driver
continues to require the `clp` build feature. Basis-dependent cut retirement is
skipped for IPM; it has no simplex basis. The separate conservative-bound
certificate routine retains its bounded auxiliary Clp solve. IPM time limits are cooperative,
checked between Newton iterations; a factorization may exceed the remaining
budget before control returns. Numerical failures are reported as errors.

During column generation, IPM uses the bound-driven accuracy rule from
Munari and Gondzio (2013), Section 3.1, Algorithm 1
(Munari and Gondzio (2013), p. 2029). It starts with `epsilon = 0.5`, then
uses `epsilon = min(1, gap / 10)` with a numerical floor of `1e-9`, where
`gap = max(0, UB - LB) / (1 + abs(UB))`. `UB` is the best artificial-free
restricted-master primal objective; `LB` comes only from exact LP pricing.
Integer incumbents, heuristic pricing estimates, and stronger integer-domain
certificates do not control IPM accuracy. New cuts reset these accuracy bounds.
Only the optimality gap is relaxed; primal and dual feasibility remain at
`1e-9`. Exhausted pricing triggers an accurate master solve and another pricing
check before separation or completion. Standalone master solves and strong
branching retain accurate solves. The requested tolerance is available as
`master_gap_tolerance` in Rust's column-generation iteration history.

This implements the paper's adaptive master accuracy. IPM still rebuilds each
restricted master; the stored-interior-iterate warm starts in Section 3.4 and
early cut generation in Algorithm 2 are not implemented by this option.

Select HiGHS for restricted-master LP solves with `--master highs`:

```sh
cargo build --release --features clp,highs
./target/release/baldes vrptw instances/R101.txt --customers 25 --master highs --log
```

HiGHS preserves the requested stabilization setting; use `--no-stabilization`
to turn it off. The native HiGHS instance persists across LP solves. Row/column
insertion, cost and coefficient edits, RHS/bound changes, and cleanup update
that instance directly, retaining its basis and reusable simplex state. The
adapter also reuses solution buffers, batches row/column insertion, and serves
an unchanged optimal LP from its cache without another native solve.

Node forks copy the sparse matrix in bulk and install the source's current
native basis into an independent HiGHS instance. A child refactorizes its own
basis once; siblings never share mutable solver state. Solutions, reduced costs,
and cut-cleanup basis statuses come from HiGHS. HiGHS owns the only native
matrix: the adapter does not retain, update, or clone a shadow Clp model.
Repeated matrix reads use a lazily populated Rust CSC snapshot, fetched in bulk
and invalidated by row/column insertion, coefficient edits, or deletion. This
avoids one native range query per column during cut replay and node restoration;
cost, RHS, and bound changes retain the coefficient snapshot.
Selecting a different backend converts the matrix on the next edit or solve,
releases the previous native model, and starts a fresh basis. Clp/IPM share Clp
matrix storage; switching between those two retains that storage.
Clp still solves the separate bounded certificate auxiliary LP. Selecting HiGHS
without the `highs` build feature
returns an explicit error. Enable HiGHS in `make config` when building with Make.

Python accepts `master_solver="highs"`; build its extension with the `highs`
feature (for example, `maturin develop --release --features highs`).
`pybaldes.has_highs` reports whether that build feature is available.

In Rust, set `RootOptions::master_solver` to `MasterSolver::Highs` or
`MasterSolver::Ipm`. For HGS seeding,
call `instance.seed_hgs(&problem, rounding, &mut options, &HgsWarmStart { ... })`
before passing those options to `root::solve` or `TreeOptions::root`. The types
`HgsWarmStart` and `HgsColumns::{Best, All, Top(n)}` are exported by `baldes::inputs`.
The returned summary reports inserted/rejected columns and any validated
incumbent. Existing initial columns are retained and duplicates are skipped.

Python exposes the same root options:

```python
result = pybaldes.solve_root(
    "instances/cvrplib/E-n22-k4.vrp", problem_type="cvrp",
    hgs_time=10.0, hgs_columns=100, hgs_seed=7, master_solver="ipm",
)
```


## Terminal progress

Routing solves are quiet by default. `--verbose`, `-v`, or `--log` enables an
aligned column-generation table on stderr. It prints iteration 1, every tenth
iteration, and the final state, followed by the CG status and work totals.
`--log-interval N` enables the same log with a positive custom interval.
The `Bucket step` column shows current active resource widths, separated by `/`.
Changes also produce a `Bucket step updated: old -> new` message independently
of that interval. Pricing implementations without a bucket grid show `--`.
Cut-count columns appear only for enabled families: `Rank1`, `Capacity`,
`2Path`, `KPath`, `Knapsack`, and `Fenchel`. Each shows the number of that
family's rows currently in the master, including inherited cuts and excluding
removed rows. Enabled families show zero before any cuts are installed.
The interval applies to each CG solve, including descendant nodes and tree
trials. Each node reports its depth, master algorithm, and stabilization mode.

```sh
./target/release/baldes vrptw instances/R101.txt --customers 25 \
  --hgs-time 2 --hgs-columns 100 --master ipm --verbose
```

`RMP obj` is the restricted-master objective and may include artificial columns;
it is not a feasible routing objective or a global lower bound. `Lower bound`
is the separately certified bound. `Incumbent` is the current objective upper
bound; a caller-supplied root upper bound still carries its feasibility
assumption. `Gap %` is `100 * max(0, incumbent - bound) / max(1, abs(incumbent))`.
Unknown or nonfinite values appear as `--`. `Min RC` is the latest pricing
minimum, `+Cols` counts inserted columns in that iteration, `Cols` is the current
master size including artificials, and `Phase` is the pricing phase. Time is
elapsed wall time within the current CG call. A CG `Optimal` status does not
by itself certify an integer routing optimum.

The logger adds no color escapes or cursor control, so redirected logs remain
readable. stdout summaries and `--json` files retain their result formats.
Standalone `hgs --verbose` uses the same HGS progress table; during warm starts,
that table and its summary appear before the column-generation log.

Rust callers set `options.cg.log_interval = Some(10)` (or the generic CG
`Options::log_interval`). Python callers use `verbose=True, log_interval=10`.
Python's interval defaults to 10 and must be positive; `verbose=False` keeps
CG silent.
