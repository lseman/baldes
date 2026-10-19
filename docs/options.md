# BALDES pricing options

[Documentation index](README.md) · [Cuts](cuts.md) ·
[Enumerated MIP](enumerated-mip.md)

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
| `BALDESstopCutGenTimeThresholdInPricing` | Double | `10.0` | [src/pricing/controller.rs](../src/pricing/controller.rs) |
| `BALDEShardTimeThresholdInPricing` | Double | `20.0` | [src/pricing/mod.rs](../src/pricing/mod.rs); [src/pricing/solve.rs](../src/pricing/solve.rs); [src/master/tree/mod.rs](../src/master/tree/mod.rs) |
| `BALDESredCostFixingTimeThreshold` | Double | `100.0` | [src/pricing/fixing_workflow.rs](../src/pricing/fixing_workflow.rs); [src/master/root/controller.rs](../src/master/root/controller.rs) |
| `BALDESnumberOfBucketsPerVertex` | Int | `25.0` | [src/pricing/mod.rs](../src/pricing/mod.rs); [src/api/options.rs](../src/api/options.rs) |
| `BALDESdynamicBucketSteps` | Int | `1.0` | [src/pricing/controller.rs](../src/pricing/controller.rs); [src/pricing/solve.rs](../src/pricing/solve.rs) |
| `BALDESuseBidirectionalSearch` | Int | `2.0` | [src/pricing/mod.rs](../src/pricing/mod.rs); [src/api/options.rs](../src/api/options.rs) |
| `BALDESapplyReducedCostFixing` | Int | `1.0` | [src/pricing/fixing_workflow.rs](../src/pricing/fixing_workflow.rs) |
| `BALDESmaxNumOfColsPerIteration` | Int | `30.0` | [src/pricing/mod.rs](../src/pricing/mod.rs); [src/pricing/solve.rs](../src/pricing/solve.rs) |
| `BALDESmaxNumOfColsPerExactIteration` | Int | `150.0` | [src/pricing/mod.rs](../src/pricing/mod.rs) |
| `BALDESmaxNumOfLabelsInEnumeration` | Int | `1000000.0` | [src/pricing/fixing_workflow.rs](../src/pricing/fixing_workflow.rs); [src/pricing/universe.rs](../src/pricing/universe.rs) |
| `BALDESmaxNumOfEnumeratedSolutions` | Int | `1000000.0` | [src/pricing/fixing_workflow.rs](../src/pricing/fixing_workflow.rs); [src/pricing/universe.rs](../src/pricing/universe.rs) |
| `BALDESinitNGneighbourhoodSize` | Int | `8.0` | [src/api/relaxation.rs](../src/api/relaxation.rs) |
| `BALDESmaxNGneighbourhoodSize` | Int | `0.0` | [src/api/relaxation.rs](../src/api/relaxation.rs); [src/master/root/controller.rs](../src/master/root/controller.rs) |
| `BALDESrankOneCutsMaxNumRows` | Int | `5.0` | [src/cuts/registry.rs](../src/cuts/registry.rs) |
| `BALDESrankOneCutsMaxNumPerRound` | Int | `100.0` | [src/cuts/registry.rs](../src/cuts/registry.rs); [src/cuts/kpath/mod.rs](../src/cuts/kpath/mod.rs) |
| `BALDESrankOneCutsMemoryType` | Int | `2.0` | [src/cuts/registry.rs](../src/cuts/registry.rs) |
| `BALDESrankOneCutsLSnumIterations` | Int | `1000.0` | [src/cuts/registry.rs](../src/cuts/registry.rs) |
| `BALDESredCostFixingFalseGap` | Double | `0.0` | [src/pricing/fixing_workflow.rs](../src/pricing/fixing_workflow.rs) |
| `BALDESmaxNumOfLabelsInHeurEnumeration` | Int | `0.0` | [src/pricing/fixing_workflow.rs](../src/pricing/fixing_workflow.rs) |
| `BALDEScheckDominInOtherBuckets` | Bool | `1.0` | [src/pricing/mod.rs](../src/pricing/mod.rs) |
| `BALDESheurLabelingStrategy` | Int | `0.0` | [src/pricing/mod.rs](../src/pricing/mod.rs) |
| `BALDESlabelDominanceStrategy` | Int | `0.0` | [src/pricing/mod.rs](../src/pricing/mod.rs); label dominance mode (0 or 1), shared with the C strategy setter |
| `BALDESdynBuckStepAdjustRatioThreshold` | Double | `500.0` | [src/pricing/controller.rs](../src/pricing/controller.rs) |
| `BALDESdynBuckStepAdjustNumBuckArcsThreshold` | Double | `10000.0` | [src/pricing/controller.rs](../src/pricing/controller.rs) |
| `BALDESdynamicNGmode` | Int | `-1.0` | [src/api/relaxation.rs](../src/api/relaxation.rs); [src/master/root/controller.rs](../src/master/root/controller.rs) |
| `BALDESmaxNumOfColumnsInDynamicNG` | Int | `100.0` | [src/api/relaxation.rs](../src/api/relaxation.rs); [src/master/root/controller.rs](../src/master/root/controller.rs) |
| `BALDESmaxCycleSizeInDynamicNG` | Int | `5.0` | [src/api/relaxation.rs](../src/api/relaxation.rs); [src/master/root/controller.rs](../src/master/root/controller.rs) |
| `BALDESmaxNGaverNeighbourhoodSize` | Int | `0.0` | [src/api/relaxation.rs](../src/api/relaxation.rs); [src/master/root/controller.rs](../src/master/root/controller.rs) |
| `BALDESuseCapacityCuts` | Bool | `1.0` | [src/master/root/mod.rs](../src/master/root/mod.rs); explicit option overrides a configured capacity separator and is inherited by children |
| `BALDEScapacityCutsSeparator` | Int | `1.0` | [src/pricing/mod.rs](../src/pricing/mod.rs); accepts the available built-in separator; external CVRPSEP (0) is rejected |
| `BALDEScapCutsMaxNumPerRound` | Int | `100.0` | [src/master/root/mod.rs](../src/master/root/mod.rs); explicit per-round override for configured capacity separation, including inherited nodes |
| `BALDESresConsKnapsackCutsMode` | Int | `-1.0` | [src/master/root/mod.rs](../src/master/root/mod.rs); [src/cuts/knapsack/mod.rs](../src/cuts/knapsack/mod.rs) |
| `BALDESresConsKnapsackCutsMaxNumPerRound` | Int | `100.0` | [src/master/root/mod.rs](../src/master/root/mod.rs); [src/cuts/knapsack/mod.rs](../src/cuts/knapsack/mod.rs) |
| `BALDESrankOneCutsArcMemoryType` | Int | `2.0` | [src/cuts/registry.rs](../src/cuts/registry.rs) |
| `BALDESrankOneCutsSpDependentMemory` | Bool | `0.0` | [src/cuts/registry.rs](../src/cuts/registry.rs) |
| `BALDESrankOneCutsMaxNumTriplets` | Int | `10000.0` | [src/cuts/registry.rs](../src/cuts/registry.rs) |
| `BALDESrankOneCutsTypeToSeparate` | Int | `0.0` | [src/cuts/registry.rs](../src/cuts/registry.rs) |
| `BALDESuseCovSetsForSepOneRowPackingCuts` | Bool | `0.0` | [src/cuts/registry.rs](../src/cuts/registry.rs) |
| `BALDESuseExactComplBoundsInEnumeration` | Bool | `1.0` | [src/pricing/universe.rs](../src/pricing/universe.rs) |

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
[enumerated integer solving](enumerated-mip.md).

## Examples and verification

Rust exposes metadata independently of constructing a graph:

```rust
use baldes::{core::options::Options, pricing::Solver};

fn main() -> baldes::Result<()> {
    println!("{:?}", Options::support("BALDESlabelDominanceStrategy")?);
    let mut solver = Solver::new(3, 1)?;
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

[`OPTION_TABLE`](../src/core/options.rs) is the source of truth for names,
types, defaults, and feature-dependent support. It lists only controls supported
in the current build. Keep this document synchronized when adding a consumer.
Bool values are numeric zero/one; integer options
require integral values. Live setters also validate operation-specific modes
and limits before changing solver state.

[Option tests](../tests/api/options.rs) cover canonical discovery in Rust/C,
input-alias compatibility, failure atomicity, dominance synchronization, and
capacity overrides
in fresh/inherited nodes. Run:

```sh
cargo test --features clp,highs --test options
cargo test --features clp --test options --test cli
```

See [Build and validation](testing.md) for the full suite. Separate feature
builds matter because MIP threshold support changes with the enabled backends.
