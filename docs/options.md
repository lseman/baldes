# RCSP option support

Audited 2026-10-18. All 58 RCSP option names, kinds, defaults and
enumeration order are retained. **With `clp,highs`, 41 options have consumers and 17 remain unsupported;
otherwise, the counts are 39 and 19.** A consumer can be conditional on an operation or configured
cut family; this count is not a claim of support for every configuration.

`Options::support(name)` and `OPTION_TABLE[i].support` expose the support status
and consumer or reason in Rust. Python exposes
`Solver.option_support(name) -> (implemented, description)`.

Unsupported options remain readable, and setting their default is
allowed for compatibility with configurations that repeat the complete
default table. A non-default change returns an error naming the missing
behavior, before updating solver state. C and Python setters propagate that
error. No option entries were deleted.

Use `Solver::set_option` to update a live solver. `Options` itself is a validated
option-value container; changing `solver.options` directly does not synchronize
controls, phase schedules or engine state. Failed live-solver setters preserve
the prior option values, controls, phases and prepared engine.

## Newly connected behavior

- `RCSPlabelDominanceStrategy` accepts 0 or 1 and reaches labeling through
  `Solver::dominance_mode`. The dedicated C strategy setter uses the same
  path, so its value agrees with the named option getter. Full-label fixing
  still selects its required dominance behavior.
- `RCSPuseCapacityCuts` and `RCSPcapCutsMaxNumPerRound` override an explicitly
  configured root capacity separator and apply after inherited cut state is
  installed. Omitting an override preserves the configuration or parent's
  settings. Enabling the option alone does not create demands or a separator.
- `RCSPcapacityCutsSeparator` selects the available native separator for
  nonzero values. Value 0 requests external CVRPSEP, which is unavailable and
  is rejected explicitly. Positive capacity cut limits are required.

The rank-one neighbourhood-size option remains unsupported in the routing
adapter: its generated cut graphs contain no packing-distance matrices or
explicit packing neighbourhoods. Passing the value into a parameter struct
would not make it effective. Exposing and reproducing that input path is
required before enabling this setting.

## Options with consumers

Paths below name consumers, not unconditional guarantees. For example, NG
initialization options are consumed by the explicit C initialization calls;
`vrptw::Instance::graph` receives its NG size as an argument. Cut options need
the corresponding cut family. The staged `Solver::price` convenience method
installs its own heuristic column schedule, while `price_phase` uses the
configured per-iteration limit. Knapsack root overrides currently apply when
constructing that separator; changing them in an inherited node does not
reconfigure its existing separator.

| Option | Default | Consumer / scope |
| --- | --- | --- |
| `RCSPstopCutGenTimeThresholdInPricing` | `10.0` | src/pricing/controller.rs |
| `RCSPhardTimeThresholdInPricing` | `20.0` | src/pricing/mod.rs; src/pricing/solve.rs; src/master/tree/mod.rs |
| `RCSPredCostFixingTimeThreshold` | `100.0` | src/pricing/fixing_workflow.rs; src/master/root/controller.rs |
| `RCSPnumberOfBucketsPerVertex` | `25.0` | src/pricing/mod.rs; src/api/options.rs |
| `RCSPdynamicBucketSteps` | `1.0` | src/pricing/controller.rs; src/pricing/solve.rs |
| `RCSPuseBidirectionalSearch` | `2.0` | src/pricing/mod.rs; src/api/options.rs |
| `RCSPapplyReducedCostFixing` | `1.0` | src/pricing/fixing_workflow.rs |
| `RCSPmaxNumOfColsPerIteration` | `30.0` | src/pricing/mod.rs; src/pricing/solve.rs |
| `RCSPmaxNumOfColsPerExactIteration` | `150.0` | src/pricing/mod.rs |
| `RCSPmaxNumOfLabelsInEnumeration` | `1000000.0` | src/pricing/fixing_workflow.rs; src/pricing/universe.rs |
| `RCSPmaxNumOfEnumeratedSolutions` | `1000000.0` | src/pricing/fixing_workflow.rs; src/pricing/universe.rs |
| `RCSPinitNGneighbourhoodSize` | `8.0` | src/api/relaxation.rs |
| `RCSPmaxNGneighbourhoodSize` | `0.0` | src/api/relaxation.rs; src/master/root/controller.rs |
| `RCSPrankOneCutsMaxNumRows` | `5.0` | src/cuts/registry.rs |
| `RCSPrankOneCutsMaxNumPerRound` | `100.0` | src/cuts/registry.rs; src/cuts/kpath/mod.rs |
| `RCSPrankOneCutsMemoryType` | `2.0` | src/cuts/registry.rs |
| `RCSPrankOneCutsLSnumIterations` | `1000.0` | src/cuts/registry.rs |
| `RCSPredCostFixingFalseGap` | `0.0` | src/pricing/fixing_workflow.rs |
| `RCSPmaxNumOfLabelsInHeurEnumeration` | `0.0` | src/pricing/fixing_workflow.rs |
| `RCSPcheckDominInOtherBuckets` | `1.0` | src/pricing/mod.rs |
| `RCSPheurLabelingStrategy` | `0.0` | src/pricing/mod.rs |
| `RCSPlabelDominanceStrategy` | `0.0` | src/pricing/mod.rs; label dominance mode (0 or 1), shared with the C strategy setter |
| `RCSPdynBuckStepAdjustRatioThreshold` | `500.0` | src/pricing/controller.rs |
| `RCSPdynBuckStepAdjustNumBuckArcsThreshold` | `10000.0` | src/pricing/controller.rs |
| `RCSPdynamicNGmode` | `-1.0` | src/api/relaxation.rs; src/master/root/controller.rs |
| `RCSPmaxNumOfColumnsInDynamicNG` | `100.0` | src/api/relaxation.rs; src/master/root/controller.rs |
| `RCSPmaxCycleSizeInDynamicNG` | `5.0` | src/api/relaxation.rs; src/master/root/controller.rs |
| `RCSPmaxNGaverNeighbourhoodSize` | `0.0` | src/api/relaxation.rs; src/master/root/controller.rs |
| `RCSPuseCapacityCuts` | `1.0` | src/master/root/mod.rs; explicit option overrides a configured capacity separator and is inherited by children |
| `RCSPcapacityCutsSeparator` | `1.0` | src/pricing/mod.rs; accepts the available native separator; external CVRPSEP (0) is rejected |
| `RCSPcapCutsMaxNumPerRound` | `100.0` | src/master/root/mod.rs; explicit per-round override for configured capacity separation, including inherited nodes |
| `RCSPresConsKnapsackCutsMode` | `-1.0` | src/master/root/mod.rs; src/cuts/knapsack/mod.rs |
| `RCSPresConsKnapsackCutsMaxNumPerRound` | `100.0` | src/master/root/mod.rs; src/cuts/knapsack/mod.rs |
| `RCSPrankOneCutsArcMemoryType` | `2.0` | src/cuts/registry.rs |
| `RCSPrankOneCutsSpDependentMemory` | `0.0` | src/cuts/registry.rs |
| `RCSPrankOneCutsMaxNumTriplets` | `10000.0` | src/cuts/registry.rs |
| `RCSPrankOneCutsTypeToSeparate` | `0.0` | src/cuts/registry.rs |
| `RCSPuseCovSetsForSepOneRowPackingCuts` | `0.0` | src/cuts/registry.rs |
| `RCSPuseExactComplBoundsInEnumeration` | `1.0` | src/pricing/universe.rs |

The two additional consumers with `clp,highs` are:

| Option | Default | Consumer / scope |
| --- | --- | --- |
| `RCSPmaxNumOfEnumSolutionsForMIP` | `10000.0` | Normal threshold used to normalize the end-of-node threshold; immediate in-CG MIP handoff is not implemented |
| `RCSPmaxNumOfEnumSolsForEndOfNodeMIP` | `0.0` | End-of-node route-count gate, raised to at least the normal threshold; requires complete enumeration and opt-in HiGHS solving |

Both accept only their defaults unless both `clp` and `highs` are enabled. See the
[enumerated integer solving](enumerated-mip.md).

## Unsupported behavior

The default-only acceptance rule does not activate the missing feature. In
particular, enumeration-to-MIP thresholds require the optional HiGHS path, the
stored gap threshold does not replace the current root driver's fixed relative
gap trigger, and stored route-identity settings do not change the branch-safe
arc-path policy.

| Option | Default | Missing behavior |
| --- | --- | --- |
| `RCSPallowRoutesWithSameVerticesSet` | `1.0` | Native route-pool equivalence control is not implemented; node enumeration preserves branch-distinct arc paths |
| `RCSPuseMetaSolver` | `0.0` | The native MetaSolver is not implemented |
| `RCSPprintLevel` | `-2.0` | Native pricing log-level control is not implemented |
| `RCSPuseCompletionBoundsInPricing` | `0.0` | Optional completion-bound pruning in ordinary pricing is not implemented |
| `RCSPdomChecksThresholdInPricing` | `1000000000000000.0` | The native dominance-check stopping threshold is not implemented |
| `RCSPlabelSplitStrategy` | `0.0` | The native configurable label-split strategy is not implemented |
| `RCSPuseComplBoundsInRedCostFixing` | `2.0` | Configurable fixing completion-bound modes are not implemented; the current pairing algorithm is fixed |
| `RCSPuseDSSRInMode` | `0.0` | DSSR reduction is not implemented |
| `RCSPmaxNumOfColumnsInDSSR` | `100.0` | DSSR reduction is not implemented |
| `RCSPmaxCycleSizeInDSSR` | `5.0` | DSSR reduction is not implemented |
| `RCSPprintNGMemories` | `0.0` | Native NG-memory diagnostic output is not implemented |
| `RCSPcliqueCutsMaxNumPerRound` | `0.0` | Clique-cut separation is not implemented |
| `RCSPcliqueCutsMaxSepTime` | `5.0` | Clique-cut separation is not implemented |
| `RCSPrankOneCutsNeighbourhoodSize` | `16.0` | The routing cut adapter has no packing-distance input; native distance-neighbourhood construction is not exposed |
| `RCSPmaxGapToRunEnumeration` | `1.0` | The native enumeration-gap trigger is not implemented; the root driver currently uses its fixed relative-gap rule |
| `RCSPuseVertexIdsInEnumeration` | `0.0` | Configurable vertex-ID enumeration identity is not implemented |
| `RCSPstopRatioForConcatenationInEnum` | `30.0` | The native enumeration concatenation stop-ratio policy is not implemented |

## Verification and next work

`tests/options.rs` checks all defaults and unsupported overrides in
Rust and the C API, setter failure atomicity, dominance-mode synchronization,
capacity enablement in fresh/inherited nodes, and per-round cut limits on two
independent fractional customer groups. Run:

```sh
cargo test --all-features --test options
cargo test --workspace --all-features
cargo check --workspace --all-targets --all-features
```

The option-audit baseline passed **145 tests**; subsequent HiGHS node-MIP
integration brings the all-feature workspace suite to **155 tests**. Default-feature and
all-feature checks passed. A freshly built Python extension smoke check passed
for all 58 support queries, unsupported-setting errors, retained defaults, and
the named dominance setter. The existing unsigned-comparison warning in
`src/hgs/params.rs` remains.

The HiGHS end-of-node MIP path now consumes its route thresholds; native
immediate handoff inside CG remains open. Recover completion-bound modes, route
identity and concatenation policies before enabling their controls. DSSR,
MetaSolver, clique separation and native diagnostic policies remain feature
work. Keep support metadata current when adding a consumer.
