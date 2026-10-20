# Clp reoptimization investigation — 2026-10-20

## Decision

Retain Clp work areas between LP solves by default, using `dual(0, 1)`.
Keep automatic scaling and dual simplex as defaults. Expose scaling and work-area
retention independently so a workload can select geometric scaling or reproduce
the former discard policy. No feasibility or optimality tolerance is relaxed.

This is a Baldes implementation investigation using the vendored Clp source.
It does not establish the compiled VRPSolver reference's LP policy or explain all
of the runtime difference against that reference.

## Native behavior

The preceding [R201 rank-one profile](performance-profiling-20261020.md)
attributed about 6% of sampled cycles to Clp matrix scaling. That motivated this
investigation; these policy experiments do not isolate scaling-kernel cost.

The former C entry point `Clp_dual(model, 0)` invokes dual simplex with
`startFinishOptions = 0`. Clp's `finish()` deletes work areas and empties the
factorization in that mode. The basis status survives, so the previous code was
already warm-starting the basis; retention changes the native workspace lifecycle.

The documented flags in `vendor/Clp/Clp/src/ClpSimplex.hpp:333` are:

| Flag | Native behavior | Decision |
| --- | --- | --- |
| 1 | Retain work areas and factorization at finish | Enabled by default |
| 2 | Use the old factorization if the row count permits | Not enabled |
| 4 | Skip initialization using change flags; documented as work in progress | Not enabled |

`ClpSimplex.cpp:9131` implements finish/retention, and `createRim()` and `startup()`
perform initialization and validity checks. Retaining storage is not a promise
that a numerical factorization remains reusable after adding rows or columns.
The implementation continues through normal initialization after model edits.

Scaling modes are off (0), equilibrium (1), geometric (2), and automatic (3).
`ClpModel.cpp:3926` invalidates the scaled matrix on a mode change, but a positive
to positive mode change does not itself clear row/column scale factors. Baldes
therefore resets scaling to zero before installing a different positive mode.
This matters when changing options on a live retained model. The option is not
reset on every solve: doing so would discard the cache we intend to retain.

Repeated scaling is affected by both workspace lifetime and matrix edits. It is
not safe to freeze scaling universally during column generation: inserted columns,
cut rows and coefficient edits still need Clp's update/invalidation logic.

## Experiment design

Artifacts are in the ignored directory `results/clp-tuning-20261020/`:

- `screen.json`: flag/scaling/algorithm screening on C202 and R201 with rank-one cuts.
- `combo.json`: interactions and a columns-only primal/dual heuristic.
- `broad.json`: one-pass screening over the 13 original chart cases.
- `benchmark.json`: repeated production-policy comparisons, binary hash, all raw samples,
  median counters and timings; `benchmark.py` is the runner.

Final comparisons use the same frozen release executable for all policies, one
warmup plus three measured runs, balanced policy order, CPUs 14/15, and one-thread
BLAS/OpenMP settings. Each case has 100 customers, NG size 8, reference NG mode,
unrounded distances, enumeration disabled, and numerical bounds. Rank-one cuts
are enabled only on the labeled cases. Every run must finish optimally and match
the reference LP within `1e-6`. Here optimal refers to the root LP/pricing
termination, not a proof of integer optimality.

The machine has other work running, as confirmed by the user. Wall-time medians
are therefore paired with CPU usage, instructions, pivots, pricing calls and peak
RSS. These results are local comparisons, not uncontended hardware records.

## Screening findings

Retaining work areas improved most of the 13 screened cases. Options 3 and 7
provided essentially the same measured trajectory and runtime as option 1 on the
two primary screen cases, so there is no evidence here to justify their additional
reuse assumptions. Always choosing primal simplex made R201 rank-one slower
(about 9.3 s versus 7.6 s in that screen). A columns-only primal heuristic did not
show a consistent advantage over retained dual simplex.

Geometric scaling has a large workload-dependent effect: C202 fell from about
12.7 s to 3.9 s in screening, while R202 grew from about 5.9 s to 6.35 s. Disabling
scaling also helped C202, but less than geometric scaling. These policies change
pivot choices and the column-generation trajectory, not just scaling overhead.
For example, C202's LP calls fell from 1,042 to 313 and simplex pivots from 30,613
to 8,324 under geometric scaling. The final LP objective agreed numerically.

The two optimizations are not additive. Geometric scaling plus retained work
areas took about 5.0 s on C202 and 2.35 s on R103 rank-one, slower than geometric
scaling with discarded work areas. Keep the two controls independent.

## Configuration

CLI examples:

```sh
# New default: automatic scaling, retained work areas
baldes vrptw instances/C202.txt --master clp

# Previous native lifecycle
baldes vrptw instances/C202.txt --master clp --clp-work-areas discard

# Optional policy worth measuring on C202
baldes vrptw instances/C202.txt --master clp --clp-scaling geometric --clp-work-areas discard
```

Rust callers configure `ClpOptions { scaling: ClpScaling::Auto,
keep_work_areas: true }` through `ClpMaster::set_clp_options` or
`RootOptions::clp_options`. Clones inherit their source policy; `continue_from`
keeps the destination's requested policy when restoring a source snapshot.
Python `solve_root` accepts `clp_scaling="auto"` and
`clp_keep_work_areas=True`. Root CLI JSON and Python `RootResult` expose
`simplex_iterations` to distinguish simplex work from column-generation rounds.
The settings affect the Clp backend.

## Validation scope

Regression tests cover all eight scaling/retention combinations, original-row
feasibility, objective and reduced-cost reconstruction, column/row insertion and
deletion, bounds, RHS/coefficient/objective edits, forks and snapshot continuation.
A scaled-row fixture also switches between positive scaling modes on live models.
CLI/Python tests cover configuration and root LP solutions. Existing iteration and
time-limit restoration tests remain applicable to the shared native solve helper.

The strict R201-10 reference trajectory is tested with discarded work areas.
A separate retained-policy test verifies the same objective and valid original
coverage. A changed valid LP policy can change cut/column counts; those changes
must not be hidden by rewriting the old reference counts.

These experiments do not measure full search-tree throughput or snapshot-memory
cost on large trees. Peak RSS below describes separate root processes only.

## Repeated production comparison

Medians of three measured root solves; seconds exclude process startup.

| Case | Former auto/discard | Auto/retain (default) | Change | Geometric/discard | Change |
| --- | ---: | ---: | ---: | ---: | ---: |
| C101 | 0.0663 | 0.0635 | -4.1% | 0.0669 | +1.0% |
| C103 | 1.1396 | 1.0156 | -10.9% | 1.1476 | +0.7% |
| C202 | 12.7810 | 10.3147 | -19.3% | 3.8948 | -69.5% |
| R101-rank1 | 1.1711 | 1.1849 | +1.2% | 1.1694 | -0.1% |
| R103-rank1 | 1.9168 | 1.7755 | -7.4% | 1.6953 | -11.6% |
| R201 | 1.5526 | 1.5392 | -0.9% | 1.6837 | +8.4% |
| R201-rank1 | 7.5367 | 7.0575 | -6.4% | 5.9685 | -20.8% |
| R202 | 5.8206 | 5.8320 | +0.2% | 6.3506 | +9.1% |

The geometric mean runtime ratio for retention across these eight selected cases is 0.938 (each case weighted equally). This is not the full 13-case chart.

| Case | Retain instructions change | Retain peak RSS change |
| --- | ---: | ---: |
| C101 | -3.8% | +2.7% |
| C103 | -8.3% | +1.1% |
| C202 | -20.3% | -6.8% |
| R101-rank1 | +0.6% | +5.8% |
| R103-rank1 | -6.4% | +7.9% |
| R201 | +0.0% | +7.5% |
| R201-rank1 | -6.8% | -4.2% |
| R202 | +0.0% | +0.2% |

Retained storage can increase peak RSS, although changes in the generated master can offset it. The data do not support a universal memory reduction claim.

Measured binary SHA-256: `17880b859d4f550b74268881e601076cfe0c2ae5a22cc69a32c3a3e88347ad91`.

Final validation: `cargo test --offline --features clp,simd --lib --tests`
passed 343 tests (one ignored), and the rebuilt Python extension passed all
59 tests in `python/tests/test_pybaldes.py`. `cargo check --all-features --tests`
also passed during integration. Release Clp build and `git diff --check` passed.
Logs are in the experiment directory.

After the scaling-cache reset fix, the final release binary replayed all three
policies on C202, R201 rank-one and R202. All nine runs matched the measured
LPs within `1e-6` and exactly matched iterations, simplex pivots, pricing calls,
columns, cuts and final master size. See `replay-final.json`. Final binary
SHA-256: `c1cba60fc9f85349bcb4de149f870bfa0664b321e0f1552f1672183221874dec`.

## Timing comparability correction

A subsequent [paired regression check](runtime-regression-check-20261020.md)
found that the chart refresh had changed affinity from CPUs 0/1 to CPUs 14/15.
Those core groups ran at different effective clocks. The policy comparisons in
this report remain comparisons within the same 14/15 placement; they should not
be compared directly with older 0/1 wall times. On the original 0/1 placement,
the saved portable and current default measured 5.6933 / 5.5844 s on R202 and
1.8307 / 1.6871 s on R103 rank-one. The chart has been refreshed consistently
on CPU 0 and CPUs 0/1, retaining the earlier raw data for provenance.
