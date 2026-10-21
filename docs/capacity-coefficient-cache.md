# Cached capacity-cut coefficients

Experiment on `perf/capacity-coefficient-cache`, based on `e293727`.
Root route-column construction and pool refresh now accumulate capacity-cut
coefficients from the existing `capacity_arc_entries` cache. This avoids a
`BTreeMap` lookup for each route edge and active cut.

Each cut is summed in route order and rounded once at the end, retaining the
existing integer coefficient semantics. Repeated visits count repeatedly. Row
retirement/remapping uses the existing cache lifecycle. Sparse physical arc IDs
are resolved through the original problem's arc index. Unsupported route/depot
representations fall back to the customer-sequence calculation.

Snapshot, fixed-route, certificate and new-cut installation calculations retain
the existing coefficient path. No public API or cut selection behavior changes.

## Measurements (October 21, 2026)

Release builds with Clp, SIMD disabled, debug information and frame pointers.
CPU: AMD Ryzen 9 9950X. Solves pinned to CPU 0/1, external numerical library thread
limits set to one. One warmup per binary; alternating baseline/candidate order.
These are root LP measurements on a shared host, not integer tree solve times.

Initial nine-run paired medians:

| Workload | Baseline seconds | Cached seconds | Time change |
| --- | ---: | ---: | ---: |
| A-n32-k5, no cuts | 0.124319 | 0.122817 | −1.2%, small control difference |
| A-n32-k5 + capacity | 0.375844 | 0.365478 | −2.8% |
| E-n22-k4 + capacity/rank-one | 0.030991 | 0.029285 | −5.5% |

Independent fifteen-run repeat:

| Workload | Baseline seconds | Cached seconds | Time change |
| --- | ---: | ---: | ---: |
| A-n32-k5 + capacity | 0.367709 | 0.359653 | **−2.2%** |
| E-n22-k4 + capacity/rank-one | 0.030525 | 0.029409 | **−3.7%** |

Final routes, objectives, bounds, cut/column counts, CG iterations, pricing calls
and simplex iterations were identical across all 74 pairs, including warmups and
the VRPTW controls. The no-cut path can also avoid the former
unconditional customer-list allocation, but its small timing difference is not
claimed as a separate speedup.

A three-run R103/rank-one control initially appeared about 9.7% slower, with
considerable timing variation. A seven-run paired recheck measured 1.768861 s
baseline and 1.777123 s cached (**+0.47%**), so the large slowdown did not persist.
The CVRP improvements are modest and should be tested on larger inputs before
making broader claims.

A follow-up `perf` sample over 100 repeated E-n22-k4 solves per variant showed
`CapacityCut::coefficient` dropping from 5.49% to 2.07% of self user-cycle
samples. Remaining calls include new-cut installation. These sample shares are
not wall-clock percentages; they support that the intended work was reduced.
Raw profiles are retained under the `profile/` results directory.

## Validation

261 tests passed with Clp enabled: all library tests plus `capacity_root`,
`column_cleanup`, `partial_solution`, `node_lifecycle`, `preprocessing` and
`branch_price`. Three new regressions cover small walks, repeated visits, sparse
and parallel arcs, fractional/negative weights, rounding, retired/remapped rows,
and routes requiring fallback.

Changed Rust files pass rustfmt checking and `git diff --check`. Clippy completed
successfully with existing repository warnings and none in the changed files.

## Reproduce

Build both the base and candidate with identical settings, preserving each
executable. This session used:

```sh
CARGO_PROFILE_RELEASE_DEBUG=1 RUSTFLAGS='-C force-frame-pointers=yes' \
CXXFLAGS='-fno-omit-frame-pointer' \
  cargo build --release --features clp --bin baldes

BALDES_PROFILE_OUT=results/capacity-cache \
BALDES_PROFILE_CASES=A-n32-k5-capacity,E-n22-k4-cuts \
BALDES_PROFILE_RUNS=15 \
  node scripts/benchmark_solvers.mjs baseline=/path/to/base cached=/path/to/candidate
```

Raw timing JSON, logs, executable hashes and metadata are retained under
`/home/seman/dev/baldes/target/capacity-cache/results/`: `paired/`, `recheck/`,
`vrptw-control/` and `vrptw-recheck/`. Executables are `baldes-baseline` and
`release/baldes` under `target/capacity-cache/`. Large build and profiling artifacts
are ignored by Git.
