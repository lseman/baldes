# IPM factorization: cached assembly and lower-triangle updates

This comparison measures the native Rust IPM before and after the assembly-map,
workspace, and dense-kernel changes. The baseline already includes the tree/node
parallel controls. These are BALDES-to-BALDES measurements, not a comparison
against HiGHS timings.

## Changes

- Compute CSC-to-front scatter positions and child-to-parent row maps during
  symbolic analysis. Share immutable maps across solver clones.
- Retain dense scratch, pivot permutations, multiplier vectors, and output
  column capacity across Newton steps. Reset scratch when the symbolic plan
  changes; solver clones start with independent empty scratch.
- Update the contiguous lower triangle and read symmetric pivot candidates from
  that triangle. Avoid mirroring the trailing matrix after every pivot. The
  previous full-matrix kernel remains an independent test oracle.
- Batch tree work over retained scratch buffers. Nested node work uses the same
  bounded Rayon pool; no workspace lock is held during nested execution.
- Recycle consumed Schur buffers through a size-binned pool, capped at 8 MiB per
  solver. The limit covers spare buffers, not live contributions or factors.
  Releasing consumed contributions avoids retaining every clique on deep trees.
- Preserve the serial up-looking fast path and release its old compressed
  factor before allocating the next output.

The dense update is still a rank-1/rank-2 kernel. Blocked panel factorization,
relaxed supernodes, and dependency-driven subtree scheduling remain future work.

## Method

Measured on an AMD Ryzen 9 9950X (16 cores, 32 logical CPUs), Linux x86_64,
with rustc 1.97.1 / LLVM 22.1.6. Both binaries use the repository's release
profile (opt-level 3, thin LTO), the same expanded benchmark, and the same input
and diagonal sequences. No compiler or other validation job ran during timing.

Each process performs one warmup and 30 measured refactorizations. Diagonals
vary as `1 + iteration / repeats`. Tables give the median of three independent
process medians, with the range of those medians where shown. Baseline/candidate
execution order alternates between repetitions. Setup, factorization, and
solve timings are collected separately. One worker is used for `off`, four for
`both` and `choose`; there is no CPU affinity pinning.

The dense-front fixture has 512 identity columns and one column touching all
512 rows. The forest has four independent 128-row components of that form;
the unbalanced forest has components of sizes 32, 64, 128, and 256. The routing
case is the captured `r201_master.txt` fixture. Every solve is checked against
a known KKT witness, with maximum absolute solution error below 1e-8.

```sh
cargo build --release --example ipm_sparse_bench
# Arguments: repetitions, threads, mode, optional exact case name.
target/release/examples/ipm_sparse_bench 30 4 both
target/release/examples/ipm_sparse_bench 30 1 off
target/release/examples/ipm_sparse_bench 30 4 choose
# Example: isolate a large front.
target/release/examples/ipm_sparse_bench 30 4 both dense-front
```

## Warm factorization

All times are microseconds. Speedup is baseline divided by candidate; values
below one indicate a slowdown. Ranges show variation across process medians,
not per-iteration confidence intervals.

### both

| Case | Backend | Before | After | After range | Speedup |
|---|---|---:|---:|---:|---:|
| diagonal | HiPO | 907.70 | 414.67 | 296.40–425.62 | 2.19× |
| diagonal | NormalEquations | 283.31 | 177.65 | 116.94–177.88 | 1.59× |
| banded | HiPO | 306.66 | 231.61 | 184.68–234.04 | 1.32× |
| banded | NormalEquations | 80.51 | 70.87 | 39.69–73.61 | 1.14× |
| overlapping | HiPO | 385.07 | 187.98 | 185.80–215.77 | 2.05× |
| overlapping | NormalEquations | 178.18 | 86.08 | 84.22–86.99 | 2.07× |
| r201-master | HiPO | 2459.59 | 683.02 | 631.91–818.28 | 3.60× |
| r201-master | NormalEquations | 599.44 | 313.70 | 286.23–315.42 | 1.91× |
| dense-front | HiPO | 110550.11 | 6054.69 | 4086.41–6168.10 | 18.26× |
| dense-front | NormalEquations | 107263.80 | 7395.00 | 5287.36–7864.87 | 14.50× |
| forest | HiPO | 653.55 | 299.92 | 226.40–325.48 | 2.18× |
| forest | NormalEquations | 766.54 | 307.48 | 299.31–367.43 | 2.49× |
| unbalanced | HiPO | 10217.10 | 1064.01 | 999.71–1088.83 | 9.60× |
| unbalanced | NormalEquations | 10342.70 | 1224.15 | 1207.97–1250.81 | 8.45× |

### off

| Case | Backend | Before | After | After range | Speedup |
|---|---|---:|---:|---:|---:|
| diagonal | HiPO | 229.72 | 202.96 | 200.91–205.90 | 1.13× |
| diagonal | NormalEquations | 77.25 | 78.94 | 78.56–83.14 | 0.98× |
| banded | HiPO | 17.97 | 15.48 | 15.22–15.52 | 1.16× |
| banded | NormalEquations | 37.13 | 29.64 | 29.53–30.02 | 1.25× |
| overlapping | HiPO | 22.57 | 22.19 | 22.07–22.26 | 1.02× |
| overlapping | NormalEquations | 77.04 | 50.56 | 50.52–50.65 | 1.52× |
| r201-master | HiPO | 221.78 | 230.16 | 227.10–231.35 | 0.96× |
| r201-master | NormalEquations | 459.33 | 267.94 | 267.44–268.07 | 1.71× |
| dense-front | HiPO | 144753.70 | 6852.66 | 6775.27–6864.73 | 21.12× |
| dense-front | NormalEquations | 146127.80 | 7066.25 | 7055.53–7081.35 | 20.68× |
| forest | HiPO | 1697.86 | 522.77 | 518.35–526.52 | 3.25× |
| forest | NormalEquations | 1804.61 | 554.43 | 550.23–557.63 | 3.25× |
| unbalanced | HiPO | 10582.49 | 996.28 | 995.67–1000.63 | 10.62× |
| unbalanced | NormalEquations | 10425.43 | 1160.17 | 1156.01–1164.15 | 8.99× |

### choose

| Case | Backend | Before | After | After range | Speedup |
|---|---|---:|---:|---:|---:|
| diagonal | HiPO | 291.86 | 228.64 | 226.89–231.33 | 1.28× |
| diagonal | NormalEquations | 99.17 | 105.00 | 104.11–105.23 | 0.94× |
| banded | HiPO | 15.63 | 15.78 | 15.59–16.02 | 0.99× |
| banded | NormalEquations | 41.48 | 41.82 | 41.81–68.74 | 0.99× |
| overlapping | HiPO | 22.83 | 23.05 | 23.01–29.29 | 0.99× |
| overlapping | NormalEquations | 82.22 | 60.45 | 59.89–86.13 | 1.36× |
| r201-master | HiPO | 235.40 | 235.11 | 233.80–237.64 | 1.00× |
| r201-master | NormalEquations | 509.70 | 308.86 | 296.69–316.96 | 1.65× |
| dense-front | HiPO | 106945.43 | 6091.26 | 5648.57–6191.26 | 17.56× |
| dense-front | NormalEquations | 107349.71 | 7331.89 | 6807.36–8112.11 | 14.64× |
| forest | HiPO | 677.14 | 319.71 | 316.59–333.69 | 2.12× |
| forest | NormalEquations | 756.22 | 362.51 | 360.59–386.18 | 2.09× |
| unbalanced | HiPO | 9923.35 | 1082.23 | 1078.16–1129.87 | 9.17× |
| unbalanced | NormalEquations | 10256.37 | 1262.81 | 1252.36–1274.31 | 8.12× |

## Setup and solve costs with four workers (`both`)

Setup times are milliseconds; solve times are microseconds. Setup includes
ordering candidates and the new immutable assembly maps. Warmup allocations
are excluded from the measured refactorization loop.

| Case | Backend | Setup before | Setup after | Solve before | Solve after |
|---|---|---:|---:|---:|---:|
| diagonal | HiPO | 2.291 | 2.785 | 40.528 | 26.538 |
| diagonal | NormalEquations | 2.073 | 2.041 | 13.349 | 12.139 |
| banded | HiPO | 0.587 | 0.700 | 17.529 | 16.029 |
| banded | NormalEquations | 0.363 | 0.395 | 6.850 | 7.000 |
| overlapping | HiPO | 0.752 | 1.033 | 28.838 | 13.810 |
| overlapping | NormalEquations | 0.535 | 0.558 | 9.590 | 6.040 |
| r201-master | HiPO | 11.351 | 10.739 | 144.332 | 57.657 |
| r201-master | NormalEquations | 5.590 | 5.448 | 23.409 | 24.538 |
| dense-front | HiPO | 552.868 | 522.025 | 147.053 | 109.645 |
| dense-front | NormalEquations | 563.528 | 541.997 | 110.235 | 97.965 |
| forest | HiPO | 28.924 | 27.753 | 40.247 | 33.279 |
| forest | NormalEquations | 49.140 | 47.535 | 36.658 | 29.219 |
| unbalanced | HiPO | 73.583 | 71.363 | 47.688 | 39.458 |
| unbalanced | NormalEquations | 86.339 | 82.084 | 44.717 | 39.708 |

## Peak resident memory

These are medians of whole-process maximum RSS from `wait4`, in MiB. A process
runs all cases and both backends, so this includes ordering, live fronts, retained
allocator storage, thread stacks, and cached buffers; it is not a per-front or
per-backend allocation count. The raw measurements retain each process peak.

| Mode | Before | After |
|---|---:|---:|
| off | 26.58 | 25.71 |
| both | 54.83 | 54.21 |
| choose | 55.16 | 55.44 |

The largest absolute solution error across all runs was 3.598e-13.

## Interpretation and limits

Compare before/after within the same mode. A faster forced-parallel path can
still be slower than the serial up-looking algorithm on a small sparse matrix.
`Choose` retains that fast path when the structural heuristic finds insufficient
work. Setup and solve costs should be included when judging end-to-end impact.

The synthetic fronts demonstrate dense-kernel and assembly behavior, not speedups
for arbitrary routing instances. These measurements do not cover NUMA scaling,
very deep elimination trees, or multiple simultaneous solver instances.

## Provenance

The implementation was informed by inspection of HiPO's
[precomputed assembly indices](https://github.com/ERGO-Code/HiGHS/blob/73cac48c5340d775a477087198611862559be250/highs/ipm/hipo/factorhighs/Analyse.cpp#L757)
and [frontal storage reuse](https://github.com/ERGO-Code/HiGHS/blob/73cac48c5340d775a477087198611862559be250/highs/ipm/hipo/factorhighs/HybridHybridFormatHandler.cpp#L25).
The Rust kernel preserves this project's pivot policy and regularization rules.

Baseline source fingerprints (the worktree snapshot immediately before these
optimizations, including the preceding parallel-control changes):

- `factor.rs`: SHA-256 `ca014762d3fe2b3a94f900196bad8c73226ebdb96312c67da9ee62196623c014`
- `sparse_solver.rs`: SHA-256 `4a493de1b8c823445075c31f4170f42ba1e54ce9af248516d1bc25cd03685ca7`

Raw results: [ipm-factorization.json](ipm-factorization.json).

## Validation

- `cargo test -p baldes --lib --test ipm`: 381 tests passed.
- `cargo test -p baldes --doc`: four doctests passed.
- `cargo fmt -p baldes --check`: passed.
- `cargo clippy -p baldes --lib --tests`: passed with existing repository warnings.

Regression coverage includes exact comparison against the previous dense kernel,
partial fronts, 1×1/2×2 pivots, an unused upper triangle containing NaNs, changed
numeric values, changed symbolic plans of the same size, appended-column symbolic
reuse, thread modes, workspace reuse, failure recovery, and the Schur cache limit.
