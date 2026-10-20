# IPM performance investigation — 2026-10-20

The R201 root slowdown was dominated by repeated sparse linear algebra and thousands of interior-point Newton iterations. Column-generation iteration counts alone concealed this work. This investigation changes the Rust implementation and measures the numerical kernels separately from the routing model sequence.

## Scope and measurement conditions

- Reference: Baldes commit `7938eca`, including the earlier objective normalization and wide-upper-bound initialization fixes.
- AMD Ryzen 9 9950X, Rust 1.97.1, release optimization, thin LTO, `CARGO_PROFILE_RELEASE_DEBUG=1`, `clp` feature, SIMD disabled for timed binaries.
- Kernel/LP benchmarks pinned to CPU 0; root runs pinned to CPUs 0,1. OpenMP, OpenBLAS and MKL thread counts set to one. Performance governor.
- Root workload: Solomon 100 customers, unrounded distances, reference NG pricing, NG size 8, IPM master, no enumeration, numerical bounds.
- Other machine activity was previously confirmed by the user. Wall times are indicative; paired repetitions and instruction counts support the comparisons. Full-root screens are single measurements unless explicitly stated otherwise.
- Frozen executables, hashes, scripts, logs, profiles and raw measurements are in the ignored local directory `results/ipm-optimization-20261020/`. They are not distributed with the source report.

Reproduce a full root run with the release CLI:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0,1 \
  target/release/baldes vrptw instances/R201.txt --master ipm --reference-ng \
  --no-enumeration --customers 100 --ng 8 --rounding none --time 240 \
  --json /tmp/r201-ipm.json
```

Append `--iterations 100` for the bounded routing comparisons.

## What the profile showed

The original profiled full R201 run took 81.72 s: 81.05 s in the master and 0.64 s in pricing. It reached optimal root termination after 312 CG iterations and 8,608 columns. An unprofiled baseline took 80.93 s.

Flat sampled cycles attributed 36.03% to numeric multifrontal factorization, 9.54% to factor solves, 8.92% to symbolic analysis, 6.91% to block ordering, 4.92% to sparse matrix multiplication and 4.51% to refinement. Allocation and deallocation added further cost. Factorization annotation highlighted child Schur assembly and copying, not only pivot arithmetic.

Each restricted master solve creates a fresh standard form, equilibration, symbolic solver and starting point. Symbolic structure is reused across Newton steps of that LP, but not across successive CG models. The original centering policy required 8,994 Newton updates in the full R201 sequence; retries and additional correction solves add work beyond that count.

## Implemented changes

1. Store child Schur complements as packed lower triangles; assemble only the stored triangle. Reuse a dense frontal workspace across fronts.
2. Keep the independent diagonal primal block in natural order, avoiding a large primal intersection graph. Retain exact minimum degree on the dual Schur graph.
3. Allow centering to respond to the affine step, bounded in `[1e-4, 0.1]`. The previous expression always returned 0.1 for valid step lengths. Original-coordinate feasibility checks, gap requirements and regularization remain in place.
4. Select a sparse up-looking LDL factorization for sufficiently sparse augmented systems. It uses the existing symbolic ordering, negative primal/positive dual pivots, and existing original-KKT refinement. Unsafe pivots retry the pivoted multifrontal implementation.
5. Store sparse factors in contiguous CSC numeric arrays with a shared immutable pattern. Refactorization owns new numeric values, so cloned solvers remain independent.
6. Cache normal-equation structure and row-to-column source positions, accumulating matrix values directly instead of rebuilding and sorting duplicate triplets on every refactorization.
7. Initialize from regularized primal/dual least-squares estimates, with positive HSD variables and unit upper-bound complementarity products. Unsafe estimates retain the unit starting point. Empty systems and very tight finite bounds keep the original initializer. This adds one factorization and two solves per eligible LP.
8. Report cumulative `ipm_iterations` in root CLI JSON and Python results. This counts completed Newton updates reported by returned IPM solves; it does not separately count retries, correction solves or initialization work. Forked masters reset the counter, like the simplex counter.

The sparse path is selected when the primal block has at least twice the dual dimension and symbolic L has at most 32 off-diagonal entries per system variable on average. These are conservative local heuristics, not a reproduction of HiPO's cost model. Tiny, non-finite and wrongly signed pivots retain the existing fallback. Dense systems continue to use multifrontal factorization.

## HiPO ideas and source evidence

The linked [HiGHS PR 2994](https://github.com/ERGO-Code/HiGHS/pull/2994) integrates HiPO into Python and related build paths. Algorithm comparisons here use the actual HiGHS source at immutable commit `2e5ad23c382ab9998d512d59987ebc2a1ecacbfa`:

- [Solver.cpp](https://github.com/ERGO-Code/HiGHS/blob/2e5ad23c382ab9998d512d59987ebc2a1ecacbfa/highs/ipm/hipo/ipm/Solver.cpp): lines 1266–1295 select up-looking factorization for inexpensive or assembly-dominated systems; lines 675–855 construct least-squares starting estimates; lines 897–916 adapt centrality corrections to step lengths.
- [UpLookingSolver.cpp](https://github.com/ERGO-Code/HiGHS/blob/2e5ad23c382ab9998d512d59987ebc2a1ecacbfa/highs/ipm/hipo/ipm/UpLookingSolver.cpp): sparse row updates through previously computed column prefixes, followed by triangular solves.
- [KktMatrix.cpp](https://github.com/ERGO-Code/HiGHS/blob/2e5ad23c382ab9998d512d59987ebc2a1ecacbfa/highs/ipm/hipo/ipm/KktMatrix.cpp): lines 239–306 fill cached normal-equation structure using row/column correspondence and scratch accumulation.
- [Analyse.cpp](https://github.com/ERGO-Code/HiGHS/blob/2e5ad23c382ab9998d512d59987ebc2a1ecacbfa/highs/ipm/hipo/factorhighs/Analyse.cpp): relaxed supernode selection evaluates estimated arithmetic and sparse-operation costs.

These are independently implemented numerical ideas. Our homogeneous algorithm and bound handling differ from HiPO, so its starting-point and centering rules cannot be transferred without numerical validation. This investigation does not benchmark Baldes against HiPO or establish solver-level performance parity.

## Full R201 screens

| Candidate | Root seconds | CG iterations | Newton updates |
| --- | ---: | ---: | ---: |
| Original | 80.935 | 312 | 8,994 with equivalent instrumented policy |
| Packed child Schur storage | 73.357 | 312 | — |
| Natural primal ordering | 68.166 | 312 | — |
| Frontal workspace reuse | 64.751 | 312 | — |
| Instrumented constant centering | 62.626 | 312 | 8,994 |
| Adaptive centering | 51.110 | 297 | 7,511 |
| Sparse left-looking prototype | 31.794 | 293 | 7,263 |
| Sparse up-looking algorithm | 31.195 | 298 | 7,499 |
| Contiguous sparse factors | 28.316 | 298 | 7,499 |
| Least-squares initialization screen | 24.888 | 297 | 6,330 |
| Final, median of three full runs | 24.624 | 297 | 6,330 |

All screens reached optimal root termination. The contiguous-factor LP objective was `1140.1301452407072`; the independent Clp bound was `1140.1301445338584` (difference about `7.1e-7`). Contiguous storage preserved all non-timing JSON output exactly relative to the preceding up-looking candidate. The initialization candidate reached objective `1140.1301447923233`, within `3e-7` of the independent bound. Its 24.89 s is about 3.25× faster than the original 80.93 s screen, with about 30% fewer Newton updates than the equivalent instrumented original policy.

The final binary completed three full runs in 24.59–24.67 s (median 24.624 s), with identical non-timing output. Against the single unprofiled original baseline, this is about 3.29× faster. Median retired instructions fell from 1.587 trillion to 508.8 billion (67.9% reduction); peak RSS fell from 131,808 to 99,132 KiB (24.8%). These instruction counts still include changes in the generated model sequence.

Changing numerical algorithms can change the dual sequence and generated columns; equal CG histories are not expected across all candidates.

The original-to-workspace comparison, before adaptive centering and sparse LDL, also used one warmup plus three paired repetitions. Median time for the first 100 CG iterations improved from 6.171 to 5.193 s on R201 and 11.692 to 9.420 s on R202; full C101 improved from 0.196 to 0.172 s. Instruction counts fell 13.1%, 13.2% and 9.1%, respectively. The 100-iteration cases ended at the requested iteration limit and are not full optimal-root benchmarks.

## Final paired routing comparisons

One warmup plus three measured repetitions per executable, alternating order, on the final implementation:

| Workload | Original median seconds | Final median seconds | Time reduction | Instruction reduction |
| --- | ---: | ---: | ---: | ---: |
| R201, first 100 CG iterations | 6.186 | 2.331 | 62.3% | 63.7% |
| R202, first 100 CG iterations | 10.677 | 4.165 | 61.0% | 63.7% |
| C101, full root | 0.197 | 0.093 | 52.5% | 51.5% |

The bounded runs deliberately stop at the CG iteration limit. Their model sequences can differ, so these are equal-workload limits, not identical LP sequences. Every repetition of each executable produced identical non-timing output. The full C101 LP objectives agree within `1e-6`.

## Kernel and fixed-LP measurements

`cargo run --offline --release --example ipm_sparse_bench -- 50` measures setup, changing-diagonal refactorization and triangular solves. Every solve is checked against a known KKT witness. The captured R201 pattern has 101 rows, 1,694 columns and 22,923 nonzeros.

The packed/order/workspace changes reduced captured-pattern factorization from about 1,345 to 1,055 microseconds. The sparse prototype reduced it further to about 290 microseconds. At that point normal equations still required about 4,060 microseconds, dominated by triplet construction and sorting. Cached assembly reduced their paired median to 453 microseconds (about 9×), with setup falling from 8.38 to 4.77 ms. The witness error remained below `4e-13`. On the same three paired process runs, sparse augmented refactorization with contiguous factors took about 225–227 microseconds.

`cargo run --offline --release --example ipm_master_bench -- 3 hipo` measures a fixed captured R201 LP, with one warmup, independently of CG. The original instrumented policy used 21 Newton steps with a relaxed gap and 22 with strict gap, taking about 47–49 ms. Adaptive centering used 22 steps in both cases: its aggregate routing improvement does not imply fewer steps for every fixed LP. Sparse factors reduced this LP to about 29 ms, and contiguous storage to about 27 ms. Cached normal-equation assembly reduced that backend from about 114 ms to 21 ms at the same 22 Newton iterations, passing the same original-coordinate residual and objective checks. This is a fixed-LP result; the routing default remains augmented LDL.

Least-squares initialization reduced the fixed captured LP from 22 to 17 Newton steps and about 23 ms, including initialization work. The general LP and phase/bound regression suites passed before the full-root screen.

## Remaining profile

A final full-root flat profile at 499 Hz, with zero lost samples and identical non-timing output to the unprofiled final run, shows the cost has moved:

| Symbol | Sampled cycles |
| --- | ---: |
| Symbolic analysis | 26.12% |
| Sparse up-looking numeric factorization | 17.17% |
| Sparse matrix multiplication | 10.36% |
| Original-KKT refinement | 9.18% |
| Forward/backward substitution combined | 11.14% |

The next measured target is symbolic analysis and repeated matrix construction across CG solves. Reusing a previous analysis requires validating the evolving matrix pattern; blindly reusing it when columns change would be incorrect. Warm starts may also reduce Newton updates further. These remain future work, not implemented behavior.

## Validation

The final source passes 368 Rust tests with `cargo test --offline --features clp,simd --lib --tests`, an all-features `cargo check`, and 80 Python tests after rebuilding the extension with the `clp` feature. New numerical regressions cover ordering/fill invariance, unsafe-pivot fallback, independent cloned factor values, repeated cached normal-equation assembly and invalidation/recovery after failure. Existing randomized KKT witnesses, scaled LPs, infeasibility/unboundedness cases, master edits, gap policy and forks continue to pass.

The relaxed-versus-strict gap fixtures now use costs `[1, 2, 8e4]` so the two targets remain distinguishable with faster convergence; their feasibility and gap assertions were retained.

## Remaining work

Cross-CG symbolic reuse, reuse of previous primal/dual iterates, relaxed supernodes, blocked dense kernels and parallel factorization remain separate opportunities. Sparse default routing and dense general LPs need different strategies; a gain on R201 does not establish a universal fastest backend.
