# VRPTW and CVRP profiling — October 21, 2026

This investigation profiles committed `main` at `fe81f9e`, with the Clp master.
It measures root LP solves, not complete integer branch-and-price trees. The
current uncommitted branching work is outside this baseline.

## Findings

The workloads have different bottlenecks. CVRP is primarily pricing-bound;
C202 and rank-one R201 spend substantial time in Clp and other work outside
pricing. Rank-one R103 has both pricing and separator costs.

| Workload | Root seconds | Pricing seconds | Outside pricing seconds | Pricing / root |
| --- | ---: | ---: | ---: | ---: |
| C202 | 10.3132 | 4.5299 | 5.7484 | 44% |
| R201 + rank-one | 7.2163 | 3.0223 | 4.1668 | 42% |
| R103 + rank-one | 1.7602 | 1.1684 | 0.5567 | 66% |
| A-n32-k5 | 0.12285 | 0.11274 | 0.00916 | 92% |
| A-n32-k5 + capacity | 0.36930 | 0.30394 | 0.06402 | 82% |
| E-n22-k4 + capacity/rank-one | 0.03027 | 0.02137 | 0.00828 | 71% |

Each timing column is its own median of three runs after a warmup. Root timing
also includes a little preparation outside the pricing/nonpricing timers.
The CLI's `master_seconds` is elapsed solver time minus pricing time: it includes
cut separation and controller work, and must not be read as pure LP time.

CPU sampling locates the work more precisely:

- **C202:** `ClpPackedMatrix::transposeTimes2` accounts for about 24% of self
  cycle samples, scaling about 6%, and the Clp solve call about 44% inclusive.
  Label sweeps and cross-bucket filtering are the other substantial costs.
- **R201 with rank-one cuts:** Clp solve accounts for about 39% inclusive;
  pricing about 40%. Matrix kernels, scaling, and rank-one local search all
  matter. The run performs 563 CG iterations, 20,192 simplex iterations and
  finishes with 825 cuts.
- **R103 with rank-one cuts:** pricing accounts for about 63% inclusive samples;
  rank-one local search is visible at about 13% inclusive. The Clp solve call
  accounts for about 13%.
- **A-n32-k5:** explicit dominance candidate scans account for roughly 27% of
  self samples in the repeated profile. The two label sweep functions account
  for another roughly 53%; those functions include inlined operations, so that
  share cannot be interpreted as extension-only cost.
- **A-n32-k5 with capacity cuts:** explicit dominance scans account for roughly
  21% of self samples. Sweeps account for roughly 47%.
- **E-n22-k4 with cuts:** capacity-cut route coefficient computation accounts
  for roughly 10% of self samples. Clock reads are also visible on this small
  instance, but that is not evidence to weaken deadline enforcement.

Percentages are approximate user-cycle samples, not wall-clock percentages.
Inclusive call-chain percentages overlap and must not be added. CVRP profiling
repeats each case twenty times to obtain more samples than a single short solve.
Some libc addresses remain unresolved. No heap-allocation profile was collected.

## Tested change: the existing optional SIMD feature

No solver source change is needed for this experiment. The scalar build uses
`--features clp`; the SIMD build uses `--features clp,simd`. The optional feature
uses runtime-dispatched AVX2 dominance candidate scans on this machine.

Paired measurements alternate execution order, use one warmup and three measured
runs per variant, and check `Optimal` status plus matching LP objectives and
bounds. Initial paired medians:

| Workload | Scalar seconds | SIMD seconds | SIMD time change |
| --- | ---: | ---: | ---: |
| C202 | 10.11374 | 11.22161 | +11.0%, variable; rechecked below |
| R201 + rank-one | 7.06804 | 7.10274 | +0.5% |
| R103 + rank-one | 1.78718 | 1.79647 | +0.5% |
| A-n32-k5 | 0.12471 | 0.10966 | **−12.1%** |
| A-n32-k5 + capacity | 0.37142 | 0.34839 | **−6.2%** |
| E-n22-k4 + cuts | 0.03136 | 0.03052 | −2.7%, small absolute difference |

A focused paired C202 recheck produced medians of **9.97328 seconds scalar**
and **10.38214 seconds SIMD**, a **4.1% slowdown**, with matching LP results
and bounds. This confirms that a universal SIMD default is not supported.

The clear measured benefit is on A-n32-k5 with and without capacity cuts.
Rank-one VRPTW shows essentially no gain. This supports testing a separate SIMD
CVRP build; it does not support enabling SIMD universally from these results.
The first C202 series has a 13.49-second SIMD outlier and substantial scalar
variation as well. The host was shared; its one-minute load average varied from
roughly 5 to 13 during this investigation. Small changes need confirmation on an
idle machine. Objective agreement is numerical; this run did not independently
certify every bound.

## Next experiments, in priority order

1. **Reduce dominance scan work on CVRP.** Start with the existing SIMD build,
   then measure candidate counts, bucket sizes, and prefix/suffix scan lengths.
   Tune bucket partitioning and scan dispatch from those distributions rather
   than adding more extension threads. Preserve cost/NG/resource dominance and
   column order, and compare total solve time as well as label counts.
2. **Reduce master work on C202/R201.** C202 performs 926 CG iterations and
   27,726 simplex iterations, with 9,623 final master columns. Test column batch
   sizes, cleanup policy and LP continuation using identical final bounds.
   Matrix/scaling work makes this at least as important as faster labeling.
   Scaling policy changes must be checked for numerical behavior and changed
   CG trajectories; simply disabling scaling is not a validated optimization.
3. **Reuse prepared capacity coefficients.** `CapacityCut::coefficient` currently
   traverses a `BTreeMap` for each route edge and cut. Root state already holds
   `capacity_arc_entries` for physical arcs. Evaluate accumulating those cached
   entries when building route columns, with a fallback for unsupported cases.
   Preserve repeated visits, depot mapping, cut cleanup and row remapping, and
   the existing rounding-after-summation semantics. The measured opportunity is
   particularly clear in E-n22-k4 with cuts.
4. **Profile rank-one separator work separately.** R103 spends a meaningful
   share in local search. Count repeated support/coefficient calculations and
   candidate evaluations before changing its algorithm or accepted cut set.
5. **Extend to integer search and larger CVRPLIB cases.** The three available
   local CVRPLIB inputs have only 16–32 nodes. Root performance does not predict
   integer tree performance, and these small CVRP cases do not establish scaling
   on hundreds of customers. Profile the updated branching implementation after
   that work is committed, and include node count, time-to-incumbent and bound
   progress alongside total time.

These are measured targets and proposed experiments, not implemented speedups.
The discarded batched extension implementation is not part of this branch.

## Reproduction and artifacts

Hardware: AMD Ryzen 9 9950X; Linux x86_64; rustc 1.97.1, LLVM 22.1.6.
Runs are pinned to CPUs 0/1 (distinct physical cores). OMP, OpenBLAS and MKL
thread limits are one. Exact forward/backward pricing retains its existing
concurrency. VRPTW uses 100 customers, unrounded distances and NG size 8.
All cases use reference neighborhoods, numerical bounds, and no enumeration.

Build an optimized binary with symbols and frame pointers, then preserve it
before changing build features:

```sh
CARGO_PROFILE_RELEASE_DEBUG=1 RUSTFLAGS='-C force-frame-pointers=yes' \
CXXFLAGS='-fno-omit-frame-pointer' \
  cargo build --release --features clp --bin baldes
cp target/release/baldes /tmp/baldes-scalar
BALDES_PROFILE_OUT=results/profile-baseline \
BALDES_PROFILE_BUILD='release, clp, debug=1, frame pointers' \
  node scripts/profile_routing.mjs /tmp/baldes-scalar baseline
```

The profiling script records binary SHA-256, commit, compiler, CPU, options,
individual JSON/log outputs, timing medians, and self/inclusive `perf` reports.
It requires Linux `perf` access. Use `BALDES_PROFILE_CASES=C202,A-n32-k5` to
select cases. A fresh output directory or label is required to avoid replacing
an earlier run. The label `baseline` enables CPU sampling.

Build `clp,simd` with the same flags, preserve that binary, and compare with the
existing paired benchmark harness:

```sh
BALDES_PROFILE_OUT=results/profile-paired BALDES_PROFILE_RUNS=3 \
  node scripts/benchmark_solvers.mjs \
  scalar=/tmp/baldes-scalar simd=/tmp/baldes-simd
```

Raw artifacts from this session are retained in
`/home/seman/dev/baldes/target/routing-profile/results-20261021/` (ignored build
storage): `baseline-summary.json`, `baseline-*-perf*.data`, text profiles, paired
JSON/logs under `paired/`, and the C202 recheck under `c202-recheck/`. Large raw
profiles and binaries are intentionally excluded from Git. Both built executables
remain under `target/routing-profile/`; the scalar copy is `baldes-baseline` and
the SIMD executable is `release/baldes`.

Validation: all six paired workloads and the C202 recheck finished `Optimal`
with LP/bound agreement. The SIMD scalar-equivalence and malformed-column tests
passed (2 tests; the manual timing test was ignored). The profiling harness passed
a four-run E-n22-k4 smoke test and Node syntax checking. No solver code was changed.
