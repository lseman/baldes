# Solver profiling — October 20, 2026

Fresh release profiles cover Solomon C202/100 without cuts and R201/100 with
rank-one cuts, using Clp, 8-NN reference NG, unrounded Euclidean distances,
numerical bounds, and enumeration disabled. SIMD is disabled to measure the
next optimization independently. Hardware is an AMD Ryzen 9 9950X; Rust is
1.97.1, with the existing release optimization level 3 and thin LTO.

## Where runtime goes

Approximate shares of sampled CPU cycles, including both labeling workers:

| Area | C202 | R201 rank-one |
| --- | ---: | ---: |
| Clp/Coin routines combined | 43.85% | 46.36% |
| Labeling sweep `run_records` specializations | 26.65% | 18.59% |
| Dominance candidate scans | 5.96% | 5.29% |
| Clp matrix scaling, included in the Clp/Coin total | 5.89% | 5.96% |

`ClpPackedMatrix::transposeTimes2` alone accounts for approximately 22.6% of
C202 and 13.2% of R201. LP matrix operations and the labeling sweep therefore
have substantially more headroom than the dominance SIMD experiment.
The sweep also inlines resource propagation and insertion; its samples should
not all be attributed to its outer loop.

Instruction-level annotation places about 6.3% of the forward ordinary
sweep's sampled cycles at the addition immediately after loading the other
worker's published label count. This is consistent with a stalled dependency
on shared count traffic, rather than expensive integer addition. The old loop
publishes and reads counts before every attempted extension, even when the
configured budget is `usize::MAX`, documented as unlimited.

## Experiments and correctness

The unlimited-budget shortcut skips count publication and reads when no cap
was requested. Both workers share the same immutable controls. Count exchange
is used only to enforce the combined label cap, so it is unnecessary for an
unlimited solve. Finite budgets retain the original per-extension exchange and
limit check; time checks are unchanged.

A regression test exercises single-direction and bidirectional solves, changes
from an unlimited budget to a rejecting finite budget, then to a sufficient
finite budget and back to unlimited, comparing returned columns. The full
Clp/SIMD library and integration suite passed on the final isolated snapshot:
339 tests, with the manual SIMD microbenchmark ignored.

The second retained change skips outgoing-arc walks when a bucket has no
active labels after filtering. Such a bucket has no label extensions, pending
insertions or expansion count to process. Component lower-bound propagation
still runs for every bucket after the component sweep, including empty ones.

A buffer-reuse experiment was rejected. It retained the active-label snapshot
and deferred same-bucket extension vectors across buckets. Despite fewer
allocations, it added approximately 1% executed instructions relative to the
unlimited-budget shortcut on R103 rank-one and did not justify its additional
retained memory and loop overhead. Its frozen binary and samples are retained
for reference.

## Measurement conditions

Other work was running on the machine, confirmed by the user. Wall-time
speedups, especially small differences, are provisional until a quiet rerun.
The final controlled comparison uses frozen executables, CPUs `14,15`, one warmup
and three measured repetitions, with balanced execution order. `perf stat`
records task-clock, instructions and cycles in addition to solver wall,
pricing and master times. OpenMP/OpenBLAS/MKL thread environment variables are
set to `1`. Every non-timing JSON field must match, including bounds, routes,
iteration history, column counts, cut counts and pricing calls.

The initial profiles used CPUs `0,1` and 997-Hz cycle sampling. Raw profiles,
annotated assembly, commands, executable hashes, frozen executables, JSON and
counter samples are retained locally in `results/profile-speed-20261020/`.
`profile-before` is the original executable used for cycle sampling. The final
comparison uses `before-final` and `empty`, built from an isolated snapshot in
`/tmp/baldes-profile-20261020/`. Hashes of every Rust source file prove that only
`src/labeling/engine/sweep.rs` differs between these snapshots. This prevents
concurrent IPM/Clp edits in the shared checkout from entering one side of the
comparison. The preliminary three-build comparison and earlier two-repeat
C202 screen are retained separately, rather than used as final estimates.

## Next profile-guided targets

1. **LP matrix products.** The largest individual hotspot is Clp's
   `transposeTimes2`. Investigate sparse matrix traversal and LP strategy/pivot
   counts before further SIMD work in the comparatively small dominance scan.
2. **Repeated LP setup/scaling.** Scaling consumes about 6% of sampled cycles.
   The inspected Clp C wrapper calls `dual(ifValuesPass)` with the default
   `startFinishOptions = 0`. Clp documents separate options to retain work
   areas and reuse a factorization. These are candidates to benchmark, not
   flags to enable blindly: row/column additions, coefficient/bound edits,
   model forks, numerical recovery and limited solves must remain correct.
3. **Resource-to-bucket lookup.** The sweep annotation includes stalls after
   floating-point division and conversion during bucket lookup. Hoisting
   invariant work may help, but reciprocal multiplication or unchecked
   conversion would need boundary tests to preserve bucket assignment and
   the existing EPSILON arithmetic.

These native LP and lookup changes were not enabled by this investigation.
Concurrent IPM/Clp changes in the shared checkout are outside the isolated
labeling comparison and remain untouched.

## Final isolated comparison

Median solver-reported wall times; negative changes mean less work/time.

| Instance | Baseline | Optimized | Wall time | CPU time | Instructions |
| --- | ---: | ---: | ---: | ---: | ---: |
| C101 | 0.066994 s | 0.066788 s | -0.31% | -0.51% | -0.45% |
| C202 | 13.288938 s | 12.947370 s | -2.57% | -3.15% | -0.43% |
| R103-rank1 | 1.937921 s | 1.917478 s | -1.05% | -1.36% | -0.26% |
| R201-rank1 | 7.620251 s | 7.514607 s | -1.39% | -1.42% | -0.56% |
| R202 | 6.018966 s | 5.857045 s | -2.69% | -3.47% | -1.67% |

The geometric-mean ratios (optimized/baseline) are 0.9839 for wall
time, 0.9801 for CPU task-clock and 0.9932 for
instructions. All five cases execute fewer instructions. These are small,
workload-dependent gains, measured with known background work; a quiet rerun
is needed before treating the wall-time percentages as stable estimates.
The comparison does not include concurrent changes to IPM or Clp in the main
checkout. All 40 warmup/measured outputs match outside the three timing
fields; the frozen snapshot passes 339 library/integration tests.

The retained production diff is limited to the two sweep shortcuts and the
finite/unlimited budget regression test. No SIMD dependency, unsafe code or
new build option is required.

A post-change C202 profile with DWARF stack sampling still attributes about
45.2% of cycles to Clp/Coin and 23.8% to the sweep specializations.
Its non-timing output also matches the control. The LP backend remains the
largest aggregate target for the next optimization round.
