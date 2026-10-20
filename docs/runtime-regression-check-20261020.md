# R202 and R103 rank-one regression check — 2026-10-20

## Finding

The apparent regression in the chart was not reproduced when the saved and current
executables ran on the original CPU placement. The chart refresh changed affinity
from CPUs 0/1 to CPUs 14/15. Those are different physical cores and different L3
cache groups on this Ryzen 9 9950X. Effective clock estimates from cycles divided
by CPU task time were about 5.5 GHz on 0/1 and 5.2 GHz on 14/15. Background load
also produced outliers. The affinity change made the chart a poor comparison with
the earlier timings; the replacement chart uses CPU 0 and CPUs 0/1.

## Paired experiment

Four configurations, two workloads, two placements, one warmup plus three measured
runs per configuration: 64 sequential solver runs. Configuration order rotates
each repetition. Placements were tested in separate blocks (14/15, then 0/1),
so temporal machine-state changes remain a limitation. All settings use Clp,
100 customers, reference NG 8, unrounded distances, no enumeration, numerical
bounds and one-thread BLAS/OpenMP environment variables. No builds ran during
these timing batches.

| Workload / CPUs | Saved portable | Saved SIMD | Current discard | Current retain (default) |
| --- | ---: | ---: | ---: | ---: |
| R202@14,15 | 6.0291 s | 5.9821 s | 6.0713 s | 6.2665 s |
| R103-rank1@14,15 | 1.9302 s | 1.9284 s | 1.9202 s | 1.7800 s |
| R202@0,1 | 5.6933 s | 5.6775 s | 5.5663 s | 5.5844 s |
| R103-rank1@0,1 | 1.8307 s | 1.8366 s | 1.8238 s | 1.6871 s |

On CPUs 0/1, the current default is 1.9% faster on R202 and 7.8% faster on
R103 rank-one than the saved portable binary. Against saved SIMD, it is 1.6%
and 8.1% faster respectively. These are local repeated-run comparisons, not
uncontended hardware records or statistical confidence intervals.

## Work and hardware evidence

| Workload / CPUs | Configuration | Instructions (billions) | Cycles (billions) | Effective GHz estimate |
| --- | --- | ---: | ---: | ---: |
| R202@14,15 | old-portable | 123.614 | 33.404 | 5.181 |
| R202@14,15 | old-simd | 121.422 | 33.331 | 5.208 |
| R202@14,15 | current-discard | 121.555 | 33.115 | 5.161 |
| R202@14,15 | current-retain | 121.561 | 33.863 | 5.148 |
| R103-rank1@14,15 | old-portable | 36.917 | 10.491 | 5.233 |
| R103-rank1@14,15 | old-simd | 36.590 | 10.459 | 5.224 |
| R103-rank1@14,15 | current-discard | 37.003 | 10.371 | 5.228 |
| R103-rank1@14,15 | current-retain | 34.617 | 9.743 | 5.236 |
| R202@0,1 | old-portable | 123.614 | 33.438 | 5.494 |
| R202@0,1 | old-simd | 121.422 | 33.422 | 5.497 |
| R202@0,1 | current-discard | 121.556 | 32.452 | 5.497 |
| R202@0,1 | current-retain | 121.562 | 32.725 | 5.517 |
| R103-rank1@0,1 | old-portable | 36.916 | 10.474 | 5.527 |
| R103-rank1@0,1 | old-simd | 36.591 | 10.540 | 5.528 |
| R103-rank1@0,1 | current-discard | 37.003 | 10.402 | 5.525 |
| R103-rank1@0,1 | current-retain | 34.619 | 9.714 | 5.528 |

The GHz figures are ratios of median hardware cycles and median task-clock
time; they are effective estimates, not direct boost-clock settings. No CPU
governor, priority, affinity of other processes or system tuning was changed.

R202 has identical complete non-timing JSON across old/new binaries, policies
and placements, including histories, routes, bounds and LP objective (the newly
added simplex counter is excluded from cross-version comparisons). Its work
remains 435 CG iterations, 470 pricing calls and 11,975 columns. Both current
policies use 14,957 simplex pivots. Current portable instructions are about 1.7%
lower than saved portable instructions. There is no evidence of added solver
work explaining the slower chart time. Retention is essentially neutral here;
the 0.3% timing difference between current policies on CPUs 0/1 is too small
to establish an advantage or regression from three samples.

R103 rank-one uses 254 CG iterations and 289 pricing calls in the saved and
current-discard paths. Current-retain uses 235 iterations and 269 calls. It
generates 357 cuts instead of 343, so it follows a different valid LP trajectory.
Instructions fall from about 36.9 to 34.6 billion, and objectives agree within
1e-6. Complete non-timing results match across placements within each policy.

Timing variation is visible even for fixed binaries/settings. Current-retain
R202 on CPUs 14/15 had a 7.125 s sample with 385 involuntary switches; the earlier
warmup took 5.883 s with the same work. The chart batch also put R103 rank-one
at about 2.03 s on two cores; the paired batch on those same cores measured
1.78 s. Such variation cannot be assigned to a source change. It does not
justify selecting only the fastest individual runs for the chart.

## Provenance and scope

Saved portable baseline: `349b84e`, reused from the optional-SIMD experiment.
Current binary is the frozen final Clp build used by the chart, with SIMD disabled.
The exact hashes, paths and arguments for all four configurations are in
`results/regression-check-20261020/paired.json`; individual JSON, perf statistics,
resource usage, logs and environment snapshot are retained alongside it.
`validation.json` records full non-timing checks across all 64 outputs.

No solver algorithm change was made for this investigation. Because the suspected
regression did not persist on matched placement, no speculative hot-path patch
was introduced. A separate experiment would be needed to claim performance on
other CPUs, models or full search trees.

## Corrected full chart

The replacement [chart](benchmark.svg) reran all 13 workloads on CPU 0 and
CPUs 0/1, one warmup and three measured runs per configuration. All 104 outputs
match the preceding 14/15 batch in every non-timing field. Raw data and scripts
are in `results/chart-refresh-cpu01-20261020/`; the earlier batch is preserved.
Geometric means: 0.7916 s (one core), 0.7693 s (two cores).
