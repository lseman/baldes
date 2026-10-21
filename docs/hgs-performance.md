# HGS performance

Run a fixed-work benchmark with:

```sh
cargo run --release --offline --example hgs_bench -- R201 300 5
```

Arguments select the Solomon instance in `instances/`, the non-improving
iteration limit, and the number of seeds (starting at zero). The benchmark
uses no time limit and reports generations, best cost, and elapsed seconds.
An improvement resets the stall counter, so total generations can exceed the
iteration limit. Compare matching seeds and check costs and generation counts
before interpreting timing differences.

## Distance and capacity bounds

The relocate and swap operators first apply their existing distance checks.
For surviving inter-route candidates, they now compute the resulting capacity
penalties before merging time-window segments. New time-window penalties are
nonnegative, so distance plus new capacity penalties minus existing route
penalties is a lower bound on the final cost change. A positive bound rejects
the candidate immediately. Candidates near zero still receive full evaluation.
The move order, random number stream, and full cost calculation are unchanged.

A `perf` cycle profile of R201 (300 stalled iterations, seeds 0–2) attributed
roughly 65% of samples to the six relocate/swap operators. This motivated
pruning their expensive candidate evaluations rather than changing the search
neighborhood or stopping criteria.

## Validation

```sh
cargo test --release --offline -p baldes --lib extensions::hgs
cargo test --release --offline -p baldes --test hgs_driver --test hgs_e2e
```

The fixed-work regression test checks pre-optimization costs and generation
counts for C201 and R201, as well as route feasibility. Existing tests cover
fractional distances, explicit distance matrices, determinism, and solver quality.

## Measured results (2026-10-21)

AMD Ryzen 9 9950X, x86-64 Linux, rustc 1.97.1, default release profile
(`opt-level=3`, thin LTO), no optional features. Each version ran seeds 0–4
three times per instance, serially, pinned to logical CPU 7 with `taskset -c 7`.
Version order alternated between passes. No builds or tests ran concurrently.
Times include HGS setup and population construction; no separate warm-up
was discarded. The table averages the three-run median for each of five seeds.

| Instance | Before (s) | After (s) | Runtime reduction |
| --- | ---: | ---: | ---: |
| C101 | 0.2040 | 0.1975 | 3.2% |
| R101 | 0.6741 | 0.6451 | 4.3% |
| C201 | 0.2166 | 0.2159 | 0.3% |
| R201 | 0.9883 | 0.8961 | 9.3% |

All 60 paired runs retained identical best costs and generation counts.
These are local timings on 100-customer Solomon instances, not a guarantee
for other instance sizes or distributions. C201 showed no material change.
Raw measurements: [CSV](../results/hgs-performance-20261021.csv).

An initial unpinned comparison had a large timing spike in one R201 batch.
CPU pinning and repeated per-seed measurements were added to reduce timing
noise; small differences such as C201 should not be interpreted as a speedup.

All 25 HGS unit tests and six integration tests passed. Formatting and diff
whitespace checks passed for the changed operators and benchmark. Strict
Clippy (`-D warnings`) is blocked by existing repository warnings.
