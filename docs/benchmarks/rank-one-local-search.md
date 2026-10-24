# Parallel four- and five-row rank-one local search

Measured October 24, 2026 on an AMD Ryzen 9 9950X (16 physical cores, 32 logical
CPUs), Rust 1.97.1, release profile (`opt-level=3`, thin LTO).
The baseline is commit `90a0d692`, including the parallel three-row enumeration.
Both CLI executables use `--features clp` with optional SIMD disabled.

## Implementation

For four- and five-row classes with at least 32 starts and more than one Rayon
worker, build starting row sets in the original RNG order, improve independent
starts in parallel, then replay deduplication, sub-cut checks and candidate
insertion in start order. This retains the same first duplicate winner, IDs,
score ties and RNG stream across successive classes. Four- and five-row stages
remain sequential. The large precomputed difference bitsets and value tables
are borrowed by all workers, rather than cloned.

Batches contain `clamp(4 * threads, 32, 256)` starts, bounding temporary storage
independently of the iteration budget. Deduplication persists across batches.
Small restart budgets, one-thread execution and six-to-eight-row classes retain
the original interleaved serial evaluation. The tuple/enumeration alternatives
are unchanged. The existing `RAYON_NUM_THREADS` setting controls the worker pool;
there is no new solver option.

## Root solves

Each variant has one warmup followed by three measured runs, alternating which
variant runs first. R103 and R201 use 100 customers, Clp, reference NG
neighbourhoods of eight customers, unrounded Euclidean distances
(`--rounding none`), rank-one cuts, numerical bounds, no fixing/enumeration,
and the default 1,000 local-search starts and maximum of five cut rows.
Native BLAS/OpenMP thread budgets are fixed to one.

The four-thread comparison pins both executables to CPUs 0–3. The 32-thread
comparison pins both to CPUs 0–31, covering the default available CPU count on
this machine. Builds and benchmark runs were sequential, with no concurrent
build during timing. These are local measurements on two instances.

| Instance | Threads | Original median (s) | Parallel median (s) | Time reduction | Original range (s) | Parallel range (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R103 | 4 | 1.645 | 1.521 | 7.6% | 1.644–1.647 | 1.514–1.798 |
| R201 | 4 | 5.793 | 5.551 | 4.2% | 5.778–5.812 | 5.514–5.551 |
| R103 | 32 | 1.630 | 1.491 | 8.5% | 1.629–1.660 | 1.488–1.492 |
| R201 | 32 | 5.830 | 5.516 | 5.4% | 5.816–5.863 | 5.458–5.535 |

Every warmup and measured solve reported `Optimal`, with matching LP/bound
values, cut counts, cut rounds, generated-column counts, pricing calls, root
iterations and simplex iterations. R103 generated 348 cuts and reached LP
1213.623943085003; R201 generated 828 cuts and reached LP 1147.8037726210775.
The four-thread R103 candidate had one slower sample (1.798 s), so its median
gain is not a guarantee for individual runs.

## Local-search kernel

The ignored unit benchmark uses deterministic synthetic columns: 100 rows,
256 elementary columns visiting six distinct rows each, ring neighbourhoods of
eight rows, and 1,000 starts. It times start generation, local improvement,
violation filtering, deduplication and candidate insertion for a single class.
Sub-cut checks are skipped, matching the default. Table construction and pool
creation are outside the timed region. One warmup
pair precedes three measured pairs, alternating order; the table gives medians.
Every output and next RNG draw must match the original serial oracle.

These synthetic columns are not captured routing LP solutions. Four-row
packing/covering and five-row packing produce no accepted candidates here;
five-row covering produces 23. Separate regression fixtures exercise accepted
four- and five-row packing/covering cuts and duplicate suppression across batches.

| Separation | Rows | Threads | Serial (ms) | Parallel (ms) | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Packing | 4 | 1 | 8.578 | 8.655 | 0.99× |
| Packing | 4 | 4 | 8.091 | 4.004 | 2.02× |
| Packing | 4 | 8 | 8.457 | 3.204 | 2.64× |
| Packing | 4 | 32 | 8.401 | 3.240 | 2.59× |
| Covering | 4 | 1 | 162.800 | 163.960 | 0.99× |
| Covering | 4 | 4 | 160.785 | 46.727 | 3.44× |
| Covering | 4 | 8 | 164.206 | 27.598 | 5.95× |
| Covering | 4 | 32 | 163.342 | 13.560 | 12.05× |
| Packing | 5 | 1 | 10.831 | 10.468 | 1.03× |
| Packing | 5 | 4 | 10.512 | 5.620 | 1.87× |
| Packing | 5 | 8 | 10.625 | 4.176 | 2.54× |
| Packing | 5 | 32 | 10.872 | 3.510 | 3.10× |
| Covering | 5 | 1 | 395.554 | 394.445 | 1.00× |
| Covering | 5 | 4 | 393.299 | 110.681 | 3.55× |
| Covering | 5 | 8 | 395.236 | 66.370 | 5.96× |
| Covering | 5 | 32 | 396.033 | 36.878 | 10.74× |

One-thread timings stayed within approximately three percent of the original.
Packing benefits less than covering because the serial construction and ordered
merge form a larger share of this workload. The root-solve timings above measure
the resulting end-to-end effect rather than extrapolating from these kernels.

## Validation and reproduction

The default-feature library suite passed 362 tests (three ignored). The new
serial-oracle tests cover all seven five-row packing and ten five-row covering
classes, successive calls with a persistent RNG/candidate arena, sub-cut checks,
neighbour restrictions, 0/1/31/32/65 starts, 1/2/4 threads, nonzero initial IDs,
accepted cuts, signed covering multipliers, and duplicate suppression spanning
multiple batches. Six-row packing is checked as a serial fallback.
Formatting, Clippy (existing repository warnings), and Node syntax validation
also completed.

Build and save the baseline executable from `90a0d692`, then build the candidate
with the same release settings:

```sh
CARGO_INCREMENTAL=0 cargo build --release --offline --features clp -j 2
# Save target/release/baldes before switching to the candidate source.
```

Run the paired root benchmark from the repository root, using a fresh output
directory each time:

```sh
BALDES_LS_OUT=results/rank-one-local-search/paired-4 \
  node scripts/benchmarks/benchmark_rank_one_local_search.mjs \
  /path/to/baseline /path/to/candidate

BALDES_LS_THREADS=32 BALDES_LS_CPUS=0-31 \
  BALDES_LS_OUT=results/rank-one-local-search/paired-32 \
  node scripts/benchmarks/benchmark_rank_one_local_search.mjs \
  /path/to/baseline /path/to/candidate

BALDES_LS_BENCH_STARTS=1000 CARGO_INCREMENTAL=0 \
  cargo test --release --offline --lib -j 2 benchmark_parallel_local_search_starts \
  -- --ignored --nocapture --test-threads=1

CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=0 cargo test --offline --lib -j 2
cargo fmt -p baldes --check
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=0 \
  cargo clippy --offline -p baldes --lib --tests -j 2
node --check scripts/benchmarks/benchmark_rank_one_local_search.mjs
```

Raw samples, binary hashes, CPU/thread settings, build/test logs and summaries
are retained locally under `results/rank-one-local-search-20261024/`.
