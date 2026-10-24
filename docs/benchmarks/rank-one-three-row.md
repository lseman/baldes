# Parallel three-row rank-one separation

Measured October 24, 2026 on an AMD Ryzen 9 9950X (16 cores / 32 logical
CPUs), Rust 1.97.1, release profile (`opt-level=3`, thin LTO), default features.

Packing and covering enumeration evaluate outer rows through the existing Rayon
pool. Indexed collection preserves `(i, j, k)` traversal order. Candidate IDs,
score accumulation order, and candidate-tree insertion order are unchanged.
The packing pair-value matrix is still built serially. Fewer than 64 rows or a
one-thread Rayon pool uses sequential evaluation. Configure the existing pool
with `RAYON_NUM_THREADS`; this change adds no new option or dedicated pool.

The ignored unit benchmark compares against the original enumeration copied
from commit `2a94f4ca`, with setup and pool construction outside timing. Each
configuration has one warmup pair followed by six measured pairs, alternating
which implementation runs first. Reported times are the upper median of six
samples. Timed work includes pair-matrix preparation (packing), enumeration,
candidate allocation and insertion, and extraction of the ordered candidate
list. Every result must exactly match the original cuts, IDs, scores and ordering.

These are **synthetic kernel workloads**, not captured routing LP solutions.
A deterministic generator makes background columns of 12 sampled row visits,
then adds three half-valued pair columns for each disjoint triangle. The printed
`columns` field is the number of background columns: actual totals are 248,
599 and 1198 for 50, 100 and 200 rows. Packing produces 751, 37 and 66 candidates
respectively; the mixed-background covering workloads produce no violated cuts
but exercise all triple scores. Separate regression tests check nonempty
packing and covering results using triangle-only fixtures.

The final measurement ran without concurrent builds. CPU affinity was not
pinned; all configurations used the same executable and explicit Rayon pools.
Raw output is retained locally in
`results/rank-one-three-row-20261024/benchmark.log`.

| Rows | Separation | Threads | Original (ms) | Parallel (ms) | Speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| 100 | Packing | 4 | 0.267 | 0.199 | 1.34× |
| 100 | Packing | 8 | 0.285 | 0.191 | 1.50× |
| 100 | Covering | 4 | 126.381 | 42.110 | 3.00× |
| 100 | Covering | 8 | 126.953 | 22.743 | 5.58× |
| 200 | Packing | 4 | 1.069 | 0.638 | 1.68× |
| 200 | Packing | 8 | 1.127 | 0.600 | 1.88× |
| 200 | Covering | 4 | 2193.272 | 747.898 | 2.93× |
| 200 | Covering | 8 | 2192.614 | 392.850 | 5.58× |

The 50-row serial fallback was approximately unchanged. One-thread packing at
100–200 rows regressed by roughly 13–16% (about 0.04–0.17 ms), because candidate
buffering adds work without concurrency. One-thread covering stayed about even.
At four threads the measured packing gain was 1.34–1.68× and covering gain was
2.93–3.00×. These figures do not establish an end-to-end root-solve improvement
or characterize the default 32-thread global pool on this machine.

Reproduce from the repository root:

```sh
CARGO_INCREMENTAL=0 cargo test --release --offline --lib -j 2 \
  benchmark_three_rows_parallel -- --ignored --nocapture --test-threads=1
```

Validation:

```sh
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=0 cargo test --offline --lib -j 2
cargo fmt -p baldes --check
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=0 cargo clippy --offline -p baldes --lib --tests -j 2
```

The library suite passed 359 tests (two ignored). Clippy completed with existing
repository warnings. The equivalence tests cover empty and small inputs, the
64-row parallel threshold, 1/2/4-thread pools, and violated fractional triangles
for both packing and covering, including a nonzero initial candidate ID.
