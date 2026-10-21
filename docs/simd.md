# SIMD dispatch and next optimization targets

The optional `simd` feature enables four-label AVX2 dominance scans.
`simd-avx512` includes it and enables eight-label AVX-512F scans on Rust 1.89+.
Both features are off by default. Runtime detection keeps binaries portable:
AVX-512F (when compiled in), AVX2, then the portable implementation. Scans
shorter than 16 remaining labels use the portable implementation.

The AVX-512F kernel uses ordered comparisons, retains the exact EPSILON
addition/subtraction order, and returns the first matching label or prefix
key stop in storage order. It widens eight packed u16 NG masks into eight
u64 lanes without requiring AVX-512BW, VL, DQ or VPOPCNTDQ. All loaded
columns are checked before pointer loads; tails use the scalar kernel.
The comparison tests exercise direct AVX2/AVX-512 paths and runtime dispatch
against an independent scalar reference, including NaNs, infinities,
subnormals, signed zero, epsilon boundaries, offsets and incomplete blocks.

## Other opportunities

These are candidates from source inspection and the existing
[routing profile](routing-profile.md), not implemented speedups.

| Priority | Location | Opportunity | Constraints and measurement needed |
| --- | --- | --- | --- |
| 1 | `src/labeling/engine/sweep.rs`, `Sweep::filter` | Cross-bucket filtering already scans `bucket.hot` columns. Reuse vector resource/NG rejection before loading cold label state. | Preserve its `key >= threshold` stop (insertion uses `>`), presence/branch state, full cut dominance, candidate order and statistics. Measure cross-bucket checks and total pricing time. |
| 2 | `src/labeling/engine/sweep.rs`, `run_records`; `src/labeling/resources.rs` | Batch disposable-resource extension across labels for a common arc. Time-window add/clamp/feasibility can share an eight-label kernel. | Requires staging data from arena records; per-label dimension loops are normally only one or two coordinates. Keep equality resources, ancestry, NG/cuts, insertion order and deadline enforcement. Measure packing cost before adding intrinsics. |
| 3 | `src/extensions/ipm/factor.rs`, `minimum_degree_bitset` | Bitset row union plus degree recomputation could use wide OR and AVX-512VPOPCNTDQ. | Requires separate `avx512vpopcntdq` runtime detection; AVX-512F alone is insufficient. Rows need enough words to amortize dispatch and reduction. Preserve self/pivot-bit clearing and deterministic tie-breaking. This concerns the IPM backend; Clp profiles do not establish its benefit. |

The existing Clp profiles also show substantial matrix/scaling time on C202
and R201. Those kernels live in the native dependency and are a separate
optimization surface. Wider Rust dominance scans cannot remove that cost.
The cached capacity coefficient work is already in the base of this branch;
the historical profile's proposal to add that cache is no longer outstanding.

## Reproduce kernel measurements

```sh
CARGO_INCREMENTAL=0 cargo test --release -p baldes --features simd-avx512 \
  --lib measure_dispatch -- --ignored --nocapture
```

The manual experiment prints scalar, AVX2, AVX-512 and dispatched times for
16, 64, 256 and 1024 labels, including immediate-hit and all-rejected cases.
Use an idle host, repeat runs and distinguish this synthetic kernel timing
from complete solver performance. For root LP comparisons, build separate
binaries with `clp`, `clp,simd` and `clp,simd-avx512` and use
`scripts/benchmark_solvers.mjs` with the same CPU affinity and inputs.

## Initial kernel measurements (Ryzen 9 9950X)

Rust 1.97.1, release optimization, LTO disabled for the kernel experiment,
200,000 calls per cell. A repeat pinned to CPU 0 measured:

| Labels, all rejected | Portable ns/call | AVX2 ns/call | AVX-512 ns/call |
| --- | ---: | ---: | ---: |
| 16 | 4 | 4 | 3 |
| 64 | 16 | 10 | 8 |
| 256 | 64 | 36 | 28 |
| 1024 | 260 | 147 | 111 |

An earlier unpinned run measured 255/145/109 ns for 1024 labels and the
same rounded values for the other rejection cases. Immediate hits took
roughly 1–2 ns regardless of width; those tiny differences are not robust
speedup evidence. The timer reports integer ns/call. These are synthetic
resource/NG rejection scans, not a claim about end-to-end speedups.

## Validation

On this branch, the library and the `main_pricing`, `porting_regressions`
and `fixing_enumeration` integration suites passed with `simd-avx512`:
181 tests passed, with the manual timing experiment ignored in the regular
suite. The manual release timing experiment also passed. AVX2-only and
no-SIMD library checks passed. Clippy completed with existing repository
warnings; the changed Rust files pass rustfmt. The repository-wide rustfmt
check reports pre-existing formatting differences outside this change.

## Root solver comparison

Same source and release profile (thin LTO), Clp backend, CPU affinity 0/1,
one warmup and three measured runs per variant, alternating order. All six
workloads finished Optimal with matching LP objectives and bounds.

| Workload | AVX2 median s | AVX-512 median s | Time change |
| --- | ---: | ---: | ---: |
| C202 | 10.47516 | 10.55580 | +0.8% |
| R201-rank1 | 6.90799 | 6.86902 | -0.6% |
| R103-rank1 | 1.75562 | 1.73619 | -1.1% |
| A-n32-k5 | 0.11408 | 0.11186 | -1.9% |
| A-n32-k5-capacity | 0.32223 | 0.32774 | +1.7% |
| E-n22-k4-cuts | 0.02915 | 0.02974 | +2.0% |

These small solver-level differences are mixed and measured on a shared host.
They do not establish a universal speedup. In particular, the capacity-cut
and E-n22 cases show small regressions; keep AVX-512 opt-in and use the
kernel improvement as evidence for long rejection scans only. Raw binaries,
per-run JSON/logs and metadata are retained under
`target/avx512-task/` in the primary repository (ignored build storage).
The paired benchmark metadata includes binary hashes, compiler and CPU.
