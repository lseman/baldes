# Native kernel experiment

The October 23, 2026 experiment evaluated additional Clp sparse-matrix kernels
and C++ batched label propagation. **No additional matrix variant was accepted.
Batched propagation is experimental and disabled by default.**

## Batched label extensions

Build with `cargo build --release --features native-pricing`. This also enables
Clp. The kernel snapshots resource/NG inputs once per bucket and preflights all
eligible labels for an arc in one native call. SSE2 processes the two resource
coordinates together; other architectures use scalar C++. Rust constructs the
surviving labels, updates cuts and branch state, and inserts them in the
original arc/label order. Same-bucket insertions remain deferred.

Only buckets with at least 32 eligible labels, two disposable resources,
ordinary service-start windows, no coupled resources and an unlimited label
budget use this path. Other cases retain scalar Rust propagation. Native code
preserves forward add-then-clip and backward clip-then-subtract semantics and
rejects non-finite resource values. It does not enable fast math or FMA.

An initial three-repetition comparison on R202 increased median pricing time
from 3.556 s to 3.781 s and root time from 5.432 s to 5.648 s. Sharing label
construction between scalar and native preflight paths reduced that regression
in a follow-up pair, but did not establish a speedup. The feature therefore
remains opt-in. Faster resource arithmetic alone is insufficient to offset
snapshot/output traffic and caller overhead on these workloads.

All initial C202, R202 and R201-rank1 runs terminated `Optimal` with identical
non-timing JSON, including LP/bounds, routes, pricing history, cuts, column
counts and simplex iterations. C202 timing varied substantially; it does not
establish a gain. These are root LP measurements, not integer solve times.

## Sparse matrix candidates

Clp already uses a paired-row SSE2 kernel that computes tableau and weight dot
products in one matrix traversal while retaining sparse-entry addition order.
The experiment retained that implementation. Additional two/four-entry manual
unrolling, unit-coefficient branching, load pipelining and interleaving
independent subset columns were slower in microbenchmarks. A scaled paired-dot
prototype was also slower. These candidates are not compiled into BALDES.

The microbenchmarks used 20,000 sparse columns, approximately 276,000–297,000
nonzeros, and 100 or 900 rows on one pinned CPU. Arithmetic reassociation was
not used: changing sparse summation order can change simplex pivot choices.

## Validation and reproduction

The native preflight has a scalar Rust oracle covering forward/backward
propagation, resource boundaries, overflow/non-finite inputs, jump arcs, NG
rejection and disabled arcs. Unsupported resource shapes are checked separately.
Run these tests with:

```sh
cargo test --features native-pricing --lib extension_batch
```

Local experiment sources, binary hashes, compiler versions, per-run JSON/logs
and summaries live under `results/native-kernels-20261023/` (ignored by Git).
`scripts/benchmarks/benchmark_native_pricing.py` alternates a saved baseline and candidate on CPU 0, with one warmup
and three measured runs per case. Native solver thread environment variables
are set to one. Both builds use release optimization, no optional Rust SIMD,
Clp automatic scaling, 100 customers, NG 8, no enumeration, numerical bounds and
explicit `--rounding none`. The starting working tree contained other changes;
binary SHA-256 values identify the actual compared executables.

To reproduce, save a release `clp` executable, build `native-pricing`, then run:

```sh
python3 scripts/benchmarks/benchmark_native_pricing.py /tmp/baldes-clp target/release/baldes \
  --output results/native-pricing-comparison --runs 3
```
