# SIMD dispatch

[Documentation index](../README.md)

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

## Build configuration

See the [build guide](../guides/build.md) for Cargo features and saved Make settings.
SIMD is optional and disabled by default; kernel timings alone do not establish
an improvement in complete solver performance.
