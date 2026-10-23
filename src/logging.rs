//! Shared build and runtime identification for opt-in solver logs.
use std::io::{self, Write};

pub(crate) fn banner(out: &mut impl Write, algorithm: &str) -> io::Result<()> {
    writeln!(out, "\n  BALDES ({})", env!("BALDES_GIT_VERSION"))?;
    writeln!(
        out,
        "  Version: {} | Target: {}-{}",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::ARCH,
        std::env::consts::OS
    )?;
    writeln!(
        out,
        "  SIMD: AVX2 {} | AVX-512 {} | Runtime path: {}",
        enabled(cfg!(feature = "simd")),
        enabled(cfg!(feature = "simd-avx512")),
        simd_path()
    )?;
    writeln!(
        out,
        "  Backends: Clp {} | HiGHS {} | Gurobi {}",
        enabled(cfg!(feature = "clp")),
        enabled(cfg!(feature = "highs")),
        enabled(cfg!(feature = "gurobi"))
    )?;
    writeln!(out, "  BALDES | {algorithm}")
}

fn enabled(value: bool) -> &'static str {
    if value {
        "enabled"
    } else {
        "disabled"
    }
}

fn simd_path() -> &'static str {
    #[cfg(all(feature = "simd-avx512", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("avx512f") {
        return "AVX-512 (AVX2/scalar fallback)";
    }
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("avx2") {
        return "AVX2 (scalar fallback)";
    }
    "scalar"
}
