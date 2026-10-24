//! Shared build and runtime identification for opt-in solver logs.
use std::io::{self, Write};

pub(crate) fn banner(out: &mut impl Write, algorithm: &str) -> io::Result<()> {
    banner_with_context(out, algorithm, None)
}

pub(crate) fn banner_with_context(
    out: &mut impl Write,
    algorithm: &str,
    context: Option<&str>,
) -> io::Result<()> {
    writeln!(
        out,
        "  BALDES | {algorithm} | {} ({}) | {}-{}",
        env!("CARGO_PKG_VERSION"),
        env!("BALDES_GIT_VERSION"),
        std::env::consts::ARCH,
        std::env::consts::OS
    )?;
    if let Some(context) = context {
        writeln!(out, "{context}")?;
    }
    let backends: Vec<_> = [
        ("Clp", cfg!(feature = "clp")),
        ("HiGHS", cfg!(feature = "highs")),
        ("Gurobi", cfg!(feature = "gurobi")),
    ]
    .into_iter()
    .filter_map(|(name, enabled)| enabled.then_some(name))
    .collect();
    writeln!(
        out,
        "  SIMD: {} | Backends: {}",
        simd_path(),
        if backends.is_empty() {
            "none".into()
        } else {
            backends.join(", ")
        }
    )
}

fn simd_path() -> &'static str {
    #[cfg(all(feature = "simd-avx512", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("avx512f") {
        return "AVX-512";
    }
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("avx2") {
        return "AVX2";
    }
    "scalar"
}
