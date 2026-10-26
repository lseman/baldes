# Test layout

Integration suites exercise the public crate interface and are grouped by
subsystem (`api`, `pricing`, `cuts`, `master`, `hgs`, `ipm`, and `instances`).
Register new integration targets explicitly in `Cargo.toml`: automatic test
discovery is disabled. Keep fixtures beside the integration suite that owns
them.

`unit/` mirrors the production module hierarchy. These files are compiled as
child modules of their production modules through `#[cfg(test)]` and `#[path]`
declarations in `src/`. They retain access to private implementation details
without exposing test helpers in the public interface. They are not standalone
integration targets and need no Cargo registration. Keep feature gates on the
module declarations, including gates for native backends and SIMD.

For example, `tests/unit/master/cg/rows_tests.rs` is attached to
`src/master/cg/rows.rs`:

```rust,ignore
#[cfg(all(test, feature = "clp"))]
#[path = "../../../tests/unit/master/cg/rows_tests.rs"]
mod tests;
```

Run the default suites with `cargo test -p baldes`, or exercise shared-master, HiGHS and multiple-window paths with
`cargo test -p baldes --features clp,highs,multi-time-windows`. Other optional
backends/kernels require their corresponding dependencies; `--all-features`
additionally enables Gurobi and CUDA and is not a portable default check. Unit tests can
be selected with `cargo test -p baldes --lib pricing::variable_costs::tests`; integration
suites use `--test`, for example `cargo test -p baldes --test main_pricing`.
