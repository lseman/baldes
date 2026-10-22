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

For example, `tests/unit/graph/resources/tests.rs` is attached to
`src/graph/resources.rs`:

```rust,ignore
#[cfg(test)]
#[path = "../../tests/unit/graph/resources/tests.rs"]
mod tests;
```

Run the default suites with `cargo test -p baldes`, or include all optional
backends and kernels with `cargo test -p baldes --all-features`. Unit tests can
be selected with `cargo test -p baldes --lib graph::resources::tests`; integration
suites use `--test`, for example `cargo test -p baldes --test main_pricing`.
