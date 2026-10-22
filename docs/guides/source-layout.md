# Source layout

Production Rust code lives in `src/`. Unit tests live in `tests/unit/`, and
public-interface integration suites live in subsystem directories under
`tests/`. See the [test guide](../../tests/README.md) for registration and
commands. Runnable demonstrations belong in `examples/`.

| Directory | Responsibility |
| --- | --- |
| `src/core/` | Errors, options, logging, and conservative floating-point arithmetic |
| `src/graph/` | Graphs, resources, neighborhoods, and bucket construction |
| `src/labeling/` | Label storage, dominance, and directional labeling engines |
| `src/pricing/` | Stateful pricing, restrictions, cut preparation, and retained route universes |
| `src/enumeration/` | Route-domain proofs and bounded path enumeration |
| `src/cuts/` | Cut families, registration, and shared separation machinery |
| `src/master/` | LP interfaces, column generation, root solving, branching, and tree search |
| `src/master/multi_pricing/` | Shared masters with independently priced networks and imported columns |
| `src/master/solvers/` | Clp and HiGHS backends and shared MIP controls |
| `src/extensions/` | HGS, IPM, electric-vehicle routing, p-step, and column elimination |
| `src/inputs/` | Instance parsing and graph construction |
| `src/bin/baldes/` | Executable entry point and command implementations |

A module with production child files uses `name/mod.rs`; a leaf module uses
`name.rs`. This keeps the parent and its implementations together. Rust module
paths stay independent of these physical filenames. Private implementation
modules group work by responsibility, while the parent defines shared data and
exports the public interface.

`pricing/mod.rs` defines the solver state. Construction, configuration, cut
preparation, restriction application, and lifecycle changes have separate
implementations. `master/multi_pricing/mod.rs` defines its shared model and
state; `pricing.rs` implements network pricing callbacks, and `solve.rs`
implements initialization and the solve workflow.

Keep test bodies and fixtures outside `src/`. Attach private unit tests with
feature-gated `#[path]` module declarations so tests do not require widening
production visibility. Keep implementation-specific test adapters gated with
`#[cfg(test)]` when they must access private fields. Extension guides belong in
`docs/extensions/`; dependency licenses remain alongside their implementation.
