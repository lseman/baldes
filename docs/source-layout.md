# Source layout

[Documentation index](README.md) · [Tree lifecycle](tree.md) · [Cuts](cuts.md)

The Rust crate groups code by solver responsibility. Directory `mod.rs` files
provide entry points and public exports; implementation files hold the routines.
Existing Rust import paths and C symbols remain available.

| Directory | Responsibility |
| --- | --- |
| [src/api/](../src/api/) | C ABI handles, graph construction, options, pricing, fixing, relaxation, solution access, and statistics |
| [src/core/](../src/core/) | Shared errors, logging, and solver options |
| [src/cuts/](../src/cuts/) | Cut registry and cut-family modules |
| [src/graph/](../src/graph/) | Graph representation, buckets, NG preparation, relaxation, and symmetry |
| [src/labeling/](../src/labeling/) | Labeling engine, label storage, and route pool |
| [src/enumeration/](../src/enumeration/) | Full-memory elementary enumeration |
| [src/pricing/](../src/pricing/) | Stateful pricing workflow, branching resources, fixing, and NG updates |
| [src/master/](../src/master/) | Restricted master, column generation, stabilization, branching constraints, root driver, and tree |
| [src/hgs/](../src/hgs/) | Hybrid genetic search: individuals, split, local search, genetic operators, population, and driver |
| [src/vrptw/](../src/vrptw/) | Solomon/CVRPLIB instances, parsing, graph construction, and optional root adapter |
| [src/bin/](../src/bin/) | Command-line application |

## Cuts

[cuts/mod.rs](../src/cuts/mod.rs) exports the registry and its public types. `capacity/`, `knapsack/`,
`kpath/`, and `rank_one/` own their respective cut families. `shared/` owns common
cut-state, graph, and container helpers. Compatibility exports retain
paths such as `cuts::state` and `cuts::separation`.

## Master and tree

[master/root/mod.rs](../src/master/root/mod.rs) owns solved-node snapshots, mutable root state, and solve
or inheritance entry points. `problem.rs` owns rows and route coefficients;
`options.rs` owns root and cut-family configuration; `controller.rs` owns
column-generation callbacks and separation scheduling.

[master/tree/mod.rs](../src/master/tree/mod.rs) owns node processing and the search queue. Its sibling
`candidates.rs`, `strong.rs`, and `diving.rs` own candidate generation, strong
branching, and diving. Typed constraints live in [master/branching.rs](../src/master/branching.rs).
LP backends remain under `master/solvers/`. Shared integer result types are in
[master/solvers/mip.rs](../src/master/solvers/mip.rs); [master/solvers/highs/integer.rs](../src/master/solvers/highs/integer.rs) transfers complete
models to HiGHS. [master/root/integer.rs](../src/master/root/integer.rs) prepares complete enumerated masters;
[master/tree/integer.rs](../src/master/tree/integer.rs) runs the optional end-of-node solve.

Public paths remain `master::root::{RootProblem, RootOptions, NodeSnapshot}`
and `master::tree::{TreeOptions, BranchingOptions}`.

## Pricing and enumeration

[pricing/mod.rs](../src/pricing/mod.rs) owns `Solver`, construction and forks, options, engine and cut
preparation, and column-generation round updates. `solve.rs` runs pricing phases;
`universe.rs` manages enumeration, inspection, and direct fixing;
`completion.rs` completes labels and restores arcs;
`fixing_workflow.rs` dispatches native fixing/enumeration modes;
`variable_costs.rs` handles variable costs and corrected bounds;
`relaxation.rs` manages dynamic NG updates and route validation.
`branching.rs`, `phases.rs`, `fixing.rs`, and `controller.rs` provide supporting
resources, phase definitions, fixing result types, and round state.

[enumeration/mod.rs](../src/enumeration/mod.rs) exports `Controls`, `Enumeration`, `Label`, `CutTables`,
`Storage`, `Path`, and `PathStorage`. Their implementation files are
`controls.rs`, `engine.rs`, `label.rs`, `storage.rs`, and `paths.rs`.

## C API and instances

C API implementation modules are private; public functions and ABI types are
re-exported through `api`. Shared handle validation and error handling stay in
[api/mod.rs](../src/api/mod.rs). Symbol names and calling conventions are unchanged.

[vrptw/mod.rs](../src/vrptw/mod.rs) owns `Customer`, `Instance`, and `Rounding`; `parsing.rs` reads
instances; `graph.rs` builds pricing graphs; `root.rs` provides the optional Clp
adapter. Public paths such as `vrptw::Instance` remain unchanged.

For new code, place routines with the component that owns their state and add
exports at its existing facade when needed. Keep numerical and branching
behavior changes separate from file organization so each diff remains
reviewable.
