# Source layout

[Documentation index](../README.md) · [Architecture](architecture.md) · [Test layout](../../tests/README.md)

The source tree groups code by the work it owns. A pricing network and an array
containing that network use the same master framework. See the
[architecture guide](architecture.md) for execution flow and adapter semantics.

## Production ownership

| Directory | Responsibility |
| --- | --- |
| `src/core/` | Errors, options, logging and conservative floating-point arithmetic |
| `src/graph/` | Graphs, resources, neighborhoods and bucket geometry |
| `src/labeling/` | Label storage, resource extension, joining and dominance |
| `src/pricing/` | Stateful network oracle, cost preparation, restrictions and retained route universes |
| `src/enumeration/` | Route-domain proofs and bounded path collection |
| `src/cuts/` | Cut definitions, registration, separation and coefficient replay |
| `src/master/model/` | Canonical shared rows/networks, registrations, original variables and snapshots |
| `src/master/routing/` | Customer/fleet registration, routing cut hooks, witnesses and certificates |
| `src/master/formulation/` | Shared restricted-master construction and dynamic row insertion |
| `src/master/cg/` | Column-generation lifecycle, artificial columns, pools, restarts and handoffs |
| `src/master/pricing/` | Scheduling and feedback aggregation across enabled network oracles |
| `src/master/preprocessing/` | Propagation, conservation rows and model/routing preprocessing adapters |
| `src/master/tree/` | Shared lifecycle, frontier, policies, history, strong branching and adapters |
| `src/master/stabilization/` | Smoothing, proximal stabilization and curvature rules |
| `src/master/solvers/` | Clp, HiGHS, Gurobi and shared integer-copy controls |
| `src/extensions/` | HGS, IPM, EVRP, p-step, robust routing and column elimination |
| `src/inputs/` | Instance parsing and graph construction |
| `src/bin/baldes/` | `main.rs`, unified `cli.rs`, and distinct CE/p-step commands |
| `src/api/` | Public low-level interface helpers |
| `python/pybaldes/` | Python data/results and unified `solve.py` entry point |
| `python/rust/src/frontend/` | One `_solve` transport and private routing/network input compilers |

`master::model::Problem` owns shared rows and a network collection.
`RoutingContext` holds that model with routing metadata. The physical source
owners for tree and preprocessing algorithms are `master/tree` and
`master/preprocessing`; their model-specific files live under those owners.

Private adapters can be attached to their data-owning Rust module through
`#[path]`. That preserves field privacy while keeping a single physical source
for the implementation. Public callers use the parent's exported interface,
not private attachment names such as `tree_adapter` or `flow_adapter`.

## Finding the right file

| Change | Start here |
| --- | --- |
| New model rows / quantity layout | `master/formulation/mod.rs` |
| LP/pricing iteration or recovery handoff | `master/cg/driver.rs`, `master/cg/feasibility.rs` |
| Initial/cut/branch artificial columns | `master/cg/rows.rs` |
| Network visitation / admission | `master/pricing/mod.rs`, `master/pricing/scheduling.rs` |
| Raw objective pricing / cost cache | `pricing/variable_costs.rs` |
| Propagation or flow rows | `master/preprocessing/`, including shared `flow/mod.rs` |
| Node dispatch / budgets | `master/tree/driver.rs` |
| Open-node priority / overflow | `master/tree/frontier.rs` |
| Routing/model/imported tree interpretation | `master/tree/adapters/` |
| CLI parsing and result output | `bin/baldes/cli.rs` |
| Python scalar/array normalization | `python/pybaldes/solve.py` |

A module with implementation children normally uses `name/mod.rs`; a leaf uses
`name.rs`. Keep shared state near its consumers, helpers private to their owner,
and established solver verbs such as `price`, `extend`, `join` and `separate`.
Split independently explainable responsibilities; avoid generic `utils` files
or moving long argument lists behind pass-through helpers.

## Tests, examples and documentation

- `tests/unit/`: private unit tests, attached with feature-gated `#[path]` modules.
- `tests/api`, `tests/master`, `tests/pricing` and other subsystem folders: public-interface integration suites, explicitly registered in `Cargo.toml`.
- `python/tests/`: Python wrapper and native-interface checks.
- `examples/`: runnable demonstrations; `instances/`: benchmark inputs.
- `docs/`: maintained Markdown, built into the documentation site.
- `internal/` and `working/`: ignored local investigations and experiments.

Test bodies and fixtures stay outside `src/`. Most unit paths mirror their
implementation owner; the `#[path]` declaration is authoritative where a private
adapter belongs to another logical Rust module.

The current separation gives one owner to master construction, network pricing
coordination, preprocessing and tree orchestration. Model-specific data and
proof hooks remain explicit adapters. A future refactor of the CG loop should
preserve its restart, feasibility, cut-remapping and enumeration contracts;
folder moves alone do not establish numerical or behavioral equivalence.
