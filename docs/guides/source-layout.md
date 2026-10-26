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
| `src/master/pricing/` | Shared network scheduling, proof admission, oracle invocation, and batch accounting |
| `src/master/formulation/` | Common master construction and dynamic row insertion |
| `src/master/tree/` | Common tree lifecycle, evaluation dispatch, frontier and queue policies |
| `src/master/routing/` | Routing rows, cuts, certificates, and the routing pricing adapter |
| `src/master/model/` | Shared masters with independently priced networks and imported columns |
| `src/master/solvers/` | Clp, HiGHS, and Gurobi adapters and shared MIP controls |
| `src/extensions/` | HGS, IPM, electric-vehicle routing, p-step, robust routing, and column elimination (`ce`) |
| `src/inputs/` | Instance parsing and graph construction |
| `src/bin/baldes/` | Executable entry point and command implementations |

A module with production child files uses `name/mod.rs`; a leaf module uses
`name.rs`. This keeps the parent and its implementations together. Rust module
paths stay independent of these physical filenames. Private implementation
modules group work by responsibility, while the parent defines shared data and
exports the public interface.

`pricing/mod.rs` defines the resource-pricing solver state. The canonical master
model lives in `master/model/mod.rs`: shared rows plus a collection of networks.
`RoutingContext` registers routing metadata around that same model; it does not
own another graph, row or coefficient definition.

`master/formulation/` constructs masters for every registration.
`master/preprocessing/` owns propagation and flow construction. `master/tree/`
owns the single tree driver, frontier, statuses, policies and branching helpers.
Its routing, general-model and imported-domain adapters supply evaluation,
branching and certificate callbacks. They do not maintain separate search loops.
Adapter files may be attached as private children of their data-owning module
with `#[path]`, keeping implementation fields private while giving each subsystem
one physical source location.

`master/pricing/` coordinates one or more pricing networks. Exact arithmetic
lives in `master/numerics.rs`. The CLI is `src/bin/baldes/cli.rs`; Python uses
`pybaldes.solve` and the `_solve` extension transport. See the
[unified solve guide](solve.md) for accepted inputs and the breaking API changes.

Keep test bodies and fixtures outside `src/`. Attach private unit tests with
feature-gated `#[path]` module declarations so tests do not require widening
production visibility. Keep implementation-specific test adapters gated with
`#[cfg(test)]` when they must access private fields. Extension guides belong in
`docs/extensions/`; dependency licenses remain alongside their implementation.


## Responsibility within large modules

The top-level directories follow solver responsibilities. Keep that grouping as
new models and algorithms arrive. In particular, multiple pricing networks
still belong to `master/`, while a specialized routing formulation belongs to
`extensions/`. Low-level route enumeration belongs to `enumeration/`; installing
its routes into a restricted master belongs to `master/`.

| Module | Internal ownership |
| --- | --- |
| `graph/coupled/kernel/mod.rs` | Shared kernel/state types, state inspection, graph attachment, and child declarations |
| `graph/coupled/kernel/preparation.rs` | Prepare definitions, configure objectives, and validate graph compatibility |
| `graph/coupled/kernel/extension.rs` | Create directional labels, extend arcs, and replay paths |
| `graph/coupled/kernel/joining.rs` | Join forward/backward states and compute the joined objective |
| `master/model/enumeration/mod.rs` | Public options/reports and snapshot enumeration entry point |
| `master/model/enumeration/collection.rs` | Collect complete or cutoff-bounded route universes under shared budgets |
| `master/model/enumeration/transfer.rs` | Install a collected universe atomically on a private master |
| `master/pricing/mod.rs` | Coordinate live and retained pricing for both model adapters |
| `master/model/enumeration/mip.rs` | Solve integer copies and check original-domain encoding |
| `master/model/enumeration/cutoff.rs` | Own the improvement-domain snapshot and expose its restricted guarantees |
| `master/cg/setup.rs` | Validate options and warm-start metadata before execution |
| `master/cg/driver.rs` | Sequence master solves, pricing, stabilization, cuts, and handoffs |

Kernel submodules with their own implementations (`front`, `relation`,
`dominance`, and `certificate`) also use `mod.rs`. Moving a file into its module
directory does not change its Rust import path.

## Naming and split decisions

Use established solver vocabulary (`price`, `extend`, `join`, `dominates`,
`separate`) when the enclosing type or module supplies the subject. Prefer a
more specific verb when several subjects share the scope: `collect_routes` and
`collect_routes_with_plan` distinguish route-universe construction from
iterator collection. Public names such as `Options`, `Output`, and `Status`
remain useful under a descriptive module path; import them with a module
qualifier when several solvers appear in one file.

Keep implementation state near its consumer. For example, column generation's
`IpmAccuracy` belongs beside its loop rather than among its public data types.
Keep private helpers private to their owning module; sibling implementations
may share them through a restricted parent visibility. Re-export public types
and entry points from their existing parent so callers do not learn the
internal file layout.

Split when a file contains independently explainable responsibilities. Keep
extension and replay together because they share the same transition rules.
Avoid extracting each short helper into its own file or creating generic
`utils` modules. A long mathematical routine may remain cohesive; line count
alone does not justify another interface.

## Remaining redesign candidates

The organization review identifies these next areas for deeper work:

- `master/cg/driver.rs`: the main loop still coordinates many
  mutable values and several restart paths. A future extraction should give
  iteration state a single owner and test warm starts, feasibility recovery,
  cut remapping, and enumeration handoffs together. Moving individual branches
  behind helpers with long mutable argument lists would retain that coupling.
- `graph/coupled/kernel/relation/mod.rs` and
  `graph/coupled/kernel/bounds.rs`: separate preparation, directional operations,
  and mathematical helpers only where they can share a small internal
  interface. Existing specialized child implementations provide starting
  points; preserve numerical evaluation order when extracting code.
- `bin/baldes/routing.rs`: consider separating command configuration from
  execution and result rendering when adding another routing command.

The canonical namespaces are `baldes::master::cg` for column generation,
`baldes::master::model` for multi-pricing, and `baldes::extensions::ce` for column
elimination. Source and mirrored unit-test
folders use those abbreviations as well. These namespace renames require Rust
callers to update their imports; descriptive algorithm method names remain
unchanged. Internal file splits preserve the remaining public interfaces and
numerical algorithms. This is a structural pass, not an exhaustive correctness
or dead-code audit of every function. Further function renames should follow call-site evidence and domain
meaning rather than a global naming substitution.
