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
| [src/ipm/](../src/ipm/README.md) | Native interior-point LP solver, HiPO augmented factorization, and normal equations |
| [src/labeling/](../src/labeling/) | Labeling engine, label storage, and route pool |
| [src/enumeration/](../src/enumeration/) | Full-memory elementary enumeration |
| [src/pricing/](../src/pricing/) | Stateful pricing workflow, branching resources, fixing, and NG updates |
| [src/master/](../src/master/) | Restricted master, column generation, stabilization, branching constraints, root driver, and tree |
| [src/hgs/](../src/hgs/) | Hybrid genetic search: individuals, split, local search, genetic operators, population, and driver |
| [src/inputs/](../src/inputs/) | Solomon/CVRPLIB instances, parsing, graph construction, and optional root adapter |
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

[master/root/incumbent.rs](../src/master/root/incumbent.rs) validates integer
route quantities, exact original row activities and scalar resource feasibility,
then evaluates an objective upper bound from original arc costs. Physical arc
chains determine the resource checks; recorded nodes must agree with them.

[master/root/domain.rs](../src/master/root/domain.rs) validates unit visit-count
cover rows, fixed-route residuals and disposable scalar resources before
collecting a `CanonicalCoverRelaxation`. This projects feasible integer routes
onto canonical physical paths; it does not establish completeness of an NG LP
or authorize cut rows or root bound publication.
`ConservativeCoverRelaxation` uses downward resource bounds to cover exact-real
feasible paths, including arithmetic boundaries missed by strict forward replay.
Its states describe a relaxation and cannot serve as integer route witnesses.

[master/root/certificate.rs](../src/master/root/certificate.rs) publishes root
bounds for eligible unit-cover models. It restores conservative physical paths,
proves optional rows over residual integer partitions, obtains auxiliary LP
duals, and checks them against raw original costs and coefficients with directed
arithmetic. Unsupported models or exhausted budgets return an unknown bound.
Auxiliary slacks supply duals but never enter the certified matrix.

[master/root/restoration.rs](../src/master/root/restoration.rs) recovers bounded,
validated original columns when exhausted pricing leaves positive artificials.
It replays physical paths, exact resources and vertex elementarity before
registration; generic CG then re-solves rather than inferring infeasibility.

[master/root/cover_rows.rs](../src/master/root/cover_rows.rs) computes exact
integer row-activity ranges over residual customer partitions of that domain.
It relaxes fleet and branch restrictions and has explicit state, work and time
limits. Passing its range test proves a row valid on those canonical covers;
resource-schedule projection and original-domain eligibility remain separate.
Its additive-weight upper proof checks every auxiliary path and uses exact
integer cover conservation to validate weighted packing rows without subset DP.

[master/tree/mod.rs](../src/master/tree/mod.rs) owns node processing and the search queue. Its sibling
`candidates.rs`, `strong.rs`, and `diving.rs` own candidate generation, strong
branching, and diving. Typed constraints live in [master/branching.rs](../src/master/branching.rs).
LP backends remain under `master/solvers/`. Shared integer result types are in
[master/solvers/mip.rs](../src/master/solvers/mip.rs); [master/solvers/highs/integer.rs](../src/master/solvers/highs/integer.rs) transfers complete
models to HiGHS. [master/root/integer.rs](../src/master/root/integer.rs) prepares complete enumerated masters;
[master/tree/integer.rs](../src/master/tree/integer.rs) runs the optional end-of-node solve.

[master/networks.rs](../src/master/networks.rs) coordinates fresh roots with
multiple independent pricing graphs and shared master rows.
[master/network_bounds.rs](../src/master/network_bounds.rs) supplies conservative
graph-walk bound certificates. See [multiple networks](networks.md) for the
coefficient mapping and current integration limits.

[master/certificates.rs](../src/master/certificates.rs) evaluates conservative
lower bounds for explicit finite matrices using original cost terms and a
verified total-column-quantity row. It shares directed arithmetic with network
bounds through `master/rounding.rs`. A finite-matrix bound applies to an original
routing domain only after column completeness or excluded-domain coverage and
row validity have been established separately.

Public paths remain `master::root::{RootProblem, RootOptions, NodeSnapshot}`
and `master::tree::{TreeOptions, BranchingOptions}`.

## Pricing and enumeration

[pricing/mod.rs](../src/pricing/mod.rs) owns `Solver`, construction and forks, options, engine and cut
preparation, and column-generation round updates. `solve.rs` runs pricing phases;
`universe.rs` manages enumeration, inspection, and direct fixing;
`completion.rs` completes labels and restores arcs;
`fixing_workflow.rs` dispatches native fixing/enumeration modes;
`variable_costs.rs` handles variable costs, route-value corrections and raw
functor repricing. A rounded winning route alone cannot certify the raw minimum:
the functor uses an independent raw-cost pricing domain, preserving logical
branch constraints and sharing the call's time budget. Heuristic phases leave
the lower bound unknown;
`relaxation.rs` manages dynamic NG updates and route validation.
`branching.rs`, `phases.rs`, `fixing.rs`, and `controller.rs` provide supporting
resources, phase definitions, fixing result types, and round state.

If a backward candidate cannot be replayed within forward resource windows,
`pricing/solve.rs` rebuilds a forward-only search under the remaining call
deadline, reloading cuts, branch resources and explicit physical exclusions.
This recovers the minimum even when the failed candidate displaced valid paths
from the bounded solution pool.

`Route.resource_profile` records the source state and the state after every arc.
Label reconstruction preserves waiting from bucket jumps instead of replacing
it with earliest arrival. Root column identities include these schedules, so
equal arc paths with different resource states remain distinct. Code constructing
`Route` literals must supply the profile; `resources` is its final state.

[enumeration/mod.rs](../src/enumeration/mod.rs) exports `Controls`, `Enumeration`, `Label`, `CutTables`,
`Storage`, `Path`, and `PathStorage`. Their implementation files are
`controls.rs`, `engine.rs`, `label.rs`, `storage.rs`, and `paths.rs`.

[enumeration/domain.rs](../src/enumeration/domain.rs) exports
`CompleteCanonicalPaths`, a bounded collector of physical vertex-simple paths
under canonical forward scalar-resource extension. It preserves parallel arc
identity and borrows the graph. Budget exhaustion discards the partial result.
It keeps one earliest resource profile per path; callers must establish any
equivalence to all routing columns or alternative schedules separately.

`CompleteResourceRelaxation` shares the bounded physical-path search while
using directed additions for disposable resources. Its `RelaxedResourcePath`
values provide resource lower bounds; extra infeasible paths may remain.
Unsupported resource kinds and arithmetic failures return an explicit failure.

## C API and instances

C API implementation modules are private; public functions and ABI types are
re-exported through `api`. Shared handle validation and error handling stay in
[api/mod.rs](../src/api/mod.rs). Symbol names and calling conventions are unchanged.

[inputs/mod.rs](../src/inputs/mod.rs) owns `Customer`, `Instance`, and `Rounding`; `parsing.rs` reads
instances; `graph.rs` builds pricing graphs; `root.rs` provides the optional Clp
adapter. The public path is `inputs::Instance`.

Standalone network, root checkpoint, labeling profile, and arena test modules
live in the sibling `baldes-verification/overlays/unit/` directory. Its runner
attaches them to an isolated source snapshot; production builds need no external files.

For new code, place routines with the component that owns their state and add
exports at its existing facade when needed. Keep numerical and branching
behavior changes separate from file organization so each diff remains
reviewable.

[enumeration/frontier.rs](../src/enumeration/frontier.rs) records cached candidate
costs rejected during enumeration, ordered join breaks, and completion rejections.
These are observations from that run's cost epoch, preserved by pricing forks;
retained-path repricing does not update them. Representative pruning and symmetric
cost transformations are marked explicitly. The observations alone do not certify
an original-domain lower bound or account for a previously restricted bucket grid.
