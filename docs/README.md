# BALDES documentation

Use one solve interface for routing inputs and explicit pricing-network models.
A single network is a collection of length one. Start with
[Your first solve](guides/first-solve.md), then read the
[unified solve interface](guides/solve.md). Commands run from the repository root.

## Choose your task

| Task | Guide |
| --- | --- |
| Build and run a small instance | [First solve](guides/first-solve.md), [build](guides/build.md) |
| Understand the shared model and engine ownership | [Architecture](guides/architecture.md), [source layout](guides/source-layout.md) |
| Look up CLI input modes and flags | [Solve interface](guides/solve.md), [CLI reference](guides/cli.md) |
| Build a Python application | [pyBaldes handbook](pybaldes/README.md) |
| Model one or many pricing networks | [Network models](solver/models.md), [Python modeling](pybaldes/networks.md) |
| Integrate external columns and pricing callbacks | [Column imports](solver/column-imports.md) |
| Run integer search and interpret its bound | [Tree lifecycle](solver/tree.md), [solver concepts](guides/concepts.md) |

## Getting started and contributing

- [Your first solve](guides/first-solve.md): CLI walkthrough, outputs and setup fixes.
- [Unified solve interface](guides/solve.md): routing/model inputs and scalar/array normalization.
- [Build and installation](guides/build.md): dependencies, features, optional backends and Make configuration.
- [Command-line reference](guides/cli.md): common and routing-specific controls, JSON model controls and output.
- [Solver concepts](guides/concepts.md): paths, LP relaxations, witnesses and proof scope.
- [Architecture](guides/architecture.md): canonical model, shared engines and adapter responsibilities.
- [Source layout](guides/source-layout.md): implementation ownership and test locations.
- [Documentation maintenance](guides/documentation.md): source checks, API synchronization and site validation.

## Python handbook

- [Overview](pybaldes/README.md) and [installation](pybaldes/installation.md).
- [Quickstart](pybaldes/quickstart.md): routing, pricing and model examples.
- [Routing inputs and options](pybaldes/routing.md): instances, inference, distances and root keywords.
- [Network models](pybaldes/networks.md): scalar/array models, shared rows, integer trees, snapshots and synchronization.
- [Stateful pricing](pybaldes/pricing.md): graph setup, dual updates and low-level `Solver` methods.
- [Coupled resources](pybaldes/resources.md): resource definitions, piecewise functions and windows.
- [Results and guarantees](pybaldes/results.md): statuses, bounds, witnesses, counters and serialization.
- [API reference](pybaldes/api.md): exported names, declared fields and callable signatures.
- [Troubleshooting](pybaldes/troubleshooting.md): build errors, concurrency and reproducibility.
- [Python entry page](guides/python.md): a short guide to the handbook.

## Shared solver and model contracts

- [Network models and shared masters](solver/models.md): rows, multiplicities, scheduling, branches and original variables.
- [Column imports](solver/column-imports.md): finite domains, external pricing, certificates and imported-source trees.
- [Tree lifecycle](solver/tree.md): shared orchestration and routing registration details.
- [Feasibility recovery](solver/feasibility.md): phase I, cut-coefficient transitions and proof scope.
- [Branching](solver/branching.md): branch families, selection and history.
- [Cuts](solver/cuts.md): separation, configuration and coefficient consistency.
- [Stabilization](solver/stabilization.md): smoothing, proximal rules and automatic curvature.
- [Enumerated MIP](solver/enumerated-mip.md) and [deluxing](solver/deluxing.md): optional integer copies and coverage requirements.

## Reference

- [Algorithms and references](reference/algorithms.md).
- [BALDES options](reference/options.md).
- [Scalar resources](reference/resources.md).
- [SIMD dispatch](reference/simd.md).

## Specialized models and algorithms

- [Coupled resources](extensions/coupled-resources.md).
- [Multiple time windows](extensions/multi-time-windows.md).
- [P-step CVRP](extensions/pstep.md).
- [Electric-vehicle routing](extensions/evrp.md).
- [Robust vehicle routing](extensions/robust.md).
- [Robust pickup and delivery](extensions/robust-pd.md).
- [Interior-point solver](extensions/ipm.md).
- [Column elimination](extensions/ce.md).

## Recorded benchmarks

- [Root relaxations](benchmarks/root.md).
- [Three-row rank-one cuts](benchmarks/rank-one-three-row.md).
- [Rank-one local search](benchmarks/rank-one-local-search.md).
- [IPM factorization](benchmarks/ipm-factorization.md).

These pages record measurements under stated source/build settings. They do not
establish current performance without a matched rerun. Root-chart reproduction
requires explicit `--rounding none`.

## Interface capabilities

| Operation | Rust | CLI | Python |
| --- | --- | --- | --- |
| Routing root relaxation | `master::routing::solve` / `Instance::solve` | `solve INSTANCE` | `solve(instance)` / `instance.solve()` |
| Explicit network root, including one network | `master::model::solve` | `solve MODEL.json` | `solve(network)`, `solve([network])`, `solve(problem)` |
| Routing integer tree | `master::tree::solve` with `TreeOptions` | `solve INSTANCE --tree` | Routing tree not exposed |
| Explicit model integer tree | `master::tree::solve` with `ModelTreeOptions` | `solve MODEL.json --tree` | `solve(problem, tree=True)` |
| Low-level RCSP pricing | `pricing::Solver` | Used within solves | `Solver` |
| Original master variables without pricing | `master::model::solve_master` | Rust-only static-variable registration | Rust-only static-variable registration |
| Complete enumerated node MIP | Model/routing snapshot interfaces | `--enumerated-mip` with supported registration | Network snapshot / model options |

Shared-master registrations require `clp`; optional HiGHS/Gurobi backend and
integer-copy features are documented beside their interfaces. Different
registrations can install different cut, branch and witness hooks while sharing
construction, CG, pricing coordination and tree dispatch.

Root `Optimal` means the configured LP relaxation converged. Tree `Optimal`
closes the configured integer-domain gap within its tolerances and proof policy.
A restricted-column incumbent alone is not a lower-bound proof. Read
[concepts](guides/concepts.md) and [results](pybaldes/results.md) before comparing
bounds across models or interfaces.

- [Python wheels and releases](guides/releases.md): supported platforms, tagged builds and release assets.
