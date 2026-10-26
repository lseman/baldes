# BALDES documentation

New to BALDES? Follow [Your first solve](guides/first-solve.md) to build the CLI,
run a small instance, and understand the result.
Commands in these guides run from the repository root.

## Choose your task

| I want to… | Start here |
| --- | --- |
| Run a small example and understand its output | [Your first solve](guides/first-solve.md) |
| Find a feasible routing solution quickly | [HGS and CLI usage](guides/cli.md) |
| Search for an integer solution with bounds | [Branch-and-price tree](solver/tree.md) |
| Build a Python application or custom pricing loop | [pyBaldes documentation](pybaldes/README.md) |
| Use multiple route families in one master | [Multi-pricing](extensions/mp.md) |
| Configure backends or troubleshoot a build | [Build and installation](guides/build.md) |

## Reading paths

- **CLI users:** first solve → CLI reference → solver concepts → tree/cut guides.
- **Python users:** pyBaldes installation → quickstart → routing, pricing, or network modeling.
- **Rust contributors:** build guide → source layout → model/solver contracts → documentation maintenance.

## Getting started

- [Your first solve](guides/first-solve.md): a complete CLI walkthrough, result interpretation, and common setup fixes.

- [Build and installation](guides/build.md): dependencies, optional backends, and Make configuration.
- [Source layout](guides/source-layout.md): production modules, CLI commands, and test organization.
- [Command-line usage](guides/cli.md): routing solves, HGS, flags, and output.
- [Solver concepts](guides/concepts.md): paths, masters, root relaxations, integer search, and proof scope.
- [Documentation maintenance](guides/documentation.md): organization, local builds, validation, and API update checks.

## pyBaldes: Python documentation

A separate handbook for the Python package, with its own entry point and API reference.

- [pyBaldes overview](pybaldes/README.md): choose routing, custom pricing, or shared-master modeling.
- [Installation](pybaldes/installation.md): package builds, optional features, backends, and capacities.
- [Quickstart](pybaldes/quickstart.md): complete in-memory and file-based examples.
- [Routing inputs and options](pybaldes/routing.md): customers, instances, distances, and every root keyword.
- [Results and guarantees](pybaldes/results.md): fields, bounds, status, exactness, and serialization.
- [Stateful pricing](pybaldes/pricing.md): graph construction, dual updates, and Solver methods.
- [Coupled resources](pybaldes/resources.md): resource models, piecewise functions, and windows.
- [Network models](pybaldes/networks.md): shared rows, solve controls, integer trees, snapshots, and synchronization.
- [API reference](pybaldes/api.md): exported types, constructors, and callable signatures.
- [Troubleshooting](pybaldes/troubleshooting.md): errors, concurrency, and reproducibility.

## Solver

- [Tree and node lifecycle](solver/tree.md): integer routing, inherited state, bounds, and limits.
- [Routing feasibility recovery](solver/feasibility.md): zero-objective phase I, objective restoration, proof scope and limits.
- [Branching](solver/branching.md): branching families and selection.
- [Cuts](solver/cuts.md): cut configuration and coefficient consistency.
- [Dual-price stabilization](solver/stabilization.md): smoothing, center updates, and retries.
- [Enumerated MIP](solver/enumerated-mip.md): optional HiGHS integer solving and proof coverage.
- [Deluxing](solver/deluxing.md): complete-universe column filtering before integer solving.

## Reference

- [Algorithms and references](reference/algorithms.md): solver components and research references.
- [BALDES options](reference/options.md): supported controls, defaults, types, and scope.
- [Scalar resources](reference/resources.md): resource dimensions and bucket indexing.
- [SIMD dispatch](reference/simd.md): optional kernels and runtime portability.

## Extensions and modeling

- [Coupled resources](extensions/coupled-resources.md): simultaneous pickup/delivery, selected values, cumulative costs and affine dependencies in the physical-graph labeling engine.
- [Multiple time windows](extensions/multi-time-windows.md): disjoint service-start domains for scalar pricing resources.

- [Multi-pricing](extensions/mp.md): shared masters, multiple route families, and assignment branching.
- [Finite column imports](extensions/column-imports.md): original-vector patterns and multiplicity constraints.
- [P-step CVRP](extensions/pstep.md): compact p-step LP relaxation.
- [Electric-vehicle routing](extensions/evrp.md): charging models, schedule replay, and current limitations.
- [Robust vehicle routing](extensions/robust.md): budgeted demand and travel-time uncertainty in bucket graph labeling, plus an independent validation reference.
- [Interior-point solver](extensions/ipm.md): native LP backend, interfaces, and implementation details.
- [Column elimination (CE)](extensions/ce.md): decision-diagram route relaxation and refinement.

## Benchmarks

- [Root benchmarks](benchmarks/root.md): published root relaxation timings and measurement setup.
- [Three-row rank-one cuts](benchmarks/rank-one-three-row.md): measured separation configurations.
- [Rank-one local search](benchmarks/rank-one-local-search.md): separator search measurements.
- [IPM factorization](benchmarks/ipm-factorization.md): factorization benchmark setup and results.

## Capabilities and interfaces

| Operation | Rust | CLI | Python | Backend |
| --- | --- | --- | --- | --- |
| RCSP pricing and enumeration | Yes | Through routing solves | Low-level `Solver` | Core Rust |
| Routing root relaxation | Yes | Default `vrptw` / `cvrp` mode | `pybaldes.solve` | Clp model store; Clp/IPM/optional HiGHS/Gurobi solves |
| Branch-and-price tree | Yes | `--tree` | Not exposed | Clp model store; Clp/IPM/optional HiGHS/Gurobi solves |
| Shared-master root and integer tree | Yes | `multi-pricing` | `solve` | Clp; HiGHS for optional enumerated MIP |
| Complete enumerated node MIP | Yes, opt-in | `--tree --enumerated-mip` | Not exposed | Clp + HiGHS |
| Generic mixed-integer master solve | `highs::solve_integer` | Not exposed directly | Not exposed | HiGHS |

The `vrptw` and `cvrp` routing tree uses one pricing network. For models with
multiple route families, use the [multi-pricing interface](extensions/mp.md),
including its CLI and Python APIs and assignment tree. Its supported branching
families and solve limitations differ from the single-network routing tree.

Root `Optimal` means the LP relaxation converged. Tree `Optimal` means the
integer routing gap closed within the configured tolerances. A feasible
restricted-column solution supplies an incumbent; its objective alone does
not prove a lower bound for the full routing problem.

## Rust module organization

Core modules implement graphs, labeling, pricing, cuts, enumeration and the
master. `master::model` coordinates one master with multiple pricing
subproblems and belongs to the core master framework. Rust callers can also use
`master::model::solve_master` for original master variables with no
pricing subproblems.

| Module | Purpose |
| --- | --- |
| `extensions::hgs` | Hybrid Genetic Search and primal heuristics |
| `extensions::ipm` | Native interior-point LP backend |
| `extensions::evrp` | Electric-vehicle routing models and charging schedules |
| `extensions::pstep` | P-step CVRP LP with parallel bucket pricing and reference enumeration |
| `master::model` | Shared rows, per-subproblem multiplicities and node lifecycle |

`extensions` groups algorithms and specialized models in the same crate; it
does not imply runtime plugins or new build features. Existing `baldes::hgs`,
`baldes::ipm`, `baldes::evrp` and `master::model` paths remain compatibility
re-exports. New Rust code should use the module paths above. CLI JSON fields
and Python names such as `NetworkProblem` and `solve` stay compatible.
