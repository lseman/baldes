# BALDES documentation

Start with the [project README](../README.md) for installation and quick examples.
Commands in these guides run from the repository root.

## Getting started

- [Build and installation](guides/build.md): dependencies, optional backends, and Make configuration.
- [Command-line usage](guides/cli.md): routing solves, HGS, flags, and output.
- [Python API](guides/python.md): installation, model inputs, solver options, and result export.

## Solver

- [Tree and node lifecycle](solver/tree.md): integer routing, inherited state, bounds, and limits.
- [Branching](solver/branching.md): branching families and selection.
- [Cuts](solver/cuts.md): cut configuration and coefficient consistency.
- [Dual-price stabilization](solver/stabilization.md): smoothing, center updates, and retries.
- [Enumerated MIP](solver/enumerated-mip.md): optional HiGHS integer solving and proof coverage.

## Reference

- [Algorithms and references](reference/algorithms.md): solver components and research references.
- [BALDES options](reference/options.md): supported controls, defaults, types, and scope.
- [Scalar resources](reference/resources.md): resource dimensions and bucket indexing.
- [SIMD dispatch](reference/simd.md): optional kernels and runtime portability.

## Extensions and modeling

- [Multiple time windows](extensions/multi-time-windows.md): disjoint service-start domains for scalar pricing resources.

- [Multi-pricing](extensions/multi-pricing.md): shared masters, multiple route families, and assignment branching.
- [Finite column imports](extensions/column-imports.md): original-vector patterns and multiplicity constraints.
- [P-step CVRP](extensions/pstep.md): compact p-step LP relaxation.
- [Electric-vehicle routing](extensions/evrp.md): charging models, schedule replay, and current limitations.

## Benchmarks

- [Root benchmarks](benchmarks/root.md): published root relaxation timings and measurement setup.

## Capabilities and interfaces

| Operation | Rust | CLI | Python | Backend |
| --- | --- | --- | --- | --- |
| RCSP pricing and enumeration | Yes | Through routing solves | Low-level `Solver` | Core Rust |
| Routing root relaxation | Yes | Default `vrptw` / `cvrp` mode | `pybaldes.solve_root` | Clp model store; Clp/IPM/optional HiGHS solves |
| Branch-and-price tree | Yes | `--tree` | Not exposed | Clp model store; Clp/IPM/optional HiGHS solves |
| Complete enumerated node MIP | Yes, opt-in | `--tree --enumerated-mip` | Not exposed | Clp + HiGHS |
| Generic mixed-integer master solve | `highs::solve_integer` | Not exposed directly | Not exposed | HiGHS |

The `vrptw` and `cvrp` routing tree uses one pricing network. For models with
multiple route families, use the [multi-pricing interface](extensions/multi-pricing.md),
including its CLI and Python APIs and assignment tree. Its supported branching
families and solve limitations differ from the single-network routing tree.

Root `Optimal` means the LP relaxation converged. Tree `Optimal` means the
integer routing gap closed within the configured tolerances. A feasible
restricted-column solution supplies an incumbent; its objective alone does
not prove a lower bound for the full routing problem.

## Rust module organization

Core modules implement graphs, labeling, pricing, cuts, enumeration and the
master. `master::multi_pricing` coordinates one master with multiple pricing
subproblems and belongs to the core master framework. Rust callers can also use
`master::multi_pricing::solve_master` for original master variables with no
pricing subproblems.

| Module | Purpose |
| --- | --- |
| `extensions::hgs` | Hybrid Genetic Search and primal heuristics |
| `extensions::ipm` | Native interior-point LP backend |
| `extensions::evrp` | Electric-vehicle routing models and charging schedules |
| `extensions::pstep` | P-step CVRP LP with parallel bucket pricing and reference enumeration |
| `master::multi_pricing` | Shared rows, per-subproblem multiplicities and node lifecycle |

`extensions` groups algorithms and specialized models in the same crate; it
does not imply runtime plugins or new build features. Existing `baldes::hgs`,
`baldes::ipm`, `baldes::evrp` and `master::networks` paths remain compatibility
re-exports. New Rust code should use the module paths above. CLI JSON fields
and Python names such as `NetworkProblem` and `solve_networks` stay compatible.
